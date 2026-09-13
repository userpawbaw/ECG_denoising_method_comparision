"""HF 동기화 — **망 없이** 계약만 고정한다.

여기서 막으려는 사고는 셋이다.
  1) 올리기 실패가 학습을 죽이는 것 (30 분짜리 판을 딸꾹질로 잃는다)
  2) 내려받기가 **이번 세션의 진행을 덮어쓰는** 것
  3) `--hf-repo` 를 안 줬는데 huggingface_hub 를 찾으러 가는 것
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ecgdn.sync_hf import HFStore


class FakeApi:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.files: list[str] = []
        self.folders: list[tuple[str, str]] = []
        self.deleted: list[str] = []
        self.repos: list[str] = []

    def create_repo(self, repo_id, **kw):
        self.repos.append(repo_id)

    def upload_file(self, *, path_or_fileobj, path_in_repo, **kw):
        if self.fail:
            raise OSError("망이 끊겼다")
        self.files.append(path_in_repo)

    def upload_folder(self, *, folder_path, path_in_repo, **kw):
        if self.fail:
            raise OSError("망이 끊겼다")
        self.folders.append((str(folder_path), path_in_repo))

    def delete_file(self, *, path_in_repo, **kw):
        self.deleted.append(path_in_repo)


def _store(tmp_path, fail=False, every=10):
    s = HFStore("u/r", prefix="depth", every=every)
    s._api, s._ready = FakeApi(fail), True
    return s


def _run(tmp_path, name="m06_l1__s0"):
    d = tmp_path / name
    d.mkdir()
    (d / "last.pt").write_bytes(b"ckpt")
    (d / "log.csv").write_text("epoch\n1\n")
    (d / "summary.json").write_text("{}")
    return d


def test_disabled_store_never_imports_the_hub(tmp_path, monkeypatch):
    """`--hf-repo` 를 안 주면 **아무것도 안 한다.** 호출부에 분기를 안 만든다."""
    import builtins
    real = builtins.__import__

    def boom(name, *a, **k):
        if name.startswith("huggingface_hub"):
            raise AssertionError("쓰지 않는데 hub 를 불렀다")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", boom)
    s = HFStore(None)
    assert not s.enabled
    assert s.pull(tmp_path) == 0
    assert s.push_epoch(_run(tmp_path), 10) is False
    assert s.push_run(tmp_path) is False
    assert s.push_all(tmp_path) is False


def test_push_only_on_multiples_of_every(tmp_path):
    s = _store(tmp_path, every=10)
    d = _run(tmp_path)
    assert [e for e in range(1, 31) if s.should_push(e)] == [10, 20, 30]
    assert s.push_epoch(d, 7) is False
    assert s.push_epoch(d, 10) is True
    assert s._api.files == ["depth/m06_l1__s0/last.pt", "depth/m06_l1__s0/log.csv"]


def test_epoch_zero_never_pushes(tmp_path):
    """0 % 10 == 0 이라 그냥 나누면 학습 시작 전에 한 번 올라간다."""
    assert _store(tmp_path).should_push(0) is False


def test_upload_failure_does_not_raise(tmp_path):
    """**이것이 이 모듈의 존재 이유다** — 통신 실패가 학습을 죽이면 안 된다."""
    s = _store(tmp_path, fail=True)
    d = _run(tmp_path)
    assert s.push_epoch(d, 10) is False      # 예외가 아니라 False
    assert s.push_run(d) is False
    assert s.push_all(tmp_path) is False


def test_pull_never_overwrites_local_files(tmp_path, monkeypatch):
    """돌고 있는 기계의 것이 최신이다. HF 것으로 덮으면 이번 세션을 지운다."""
    remote = tmp_path / "remote" / "depth" / "m06_l1__s0"
    remote.mkdir(parents=True)
    (remote / "log.csv").write_text("지난 세션\n")
    (remote / "last.pt").write_bytes("지난 체크포인트".encode())

    local = tmp_path / "local"
    (local / "m06_l1__s0").mkdir(parents=True)
    (local / "m06_l1__s0" / "log.csv").write_text("이번 세션\n")

    s = _store(tmp_path)
    import sys, types
    mod = types.ModuleType("huggingface_hub")
    mod.snapshot_download = lambda *a, **k: str(tmp_path / "remote")
    mod.HfApi = lambda: s._api
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)

    assert s.pull(local) == 1                                   # last.pt 만 새로 왔다
    assert (local / "m06_l1__s0" / "log.csv").read_text() == "이번 세션\n"
    assert (local / "m06_l1__s0" / "last.pt").exists()


def test_finished_run_upload_drops_the_dead_checkpoint(tmp_path):
    """끝난 판의 `last.pt` 는 재개에 안 쓰인다 — 저장소에 남기지 않는다."""
    s = _store(tmp_path)
    d = _run(tmp_path)
    assert s.push_run(d) is True
    assert s._api.folders == [(str(d), "depth/m06_l1__s0")]
    assert s._api.deleted == ["depth/m06_l1__s0/last.pt"]


def test_trainer_calls_back_every_epoch_and_survives_a_bad_callback():
    """학습기 쪽 계약. 콜백이 터져도 학습이 계속돼야 한다."""
    torch = pytest.importorskip("torch")
    from ecgdn.config import TrainCfg
    from ecgdn.data.dataset import ECGDenoiseDataset
    from ecgdn.data.sources import get_source
    from ecgdn.models import build_model, make_loss
    from ecgdn.train import Trainer
    import tempfile

    src = get_source("synthetic", n_records=2, dur_s=20.0)
    ds = ECGDenoiseDataset(src, split="train", win=256, hop=256,
                           max_per_record=4, frontend=False)
    seen: list[int] = []

    def cb(ep, row):
        seen.append(ep)
        raise RuntimeError("올리기 실패")

    with tempfile.TemporaryDirectory() as td:
        t = Trainer(build_model("resunet1d", chs=(4, 6, 8)), make_loss("L1"), ds, ds,
                    TrainCfg(epochs=2, batch_size=2, patience=5),
                    out_dir=td, device="cpu", num_workers=0, on_epoch_end=cb)
        st = t.fit()
    assert seen == [1, 2]          # 매 epoch 불렸고
    assert st.epoch == 2           # 예외에도 학습이 끝까지 갔다
