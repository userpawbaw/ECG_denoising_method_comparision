"""학습 재개 — 중단된 학습을 처음부터 다시 돌리지 않기 위한 최소 보장.

이 기능은 실제로 두 번 학습을 잃고 나서 만들었다 (docs/99_status.md M-4).
가중치만 복원하는 '재개' 는 재개가 아니다 — optimizer 모멘텀, LR 스케줄 위치,
early stopping 이 보는 best_epoch 이 함께 돌아와야 같은 학습이 이어진다.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecgdn.config import TrainCfg
from ecgdn.train import Trainer


class _TinyDS:
    """학습 루프를 돌리기 위한 최소 데이터셋 (채널 우선 (1, N))."""

    def __init__(self, n=16, win=256, seed=0):
        g = np.random.default_rng(seed)
        self.x = g.standard_normal((n, win)).astype(np.float32) * 0.1
        self.y = self.x + g.standard_normal((n, win)).astype(np.float32) * 0.05

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return (self.y[i][None, :], self.x[i][None, :],
                {"scale": np.float32(1.0), "snr": np.float32(10.0),
                 "record": "T", "start": np.int64(0)})


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c = torch.nn.Conv1d(1, 1, 5, padding=2)

    def forward(self, y):
        return self.c(y)


def _loss(xhat, x, s_hat=None, s=None):
    """DenoiseLoss 와 같은 계약: (total, parts) 이고 parts 에 'total' 이 들어간다."""
    t = torch.nn.functional.l1_loss(xhat, x)
    return t, {"total": t}


def _make(tmp_path, epochs):
    torch.manual_seed(0)
    cfg = TrainCfg(epochs=epochs, batch_size=4, patience=99, lr=1e-3)
    return Trainer(_TinyModel(), _loss, _TinyDS(), _TinyDS(seed=1), cfg,
                   out_dir=tmp_path / "run", device="cpu")


def test_resume_continues_instead_of_restarting(tmp_path):
    st = _make(tmp_path, 2).fit()
    assert st.epoch == 2

    t2 = _make(tmp_path, 4)
    assert t2.try_resume() is True
    assert t2.state.epoch == 2, "재개 지점이 복원되지 않았다"
    st2 = t2.fit()

    assert st2.epoch == 4
    # 이어서 돈 epoch 만 추가돼야 한다 — 처음부터 다시 돌면 6행이 된다
    assert [r["epoch"] for r in st2.history] == [1, 2, 3, 4]

    log = (tmp_path / "run" / "log.csv").read_text().strip().splitlines()
    assert [l.split(",")[0] for l in log[1:]] == ["1", "2", "3", "4"], \
        "log.csv 가 덮어쓰이거나 끊겼다"


def test_resume_restores_optimizer_and_best_tracking(tmp_path):
    t1 = _make(tmp_path, 2)
    t1.fit()
    best_metric, best_epoch = t1.state.best_metric, t1.state.best_epoch

    t2 = _make(tmp_path, 4)
    t2.try_resume()
    assert t2.state.best_metric == pytest.approx(best_metric)
    assert t2.state.best_epoch == best_epoch, "early stopping 기준이 초기화됐다"
    assert t2.state.step > 0, "LR 스케줄 위치가 복원되지 않았다"

    # optimizer 상태(모멘텀)가 실제로 실려 있어야 한다
    ck = torch.load(tmp_path / "run" / "last.pt", map_location="cpu", weights_only=False)
    assert "opt" in ck and ck["opt"]["state"], "last.pt 에 optimizer 상태가 없다"


def test_resume_without_checkpoint_is_a_clean_start(tmp_path):
    t = _make(tmp_path, 1)
    assert t.try_resume() is False
    assert t.state.epoch == 0


def test_checkpoint_records_the_epoch_it_was_saved_at(tmp_path):
    """이전에는 저장 시점보다 epoch 이 1 작게 기록됐다."""
    _make(tmp_path, 3).fit()
    ck = torch.load(tmp_path / "run" / "last.pt", map_location="cpu", weights_only=False)
    assert ck["epoch"] == 3


def test_already_finished_run_does_not_retrain(tmp_path):
    _make(tmp_path, 2).fit()
    t2 = _make(tmp_path, 2)
    t2.try_resume()
    st = t2.fit()
    assert st.epoch == 2
    assert len(st.history) == 2, "끝난 학습을 다시 돌렸다"
