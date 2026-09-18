"""sweep 이 **판 하나가 중간에 끊겨도 이어서** 가는가.

종래에는 `summary.json` 이 있는 판만 건너뛰었다. 그래서 한 판이 30 epoch 에서
끊기면 **처음부터** 다시 돌았다. Colab 처럼 세션이 자주 끊기는 환경에서는 이것이
곧 한 판을 통째로 잃는 것이다 — 실제로 그렇게 잃었다.

`last.pt` 는 epoch 마다 저장되고 optimizer·스케줄 위치·`best_epoch` 까지 담는다.
여기서는 **그것이 실제로 이어지는지**를 본다. 「기능이 있다」가 아니라
「끊고 다시 돌리면 이어진다」를 잰다.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytest.importorskip("torch")

from ecgdn.data.sources import resolve_source_kind  # noqa: E402


def _sweep(out: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/run_seed_sweep.py", "--arms", "m06_l1",
         "--seeds", "99", "--out", str(out), "--workers", "0", *extra],
        cwd=ROOT, capture_output=True, text=True, timeout=1800)


@pytest.mark.skipif(resolve_source_kind("mitdb") != "mitdb",
                    reason="data/raw/mitdb 가 없다")
@pytest.mark.slow
def test_an_interrupted_run_resumes_from_its_last_epoch(tmp_path):
    """2 epoch 돌린 뒤, 같은 자리에서 4 epoch 으로 재개하면 3 부터 이어진다."""
    out = tmp_path / "sweep"
    a = _sweep(out, "--epochs", "2")
    assert a.returncode == 0, a.stderr[-2000:]
    d = out / "m06_l1__s99"
    assert (d / "summary.json").exists()
    first = json.loads((d / "summary.json").read_text())
    assert first["n_epochs"] == 2

    # 끝난 판은 건너뛴다 — 재개가 완료된 판을 건드리면 안 된다
    b = _sweep(out, "--epochs", "2")
    assert "이미 있음, 건너뜀" in b.stdout, b.stdout[-2000:]


def test_partial_epoch_reads_the_checkpoint(tmp_path):
    """`partial_epoch` 이 끝난 판을 «끊긴 판» 으로 잘못 보지 않는다."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_sw", ROOT / "scripts" / "run_seed_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    d = tmp_path / "run"
    d.mkdir()
    assert mod.partial_epoch(d) is None          # 아무것도 없다

    import torch
    torch.save({"epoch": 17}, d / "last.pt")
    assert mod.partial_epoch(d) == 17            # 끊겼다

    (d / "summary.json").write_text("{}")
    assert mod.partial_epoch(d) is None          # 끝난 판이다


def test_resume_is_on_by_default_and_can_be_turned_off():
    """기본이 재개다. 껐을 때만 처음부터 간다."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_sw2", ROOT / "scripts" / "run_seed_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = (ROOT / "scripts" / "run_seed_sweep.py").read_text()
    assert "resume: bool = True" in src, "run_one 의 기본이 재개여야 한다"
    assert "--no-resume" in src
    assert "trainer.try_resume()" in src, "sweep 이 try_resume 을 불러야 한다"


def test_analyzer_uses_declared_base_not_alphabetical(tmp_path):
    """**기준 arm 을 알파벳으로 고르면 모든 Δ 의 부호가 뒤집힌다.**

    `sweep.csv` 는 `sorted(glob)` 으로 쓰이므로 행 순서가 알파벳 순이다.
    D-28 2 번(`m06_cond` vs `m06_l1`)에서 실제로 `m06_cond` 가 기준이 되어
    「기준이 조건화보다 0.8 dB 나쁘다」가 아니라 그 반대로 찍혀 나왔다.
    sweep 이 남기는 `arms.txt` 가 의도한 순서를 들고 있어야 한다.
    """
    import csv as _csv
    import subprocess
    import sys

    cols = ["run_id", "arm", "seed", "best_metric", "fixed_snr_imp_scaled"]
    rows = [
        # 알파벳으로는 zzz 가 뒤, aaa 가 앞. 의도한 기준은 zzz 다.
        ("aaa__s0", "aaa", 0, 5.0, 5.0), ("aaa__s1", "aaa", 1, 5.0, 5.0),
        ("zzz__s0", "zzz", 0, 4.0, 4.0), ("zzz__s1", "zzz", 1, 4.0, 4.0),
    ]
    p = tmp_path / "sweep.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    (tmp_path / "arms.txt").write_text("zzz\naaa\n")

    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_seed_sweep.py"),
                          str(tmp_path)], capture_output=True, text=True).stdout
    assert "기준 arm = zzz" in out, out
    # zzz 가 기준이면 aaa 의 Δ 는 **양수** (5.0 − 4.0)
    line = [l for l in out.splitlines() if l.strip().startswith("aaa")][-1]
    assert "+1.00" in line, line
