"""D-38 E1 암기 시험의 전제를 고정한다.

결과를 읽는 근거가 세 가지 전제 위에 서 있다 — ① 고정 잡음이 정말 고정이다,
② 「새 잡음」은 정말 다른 잡음이고 **같은 ECG** 다, ③ 결과 파일에 판정에 필요한
칸이 있다. 하나라도 무너지면 ①·② 의 차이가 외움이 아니라 배관 차이가 된다.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import memorize_test as mt  # noqa: E402
from train import build_datasets, load_cfg  # noqa: E402

CFG = ROOT / "configs" / "m06_l1_quarter.yaml"


def _small_ds():
    cfg = load_cfg(str(CFG))
    cfg["data"].update(source="synthetic", n_train=2, n_val=1, n_test=1, dur_s=60.0)
    _, tr, va = build_datasets(cfg)
    return tr, va


def test_pick_is_deterministic_and_sorted():
    a, b = mt.pick(500, 32, seed=0), mt.pick(500, 32, seed=0)
    assert a == b and a == sorted(a) and len(set(a)) == 32
    assert mt.pick(500, 32, seed=1) != a


def test_fixed_noise_is_fixed_and_fresh_noise_is_new_on_the_same_ecg():
    tr, _ = _small_ds()
    idx = mt.pick(len(tr), 8, seed=0)
    y1, x1, s1 = mt.stack(tr, idx)
    y2, _, _ = mt.stack(tr, idx)
    assert torch.equal(y1, y2), "set_epoch 없이 두 번 뽑았는데 잡음이 달라졌다"
    tr.set_epoch(mt.FRESH_EPOCH)
    y3, x3, s3 = mt.stack(tr, idx)
    assert not torch.equal(y1, y3), "새 잡음이 이전 잡음과 같다"
    # 참조는 물리 스케일에서 같아야 한다 (정규화는 noisy 의 scale 로 해서 달라진다)
    assert np.allclose(x1.numpy() * s1[:, None, None], x3.numpy() * s3[:, None, None],
                       rtol=1e-5, atol=1e-6)


def test_run_writes_the_columns_the_verdict_needs(tmp_path):
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "memorize_test.py"), "-c", str(CFG),
         "--out", str(tmp_path), "--source", "synthetic", "--n", "16",
         "--steps", "4", "--every", "2", "--batch", "8", "--threads", "1"],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    s = json.loads((tmp_path / "m06_l1_quarter" / "summary.json").read_text())
    for k in ("n_params", "min_fixed_mse", "final", "weight_decay", "torch"):
        assert k in s
    assert s["weight_decay"] == 0.0
    for k in ("fixed_mse", "fresh_mse", "val_mse"):
        assert k in s["final"]
    head = (tmp_path / "m06_l1_quarter" / "log.csv").read_text().splitlines()[0]
    assert "fresh_mse" in head and "val_mse" in head


def _run(out, *extra):
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "memorize_test.py"), "-c", str(CFG),
         "--out", str(out), "--source", "synthetic", "--n", "16", "--steps", "6",
         "--every", "2", "--batch", "8", "--threads", "1", *extra],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_resume_gives_the_same_run_as_no_interruption(tmp_path):
    """하네스가 백그라운드를 약 30 분에 끊는다 — 끊겼다 이어도 **같은 판**이어야 한다 (O-19)."""
    _run(tmp_path / "a")
    out = _run(tmp_path / "b", "--stop-after", "2")
    assert "재개점만" in out
    assert (tmp_path / "b" / "m06_l1_quarter" / "last.pt").exists()
    assert not (tmp_path / "b" / "m06_l1_quarter" / "summary.json").exists()
    assert "재개" in _run(tmp_path / "b")
    sa = json.loads((tmp_path / "a" / "m06_l1_quarter" / "summary.json").read_text())
    sb = json.loads((tmp_path / "b" / "m06_l1_quarter" / "summary.json").read_text())
    for k in ("fixed_mse", "fresh_mse", "val_mse"):
        assert abs(sa["final"][k] - sb["final"][k]) < 1e-7, (k, sa["final"][k], sb["final"][k])
    assert not (tmp_path / "b" / "m06_l1_quarter" / "last.pt").exists(), "끝났는데 재개점이 남았다"
    rows = (tmp_path / "b" / "m06_l1_quarter" / "log.csv").read_text().splitlines()
    assert len(rows) == 1 + 3, "재개 뒤 log.csv 가 지난 줄을 잃었거나 겹쳤다"
