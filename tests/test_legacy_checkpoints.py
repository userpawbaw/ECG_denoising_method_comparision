"""주 실험의 체크포인트가 **지금 코드로 열리는가** (O-38).

`n_blocks`(D-27)를 넣으며 인코더 키가 `enc.{i}.c1` → `enc.{i}.0.c1` 로 바뀌었고,
그 전에 학습된 주 체크포인트가 전부 안 열렸다. 다른 시험은 실제 체크포인트를
읽지 않아 이것을 못 봤고, 실시간 시연 브리지가 `M06` 을 켜는 순간 죽었다.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tracked_ckpts() -> list[str]:
    out = subprocess.run(["git", "ls-files", "results/d0", "results/d1"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    return sorted(p for p in out if p.endswith("/best.pt"))


# 모델 종류마다 하나 — 기본 묶음에서 돈다. 전부는 `-m slow` (약 75 초).
REPRESENTATIVE = {"results/d1/m06_l1/best.pt", "results/d1/m08_l1/best.pt",
                  "results/d1/m09_l1/best.pt", "results/d1/m10_l1/best.pt"}


@pytest.mark.parametrize("ckpt", sorted(REPRESENTATIVE))
def test_representative_checkpoint_loads(ckpt: str):
    pytest.importorskip("torch")
    from ecgdn.methods.dl_wrapper import load_checkpoint
    load_checkpoint(ROOT / ckpt)


@pytest.mark.slow
@pytest.mark.parametrize("ckpt", [c for c in _tracked_ckpts() if c not in REPRESENTATIVE])
def test_tracked_checkpoint_loads(ckpt: str):
    pytest.importorskip("torch")
    from ecgdn.methods.dl_wrapper import load_checkpoint
    load_checkpoint(ROOT / ckpt)


def test_partial_remap_is_refused():
    """키가 일부만 맞는 재매핑은 하지 않는다 — 원래 오류가 그대로 나야 한다."""
    torch = pytest.importorskip("torch")
    from ecgdn.models import ResUNet1D, upgrade_state_dict
    m = ResUNet1D()
    sd = {k.replace(".0.c1.", ".c1."): v for k, v in m.state_dict().items()}
    sd.pop(next(iter(sd)))                          # 하나를 빼서 부분 일치로 만든다
    assert upgrade_state_dict(m, sd) is sd
    with pytest.raises(RuntimeError):
        m.load_state_dict(sd)
    assert isinstance(torch.zeros(1), torch.Tensor)


def test_legacy_layout_round_trips():
    pytest.importorskip("torch")
    from ecgdn.models import ResUNet1D, upgrade_state_dict
    m = ResUNet1D()
    new = m.state_dict()
    old = {k.replace(".0.c1.", ".c1.").replace(".0.c2.", ".c2.")
            .replace(".0.n1.", ".n1.").replace(".0.n2.", ".n2.")
           if k.startswith("enc.") else k: v for k, v in new.items()}
    assert set(old) != set(new)
    assert set(upgrade_state_dict(m, old)) == set(new)
