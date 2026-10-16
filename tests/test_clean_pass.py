"""D-40 — `L1C` 손실과 `clean_pass_snr` 지표의 전제를 고정한다."""
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ecgdn.models.losses import make_loss  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _sweep():
    import sys
    if str(ROOT / "scripts") not in sys.path:          # `_bootstrap` 을 찾게
        sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "_sw", ROOT / "scripts" / "run_seed_sweep.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_l1c_is_l1_plus_half_the_clean_term():
    torch.manual_seed(0)
    x = torch.randn(8, 1, 64)
    xh = x + 0.1 * torch.randn_like(x)
    xc = x[:2] + 0.3 * torch.randn_like(x[:2])
    l1, _ = make_loss("L1")(xh, x, None, None)
    l1c = make_loss("L1C")
    assert l1c.needs_clean_pass and abs(l1c.clean_frac - 0.25) < 1e-12
    tot, parts = l1c(xh, x, None, None, xhat_clean=xc)
    clean = torch.nn.functional.mse_loss(xc, x[:2])
    assert torch.allclose(tot, l1 + 0.5 * clean, atol=1e-7)


class _DS:
    """고정 평가 데이터셋 흉내 — (y, x, meta) 를 낸다."""
    def __init__(self, n=5):
        g = torch.Generator().manual_seed(1)
        self.x = [torch.randn(1, 128, generator=g).numpy() for _ in range(n)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.x[i], {}


def test_clean_pass_snr_identity_is_huge_and_zero_output_is_zero_db():
    sw = _sweep()

    class Id(torch.nn.Module):
        def forward(self, v):
            return v

    class Zero(torch.nn.Module):
        def forward(self, v):
            return torch.zeros_like(v)

    ds = _DS()
    assert sw.clean_pass_snr(Id(), ds, "cpu") > 100
    assert abs(sw.clean_pass_snr(Zero(), ds, "cpu")) < 1e-5


def test_clean_pass_is_opt_in():
    """안 주면 열 구성이 예전과 같아야 한다 — 돌고 있는 sweep 이 재개될 때 섞인다."""
    src = (ROOT / "scripts" / "run_seed_sweep.py").read_text()
    assert "if clean_pass and" in src
    assert 'action="store_true"' in src.split('"--clean-pass"')[1][:80]
