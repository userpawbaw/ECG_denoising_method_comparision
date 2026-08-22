import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecgdn.data.synthetic import synth_ecg  # noqa: E402
from ecgdn.methods.dl_wrapper import DLDenoiser  # noqa: E402
from ecgdn.models import build_model  # noqa: E402


class _Identity(torch.nn.Module):
    def forward(self, x):
        return x


class _Half(torch.nn.Module):
    def forward(self, x):
        return 0.5 * x


@pytest.mark.parametrize("n", [1024, 3000, 12345])
def test_length_preserved_and_identity(n):
    x = np.random.default_rng(n).standard_normal(n)
    d = DLDenoiser(model=_Identity(), name="M06")
    out = d(x, 250.0)
    assert out.shape == x.shape
    assert np.max(np.abs(out - x)) < 1e-4


def test_scaling_is_undone():
    """모델이 0.5배로 만들면 출력도 정확히 0.5배여야 한다 (정규화가 새는지 확인)."""
    x = synth_ecg(30.0, seed=2).x
    d = DLDenoiser(model=_Half(), name="M06")
    out = d(x, 250.0)
    assert np.max(np.abs(out - 0.5 * x)) < 1e-4


def test_no_boundary_discontinuity():
    """hop 배수 위치에서 1차 차분이 튀지 않아야 한다."""
    s = synth_ecg(40.0, seed=3)
    m = build_model("resunet1d")
    torch.manual_seed(0)
    for p in m.parameters():                   # 무작위 비선형 응답을 만들기 위해
        p.data.add_(0.01 * torch.randn_like(p))
    d = DLDenoiser(model=m, name="M06", batch=8)
    out = d(s.x, s.fs)
    dif = np.abs(np.diff(out))
    hops = np.arange(d.hop, len(dif) - 1, d.hop)
    bg = np.std(dif)
    assert np.max(dif[hops]) < np.max(dif) + 1e-12
    assert np.mean(dif[hops]) < np.mean(dif) + 3 * bg


def test_real_model_shapes():
    s = synth_ecg(20.0, seed=4)
    for name in ("resunet1d", "wavelet_unet"):
        d = DLDenoiser(model=build_model(name), name="M06", batch=4)
        out = d(s.x, s.fs)
        assert out.shape == s.x.shape and np.all(np.isfinite(out))
