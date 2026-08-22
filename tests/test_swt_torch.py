"""STEP 21 착수 조건: pywt 와 1e-6 이내 일치 + gradcheck 통과."""
import numpy as np
import pytest
import pywt
import torch

from ecgdn.models.swt_torch import TorchISWT, TorchSWT


@pytest.mark.parametrize("wavelet", ["sym4", "db4", "sym6", "coif2"])
@pytest.mark.parametrize("level", [1, 3, 5])
def test_forward_matches_pywt(wavelet, level):
    n = 1024
    x = np.random.default_rng(level).standard_normal(n)
    ref = np.stack(pywt.swt(x, wavelet, level=level, trim_approx=True, norm=False))
    out = TorchSWT(wavelet, level)(torch.tensor(x)[None, None, :]).numpy()[0]
    assert out.shape == ref.shape
    err = float(np.max(np.abs(out - ref)))
    assert err < 1e-6, err


@pytest.mark.parametrize("wavelet", ["sym4", "db4", "sym6"])
@pytest.mark.parametrize("level", [1, 3, 5])
def test_inverse_roundtrip(wavelet, level):
    n = 1024
    x = torch.tensor(np.random.default_rng(level + 10).standard_normal(n))[None, None, :]
    c = TorchSWT(wavelet, level)(x)
    xr = TorchISWT(wavelet, level)(c)
    assert float(torch.max(torch.abs(xr - x))) < 1e-6


def test_inverse_matches_pywt_iswt():
    n, wavelet, level = 1024, "sym4", 5
    x = np.random.default_rng(99).standard_normal(n)
    c = pywt.swt(x, wavelet, level=level, trim_approx=True, norm=False)
    ref = np.asarray(pywt.iswt(c, wavelet, norm=False))
    ct = torch.tensor(np.stack(c))[None]
    out = TorchISWT(wavelet, level)(ct).numpy()[0, 0]
    assert float(np.max(np.abs(out - ref))) < 1e-6


def test_gradcheck():
    torch.manual_seed(0)
    x = torch.randn(1, 1, 64, dtype=torch.float64, requires_grad=True)
    swt, iswt = TorchSWT("db2", 2), TorchISWT("db2", 2)
    assert torch.autograd.gradcheck(lambda v: iswt(swt(v)).sum(), (x,), eps=1e-6,
                                    atol=1e-8)
    assert torch.autograd.gradcheck(lambda v: swt(v).pow(2).sum(), (x,), eps=1e-6,
                                    atol=1e-8)


def test_batch_and_float32():
    x = torch.randn(4, 1, 1024)
    c = TorchSWT("sym4", 5)(x)
    assert c.shape == (4, 6, 1024) and c.dtype == torch.float32
    xr = TorchISWT("sym4", 5)(c)
    assert float(torch.max(torch.abs(xr - x))) < 1e-3


def test_length_must_be_multiple():
    with pytest.raises(ValueError):
        TorchSWT("sym4", 5)(torch.randn(1, 1, 1000))
