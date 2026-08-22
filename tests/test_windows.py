import numpy as np
import pytest

from ecgdn.data.windows import frame, overlap_add, process_framed, analysis_window
from ecgdn.config import WIN, HOP


@pytest.mark.parametrize("n", [1, 7, 100, 511, 512, 1023, 1024, 1025, 5000, 7777])
def test_perfect_reconstruction(n):
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n)
    frames, pl, pr = frame(x, WIN, HOP)
    y = overlap_add(frames, n, pl, HOP)
    assert y.shape == x.shape
    assert np.max(np.abs(y - x)) < 1e-10, np.max(np.abs(y - x))


@pytest.mark.parametrize("n", [333, 1024, 4321])
def test_process_framed_identity(n):
    rng = np.random.default_rng(n + 1)
    x = rng.standard_normal(n)
    y = process_framed(x, lambda f: f, WIN, HOP)
    assert np.max(np.abs(y - x)) < 1e-10


def test_window_cola():
    """sqrt(Hann) 50% overlap 에서 w^2 합이 상수."""
    w = analysis_window(WIN) ** 2
    acc = np.zeros(4 * WIN)
    for k in range(0, 3 * WIN, HOP):
        acc[k:k + WIN] += w
    mid = acc[WIN:2 * WIN]
    assert np.allclose(mid, 1.0, atol=1e-12)


def test_small_window():
    x = np.arange(64, dtype=float)
    f, pl, pr = frame(x, 16, 8)
    y = overlap_add(f, 64, pl, 8)
    assert np.max(np.abs(y - x)) < 1e-10
