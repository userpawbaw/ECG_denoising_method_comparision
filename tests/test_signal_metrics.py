import numpy as np
import pytest

from ecgdn.data.synthetic import synth_ecg
from ecgdn.eval.signal_metrics import (metrics_signal, optimal_gain, prdn,
                                       pearson_cc, rmse, snr_out_scaled,
                                       snr_out_strict)


@pytest.fixture(scope="module")
def x():
    return synth_ecg(20.0, seed=3).x


def test_perfect_reconstruction(x):
    m = metrics_signal(x, None, x.copy())
    assert np.isinf(m["snr_out_strict"]) and m["snr_out_strict"] > 0
    assert np.isinf(m["snr_out_scaled"]) and m["snr_out_scaled"] > 0
    assert m["rmse"] == 0.0
    assert abs(m["cc"] - 1.0) < 1e-12
    assert abs(m["gain_bias"] - 1.0) < 1e-12
    assert m["prdn"] == 0.0


def test_scale_bias_is_separated(x):
    """A-1 의 핵심: 0.5 배 축소는 strict 에서 6.02 dB, scaled 에서 inf."""
    xhat = 0.5 * x
    assert abs(snr_out_strict(x, xhat) - 10 * np.log10(4.0)) < 1e-9
    assert np.isinf(snr_out_scaled(x, xhat))
    assert abs(optimal_gain(x, xhat) - 2.0) < 1e-12


def test_dc_offset_invariance(x):
    """DC 규약: 상수 오프셋은 모든 지표에 영향을 주지 않는다."""
    xhat = x + 0.37
    m = metrics_signal(x, None, xhat)
    assert m["snr_out_strict"] > 200.0            # 부동소수 오차만 남음
    assert abs(m["cc"] - 1.0) < 1e-12
    assert m["prdn"] == pytest.approx(0.0, abs=1e-9)
    assert m["rmse"] == pytest.approx(0.0, abs=1e-12)


def test_all_metrics_dc_consistent(x):
    """x 와 xhat 에 서로 다른 오프셋을 줘도 지표가 동일해야 한다."""
    rng = np.random.default_rng(9)
    xhat = x + 0.02 * rng.standard_normal(x.size)
    a = metrics_signal(x, None, xhat)
    b = metrics_signal(x + 5.0, None, xhat - 3.0)
    for k in a:
        if np.isfinite(a[k]):
            assert abs(a[k] - b[k]) < 1e-8, k


def test_snr_improvement_of_identity(x):
    rng = np.random.default_rng(0)
    n = rng.standard_normal(x.size)
    n *= np.sqrt(np.var(x) / np.var(n) / 10 ** (10 / 10))   # 10 dB
    y = x + n
    m = metrics_signal(x, y, y)          # identity denoiser
    assert abs(m["snr_in"] - 10.0) < 1e-6
    assert abs(m["snr_imp_strict"]) < 1e-9


def test_prdn_matches_definition(x):
    rng = np.random.default_rng(4)
    xhat = x + 0.01 * rng.standard_normal(x.size)
    # DC 규약: 분자도 평균 제거 후 오차
    x0, h0 = x - x.mean(), xhat - xhat.mean()
    expected = 100 * np.sqrt(np.sum((x0 - h0) ** 2) / np.sum(x0 ** 2))
    assert abs(prdn(x, xhat) - expected) < 1e-9


def test_cc_is_scale_invariant(x):
    assert abs(pearson_cc(x, 3.0 * x + 1.0) - 1.0) < 1e-12
