import numpy as np
import pytest

import ecgdn.methods  # noqa: F401  (레지스트리 등록)
from ecgdn.config import SWTCfg
from ecgdn.data.mixer import mix_at_snr
from ecgdn.data.noise import awgn, mixed_noise
from ecgdn.data.synthetic import synth_ecg
from ecgdn.eval.engine import evaluate
from ecgdn.methods.wavelet import (DWTDenoiser, SWTDenoiser, mad_sigma,
                                   qrs_protection_mask, threshold)
from ecgdn.registry import build
from ecgdn.utils import rng


def test_mad_sigma_on_gaussian():
    x = np.random.default_rng(0).standard_normal(200000)
    assert abs(mad_sigma(x) - 1.0) < 0.02


@pytest.mark.parametrize("mode", ["soft", "hard", "garrote"])
def test_threshold_zero_lambda_is_identity(mode):
    d = np.random.default_rng(1).standard_normal(1000)
    assert np.allclose(threshold(d, 0.0, mode), d)


def test_soft_shrinks_garrote_less():
    """soft 는 큰 계수에서도 lambda 만큼 줄인다. garrote 는 거의 줄이지 않는다."""
    d = np.array([10.0])
    lam = 1.0
    assert threshold(d, lam, "soft")[0] == pytest.approx(9.0)
    assert threshold(d, lam, "garrote")[0] == pytest.approx(10.0 * (1 - 0.01))
    assert threshold(d, lam, "hard")[0] == pytest.approx(10.0)


def test_qrs_protection_mask():
    g = qrs_protection_mask(1000, np.array([500]), 250.0, 60.0)
    assert g[500] == pytest.approx(1.0)
    assert g[0] == 0.0 and g[-1] == 0.0
    assert np.all((g >= 0) & (g <= 1))


@pytest.mark.parametrize("cls", [SWTDenoiser, DWTDenoiser])
def test_zero_k_is_perfect_reconstruction(cls):
    """k=0 이면 threshold 가 걸리지 않아 완전 재구성되어야 한다."""
    s = synth_ecg(30.0, seed=3)
    d = cls(SWTCfg(k=(0.0,) * 5, protect_qrs=False), use_frontend=False)
    out = d(s.x, s.fs)
    assert np.max(np.abs(out - s.x)) < 1e-9


def test_swt_beats_identity_on_synthetic():
    """STEP 11 DoD: 합성 ECG SNR 5 dB 에서 snr_imp_scaled > 4 dB."""
    s = synth_ecg(90.0, seed=4)
    n, _ = mixed_noise(len(s.x), s.fs, rng("wv"))
    y, _, _ = mix_at_snr(s.x, n, 5.0)
    m = evaluate(s.x, y, build("M04")(y, s.fs), s.fs, r_peaks_ref=s.r_peaks,
                 do_morph=False, do_spectral=False)
    assert m["snr_imp_scaled"] > 4.0, m["snr_imp_scaled"]


def test_oracle_beats_practical_wavelet():
    """STEP 11 DoD: B01(oracle) 이 M04 보다 항상 높아야 한다 (아니면 구현 버그)."""
    s = synth_ecg(90.0, seed=6)
    y, _, _ = mix_at_snr(s.x, awgn(len(s.x), s.fs, rng("wv2")), 10.0)
    m4 = evaluate(s.x, y, build("M04")(y, s.fs), s.fs, r_peaks_ref=s.r_peaks,
                  do_morph=False, do_spectral=False)
    b1 = build("B01")
    m1 = evaluate(s.x, y, b1(y, s.fs, {"x_clean": s.x}), s.fs, r_peaks_ref=s.r_peaks,
                  do_morph=False, do_spectral=False)
    assert m1["snr_imp_scaled"] > m4["snr_imp_scaled"]


def test_oracle_requires_clean():
    s = synth_ecg(20.0, seed=7)
    with pytest.raises(ValueError):
        build("B01")(s.x, s.fs)          # ctx 없음
