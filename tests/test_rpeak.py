import numpy as np

from ecgdn.config import FS
from ecgdn.data.synthetic import synth_ecg
from ecgdn.eval.rpeak import detect_rpeaks, hr_from_rr, match_peaks, metrics_rpeak


def test_match_identical():
    ref = np.array([100, 350, 600, 850])
    pairs, tp, fp, fn = match_peaks(ref, ref, FS)
    assert tp == 4 and fp == 0 and fn == 0
    assert np.array_equal(pairs[:, 0], pairs[:, 1])


def test_match_is_one_to_one():
    """tolerance 안에 2개가 들어와도 1:1 이어야 한다."""
    ref = np.array([1000])
    test = np.array([1000, 1005, 1010])       # 전부 75 ms 이내
    pairs, tp, fp, fn = match_peaks(ref, test, FS)
    assert tp == 1 and fp == 2 and fn == 0


def test_match_beyond_tolerance():
    ref = np.array([1000])
    test = np.array([1000 + int(0.2 * FS)])   # 200 ms > 75 ms
    _, tp, fp, fn = match_peaks(ref, test, FS)
    assert tp == 0 and fp == 1 and fn == 1


def test_metrics_perfect():
    s = synth_ecg(60.0, seed=11)
    m = metrics_rpeak(s.x, s.x, s.fs, r_peaks_ref=s.r_peaks)
    assert m["se"] > 0.99 and m["ppv"] > 0.99
    assert m["rpeak_mae_ms"] < 10.0
    assert abs(m["hr_err_bpm"]) < 0.5


def test_timing_bias_detected():
    s = synth_ecg(60.0, seed=12)
    shift = int(round(0.008 * s.fs))          # +8 ms
    xhat = np.roll(s.x, shift)
    m = metrics_rpeak(s.x, xhat, s.fs, r_peaks_ref=s.r_peaks)
    assert abs(m["rpeak_bias_ms"] - 8.0) < 4.0


def test_hr_from_rr():
    assert abs(hr_from_rr(np.full(10, 60 / 75.0)) - 75.0) < 1e-9
