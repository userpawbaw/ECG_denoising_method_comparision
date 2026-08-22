import numpy as np
import pytest

from ecgdn.data.mixer import mix_at_snr, measure_snr
from ecgdn.data.noise import NOISE_FNS
from ecgdn.data.synthetic import synth_ecg
from ecgdn.utils import rng


@pytest.mark.parametrize("snr", [-5, 0, 5, 10, 15, 20])
@pytest.mark.parametrize("kind", list(NOISE_FNS))
def test_mix_hits_target_snr(snr, kind):
    s = synth_ecg(20.0, seed=1)
    g = rng("mix", kind, snr)
    n = NOISE_FNS[kind](len(s.x), s.fs, g)
    y, ns, actual = mix_at_snr(s.x, n, snr)
    assert abs(actual - snr) < 1e-6
    assert np.max(np.abs(y - (s.x + ns))) < 1e-12


def test_snr_offset_invariant():
    s = synth_ecg(10.0, seed=2)
    g = rng("m2")
    n = NOISE_FNS["awgn"](len(s.x), s.fs, g)
    a = measure_snr(s.x, n)
    b = measure_snr(s.x + 3.7, n + 100.0)
    assert abs(a - b) < 1e-9
