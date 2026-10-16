"""D-39 E2′ — D0 기록에 넣는 「고유 잡음」의 전제를 고정한다.

E2′ 의 판정은 **E2 와 seed 별로 짝짓는 것**에 서 있다. 그러려면 옵션을 안 줄 때
예전과 비트 단위로 같아야 하고, 줄 때는 잡음이 기록마다 고정이어야 하며, 「더하는
잡음」(NSTDB)과 종류가 겹치지 않아야 한다.
"""
import numpy as np
import pytest

from ecgdn.data import noise as noise_mod
from ecgdn.data.mixer import measure_snr
from ecgdn.data.sources import SyntheticSource, get_source
from ecgdn.data.nstdb import NSTDB_KINDS


def _src(**kw):
    return SyntheticSource(n_train=2, n_val=1, n_test=1, dur_s=30.0, **kw)


def test_off_is_bit_identical_to_before():
    a = _src().get("S000").x
    b = _src(ref_noise_snr=None).get("S000").x
    assert np.array_equal(a, b)


def test_on_mixes_at_exactly_that_snr_and_keeps_the_beats():
    clean = _src().get("S001")
    noisy = _src(ref_noise_snr=20.0).get("S001")
    n = noisy.x - clean.x
    # 이 저장소의 SNR 정의(평균 제거 파워)로 잰다 — mix_at_snr 와 같은 자
    assert abs(measure_snr(clean.x, n) - 20.0) < 1e-6
    # 형태 축은 그대로다 — R-peak 와 라벨이 같다
    assert np.array_equal(clean.r_peaks, noisy.r_peaks)
    assert np.array_equal(clean.symbols, noisy.symbols)


def test_noise_is_fixed_per_record():
    a = _src(ref_noise_snr=20.0).get("S002").x
    b = _src(ref_noise_snr=20.0).get("S002").x
    c = _src(ref_noise_snr=20.0).get("S003").x
    assert np.array_equal(a, b), "같은 기록인데 잡음이 다시 뽑혔다"
    assert not np.array_equal(a[:100], c[:100])


def test_never_uses_nstdb_kinds(monkeypatch):
    seen = []
    real = noise_mod.mixed_noise

    def spy(n, fs, gen, kinds=None, **kw):
        seen.append(tuple(kinds or ()))
        return real(n, fs, gen, kinds=kinds, **kw)

    monkeypatch.setattr(noise_mod, "mixed_noise", spy)
    _src(ref_noise_snr=20.0).get("S000")
    assert seen and not set(seen[0]) & set(NSTDB_KINDS)


def test_refuses_mitdb():
    from pathlib import Path
    if not any(Path("data/raw/mitdb").glob("*.hea")):
        pytest.skip("MIT-BIH 없음")
    with pytest.raises(ValueError):
        get_source("mitdb", ref_noise_snr=20.0)
