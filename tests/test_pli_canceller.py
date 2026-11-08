"""전원선 고조파 추적 제거기 (D-52) — 노치가 못 하던 자리를 실제로 메우는가.

잡음은 실보드에서 본 크기(60 Hz 가 ECG p-p 의 3 배)에 SMPS 형 고조파를 얹는다.
250 Hz 로 직접 샘플하므로 3 · 4 · 5 차 이상은 나이퀴스트 아래로 **접혀** 들어온다.
"""
import numpy as np
import pytest
from scipy import signal as sps

from ecgdn.data.synthetic import synth_ecg
from ecgdn.realtime.frontend_modes import PLI_MODES, build_fe
from ecgdn.realtime.pli_canceller import HarmonicCanceller, PliRemover

FS = 250.0
HARM = {1: 1.0, 3: 0.30, 4: 0.05, 5: 0.15, 7: 0.08, 9: 0.04}


def _ecg(dur=40.0, seed=3):
    c = synth_ecg(dur, fs=FS, seed=seed).x
    return c, np.ptp(c)


def _pli(n, amp, f0=60.0, on_s=0.0):
    t = np.arange(n) / FS
    f = f0(t) if callable(f0) else np.full(n, float(f0))
    ph = 2 * np.pi * np.cumsum(f) / FS
    return (t >= on_s) * amp * sum(a * np.sin(k * ph + k) for k, a in HARM.items())


def _notch(x):
    for f in (60.0, 120.0):
        b, a = sps.iirnotch(f, 30, FS)
        x = sps.lfilter(b, a, x)
    return x


def _pp(x, c, A, from_s=8.0):
    s = slice(int(from_s * FS), None)
    return float(np.ptp((x - c)[s]) / A)


@pytest.mark.parametrize("f0", [60.0, 59.8, 60.2])
def test_removes_drifting_mains_with_folded_harmonics(f0):
    """고정 노치는 ±0.2 Hz · 접힌 고조파에서 ECG 보다 큰 잔류를 남긴다. 제거기는 그 1/10 아래로."""
    c, A = _ecg()
    y = c + _pli(c.size, 1.5 * A, f0)
    notch = _pp(_notch(y), c, A)
    hc = _pp(HarmonicCanceller(FS).push(y), c, A)
    assert notch > 1.0                       # 노치의 한계가 시험 조건에 실제로 있다
    assert hc < 0.15 and hc < notch / 10


def test_follows_clock_error():
    """보드 클럭이 0.5 % 틀리면 화면의 60 Hz 는 59.7 Hz 로 보인다 — f 가 따라가야 한다."""
    c, A = _ecg()
    y = c + _pli(c.size, 1.5 * A, 60.0 / 1.005)
    h = HarmonicCanceller(FS)
    out = h.push(y)
    assert abs(h.f - 60.0 / 1.005) < 0.02
    assert _pp(out, c, A) < 0.15


def test_reacquires_when_mains_appears_later():
    """시작 뒤에 충전기를 꽂는 경우 — D-50 이전 노치가 놓치던 자리."""
    c, A = _ecg()
    y = c + _pli(c.size, 1.5 * A, 60.15, on_s=20.0)
    out = HarmonicCanceller(FS).push(y)
    assert _pp(out, c, A, from_s=25.0) < 0.15


def test_leaves_clean_ecg_alone():
    """잡음이 없을 때 ECG 를 바꾸는 양이 노치 Q30 보다 크지 않다."""
    c, A = _ecg()
    hc = _pp(HarmonicCanceller(FS).push(c), c, A)
    assert hc <= _pp(_notch(c), c, A) + 0.005
    assert hc < 0.04


def test_remover_tolerates_baseline_wander():
    """기저선이 크면 혁신이 커져 판정이 흔들린다 — PliRemover 는 HPF 사본에서 추정한다."""
    c, A = _ecg()
    t = np.arange(c.size) / FS
    base = 0.5 * A * np.sin(2 * np.pi * 0.3 * t)
    y = c + base + _pli(c.size, 1.5 * A, 59.85)
    out = PliRemover(FS).push(y)
    ref = PliRemover(FS).push(c + base)
    assert _pp(out, ref, A) < 0.3
    assert _pp(_notch(y), _notch(c + base), A) > 1.0


@pytest.mark.parametrize("mode", ["causal", "zerophase", "median"])
def test_every_fe_mode_uses_the_canceller_by_default(mode):
    """세 front-end 모드가 모두 기본으로 제거기를 쓰고, 블록으로 나눠 넣어도 결과가 같다."""
    c, A = _ecg(30.0)
    y = c + _pli(c.size, 1.5 * A, 59.8)
    outs = []
    for sig in (y, c):
        fe = build_fe(mode, FS)
        outs.append(np.concatenate([fe.push(sig[i:i + 16]) for i in range(0, sig.size, 16)]))
    m = min(map(len, outs))
    assert _pp(outs[0][:m], outs[1][:m], A, from_s=10.0) < 0.2
    fe_notch = build_fe(mode, FS, pli="notch")
    on = np.concatenate([fe_notch.push(y[i:i + 16]) for i in range(0, y.size, 16)])
    assert _pp(on[:m], outs[1][:m], A, from_s=10.0) > 0.5     # 노치는 여기서 못 뺀다


def test_unknown_pli_mode_names_the_choices():
    with pytest.raises(ValueError, match="canceller"):
        build_fe("causal", FS, pli="wavelet")
    assert set(PLI_MODES) == {"canceller", "notch"}


def test_block_size_does_not_change_output():
    c, A = _ecg(20.0)
    y = c + _pli(c.size, 1.5 * A, 59.9)
    whole = PliRemover(FS).push(y)
    r = PliRemover(FS)
    parts = np.concatenate([r.push(y[i:i + 7]) for i in range(0, y.size, 7)])
    assert np.allclose(whole, parts)
