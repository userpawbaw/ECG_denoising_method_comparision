"""M05S — 스트리밍 Sameni EKF (D-40). 오프라인 경로와 실시간 브리지 연결을 고정한다."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ecgdn.data.mixer import mix_at_snr
from ecgdn.data.noise import mixed_noise
from ecgdn.data.synthetic import synth_ecg
from ecgdn.methods import build
from ecgdn.methods.frontend import FrontEnd
from ecgdn.methods.kalman_stream import StreamingSameni

ROOT = Path(__file__).resolve().parent.parent
FS = 250


def _case(seed=1, snr_in=6.0, dur=24):
    clean = synth_ecg(duration_s=dur, fs=FS, seed=seed).x
    noise, _ = mixed_noise(clean.size, FS, np.random.default_rng(100 + seed))
    y, _, _ = mix_at_snr(clean, noise, snr_in)
    return clean, y


def _snr(ref, x):
    return 10 * np.log10(np.sum(ref ** 2) / np.sum((ref - x) ** 2))


def test_m05s_keeps_length_and_beats_the_front_end():
    """스윕(D-40)에서 입력 6 dB 평균: FE 7.9 · M05S 14.9 · M05 16.4. 여기서는 «FE 보다 3 dB 넘게» 만 고정한다."""
    clean, y = _case()
    fe = FrontEnd()
    ref = fe(clean, FS)
    m = build("M05S")
    t = time.perf_counter()
    out = m(y, FS)
    dt = time.perf_counter() - t
    assert out.shape == y.shape and np.all(np.isfinite(out))
    assert m.last_info["initialized"], "적합(init)이 안 됐다"
    sl = slice(8 * FS, 22 * FS)
    gain = _snr(ref[sl], out[sl]) - _snr(ref[sl], fe(y, FS)[sl])
    assert gain > 3.0, f"M05S 가 front-end 보다 {gain:.2f} dB 밖에 안 낫다"
    assert dt / (y.size / FS) < 0.5, f"오프라인 RTF {dt / (y.size / FS):.2f} — 스트리밍인데 너무 느리다"


def test_the_rr_phase_noise_is_what_makes_it_work():
    """ω 를 외삽하는 대가를 위상 과정잡음에 넣지 않으면 나빠진다 (D-40 결과: 12 경우 평균 14.5 -> 3.6 dB).

    경우마다 편차가 커서, 이 한 경우(seed 1 · 6 dB)는 10.3 대 8.8 dB 다 — 문턱을 1 dB 로 둔다."""
    clean, y = _case()
    fe = FrontEnd()
    ref = fe(clean, FS)
    sl = slice(8 * FS, 22 * FS)
    on = build("M05S")(y, FS)
    off = build("M05S", q_rr=False)(y, FS)
    assert _snr(ref[sl], on[sl]) > _snr(ref[sl], off[sl]) + 1.0


def test_streaming_core_speaks_the_bridge_interface():
    s = StreamingSameni(FS)
    for attr in ("push", "reset", "origin", "latency_s", "warmup_s", "runs_per_s", "hop", "n_runs"):
        assert hasattr(s, attr), attr
    assert abs(s.latency_s - (36 + 12 + 12) / FS) < 1e-9     # 기본값 240 ms (D-40)


def test_bridge_presets_m05_hop_and_builds_m05s_as_a_stream():
    src = (ROOT / "scripts" / "serial_bridge.py").read_text()
    assert 'DEFAULT_HOP = {"M05": 128, "M05f": 128}' in src
    assert "hop_for.get(n, DEFAULT_HOP.get(n, args.hop))" in src
    assert "StreamingSameni(FS, hop=hop_n, background=True)" in src


def test_fitting_never_blocks_the_stream_in_background_mode():
    """실보드에서 «시작 20 초 안에 멈췄다가 돌아온다» 였다 (O-38). init 순간 본체에서 wfdb import
    (약 1 s) + XQRS + least_squares 가 돌아 push 하나가 0.5~1.9 s 걸렸다. 백그라운드면 짧아야 한다."""
    _, y = _case(seed=0, snr_in=0.0, dur=20)
    x = FrontEnd()(y, FS)
    s = StreamingSameni(FS, background=True)
    worst = 0.0
    for i in range(0, x.size, 12):
        t = time.perf_counter()
        s.push(x[i:i + 12])
        worst = max(worst, time.perf_counter() - t)
        time.sleep(0.0005)                 # 작업 스레드에 숨 쉴 틈 (실시간이면 48 ms 다)
    t_end = time.perf_counter() + 10.0
    while s.p is None and time.perf_counter() < t_end:
        s.push(np.zeros(0))
        time.sleep(0.01)
    assert s.p is not None, "백그라운드 적합이 끝나지 않았다"
    assert worst < 0.3, f"push 하나가 {worst * 1000:.0f} ms 막혔다 — 적합이 본체에서 돈다"
