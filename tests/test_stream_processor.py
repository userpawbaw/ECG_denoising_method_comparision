"""R-4 스트리밍 처리기.

시연이 보고서와 **다른 것**을 보이면 아무도 알아채지 못한다. 그래서 이
파일이 고정하는 것은 세 가지다 — 배관이 신호를 건드리지 않는가, 지연
관계식이 지켜지는가, 그리고 **front-end 가 두 번 걸리지 않는가.**
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ecgdn.methods  # noqa: E402,F401  레지스트리 등록
from ecgdn.realtime import StreamProcessor  # noqa: E402
from ecgdn.realtime.frontend_stream import StreamingFrontEnd  # noqa: E402
from ecgdn.registry import build  # noqa: E402

FS = 250.0


def _sig(n: int = 6000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    return (np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 17 * t)
            + 0.05 * rng.normal(size=n))


# ------------------------------------------------------------------ 배관
def test_identity_method_streams_the_signal_unchanged():
    """항등 방법을 흘리면 **입력이 그대로 나와야** 한다.

    배관(링버퍼 · Hann² OLA · 내보내기 규칙)이 신호를 건드리는지 보는 검사다.
    이것이 통과하지 않으면 뒤의 어떤 수치도 방법의 성능이 아니다.
    """
    y = _sig()
    sp = StreamProcessor(build("M00"), fs=FS, win=1024, hop=12, d=12,
                         frontend="none")
    out = sp.run(y, block=25)
    lo = sp.origin + sp.win          # OLA 가중이 다 쌓인 뒤부터
    a = out[lo - sp.origin:]
    b = y[lo:lo + a.size]
    assert a.size > FS * 5, "출력이 너무 짧다"
    assert np.max(np.abs(a - b)) < 1e-9, "배관이 신호를 바꿨다"


def test_block_size_does_not_change_the_output():
    """시리얼 패킷 크기가 결과를 바꾸면 안 된다 — `hop` 의 배수가 아니어도."""
    y = _sig()
    outs = []
    for blk in (1, 7, 25, 64, 250):
        sp = StreamProcessor(build("M00"), fs=FS, win=1024, hop=12, d=12,
                             frontend="none")
        outs.append(sp.run(y, block=blk))
    # 끝부분은 블록 경계에 따라 마지막 window 가 돌았는지가 달라 OLA 가중이
    # 덜 쌓여 있다. 그 ramp-out 을 뺀 구간을 본다.
    n = min(o.size for o in outs) - 1024
    for o in outs[1:]:
        assert np.max(np.abs(o[:n] - outs[0][:n])) < 1e-12


# ------------------------------------------------------------------ 지연
def test_latency_is_d_plus_hop():
    """`L = (d + hop)/fs` (30_realtime_demo 2.2). `d` 만 줄여도 소용없다."""
    sp = StreamProcessor(build("M00"), fs=FS, win=1024, hop=13, d=12,
                         frontend="none")
    assert sp.latency_s == pytest.approx(25 / FS)
    assert sp.runs_per_s == pytest.approx(FS / 13)


def test_output_never_runs_ahead_of_the_promised_latency():
    """샘플 `p` 는 `p + d + hop` 이 도착하기 **전에** 나오면 안 된다.

    앞질러 내보내면 그 샘플은 약속한 미래 문맥을 못 받은 것이다.
    """
    sp = StreamProcessor(build("M00"), fs=FS, win=512, hop=16, d=8,
                         frontend="none")
    y = _sig(4000)
    emitted, fed = sp.origin, 0
    for i in range(0, y.size, 10):
        blk = y[i:i + 10]
        fed += blk.size
        emitted += sp.push(blk).size
        assert emitted <= fed - sp.d, (
            f"{fed} 표본을 받았는데 {emitted} 까지 내보냈다 (d={sp.d})")


def test_bad_parameters_are_refused():
    with pytest.raises(ValueError):
        StreamProcessor(build("M00"), hop=0)
    with pytest.raises(ValueError):
        StreamProcessor(build("M00"), win=64, d=64)
    with pytest.raises(ValueError):
        StreamProcessor(build("M00"), frontend="filtfilt")


# ------------------------------------------------- front-end 를 두 번 걸지 않기
def test_refuses_a_method_that_still_has_its_own_frontend():
    """인과 FE 를 앞단에 두면서 방법의 FE 를 켜 두면 **두 번 걸린다.**

    조용히 통과시키면 시연이 보고서와 다른 신호를 보이게 되고, 파형만 봐서는
    알 수 없다. 그래서 만들 때 막는다.
    """
    m = build("M04")                     # 기본값이 use_frontend=True
    assert m.fe is not None, "M04 가 자체 front-end 를 들고 있어야 한다"
    with pytest.raises(ValueError, match="두 번"):
        StreamProcessor(m, frontend="causal")
    StreamProcessor(build("M04", use_frontend=False), frontend="causal")


def test_causal_frontend_removes_the_baseline_without_seeing_the_future():
    """인과 FE 는 블록 단위로 들어와도 같은 결과를 낸다 (상태를 유지한다)."""
    y = _sig() + 3.0                      # 큰 DC
    a = StreamingFrontEnd(FS)
    b = StreamingFrontEnd(FS)
    out_a = np.concatenate([a.push(y[i:i + 25]) for i in range(0, y.size, 25)])
    out_b = np.concatenate([b.push(y[i:i + 137]) for i in range(0, y.size, 137)])
    assert np.max(np.abs(out_a - out_b)) < 1e-9, "블록 크기가 결과를 바꿨다"
    assert abs(out_a[2000:].mean()) < 0.05, "기저선이 안 빠졌다"


def test_notch_decision_is_frozen_after_warmup():
    """잡음이 들락날락할 때 필터가 켜졌다 꺼졌다 하면 그 자체가 인공물이다."""
    fe = StreamingFrontEnd(FS, decide_after_s=4.0)      # 1000 표본
    y = _sig(3000)
    fe.push(y[:600])
    assert not fe._decided, "판정 표본이 덜 찼는데 정했다"
    fe.push(y[600:1200])
    assert fe._decided
    n = len(fe._stages)
    fe.push(y[1200:])
    assert len(fe._stages) == n, "warm-up 이후에 필터가 추가됐다"


# ------------------------------------------------------------------ 산출물
def test_verification_numbers_are_recorded():
    """`plumbing_db` 는 0 근처여야 한다 — 배관이 성능을 깎으면 안 된다."""
    import json
    p = ROOT / "results" / "stream_verify.json"
    if not p.exists():
        pytest.skip("stream_verify.json 없음 — verify_stream_processor.py 를 돌린다")
    rows = json.loads(p.read_text())["rows"]
    assert rows
    bad = [r for r in rows if abs(r["plumbing_db"]) > 1.5]
    assert not bad, f"배관 손해가 큰 설정: {[(r['method'], r['plumbing_db']) for r in bad]}"
