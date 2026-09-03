"""시리얼 브리지 (`scripts/serial_bridge.py`) — R-5/R-6.

하드웨어가 없으므로 **모의 보드**(`ReplaySource`)가 진짜 보드와 같은 선
규격으로 말하는지, 그리고 화면에 보내기 전 **방법끼리 시각이 맞는지**를
고정한다. 둘 다 틀려도 화면은 그럴듯하게 나오므로 눈으로는 못 잡는다.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from ecgdn.realtime.serial_link import AsciiParser, BinaryParser

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "serial_bridge", ROOT / "scripts" / "serial_bridge.py")
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as e:                       # pragma: no cover
        pytest.skip(f"serial_bridge 를 못 읽었다: {e}")
    return m


# ------------------------------------------------------- 모의 보드가 진짜처럼
def test_replay_source_speaks_the_real_wire_format():
    """모의 보드가 실제 파서로 읽혀야 대체 경로로서 의미가 있다."""
    m = _mod()
    src = m.ReplaySource("synth", 250, True, drift_ppm=0.0)
    p = BinaryParser()
    got = 0
    t0 = time.perf_counter()
    while got < 100 and time.perf_counter() - t0 < 5.0:
        ch = p.feed(src.read())
        got += len(ch)
        assert ch.n_bad == 0, "모의 보드가 깨진 프레임을 냈다"
        assert ch.n_lost == 0, "드롭을 안 켰는데 손실이 생겼다"
    assert got >= 100


def test_replay_source_ascii_mode_matches_the_ide_format():
    m = _mod()
    src = m.ReplaySource("synth", 250, False, drift_ppm=0.0)
    p = AsciiParser()
    t0 = time.perf_counter()
    got = 0
    while got < 20 and time.perf_counter() - t0 < 5.0:
        got += len(p.feed(src.read()))
    assert got >= 20


def test_replay_source_drops_show_up_as_sequence_gaps():
    """모의 드롭이 **파서에서 손실로 보여야** 한다 — 그래야 대응을 시험할 수 있다."""
    m = _mod()
    src = m.ReplaySource("synth", 250, True, drift_ppm=0.0, drop_rate=0.2, seed=3)
    p = BinaryParser()
    lost = n = 0
    t0 = time.perf_counter()
    while n < 300 and time.perf_counter() - t0 < 5.0:
        ch = p.feed(src.read())
        n += len(ch)
        lost += ch.n_lost
    assert lost > 0, "드롭을 켰는데 손실이 하나도 안 잡혔다"


# ------------------------------------------------------------------ 정렬
def test_aligner_emits_only_the_range_every_method_has():
    """한 방법이 아직 못 낸 구간을 내보내면 화면에서 시각이 어긋난다."""
    m = _mod()
    al = m.Aligner(["A", "B"])
    al.add("A", 0, np.arange(10.0))
    idx, out = al.take()
    assert out == {}, "B 가 아직 아무것도 안 냈는데 내보냈다"
    al.add("B", 4, np.arange(3.0))
    idx, out = al.take()
    assert idx == 4 and len(out["A"]) == len(out["B"]) == 3
    assert out["A"] == [4.0, 5.0, 6.0]


def test_aligner_never_repeats_or_skips_a_sample():
    m = _mod()
    al = m.Aligner(["A", "B"])
    seen: list[float] = []
    for k in range(5):
        al.add("A", 0, np.arange(k * 7.0, k * 7.0 + 7))
        al.add("B", 0, np.arange(k * 7.0, k * 7.0 + 7))
        idx, out = al.take()
        if out:
            assert idx == len(seen), f"연속이 끊겼다: {idx} vs {len(seen)}"
            seen.extend(out["A"])
    assert seen == list(np.arange(float(len(seen))))


def test_aligner_takes_nothing_twice():
    m = _mod()
    al = m.Aligner(["A"])
    al.add("A", 0, np.arange(5.0))
    assert al.take()[1]["A"] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert al.take()[1] == {}


# --------------------------------------------------- 이중 필터를 막는 규칙
def test_front_end_methods_are_not_given_their_own_processor():
    """`M_FE`·`M01` 은 **필터가 곧 방법**이다 — 단 **front-end 가 대역통과일 때만.**

    실시간 경로는 앞단에 FE 를 이미 한 번 걸었으므로, 대역통과 FE 에서 이들에
    처리기를 또 붙이면 같은 필터가 두 번 걸린다 — R-4 에서 −15 dB 를 만든 그
    실수다 (F-25). `verify_stream_processor.py` 가 이들을 건너뛰는 것과 같은 이유다.

    **중앙값 front-end 는 대역통과가 아니다.** 거기서 `M01` 을 FE 출력으로
    대체하면 «대역통과+노치» 라고 이름 붙인 자리에 **전혀 다른 필터의 출력**이
    나간다. 그래서 대체는 모드에 따라 갈려야 한다.
    """
    from ecgdn.realtime.frontend_modes import FE_MODES, fe_intrinsic
    m = _mod()
    assert m.ALWAYS_INTRINSIC == {"M_FE"}
    for mode, spec in FE_MODES.items():
        got = fe_intrinsic(mode)
        assert "M_FE" in got, f"{mode}: M_FE 는 언제나 FE 출력 그 자체다"
        if spec["bandpass"]:
            assert got == {"M_FE", "M01", "M01d"}, f"{mode}: 이중 필터가 된다 (F-25)"
        else:
            assert "M01" not in got, \
                f"{mode}: 대역통과가 아닌 FE 출력을 M01 이라고 내보내고 있다"
    src = ROOT / "scripts" / "serial_bridge.py"
    assert "split_names(" in src.read_text(), \
        "모드별로 FE 내재 방법을 가르는 분기가 사라졌다"


def test_the_live_page_never_shows_a_performance_number():
    """모드 A 에는 참값이 없다 — SNR 을 띄우면 그것은 지어낸 숫자다(난관 6)."""
    html = (ROOT / "demo" / "live.html").read_text()
    for bad in ("SNR 개선", "dB</", "snr_imp", "축 평균"):
        assert bad not in html, f"실측 화면에 성능 수치가 들어갔다: {bad}"
