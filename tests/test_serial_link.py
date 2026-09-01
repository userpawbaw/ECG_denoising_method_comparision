"""아두이노 선 규격 파서 (`ecgdn/realtime/serial_link.py`) — R-5.

여기서 고정하는 것은 **잃은 것을 잃었다고 말하는가**다. 시리얼에서 샘플이
빠지는 것 자체는 막을 수 없다(보드의 송신 버퍼가 64 바이트다). 막아야 하는
것은 **빠진 것을 조용히 이어 붙여 시간축이 밀리는 일**이다 — 그러면 R-peak
간격이 틀리고, 그것은 화면에서 잡음처럼 보인다.
"""
from __future__ import annotations

import numpy as np
import pytest

from ecgdn.realtime.serial_link import (SYNC, AsciiParser, BinaryParser,
                                        CausalDecimator, adc_to_mv)


def frame(seq: int, val: int) -> bytes:
    lo, hi = val & 0xFF, (val >> 8) & 0xFF
    return bytes([SYNC, seq & 0xFF, lo, hi, SYNC ^ (seq & 0xFF) ^ lo ^ hi])


def stream(vals, start=0) -> bytes:
    return b"".join(frame(start + i, v) for i, v in enumerate(vals))


# ------------------------------------------------------------------ 기본
def test_frames_round_trip_exactly():
    vals = [0, 1, 511, 512, 1023, 7]
    ch = BinaryParser().feed(stream(vals))
    assert list(ch.x) == [float(v) for v in vals]
    assert ch.ok.all() and ch.n_lost == 0 and ch.n_bad == 0


def test_a_frame_split_across_reads_is_not_lost():
    """USB CDC 는 **패킷 단위**로 온다 — 프레임 한가운데서 잘려 도착한다."""
    data = stream([100, 200, 300])
    p = BinaryParser()
    got = []
    for i in range(0, len(data), 3):            # 3 바이트씩: 프레임 경계와 어긋난다
        got.extend(p.feed(data[i:i + 3]).x.tolist())
    assert got == [100.0, 200.0, 300.0]


def test_sync_byte_inside_payload_does_not_break_framing():
    """`0xA5` 는 payload 에도 나온다 — 동기 바이트만 보면 프레임이 어긋난다."""
    ch = BinaryParser().feed(stream([0xA5, 0x1A5, 0xA5A & 0x3FF]))
    assert len(ch) == 3 and ch.n_bad == 0


# ------------------------------------------------------------------ 손실
def test_a_gap_is_filled_so_the_time_axis_survives():
    """빠진 자리를 **채워야** 한다. 이어 붙이면 이후 신호가 앞당겨진다."""
    p = BinaryParser()
    p.feed(frame(0, 500))
    ch = p.feed(frame(4, 600))                  # seq 1,2,3 이 없다
    assert ch.n_lost == 3
    assert len(ch) == 4, "빠진 3 개를 채우지 않으면 시간축이 밀린다"
    assert list(ch.ok) == [False, False, False, True]
    assert list(ch.x[:3]) == [500.0] * 3, "채움값은 직전 유효 샘플이다"


def test_gap_count_wraps_correctly_at_the_byte_boundary():
    p = BinaryParser()
    p.feed(frame(254, 500))
    ch = p.feed(frame(1, 600))                  # 255, 0 이 없다
    assert ch.n_lost == 2


def test_a_gap_too_large_to_count_is_reported_not_guessed():
    """seq 는 1 바이트다 — 255 를 넘으면 **몇 바퀴인지 알 수 없다.**"""
    p = BinaryParser()
    p.feed(frame(0, 500))
    ch = p.feed(frame(250, 600))
    assert ch.gap_unknown, "셀 수 없는 손실을 숫자로 지어내면 안 된다"


def test_lead_off_holds_the_last_value_instead_of_a_cliff():
    """`0xFFFF` 를 그대로 흘리면 필터에 거대한 계단이 들어간다."""
    p = BinaryParser()
    p.feed(frame(0, 500))
    ch = p.feed(frame(1, 0xFFFF) + frame(2, 510))
    assert ch.n_leadoff == 1
    assert ch.x[0] == 500.0 and not ch.ok[0]


# ------------------------------------------------------------------ 손상
def test_a_corrupted_frame_is_dropped_and_the_stream_resyncs():
    data = bytearray(stream([100, 200, 300]))
    data[7] ^= 0xFF                              # 두 번째 프레임을 깨뜨린다
    ch = BinaryParser().feed(bytes(data))
    assert ch.n_bad > 0
    assert 100.0 in ch.x and 300.0 in ch.x, "깨진 뒤 프레임을 되찾아야 한다"


def test_garbage_prefix_does_not_swallow_the_first_real_frame():
    """검사합 불일치에서 **프레임 길이만큼** 밀면 진짜 프레임을 건너뛴다.

    한 바이트씩 밀어야 한다. 아래 앞잡음은 검사합을 통과하지 못하도록
    골랐다 — `0xA5 01 02 03` 처럼 XOR 이 0 이 되는 조합은 **우연히 유효한
    프레임처럼 보인다**(1/256). 그 경우의 두 번째 방어선은 seq 검사다.
    """
    ch = BinaryParser().feed(bytes([SYNC, 1, 2, 4]) + stream([777]))
    assert 777.0 in ch.x and ch.n_bad > 0


# --------------------------------------------------------------- 데시메이션
def test_decimation_keeps_the_grid_across_block_boundaries():
    """블록마다 0 부터 다시 뽑으면 **샘플 간격이 들쭉날쭉해진다.**"""
    d = CausalDecimator(500, 250)
    p = BinaryParser()
    n = 0
    for i in range(0, 300, 7):                  # 7 개씩: 2 의 배수가 아니다
        n += len(d(p.feed(stream(list(range(i, min(i + 7, 300))), start=i))))
    assert n == 150, f"500 Hz 300 샘플이면 250 Hz 150 샘플이어야 한다 (얻은 값 {n})"


def test_decimation_marks_a_window_bad_if_any_source_sample_was_bad():
    """필터가 나쁜 값을 이웃으로 퍼뜨린다 — 뽑은 샘플만 보면 오염을 놓친다."""
    d = CausalDecimator(500, 250)
    p = BinaryParser()
    p.feed(frame(0, 500))
    ch = d(p.feed(frame(2, 500) + frame(3, 500)))   # seq 1 손실 -> ok[0]=False
    assert not ch.ok[0]


def test_non_integer_ratio_is_refused_rather_than_approximated():
    with pytest.raises(ValueError):
        CausalDecimator(360, 250)


def test_identity_decimator_passes_the_chunk_through():
    d = CausalDecimator(250, 250)
    ch = BinaryParser().feed(stream([1, 2, 3]))
    assert list(d(ch).x) == [1.0, 2.0, 3.0]


# ------------------------------------------------------------------ ASCII
def test_ascii_parser_reads_the_ide_format_and_marks_lead_off():
    ch = AsciiParser().feed(b"# logger ready\n0,512\n2,514\n4,-1\n")
    assert list(ch.x) == [512.0, 514.0, 514.0]
    assert list(ch.ok) == [True, True, False] and ch.n_leadoff == 1


def test_ascii_parser_holds_an_incomplete_line_until_it_finishes():
    p = AsciiParser()
    assert len(p.feed(b"0,51")) == 0
    assert list(p.feed(b"2\n").x) == [512.0]


# ------------------------------------------------------------------ 단위
def test_adc_to_mv_divides_by_the_analog_gain():
    """이득을 안 나누면 단위가 **회로 출력**이지 전극 단이 아니다."""
    mv = adc_to_mv(np.array([1023.0]), bits=10, vref=5.0, gain=1000.0)
    assert np.isclose(mv[0], 5.0)               # 5 V / 1000 = 5 mV
