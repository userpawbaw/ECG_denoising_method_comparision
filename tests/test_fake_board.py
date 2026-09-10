"""가상 보드 (`ecgdn/realtime/fake_board.py`) — R-7.

여기서 고정하는 것은 **가상 보드가 진짜 보드와 같은 것을 말하는가**다.
시뮬레이터가 실제와 어긋나면 «가상으로 다 해 봤다» 가 거짓이 되고, 그 거짓은
시연 당일에만 드러난다. 그래서 `.ino` 를 **읽어서 대조한다** — 펌웨어를 고치고
이 파일을 안 고치면 여기서 걸린다.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from ecgdn.realtime.fake_board import (BOOT_BANNER, BOOTLOADER_S,
                                       SERIAL_TX_BUFFER_SIZE, FakeBoard,
                                       UsbPipe)
from ecgdn.realtime.serial_link import (LEADOFF, SYNC, AsciiParser,
                                        BinaryParser)

ROOT = Path(__file__).resolve().parent.parent
INO = ROOT / "hardware" / "arduino_ecg_logger" / "arduino_ecg_logger.ino"


def run(board: FakeBoard, to_s: float, step: float = 0.004,
        start: float = 0.0) -> bytes:
    """가상 시계로 `start` 부터 `to_s` 까지 돌린다.

    실시간으로 기다리면 1 kHz 를 3 분 재는 데 3 분이 걸리고, 그러면 아무도
    안 돌린다. 보드의 시계는 `poll(now)` 가 주는 값이라 **되감으면 안 된다** —
    이어서 돌릴 때는 `start` 를 앞 구간의 끝으로 준다.
    """
    out = bytearray()
    t = start
    while t < to_s:
        t += step
        out += board.poll(t)
    return bytes(out)


# ------------------------------------------------- 펌웨어와 어긋나지 않는가
def test_the_sketch_still_exists_where_we_think_it_does():
    assert INO.exists(), f"펌웨어를 못 찾겠다: {INO}"


def test_frame_layout_matches_the_sketch():
    """`0xA5` · 5 바이트 · XOR 검사합 — 셋 다 스케치에서 읽어 대조한다."""
    src = INO.read_text()
    assert "0xA5" in src, "스케치의 동기 바이트가 바뀌었다"
    assert SYNC == 0xA5
    assert "Serial.availableForWrite() < 5" in src, \
        "BINARY 의 송신 버퍼 임계값이 바뀌었다 — fake_board 의 _MIN_FREE 도 고칠 것"
    assert "Serial.availableForWrite() < 16" in src, \
        "ASCII 의 송신 버퍼 임계값이 바뀌었다 — fake_board 의 _MIN_FREE 도 고칠 것"


def test_default_fs_and_baud_match_the_sketch():
    src = INO.read_text()
    baud = int(re.search(r"SERIAL_BAUD\s*=\s*(\d+)", src).group(1))
    boot_fs = int(re.search(r"void setup\(\).*?set_fs\((\d+)\)", src,
                            re.S).group(1))
    b = FakeBoard(np.full(100, 512), fs=boot_fs, baud=baud)
    assert b.fs == boot_fs == 500, "스케치가 켜질 때의 fs 가 바뀌었다"
    assert b.baud == baud == 115200
    assert b.mode == "ascii", "스케치는 ASCII 로 켜진다 (IDE 플로터 호환)"


def test_every_command_the_sketch_knows_is_implemented():
    """스케치의 `case 'x':` 를 전부 긁어 가상 보드가 아는지 본다."""
    src = INO.read_text()
    cmds = set(re.findall(r"case '(.)':", src))
    assert cmds == set("ab251r?"), f"스케치의 명령 목록이 바뀌었다: {cmds}"
    b = FakeBoard(np.full(100, 512), fs=500)
    b.feed_command(b"2")
    assert b.fs == 250
    b.feed_command(b"5")
    assert b.fs == 500
    b.feed_command(b"1")
    assert b.fs == 1000
    b.feed_command(b"b")
    assert b.mode == "bin"
    b.feed_command(b"a")
    assert b.mode == "ascii"


def test_boot_banner_is_what_the_sketch_prints():
    src = INO.read_text()
    assert "# logger ready" in src
    b = FakeBoard(np.full(100, 512))
    first = b.poll(0.0)
    assert first.startswith(BOOT_BANNER)
    assert b"# ecgstream v1 fs=500 mode=ascii" in first


# ------------------------------------------------------------- 선 규격
def test_binary_frames_read_back_with_the_real_parser():
    b = FakeBoard(np.full(2000, 700), fs=250, mode="bin")
    b.poll(0.0)                                   # 부팅 배너를 버린다
    data = run(b, 2.0)
    ch = BinaryParser().feed(data)
    assert ch.n_bad == 0, "가상 보드가 깨진 프레임을 냈다"
    assert ch.n_lost == 0
    assert len(ch) > 400
    assert set(np.unique(ch.x)) == {700.0}


def test_ascii_lines_end_with_crlf_like_arduino_println():
    """`Serial.println` 은 **CR+LF** 를 붙인다 — `ReplaySource` 는 LF 만 쓴다.

    파서가 `strip()` 하므로 결과는 같지만, «같은 선 규격» 이라고 말하려면
    가상 보드 쪽이 실제와 같아야 한다.
    """
    b = FakeBoard(np.full(2000, 700), fs=250, mode="ascii")
    b.poll(0.0)
    data = run(b, 1.0)
    body = data.split(b"\r\n", 1)[1] if data.startswith(b"#") else data
    assert b"\r\n" in body
    ch = AsciiParser().feed(data)
    assert ch.n_bad == 0, "헤더 줄이 파싱 실패로 세어졌다"
    assert len(ch) > 200
    assert set(np.unique(ch.x)) == {700.0}


def test_lead_off_is_marked_not_measured():
    b = FakeBoard(np.full(2000, 700), fs=250, mode="bin",
                  leadoff_every_s=1.0, leadoff_len_s=0.5)
    b.poll(0.0)
    ch = BinaryParser().feed(run(b, 2.0))
    assert ch.n_leadoff > 100, "lead-off 구간이 표시되지 않았다"
    # 표시일 뿐 값이 아니다 — 직전 값을 유지하고 `ok=False` 로 남는다
    assert not ch.ok.all()
    assert float(np.max(ch.x)) <= 1023.0, f"lead-off 원값({LEADOFF})이 새어 나왔다"


def test_the_board_clock_is_not_the_pc_clock():
    """드리프트가 실제로 샘플 수를 바꾼다 — 이것이 없으면 «벽시계로 세기» 가 통과한다."""
    fast = FakeBoard(np.full(4000, 512), fs=250, mode="bin", drift_ppm=+5000.0)
    slow = FakeBoard(np.full(4000, 512), fs=250, mode="bin", drift_ppm=-5000.0)
    run(fast, 10.0)
    run(slow, 10.0)
    assert fast.k > slow.k, "클럭 오차가 샘플 수에 안 나타난다"
    assert abs(fast.k / 10.0 - 251.25) < 1.0
    assert abs(slow.k / 10.0 - 248.75) < 1.0


# --------------------------------------------- 드롭은 확률이 아니라 대역이 정한다
@pytest.mark.parametrize("mode,fs,baud,want_drop", [
    ("bin", 250, 115200, False),      # 시연 설정
    ("bin", 500, 115200, False),
    ("bin", 1000, 115200, False),     # 5 kB/s — 115200 으로 **충분하다**
    ("ascii", 500, 115200, False),    # 수집 설정
    ("ascii", 1000, 115200, True),    # 12 kB/s — 넘는다
    ("ascii", 1000, 250000, False),   # baud 를 올리면 들어온다
])
def test_transmit_buffer_overflow_follows_the_bandwidth_sum(mode, fs, baud,
                                                            want_drop):
    """`docs/30_realtime_demo.md` 6.2 (1) 의 표를 **재현**한다.

    펌웨어가 샘플을 버리는 조건은 확률이 아니라 `availableForWrite() < N` 이고,
    그것을 정하는 것은 baud · fs · 형식이다.
    """
    b = FakeBoard(np.full(4000, 512), fs=fs, baud=baud, mode=mode)
    run(b, 180.0, step=0.01)
    got = b.dropped > 0
    assert got is want_drop, (
        f"{mode} {fs} Hz @{baud}: 드롭 {b.dropped}/{b.k} — 예상과 다르다")


def test_ascii_at_1khz_survives_the_first_minute_then_fails():
    """**시작 100 초까지는 멀쩡하다.** 짧게 재면 통과하고 시연 중에 무너진다.

    `Serial.print(t_ms)` 가 보내는 바이트 수가 `t_ms` 의 **자릿수**라, 한 줄이
    100 초를 넘는 순간 11 B 에서 12 B 가 되고 그때 115200 을 넘어선다.
    """
    b = FakeBoard(np.full(4000, 512), fs=1000, baud=115200, mode="ascii")
    run(b, 90.0, step=0.01)
    early = b.dropped
    run(b, 180.0, step=0.01, start=90.0)
    assert early == 0, f"첫 90 초에 이미 {early} 개를 버렸다"
    assert b.dropped > 1000, "100 초 뒤에도 안 무너진다 — 대역 모형을 다시 볼 것"


def test_buffer_threshold_is_the_ring_buffer_size_minus_one():
    """AVR 의 `availableForWrite()` 는 링버퍼라 **63** 이 최대다.

    경계를 시간으로 재면 배출이 섞이므로 여기서는 버퍼 잔량을 직접 놓고 본다.
    """
    assert SERIAL_TX_BUFFER_SIZE == 64
    b = FakeBoard(np.full(100, 512), fs=250, mode="bin", baud=115200)
    b.poll(0.0)
    b._pending = SERIAL_TX_BUFFER_SIZE - 1 - 5      # 자리가 딱 5 B
    before = b.dropped
    b._one_sample()
    assert b.dropped == before, "자리가 5 B 인데 버렸다"
    b._pending = SERIAL_TX_BUFFER_SIZE - 1 - 4      # 5 B 를 못 넣는다
    b._one_sample()
    assert b.dropped == before + 1, "자리가 4 B 인데 안 버렸다"


# --------------------------------------------------- 구 스케치가 꽂힌 판
def test_an_old_sketch_ignores_commands_and_keeps_its_own_fs():
    b = FakeBoard(np.full(100, 512), fs=500, accept_commands=False)
    b.feed_command(b"2b")
    assert b.fs == 500, "명령을 모르는 판인데 fs 가 바뀌었다"
    assert b.mode == "ascii", "명령을 모르는 판인데 형식이 바뀌었다"


def test_changing_fs_mid_stream_resets_the_sequence_number():
    """`set_fs()` 는 `seq` 를 0 으로 돌린다 — 그래서 **명령은 시작할 때만** 보낸다.

    바이너리가 흐르는 도중에 fs 를 바꾸면 PC 는 seq 점프를 «셀 수 없는 손실»
    로 읽고 처리기를 리셋한다. 브리지가 fs 를 먼저, 형식을 나중에 보내는
    이유가 이것이다.
    """
    b = FakeBoard(np.full(2000, 512), fs=250, mode="bin")
    b.poll(0.0)
    p = BinaryParser()
    p.feed(run(b, 1.0))
    b.feed_command(b"5")
    ch = p.feed(run(b, 1.0))
    assert ch.gap_unknown or ch.n_lost > 0, \
        "seq 가 0 으로 돌아갔는데 PC 쪽에서 아무 일도 안 생겼다"

# ------------------------------------------------- DTR 리셋 (열면 처음부터)
def test_the_bootloader_is_silent_and_then_the_banner_comes():
    """리셋 뒤 부트로더가 도는 동안 보드는 **한 바이트도 안 보낸다.**

    브리지가 포트를 연 뒤 2 초를 기다리는 이유가 이것이다.
    """
    b = FakeBoard(np.full(2000, 700), fs=250, mode="ascii", boot_s=BOOTLOADER_S)
    quiet = run(b, BOOTLOADER_S - 0.1, step=0.01)
    assert quiet == b"", f"부트로더가 도는 중에 {len(quiet)} B 가 나갔다"
    after = run(b, BOOTLOADER_S + 0.5, step=0.01, start=BOOTLOADER_S - 0.1)
    assert after.startswith(BOOT_BANNER), "부트로더가 끝났는데 배너가 없다"
    assert b"# ecgstream" in after


def test_a_command_sent_during_the_bootloader_is_eaten():
    """**성급하게 보낸 명령은 사라진다** — 부트로더가 먹고 스케치는 못 본다.

    이것이 없으면 「왜 2 초를 기다리나」가 주석으로만 남고, 대기를 줄이는
    변경이 테스트를 통과해 버린다.
    """
    b = FakeBoard(np.full(2000, 700), fs=500, mode="ascii", boot_s=BOOTLOADER_S)
    run(b, 0.3, step=0.01)
    b.feed_command(b"2b")                       # 부트로더가 도는 중
    assert b.fs == 500 and b.mode == "ascii", "부트로더 중의 명령이 먹혔어야 한다"
    run(b, BOOTLOADER_S + 0.2, step=0.01, start=0.3)
    b.feed_command(b"2b")                       # 스케치가 도는 중
    assert b.fs == 250 and b.mode == "bin", "스케치가 명령을 못 받았다"


def test_reset_puts_the_board_back_to_its_power_on_state():
    b = FakeBoard(np.full(2000, 700), fs=500, mode="ascii", boot_s=0.0)
    run(b, 2.0)
    b.feed_command(b"2b")
    assert (b.fs, b.mode) == (250, "bin")
    b.reset(0.0)
    assert (b.fs, b.mode) == (500, "ascii"), "리셋했는데 명령이 살아 있다"
    assert b.seq == 0 and b.k == 0 and b.dropped == 0


# --------------------------------------------- USB 는 바이트를 뭉쳐서 배달한다
def test_usb_delivers_in_packets_not_byte_by_byte():
    pipe = UsbPipe(poll_ms=1.0, packet=64)
    pipe.push(b"x" * 500)
    assert pipe.pop(0.0) == b"x" * 64, "한 폴링에 한 패킷보다 많이 나갔다"
    assert len(pipe.pop(0.0005)) == 0, "폴링 간격 전에 또 나갔다"
    assert len(pipe.pop(0.0011)) == 64


def test_a_host_hiccup_holds_everything_then_lets_it_go_at_once():
    """호스트가 잠깐 안 가져가면 그동안 쌓인 것이 **한 덩어리로** 나간다."""
    pipe = UsbPipe(poll_ms=1.0, packet=64, hiccup_every_s=1.0, hiccup_ms=200.0)
    pipe.pop(0.5)                                # 정상 구간에서 시계를 맞춘다
    pipe.push(b"y" * 4000)
    assert pipe.pop(1.05) == b"", "멈춤 구간인데 내보냈다"
    burst = pipe.pop(1.30)                       # 멈춤이 끝난 직후
    assert len(burst) > 1000, f"뭉쳐 있던 것이 안 나왔다 ({len(burst)} B)"


def test_the_host_buffer_can_overflow_and_says_so():
    """호스트가 오래 안 가져가면 **아무도 세지 않는 자리**에서 바이트가 사라진다."""
    pipe = UsbPipe(poll_ms=1.0, packet=64, buffer_bytes=1000)
    pipe.push(b"z" * 1500)
    assert pipe.overflow == 500
    assert pipe.pending == 1000

