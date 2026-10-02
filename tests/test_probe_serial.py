"""`scripts/probe_serial.py` — 보드가 말하는 것을 **다섯 가지로 갈라 읽는가**.

이 도구는 «브리지가 바이트는 오는데 샘플이 하나도 안 맞는다» 고 했을 때 쓴다. 그때
도구가 보드를 **틀리게** 말하면 — 예컨대 브리지가 못 읽는 줄을 «정상» 이라고 하면 —
사용자는 없는 문제를 쫓거나 있는 문제를 놓친다. 실보드 첫 투입에서 실제로 필요했던
도구라(브리지는 «안 맞는다» 까지만 말하고 «무엇이 왔는지» 는 말하지 않았다) 판정을
**브리지 파서와 같은 규칙** 위에 고정한다.

고정하는 것:
1. 프로브가 «받는다» 고 한 줄을 `AsciiParser` 가 실제로 받는다 (그 반대도) — 규칙이 둘이라 갈라지기 쉽다
2. 동기 바이트·프레임 길이가 `BinaryParser` 와 같다
3. 정상 · BINARY · 형식 다름 · 깨진 바이트 · 무응답을 서로 구분한다
4. 중간에 보드가 리셋돼 `t_ms` 가 되돌아가도 샘플률이 안 틀린다
5. pyserial 없이 읽힌다 (옵션 문서 생성기가 이 파일을 import 한다)
6. 가상 포트로 `main()` 을 끝까지 태운다
"""
from __future__ import annotations

import importlib.util
import os
import random
import sys
import threading
import time
from pathlib import Path

import pytest

from ecgdn.realtime.serial_link import FRAME_LEN, SYNC, AsciiParser, BinaryParser

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "probe_serial.py"


def _load():
    spec = importlib.util.spec_from_file_location("probe_serial", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


probe = _load()


# ---------------------------------------------------------------- 보드가 보낼 법한 바이트
def ascii_stream(n: int = 600, dt: int = 2, t0: int = 0) -> bytes:
    """스케치의 ASCII 출력: `t_ms,adc\\r\\n` (`Serial.println` 은 CRLF 다)."""
    rnd = random.Random(0)
    return b"".join(b"%d,%d\r\n" % (t0 + i * dt, rnd.randrange(300, 700)) for i in range(n))


BANNER = (b"# logger ready\r\n"
          b"# ecgstream v1 fs=500 mode=ascii bits=10 vref=5.00 dropped=0\r\n")


def binary_stream(n: int = 400) -> bytes:
    """스케치의 BINARY 프레임: [0xA5][seq][lo][hi][xor]."""
    out = bytearray()
    for i in range(n):
        val = 300 + (i * 7) % 400
        lo, hi, seq = val & 0xFF, val >> 8, i & 0xFF
        out += bytes([SYNC, seq, lo, hi, SYNC ^ seq ^ lo ^ hi])
    return bytes(out)


def verdict_of(buf: bytes) -> str:
    return probe.verdict(probe.analyse(buf))


# ------------------------------------------------------------ 1 · 2. 브리지 파서와 같은 규칙
@pytest.mark.parametrize("line", [
    b"4,793", b"10,-1", b"12,512", b"0,0", b"12, 512", b"12,512 ",   # 받는 것
    b"1,2,3", b"abc,1", b"1,2.5", b"512", b"t_ms,adc", b"1,", b",1",  # 못 받는 것
])
def test_probe_accepts_exactly_what_the_bridge_parser_accepts(line):
    """한 줄을 프로브가 «받는다» 고 하면 `AsciiParser` 도 샘플 하나로 받아야 한다."""
    n_probe = probe.analyse(line + b"\r\n")["ok"]
    n_bridge = AsciiParser().feed(line + b"\r\n").x.size
    assert n_probe == n_bridge, (
        f"{line!r}: 프로브는 {n_probe} 줄, 브리지는 {n_bridge} 샘플 — 규칙이 갈라졌다")


def test_comment_lines_are_not_counted_as_data():
    """배너(`#` 로 시작)는 데이터 줄이 아니다 — 브리지도 건너뛴다."""
    a = probe.analyse(BANNER + ascii_stream(30))
    assert len(a["data"]) == 30 and a["ok"] == 30 and a["banner"]
    assert AsciiParser().feed(BANNER + ascii_stream(30)).x.size == 30


def test_frame_constants_match_the_binary_parser():
    assert probe.SYNC == SYNC and probe.FRAME_LEN == FRAME_LEN


def test_probe_counts_the_frames_the_binary_parser_accepts():
    buf = binary_stream(400)
    assert probe.analyse(buf)["frames"] == BinaryParser().feed(buf).x.size == 400


# ---------------------------------------------------------------- 3. 다섯 가지를 가른다
def test_good_ascii_stream():
    a = probe.analyse(BANNER + ascii_stream())
    assert probe.verdict(a) == "good" and a["banner"]
    assert probe.rate_hz(a["data"]) == pytest.approx(500, rel=0.01)


def test_good_even_when_both_ends_are_partial_lines():
    """OS 버퍼에서 읽은 덩어리는 줄 중간에서 시작하고 끝난다."""
    buf = ascii_stream(600)
    assert verdict_of(buf[7:-5]) == "good"


def test_lead_off_lines_still_count_as_good():
    """전극이 떨어져 `-1` 이 흘러도 보드와 포트는 정상이다."""
    buf = b"".join(b"%d,-1\r\n" % (i * 2) for i in range(300))
    assert verdict_of(buf) == "good"


def test_a_few_lines_are_not_enough_to_call_it_good():
    assert verdict_of(ascii_stream(MIN := probe.MIN_LINES - 1)) != "good"
    assert MIN == probe.MIN_LINES - 1


def test_binary_frames_are_recognised_as_binary():
    assert verdict_of(binary_stream()) == "binary"


def _other_format(name: str) -> bytes:
    """사용자가 실제로 겪은 모양 - 줄바꿈은 규칙적인데 브리지가 못 받는 형식."""
    if name == "one_column":
        return b"".join(b"%d\r\n" % (400 + i % 300) for i in range(300))
    if name == "three_columns":
        return b"".join(b"%d,%d,0\r\n" % (i * 2, 400 + i % 300) for i in range(300))
    return b"".join(b"%d,%.2f\r\n" % (i * 2, 2.0 + (i % 9) / 10) for i in range(300))


@pytest.mark.parametrize("name,hint", [
    ("one_column", "1 칸"),
    ("three_columns", "3 칸"),
    ("float_column", "정수가 아니다"),
])
def test_text_in_another_format_is_not_called_good(name, hint):
    buf = _other_format(name)
    a = probe.analyse(buf)
    assert probe.verdict(a) == "text", name
    assert a["ok"] == 0
    assert hint in probe.text_hint(a)
    assert AsciiParser().feed(buf).x.size == 0          # 브리지도 정말 못 받는다


def test_random_bytes_are_garbage():
    rnd = random.Random(1)
    assert verdict_of(bytes(rnd.randrange(256) for _ in range(20000))) == "garbage"


def test_no_bytes_is_empty():
    assert verdict_of(b"") == "empty"


# ---------------------------------------------------------------- 4. 샘플률
def test_rate_survives_a_board_reset_in_the_middle():
    """`t_ms` 가 중간에 되돌아가도(보드 리셋) 처음·끝 차이로 재지 않는다."""
    buf = ascii_stream(300, dt=2) + ascii_stream(300, dt=2)
    a = probe.analyse(buf)
    assert probe.rate_hz(a["data"]) == pytest.approx(500, rel=0.01)


@pytest.mark.parametrize("dt,hz", [(1, 1000), (2, 500), (4, 250)])
def test_rate_for_the_three_board_rates(dt, hz):
    a = probe.analyse(ascii_stream(300, dt=dt))
    assert probe.rate_hz(a["data"]) == pytest.approx(hz, rel=0.01)


def test_rate_is_unknown_without_enough_lines():
    assert probe.rate_hz(probe.analyse(ascii_stream(5))["data"]) is None


# ---------------------------------------------------------------- 5. pyserial 없이
def test_importing_the_probe_does_not_need_pyserial(monkeypatch):
    """옵션 문서 생성기(`make_cli_reference.py`)가 이 파일을 import 한다.

    pyserial 이 없는 환경(CI)에서 import 가 터지면 문서에서 이 스크립트가 조용히 빠진다.
    """
    monkeypatch.setitem(sys.modules, "serial", None)         # `import serial` 이 ImportError
    _load()


# ---------------------------------------------------------------- 6. 가상 포트로 끝까지
pytest.importorskip("serial", reason="pyserial 이 있어야 main() 을 태운다")
posix_only = pytest.mark.skipif(not hasattr(os, "openpty"),
                                reason="가상 포트를 못 만든다 (윈도우)")


class Feeder:
    """PTY 한쪽에 `payload` 를 반복해서 쓰는 가짜 보드. `port` 가 프로브에 줄 이름이다."""

    def __init__(self, payload: bytes, per_s: int = 5000):
        import tty
        self.master, self.slave = os.openpty()
        tty.setraw(self.master)
        tty.setraw(self.slave)
        self.port = os.ttyname(self.slave)
        self._payload, self._rate = payload, per_s
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        t0, sent = time.time(), 0
        while not self._stop.is_set():
            want = int((time.time() - t0) * self._rate)
            while sent < want and not self._stop.is_set():
                i = sent % len(self._payload)
                try:
                    os.write(self.master, self._payload[i:i + 16])
                except OSError:
                    return
                sent += 16
            time.sleep(0.003)

    def close(self):
        self._stop.set()
        self._t.join(timeout=2)
        for fd in (self.master, self.slave):
            try:
                os.close(fd)
            except OSError:
                pass


def _run_main(monkeypatch, capsys, *argv) -> tuple[int, str]:
    monkeypatch.setattr(sys, "argv", ["probe_serial.py", *argv])
    code = probe.main()
    return code, capsys.readouterr().out


@posix_only
def test_main_reports_a_healthy_board_with_exit_code_0(monkeypatch, capsys):
    fb = Feeder(BANNER + ascii_stream(2000))
    try:
        code, out = _run_main(monkeypatch, capsys, "--port", fb.port, "--baud", "115200",
                              "--seconds", "1.5")
    finally:
        fb.close()
    assert code == 0, out
    assert "정상이다" in out and "약 500 Hz" in out


@posix_only
def test_main_flags_a_board_speaking_another_format(monkeypatch, capsys):
    other = b"".join(b"%d,%d,0\r\n" % (i * 2, 400 + i % 300) for i in range(2000))
    fb = Feeder(other)
    try:
        code, out = _run_main(monkeypatch, capsys, "--port", fb.port, "--baud", "115200",
                              "--seconds", "1.5")
    finally:
        fb.close()
    assert code == 1, out
    assert "3 칸" in out and "다시 업로드" in out


@posix_only
def test_main_stops_at_the_first_baud_that_reads(monkeypatch, capsys):
    """흔한 baud 를 다 돌리지 않는다 — 읽히는 곳에서 멈춘다 (9 개면 40 초가 든다)."""
    fb = Feeder(BANNER + ascii_stream(2000))
    try:
        code, out = _run_main(monkeypatch, capsys, "--port", fb.port, "--seconds", "1.5")
    finally:
        fb.close()
    assert code == 0
    assert sum(1 for ln in out.splitlines() if ln.strip().startswith(("115200", "9600"))) == 1


def test_main_without_a_port_lists_ports_and_exits_2(monkeypatch, capsys):
    code, out = _run_main(monkeypatch, capsys)
    assert code == 2 and "열 수 있는 포트" in out and "--port 가 필요하다" in out


def test_main_list_exits_0(monkeypatch, capsys):
    code, out = _run_main(monkeypatch, capsys, "--list")
    assert code == 0 and "열 수 있는 포트" in out


def test_main_tells_which_ports_exist_when_the_port_cannot_be_opened(monkeypatch, capsys):
    code, out = _run_main(monkeypatch, capsys, "--port", "/dev/no_such_port_xyz",
                          "--baud", "115200", "--seconds", "3")
    assert code == 1
    assert "포트를 못 연다" in out and "열 수 있는 포트" in out


# ------------------------------------------------------------------ --read-timing (F-55)
def test_read_timing_says_blocks_when_every_read_fills_the_request():
    """윈도우 실보드에서 녹화로 잰 모양 — 4096 B 를 1.49 s 마다 (F-55)."""
    p = _load()
    s = p.timing_summary([(1.49, 4096)] * 8 + [(0.05, 0)])
    assert s["n_med"] == 4096 and s["calls"] == 8
    assert p.timing_verdict(s) == "blocks"


def test_read_timing_says_ok_when_reads_return_on_the_timeout():
    """리눅스 pyserial + 가상 보드에서 잰 모양 — 50 ms · 132 B."""
    p = _load()
    s = p.timing_summary([(0.05, 132)] * 50)
    assert p.timing_verdict(s) == "ok"


def test_read_timing_ignores_empty_reads_and_reports_none_without_bytes():
    p = _load()
    assert p.timing_verdict(p.timing_summary([(0.05, 0)] * 10)) == "none"
