"""실제 시리얼 경로 (`SerialSource`) — R-7. **가상 포트로 끝까지 태운다.**

`--replay` 는 브리지 **안에서** 바이트를 만들므로, 시연 당일 실제로 타는 길
— pyserial 이 포트를 열고, 보드에 명령을 보내고, OS 버퍼에서 읽는 경로 —
가 **한 줄도 실행되지 않는다.** 여기서는 `os.openpty()` 로 커널이 만든 진짜
tty 한 쌍을 열고 한쪽에 `FakeBoard` 를 붙여 그 길을 통째로 태운다.

고정하는 것 넷 — 넷 다 시연 당일에 실제로 일어나는 일이다:

1. **명령이 보드에 닿는가** (`--board-fs 250` 이 정말 250 Hz 를 만드는가)
2. **포트가 배타 자원인가** (IDE 나 두 번째 브리지가 붙으면 막히는가 — O-30)
3. **선이 끊기면 알아채는가** (조용히 멈추지 않는가 — O-30)
4. **부팅 잡음이 신호에 안 섞이는가**
"""
from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from ecgdn.realtime.fake_board import FakeBoard
from ecgdn.realtime.serial_link import BinaryParser

ROOT = Path(__file__).resolve().parent.parent

pytest.importorskip("serial", reason="pyserial 이 있어야 실경로를 태울 수 있다")
if not hasattr(os, "openpty"):                              # pragma: no cover
    pytest.skip("가상 포트를 못 만든다 (윈도우는 com0com 을 쓴다)",
                allow_module_level=True)


def _load(name: str):
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as e:                                   # pragma: no cover
        pytest.skip(f"{name} 을 못 읽었다: {e}")
    return m


class VirtualBoard:
    """가상 아두이노를 스레드로 돌린다. `port` 가 브리지에 줄 이름이다."""

    def __init__(self, **kw):
        self.fa = _load("fake_arduino")
        self.master, self.slave, self.port = self.fa.open_virtual_port()
        kw.setdefault("fs", 500)
        self.board = FakeBoard(np.full(4000, 700), **kw)
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                cmd = os.read(self.master, 256)
            except (BlockingIOError, OSError):
                cmd = b""
            if cmd:
                self.board.feed_command(cmd)
            data = self.board.poll(time.perf_counter())
            if data:
                try:
                    os.write(self.master, data)
                except (BlockingIOError, OSError):
                    pass
            time.sleep(0.002)

    def unplug(self) -> None:
        """케이블을 뽑는다."""
        self._stop.set()
        self._th.join(timeout=1.0)
        for fd in (self.master, self.slave):
            try:
                os.close(fd)
            except OSError:
                pass

    def close(self) -> None:
        if not self._stop.is_set():
            self.unplug()


@pytest.fixture
def board():
    b = VirtualBoard(mode="ascii")
    yield b
    b.close()


# ----------------------------------------------- 명령이 보드에 닿는가
def test_the_bridge_actually_switches_the_board_to_250_hz(board):
    """`--board-fs 250` 은 **보드에게 보내는 명령**이다. 닿는지 여기서 본다."""
    m = _load("serial_bridge")
    src = m.SerialSource(board.port, 115200, 250, True, settle_s=0.3)
    try:
        assert board.board.fs == 250, "fs 명령이 보드에 안 닿았다"
        assert board.board.mode == "bin", "형식 명령이 보드에 안 닿았다"
        p = BinaryParser()
        got = 0
        t0 = time.perf_counter()
        while got < 200 and time.perf_counter() - t0 < 5.0:
            got += len(p.feed(src.read()))
        assert got >= 200, "실경로로 샘플이 안 들어온다"
    finally:
        src.close()


def test_the_boot_banner_never_reaches_the_signal(board):
    """포트를 열면 보드가 리셋되고 `# logger ready` 를 흘린다 — 그것을 버려야 한다."""
    m = _load("serial_bridge")
    src = m.SerialSource(board.port, 115200, 250, True)
    try:
        raw = b""
        t0 = time.perf_counter()
        while len(raw) < 400 and time.perf_counter() - t0 < 5.0:
            raw += src.read()
        assert b"# logger ready" not in raw, "부팅 배너가 신호에 섞였다"
        assert b"# ecgstream" not in raw, "헤더가 신호에 섞였다"
        ch = BinaryParser().feed(raw)
        assert len(ch) > 50
        # 재동기에 쓰인 몇 바이트는 있을 수 있지만, 프레임의 대부분은 맞아야 한다
        assert ch.n_bad < 10, f"프레임이 계속 깨진다 (bad={ch.n_bad})"
    finally:
        src.close()


# --------------------------------------------------- 포트는 배타 자원이다
def test_a_second_bridge_cannot_take_the_port(board):
    """**이것이 없으면 두 브리지가 에러 없이 바이트를 나눠 가진다** (O-30).

    양쪽 화면이 손실 50 % 로 보이고, 그 증상은 전극 문제와 구별되지 않는다.
    """
    m = _load("serial_bridge")
    first = m.SerialSource(board.port, 115200, 250, True, settle_s=0.3)
    try:
        with pytest.raises(SystemExit) as e:
            m.SerialSource(board.port, 115200, 250, True, settle_s=0.3)
        assert "포트를 열 수 없다" in str(e.value)
        assert "시리얼 모니터" in str(e.value), "무엇을 닫아야 하는지 안 적혀 있다"
    finally:
        first.close()


def test_without_the_exclusive_flag_two_readers_split_the_bytes(board):
    """왜 배타 잠금이 필요한지 — **잠그지 않으면 조용히 나눠 갖는다.**"""
    import serial

    a = serial.Serial(board.port, 115200, timeout=0.05)
    b = serial.Serial(board.port, 115200, timeout=0.05)
    try:
        t0 = time.perf_counter()
        na = nb = 0
        while time.perf_counter() - t0 < 0.6:
            na += len(a.read(200))
            nb += len(b.read(200))
        assert na > 0 and nb > 0, "둘 다 읽지는 못했다 — 전제가 바뀌었다"
    finally:
        a.close()
        b.close()


# ------------------------------------------------------- 선이 끊기면
def test_pulling_the_cable_raises_instead_of_going_quiet(board):
    """읽기가 조용히 죽으면 화면은 «연결됨» 인 채 파형만 멈춘다 (O-30)."""
    m = _load("serial_bridge")
    src = m.SerialSource(board.port, 115200, 250, True, settle_s=0.3)
    try:
        assert src.read() is not None
        board.unplug()
        with pytest.raises(m.LinkLost):
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 3.0:
                src.read()
            pytest.fail("케이블을 뽑았는데 3 초 동안 아무 일도 안 났다")
    finally:
        try:
            src.close()
        except Exception:
            pass


def test_link_loss_is_a_distinct_error_type():
    """`LinkLost` 여야 읽기 스레드가 그것만 잡고 나머지는 그대로 터진다."""
    m = _load("serial_bridge")
    assert issubclass(m.LinkLost, RuntimeError)


# ------------------------------------------- 구 스케치가 꽂혀 있으면
def test_an_old_sketch_keeps_its_own_fs_and_the_bridge_can_see_it():
    """명령을 모르는 판은 500 Hz 를 계속 준다 — **에러 없이 시간축만 틀린다**(F-42).

    브리지가 이것을 잡는 근거는 «실측 샘플률» 하나뿐이므로, 그 값이 정말
    보드를 따라가는지 여기서 고정한다.
    """
    b = VirtualBoard(mode="bin", accept_commands=False, fs=500)
    m = _load("serial_bridge")
    try:
        src = m.SerialSource(b.port, 115200, 250, True, settle_s=0.3)
        try:
            assert b.board.fs == 500, "명령을 모르는 판인데 fs 가 바뀌었다"
            p = BinaryParser()
            n = 0
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 3.0:
                n += len(p.feed(src.read()))
            meas = n / (time.perf_counter() - t0)
            assert meas > 250 * 1.5, (
                f"실측 {meas:.0f} Hz — 250 Hz 라고 믿으면 시간축이 배로 틀린다")
        finally:
            src.close()
    finally:
        b.close()
