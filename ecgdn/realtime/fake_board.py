"""R-7: **펌웨어를 파이썬으로 옮긴 것.** 가상 보드가 진짜 보드처럼 말하게 한다.

    board = FakeBoard(counts, fs=500, baud=115200)
    board.feed_command(b"2b")               # PC 가 보낸 명령
    data = board.poll(time.perf_counter())  # 선으로 나갈 바이트

왜 `ReplaySource` 로 부족한가
-----------------------------
`scripts/serial_bridge.py` 의 `--replay` 는 **PC 안에서 바이트를 만들어 준다.**
그것으로 파서·처리기·화면은 검증되지만, 시연 당일에 실제로 쓰는 길은
`SerialSource` — **pyserial 이 포트를 열고, 명령을 보내고, OS 버퍼에서 읽는**
경로다. 그 경로는 `--replay` 에서 **한 줄도 실행되지 않는다.**

이 모듈은 그 구멍을 메운다. `scripts/fake_arduino.py` 가 이것을 **가상 시리얼
포트(PTY)** 뒤에 놓으면, 브리지는 자기가 실제 보드에 붙었는지 알 방법이 없다.

`ReplaySource` 와 무엇이 다른가
------------------------------
| | `ReplaySource` | `FakeBoard` |
|---|---|---|
| 어디서 도나 | 브리지 프로세스 안 | 별도 프로세스 + **진짜 포트** |
| 명령(`2`·`b`·`?`) | **안 받는다** | 펌웨어와 같은 상태 머신 |
| 부팅 헤더 | 없다 | `# logger ready` + `# ecgstream v1 …` |
| 줄 끝 | `\\n` | **`\\r\\n`** (Arduino `println` 이 그렇다) |
| 드롭 | `--drop` 확률 | **송신 버퍼 64 B 가 차면** — baud 가 정한다 |

마지막 줄이 이 모듈의 핵심이다. 펌웨어가 샘플을 버리는 조건은 확률이 아니라
`availableForWrite() < 5` 이고, 그것은 **baud · fs · 형식이 함께 정한다.**
그래서 "ASCII 는 1 kHz 를 못 버틴다"(`docs/30_realtime_demo.md` 6.2 (1))가
여기서 **재현되는 사실**이 된다 — 표에 적힌 계산이 아니라.

원본은 `hardware/arduino_ecg_logger/arduino_ecg_logger.ino` 이고,
`tests/test_fake_board.py` 가 이 파일과 그 `.ino` 가 어긋나지 않는지 본다.
"""
from __future__ import annotations

import numpy as np

__all__ = ["FakeBoard", "UsbPipe", "SERIAL_TX_BUFFER_SIZE", "ADC_MAX",
           "BOOT_BANNER", "BOOTLOADER_S"]

# AVR `HardwareSerial` 의 송신 링버퍼. `availableForWrite()` 는 링버퍼라
# 한 칸을 못 쓰므로 **최대 63** 을 돌려준다 — 이 −1 이 있어야 경계가 맞는다.
SERIAL_TX_BUFFER_SIZE = 64

ADC_MAX = 1023
LEADOFF_RAW = 0xFFFF
BOOT_BANNER = b"# logger ready\r\n"

# Uno 의 부트로더는 리셋 뒤 새 스케치를 기다렸다가 넘어간다. 그동안 보드는
# **아무것도 안 보낸다** — 브리지가 2 초를 기다리는 이유가 이것이다.
BOOTLOADER_S = 1.6

# 펌웨어의 임계값. BINARY 는 프레임 5 B, ASCII 는 한 줄 최대 약 16 B 다.
_MIN_FREE = {"bin": 5, "ascii": 16}


class FakeBoard:
    """`arduino_ecg_logger.ino` 의 상태 머신. **시계는 밖에서 준다.**

    시계를 인자로 받는 이유는 테스트다 — 실시간으로 기다리면 1 kHz 를 재는 데
    분 단위가 걸리고, 그러면 아무도 안 돌린다. `poll(now)` 에 가상 시각을 넣으면
    같은 코드가 **즉시** 돈다.
    """

    def __init__(self, counts: np.ndarray, *, fs: int = 500,
                 baud: int = 115200, mode: str = "ascii",
                 drift_ppm: float = 0.0, leadoff_every_s: float = 0.0,
                 leadoff_len_s: float = 0.7,
                 tx_buffer: int = SERIAL_TX_BUFFER_SIZE,
                 accept_commands: bool = True,
                 boot_s: float = 0.0):
        self.counts = np.asarray(counts, dtype=np.int64)
        if self.counts.size == 0:
            raise ValueError("보낼 것이 없다 — counts 가 비었다")
        self.baud = int(baud)
        self.tx_buffer = int(tx_buffer)
        self.drift = 1.0 + drift_ppm * 1e-6
        self.leadoff_every = float(leadoff_every_s)
        self.leadoff_len = float(leadoff_len_s)
        self.mode = "bin" if mode.startswith("b") else "ascii"
        # **명령을 모르는 판을 흉내낸다.** 스케치가 구 버전이면 `'2'` 를 받고도
        # 500 Hz 를 계속 준다. 그러면 PC 는 250 Hz 라 믿고 처리하고, 시간축이
        # 두 배로 틀린다 — **에러는 안 나고 심박수만 절반이 된다** (F-42).
        self.accept_commands = bool(accept_commands)
        self.dropped = 0
        self.seq = 0
        self.n_sent = 0                 # 실제로 선에 나간 샘플
        self.k = 0                      # 보드가 뜬 샘플 번호 (드롭해도 진행)
        self._pending = 0.0             # 송신 버퍼에 남은 바이트
        self._t_open: float | None = None
        self._t_board = 0.0             # 보드 시계 [s]
        self._t_next = 0.0              # 다음 샘플 시각
        self._t_drained = 0.0           # 여기까지 UART 가 빼냈다
        self.boot_s = float(boot_s)     # 부트로더가 조용한 시간
        self._boot_fs = int(fs)
        # **리셋은 그 스케치의 켜질 때 상태로 돌아간다.** 지금 펌웨어는 ASCII 로
        # 켜지지만, BINARY 로 구워진 판을 흉내낼 때는 그쪽이 «켜질 때» 다.
        self._boot_mode = self.mode
        self._banner_at: float | None = None
        self._out = bytearray()
        self._set_fs(int(fs), boot=True)
        self._t_next = self.boot_s
        self._banner_at = self.boot_s if self.boot_s > 0 else None
        if self._banner_at is None:
            self._out += BOOT_BANNER
            self._header()

    # ------------------------------------------------------------ 펌웨어 명령
    def _set_fs(self, fs: int, boot: bool = False) -> None:
        """`set_fs()`. **seq 와 드롭 계수까지 0 으로 돌린다** — 펌웨어가 그렇다."""
        if fs <= 0:
            raise ValueError("fs 는 양수여야 한다")
        self.fs = int(fs)
        self.period = 1.0 / self.fs
        self.seq = 0
        self.dropped = 0
        if not boot:
            # 시각 기준도 함께 돌아간다(`t0_us = micros()`). 이것 때문에 명령을
            # **바이너리 프레임이 흐르는 도중에** 보내면 seq 가 0 으로 튀고,
            # PC 는 그것을 «셀 수 없는 손실» 로 읽는다 — 6.2 의 순서 규칙.
            self._t_next = max(self._t_board, self.boot_s)

    def _header(self) -> None:
        self._out += (f"# ecgstream v1 fs={self.fs} "
                      f"mode={'bin' if self.mode == 'bin' else 'ascii'} "
                      f"bits=10 vref=5.00 dropped={self.dropped}\r\n").encode()

    def reset(self, now: float) -> None:
        """**DTR 리셋.** 포트를 열면 ATmega328P 가 리셋되고 `setup()` 부터 다시 돈다.

        PTY 에는 modem control line 이 없어(`TIOCMGET` 이 ENOTTY 다) DTR 자체는
        흉내낼 수 없다. 그러나 **관찰 가능한 결과**는 같다 — 포트가 열리는
        순간 보드가 처음부터 시작하고, 부트로더가 끝날 때까지 조용하다.
        `scripts/fake_arduino.py` 가 열림을 감지해 이것을 부른다.
        """
        self.dropped = 0
        self.seq = 0
        self.n_sent = 0
        self.k = 0
        self._pending = 0.0
        self.mode = self._boot_mode         # 지금 펌웨어는 ASCII 로 켜진다
        self._t_open = now
        self._t_board = 0.0
        self._t_drained = 0.0
        self.fs = self._boot_fs
        self.period = 1.0 / self.fs
        # **부트로더가 끝나기 전에는 한 바이트도 안 나간다.** 이것이 없으면
        # 「열자마자 2 초를 버린다」는 브리지의 대기가 왜 필요한지 안 드러난다.
        self._t_next = self.boot_s
        self._out = bytearray()
        self._banner_at = self.boot_s

    def _maybe_banner(self) -> None:
        if self._banner_at is not None and self._t_board >= self._banner_at:
            self._banner_at = None
            self._out += BOOT_BANNER
            self._header()

    def feed_command(self, data: bytes) -> None:
        """`handle_command()`. 한 바이트씩, 받은 순서대로.

        **부트로더가 도는 동안 온 바이트는 스케치에 닿지 않는다.** 리셋 직후의
        1.6 초는 부트로더가 새 스케치를 기다리는 시간이고, 그때 온 것은
        부트로더가 먹는다. 브리지가 포트를 연 뒤 2 초를 기다렸다가 명령을
        보내는 이유가 이것이다 — 안 기다리면 `'2'`·`'b'` 가 사라지고 보드는
        기본값(500 Hz · ASCII)으로 남는다.
        """
        if self._t_board < self.boot_s:
            return
        if not self.accept_commands:
            return
        for c in data:
            ch = bytes([c])
            if ch == b"a":
                self.mode = "ascii"
                self._header()
            elif ch == b"b":
                self.mode = "bin"          # 헤더를 안 찍는다 — 프레임에 섞인다
            elif ch == b"2":
                self._set_fs(250)
            elif ch == b"5":
                self._set_fs(500)
            elif ch == b"1":
                self._set_fs(1000)
            elif ch == b"r":
                self._set_fs(self.fs)
            elif ch == b"?":
                self._header()
            # 그 밖(개행 등)은 무시한다

    # -------------------------------------------------------------- 선으로
    def poll(self, now: float) -> bytes:
        """`loop()` 를 `now` 까지 돌리고, 선으로 나갈 바이트를 준다."""
        if self._t_open is None:
            self._t_open = now
            self._t_board = 0.0
            self._t_drained = 0.0
            # **부트로더가 끝나야 첫 샘플이 나간다.** 여기서 0 으로 두면
            # 「열자마자 조용한 1.6 초」가 사라진다.
            self._t_next = self.boot_s
        # **보드의 시계는 PC 의 시계가 아니다.** Uno 는 세라믹 레조네이터라
        # 0.3 % 쯤 어긋나고, 그 어긋남이 있어야 «PC 벽시계로 샘플을 센다» 는
        # 구현이 여기서 무너진다 (6.2 (3)).
        self._t_board = (now - self._t_open) * self.drift
        self._maybe_banner()
        # **샘플 하나마다 그 시각까지 배출한다.** 블록 단위로 배출하면 결과가
        # `poll` 을 얼마나 자주 부르느냐에 따라 달라진다 — 실제 보드는 `loop()`
        # 안에서 한 샘플씩 처리하고 그 사이에도 UART 는 계속 비워지므로,
        # 여기서도 그렇게 해야 드롭이 **호출자와 무관한 사실**이 된다.
        while self._t_board >= self._t_next:
            self._drain_to(self._t_next)
            self._one_sample()
            self._t_next += self.period
        self._drain_to(self._t_board)
        out, self._out = bytes(self._out), bytearray()
        return out

    def _drain_to(self, t: float) -> None:
        """UART 는 초당 `baud/10` 바이트를 빼낸다 (시작 1 + 데이터 8 + 정지 1)."""
        dt = max(t - self._t_drained, 0.0)
        self._t_drained = max(t, self._t_drained)
        self._pending = max(0.0, self._pending - dt * self.baud / 10.0)

    def _one_sample(self) -> None:
        lead = (self.leadoff_every > 0.0
                and (self._t_next % self.leadoff_every) < self.leadoff_len)
        val = LEADOFF_RAW if lead else int(self.counts[self.k % self.counts.size])
        self.k += 1

        # **자리가 없으면 버리고 센다.** 블록하면 fs 가 조용히 흔들리고,
        # 그러면 R-peak 간격이 전부 틀린다 (`.ino` 의 같은 주석).
        free = self.tx_buffer - 1 - int(self._pending)
        if free < _MIN_FREE[self.mode]:
            self.seq = (self.seq + 1) & 0xFF
            self.dropped += 1
            return

        if self.mode == "bin":
            lo, hi = val & 0xFF, (val >> 8) & 0xFF
            frame = bytes([0xA5, self.seq, lo, hi, 0xA5 ^ self.seq ^ lo ^ hi])
        else:
            # `Serial.println` 은 **CR+LF** 를 붙인다. lead-off 는 `-1`.
            t_ms = int(self._t_next * 1000.0)
            frame = f"{t_ms},{-1 if val == LEADOFF_RAW else val}\r\n".encode()
        self._out += frame
        self._pending += len(frame)
        self.seq = (self.seq + 1) & 0xFF
        self.n_sent += 1


def synth_counts(duration_s: float, fs: int, seed: int = 7,
                 snr_db: float = 8.0) -> np.ndarray:
    """AD8232 + 10 bit ADC 출력 흉내. `ReplaySource._make` 와 같은 척도다."""
    from ..data.mixer import mix_at_snr
    from ..data.noise import mixed_noise
    from ..data.synthetic import synth_ecg

    gen = np.random.default_rng(seed)
    clean = synth_ecg(duration_s=duration_s, fs=fs, seed=seed).x
    noise, _ = mixed_noise(clean.size, fs, gen)
    x, _, _ = mix_at_snr(clean, noise, snr_db)
    x = np.asarray(x, dtype=np.float64)
    s = np.percentile(np.abs(x - np.median(x)), 99) or 1.0
    # 중앙 512, 진폭이 ADC 범위의 약 1/3 — 실제 AD8232 + 5 V 에서 그 정도다.
    return np.clip(512 + (x - np.median(x)) / s * 170, 0, ADC_MAX).round()


class UsbPipe:
    """**USB CDC 는 바이트를 한 개씩 배달하지 않는다.** 그 뭉침을 흉내낸다.

    Uno 에는 칩이 둘이다 — 스케치가 도는 ATmega328P 와, UART 를 USB 로 바꾸는
    ATmega16U2. 뒤엣것이 UART 에서 받은 바이트를 **bulk 패킷(최대 64 B)** 에
    담고, 호스트는 **1 ms 프레임마다 폴링**해 가져간다. 그래서 PC 가 보는 것은
    「250 Hz 로 고르게 오는 샘플」이 아니라 **폴링 간격 단위의 덩어리**다.

    이것이 왜 검증거리인가 — `docs/30_realtime_demo.md` 6.2 (3) 은 「시간축의
    주인은 보드이고 화면은 도착한 만큼 진행한다」고 정해 뒀다. 그 규칙이
    맞는지는 **도착이 고르지 않을 때만** 드러난다. 고르게 오면 벽시계로 세는
    잘못된 구현도 똑같이 잘 돈다.

        pipe = UsbPipe(poll_ms=1.0, packet=64, hiccup_every_s=2.0, hiccup_ms=40)
        pipe.push(board.poll(now))          # 보드가 UART 로 낸 것
        data = pipe.pop(now)                # 호스트가 실제로 받는 것

    `hiccup` 은 **호스트가 잠깐 안 가져가는 구간**이다(다른 프로세스가 CPU 를
    쥐거나 USB 대역을 나눠 쓸 때 실제로 생긴다). 그동안 쌓인 것이 그 뒤에
    한꺼번에 나가므로, 뭉침의 극단이 여기서 만들어진다.
    """

    def __init__(self, poll_ms: float = 1.0, packet: int = 64,
                 hiccup_every_s: float = 0.0, hiccup_ms: float = 0.0,
                 buffer_bytes: int = 8192):
        self.poll_s = max(float(poll_ms), 1e-6) / 1000.0
        self.packet = int(packet)
        self.hiccup_every = float(hiccup_every_s)
        self.hiccup_s = float(hiccup_ms) / 1000.0
        self.buffer_bytes = int(buffer_bytes)
        self.overflow = 0                  # 호스트 버퍼가 넘쳐 잃은 바이트
        self._buf = bytearray()
        self._t_next: float | None = None

    def push(self, data: bytes) -> None:
        if not data:
            return
        self._buf += data
        if len(self._buf) > self.buffer_bytes:
            # 실제로도 여기는 잃는 자리다 — 호스트가 오래 안 가져가면 tty
            # 버퍼가 넘치고, 넘친 바이트는 **아무도 세지 않는다.**
            drop = len(self._buf) - self.buffer_bytes
            del self._buf[:drop]
            self.overflow += drop

    def pop(self, now: float) -> bytes:
        if self._t_next is None:
            self._t_next = now
        if self.hiccup_every > 0.0 and (now % self.hiccup_every) < self.hiccup_s:
            return b""                     # 호스트가 지금 안 가져간다
        if now < self._t_next:
            return b""
        n = int((now - self._t_next) // self.poll_s) + 1
        self._t_next += n * self.poll_s
        take = min(len(self._buf), n * self.packet)
        if take == 0:
            return b""
        out = bytes(self._buf[:take])
        del self._buf[:take]
        return out

    @property
    def pending(self) -> int:
        return len(self._buf)

