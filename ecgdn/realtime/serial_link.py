"""R-5: 아두이노 시리얼 -> 샘플 스트림. **선 위에서 잃은 것을 숨기지 않는다.**

    parser = BinaryParser()
    for data in serial_reads():
        ch = parser.feed(data)
        sp.push(ch.x)                     # StreamProcessor 로

왜 파서를 따로 두는가
--------------------
`scripts/serial_bridge.py` 는 스레드·소켓·모델을 함께 다루므로 **하드웨어
없이는 못 돌린다.** 선에서 오는 바이트를 샘플로 바꾸는 규칙은 그것과 무관한
순수 계산이고, **여기서 틀리면 파형이 조용히 틀린다** — 그래서 갈라 놓고
테스트로 고정한다.

선에서 실제로 생기는 일 넷
-------------------------
**(1) 샘플 손실.** USB CDC 는 바이트를 잃지 않지만(재전송한다), 보드 쪽
송신 버퍼가 차면 펌웨어가 **샘플을 버린다**(그렇게 하라고 짰다 — 블록하면
fs 가 조용히 흔들린다). 그래서 프레임에 `seq` 를 실었고 여기서 그것을 센다.

**손실을 그냥 이어 붙이면 안 된다.** 100 샘플이 빠진 자리를 이어 붙이면
그 뒤 신호가 0.4 s 앞당겨지고, R-peak 간격이 틀어진다. 그래서 **빠진 수만큼
직전 값을 채워 넣고**(`ok=False` 로 표시) 시간축을 지킨다. 채운 구간은
화면에서 회색으로 그린다 — 없는 것을 있는 척하지 않는다.

**(2) lead-off.** 전극이 떨어지면 펌웨어가 `0xFFFF` 를 보낸다. 이것을 숫자로
그대로 흘리면 필터에 거대한 계단이 들어가고, IIR 상태가 망가져 **전극을 다시
붙인 뒤에도 몇 초간 파형이 엉킨다.** 그래서 직전 값을 유지하고 표시만 남긴다.

**(3) 프레임 깨짐.** 동기 바이트 `0xA5` 는 payload 에도 나온다. 그래서
검사합을 함께 보고, 맞지 않으면 한 바이트 밀어 다시 찾는다.

**(4) 시각.** PC 의 벽시계로 샘플 수를 세면 안 된다 — 보드의 클럭(Uno 는
세라믹 레조네이터, 약 ±0.5 %)과 어긋나 서서히 밀린다. **시간축의 주인은
보드**이고, PC 는 도착한 만큼만 그린다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps

__all__ = ["SampleChunk", "BinaryParser", "AsciiParser", "CausalDecimator",
           "adc_to_mv", "SYNC", "FRAME_LEN", "LEADOFF"]

SYNC = 0xA5
FRAME_LEN = 5
LEADOFF = 0xFFFF

# seq 는 1 바이트라 **255 샘플까지만** 손실을 셀 수 있다. 그보다 크게 빠지면
# 몇 바퀴 돌았는지 알 수 없으므로 세는 대신 "모른다" 로 보고한다.
MAX_GAP = 200


@dataclass
class SampleChunk:
    """한 번의 `feed` 로 확정된 샘플들."""
    x: np.ndarray                      # ADC counts (lead-off/손실은 채운 값)
    ok: np.ndarray                     # 실제로 측정된 샘플인가
    n_lost: int = 0                    # seq 로 확인한 손실 (채워 넣은 수)
    n_leadoff: int = 0
    n_bad: int = 0                     # 검사합 불일치로 버린 바이트
    gap_unknown: bool = False          # 손실이 커서 셀 수 없었다 -> 리셋 권고

    def __len__(self) -> int:
        return int(self.x.size)


def _empty() -> SampleChunk:
    return SampleChunk(np.zeros(0), np.zeros(0, dtype=bool))


@dataclass
class _Base:
    last_val: float = 512.0            # 채워 넣을 값 (직전 유효 샘플)
    _out: list[float] = field(default_factory=list)
    _ok: list[bool] = field(default_factory=list)

    def _emit(self, val: float | None) -> None:
        """`None` 이면 '없는 샘플' — 직전 값으로 채우고 `ok=False`."""
        if val is None:
            self._out.append(self.last_val)
            self._ok.append(False)
        else:
            self.last_val = val
            self._out.append(val)
            self._ok.append(True)

    def _take(self, **kw) -> SampleChunk:
        ch = SampleChunk(np.asarray(self._out, dtype=np.float64),
                         np.asarray(self._ok, dtype=bool), **kw)
        self._out, self._ok = [], []
        return ch


class BinaryParser(_Base):
    """5 바이트 프레임 파서. 재동기 + 손실 계수."""

    def __init__(self) -> None:
        super().__init__()
        self._buf = bytearray()
        self._seq: int | None = None

    def reset(self) -> None:
        self._buf.clear()
        self._seq = None

    def feed(self, data: bytes) -> SampleChunk:
        self._buf.extend(data)
        n_lost = n_leadoff = n_bad = 0
        unknown = False
        b = self._buf
        i = 0
        while len(b) - i >= FRAME_LEN:
            if b[i] != SYNC:
                i += 1
                n_bad += 1
                continue
            seq, lo, hi, x = b[i + 1], b[i + 2], b[i + 3], b[i + 4]
            if (SYNC ^ seq ^ lo ^ hi) != x:
                # 동기 바이트처럼 보였을 뿐이다. **한 바이트만** 민다 —
                # 프레임 단위로 밀면 진짜 프레임을 건너뛴다.
                i += 1
                n_bad += 1
                continue
            if self._seq is not None:
                gap = (seq - self._seq - 1) & 0xFF
                if gap:
                    if gap > MAX_GAP:
                        unknown = True          # 몇 바퀴 돌았는지 알 수 없다
                    else:
                        n_lost += gap
                        for _ in range(gap):
                            self._emit(None)
            self._seq = seq
            val = lo | (hi << 8)
            if val == LEADOFF:
                n_leadoff += 1
                self._emit(None)
            else:
                self._emit(float(val))
            i += FRAME_LEN
        del b[:i]
        return self._take(n_lost=n_lost, n_leadoff=n_leadoff, n_bad=n_bad,
                          gap_unknown=unknown)


class AsciiParser(_Base):
    """`t_ms,adc` 줄 파서 — 아두이노 IDE 플로터와 같은 형식.

    **손실을 셀 수 없다.** `t_ms` 로 추정할 수는 있지만 ms 해상도라 fs 가
    500 Hz 면 한 샘플이 2 ms 로 반올림 오차와 같은 크기다. 그래서 이 모드는
    **디버깅·수집용**이고, 시연 경로는 BINARY 를 쓴다.
    """

    def __init__(self) -> None:
        super().__init__()
        self._buf = bytearray()

    def reset(self) -> None:
        self._buf.clear()

    def feed(self, data: bytes) -> SampleChunk:
        self._buf.extend(data)
        n_leadoff = n_bad = 0
        lines = self._buf.split(b"\n")
        self._buf = bytearray(lines.pop())          # 마지막은 미완성일 수 있다
        for ln in lines:
            s = ln.strip()
            if not s or s.startswith(b"#"):
                continue
            parts = s.split(b",")
            if len(parts) != 2:
                n_bad += 1
                continue
            try:
                v = int(parts[1])
            except ValueError:
                n_bad += 1
                continue
            if v < 0:
                n_leadoff += 1
                self._emit(None)
            else:
                self._emit(float(v))
        return self._take(n_leadoff=n_leadoff, n_bad=n_bad)


class CausalDecimator:
    """정수배 인과 데시메이션. **가능하면 쓰지 말 것.**

    보드를 250 Hz 로 돌리면 리샘플이 아예 없고 학습 경로와 **같은 신호**가 된다.
    이 클래스는 이미 500 Hz 로 구워진 보드를 그대로 쓰는 경우의 대비책이다.

    오프라인 경로(`resample_to`)는 다상 FIR 이라 영위상에 가깝고, 여기서는 그것을
    못 쓴다 — 미래를 보기 때문이다(F-25 와 같은 이유). 대신 차단 100 Hz 인
    인과 Butterworth 를 앞에 두는데, 이 값은 우연이 아니라 **공통 front-end 의
    `lp_hz` 와 같은 값**이다. 즉 front-end 가 어차피 버릴 대역만 접힌다.
    """

    def __init__(self, fs_in: float, fs_out: float, order: int = 4,
                 cutoff_hz: float = 100.0):
        if fs_in <= 0 or fs_out <= 0:
            raise ValueError("fs 는 양수여야 한다")
        ratio = fs_in / fs_out
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(
                f"정수배만 지원한다: {fs_in:g} -> {fs_out:g} (비 {ratio:.3f}). "
                "보드의 fs 를 250·500·1000 중에서 고를 것.")
        self.m = int(round(ratio))
        self.fs_in, self.fs_out = float(fs_in), float(fs_out)
        if self.m == 1:
            self._sos = None
        else:
            nyq_out = fs_out / 2.0
            fc = min(cutoff_hz, 0.9 * nyq_out)
            self._sos = sps.butter(order, fc / (fs_in / 2.0),
                                   btype="lowpass", output="sos")
            self._zi = sps.sosfilt_zi(self._sos) * 0.0
        self._phase = 0                    # 데시메이션 격자 위치 (블록 경계 유지)

    def __call__(self, ch: SampleChunk) -> SampleChunk:
        if self.m == 1 or ch.x.size == 0:
            return ch
        y, self._zi = sps.sosfilt(self._sos, ch.x, zi=self._zi)
        idx = np.arange(self._phase, ch.x.size, self.m)
        self._phase = (self._phase - ch.x.size) % self.m
        if idx.size == 0:
            return SampleChunk(np.zeros(0), np.zeros(0, dtype=bool),
                               n_lost=ch.n_lost, n_leadoff=ch.n_leadoff,
                               n_bad=ch.n_bad, gap_unknown=ch.gap_unknown)
        # **`ok` 는 창 안에 하나라도 나쁜 샘플이 있으면 나쁘다.** 필터가
        # 그 값을 이웃으로 퍼뜨리기 때문에, 뽑은 샘플만 보면 오염을 놓친다.
        pad = -(-ch.ok.size // self.m) * self.m
        okm = np.pad(ch.ok, (0, pad - ch.ok.size), constant_values=True)
        okm = okm.reshape(-1, self.m).all(axis=1)[: idx.size]
        return SampleChunk(y[idx], okm, n_lost=ch.n_lost, n_leadoff=ch.n_leadoff,
                           n_bad=ch.n_bad, gap_unknown=ch.gap_unknown)


def adc_to_mv(counts: np.ndarray, bits: int = 10, vref: float = 5.0,
              gain: float = 1100.0) -> np.ndarray:
    """ADC 카운트를 **전극 단 mV** 로. 이득을 안 나누면 단위가 회로 출력이다."""
    full = float(2 ** bits - 1)
    return np.asarray(counts, dtype=np.float64) / full * vref / gain * 1e3
