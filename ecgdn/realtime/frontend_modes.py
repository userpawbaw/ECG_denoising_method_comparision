"""실시간 front-end **세 가지**를 같은 계약으로 묶는다 (F-27 · F-29 · F-30).

왜 세 가지인가
-------------
«위상 왜곡» 과 «평활 대역이 중심선에 붙는가» 는 **같은 방향으로 못 간다**.
어느 쪽을 더 싫어하는지는 **보는 사람이 정할 문제**라, 화면에서 바꿔 볼 수
있게 셋을 나란히 둔다 (`docs/13_lookahead_fe.md` · `docs/14_median_vs_zerophase.md`):

| 모드 | 지연 | QRS 직후 준위 이동 d1 | T-P 준위 산포 d1 | 성격 |
|---|---|---|---|---|
| `causal`  | **0 ms**  | 5.7 %R | 4.7 %R | 선형. 가장 빠르고 가장 찌그러진다 |
| `zerophase` | 548 ms | **0.1 %R** | 2.5 %R | 선형. 위상 왜곡이 원리적으로 0 |
| `median`  | 448 ms | 4.7 %R | **1.0 %R** | **비선형.** 가장 평평하지만 중첩이 깨진다 |

(지연은 실시간 브리지 설정 기준 — FE hop 24 ms, 교차 페이드 24 ms.)

(4)번 열은 처음에 「S 골 오차」라고 불렀는데 **그 이름이 틀렸다** — S 골 자체는
어느 방식도 안 깎는다(`docs/16` §1). 재는 것은 QRS 옆의 기준 준위 = ST 이동이다.

지연에는 **교차 페이드**가 들어 있다. 그것이 없으면 블록 경계에 계단이 생겨
T-P 구간에서 내부 차분의 **23 배**로 튄다 (F-36). 기본값은 `hop` 과 같다 —
이음매의 간격이 hop 이라 거기에 묶는 것이 자연스럽고, **hop 을 줄이면 지연도
함께 준다.** 그래서 FE hop 을 48 -> 24 ms 로 줄이면 교차 페이드를 넣고도
지연이 548 ms 로 **예전과 같다** (이음매 15.0 -> 0.65, 오프라인차 8.5 -> 6.2 %R).
FE 는 블록당 0.57 ms 라 hop 을 줄여도 예산의 2 % 다.

계약 — **지연이 있어도 표본 번호는 안 밀린다**
---------------------------------------------
`push(block) -> ndarray` 는 **그 시점에 확정된 출력만** 낸다. 길이는 입력과
다를 수 있다(초반에는 0). 중요한 것은 **낸 표본이 입력 표본 번호 `n_emitted`
부터 차례로 대응한다**는 것이다 — 즉 «같은 표본을 늦게 낼 뿐» 시간축이
밀리지 않는다.

그래서 `Aligner` 와 화면의 원시 패널은 **손댈 필요가 없다.** 지연이 있는
front-end 를 «표본 번호를 바꾸는 것» 으로 만들었다면 정렬기·원시 버퍼·커서가
전부 따라 움직여야 했을 것이다.

`latency_samples` 는 «첫 출력이 나오기까지 몇 표본을 더 받아야 하는가» 이고,
화면은 이것을 지연 표시에 더한다.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

from ..config import DEFAULT_FE, DEFAULT_FE_CAUSAL, FS
from .frontend_stream import StreamingFrontEnd

__all__ = ["FE_MODES", "fe_intrinsic", "build_fe",
           "CausalFE", "BlockZeroPhaseFE", "MedianBaselineFE"]

# 화면에 그대로 쓰는 이름표. 코드와 UI 가 갈라지지 않게 **여기 하나만** 둔다.
#
# `bandpass` 는 «이 front-end 가 대역통과+노치인가» 다. 실시간 경로에서 `M01`
# (대역통과+노치)의 대응물을 front-end 출력으로 **대체해도 되는지**를 이것이
# 정한다 — 대역통과이면 같은 필터를 두 번 거는 셈이라 대체가 맞고(F-25),
# 중앙값은 대역통과가 아니므로 **대체하면 안 된다.** `M01` 은 그 위에서
# 따로 돌아야 하는 별개의 방법이다.
FE_MODES = {
    "causal": dict(label="인과 o1 · 0.5 Hz", bandpass=True,
                   note="지연 0 — 가장 빠르다. QRS 뒤가 찌그러진다"),
    "zerophase": dict(label="블록 영위상 · 미리보기 0.5 s", bandpass=True,
                      note="위상 왜곡 0 — ST 준위가 기준과 겹친다"),
    "median": dict(label="중앙값 200+600 ms", bandpass=False,
                   note="가장 평평하다. 비선형이라 평가 기준이 흔들린다"),
}


def fe_intrinsic(mode: str) -> set[str]:
    """이 front-end 에서 «필터가 곧 방법» 이 되는 이름들.

    `M_FE`(front-end 만)는 어느 모드에서나 front-end 출력 그 자체다.
    `M01`/`M01d` 는 **front-end 가 대역통과일 때만** 그렇다.
    """
    base = {"M_FE"}
    return base | ({"M01", "M01d"} if FE_MODES[mode]["bandpass"] else set())


class _Base:
    """공통 — 「확정된 것만 낸다」 를 여기서 구현한다."""

    latency_samples = 0

    def __init__(self, fs: float = FS):
        self.fs = float(fs)
        self._buf: list[float] = []      # 아직 안 낸 것을 포함한 과거 전부(창 길이만큼만 유지)
        self._n_seen = 0                 # 받은 표본 수
        self._n_out = 0                  # 낸 표본 수

    def push(self, block):               # pragma: no cover - 하위에서 구현
        raise NotImplementedError

    def reset(self) -> None:
        self._buf, self._n_seen, self._n_out = [], 0, 0


class CausalFE(_Base):
    """지금 쓰는 것. `StreamingFrontEnd` 를 그대로 감싼다 — 지연 0."""

    latency_samples = 0

    def __init__(self, fs: float = FS, cfg=DEFAULT_FE_CAUSAL):
        super().__init__(fs)
        self._fe = StreamingFrontEnd(fs, cfg)

    def push(self, block):
        out = self._fe.push(block)
        self._n_seen += len(block)
        self._n_out += len(out)
        return out

    def reset(self) -> None:
        super().reset()
        self._fe.reset()


class _Windowed(_Base):
    """미래 `look` 표본을 기다렸다가 창 하나를 통째로 처리하는 방식의 공통 뼈대.

    `emitted` 부터 `hop` 개를 내려면 `emitted + hop + look` 까지 받아야 한다.
    과거 `past` 는 **이미 받아 둔 것이라 지연을 안 만든다** — 넉넉히 준다
    (처음 짤 때 과거를 미래와 같이 줄였다가 인과보다 나쁜 값이 나왔다, D-19 후속).
    """

    def __init__(self, fs: float = FS, look_s: float = 0.5, past_s: float = 4.0,
                 hop_s: float = 0.096, xfade_s: float | None = None):
        super().__init__(fs)
        self.look = int(round(look_s * fs))
        self.past = int(round(past_s * fs))
        self.hop = max(1, int(round(hop_s * fs)))
        # **기본은 hop 과 같다.** 이음매의 간격이 hop 이므로 그것에 묶는 것이
        # 자연스럽고, hop 을 줄이면 지연도 함께 준다. 0 을 주면 끌 수 있다
        # (회귀 시험이 «껐을 때 결함이 나타나는지» 를 확인하는 데 쓴다).
        self.xfade = self.hop if xfade_s is None else max(0, int(round(xfade_s * fs)))
        # 교차 페이드는 **지연을 산다.** 표본 n 은 그것을 덮는 마지막 블록이
        # 처리된 뒤에 확정되고, 그 블록은 n 보다 xfade 만큼 뒤에서 시작한다.
        self.latency_samples = self.look + self.hop + self.xfade
        # 겹쳐 더한 값과 가중치 합. **numpy 로 둔다** — 파이썬 리스트에서
        # 앞을 하나씩 지우면 매번 전체가 밀려 O(n^2) 이 된다 (F-31 과 같은 함정).
        self._acc = np.zeros(0)
        self._wsum = np.zeros(0)
        self._acc0 = 0                    # _acc[0] 의 절대 표본 번호

    def _process(self, w: np.ndarray) -> np.ndarray:   # pragma: no cover
        raise NotImplementedError

    def reset(self) -> None:
        super().reset()
        self._acc, self._wsum, self._acc0 = np.zeros(0), np.zeros(0), 0

    def _fade(self, m: int) -> np.ndarray:
        """양 끝을 raised-cosine 으로 눕힌 가중치. 겹치는 두 조각의 합이 1 이다."""
        w = np.ones(m)
        k = min(self.xfade, m // 2)
        if k > 0:
            r = 0.5 * (1 - np.cos(np.pi * np.arange(1, k + 1) / (k + 1)))
            w[:k] = r
            w[-k:] = r[::-1]
        return w

    def push(self, block):
        block = np.asarray(block, dtype=np.float64).ravel()
        self._buf.extend(block.tolist())
        self._n_seen += block.size
        out: list[float] = []
        while self._n_seen >= self._n_out + self.hop + self.look:
            base = self._n_seen - len(self._buf)          # _buf[0] 의 절대 번호
            a = max(base, self._n_out - self.past)
            b = min(self._n_seen, self._n_out + self.hop + self.look)
            w = np.asarray(self._buf[a - base: b - base], dtype=np.float64)
            v = self._process(w)
            if self.xfade == 0:
                k0 = self._n_out - a
                out.extend(v[k0: k0 + self.hop].tolist())
                self._n_out += self.hop
            else:
                out.extend(self._blend(v, a, b))
            # 더 필요 없는 과거는 버린다 — 무한히 자라면 안 된다
            keep = self.past + self.hop + self.look + self.hop + self.xfade
            if len(self._buf) > keep:
                del self._buf[:len(self._buf) - keep]
        return np.asarray(out, dtype=np.float64)

    def _blend(self, v, a, b) -> list[float]:
        """창 하나의 결과를 누적기에 **겹쳐 더하고**, 확정된 것만 돌려준다.

        같은 표본을 서로 다른 창으로 계산한 값의 가중 평균이 된다. 창 위치에
        따른 오차는 부호가 오가므로 평균에서 상쇄된다 — 그래서 이음매가
        사라질 뿐 아니라 **오프라인 근사도까지 좋아진다** (F-36).
        """
        X, H = self.xfade, self.hop
        s0, s1 = max(a, self._n_out - X), min(b, self._n_out + H + X)
        if self._acc.size == 0:
            self._acc0 = s0
        need = s1 - self._acc0
        if need > self._acc.size:                     # 누적기를 오른쪽으로 넓힌다
            grow = need - self._acc.size
            self._acc = np.concatenate([self._acc, np.zeros(grow)])
            self._wsum = np.concatenate([self._wsum, np.zeros(grow)])
        seg = v[s0 - a: s1 - a]
        wgt = self._fade(seg.size)
        i = s0 - self._acc0
        self._acc[i: i + seg.size] += seg * wgt
        self._wsum[i: i + seg.size] += wgt
        self._n_out += H
        # 다음 창은 `_n_out - X` 부터 덮는다 — 그 앞은 더 안 바뀌므로 확정이다
        k = min(max(0, self._n_out - X - self._acc0), self._acc.size)
        if k == 0:
            return []
        ready = np.where(self._wsum[:k] > 1e-12, self._acc[:k] / self._wsum[:k], 0.0)
        self._acc, self._wsum = self._acc[k:], self._wsum[k:]
        self._acc0 += k
        return ready.tolist()


class BlockZeroPhaseFE(_Windowed):
    """창 안에서 `filtfilt` — **위상 왜곡이 원리적으로 0** 이다.

    창 [emitted−past, emitted+hop+look) 의 자료는 그 시점에 전부 손에 있다.
    filtfilt 의 가장자리 처리는 창 안에서만 하므로 미래를 더 훔치지 않는다.
    """

    def __init__(self, fs: float = FS, look_s: float = 0.5, past_s: float = 4.0,
                 hop_s: float = 0.096, cfg=DEFAULT_FE, xfade_s: float | None = None):
        super().__init__(fs, look_s, past_s, hop_s, xfade_s)
        nyq = fs / 2.0
        self._hp = sps.butter(cfg.order, cfg.hp_hz / nyq, btype="highpass", output="sos")
        self._lp = sps.butter(cfg.order, cfg.lp_hz / nyq, btype="lowpass", output="sos")

    def _process(self, w):
        pad = max(1, min(w.size - 1, w.size // 2))
        v = sps.sosfiltfilt(self._hp, w, padtype="odd", padlen=pad)
        return sps.sosfiltfilt(self._lp, v, padtype="odd", padlen=pad)


class MedianBaselineFE(_Windowed):
    """기저선을 **비선형으로 추정**해 뺀다 — 가장 평평하지만 선형이 아니다.

    창 `w1`(200 ms)은 QRS(약 80 ms)보다 넓어 QRS 가 중앙값이 될 수 없다.
    `w2`(600 ms)는 T 파(안정 시 약 200 ms)보다 넓어야 T 를 안 깎는다 —
    300 ms 로 줄이면 T 를 16 %R 깎는다 (F-29).
    """

    def __init__(self, fs: float = FS, w1_s: float = 0.2, w2_s: float = 0.6,
                 past_s: float = 4.0, hop_s: float = 0.096, cfg=DEFAULT_FE,
                 xfade_s: float | None = 0.0):
        # **기본은 끔.** 중앙값은 창 w1·w2 안의 자료만 보는 **국소 연산**이라
        # 창이 hop 만큼 미끄러져도 결과가 거의 안 바뀐다 — 이음매 비를 재 보면
        # 교차 페이드 없이도 0.95 다. 반면 `filtfilt` 는 **창 전체의 함수**라
        # 창이 바뀌면 값이 바뀐다(비 23). 그래서 교차 페이드가 필요한 쪽은
        # 영위상뿐이고, 여기서 켜면 지연만 hop 만큼 는다 (F-36).
        self.w1 = _odd(int(round(w1_s * fs)))
        self.w2 = _odd(int(round(w2_s * fs)))
        super().__init__(fs, look_s=(self.w1 // 2 + self.w2 // 2) / fs, past_s=past_s,
                         hop_s=hop_s, xfade_s=xfade_s)
        self._lp = sps.butter(cfg.order, cfg.lp_hz / (fs / 2), btype="lowpass",
                              output="sos")

    def _process(self, w):
        # **가장자리를 edge 로 채운다.** scipy 의 medfilt 는 창이 자료보다 길면
        # 0 으로 채우는데, 시작 구간에서 그러면 기저선 추정이 0 쪽으로 끌려가
        # 첫 몇 초가 통째로 틀어진다. (첫 검증에서 경고로 잡혔다.)
        pad_m = self.w2 // 2 + 1
        wp = np.pad(w, (pad_m, pad_m), mode="edge")
        base = sps.medfilt(sps.medfilt(wp, self.w1), self.w2)[pad_m:-pad_m]
        v = w - base
        pad = max(1, min(v.size - 1, v.size // 2))
        return sps.sosfiltfilt(self._lp, v, padtype="odd", padlen=pad)


def _odd(n: int) -> int:
    return n + 1 if n % 2 == 0 else n


def build_fe(mode: str, fs: float = FS, hop_s: float = 0.096):
    """이름 하나로 고른다. 모르는 이름은 **조용히 넘어가지 않고** 죽는다."""
    if mode not in FE_MODES:
        raise ValueError(f"front-end 모드가 아니다: {mode!r} (가능: {sorted(FE_MODES)})")
    if mode == "causal":
        return CausalFE(fs)
    if mode == "zerophase":
        return BlockZeroPhaseFE(fs, hop_s=hop_s)
    return MedianBaselineFE(fs, hop_s=hop_s)
