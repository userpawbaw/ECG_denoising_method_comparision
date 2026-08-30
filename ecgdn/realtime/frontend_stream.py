"""인과(causal) front-end — **영위상 필터는 실시간으로 못 돌린다.**

왜 필요한가
----------
오프라인 front-end 는 `sosfiltfilt`(영위상)다. 위상 왜곡이 없어 R-peak 타이밍이
밀리지 않기 때문인데, 대가로 **미래를 본다.** 그리고 그 양이 작지 않다 —
`FrontEnd._run` 은 0.5 Hz HPF 의 링잉 시간을 `8 × order / f_c` 로 잡는다:

    8 × 4 / 0.5 Hz = **64 초**

즉 한 샘플을 영위상으로 거르려면 앞뒤로 수십 초가 있어야 한다. 시연 지연
목표는 100 ms 다. **원리적으로 양립하지 않는다.**

실측이 그것을 그대로 보였다 `[측정]` — 링버퍼마다 `filtfilt` 를 다시 거는
방식으로 짰더니 오프라인 대비 **−15.0 dB** 였고, 같은 신호에 front-end 를
미리 걸고 방법의 FE 를 끄자 **−0.02 dB**(상관 1.0000)로 붙었다. 스트리밍
배관은 처음부터 정확했고 틀린 것은 **필터를 어디서 도느냐** 하나였다.

그래서 이 모듈
-------------
같은 필터 **설계**를 쓰되 `sosfilt` + 상태 유지로 **한 방향**만 돈다.
위상 왜곡이 생기고, 그 대가는 `scripts/verify_stream_processor.py` 가 잰다 —
"실시간에서는 이만큼 다르다" 를 숫자로 두는 것이 이 모듈의 목적이다.

notch 를 걸지 말지는 **warm-up 구간에서 한 번 정하고 고정한다.** 매 블록마다
다시 판정하면 잡음이 들락날락할 때 필터가 켜졌다 꺼졌다 하면서 그 자체가
인공물이 된다.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

from ..config import DEFAULT_FE, FrontEndCfg
from ..eval.spectral import pli_ratio

__all__ = ["StreamingFrontEnd"]


class StreamingFrontEnd:
    """상태를 유지하는 인과 front-end. `push(block) -> filtered block`."""

    def __init__(self, fs: float, cfg: FrontEndCfg = DEFAULT_FE,
                 decide_after_s: float = 4.0):
        self.fs = float(fs)
        self.cfg = cfg
        self.decide_after = int(round(decide_after_s * self.fs))
        self._probe: list[np.ndarray] = []      # notch 판정용 warm-up 표본
        self._decided = False
        self._stages: list[tuple[np.ndarray, np.ndarray]] = []   # (sos, zi)
        self._build_fixed()

    # --------------------------------------------------------------- 설계
    def _sos(self, kind: str, f0: float):
        nyq = self.fs / 2.0
        if kind == "hp":
            return sps.butter(self.cfg.order, f0 / nyq, btype="highpass", output="sos")
        if kind == "lp":
            return sps.butter(self.cfg.order, f0 / nyq, btype="lowpass", output="sos")
        b, a = sps.iirnotch(f0, self.cfg.notch_q, self.fs)
        return sps.tf2sos(b, a)

    def _add(self, sos: np.ndarray) -> None:
        self._stages.append((sos, sps.sosfilt_zi(sos)))

    def _build_fixed(self) -> None:
        """HPF·LPF 는 조건과 무관하다 — 바로 만든다."""
        c, nyq = self.cfg, self.fs / 2.0
        if c.hp_hz and c.hp_hz > 0:
            self._add(self._sos("hp", c.hp_hz))
        if c.lp_hz and c.lp_hz < nyq * 0.98:
            self._add(self._sos("lp", c.lp_hz))

    def _decide_notch(self, warm: np.ndarray) -> None:
        """PLI 가 있는지 **한 번** 판정하고 그대로 간다."""
        nyq = self.fs / 2.0
        for f0 in self.cfg.notch_hz:
            if f0 >= nyq * 0.98:
                continue
            if self.cfg.auto_notch and pli_ratio(warm, self.fs, f0) < self.cfg.pli_ratio_thresh:
                continue
            self._add(self._sos("notch", f0))
        self._decided = True

    # --------------------------------------------------------------- 입력
    def push(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float64).ravel()
        if not block.size:
            return block
        if not self._decided:
            self._probe.append(block)
            got = int(sum(p.size for p in self._probe))
            if got >= self.decide_after:
                self._decide_notch(np.concatenate(self._probe))
                self._probe = []
        out = block
        for i, (sos, zi) in enumerate(self._stages):
            # 첫 호출의 zi 는 **첫 표본 값으로 정규화**한다. 0 에서 시작하면
            # 신호의 DC 만큼이 계단 입력이 되어 필터가 수 초간 울린다.
            if not getattr(self, f"_primed{i}", False):
                zi = zi * float(out[0])
                setattr(self, f"_primed{i}", True)
            out, zi = sps.sosfilt(sos, out, zi=zi)
            self._stages[i] = (sos, zi)
        return out

    def reset(self) -> None:
        self._probe, self._decided, self._stages = [], False, []
        for i in range(16):
            if hasattr(self, f"_primed{i}"):
                delattr(self, f"_primed{i}")
        self._build_fixed()
