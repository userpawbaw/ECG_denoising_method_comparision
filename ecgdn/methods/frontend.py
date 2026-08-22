"""공통 front-end (docs/01_design.md 2.1).

**모든 방법에 동일하게 적용한다.** 특정 방법 앞에만 붙이면 비교가 불공정해진다.
실험은 fe=on / fe=off 두 조건을 모두 돌린다.

  FE1 zero-phase Butterworth HPF (0.5 Hz)   : baseline wander
  FE2 zero-phase Butterworth LPF (100 Hz)   : 대역 제한
  FE3 IIR notch (60/120 Hz)                 : **PSD 로 PLI 존재를 판정한 뒤에만**

zero-phase(filtfilt) 를 쓰는 이유: 위상 왜곡이 없어 R-peak timing 이 밀리지 않는다.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal as sps

from ..config import DEFAULT_FE, FrontEndCfg
from ..eval.spectral import pli_ratio
from ..registry import register_method
from .base import BaseDenoiser

__all__ = ["FrontEnd", "apply_frontend"]

def _safe_filtfilt(sos: np.ndarray, x: np.ndarray, ring_s: float, fs: float) -> np.ndarray:
    """충분한 padlen 을 강제한 zero-phase 필터.

    **중요**: scipy 의 기본 padlen 은 3*(2*n_sections+1) (4차 SOS 에서 15 샘플)이다.
    0.5 Hz HPF 는 수 초 동안 울리므로 15 샘플 패딩으로는 경계 트랜지언트가 남아
    깨끗한 신호에도 큰 왜곡을 만든다. (실측: distortion floor 8.9 dB)
    필터의 링잉 시간(ring_s)에 맞춰 padlen 을 직접 지정한다.
    """
    need = int(round(ring_s * fs))
    padlen = min(max(need, 3 * (2 * sos.shape[0] + 1)), x.size - 1)
    if padlen <= 0 or x.size <= 3:
        return x.copy()
    return sps.sosfiltfilt(sos, x, padtype="odd", padlen=padlen)


class FrontEnd(BaseDenoiser):
    name = "M_FE"

    def __init__(self, cfg: FrontEndCfg = DEFAULT_FE):
        self.cfg = cfg

    def _run(self, y: np.ndarray, fs: float, ctx: dict[str, Any]) -> np.ndarray:
        c = self.cfg
        x = y
        nyq = fs / 2.0

        if c.hp_hz and c.hp_hz > 0:
            sos = sps.butter(c.order, c.hp_hz / nyq, btype="highpass", output="sos")
            # 링잉 시간 ~ order / cutoff. 여유 있게 잡는다.
            x = _safe_filtfilt(sos, x, ring_s=8.0 * c.order / c.hp_hz, fs=fs)
        if c.lp_hz and c.lp_hz < nyq * 0.98:
            sos = sps.butter(c.order, c.lp_hz / nyq, btype="lowpass", output="sos")
            x = _safe_filtfilt(sos, x, ring_s=8.0 * c.order / c.lp_hz, fs=fs)

        for f0 in c.notch_hz:
            if f0 >= nyq * 0.98:
                continue
            if c.auto_notch and pli_ratio(x, fs, f0) < c.pli_ratio_thresh:
                continue                       # PLI 가 없으면 걸지 않는다
            b, a = sps.iirnotch(f0, c.notch_q, fs)
            sos = sps.tf2sos(b, a)
            # notch 의 링잉 시간 ~ Q / (pi * f0)
            x = _safe_filtfilt(sos, x, ring_s=8.0 * c.notch_q / (np.pi * f0), fs=fs)
        return x


def apply_frontend(y: np.ndarray, fs: float, cfg: FrontEndCfg = DEFAULT_FE) -> np.ndarray:
    return FrontEnd(cfg)(y, fs)


@register_method("M_FE", family="frontend", label="Front-end only (HPF+LPF+auto notch)")
def _build_fe(**kw):
    return FrontEnd(FrontEndCfg(**kw) if kw else DEFAULT_FE)
