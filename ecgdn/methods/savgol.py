"""M02 Savitzky-Golay — 국소 다항식 적합.

이동평균과 달리 국소 다항식을 맞추므로 peak 형태를 비교적 잘 보존한다.
window_length 는 홀수여야 하고, ms 단위로 지정해 fs 에 무관하게 만든다.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from ..registry import register_method
from .base import BaseDenoiser
from .frontend import FrontEnd


class SavGol(BaseDenoiser):
    def __init__(self, win_ms: float = 40.0, polyorder: int = 3,
                 use_frontend: bool = True, name: str = "M02"):
        self.name = name
        self.win_ms = float(win_ms)
        self.polyorder = int(polyorder)
        self.fe = FrontEnd() if use_frontend else None

    def _run(self, y: np.ndarray, fs: float, ctx: dict[str, Any]) -> np.ndarray:
        x = self.fe(y, fs) if self.fe is not None else y
        w = int(round(self.win_ms * 1e-3 * fs))
        w = max(self.polyorder + 2, w)
        if w % 2 == 0:
            w += 1
        if w >= x.size:
            return x
        return savgol_filter(x, w, self.polyorder, mode="interp")


@register_method("M02", family="classical", label="Savitzky-Golay (40 ms, order 3)")
def _s1(**kw):
    return SavGol(**kw)
