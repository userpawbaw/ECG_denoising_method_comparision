"""M01 Bandpass + Notch — 가장 단순한 현실적 baseline.

두 대역 설정을 제공한다.
  monitoring : 0.5 - 40 Hz   (임상 monitoring 관행, 잡음에 강하지만 QRS 고주파를 자른다)
  diagnostic : 0.5 - 100 Hz  (공통 front-end 와 동일; 형태 보존 우선)
설정에 따라 '잡음 제거 vs 형태 보존' 의 trade-off 가 그대로 드러나므로 둘 다 평가한다.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..config import DEFAULT_FE, FrontEndCfg
from ..registry import register_method
from .base import BaseDenoiser
from .frontend import FrontEnd


class BandpassNotch(BaseDenoiser):
    def __init__(self, lp_hz: float = 40.0, hp_hz: float = 0.5,
                 cfg: FrontEndCfg = DEFAULT_FE, name: str = "M01"):
        self.name = name
        self.fe = FrontEnd(FrontEndCfg(
            hp_hz=hp_hz, lp_hz=lp_hz, order=cfg.order, notch_hz=cfg.notch_hz,
            notch_q=cfg.notch_q, auto_notch=cfg.auto_notch,
            pli_ratio_thresh=cfg.pli_ratio_thresh))

    def _run(self, y: np.ndarray, fs: float, ctx: dict[str, Any]) -> np.ndarray:
        return self.fe(y, fs)


@register_method("M01", family="classical", label="Bandpass 0.5-40 Hz + auto notch")
def _b1(use_frontend: bool = True, **kw):
    # M01 은 front-end 그 자체다. use_frontend 는 무시한다 (끄면 identity 가 된다).
    return BandpassNotch(lp_hz=40.0, name="M01", **kw)


@register_method("M01d", family="classical", label="Bandpass 0.5-100 Hz + auto notch")
def _b2(use_frontend: bool = True, **kw):
    return BandpassNotch(lp_hz=100.0, name="M01d", **kw)
