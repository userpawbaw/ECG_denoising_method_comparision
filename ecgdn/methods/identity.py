"""M00 Identity — 하한선(lower bound). 파이프라인 정합성 확인에도 쓴다."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..registry import register_method
from .base import BaseDenoiser


class Identity(BaseDenoiser):
    name = "M00"

    def _run(self, y: np.ndarray, fs: float, ctx: dict[str, Any]) -> np.ndarray:
        return y.copy()


@register_method("M00", family="baseline", label="Identity (no-op)")
def _build():
    return Identity()
