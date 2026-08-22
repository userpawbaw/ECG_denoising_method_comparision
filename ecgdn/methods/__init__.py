"""기법 패키지. import 시 모든 방법이 레지스트리에 등록된다."""
from __future__ import annotations

from .base import BaseDenoiser, Denoiser  # noqa: F401
from . import identity, frontend, bandpass, savgol, wavelet, kalman_sameni  # noqa: F401

try:                                       # oracle 은 선택적
    from . import oracle                   # noqa: F401
except ImportError:                        # pragma: no cover
    pass

from ..registry import available, build, meta  # noqa: F401

__all__ = ["BaseDenoiser", "Denoiser", "available", "build", "meta"]
