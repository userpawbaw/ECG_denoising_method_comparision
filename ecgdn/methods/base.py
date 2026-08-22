"""Denoiser 계약 (docs/00_review.md B-5).

모든 기법은 이 하나의 시그니처를 따른다. 그래야 실험 스크립트가 방법을 몰라도 된다.

    x_hat = denoiser(y, fs, ctx=None)

규칙
----
  * 반환 길이 == 입력 길이, 같은 물리 스케일(mV).
  * `ctx` 에는 선택적 부가정보만 넣는다 (예: 'r_peaks').
  * **`ctx` 에 clean 신호를 넣는 것은 oracle 계열에만 허용**되며,
    그 경우 `needs_clean=True` 로 등록하고 name 이 'oracle_' 로 시작해야 한다.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Denoiser(Protocol):
    name: str

    def __call__(self, y: np.ndarray, fs: float,
                 ctx: dict[str, Any] | None = None) -> np.ndarray: ...


class BaseDenoiser:
    """공통 구현: 길이 검증 + float64 변환."""

    name: str = "base"
    needs_clean: bool = False

    def __call__(self, y: np.ndarray, fs: float,
                 ctx: dict[str, Any] | None = None) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel()
        if y.size == 0:
            raise ValueError("empty input")
        if self.needs_clean:
            if not self.name.startswith("oracle_"):
                raise RuntimeError(
                    f"{type(self).__name__} uses clean signal but name '{self.name}' "
                    "does not start with 'oracle_'")
            if ctx is None or "x_clean" not in ctx:
                raise ValueError(f"{self.name} requires ctx['x_clean']")
        out = self._run(y, float(fs), ctx or {})
        out = np.asarray(out, dtype=np.float64).ravel()
        if out.size != y.size:
            raise RuntimeError(f"{self.name}: length changed {y.size} -> {out.size}")
        if not np.all(np.isfinite(out)):
            raise RuntimeError(f"{self.name}: non-finite output")
        return out

    def _run(self, y: np.ndarray, fs: float, ctx: dict[str, Any]) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
