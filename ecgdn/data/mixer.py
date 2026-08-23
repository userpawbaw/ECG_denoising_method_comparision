"""SNR 통제 합성 (docs/01_design.md 3.5)."""
from __future__ import annotations

import numpy as np

from ..utils import power

__all__ = ["mix_at_snr", "measure_snr", "scale_noise_to_snr"]


def measure_snr(x: np.ndarray, n: np.ndarray) -> float:
    """SNR = 10 log10( P(x) / P(n) ). 평균 제거 파워 사용."""
    pn = power(n)
    if pn <= 0:
        return float("inf")
    return float(10.0 * np.log10(power(x) / pn))


def scale_noise_to_snr(x: np.ndarray, n: np.ndarray, snr_db: float) -> np.ndarray:
    """목표 SNR 이 되도록 잡음을 스케일링."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = np.asarray(n, dtype=np.float64).ravel()
    if x.size != n.size:
        raise ValueError(f"length mismatch: {x.size} vs {n.size}")
    pn = power(n)
    if pn <= 0:
        raise ValueError("noise has zero power")
    g = np.sqrt(power(x) / (pn * 10.0 ** (snr_db / 10.0)))
    return g * (n - n.mean())


def mix_at_snr(x: np.ndarray, n: np.ndarray, snr_db: float,
               check: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """y = x + n'  (n' 은 목표 SNR 로 스케일된 잡음).

    Returns
    -------
    (y, n_scaled, actual_snr_db)
    """
    ns = scale_noise_to_snr(x, n, snr_db)
    y = np.asarray(x, dtype=np.float64).ravel() + ns
    actual = measure_snr(x, ns)
    if check and not np.isclose(actual, snr_db, atol=1e-6):
        raise AssertionError(f"SNR mismatch: target={snr_db}, actual={actual}")
    return y, ns, actual
