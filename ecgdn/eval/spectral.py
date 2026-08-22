"""주파수 영역 지표 (docs/02_procedure.md STEP 08).

주의 (docs/00_review.md 참조):
    "ECG = 0.5~40 Hz, noise = 그 위" 처럼 대역을 임의로 잘라 판정하지 않는다.
    QRS 에는 상당한 고주파 성분이 있다. 항상 **reference PSD 와 비교**한다.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

from ..config import BAND_EDGES, PSD_BAND

__all__ = ["welch_psd", "psd_logdist", "band_power", "band_power_err",
           "has_pli", "pli_ratio", "metrics_spectral"]


def welch_psd(x: np.ndarray, fs: float, nperseg: int | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).ravel()
    n = nperseg or min(1024, max(64, x.size // 8 * 2))
    n = min(n, x.size)
    f, p = sps.welch(x - x.mean(), fs=fs, nperseg=n, noverlap=n // 2, detrend="constant")
    return f, p


def psd_logdist(x: np.ndarray, xhat: np.ndarray, fs: float,
                band: tuple[float, float] = PSD_BAND, floor_db: float = -120.0) -> float:
    """log-PSD 평균 절대차 [dB]. 스펙트럼 형태를 얼마나 보존했는가."""
    f, px = welch_psd(x, fs)
    _, ph = welch_psd(xhat, fs)
    m = (f >= band[0]) & (f <= min(band[1], f[-1]))
    if not np.any(m):
        return float("nan")
    a = np.maximum(10 * np.log10(np.maximum(px[m], 1e-30)), floor_db)
    b = np.maximum(10 * np.log10(np.maximum(ph[m], 1e-30)), floor_db)
    return float(np.mean(np.abs(a - b)))


def band_power(x: np.ndarray, fs: float,
               bands: tuple[tuple[float, float], ...] = BAND_EDGES) -> np.ndarray:
    f, p = welch_psd(x, fs)
    out = []
    for lo, hi in bands:
        m = (f >= lo) & (f <= min(hi, f[-1]))
        out.append(float(np.trapezoid(p[m], f[m])) if np.any(m) else 0.0)
    return np.asarray(out)


def band_power_err(x: np.ndarray, xhat: np.ndarray, fs: float,
                   bands: tuple[tuple[float, float], ...] = BAND_EDGES) -> np.ndarray:
    """대역별 상대 파워 오차. 어떤 대역을 과하게 잘랐는지 보여준다."""
    a = band_power(x, fs, bands)
    b = band_power(xhat, fs, bands)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(b - a) / np.maximum(a, 1e-30)


def pli_ratio(x: np.ndarray, fs: float, f0: float = 60.0, half_bw: float = 1.0,
              bg: tuple[tuple[float, float], ...] = ((55.0, 58.0), (62.0, 65.0))) -> float:
    """f0 주변 파워 / 인접 배경대역 중앙 파워밀도. PLI 존재 판정용."""
    f, p = welch_psd(x, fs, nperseg=min(2048, len(np.asarray(x).ravel())))
    if f[-1] < f0 + half_bw:
        return 0.0
    m = np.abs(f - f0) <= half_bw
    if not np.any(m):
        return 0.0
    bgm = np.zeros_like(f, dtype=bool)
    for lo, hi in bg:
        bgm |= (f >= lo) & (f <= hi)
    if not np.any(bgm):
        return 0.0
    base = float(np.median(p[bgm]))
    return float(np.max(p[m]) / max(base, 1e-30))


def has_pli(x: np.ndarray, fs: float, f0: float = 60.0, thresh: float = 10.0) -> bool:
    """PSD 로 PLI 존재를 판정 (docs/01_design.md 2.1). 무조건 notch 를 걸지 않기 위함."""
    return pli_ratio(x, fs, f0) >= thresh


def metrics_spectral(x: np.ndarray, xhat: np.ndarray, fs: float) -> dict[str, float]:
    out = {"psd_logdist": psd_logdist(x, xhat, fs)}
    err = band_power_err(x, xhat, fs)
    for (lo, hi), e in zip(BAND_EDGES, err):
        out[f"bp_err_{lo:g}_{hi:g}"] = float(e)
    out["pli_ratio_hat"] = pli_ratio(xhat, fs)
    return out
