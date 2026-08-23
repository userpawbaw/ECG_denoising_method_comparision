"""M03 DWT / M04 SWT thresholding (docs/00_review.md B-1, STEP 11).

본 프로젝트의 SWT 는 교과서적 universal threshold 를 그대로 쓰지 않는다.
세 가지를 추가한다.

  1) **잡음 sigma 는 최고주파 band 하나에서만 추정한다 (level 별 MAD 금지)**
     fs=250 Hz 에서 각 detail band 의 대략적 통과대역과 실측 band SNR(입력 10 dB, AWGN):
        D1 62.5-125 Hz  (-13.2 dB)   D2 31-62.5 (+0.5)   D3 15.6-31 (+12.3)
        D4 7.8-15.6 (+17.5)          D5 3.9-7.8 (+18.5)  A5 0-3.9 (+19.5)
     즉 **D3 이상은 ECG 가 지배적**이라 그 band 의 MAD 는 잡음이 아니라 ECG 를 잰다
     (D4 에서 2 배, D5 에서 5 배 과대추정). 그 sigma 로 threshold 를 만들면 QRS 를 잘라낸다.
     SWT(norm=False) 는 백색잡음의 계수 분산을 level 에 무관하게 보존하므로,
     **D1(또는 D2) 하나에서 추정한 sigma 를 전 level 에 적용**하는 것이 맞다.
     level 별 가중치 k 는 그 위에서 미세조정용으로만 쓴다.

  2) **QRS 구간 보호**
     R-peak 주변 +-protect_ms 에서 threshold 를 rho 배로 낮춘다.
     '고주파 = 잡음' 가정이 깨지는 유일한 구간이 QRS 라서, 여기만 예외 처리하는 것이
     비용 대비 효과가 가장 크다.

  3) **garrote 를 기본 threshold 함수로**
     soft 는 살아남은 계수에서도 항상 lambda 만큼을 빼므로 진폭이 체계적으로 줄어든다
     (docs/00_review.md A-1 의 gain bias). garrote 는 큰 계수에서 그 편향이 사라진다.

구현 주의
--------
  * `pywt.swt` 는 신호 길이가 2^level 의 배수여야 한다 -> 반드시 padding.
  * `norm=False` 를 쓴다. SWT(비추출)에서 백색잡음의 계수 분산이 level 마다 1 로 유지되어
    MAD 기반 sigma 추정과 universal threshold 가 그대로 성립한다.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pywt

from ..config import DEFAULT_SWT, SWTCfg
from ..registry import register_method
from .base import BaseDenoiser
from .frontend import FrontEnd

__all__ = ["mad_sigma", "threshold", "qrs_protection_mask", "SWTDenoiser", "DWTDenoiser"]


def mad_sigma(d: np.ndarray) -> float:
    """MAD 기반 잡음 표준편차 추정. 0.6745 = 표준정규의 MAD."""
    d = np.asarray(d, dtype=np.float64).ravel()
    if d.size == 0:
        return 0.0
    return float(np.median(np.abs(d - np.median(d))) / 0.6745)


def threshold(d: np.ndarray, lam: np.ndarray | float, mode: str = "garrote") -> np.ndarray:
    """lam 은 스칼라 또는 d 와 같은 길이(시변 threshold)."""
    d = np.asarray(d, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    a = np.abs(d)
    if mode == "hard":
        return np.where(a > lam, d, 0.0)
    if mode == "soft":
        return np.sign(d) * np.maximum(a - lam, 0.0)
    if mode == "garrote":
        with np.errstate(divide="ignore", invalid="ignore"):
            g = 1.0 - (lam ** 2) / np.maximum(d ** 2, 1e-300)
        return np.where(a > lam, d * g, 0.0)
    raise ValueError(f"unknown threshold mode: {mode}")


def qrs_protection_mask(n: int, r_peaks: np.ndarray, fs: float,
                        protect_ms: float) -> np.ndarray:
    """R-peak 중심 raised-cosine 창. 값 1 = 완전 보호, 0 = 보호 없음."""
    g = np.zeros(n, dtype=np.float64)
    if r_peaks is None or len(r_peaks) == 0:
        return g
    w = max(1, int(round(protect_ms * 1e-3 * fs)))
    k = np.arange(-w, w + 1)
    bump = 0.5 * (1.0 + np.cos(np.pi * k / w))          # 1 at center, 0 at edges
    for p in np.asarray(r_peaks, dtype=int).ravel():
        i0, i1 = p - w, p + w + 1
        a, b = max(0, i0), min(n, i1)
        if b <= a:
            continue
        g[a:b] = np.maximum(g[a:b], bump[a - i0:b - i0])
    return g


def _pad_to_multiple(x: np.ndarray, m: int) -> tuple[np.ndarray, int, int]:
    n = x.size
    total = int(np.ceil(n / m) * m)
    extra = total - n
    left = extra // 2
    right = extra - left
    mode = "reflect" if n > max(left, right) + 1 else "edge"
    return np.pad(x, (left, right), mode=mode), left, right


class _WaveletBase(BaseDenoiser):
    def __init__(self, cfg: SWTCfg = DEFAULT_SWT, use_frontend: bool = True,
                 name: str = "M04"):
        self.cfg = cfg
        self.name = name
        self.fe = FrontEnd() if use_frontend else None

    # --- 하위 클래스가 구현
    def _decompose(self, x: np.ndarray) -> list[np.ndarray]: ...
    def _reconstruct(self, coeffs: list[np.ndarray], n: int) -> np.ndarray: ...

    def _sigma(self, coeffs: list[np.ndarray], j: int) -> float:
        """threshold 에 쓸 잡음 sigma. src 에 따라 전역/level 별."""
        src = getattr(self.cfg, "sigma_source", "d1")
        if src == "level":
            return mad_sigma(coeffs[j])
        if src == "d1":
            return mad_sigma(coeffs[-1])
        if src == "d2":
            return mad_sigma(coeffs[-2]) if len(coeffs) >= 3 else mad_sigma(coeffs[-1])
        if src == "min12":
            a = mad_sigma(coeffs[-1])
            b = mad_sigma(coeffs[-2]) if len(coeffs) >= 3 else a
            return min(a, b)
        raise ValueError(f"unknown sigma_source: {src}")

    def _levels_k(self) -> np.ndarray:
        """coeffs 순서([cA_L, cD_L, ..., cD_1])에 맞춘 k 벡터."""
        k = np.asarray(self.cfg.k, dtype=np.float64)   # (k_D1, ..., k_DL)
        L = self.cfg.level
        if k.size < L:
            k = np.concatenate([k, np.full(L - k.size, k[-1] if k.size else 1.0)])
        return k[:L][::-1]                              # -> (k_DL, ..., k_D1)

    def _run(self, y: np.ndarray, fs: float, ctx: dict[str, Any]) -> np.ndarray:
        c = self.cfg
        x = self.fe(y, fs) if self.fe is not None else y

        r_peaks = ctx.get("r_peaks")
        if c.protect_qrs and r_peaks is None:
            from ..eval.rpeak import detect_rpeaks
            r_peaks = detect_rpeaks(x, fs)

        xp, pl, pr = _pad_to_multiple(x, 2 ** c.level)
        coeffs = self._decompose(xp)
        n_full = xp.size

        gmask_full = (qrs_protection_mask(n_full, np.asarray(r_peaks, dtype=int) + pl, fs,
                                          c.protect_ms)
                      if (c.protect_qrs and r_peaks is not None and len(r_peaks)) else None)

        kvec = self._levels_k()
        out = [coeffs[0] if not c.threshold_approx else coeffs[0]]
        for j, d in enumerate(coeffs[1:], start=1):
            sigma = self._sigma(coeffs, j)
            lam = kvec[j - 1] * sigma * np.sqrt(2.0 * np.log(max(d.size, 2)))
            if gmask_full is not None:
                g = gmask_full
                if g.size != d.size:                     # DWT: level 별 길이가 다르다
                    g = np.interp(np.linspace(0, 1, d.size),
                                  np.linspace(0, 1, g.size), g)
                lam = lam * (1.0 - (1.0 - c.protect_rho) * g)
            out.append(threshold(d, lam, c.mode))

        xr = self._reconstruct(out, n_full)
        return xr[pl:pl + x.size]


class SWTDenoiser(_WaveletBase):
    """M04 — Stationary(undecimated) Wavelet Transform thresholding."""

    def _decompose(self, x):
        return list(pywt.swt(x, self.cfg.wavelet, level=self.cfg.level,
                             trim_approx=True, norm=False))

    def _reconstruct(self, coeffs, n):
        return np.asarray(pywt.iswt(list(coeffs), self.cfg.wavelet, norm=False))[:n]


class DWTDenoiser(_WaveletBase):
    """M03 — 일반 DWT thresholding (SWT 의 대조군).

    downsampling 이 있어 translation-equivariance 를 잃는다. 그 차이를 보기 위한 대조군.
    """

    def _decompose(self, x):
        return list(pywt.wavedec(x, self.cfg.wavelet, level=self.cfg.level,
                                 mode="periodization"))

    def _reconstruct(self, coeffs, n):
        return np.asarray(pywt.waverec(list(coeffs), self.cfg.wavelet,
                                       mode="periodization"))[:n]


# --------------------------------------------------------------------- 등록
@register_method("M03", family="timefreq", label="DWT soft threshold (sym4, L5)")
def _m03(use_frontend: bool = True, **kw):
    cfg = SWTCfg(mode="soft", protect_qrs=False, **kw)
    return DWTDenoiser(cfg, use_frontend=use_frontend, name="M03")


@register_method("M04", family="timefreq",
                 label="SWT adaptive threshold (level-k + QRS protect, garrote)")
def _m04(use_frontend: bool = True, **kw):
    return SWTDenoiser(SWTCfg(**kw) if kw else DEFAULT_SWT,
                       use_frontend=use_frontend, name="M04")


@register_method("M04np", family="timefreq",
                 label="SWT adaptive threshold, QRS protection OFF (ablation)")
def _m04np(use_frontend: bool = True, **kw):
    """M04 에서 **QRS 보호만** 끈 조건.

    M04s 는 sigma 출처/threshold 함수/QRS 보호를 동시에 바꿔 교란되어 있다.
    'ECG-aware DSP 의 이득' 을 주장하려면 이 한 축만 분리해야 한다.
    """
    base = SWTCfg(**kw) if kw else DEFAULT_SWT
    cfg = SWTCfg(**{**base.__dict__, "protect_qrs": False})
    return SWTDenoiser(cfg, use_frontend=use_frontend, name="M04np")


@register_method("M04s", family="timefreq", label="SWT soft threshold (uniform k, no protect)")
def _m04s(use_frontend: bool = True, **kw):
    cfg = SWTCfg(mode="soft", k=(1.0,) * 5, protect_qrs=False, **kw)
    return SWTDenoiser(cfg, use_frontend=use_frontend, name="M04s")
