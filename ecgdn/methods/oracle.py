"""성능 상한(bound) — 실제로 쓸 수 없지만 '어디까지 갈 수 있는가' 를 준다.

docs/00_review.md C-2. 이 상한이 있어야 다음 질문에 정량적으로 답할 수 있다.

  * `B01` (oracle wavelet) : **wavelet 계열의 상한.**
    M04 가 B01 에 가까우면 -> threshold 튜닝은 끝났다. 더 얻으려면 표현(representation)
    자체를 바꿔야 한다  =>  hybrid / 딥러닝을 도입할 정량적 근거가 된다.
  * `B02` (oracle Wiener) : **선형 시불변 필터의 상한.**
    어떤 방법이 B02 를 넘으면 -> 비선형/시변 처리가 실제로 필요하다는 증거.

두 방법 모두 clean 신호를 참조하므로 이름이 반드시 'oracle_' 로 시작한다
(BaseDenoiser 가 강제 검사한다).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pywt

from ..config import DEFAULT_SWT, SWTCfg
from ..registry import register_method
from .base import BaseDenoiser
from .frontend import FrontEnd
from .wavelet import _pad_to_multiple

__all__ = ["OracleWaveletDenoiser", "OracleWienerDenoiser"]


class OracleWaveletDenoiser(BaseDenoiser):
    """B01 — oracle diagonal estimator.

    각 SWT 계수를 살릴지 죽일지를 clean 계수의 크기로 결정한다.

        keep_j[n] = 1  if  s_clean_j[n]^2 > sigma_j^2   else 0

    이는 대각 추정기(계수를 개별적으로 keep/kill 하는 방식) 중 최적에 가까운 성능이며,
    threshold 설계로 도달 가능한 상한 역할을 한다.
    """

    name = "oracle_B01"
    needs_clean = True

    def __init__(self, cfg: SWTCfg = DEFAULT_SWT, use_frontend: bool = True):
        self.cfg = cfg
        # M04 와 동일한 front-end 를 준다. 그래야 'wavelet thresholding 으로 얼마나 더
        # 갈 수 있는가' 라는 질문에 대한 공정한 상한이 된다.
        # (A5 는 thresholding 하지 않으므로 front-end 없이는 baseline 계열 잡음을
        #  전혀 제거하지 못해 상한 역할을 못 한다)
        self.fe = FrontEnd() if use_frontend else None

    def _swt(self, x: np.ndarray) -> list[np.ndarray]:
        return list(pywt.swt(x, self.cfg.wavelet, level=self.cfg.level,
                             trim_approx=True, norm=False))

    def _run(self, y: np.ndarray, fs: float, ctx: dict[str, Any]) -> np.ndarray:
        x_clean = np.asarray(ctx["x_clean"], dtype=np.float64).ravel()
        if x_clean.size != y.size:
            raise ValueError("x_clean length mismatch")
        if self.fe is not None:
            y = self.fe(y, fs)
            x_clean = self.fe(x_clean, fs)     # 잡음 추정도 같은 필터를 통과한 기준으로
        n_noise = y - x_clean

        m = 2 ** self.cfg.level
        yp, pl, _ = _pad_to_multiple(y, m)
        xp, _, _ = _pad_to_multiple(x_clean, m)
        np_, _, _ = _pad_to_multiple(n_noise, m)

        cy, cx, cn = self._swt(yp), self._swt(xp), self._swt(np_)
        out = [cy[0]]                                   # 근사계수는 유지
        for j in range(1, len(cy)):
            sigma2 = float(np.var(cn[j]))
            keep = (cx[j] ** 2) > sigma2
            out.append(np.where(keep, cy[j], 0.0))
        xr = np.asarray(pywt.iswt(out, self.cfg.wavelet, norm=False))
        return xr[pl:pl + y.size]


class OracleWienerDenoiser(BaseDenoiser):
    """B02 — 참 PSD 를 아는 주파수영역 Wiener 필터.

        H(f) = Sx(f) / (Sx(f) + Sn(f))

    선형 시불변 필터가 낼 수 있는 최선(정상 신호/잡음 가정)이다.
    PSD 는 Welch 로 추정한 뒤 FFT 격자에 보간해 적용한다.
    """

    name = "oracle_B02"
    needs_clean = True

    def __init__(self, nperseg: int = 1024, smooth: int = 3, use_frontend: bool = True):
        self.nperseg = int(nperseg)
        self.smooth = int(smooth)
        self.fe = FrontEnd() if use_frontend else None

    def _run(self, y: np.ndarray, fs: float, ctx: dict[str, Any]) -> np.ndarray:
        from scipy.signal import welch

        x_clean = np.asarray(ctx["x_clean"], dtype=np.float64).ravel()
        if self.fe is not None:
            y = self.fe(y, fs)
            x_clean = self.fe(x_clean, fs)
        n_noise = y - x_clean
        nps = min(self.nperseg, y.size)

        f, sx = welch(x_clean - x_clean.mean(), fs=fs, nperseg=nps)
        _, sn = welch(n_noise - n_noise.mean(), fs=fs, nperseg=nps)
        if self.smooth > 1:
            k = np.ones(self.smooth) / self.smooth
            sx = np.convolve(sx, k, mode="same")
            sn = np.convolve(sn, k, mode="same")

        n = y.size
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        sx_i = np.interp(freqs, f, sx)
        sn_i = np.interp(freqs, f, sn)
        h = sx_i / np.maximum(sx_i + sn_i, 1e-30)

        mean = y.mean()
        Y = np.fft.rfft(y - mean)
        return np.fft.irfft(Y * h, n=n) + mean


@register_method("B01", family="oracle",
                 label="Oracle wavelet threshold (wavelet thresholding 계열의 상한)",
                 needs_clean=True)
def _b01(use_frontend: bool = True, **kw):
    return OracleWaveletDenoiser(SWTCfg(**kw) if kw else DEFAULT_SWT,
                                 use_frontend=use_frontend)


@register_method("B02", family="oracle",
                 label="Oracle Wiener (LTI 계열의 상한)",
                 needs_clean=True)
def _b02(use_frontend: bool = True, **kw):
    return OracleWienerDenoiser(use_frontend=use_frontend, **kw)
