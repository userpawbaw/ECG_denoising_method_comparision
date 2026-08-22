"""미분 가능한 SWT / ISWT (docs/00_review.md B-2, STEP 21).

**왜 필요한가**
    wavelet 을 '전처리' 가 아니라 '신경망의 표현공간' 으로 쓰려면, 역변환까지 학습
    그래프 안에 있어야 subband 도메인 손실과 시간 도메인 손실을 동시에 걸 수 있다.
    SWT/ISWT 는 전부 선형 FIR 연산이므로 conv1d 로 그대로 구현되고 자동미분이 통한다.

**정렬 규약** (pywt 와 수치적으로 동일하도록 실측으로 유도)
    level j: dilation d = 2^(j-1),  shift s = -(L//2 - 1) * d      (L = 필터 길이)
    분해:  a[n] = sum_k lo_rev[k] * x[(n + s + d k) mod N]
           d[n] = sum_k hi_rev[k] * x[(n + s + d k) mod N]
    복원:  x[n] = 0.5 * ( sum_k lo_rev[k] a[(n - s - d k) mod N]
                        + sum_k hi_rev[k] d[(n - s - d k) mod N] )
    (직교 wavelet 에서 프레임 연산자가 level 당 2*I 이므로 0.5 배가 붙는다)

계수 순서는 pywt 의 `trim_approx=True` 와 같다: [cA_L, cD_L, ..., cD_1].
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TorchSWT", "TorchISWT", "swt_filters", "SWT_COEFF_NAMES"]


def SWT_COEFF_NAMES(level: int) -> list[str]:
    return [f"A{level}"] + [f"D{j}" for j in range(level, 0, -1)]


def swt_filters(wavelet: str) -> tuple[np.ndarray, np.ndarray]:
    """(lo_rev, hi_rev) — pywt 분해필터의 역순."""
    import pywt

    w = pywt.Wavelet(wavelet)
    return (np.asarray(w.dec_lo, dtype=np.float64)[::-1].copy(),
            np.asarray(w.dec_hi, dtype=np.float64)[::-1].copy())


def _circ_corr(x: torch.Tensor, w: torch.Tensor, dil: int) -> torch.Tensor:
    """out[b,c,n] = sum_k w[k] * x[b,c,(n + dil*k) mod N]  (채널별 동일 필터)."""
    b, c, n = x.shape
    L = w.shape[-1]
    pad = (L - 1) * dil
    if pad > 0:
        reps = int(np.ceil(pad / n))
        xp = torch.cat([x] + [x] * reps, dim=-1)[..., : n + pad]
    else:
        xp = x
    ww = w.view(1, 1, -1).to(xp.dtype).expand(c, 1, L)
    return F.conv1d(xp, ww, dilation=dil, groups=c)


class TorchSWT(nn.Module):
    """(B, 1, N) -> (B, level+1, N), 계수 순서 [cA_L, cD_L, ..., cD_1]."""

    def __init__(self, wavelet: str = "sym4", level: int = 5):
        super().__init__()
        self.wavelet, self.level = wavelet, int(level)
        lo, hi = swt_filters(wavelet)
        self.L = int(lo.size)
        self.register_buffer("lo", torch.tensor(lo, dtype=torch.float64))
        self.register_buffer("hi", torch.tensor(hi, dtype=torch.float64))

    def _shift(self, dil: int) -> int:
        return -(self.L // 2 - 1) * dil

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != 1:
            raise ValueError(f"expected (B,1,N), got {tuple(x.shape)}")
        n = x.shape[-1]
        if n % (2 ** self.level) != 0:
            raise ValueError(f"length {n} must be a multiple of 2^{self.level}")
        lo = self.lo.to(x.dtype)
        hi = self.hi.to(x.dtype)
        cur = x
        details: list[torch.Tensor] = []
        for j in range(1, self.level + 1):
            dil = 2 ** (j - 1)
            s = self._shift(dil)
            z = torch.roll(cur, shifts=-s, dims=-1)
            details.append(_circ_corr(z, hi, dil))
            cur = _circ_corr(z, lo, dil)
        return torch.cat([cur] + details[::-1], dim=1)


class TorchISWT(nn.Module):
    """(B, level+1, N) -> (B, 1, N). TorchSWT 의 정확한 역변환."""

    def __init__(self, wavelet: str = "sym4", level: int = 5):
        super().__init__()
        self.wavelet, self.level = wavelet, int(level)
        lo, hi = swt_filters(wavelet)
        self.L = int(lo.size)
        # 복원은 f_rev 를 '되집어' 쓰므로 원래(비역순) 필터가 필요하다
        self.register_buffer("lo_f", torch.tensor(lo[::-1].copy(), dtype=torch.float64))
        self.register_buffer("hi_f", torch.tensor(hi[::-1].copy(), dtype=torch.float64))

    def _shift(self, dil: int) -> int:
        return -(self.L // 2 - 1) * dil

    def _adj(self, a: torch.Tensor, f: torch.Tensor, dil: int, s: int) -> torch.Tensor:
        v = _circ_corr(torch.roll(a, shifts=dil * (self.L - 1), dims=-1), f, dil)
        return torch.roll(v, shifts=s, dims=-1)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        if c.dim() != 3 or c.shape[1] != self.level + 1:
            raise ValueError(f"expected (B,{self.level + 1},N), got {tuple(c.shape)}")
        lo_f = self.lo_f.to(c.dtype)
        hi_f = self.hi_f.to(c.dtype)
        cur = c[:, :1]
        for j in range(self.level, 0, -1):
            dil = 2 ** (j - 1)
            s = self._shift(dil)
            d = c[:, self.level - j + 1: self.level - j + 2]
            cur = 0.5 * (self._adj(cur, lo_f, dil, s) + self._adj(d, hi_f, dil, s))
        return cur
