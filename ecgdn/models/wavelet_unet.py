"""M08 — Wavelet-subband Residual U-Net (docs/00_review.md B-2, STEP 23).

**핵심 차별점**
    wavelet 을 '전처리 필터' 가 아니라 **신경망의 입력 표현공간**으로 쓴다.
    D1 을 '고주파니까 버린다' 고 사람이 정하지 않고,
    "D1 안에서 무엇을 남기고 무엇을 지울지" 를 네트워크가 학습하게 한다.

구조
----
    y (B,1,N)
      -> TorchSWT(level=5)            -> s (B,6,N)   [A5,D5,D4,D3,D2,D1]
      -> band-wise Conv1d(groups=6)                  대역별 독립 처리 (대역 특성 보존)
      -> 1x1 fusion                   -> (B,C,N)
      -> Residual U-Net backbone      -> (B,6,N)     예측된 subband 잡음 n̂
      -> ŝ = s - n̂
      -> TorchISWT                    -> x̂ (B,1,N)

  head 는 0 으로 초기화한다. 따라서 학습 시작 시점에 n̂ = 0 이고
  `ISWT(SWT(y)) = y` 이므로 **출력이 정확히 identity** 에서 출발한다.

주의: 첫 band-wise conv 에는 정규화를 넣지 않는다.
      대역별 진폭 자체가 '잡음이 얼마나 큰가' 를 알려주는 정보이기 때문이다.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import _gn
from .resunet1d import ResUNet1D
from .swt_torch import SWT_COEFF_NAMES, TorchISWT, TorchSWT

__all__ = ["WaveletSubbandUNet"]


class WaveletSubbandUNet(nn.Module):
    def __init__(self, wavelet: str = "sym4", level: int = 5,
                 band_ch: int = 8, fuse_ch: int = 32,
                 chs: tuple[int, ...] = (24, 32, 48, 64, 96),
                 k: int = 9, k_stem: int = 15,
                 time_residual: bool = False):
        super().__init__()
        self.level = int(level)
        self.n_band = self.level + 1
        self.band_names = SWT_COEFF_NAMES(self.level)
        self.time_residual = bool(time_residual)
        self.in_ch = 1

        self.swt = TorchSWT(wavelet, level)
        self.iswt = TorchISWT(wavelet, level)

        # 대역별 독립 처리 (정규화 없음 — 대역 진폭 정보를 보존)
        self.band = nn.Sequential(
            nn.Conv1d(self.n_band, self.n_band * band_ch, k, padding=k // 2,
                      groups=self.n_band),
            nn.SiLU(),
            nn.Conv1d(self.n_band * band_ch, self.n_band * band_ch, k,
                      padding=k // 2, groups=self.n_band),
            nn.SiLU(),
        )
        self.fuse = nn.Sequential(nn.Conv1d(self.n_band * band_ch, fuse_ch, 1),
                                  _gn(fuse_ch), nn.SiLU())
        self.backbone = ResUNet1D(in_ch=fuse_ch, out_ch=self.n_band, chs=chs,
                                  k=k, k_stem=k_stem, residual=False)
        # backbone.head 는 이미 0 초기화되어 있다 -> n̂ = 0 -> identity 출발

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def receptive_field_samples(self) -> int:
        return self.backbone.receptive_field_samples

    def forward(self, y: torch.Tensor, return_bands: bool = False):
        s = self.swt(y)                       # (B, n_band, N)
        h = self.fuse(self.band(s))
        n_hat = self.backbone(h)              # 예측된 subband 잡음
        s_hat = s - n_hat
        x_hat = self.iswt(s_hat)
        if self.time_residual:
            x_hat = y - (y - x_hat)           # 항등식 — 확장 지점 표시용
        if return_bands:
            return x_hat, s, s_hat
        return x_hat
