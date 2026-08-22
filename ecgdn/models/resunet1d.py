"""M06 — Residual 1D U-Net (docs/01_design.md 4.1).

**residual learning**: 네트워크는 clean ECG 가 아니라 **잡음 n̂** 을 예측하고
`x̂ = y - n̂` 로 만든다.
  1) ECG 전체를 다시 그릴 필요가 없어 학습이 쉽다.
  2) 진폭 편향(docs/00_review.md A-1)이 잘 생기지 않는다 — 출력의 대부분이 입력에서 온다.
  3) 없는 파형을 만들어내는(hallucination) 경향이 줄어든다.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBlock, Down, ResBlock, Up, receptive_field

__all__ = ["ResUNet1D"]


class ResUNet1D(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1,
                 chs: tuple[int, ...] = (24, 32, 48, 64, 96),
                 k_stem: int = 15, k: int = 9, n_bottleneck: int = 2,
                 residual: bool = True):
        super().__init__()
        self.residual = bool(residual)
        self.in_ch, self.out_ch = in_ch, out_ch
        self.chs = tuple(chs)
        depth = len(chs) - 1

        self.stem = ConvBlock(in_ch, chs[0], k_stem)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for i in range(depth):
            self.enc.append(ResBlock(chs[i], k))
            self.down.append(Down(chs[i], chs[i + 1], k))
        self.bottleneck = nn.Sequential(*[ResBlock(chs[-1], k) for _ in range(n_bottleneck)])

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.up.append(Up(chs[i], chs[i - 1], k))
            self.dec.append(nn.Sequential(ConvBlock(2 * chs[i - 1], chs[i - 1], k),
                                          ResBlock(chs[i - 1], k)))
        self.head = nn.Conv1d(chs[0], out_ch, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)      # 초기 출력 = 0 -> 초기 x̂ = y (identity 근방에서 시작)

        self._rf = receptive_field(
            [k_stem] + [k, k, k] * depth + [k] * (2 * n_bottleneck),
            [1] + [1, 1, 2] * depth + [1] * (2 * n_bottleneck))

    @property
    def receptive_field_samples(self) -> int:
        return int(self._rf)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        h = self.stem(y)
        skips = []
        for enc, dn in zip(self.enc, self.down):
            h = enc(h)
            skips.append(h)
            h = dn(h)
        h = self.bottleneck(h)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            h = up(h, target_len=s.shape[-1])
            h = dec(torch.cat([h, s], dim=1))
        out = self.head(h)
        if self.residual and self.in_ch == self.out_ch:
            return y - out                 # out = 예측된 잡음
        return out
