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
                 n_blocks: int = 1, residual: bool = True):
        super().__init__()
        self.residual = bool(residual)
        self.in_ch, self.out_ch = in_ch, out_ch
        self.chs = tuple(chs)
        depth = len(chs) - 1        # **다운샘플 횟수**이지 잔차 깊이가 아니다
        # `n_blocks` = **해상도 하나당 쌓는 ResBlock 수.** 기본 1 — 종래와 같다.
        #
        # 왜 뺐나: residual learning 의 대표 이점은 «깊어도 학습이 된다» 인데,
        # 레벨당 블록이 1 개면 그 이점을 쓰지 않는다. 용량 실험(D-23 B · F-42)이
        # **폭만** 바꿨던 것도 이것과 짝이다 — 깊이 축이 아예 없었다.
        # `chs` 를 늘리는 것과 달리 이쪽은 **수용영역도 함께 넓힌다.**
        n_blocks = max(1, int(n_blocks))
        self.n_blocks = n_blocks

        self.stem = ConvBlock(in_ch, chs[0], k_stem)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for i in range(depth):
            self.enc.append(nn.Sequential(*[ResBlock(chs[i], k)
                                            for _ in range(n_blocks)]))
            self.down.append(Down(chs[i], chs[i + 1], k))
        self.bottleneck = nn.Sequential(*[ResBlock(chs[-1], k) for _ in range(n_bottleneck)])

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.up.append(Up(chs[i], chs[i - 1], k))
            self.dec.append(nn.Sequential(
                ConvBlock(2 * chs[i - 1], chs[i - 1], k),
                *[ResBlock(chs[i - 1], k) for _ in range(n_blocks)]))
        self.head = nn.Conv1d(chs[0], out_ch, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)      # 초기 출력 = 0 -> 초기 x̂ = y (identity 근방에서 시작)

        # 레벨마다 ResBlock n_blocks 개(각 conv 2 개, stride 1) + Down(conv 1 개, stride 2)
        per_level_k = [k] * (2 * n_blocks) + [k]
        per_level_s = [1] * (2 * n_blocks) + [2]
        self._rf = receptive_field(
            [k_stem] + per_level_k * depth + [k] * (2 * n_bottleneck),
            [1] + per_level_s * depth + [1] * (2 * n_bottleneck))

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
