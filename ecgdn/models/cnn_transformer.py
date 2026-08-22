"""M09 — CNN + Transformer (docs/01_design.md 4장, 확장 모델).

**연구 질문**: 긴 시간 문맥(여러 beat 의 리듬)이 denoising 에 실제로 도움이 되는가?

  CNN  : 국소 파형(QRS 형태)에 강한 inductive bias
  Attn : beat 사이의 장거리 의존성

구조
    y (B,1,N)
      -> CNN stem + 2단 다운샘플            (B, C, N/4)   토큰 256 개 (N=1024)
      -> 학습형 positional embedding
      -> TransformerEncoder x depth
      -> 2단 업샘플 + skip
      -> head -> 예측 잡음 n̂,  x̂ = y - n̂

M06/M08 과 마찬가지로 head 를 0 초기화해 **identity 에서 출발**한다.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBlock, Down, ResBlock, Up, receptive_field

__all__ = ["CNNTransformer"]


class CNNTransformer(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, chs: tuple[int, ...] = (32, 64, 96),
                 d_model: int = 96, nhead: int = 4, depth: int = 3,
                 ff_mult: int = 2, dropout: float = 0.0, max_len: int = 4096,
                 k: int = 9, k_stem: int = 15, residual: bool = True):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.residual = bool(residual)

        self.stem = ConvBlock(in_ch, chs[0], k_stem)
        self.enc1 = ResBlock(chs[0], k)
        self.dn1 = Down(chs[0], chs[1], k)
        self.enc2 = ResBlock(chs[1], k)
        self.dn2 = Down(chs[1], chs[2], k)

        self.proj_in = nn.Conv1d(chs[2], d_model, 1)
        self.pos = nn.Parameter(torch.zeros(1, max_len // 4 + 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=ff_mult * d_model,
                                           dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.attn = nn.TransformerEncoder(layer, num_layers=depth)
        self.proj_out = nn.Conv1d(d_model, chs[2], 1)

        self.up2 = Up(chs[2], chs[1], k)
        self.dec2 = nn.Sequential(ConvBlock(2 * chs[1], chs[1], k), ResBlock(chs[1], k))
        self.up1 = Up(chs[1], chs[0], k)
        self.dec1 = nn.Sequential(ConvBlock(2 * chs[0], chs[0], k), ResBlock(chs[0], k))
        self.head = nn.Conv1d(chs[0], out_ch, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # attention 은 전 토큰을 보므로 실효 수용영역은 window 전체다.
        self._rf_conv = receptive_field([k_stem, k, k, k, k, k, k], [1, 1, 2, 1, 2, 1, 1])

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def receptive_field_samples(self) -> int:
        return -1          # attention 으로 인해 window 전체 (유한 conv RF 로 표현 불가)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        h0 = self.stem(y)
        s1 = self.enc1(h0)
        h1 = self.dn1(s1)
        s2 = self.enc2(h1)
        h2 = self.dn2(s2)

        t = self.proj_in(h2).transpose(1, 2)             # (B, T, d)
        if t.shape[1] > self.pos.shape[1]:
            raise ValueError(f"sequence {t.shape[1]} > max_len/4 {self.pos.shape[1]}")
        t = self.attn(t + self.pos[:, : t.shape[1]])
        h2 = self.proj_out(t.transpose(1, 2))

        h = self.dec2(torch.cat([self.up2(h2, target_len=s2.shape[-1]), s2], dim=1))
        h = self.dec1(torch.cat([self.up1(h, target_len=s1.shape[-1]), s1], dim=1))
        out = self.head(h)
        if self.residual and self.in_ch == self.out_ch:
            return y - out
        return out
