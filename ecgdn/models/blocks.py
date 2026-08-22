"""공통 빌딩 블록 (docs/01_design.md 4.1).

설계 결정
--------
  * **GroupNorm** 을 쓴다 (BatchNorm 아님). ECG window 는 환자/구간마다 진폭 분포가
    크게 달라 batch 통계가 불안정하다. 추론 시 batch 크기에도 의존하지 않는다.
  * activation 은 SiLU. ReLU 보다 미분이 매끄러워 회귀(재구성) 과제에 유리하다.
  * kernel 은 홀수만 사용 -> 'same' padding 이 정확히 맞아 길이가 보존된다.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ConvBlock", "ResBlock", "Down", "Up", "receptive_field"]


def _gn(ch: int, max_groups: int = 8) -> nn.GroupNorm:
    g = max(1, min(max_groups, ch))
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, ch)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 9):
        super().__init__()
        if k % 2 == 0:
            raise ValueError("kernel must be odd")
        self.f = nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2),
                               _gn(cout), nn.SiLU())

    def forward(self, x):
        return self.f(x)


class ResBlock(nn.Module):
    """Conv-GN-SiLU-Conv-GN + skip -> SiLU."""

    def __init__(self, ch: int, k: int = 9):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, k, padding=k // 2)
        self.n1 = _gn(ch)
        self.c2 = nn.Conv1d(ch, ch, k, padding=k // 2)
        self.n2 = _gn(ch)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return self.act(x + h)


class Down(nn.Module):
    """stride-2 conv 로 다운샘플 (max-pool 대신 — 위치 정보 손실이 적다)."""

    def __init__(self, cin: int, cout: int, k: int = 9):
        super().__init__()
        self.f = nn.Sequential(nn.Conv1d(cin, cout, k, stride=2, padding=k // 2),
                               _gn(cout), nn.SiLU())

    def forward(self, x):
        return self.f(x)


class Up(nn.Module):
    """선형 보간 업샘플 + conv. transposed conv 의 체커보드 아티팩트를 피한다."""

    def __init__(self, cin: int, cout: int, k: int = 9):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.f = ConvBlock(cin, cout, k)

    def forward(self, x, target_len: int | None = None):
        x = self.up(x)
        if target_len is not None and x.shape[-1] != target_len:
            x = x[..., :target_len] if x.shape[-1] > target_len else \
                torch.nn.functional.pad(x, (0, target_len - x.shape[-1]), mode="replicate")
        return self.f(x)


def receptive_field(kernels: list[int], strides: list[int]) -> int:
    """1D 수용영역 계산. 설계 근거 기록용 (docs/02_procedure.md STEP 17)."""
    rf, jump = 1, 1
    for k, s in zip(kernels, strides):
        rf += (k - 1) * jump
        jump *= s
    return rf
