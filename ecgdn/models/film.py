"""FiLM 조건화 — 스칼라 SNR 로 각 ResBlock 을 변조한다 (D-28 2 번).

왜 FiLM 인가
-----------
DRUNet 은 잡음 준위를 **입력 채널로 붙인다**. 1D 에서 그대로 하면 «상수 채널
하나»가 되고, 첫 conv 가 그것을 무시하도록 학습돼도 손실이 거의 오르지 않는다.
FiLM 은 조건이 **모든 레벨의 활성에 직접 닿아** 무시할 수가 없다.

어디에 거는가 — **ResBlock 의 두 번째 GroupNorm 뒤, skip 을 더하기 직전**이다.
이 망은 잔차 학습(residual)을 한다: 출력이 `y - n̂` 이라 블록이 만드는 것은
«얼마나 고칠지»이고, SNR 에 따라 변해야 하는 것도 바로 그 양이다. 여기에 γ 를
걸면 «고 SNR 에서는 덜 건드린다»가 한 번의 곱으로 표현된다. skip 경로가 그대로
남아 γ→0 이 «아무것도 안 함»이 된다.

**초기값에서 조건 없음과 비트 단위로 같다** — head 를 0 으로 두고 γ = 1 + dγ 로
쓰기 때문이다. `tests/test_film_cond.py` 가 고정한다.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBlock, ResBlock

__all__ = ["CondEmbed", "ResStack"]


class CondEmbed(nn.Module):
    """스칼라 SNR(dB) → 조건 벡터.

    **푸리에(사인) 특징을 쓰지 않는다.** 확산모형의 timestep 과 달리 SNR
    의존은 매끄럽고 단조로울 것으로 기대되고, 사인 특징은 학습 범위 밖에서
    **되감긴다** — D-28 4 번이 ±5 dB 교란을 줄 예정이라 되감김은 그대로
    「추정 오차에 강건한가」라는 질문을 망친다.
    """

    def __init__(self, dim: int = 32, lo: float = -5.0, hi: float = 20.0):
        super().__init__()
        if hi <= lo:
            raise ValueError("cond_hi 는 cond_lo 보다 커야 한다")
        self.dim, self.lo, self.hi = int(dim), float(lo), float(hi)
        self.f = nn.Sequential(nn.Linear(1, self.dim), nn.SiLU(),
                               nn.Linear(self.dim, self.dim), nn.SiLU())

    def normalize(self, snr: torch.Tensor) -> torch.Tensor:
        """학습 범위를 [-1, 1] 로. **자르지 않는다** — 범위 밖도 그대로 외삽한다."""
        return 2.0 * (snr - self.lo) / (self.hi - self.lo) - 1.0

    def forward(self, snr: torch.Tensor) -> torch.Tensor:
        return self.f(self.normalize(snr.reshape(-1, 1)))


class ResStack(nn.Module):
    """(선택적 ConvBlock) + ResBlock n 개 — 블록마다 FiLM 을 건다.

    `nn.Sequential` 을 쓰지 않는 이유는 조건 벡터를 블록마다 넘겨야 해서다.
    **조건을 끄면 `ResUNet1D` 는 종래대로 `nn.Sequential` 을 쓴다** — 여기로
    갈아끼우면 state_dict 키가 `enc.0.0.c1` 에서 `enc.0.blocks.0.c1` 로 바뀌어
    **지금까지 학습한 체크포인트가 전부 안 열린다**(F-9 계열의 사고).
    """

    def __init__(self, ch: int, k: int, n: int, cond_dim: int,
                 cin: int | None = None):
        super().__init__()
        self.pre = ConvBlock(cin, ch, k) if cin is not None else None
        self.blocks = nn.ModuleList([ResBlock(ch, k) for _ in range(n)])
        self.heads = nn.ModuleList([nn.Linear(cond_dim, 2 * ch) for _ in range(n)])
        for h in self.heads:
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)      # γ = 1 + 0, β = 0 -> 초기엔 조건 없음과 동일

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.pre is not None:
            x = self.pre(x)
        for blk, head in zip(self.blocks, self.heads):
            g, b = head(emb).chunk(2, dim=1)
            x = blk(x, film=(1.0 + g[..., None], b[..., None]))
        return x
