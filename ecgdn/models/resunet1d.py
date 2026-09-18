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
from .film import CondEmbed, ResStack

__all__ = ["ResUNet1D"]


class ResUNet1D(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1,
                 chs: tuple[int, ...] = (24, 32, 48, 64, 96),
                 k_stem: int = 15, k: int = 9, n_bottleneck: int = 2,
                 n_blocks: int = 1, residual: bool = True,
                 cond: bool = False, cond_dim: int = 32,
                 cond_lo: float = -5.0, cond_hi: float = 20.0):
        super().__init__()
        self.residual = bool(residual)
        # 조건화(D-28 2 번). **끄면 종래와 state_dict 키까지 같다** — 켤 때만
        # `ResStack` 으로 갈아끼운다 (`ecgdn/models/film.py` 머리말).
        self.cond = bool(cond)
        self.needs_cond = self.cond
        self.embed = CondEmbed(cond_dim, cond_lo, cond_hi) if self.cond else None
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
            self.enc.append(self._stack(chs[i], k, n_blocks, cond_dim))
            self.down.append(Down(chs[i], chs[i + 1], k))
        self.bottleneck = self._stack(chs[-1], k, n_bottleneck, cond_dim)

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.up.append(Up(chs[i], chs[i - 1], k))
            self.dec.append(self._stack(chs[i - 1], k, n_blocks, cond_dim,
                                        cin=2 * chs[i - 1]))
        self.head = nn.Conv1d(chs[0], out_ch, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)      # 초기 출력 = 0 -> 초기 x̂ = y (identity 근방에서 시작)

        # 레벨마다 ResBlock n_blocks 개(각 conv 2 개, stride 1) + Down(conv 1 개, stride 2)
        per_level_k = [k] * (2 * n_blocks) + [k]
        per_level_s = [1] * (2 * n_blocks) + [2]
        self._rf = receptive_field(
            [k_stem] + per_level_k * depth + [k] * (2 * n_bottleneck),
            [1] + per_level_s * depth + [1] * (2 * n_bottleneck))

    def _stack(self, ch: int, k: int, n: int, cond_dim: int,
               cin: int | None = None) -> nn.Module:
        """조건이 켜지면 `ResStack`, 아니면 종래의 `nn.Sequential`."""
        if self.cond:
            return ResStack(ch, k, n, cond_dim, cin=cin)
        pre = [ConvBlock(cin, ch, k)] if cin is not None else []
        return nn.Sequential(*pre, *[ResBlock(ch, k) for _ in range(n)])

    @property
    def clean_cond_snr(self) -> float:
        """clean 을 통과시킬 때 줄 조건값 (`L6`).

        clean 의 참 SNR 은 ∞ 지만, 학습 범위 밖으로 멀리 보내면 조건 MLP 가
        외삽 구간에서만 쓰이는 값을 배우게 된다. **학습 범위의 위끝**을 준다 —
        「우리가 본 것 중 가장 깨끗함」이 정확히 이 뜻이다.
        """
        return float(self.embed.hi) if self.cond else float("nan")

    @property
    def receptive_field_samples(self) -> int:
        return int(self._rf)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _embed(self, y: torch.Tensor, snr) -> torch.Tensor | None:
        """조건 벡터. **조용한 실패를 막는 게 목적**이다.

        조건화 모델에 SNR 을 안 주면 0 으로 때우는 대신 **터뜨린다** — 안
        그러면 「조건화가 안 통하더라」라는 틀린 결론이 나온다.
        """
        if not self.cond:
            if snr is not None:
                raise ValueError("cond=False 인 모델에 snr 을 줬다 "
                                 "(설정에서 cond: true 를 빠뜨리지 않았는지 볼 것)")
            return None
        if snr is None:
            raise ValueError("cond=True 인 모델은 snr 이 필요하다 "
                             "(학습·평가 양쪽에서 meta['snr'] 을 넘길 것)")
        t = torch.as_tensor(snr, dtype=torch.float32, device=y.device)
        t = t.reshape(-1)
        if t.numel() == 1 and y.shape[0] != 1:
            t = t.expand(y.shape[0])
        if t.numel() != y.shape[0]:
            raise ValueError(f"snr 개수 {t.numel()} 가 배치 {y.shape[0]} 와 다르다")
        return self.embed(t)

    def forward(self, y: torch.Tensor,
                snr: torch.Tensor | None = None) -> torch.Tensor:
        emb = self._embed(y, snr)
        h = self.stem(y)
        skips = []
        for enc, dn in zip(self.enc, self.down):
            h = enc(h, emb) if self.cond else enc(h)
            skips.append(h)
            h = dn(h)
        h = self.bottleneck(h, emb) if self.cond else self.bottleneck(h)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            h = up(h, target_len=s.shape[-1])
            hs = torch.cat([h, s], dim=1)
            h = dec(hs, emb) if self.cond else dec(hs)
        out = self.head(h)
        if self.residual and self.in_ch == self.out_ch:
            return y - out                 # out = 예측된 잡음
        return out
