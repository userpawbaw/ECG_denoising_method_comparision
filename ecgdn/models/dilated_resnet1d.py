"""M10 — Dilated ResNet 1D (**downsampling 없는 M06**).

## 무엇을 묻는 모델인가

`M06`(ResUNet1D)은 encoder-decoder 구조라 **시간 해상도를 반씩 줄였다가
다시 복원**한다. 그 과정에서 QRS 의 고주파 성분이 뭉개질 수 있다.

발표용 그림 S3(beat 평균 템플릿)에서 실제로 그 흔적이 보인다 — `M08` 의
템플릿 오차가 QRS 부근에 몰려 있고, **downsampling 이 없는 `M04`(SWT)의
오차는 거의 평평하다.**

이 모델은 그 가설을 직접 겨냥한다. **downsampling 을 dilation 으로 바꿔
시간 해상도를 처음부터 끝까지 유지**하고, 나머지는 `M06` 과 같게 둔다.

## TCN 과의 관계

Temporal Convolutional Network(Bai et al. 2018)의 핵심 요소 두 가지 중
**dilated residual stack 은 같고 causal masking 은 쓰지 않는다.**

인과 마스킹을 빼는 이유는 이 비교를 오염시키지 않기 위해서다. 공통
front-end 가 `sosfiltfilt`(zero-phase, 비인과)이므로 이 모델만 인과로
만들어도 파이프라인은 여전히 비인과이고, 그러면 '해상도 유지의 효과' 와
'인과 제약의 대가' 가 한 숫자에 섞인다 (F-10 과 같은 구조의 실수).

인과성은 별도 축으로 물어야 하고, 그때는 front-end 도 함께 인과로 바꿔야
한다 — 그건 모든 방법의 수치를 바꾸는 큰 변경이다.

## 공정성 — `M06` 과 무엇을 맞췄나

| | `M06` | `M10` |
|---|---|---|
| 파라미터 | 976,489 | **979,926** (+0.4 %) |
| 수용영역 | 887 샘플 (3.55 s) | **895 샘플** (+0.9 %) |
| residual 구조 | `x̂ = y - n̂`, head zero-init | 동일 |
| 블록 | GroupNorm + SiLU, 홀수 kernel | 동일 (`blocks.py` 재사용) |
| **시간 해상도** | 1/2 씩 4회 축소 후 복원 | **끝까지 유지** ← 유일한 차이 |

**연산량은 맞추지 않았다.** 전 구간을 full resolution 으로 도는 것은
원리적으로 더 비싸고, 그 비용이 이 설계의 대가다. 그래서 `RTF`(계산 비용)를
결과표에 함께 싣는다 — 성능이 올라도 비용이 몇 배면 판단이 달라진다.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBlock, receptive_field

__all__ = ["DilatedResNet1D"]


def _gn(ch: int, max_groups: int = 8) -> nn.GroupNorm:
    g = max(1, min(max_groups, ch))
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, ch)


class DilatedResBlock(nn.Module):
    """`blocks.ResBlock` 과 같되 dilation 을 받는다.

    `padding = dilation * (k // 2)` 라야 길이가 보존된다 — 'same' padding 의
    dilation 판이다. 이걸 틀리면 길이가 조용히 줄어들고, U-Net 과 달리
    복원 단계가 없어 출력이 입력보다 짧아진다.
    """

    def __init__(self, ch: int, k: int = 9, dilation: int = 1):
        super().__init__()
        if k % 2 == 0:
            raise ValueError("kernel must be odd")
        p = dilation * (k // 2)
        self.c1 = nn.Conv1d(ch, ch, k, padding=p, dilation=dilation)
        self.n1 = _gn(ch)
        self.c2 = nn.Conv1d(ch, ch, k, padding=p, dilation=dilation)
        self.n2 = _gn(ch)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return self.act(x + h)


class DilatedResNet1D(nn.Module):
    # 기본값은 **M06 에 맞춰 역산한 것**이다 (params +0.4 %, RF +0.9 %).
    # dilation 마지막 항이 32 가 아니라 24 인 이유: 32 로 두면 RF 가 1023 이
    # 되어 M06(887)보다 15 % 넓어진다. 그러면 M10 이 이겼을 때 '해상도 유지'
    # 때문인지 '문맥이 넓어서' 인지 가릴 수 없다 (F-10 과 같은 교란).
    def __init__(self, in_ch: int = 1, out_ch: int = 1, ch: int = 95,
                 k_stem: int = 15, k: int = 9,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 24),
                 residual: bool = True):
        super().__init__()
        self.residual = bool(residual)
        self.in_ch, self.out_ch = in_ch, out_ch
        self.ch, self.dilations = int(ch), tuple(dilations)

        self.stem = ConvBlock(in_ch, ch, k_stem)
        self.blocks = nn.ModuleList(
            [DilatedResBlock(ch, k, d) for d in dilations])
        self.head = nn.Conv1d(ch, out_ch, 1)
        # M06 과 동일: 초기 출력 0 -> 초기 x̂ = y (identity 근방에서 시작).
        # hallucination 억제의 근거이기도 하다 (models/README.md).
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # stride 가 전부 1 이므로 jump 가 커지지 않는다 — 수용영역이 dilation
        # 합에만 비례한다.
        ks = [k_stem] + [k, k] * len(dilations)
        self._rf = 1 + (k_stem - 1) + sum(2 * (k - 1) * d for d in dilations)
        assert len(ks) == 1 + 2 * len(dilations)

    @property
    def receptive_field_samples(self) -> int:
        return int(self._rf)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        h = self.stem(y)
        for b in self.blocks:
            h = b(h)
        out = self.head(h)
        if self.residual and self.in_ch == self.out_ch:
            return y - out                 # out = 예측된 잡음
        return out
