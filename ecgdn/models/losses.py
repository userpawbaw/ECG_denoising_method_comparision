"""손실 함수 (docs/01_design.md 4.2).

**단계적 도입 원칙**: L1 -> L2 -> L3 -> L4 순서로 각각 학습해 ablation 표를 만든다.
한 번에 다 넣으면 성능이 좋아졌을 때 '왜' 좋아졌는지 알 수 없다.

    L1 = MSE
    L2 = MSE + 0.5 * MAE
    L3 = L2 + λd * L_diff          # QRS 의 날카로운 형태 보존
    L4 = L3 + λw * L_band          # subband 도메인 (M08 전용), QRS 대역 가중
    L5 = L3, 단 window 별 잡음 전력으로 정규화   # 모든 SNR 을 같은 무게로
    L6 = L3 + λc * ‖model(clean) − clean‖²      # 깨끗한 입력을 건드리지 않게

**L5 · L6 이 겨냥하는 것** (91_report.md 5.8.2 · 5.3):

- `L5` — MSE 는 오차가 큰 저 SNR window 에 압도적으로 무겁게 실린다. D1 −5 dB
  의 rmse 가 20 dB 의 12 배다. 그래서 모델이 어려운 구간으로 학습되고 쉬운
  구간에 적용된다. 고 SNR 열세(교차점 13 dB)가 그 결과다.
- `L6` — EXP-C(잡음 없는 clean 통과)에서 딥러닝의 출력 SNR 이 22~24 dB 로
  SWT 의 40.14 dB 보다 **17 dB 낮다.** 현재 어떤 손실도 이것을 겨냥하지 않는다.
  residual 구조라 모델 출력이 곧 예측 잡음이므로, **입력이 이미 깨끗하면
  정답은 "아무것도 빼지 않는 것"** 이다.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DenoiseLoss", "diff_loss", "band_loss", "make_loss",
           "per_window_loss", "snr_stratum_weights", "realized_snr_db",
           "SNR_STRATA_DB", "LOSS_NAMES"]


def diff_loss(xhat: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    """1차(또는 2차) 차분의 MSE. 시간 미분을 맞추면 QRS 의 sharp edge 가 보존된다."""
    a, b = xhat, x
    for _ in range(order):
        a = a[..., 1:] - a[..., :-1]
        b = b[..., 1:] - b[..., :-1]
    return F.mse_loss(a, b)


def band_loss(s_hat: torch.Tensor, s: torch.Tensor,
              weights: torch.Tensor | None = None) -> torch.Tensor:
    """subband 도메인 MSE. weights 는 (n_band,) — QRS 대역(D3/D4)에 큰 값을 준다.

    계수 순서는 [A_L, D_L, ..., D_1] (TorchSWT 와 동일).
    """
    e = (s_hat - s) ** 2                       # (B, n_band, N)
    per_band = e.mean(dim=(0, 2))              # (n_band,)
    if weights is None:
        return per_band.mean()
    w = weights.to(per_band.device, per_band.dtype)
    return (per_band * w).sum() / w.sum()


def per_window_loss(xhat: torch.Tensor, x: torch.Tensor,
                    w_mae: float = 0.0, w_diff: float = 0.0) -> torch.Tensor:
    """window 별 복합 손실 `(B,)`. 배치 평균을 내면 기존 L1~L3 와 **정확히 같다.**

    모든 window 의 길이가 같으므로 `mean_i(mean_t(·)) == mean_{i,t}(·)` 이다.
    이 항등식이 L5 를 "L3 에 가중치만 붙인 것" 으로 만들어 준다 (테스트로 고정).
    """
    dims = tuple(range(1, xhat.dim()))
    out = ((xhat - x) ** 2).mean(dim=dims)
    if w_mae:
        out = out + w_mae * (xhat - x).abs().mean(dim=dims)
    if w_diff:
        d = (xhat[..., 1:] - xhat[..., :-1]) - (x[..., 1:] - x[..., :-1])
        out = out + w_diff * (d ** 2).mean(dim=dims)
    return out


# EXP-A 의 평가 SNR 격자와 같은 경계. 각 구간이 학습에서 같은 무게를 갖게 한다.
SNR_STRATA_DB = (0.0, 5.0, 10.0, 15.0, 20.0)


def realized_snr_db(y: torch.Tensor, x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    """window 가 **실제로** 받은 SNR `(B,)`.

    데이터셋 메타의 `snr` 은 구간 전체에 대한 **목표값**이다. 잡음(특히 EM/MA)
    은 폭발적이라 window 마다 실제 SNR 이 크게 다르다 — 실측하니 목표 −5~20 dB
    범위에서 실제로는 **−9~80 dB** 였다 `[측정]`. 그래서 메타가 아니라 여기서
    다시 잰다.
    """
    dims = tuple(range(1, y.dim()))
    s = (x ** 2).mean(dim=dims)
    p = ((y - x) ** 2).mean(dim=dims)
    return 10.0 * torch.log10(s.clamp_min(eps) / p.clamp_min(eps))


def snr_stratum_weights(per_window: torch.Tensor, y: torch.Tensor, x: torch.Tensor,
                        edges: tuple[float, ...] = SNR_STRATA_DB,
                        eps: float = 1e-20) -> torch.Tensor:
    """실제 SNR 구간마다 **손실 기여가 같아지도록** 하는 window 가중치 `(B,)`.

    가중치 합이 1 인 가중 **평균**의 가중치이므로 값이 항상 `min(per_window)`
    와 `max` 사이에 있다 — **손실 크기가 바뀌지 않아** 같은 learning rate 를
    그대로 쓸 수 있다. 모든 window 가 한 구간에 있으면 균등 가중이 되어 원래
    손실로 정확히 환원된다. 정규화 분모는 `detach` 하므로 가중치 자체로는
    gradient 가 흐르지 않는다.

    **무엇을 고치는가.** 학습된 M06 으로 실측하니 실제 SNR 최저 5분위가 손실의
    **35~42 %**, 최고 5분위가 **8.8~10 %** 였다 — 저 SNR 로 **4.2~4.8 배**
    쏠려 있다 `[측정]`. 딥러닝이 지는 곳이 고 SNR(교차점 13 dB)이라는 것과
    방향이 맞는다.

    **버린 설계 둘** (둘 다 실측이 기각했다):

    1. `1/p` (잡음 전력의 역수) 가중 — `p` 는 배치 안에서 10⁷~10⁸ 배 퍼지는데
       손실 기여는 20 배밖에 안 퍼진다. 모델 오차가 잔여 잡음이 아니라 **자기
       왜곡 바닥**에 지배되기 때문이다. 이 가중은 거의 잡음 없는 window 하나에
       배치를 몰아준다 (퍼짐이 20 배 → 3,780 배로 **악화**).
    2. 구간별 **개수** 균등 — 고 SNR 구간에 window 가 3 배 많아서, 개수를
       맞추면 오히려 저 SNR 기여가 20 % → 33 % 로 **늘었다.**

    맞춰야 하는 것은 개수가 아니라 **기여**다. 구간 `k` 의 현재 기여
    `S_k = Σ_{i∈k} per_window_i` 로 나누면 모든 구간이 같은 몫을 갖는다.

    구간 경계는 EXP-A 의 평가 SNR 격자와 같다 — **평가되는 대역마다 학습
    무게가 같다**는 뜻이 되어 해석이 분명해진다.
    """
    snr = realized_snr_db(y, x)
    idx = torch.bucketize(snr, torch.tensor(edges, device=snr.device, dtype=snr.dtype))
    n_str = len(edges) + 1
    share = torch.zeros(n_str, device=snr.device, dtype=per_window.dtype)
    share.scatter_add_(0, idx, per_window.detach())
    u = 1.0 / share.clamp_min(eps)[idx]
    return u / u.sum()


class DenoiseLoss(nn.Module):
    """설정 가능한 복합 손실. 각 항의 값을 dict 로 함께 반환해 로깅한다."""

    def __init__(self, w_mse: float = 1.0, w_mae: float = 0.0,
                 w_diff: float = 0.0, w_band: float = 0.0,
                 band_weights: tuple[float, ...] | None = None,
                 snr_balance: bool = False,
                 w_clean: float = 0.0, clean_frac: float = 0.25):
        super().__init__()
        self.w_mse, self.w_mae = float(w_mse), float(w_mae)
        self.w_diff, self.w_band = float(w_diff), float(w_band)
        # L5: MSE 항을 window 별 잡음 전력으로 정규화한다 (MAE/diff 는 그대로).
        self.snr_balance = bool(snr_balance)
        # L6: 깨끗한 입력에 대해 모델이 아무것도 하지 않게 하는 벌점.
        self.w_clean, self.clean_frac = float(w_clean), float(clean_frac)
        self.register_buffer(
            "band_weights",
            torch.tensor(band_weights, dtype=torch.float32)
            if band_weights is not None else torch.empty(0))

    @property
    def needs_noisy_input(self) -> bool:
        """`forward` 에 `y`(잡음 섞인 입력)가 필요한가. L5 만 True."""
        return self.snr_balance

    @property
    def needs_clean_pass(self) -> bool:
        """학습 루프가 clean 입력으로 forward 를 한 번 더 돌려야 하는가. L6 만 True."""
        return self.w_clean > 0.0

    def forward(self, xhat, x, s_hat=None, s=None, y=None, xhat_clean=None):
        """`y` 는 잡음 섞인 입력(L5 용), `xhat_clean` 은 clean 통과 출력(L6 용).

        필요한데 없으면 **조용히 건너뛰지 않고** 예외를 낸다 — 손실 설정과
        학습 루프가 어긋난 채로 도는 것이 이 프로젝트에서 반복된 실패 형태다.
        """
        parts: dict[str, torch.Tensor] = {}

        # 시간영역 항(mse+mae+diff)은 window 별로 계산한다. L5 는 여기에
        # 가중치를 곱할 뿐이고, 가중치가 균일하면 기존 L1~L3 와 동일하다.
        pw = per_window_loss(xhat, x, self.w_mae, self.w_diff)
        if self.snr_balance:
            if y is None:
                raise ValueError(
                    "snr_balance=True 인데 y(잡음 입력)가 없다. "
                    "학습 루프가 loss_fn.needs_noisy_input 을 보고 y 를 넘겨야 한다.")
            time_term = (pw * snr_stratum_weights(pw, y, x)).sum()
        else:
            time_term = pw.mean()
        # 로깅용으로 항을 쪼개 둔다 (가중치 없는 값 — 런 간 비교가 되도록)
        parts["mse"] = F.mse_loss(xhat, x)
        if self.w_mae:
            parts["mae"] = F.l1_loss(xhat, x)
        if self.w_diff:
            parts["diff"] = diff_loss(xhat, x)
        total = self.w_mse * time_term

        if self.w_band and s_hat is not None and s is not None:
            bw = self.band_weights if self.band_weights.numel() else None
            parts["band"] = band_loss(s_hat, s, bw)
            total = total + self.w_band * parts["band"]

        if self.w_clean:
            if xhat_clean is None:
                raise ValueError(
                    "w_clean>0 인데 xhat_clean 이 없다. 학습 루프가 "
                    "loss_fn.needs_clean_pass 를 보고 clean forward 를 돌려야 한다.")
            # **주 항과 같은 복합 손실**을 clean 통과 출력에 적용한다. mse 만
            # 쓰면 이 항이 손실의 1~4 % 밖에 안 되는데, 주 항의 90 % 가 MAE 라
            # 그렇다 [측정]. 같은 형태를 써야 w_clean 이 "주 항 대비 몇 배"
            # 라는 뜻을 갖는다.
            xc = x[:xhat_clean.shape[0]]
            parts["clean"] = per_window_loss(
                xhat_clean, xc, self.w_mae, self.w_diff).mean()
            total = total + self.w_clean * parts["clean"]

        parts["total"] = total
        return total, parts


# 계수 순서 [A5, D5, D4, D3, D2, D1] 에서 QRS 주 대역(D3, D4)에 2배 가중
BAND_W_QRS = (1.0, 1.0, 2.0, 2.0, 1.0, 1.0)

_PRESETS = {
    "L1": dict(w_mse=1.0),
    "L2": dict(w_mse=1.0, w_mae=0.5),
    "L3": dict(w_mse=1.0, w_mae=0.5, w_diff=0.3),
    "L4": dict(w_mse=1.0, w_mae=0.5, w_diff=0.3, w_band=0.2,
               band_weights=BAND_W_QRS),
    # L5/L6 은 둘 다 L3 를 바탕으로 항을 하나만 바꾸거나 더한다. L4 를 바탕으로
    # 하지 않은 것은 L4 가 M08 전용(subband)이라 M06 에 적용할 수 없어서다.
    "L5": dict(w_mse=1.0, w_mae=0.5, w_diff=0.3, snr_balance=True),
    "L6": dict(w_mse=1.0, w_mae=0.5, w_diff=0.3, w_clean=0.5, clean_frac=0.25),
}

LOSS_NAMES = tuple(_PRESETS)


def make_loss(name: str = "L1", **override) -> DenoiseLoss:
    if name not in _PRESETS:
        raise KeyError(f"unknown loss preset {name!r}; choose from {sorted(_PRESETS)}")
    cfg = dict(_PRESETS[name]); cfg.update(override)
    return DenoiseLoss(**cfg)
