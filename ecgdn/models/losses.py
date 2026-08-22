"""손실 함수 (docs/01_design.md 4.2).

**단계적 도입 원칙**: L1 -> L2 -> L3 -> L4 순서로 각각 학습해 ablation 표를 만든다.
한 번에 다 넣으면 성능이 좋아졌을 때 '왜' 좋아졌는지 알 수 없다.

    L1 = MSE
    L2 = MSE + 0.5 * MAE
    L3 = L2 + λd * L_diff          # QRS 의 날카로운 형태 보존
    L4 = L3 + λw * L_band          # subband 도메인 (M08 전용), QRS 대역 가중
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DenoiseLoss", "diff_loss", "band_loss", "make_loss"]


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


class DenoiseLoss(nn.Module):
    """설정 가능한 복합 손실. 각 항의 값을 dict 로 함께 반환해 로깅한다."""

    def __init__(self, w_mse: float = 1.0, w_mae: float = 0.0,
                 w_diff: float = 0.0, w_band: float = 0.0,
                 band_weights: tuple[float, ...] | None = None):
        super().__init__()
        self.w_mse, self.w_mae = float(w_mse), float(w_mae)
        self.w_diff, self.w_band = float(w_diff), float(w_band)
        self.register_buffer(
            "band_weights",
            torch.tensor(band_weights, dtype=torch.float32)
            if band_weights is not None else torch.empty(0))

    def forward(self, xhat, x, s_hat=None, s=None):
        parts: dict[str, torch.Tensor] = {}
        total = xhat.new_zeros(())
        if self.w_mse:
            parts["mse"] = F.mse_loss(xhat, x)
            total = total + self.w_mse * parts["mse"]
        if self.w_mae:
            parts["mae"] = F.l1_loss(xhat, x)
            total = total + self.w_mae * parts["mae"]
        if self.w_diff:
            parts["diff"] = diff_loss(xhat, x)
            total = total + self.w_diff * parts["diff"]
        if self.w_band and s_hat is not None and s is not None:
            bw = self.band_weights if self.band_weights.numel() else None
            parts["band"] = band_loss(s_hat, s, bw)
            total = total + self.w_band * parts["band"]
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
}


def make_loss(name: str = "L1", **override) -> DenoiseLoss:
    if name not in _PRESETS:
        raise KeyError(f"unknown loss preset {name!r}; choose from {sorted(_PRESETS)}")
    cfg = dict(_PRESETS[name]); cfg.update(override)
    return DenoiseLoss(**cfg)
