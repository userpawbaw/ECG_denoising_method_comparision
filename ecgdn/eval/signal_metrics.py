"""신호 재구성 지표 (docs/02_procedure.md STEP 05).

핵심 설계 결정 (docs/00_review.md A-1):
    soft-threshold / Kalman / MSE 학습 DL 은 모두 **구조적으로 출력 진폭을 줄인다**.
    이걸 보정 없이 SNR 로 재면 "파형은 정확한데 크기만 작은 출력" 이
    "파형이 틀어진 출력" 과 같은 점수를 받는다.
    따라서 strict / scaled / gain_bias 세 값을 **항상 함께** 리포트한다.

DC 규약
-------
    **모든 지표는 x, xhat 양쪽의 평균을 제거한 뒤 계산한다.**
    이유: (1) ECG 의 절대 전위 기준선은 임의값이고 공통 front-end HPF 가 이미 DC 를
    제거한다. (2) 지표 간 정의를 통일해야 SNR/RMSE/PRDN/CC 가 서로 모순되지 않는다.
    느린 baseline wander 잔차는 평균 제거로 숨지 않으므로 평가력은 유지된다.
"""
from __future__ import annotations

import numpy as np

from ..utils import as_float64, power

__all__ = [
    "snr_db", "optimal_gain", "snr_out_strict", "snr_out_scaled",
    "rmse", "mae", "prdn", "pearson_cc", "metrics_signal",
]


def snr_db(x: np.ndarray, err: np.ndarray) -> float:
    """10 log10( P(x) / P(err) ). 평균 제거 파워."""
    pe = power(err)
    if pe <= 0:
        return float("inf")
    px = power(x)
    if px <= 0:
        return float("-inf")
    return float(10.0 * np.log10(px / pe))


def optimal_gain(x: np.ndarray, xhat: np.ndarray) -> float:
    """alpha* = argmin_a ||a*xhat - x||^2  (평균 제거 후).

    1 보다 크면 출력이 참값보다 **작다**는 뜻. 표에 그대로 싣는다.
    """
    x0, h0 = as_float64(x, xhat)
    x0 = x0 - x0.mean()
    h0 = h0 - h0.mean()
    den = float(h0 @ h0)
    if den <= 0:
        return float("nan")
    return float((x0 @ h0) / den)


def _prep(x: np.ndarray, xhat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """DC 규약: 양쪽 모두 평균 제거. 모든 지표의 공통 입구."""
    x0, h0 = as_float64(x, xhat)
    if x0.size != h0.size:
        raise ValueError(f"length mismatch x={x0.size} xhat={h0.size}")
    return x0 - x0.mean(), h0 - h0.mean()


def snr_out_strict(x: np.ndarray, xhat: np.ndarray) -> float:
    """보정 없는 출력 SNR — '그대로 써도 되는가' 를 잰다."""
    x0, h0 = _prep(x, xhat)
    return snr_db(x0, h0 - x0)


def snr_out_scaled(x: np.ndarray, xhat: np.ndarray) -> float:
    """최적 이득 보정 후 출력 SNR — '파형 구조 자체가 맞는가' 를 잰다."""
    x0, h0 = _prep(x, xhat)
    a = optimal_gain(x0, h0)
    if not np.isfinite(a):
        return float("-inf")
    return snr_db(x0, a * h0 - x0)


def rmse(x: np.ndarray, xhat: np.ndarray) -> float:
    x0, h0 = _prep(x, xhat)
    return float(np.sqrt(np.mean((x0 - h0) ** 2)))


def mae(x: np.ndarray, xhat: np.ndarray) -> float:
    x0, h0 = _prep(x, xhat)
    return float(np.mean(np.abs(x0 - h0)))


def prdn(x: np.ndarray, xhat: np.ndarray) -> float:
    """정규화 PRD [%]. 분모에 평균 제거 사용 (docs/00_review.md A-4).

    PRD1(raw) 은 계산하지 않는다 — DC offset 이 있으면 오차를 과소평가한다.
    """
    x0, h0 = _prep(x, xhat)
    den = float(np.sum(x0 ** 2))
    if den <= 0:
        return float("nan")
    return float(100.0 * np.sqrt(np.sum((x0 - h0) ** 2) / den))


def pearson_cc(x: np.ndarray, xhat: np.ndarray) -> float:
    x0, h0 = _prep(x, xhat)
    d = float(np.sqrt((x0 @ x0) * (h0 @ h0)))
    if d <= 0:
        return float("nan")
    return float((x0 @ h0) / d)


def metrics_signal(x: np.ndarray, y: np.ndarray | None, xhat: np.ndarray) -> dict[str, float]:
    """전 신호 지표를 한 번에.

    Parameters
    ----------
    x    : clean reference
    y    : noisy input (None 이면 snr_in / snr_imp 는 NaN)
    xhat : denoised output
    """
    x0, h0 = _prep(x, xhat)

    out: dict[str, float] = {}
    if y is not None:
        y0, _ = _prep(y, y)
        out["snr_in"] = snr_db(x0, y0 - x0)
    else:
        out["snr_in"] = float("nan")

    out["snr_out_strict"] = snr_out_strict(x0, h0)
    out["snr_out_scaled"] = snr_out_scaled(x0, h0)
    out["snr_imp_strict"] = out["snr_out_strict"] - out["snr_in"]
    out["snr_imp_scaled"] = out["snr_out_scaled"] - out["snr_in"]
    out["gain_bias"] = optimal_gain(x0, h0)
    out["rmse"] = rmse(x0, h0)
    out["mae"] = mae(x0, h0)
    out["prdn"] = prdn(x0, h0)
    out["cc"] = pearson_cc(x0, h0)
    return out
