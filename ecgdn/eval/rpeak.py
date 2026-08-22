"""R-peak 기반 지표 (docs/02_procedure.md STEP 06).

공정성 규약 (docs/01_design.md 5.4):
    **모든 방법의 출력에 동일한 검출기(wfdb xqrs)를 적용한다.**
    방법마다 다른 검출기를 쓰면 최종 차이가 denoising 때문인지 검출기 때문인지 알 수 없다.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..config import RPEAK_TOL_MS

__all__ = ["detect_rpeaks", "match_peaks", "metrics_rpeak", "rr_intervals", "hr_from_rr"]


def detect_rpeaks(x: np.ndarray, fs: float, refine_ms: float = 0.0) -> np.ndarray:
    """wfdb XQRS 검출기. 실패하면 빈 배열.

    refine_ms > 0 이면 검출 위치 주변에서 |x| 최대점으로 미세보정한다.
    (기본 0 — 보정은 방법마다 다른 편향을 만들 수 있어 끈다)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < int(2 * fs):
        return np.empty(0, dtype=int)
    try:
        from wfdb.processing import xqrs_detect
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pk = np.asarray(xqrs_detect(x, fs=int(round(fs)), verbose=False), dtype=int)
    except Exception:
        return np.empty(0, dtype=int)
    pk = pk[(pk >= 0) & (pk < x.size)]
    if refine_ms > 0 and pk.size:
        w = max(1, int(round(refine_ms * 1e-3 * fs)))
        out = []
        for p in pk:
            a, b = max(0, p - w), min(x.size, p + w + 1)
            out.append(a + int(np.argmax(np.abs(x[a:b]))))
        pk = np.asarray(out, dtype=int)
    return np.unique(pk)


def match_peaks(ref: np.ndarray, test: np.ndarray, fs: float,
                tol_ms: float = RPEAK_TOL_MS) -> tuple[np.ndarray, int, int, int]:
    """허용오차 내 **1:1 최소비용 매칭**.

    그리디로 하면 tolerance 안에 2개가 들어올 때 지표가 부풀려진다. (STEP 06 사양)

    Returns
    -------
    (pairs, tp, fp, fn) : pairs 는 (M,2) [ref_idx, test_idx] 샘플 인덱스
    """
    ref = np.asarray(ref, dtype=int).ravel()
    test = np.asarray(test, dtype=int).ravel()
    tol = tol_ms * 1e-3 * fs
    if ref.size == 0 or test.size == 0:
        return np.empty((0, 2), dtype=int), 0, int(test.size), int(ref.size)

    from scipy.optimize import linear_sum_assignment
    cost = np.abs(ref[:, None].astype(np.float64) - test[None, :].astype(np.float64))
    big = float(cost.max()) + 10 * tol + 1.0
    masked = np.where(cost <= tol, cost, big)
    ri, ti = linear_sum_assignment(masked)
    keep = cost[ri, ti] <= tol
    ri, ti = ri[keep], ti[keep]

    pairs = np.stack([ref[ri], test[ti]], axis=1)
    tp = int(pairs.shape[0])
    return pairs, tp, int(test.size - tp), int(ref.size - tp)


def rr_intervals(peaks: np.ndarray, fs: float) -> np.ndarray:
    p = np.asarray(peaks, dtype=np.float64).ravel()
    return np.diff(p) / fs if p.size >= 2 else np.empty(0)


def hr_from_rr(rr: np.ndarray) -> float:
    """RR 중앙값 기반 심박수 [bpm]. 평균은 이상치에 취약하므로 중앙값."""
    rr = np.asarray(rr, dtype=np.float64).ravel()
    rr = rr[(rr > 0.25) & (rr < 3.0)]
    return float(60.0 / np.median(rr)) if rr.size else float("nan")


def metrics_rpeak(x_ref: np.ndarray, xhat: np.ndarray, fs: float,
                  r_peaks_ref: np.ndarray | None = None,
                  tol_ms: float = RPEAK_TOL_MS) -> dict[str, float]:
    """R-peak 검출/타이밍 지표.

    Parameters
    ----------
    r_peaks_ref : 이미 알고 있는 참 R-peak (합성 데이터/annotation). None 이면 x_ref 에서 검출.
    """
    ref = np.asarray(r_peaks_ref, dtype=int) if r_peaks_ref is not None \
        else detect_rpeaks(x_ref, fs)
    det = detect_rpeaks(xhat, fs)
    pairs, tp, fp, fn = match_peaks(ref, det, fs, tol_ms)

    out: dict[str, float] = {
        "n_ref_beats": float(ref.size),
        "n_det_beats": float(det.size),
        "tp": float(tp), "fp": float(fp), "fn": float(fn),
        "se": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "ppv": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
    }
    se, ppv = out["se"], out["ppv"]
    out["f1"] = float(2 * se * ppv / (se + ppv)) if (se and ppv and se + ppv > 0) else float("nan")

    if tp:
        err_ms = (pairs[:, 1] - pairs[:, 0]) / fs * 1e3
        out["rpeak_mae_ms"] = float(np.mean(np.abs(err_ms)))
        out["rpeak_bias_ms"] = float(np.mean(err_ms))
        out["rpeak_p95_ms"] = float(np.percentile(np.abs(err_ms), 95))
    else:
        out["rpeak_mae_ms"] = out["rpeak_bias_ms"] = out["rpeak_p95_ms"] = float("nan")

    rr_ref, rr_det = rr_intervals(ref, fs), rr_intervals(det, fs)
    out["hr_ref_bpm"] = hr_from_rr(rr_ref)
    out["hr_hat_bpm"] = hr_from_rr(rr_det)
    out["hr_err_bpm"] = abs(out["hr_hat_bpm"] - out["hr_ref_bpm"])

    # 매칭된 beat 로부터 RR 오차 (연속 매칭 쌍에 대해서만)
    if tp >= 2:
        rr_r = np.diff(pairs[:, 0]) / fs * 1e3
        rr_t = np.diff(pairs[:, 1]) / fs * 1e3
        out["rr_mae_ms"] = float(np.mean(np.abs(rr_t - rr_r)))
    else:
        out["rr_mae_ms"] = float("nan")
    return out
