"""평가 엔진 통합 (docs/02_procedure.md STEP 09).

계약
----
    evaluate(x_clean, y_noisy, x_hat, fs, ...) -> dict[str, float]

guard band 규약 (docs/01_design.md 2.0)
    **방법에는 전체 구간을 주고, 지표는 양끝 EVAL_GUARD_S 초를 제외하고 계산한다.**
    zero-phase HPF 링잉 / Kalman 수렴 / wavelet·OLA 패딩 경계가 지표를 오염시키기 때문.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from ..config import EVAL_GUARD_S
from .morphology import metrics_morph, ref_cache
from .rpeak import detect_rpeaks, metrics_rpeak
from .signal_metrics import metrics_signal
from .spectral import metrics_spectral

__all__ = ["trim_guard", "evaluate", "evaluate_many", "to_long_frame",
           "make_ref_cache"]


def trim_guard(n: int, fs: float, guard_s: float = EVAL_GUARD_S) -> slice:
    """양끝 guard 를 제외한 slice. 구간이 너무 짧으면 전체를 쓴다."""
    g = int(round(guard_s * fs))
    if n <= 2 * g + int(round(2 * fs)):
        return slice(0, n)
    return slice(g, n - g)


def evaluate(x: np.ndarray, y: np.ndarray | None, xhat: np.ndarray, fs: float, *,
             r_peaks_ref: np.ndarray | None = None,
             guard_s: float = EVAL_GUARD_S,
             do_morph: bool = True, do_spectral: bool = True,
             cache: dict | None = None) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64).ravel()
    xhat = np.asarray(xhat, dtype=np.float64).ravel()
    if x.size != xhat.size:
        raise ValueError(f"length mismatch: x={x.size} xhat={xhat.size}")
    sl = trim_guard(x.size, fs, guard_s)

    xt, ht = x[sl], xhat[sl]
    yt = np.asarray(y, dtype=np.float64).ravel()[sl] if y is not None else None

    if r_peaks_ref is None:
        rp = detect_rpeaks(xt, fs)
    else:
        rp = np.asarray(r_peaks_ref, dtype=int) - sl.start
        rp = rp[(rp >= 0) & (rp < xt.size)]

    out: dict[str, float] = {}
    out.update(metrics_signal(xt, yt, ht))
    out.update(metrics_rpeak(xt, ht, fs, r_peaks_ref=rp))
    if do_morph:
        out.update(metrics_morph(xt, ht, fs, rp, cache=cache))
    if do_spectral:
        out.update(metrics_spectral(xt, ht, fs))
    out["eval_len_s"] = float(xt.size / fs)
    return out


def make_ref_cache(x: np.ndarray, fs: float, r_peaks_ref=None,
                   guard_s: float = EVAL_GUARD_S, do_morph: bool = True) -> dict | None:
    """구간 하나에 대해 reference 쪽 계산을 1회만 하고 재사용한다.

    guard 를 적용한 뒤의 좌표계에서 만들어야 `evaluate` 와 정합한다.
    """
    if not do_morph:
        return None
    x = np.asarray(x, dtype=np.float64).ravel()
    sl = trim_guard(x.size, fs, guard_s)
    xt = x[sl]
    if r_peaks_ref is None:
        rp = detect_rpeaks(xt, fs)
    else:
        rp = np.asarray(r_peaks_ref, dtype=int) - sl.start
        rp = rp[(rp >= 0) & (rp < xt.size)]
    return ref_cache(xt, fs, rp)


def evaluate_many(x: np.ndarray, y: np.ndarray, methods: dict[str, Any], fs: float, *,
                  r_peaks_ref: np.ndarray | None = None,
                  ctx: dict[str, Any] | None = None,
                  guard_s: float = EVAL_GUARD_S,
                  do_morph: bool = True, do_spectral: bool = True,
                  timing: bool = True) -> dict[str, dict[str, float]]:
    """여러 방법을 **동일한 y** 에 적용하고 평가한다.

    공정성: 모든 방법이 같은 입력, 같은 guard, 같은 R-peak 검출기를 쓴다.
    """
    import time

    base_ctx = dict(ctx or {})
    cache = make_ref_cache(x, fs, r_peaks_ref, guard_s, do_morph)
    res: dict[str, dict[str, float]] = {}
    for name, fn in methods.items():
        c = dict(base_ctx)
        if getattr(fn, "needs_clean", False):
            c["x_clean"] = x
        t0 = time.perf_counter()
        xhat = fn(y, fs, c)
        dt = time.perf_counter() - t0
        m = evaluate(x, y, xhat, fs, r_peaks_ref=r_peaks_ref, guard_s=guard_s,
                     do_morph=do_morph, do_spectral=do_spectral, cache=cache)
        if timing:
            m["latency_s"] = float(dt)
            m["rtf"] = float(dt / (len(np.asarray(y).ravel()) / fs))
        res[name] = m
    return res


def to_long_frame(records: Iterable[dict[str, Any]]):
    """[{**keys, 'metrics': {...}}, ...] -> long-format DataFrame.

    컬럼: 식별자들 + method + metric + value
    """
    import pandas as pd

    rows = []
    for rec in records:
        meta = {k: v for k, v in rec.items() if k != "metrics"}
        for metric, value in rec["metrics"].items():
            rows.append({**meta, "metric": metric, "value": float(value)})
    return pd.DataFrame(rows)
