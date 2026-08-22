"""방법 간 비교의 통계 처리 (docs/00_review.md C-6).

원칙
----
  * 집계 단위는 **record**. window 는 서로 독립이 아니므로 p 값이 과대해진다.
  * 같은 record 에 여러 방법을 적용했으므로 **paired** 검정이 맞다.
  * 다중비교는 Holm-Bonferroni 보정.
  * p 값만 쓰면 "유의하지만 0.2 dB 차이" 같은 무의미한 주장이 되므로
    **효과크기(rank-biserial)** 를 함께 보고한다.
"""
from __future__ import annotations

import numpy as np

__all__ = ["paired_wilcoxon", "holm", "rank_biserial", "compare_methods",
           "summarize"]


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """paired rank-biserial correlation. r = (R+ - R-) / (R+ + R-), 범위 [-1, 1]."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    d = d[np.isfinite(d) & (d != 0)]
    if d.size == 0:
        return 0.0
    from scipy.stats import rankdata
    r = rankdata(np.abs(d))
    rp, rm = float(r[d > 0].sum()), float(r[d < 0].sum())
    tot = rp + rm
    return float((rp - rm) / tot) if tot > 0 else 0.0


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(statistic, p). 표본이 부족하면 (nan, nan)."""
    from scipy.stats import wilcoxon
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 6 or np.allclose(a, b):
        return float("nan"), float("nan")
    try:
        s, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return float(s), float(p)
    except Exception:
        return float("nan"), float("nan")


def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni 보정. NaN 은 통과."""
    p = np.asarray(pvals, dtype=np.float64)
    out = np.full_like(p, np.nan)
    idx = np.where(np.isfinite(p))[0]
    if idx.size == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = order.size
    prev = 0.0
    for rank, i in enumerate(order):
        v = min(1.0, (m - rank) * p[i])
        prev = max(prev, v)
        out[i] = prev
    return out


def summarize(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(n=0, mean=np.nan, std=np.nan, median=np.nan, q1=np.nan, q3=np.nan)
    q1, q3 = np.percentile(v, [25, 75])
    return dict(n=int(v.size), mean=float(v.mean()), std=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                median=float(np.median(v)), q1=float(q1), q3=float(q3))


def compare_methods(df, metric: str, baseline: str, *, unit: str = "record",
                    method_col: str = "method"):
    """baseline 대비 각 방법의 paired 검정표.

    df : long-format (unit, method, metric, value)
    """
    import pandas as pd

    sub = df[df["metric"] == metric]
    wide = sub.pivot_table(index=unit, columns=method_col, values="value", aggfunc="mean")
    if baseline not in wide.columns:
        raise KeyError(f"baseline '{baseline}' not in {list(wide.columns)}")

    rows = []
    for m in wide.columns:
        if m == baseline:
            continue
        pair = wide[[m, baseline]].dropna()
        if pair.empty:
            continue
        a, b = pair[m].to_numpy(), pair[baseline].to_numpy()
        stat, p = paired_wilcoxon(a, b)
        s = summarize(a)
        rows.append(dict(method=m, metric=metric, n=len(a),
                         mean=s["mean"], std=s["std"], median=s["median"],
                         baseline_mean=float(np.nanmean(b)),
                         delta_mean=float(np.nanmean(a - b)),
                         stat=stat, p=p, effect_r=rank_biserial(a, b)))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm(out["p"].to_numpy())
        out = out.sort_values("delta_mean", ascending=False).reset_index(drop=True)
    return out
