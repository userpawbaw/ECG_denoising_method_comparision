"""결과 그림 (docs/02_procedure.md STEP 27).

모든 그림은 `results/` 의 산출물만 읽어서 재생성된다 (원본 데이터 재접근 불필요).
그림 안의 텍스트는 영문으로 쓴다 (matplotlib 기본 폰트에 한글이 없다).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = ["waveform_stack", "psd_compare", "snr_curve", "pareto_scatter",
           "noise_heatmap", "beat_type_bars", "training_curves", "METHOD_ORDER",
           "COLORS"]

METHOD_ORDER = ["M00", "M01", "M02", "M03", "M04", "M05f", "M05",
                "M06", "M07", "M08", "B01", "B02"]

COLORS = {
    "M00": "#9e9e9e", "M01": "#8c564b", "M02": "#e377c2", "M03": "#7f7f7f",
    "M04": "#1f77b4", "M05f": "#98df8a", "M05": "#2ca02c",
    "M06": "#ff7f0e", "M07": "#d62728", "M08": "#9467bd",
    "B01": "#17becf", "B02": "#bcbd22",
}
BOUND_STYLE = {"B01": "--", "B02": ":"}


def _order(methods):
    known = [m for m in METHOD_ORDER if m in methods]
    return known + [m for m in methods if m not in METHOD_ORDER]


def waveform_stack(x, y, outputs: dict[str, np.ndarray], fs: float,
                   t0: float = 0.0, dur: float = 4.0, path: str | Path | None = None,
                   title: str = "", labels: dict[str, str] | None = None):
    """F1/F2 — 동일 y축으로 겹쳐 쌓은 파형 비교."""
    i0, i1 = int(t0 * fs), int((t0 + dur) * fs)
    i1 = min(i1, len(x))
    t = np.arange(i0, i1) / fs
    names = ["clean", "noisy"] + _order(list(outputs))
    sig = {"clean": x, "noisy": y, **outputs}
    lo = min(float(np.min(v[i0:i1])) for v in sig.values())
    hi = max(float(np.max(v[i0:i1])) for v in sig.values())
    pad = 0.08 * (hi - lo + 1e-9)

    fig, axes = plt.subplots(len(names), 1, figsize=(11, 1.15 * len(names)),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, nm in zip(axes, names):
        c = COLORS.get(nm, "#333333")
        if nm == "clean":
            c = "#000000"
        elif nm == "noisy":
            c = "#c0392b"
        ax.plot(t, sig[nm][i0:i1], lw=0.9, color=c)
        lab = (labels or {}).get(nm, nm)
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=9)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[-1].set_xlabel("time [s]")
    if title:
        axes[0].set_title(title, fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140); plt.close(fig)
    return fig


def psd_compare(x, y, outputs: dict[str, np.ndarray], fs: float,
                path: str | Path | None = None, title: str = "PSD"):
    """F3 — 어떤 주파수 성분을 없앴는지 보여준다."""
    from scipy.signal import welch

    fig, ax = plt.subplots(figsize=(9, 5))
    nps = min(2048, len(x))
    f, p = welch(x - x.mean(), fs=fs, nperseg=nps)
    ax.semilogy(f, p, lw=2.0, color="k", label="clean")
    f, p = welch(y - y.mean(), fs=fs, nperseg=nps)
    ax.semilogy(f, p, lw=1.0, color="#c0392b", alpha=0.7, label="noisy")
    for nm in _order(list(outputs)):
        v = outputs[nm]
        f, p = welch(v - v.mean(), fs=fs, nperseg=nps)
        ax.semilogy(f, p, lw=1.1, color=COLORS.get(nm, None),
                    ls=BOUND_STYLE.get(nm, "-"), label=nm)
    ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("PSD")
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140); plt.close(fig)
    return fig


def snr_curve(df, metric: str = "snr_imp_scaled", path: str | Path | None = None,
              real_snr: float | None = None, title: str | None = None,
              ylabel: str | None = None):
    """F4 — 입력 SNR 별 성능 곡선. real_snr 이 주어지면 그 위치에 수직선."""
    piv = df[df.metric == metric].pivot_table(index="snr_in_target", columns="method",
                                              values="value", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for nm in _order(list(piv.columns)):
        ax.plot(piv.index, piv[nm], marker="o", ms=4, lw=1.6,
                color=COLORS.get(nm, None), ls=BOUND_STYLE.get(nm, "-"), label=nm)
    if real_snr is not None:
        ax.axvline(real_snr, color="#444444", ls="-.", lw=1.2)
        ax.text(real_snr, ax.get_ylim()[1], f"  measured device SNR ≈ {real_snr:.1f} dB",
                va="top", fontsize=8, color="#444444")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("input SNR [dB]")
    ax.set_ylabel(ylabel or f"{metric} [dB]")
    ax.set_title(title or "Performance vs input SNR")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140); plt.close(fig)
    return fig


def pareto_scatter(removal: dict[str, float], preservation: dict[str, float],
                   path: str | Path | None = None,
                   xlabel: str = "noise removal: snr_imp_scaled [dB]",
                   ylabel: str = "signal preservation: distortion floor [dB]",
                   title: str = "Noise removal vs signal preservation"):
    """F5 — 두 목표의 trade-off. 오른쪽 위가 좋다."""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for nm in _order(list(removal)):
        if nm not in preservation:
            continue
        xv, yv = removal[nm], preservation[nm]
        if not (np.isfinite(xv) and np.isfinite(yv)):
            continue
        ax.scatter(xv, yv, s=90, color=COLORS.get(nm, None),
                   edgecolor="k", linewidth=0.6, zorder=3)
        ax.annotate(nm, (xv, yv), textcoords="offset points", xytext=(7, 4), fontsize=9)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140); plt.close(fig)
    return fig


def noise_heatmap(df, metric: str = "snr_imp_scaled", path: str | Path | None = None,
                  title: str | None = None):
    """F6 — 잡음 종류 x 방법 히트맵."""
    piv = df[df.metric == metric].pivot_table(index="method", columns="cond",
                                              values="value", aggfunc="mean")
    piv = piv.reindex(_order(list(piv.index)))
    fig, ax = plt.subplots(figsize=(1.3 * len(piv.columns) + 3, 0.45 * len(piv) + 2.4))
    v = piv.to_numpy(dtype=float)
    im = ax.imshow(v, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=30,
                                                           ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=9)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isfinite(v[i, j]):
                ax.text(j, i, f"{v[i, j]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title(title or f"{metric} by noise type")
    fig.colorbar(im, ax=ax, label="dB", shrink=0.8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140); plt.close(fig)
    return fig


def beat_type_bars(data: dict[str, dict[str, float]], path: str | Path | None = None,
                   ylabel: str = "beat template CC", title: str = "By beat type"):
    """F7 — beat type 별 층화 결과. data[method][beat_type] = value."""
    methods = _order(list(data))
    types = sorted({t for d in data.values() for t in d})
    w = 0.8 / max(len(types), 1)
    fig, ax = plt.subplots(figsize=(1.0 * len(methods) + 3, 5))
    xs = np.arange(len(methods))
    for i, t in enumerate(types):
        vals = [data[m].get(t, np.nan) for m in methods]
        ax.bar(xs + i * w - 0.4 + w / 2, vals, width=w, label=f"beat {t}")
    ax.set_xticks(xs); ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140); plt.close(fig)
    return fig


def training_curves(runs: dict[str, "object"], path: str | Path | None = None):
    """학습 곡선 (val snr_imp_scaled). runs[name] = DataFrame(log.csv)."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for nm, d in runs.items():
        ax[0].plot(d["epoch"], d["val_snr_imp_scaled"], lw=1.5, label=nm)
        ax[1].semilogy(d["epoch"], d["val_loss"], lw=1.5, label=nm)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("val snr_imp_scaled [dB]")
    ax[0].set_title("validation performance"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val loss")
    ax[1].set_title("validation loss"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140); plt.close(fig)
    return fig
