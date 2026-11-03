"""판넬(A1) 전용 그림 — 판넬에 놓일 크기 그대로 그려 그림 속 글자가 인쇄 크기로 14 pt 이상이 되게 한다.

    python scripts/make_panel_figs.py            # 전부
    python scripts/make_panel_figs.py --only F5

산출: `results/panel/figs/*.png` (300 dpi). `build_panel.py` 가 `SIZES` 의 크기로 **축소 없이** 놓는다.

발표용 그림(`make_slides.py`)을 판넬 칸에 줄여 넣으면 그림 속 글자가 2~8 pt 로 인쇄되었다
(사용자 피드백 2026-11-03). 그래서 판넬 그림은 따로 그린다. 원칙은 세 가지다.

1. 그림 크기(cm)를 판넬 칸과 같게 두고 글자 크기를 pt 로 정한다 — 축소가 없으므로 pt 가 그대로 인쇄된다.
2. 요소를 줄인다 — 제목 · 부제 · 범례 문장은 판넬 캡션으로 보내고, 그림에는 축 이름과 직접 표지만 둔다.
3. 데이터는 발표용 그림과 같은 경로로 읽는다(`make_slides` 의 함수와 상수를 그대로 쓴다).

글꼴은 `make_slides.slide_style()`(NanumGothic 우선)로 고르고 `axes.unicode_minus` 를 끈다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np

import make_slides as MS
from ecgdn.utils import ensure_dir, save_manifest

OUT = Path("results/panel/figs")
CM = 1 / 2.54
FS = 14            # 그림 속 기본 글자 크기 [pt] — 인쇄 크기 그대로
FS_S = 13          # 눈금 숫자

# 판넬 칸 크기 [cm] — build_panel.py 가 이 크기로 놓는다
SIZES = {
    "F4": (18.2, 9.8),     # 결과: 입력 SNR 교차점
    "F5": (18.2, 9.8),     # 결과: 칼만 가성 QRS군
    "F6": (18.6, 9.8),     # 결과: 잡음 종류별 출력
    "F7": (22.0, 11.0),    # 결과: 개선 시도 16종 (16 행)
    "T1": (11.6, 8.6),     # 이론 카드 ①: SWT 임계값
    "T2": (11.6, 8.6),     # 이론 카드 ②: 칼만 모델
}


def _ko_font():
    """한글 글꼴(NanumGothic 등) 선택, 음수 눈금 깨짐 방지, 판넬용 글자 크기."""
    MS.slide_style()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": FS, "axes.labelsize": FS,
                         "xtick.labelsize": FS_S, "ytick.labelsize": FS_S, "axes.titlesize": FS,
                         "legend.fontsize": FS_S, "axes.linewidth": 0.9})


def _fig(key, **kw):
    w, h = SIZES[key]
    return plt.subplots(figsize=(w * CM, h * CM), **kw)


def _save(fig, key, name):
    w, h = SIZES[key]
    fig.set_size_inches(w * CM, h * CM)
    fig.savefig(OUT / f"{name}.png", dpi=300)       # bbox 를 자르지 않는다 — 크기가 곧 판넬 칸이다
    plt.close(fig)
    print(f"  {OUT / name}.png  ({w} x {h} cm)")


# ------------------------------------------------------------------ F4 교차점
def f4_crossover():
    import pandas as pd

    from ecgdn.eval.stats import compare_methods
    fig, ax = _fig("F4")
    styles = {"d0": (MS.C["M01"], "합성 ECG", "o"), "d1": (MS.C["M04"], "MIT-BIH", "s")}
    for tag, (col, lab, mk) in styles.items():
        df = pd.read_parquet(Path("results") / tag / "exp_a" / "metrics.parquet")
        xs, ys, sig = [], [], []
        for s in sorted(df.snr_in_target.dropna().unique()):
            t = compare_methods(df[df.snr_in_target == s], "snr_imp_scaled", "M_FE")
            r = t[t.method == "M08"]
            if r.empty:
                continue
            xs.append(float(s)); ys.append(float(r.iloc[0].delta_mean)); sig.append(bool(r.iloc[0].p_holm < 0.05))
        xs, ys, sig = np.array(xs), np.array(ys), np.array(sig)
        ax.plot(xs, ys, color=col, lw=2.6, zorder=3)
        ax.scatter(xs[sig], ys[sig], s=70, color=col, marker=mk, zorder=4, edgecolor=MS.SURFACE, lw=1.2)
        ax.scatter(xs[~sig], ys[~sig], s=70, facecolor=MS.SURFACE, marker=mk, zorder=4, edgecolor=col, lw=1.8)
        k = 3 if tag == "d0" else 1
        ax.annotate(lab, (xs[k], ys[k]), xytext=(0, 12 if tag == "d0" else -26), textcoords="offset points",
                    color=col, fontsize=FS, fontweight="bold", ha="center")
        for i in range(len(xs) - 1):
            if ys[i] > 0 >= ys[i + 1] and tag == "d1":
                zx = xs[i] + (xs[i + 1] - xs[i]) * ys[i] / (ys[i] - ys[i + 1])
                ax.axvspan(zx, xs.max() + 1.5, color=col, alpha=0.08, lw=0, zorder=1)
                ax.axvline(zx, color=col, ls=":", lw=1.6)
                ax.text(zx + 0.4, 0.96, f"약 {zx:.0f} dB 이상:\n공통 전처리가\n더 높음", transform=ax.get_xaxis_transform(),
                        va="top", fontsize=FS, color=col)
    ax.axhline(0, color=MS.INK2, lw=1.1)
    ax.set_xlabel("입력 SNR [dB]")
    ax.set_ylabel("공통 전처리 대비 [dB]")
    ax.set_xlim(-6.5, 21.5)
    ax.text(0.02, 0.04, "채운 표식: 유의(Holm 보정)", transform=ax.transAxes, fontsize=FS_S, color=MS.INK2)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.17)
    _save(fig, "F4", "F4_crossover")


# ------------------------------------------------------------------ F5 칼만 가성 QRS군
def f5_phantom():
    summ = json.loads(Path("results/d1/phantom/summary.json").read_text())
    ex = np.load("results/d1/phantom/example.npz", allow_pickle=True)
    fs = float(ex["fs"])
    t = (np.arange(len(ex["clean"])) - (int(ex["center"]) - int(ex["i0"]))) / fs
    ref = np.asarray(ex["ref_peaks"])
    tol = int(round(0.075 * fs))
    w, h = SIZES["F5"]
    fig = plt.figure(figsize=(w * CM, h * CM))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.55, 1.0], wspace=0.55, hspace=0.25,
                          left=0.17, right=0.97, top=0.96, bottom=0.17)
    rows = [("noisy", "입력"), ("M05", "칼만\n평활기"), ("M06", "잔차\nU-Net")]
    lo, hi = np.percentile(np.r_[ex["clean"], ex["M06"]], [0.5, 99.5])
    pad = 0.35 * (hi - lo)
    for k, (key, lab) in enumerate(rows):
        ax = fig.add_subplot(gs[k, 0])
        ax.plot(t, ex["clean"], color=MS.CLEAN, lw=2.6)
        ax.plot(t, ex[key], color=MS.NOISY if key == "noisy" else MS.OUTK, lw=0.7 if key == "noisy" else 1.1)
        if key != "noisy":
            for d in np.asarray(ex[f"det_{key}"]):
                if ref.size == 0 or np.abs(ref - d).min() > tol:
                    ax.axvspan(t[d] - 0.2, t[d] + 0.2, color=MS.PHANTOM, alpha=0.22, lw=0)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_yticks([]); ax.grid(False)
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=FS)
        if k < 2:
            ax.set_xticklabels([])
    ax.set_xlabel("시간 [s]")
    ax = fig.add_subplot(gs[:, 1])
    ms = [("M_FE", "공통 전처리"), ("M04", "SWT"), ("M05", "칼만 평활기"), ("M06", "잔차 U-Net")]
    thr = str(summ["threshold"])
    vals = [summ["methods"][m]["beatlike"][thr] for m, _ in ms]
    cols = [MS.PHANTOM if m == "M05" else "#a8a59c" for m, _ in ms]
    yy = np.arange(len(ms))
    ax.barh(yy, vals, color=cols, height=0.62)
    for y, v in zip(yy, vals):
        ax.text(v + max(vals) * 0.03, y, f"{v}", va="center", fontsize=FS)
    ax.set_yticks(yy); ax.set_yticklabels([n for _, n in ms], fontsize=FS)
    ax.invert_yaxis(); ax.set_xlim(0, max(vals) * 1.35)
    ax.set_xlabel("가성 QRS군 [건]")
    ax.grid(axis="y", visible=False)
    _save(fig, "F5", "F5_phantom")


# ------------------------------------------------------------------ F6 잡음 종류별 출력
def f6_noise(cache):
    kinds = [("bw", "기저선\n변동"), ("ma", "근전도"), ("impulse", "임펄스")]
    cols = [("noisy", "입력"), ("M_FE", "공통 전처리"), ("M04", "SWT"), ("M08", "딥러닝")]
    w, h = SIZES["F6"]
    fig, axes = plt.subplots(len(kinds), len(cols), figsize=(w * CM, h * CM), sharex="row", sharey="row",
                             gridspec_kw=dict(left=0.12, right=0.995, top=0.91, bottom=0.03, wspace=0.06,
                                              hspace=0.12))
    for r, (kind, klab) in enumerate(kinds):
        d = cache[("mitdb", -5.0, kind)]
        i0, i1 = MS._window(d, 2.0)
        t = (np.arange(i0, i1) - i0) / d["fs"]
        allv = np.concatenate([d["x"][i0:i1], d["y"][i0:i1]] + [d["outs"][m][i0:i1] for m, _ in cols[1:]])
        lo, hi = np.percentile(allv, [0.2, 99.8])
        ylim = (lo - 0.05 * (hi - lo), hi + 0.45 * (hi - lo))   # 위쪽 여백에 dB 표지를 둔다
        for c, (m, mlab) in enumerate(cols):
            ax = axes[r, c]
            ax.plot(t, d["x"][i0:i1], color=MS.CLEAN, lw=2.2)
            sig = d["y"] if m == "noisy" else d["outs"][m]
            ax.plot(t, sig[i0:i1], color=MS.NOISY if m == "noisy" else MS.C.get(m, MS.OUTK), lw=0.9)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            ax.set_ylim(*ylim)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if m != "noisy":
                v = MS._snr_imp(d["x"], d["y"], d["outs"][m], d["fs"])
                ax.text(0.5, 0.99, f"{v:+.0f} dB", transform=ax.transAxes, ha="center", va="top",
                        fontsize=FS, fontweight="bold", color=MS.C.get(m, MS.OUTK))
            if r == 0:
                ax.set_title(mlab, fontsize=FS, color=MS.C.get(m, MS.INK))
            if c == 0:
                ax.set_ylabel(klab, rotation=0, ha="right", va="center", fontsize=FS)
    _save(fig, "F6", "F6_noise")


# ------------------------------------------------------------------ F7 개선 시도 16종
SHORT = {"SWT 전처리 직렬 결합": "SWT 직렬 결합", "웨이블릿 특징 공간 결합": "웨이블릿 특징 공간",
         "CNN + Transformer": "CNN-Transformer", "해상도 유지 (팽창 합성곱)": "시간 해상도 유지",
         "깊이 2배 (파라미터 수 동일)": "깊이 2배", "깊이 4배 (파라미터 수 동일)": "깊이 4배",
         "잔차 블록 4배 (파라미터 1.9배)": "잔차 블록 4배", "정규화 제거 (EDSR 방식)": "정규화 제거",
         "가중치 지수이동평균 (EMA) ※": "가중치 EMA", "손실: MAE와 미분 오차 추가": "손실: MAE · 미분",
         "손실: 무잡음 입력 보존항 추가": "손실: 보존항", "SNR 조건부 특징 변조 (FiLM)": "SNR 조건화",
         "보존항 손실에 SNR 조건화 추가": "보존항 + 조건화",
         "입력 SNR 15~20 dB 전용 학습 모델\n(회수 가능한 개선량의 상한 추정)": "15~20 dB 전용 모델"}


def f7_journey():
    """변경 16종의 점 그림(forest plot). 행 이름을 짧게 하고 범주는 색과 구분선으로 나타낸다."""
    keys = [("fixed_snr_imp_scaled", "전 구간"), ("band_15_20", "15~20 dB 구간")]
    w, h = SIZES["F7"]
    fig, axes = plt.subplots(1, 2, figsize=(w * CM, h * CM), sharey=True,
                             gridspec_kw=dict(left=0.235, right=0.99, top=0.93, bottom=0.13, wspace=0.07))
    ypos, y, last = [], 0.0, None
    for g, *_ in MS.JOURNEY:
        if last is not None and g != last:
            y += 0.6
        ypos.append(y); y += 1.0; last = g
    for ax, (key, title) in zip(axes, keys):
        ax.axvline(0, color=MS.INK2, lw=1.0)
        for (g, lab, sw, arm, base, bsw), yy in zip(MS.JOURNEY, ypos):
            r = MS._paired(sw, arm, base, key, bsw)
            if r is None:
                continue
            m, d, p = r
            sig = bool(p < 0.05)
            if sw == "d41" and bsw == "l6cond":          # 다른 환경 기준 — 같은 환경에서도 버틸 때만 유의(F-57)
                same = MS._paired(sw, arm, base, key, "d40_ref")
                if same is None or abs(same[0]) < 0.1 or np.sign(same[0]) != np.sign(m):
                    sig = False
            c = MS.GROUP_C[g]
            ax.plot([d.min(), d.max()], [yy, yy], color=c, lw=1.6, alpha=0.45, solid_capstyle="round")
            ax.scatter([m], [yy], s=46, color=c if sig else MS.SURFACE, edgecolor=c, lw=1.4, zorder=3)
        ax.set_title(title, fontsize=FS)
        ax.set_xlabel("기준 모델 대비 [dB]")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([SHORT.get(lab, lab) for _, lab, *_ in MS.JOURNEY], fontsize=FS_S)
    for tl, (g, *_ ) in zip(axes[0].get_yticklabels(), MS.JOURNEY):
        tl.set_color(MS.GROUP_C[g] if g == "학습 목표" else MS.INK)
    axes[0].set_ylim(ypos[-1] + 0.7, -0.7)
    _save(fig, "F7", "F7_journey")


# ------------------------------------------------------------------ T1 SWT 임계값
def t1_swt(cache):
    import pywt
    import sys

    from ecgdn.methods.wavelet import mad_sigma, qrs_protection_mask
    sys.path.insert(0, "scripts")
    from run_exp import build_methods  # noqa: PLC0415
    d = cache[("synthetic", 5.0, "awgn")]
    m04 = build_methods({"methods": ["M04"], "frontend": True}, d["tag"])["M04"]
    cfg, fs = m04.cfg, d["fs"]
    x = m04.fe(d["y"], fs)
    i0, _ = MS._window(d, dur=2.56)
    n = 512
    seg = x[i0:i0 + n]
    rp = d["r_peaks"][(d["r_peaks"] >= i0) & (d["r_peaks"] < i0 + n)] - i0
    co = pywt.swt(seg, cfg.wavelet, level=cfg.level)
    det = [c[1] for c in co][::-1]
    sigma = mad_sigma(det[1] if getattr(cfg, "sigma_source", "d1") == "d2" else det[0])
    kv = np.asarray(cfg.k, float)
    g = qrs_protection_mask(n, rp, fs, cfg.protect_ms) if cfg.protect_qrs else np.zeros(n)
    t = np.arange(n) / fs
    j = 2                                                     # D3 (16~31 Hz) — QRS 에너지가 큰 단
    lam = kv[min(j, kv.size - 1)] * sigma * np.sqrt(2 * np.log(n)) * (1 - (1 - cfg.protect_rho) * g)
    out = m04(d["y"], fs, {})[i0:i0 + n]
    w, h = SIZES["T1"]
    fig, axes = plt.subplots(3, 1, figsize=(w * CM, h * CM), sharex=True,
                             gridspec_kw=dict(left=0.2, right=0.99, top=0.98, bottom=0.17, hspace=0.12))
    axes[0].plot(t, seg, color=MS.NOISY, lw=0.8)
    axes[0].set_ylabel("입력", rotation=0, ha="right", va="center")
    axes[1].fill_between(t, -lam, lam, color=MS.C["M04"], alpha=0.25, lw=0)
    axes[1].plot(t, det[j], color=MS.OUTK, lw=0.8)
    axes[1].set_ylabel("계수\n+ 임계값", rotation=0, ha="right", va="center")
    axes[2].plot(t, d["x"][i0:i0 + n], color=MS.CLEAN, lw=2.6)
    axes[2].plot(t, out, color=MS.C["M04"], lw=1.1)
    axes[2].set_ylabel("출력", rotation=0, ha="right", va="center")
    for a in axes:
        a.set_yticks([]); a.grid(False)
    axes[2].set_xlabel("시간 [s]")
    _save(fig, "T1", "T1_swt")


# ------------------------------------------------------------------ T2 칼만 모델
def t2_kalman(cache):
    from ecgdn.methods.kalman_sameni import SameniKalman, _kernel_sum
    d = cache[("mitdb", 5.0, "mixed")]
    k = SameniKalman(smooth=True)
    k(d["y"], d["fs"], {})
    fit = k.last_info["fit"]
    grid = np.asarray(fit.template_grid)
    fig, ax = _fig("T2")
    ax.plot(grid, np.asarray(fit.template), color=MS.CLEAN, lw=3.0, label="평균 박동")
    for a_, b_, t_ in zip(np.asarray(fit.alpha), np.asarray(fit.b), np.asarray(fit.theta)):
        ax.plot(grid, _kernel_sum(grid, np.array([a_]), np.array([b_]), np.array([t_])), color=MS.OUTK,
                lw=0.8, alpha=0.6)
    ax.plot(grid, np.asarray(fit.fitted), color=MS.C["M05"], lw=1.8, label="가우시안 합")
    ax.set_xlabel("심장 위상 θ [rad]")
    ax.set_yticks([]); ax.grid(False)
    ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels(["-π", "0", "π"])
    ax.legend(loc="upper left", frameon=False, fontsize=FS_S)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.97, bottom=0.2)
    _save(fig, "T2", "T2_kalman")


def main() -> int:
    ap = argparse.ArgumentParser(description="판넬(A1) 전용 그림: 판넬 칸 크기 그대로, 그림 속 글자 14 pt 이상")
    ap.add_argument("--only", nargs="*", default=None, help="F4 F5 F6 F7 T1 T2 중 일부만 생성")
    a = ap.parse_args()
    want = set(a.only or ["F4", "F5", "F6", "F7", "T1", "T2"])
    _ko_font()
    ensure_dir(OUT)
    need = set()
    if "F6" in want:
        need |= {("mitdb", -5.0, k) for k in ("bw", "ma", "impulse")}
    if "T1" in want:
        need |= {("synthetic", 5.0, "awgn")}
    if "T2" in want:
        need |= {("mitdb", 5.0, "mixed")}
    cache = {}
    for src, snr, kind in sorted(need):
        print(f"[prepare] {src} {snr:+.0f} dB {kind}")
        cache[(src, snr, kind)] = MS.prepare(src, snr, kind, methods=("M_FE", "M04"))
    if "F4" in want:
        f4_crossover()
    if "F5" in want:
        f5_phantom()
    if "F6" in want:
        f6_noise(cache)
    if "F7" in want:
        f7_journey()
    if "T1" in want:
        t1_swt(cache)
    if "T2" in want:
        t2_kalman(cache)
    save_manifest(OUT, cfg={"panel_figs": sorted(want), "sizes_cm": SIZES},
                  sources=["scripts/make_panel_figs.py", "scripts/make_slides.py"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
