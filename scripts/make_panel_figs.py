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
    "F4": (12.0, 7.8),     # 결과: 입력 SNR 교차점          ┐ 결과 구역 위 줄 = 그림 4 · 5 · 6(왼쪽 37 cm),
    "F6": (12.0, 7.8),     # 결과: 잡음 종류별 출력(그림 5)  │ 아래 줄 = 그림 7(전폭), 오른쪽 18 cm 는 고찰
    "F5": (12.0, 7.8),     # 결과: 칼만 가성 QRS군(그림 6)   ┘
    "F7": (37.0, 13.2),    # 결과: 딥러닝 개선 과정의 단계 흐름(결과 구역 아래 줄 전폭)
    "T1": (13.4, 8.6),     # 이론 카드 ①: SWT 임계값 (입력 | D1~D5 | 출력)
    "T2": (13.4, 8.6),     # 이론 카드 ②: 칼만 모델 (1 주기 모델 | 위상 · 출력)
    "T3": (13.4, 8.6),     # 이론 카드 ③: 잔차 학습 U-Net (높이 = 신호 길이, 겹 = 채널 수)
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
    """PNG(300 dpi)와 편집용 SVG 를 같은 크기로 저장한다.

    SVG 는 글자를 글자로 남긴다(`svg.fonttype: none`). PowerPoint 에서 「도형으로 변환」하면 선 · 글자 · 색을
    하나씩 고칠 수 있다. 글꼴 이름은 판넬을 여는 Windows 에 있는 맑은 고딕을 먼저 쓰도록 바꾼다.
    """
    w, h = SIZES[key]
    fig.set_size_inches(w * CM, h * CM)
    fig.savefig(OUT / f"{name}.png", dpi=300)       # bbox 를 자르지 않는다 — 크기가 곧 판넬 칸이다
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(OUT / f"{name}.svg")
    _svg_font(OUT / f"{name}.svg")
    plt.close(fig)
    print(f"  {OUT / name}.png · .svg  ({w} x {h} cm)")


def _svg_font(path: Path) -> None:
    """SVG 의 한글 글꼴 지정을 「맑은 고딕 → 나눔고딕」 순서로 바꾼다."""
    import re
    src = path.read_text(encoding="utf-8")
    src = re.sub(r"font-family:\s*([^;\"]*)", lambda m: "font-family: 'Malgun Gothic', " + m.group(1)
                 if "NanumGothic" in m.group(1) else m.group(0), src)
    path.write_text(src, encoding="utf-8")


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
                ax.text(zx + 0.35, 0.97, f"약 {zx:.0f} dB\n이상에서\n공통 전처리가\n더 높다",
                        transform=ax.get_xaxis_transform(), va="top", fontsize=FS_S, color=col, linespacing=1.1)
    ax.axhline(0, color=MS.INK2, lw=1.1)
    ax.set_xlabel("입력 SNR [dB]")
    ax.set_ylabel("공통 전처리 대비 [dB]")
    ax.set_xlim(-6.5, 22.8)
    ax.text(0.02, 0.04, "채운 표식: 유의(Holm 보정)", transform=ax.transAxes, fontsize=FS_S, color=MS.INK2)
    fig.subplots_adjust(left=0.165, right=0.98, top=0.97, bottom=0.17)
    _save(fig, "F4", "F4_crossover")


# ------------------------------------------------------------------ F5 칼만 가성 QRS군
def f5_phantom():
    """발표용 S15 의 구성을 유지한다: 왼쪽 = 다섯 출력의 파형과 위양성 검출 위치(음영),
    오른쪽 = 기법별 가성 QRS군 수(진한 막대 MIT-BIH, 옅은 막대 합성 ECG). 제목 · 설명문은 캡션 · 고찰로."""
    summ = {t: json.loads(Path(f"results/{t}/phantom/summary.json").read_text()) for t in ("d1", "d0")
            if Path(f"results/{t}/phantom/summary.json").exists()}
    ex = np.load("results/d1/phantom/example.npz", allow_pickle=True)
    fs = float(ex["fs"])
    t = (np.arange(len(ex["clean"])) - (int(ex["center"]) - int(ex["i0"]))) / fs
    ref = np.asarray(ex["ref_peaks"])
    tol = int(round(0.075 * fs))
    w, h = SIZES["F5"]
    fig = plt.figure(figsize=(w * CM, h * CM))
    rows = [("noisy", "입력"), ("M_FE", "공통 전처리"), ("M04", "SWT"), ("M05", "칼만 평활기"), ("M06", "잔차 U-Net")]
    # 오른쪽 막대는 왼쪽 파형의 2~5 행과 같은 높이에 둔다 — 기법 이름은 왼쪽 행 이름 하나로 읽는다
    gs = fig.add_gridspec(len(rows), 2, width_ratios=[1.6, 1.0], wspace=0.06, hspace=0.12,
                          left=0.22, right=0.985, top=0.97, bottom=0.2)
    lo, hi = np.percentile(np.r_[ex["clean"], ex["M06"]], [0.5, 99.5])
    pad = 0.35 * (hi - lo)
    for k, (key, lab) in enumerate(rows):
        ax = fig.add_subplot(gs[k, 0])
        ax.plot(t, ex["clean"], color=MS.CLEAN, lw=2.2)
        ax.plot(t, ex[key], color=MS.NOISY if key == "noisy" else MS.OUTK, lw=0.6 if key == "noisy" else 0.9)
        if key != "noisy":
            for d in np.asarray(ex[f"det_{key}"]):
                if ref.size == 0 or np.abs(ref - d).min() > tol:
                    ax.axvspan(t[d] - 0.2, t[d] + 0.2, color=MS.PHANTOM, alpha=0.22, lw=0)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_yticks([]); ax.grid(False)
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=FS_S)
        if k < len(rows) - 1:
            ax.set_xticklabels([])
    ax.set_xlabel("시간 [s]")
    ax = fig.add_subplot(gs[1:, 1])
    ms = [("M_FE", "공통 전처리"), ("M04", "SWT"), ("M05", "칼만 평활기"), ("M06", "잔차 U-Net")]
    yy = np.arange(len(ms))
    top = 0
    for tag, off, alpha in (("d1", -0.19, 1.0), ("d0", 0.19, 0.4)):
        if tag not in summ:
            continue
        thr = str(summ[tag]["threshold"])
        vals = [summ[tag]["methods"][m]["beatlike"][thr] for m, _ in ms]
        top = max(top, max(vals))
        ax.barh(yy + off, vals, height=0.36, alpha=alpha,
                color=[MS.PHANTOM if m == "M05" else "#8f8a80" for m, _ in ms])
        for y, v in zip(yy + off, vals):
            ax.text(v + top * 0.03, y, f"{v}", va="center", fontsize=FS_S)
    ax.set_yticks([])
    ax.set_ylim(len(ms) - 0.5, -0.5); ax.set_xlim(0, top * 1.35)
    ax.set_xlabel("가성 QRS군 [건]")
    lg = fig.add_subplot(gs[0, 1]); lg.axis("off")
    lg.text(0.06, 0.5, "진한: MIT-BIH\n옅은: 합성 ECG", transform=lg.transAxes, va="center",
            fontsize=FS_S, color=MS.INK2, linespacing=1.15)
    ax.grid(axis="y", visible=False)
    _save(fig, "F5", "F5_phantom")


# ------------------------------------------------------------------ F6 잡음 종류별 출력
def f6_noise(cache):
    kinds = [("bw", "기저선\n변동"), ("ma", "근전도"), ("impulse", "임펄스")]
    cols = [("noisy", "입력"), ("M_FE", "공통 전처리"), ("M04", "SWT"), ("M08", "딥러닝")]
    w, h = SIZES["F6"]
    fig, axes = plt.subplots(len(kinds), len(cols), figsize=(w * CM, h * CM), sharex="row", sharey="row",
                             gridspec_kw=dict(left=0.14, right=0.995, top=0.91, bottom=0.03, wspace=0.06,
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

# 단계: (번호, 질문, 목적, 시도 목록, 판정, 배경색). 시도 = (이름, 설명, [(sweep, arm, 기준 arm, 기준 sweep, 점 표지)])
# 설명은 「무엇을 바꾸었나」를 한 문장으로 적는다. 판넬을 보는 사람은 M07 · L6 같은 기호를 모른다(사용자 피드백 2026-11-04).
# 판정 문장의 {키} 는 그림을 그릴 때 Holm 보정 p 값으로 채운다. 키 = 「arm__all」(전 구간) · 「arm__hi」(15~20 dB).
# 문안은 개조식 · 말머리 기호 두 개(사용자 지시 2026-11-04). 시도 설명은 「무엇을 바꾸었나」를 명사형으로 적는다.
FLOW = [
    ("②", "구조 변경", "웨이블릿 · 해상도 · 문맥 길이", [
        ("SWT 직렬 결합", "SWT 필터 출력을 U-Net에 바로 연결", [("structure", "m07_l1", "m06_l1", None)]),
        ("웨이블릿 부대역 입력", "SWT로 분해한 대역 계수 6개를 6채널 입력으로 사용",
         [("structure", "m08_l1", "m06_l1", None)]),
        ("다운샘플링 제거", "dilated convolution으로 해상도 유지",
         [("structure", "m10_l1", "m06_l1", None)]),
        ("CNN-Transformer", "어텐션으로 장거리 의존성 학습", [("structure", "m09_l1", "m06_l1", None)]),
    ], "• 성능 향상 구조 없음\n• SWT 직렬 결합: 성능 저하\n   (p_Holm = {m07_l1__all:.2f})", "#f1f0ec"),
    ("③", "모델 규모 변경", "모델 용량 부족 여부 확인", [
        ("채널 수 1/2 · 1/4", "각 층의 채널 수 축소", [("structure", "m06_l1_half", "m06_l1", None),
                                              ("structure", "m06_l1_quarter", "m06_l1", None)]),
        ("깊이 2배 · 4배", "파라미터 수 유지, 잔차 블록 수 증가", [("depth", "m06_l1_deep2", "m06_l1", None),
                                                     ("depth", "m06_l1_deep4", "m06_l1", None)]),
        ("잔차 블록 4배", "단계당 잔차 블록 1개 → 4개(파라미터 1.9배)", [("blocks", "m06_l1_nb4", "m06_l1", None)]),
    ], "• 유의한 차이 없음\n   (p_Holm ≥ 0.05)\n• 용량 부족은 원인이 아님", "#f1f0ec"),
    ("④", "학습 기법 변경", "영상 복원 분야 기법", [
        ("정규화 계층 제거", "특징 값 범위의 유연성 확보 목적(EDSR)", [("d41", "m06_l1_nonorm", "m06_l1", "l6cond")]),
        ("가중치 EMA", "학습 잡음 감소 · 일반화 개선 목적", [("d41", "m06_l1_ema", "m06_l1", "l6cond")]),
    ], "• 성능 향상 없음\n• 정규화 제거: 저하\n   경향(p_Holm = {m06_l1_nonorm__all:.2f})", "#f1f0ec"),
    ("⑤", "원인 분석", "고 SNR 대역 전용 학습", [
        ("15~20 dB 전용 모델", "회수 가능한 개선량의 상한 추정", [("band", "m06_l1_band_hi", "m06_l1", None)]),
    ], "• 15~20 dB: {m06_l1_band_hi__hi_d:+.1f} dB\n   n = 3, p_Holm = {m06_l1_band_hi__hi:.2f}\n"
       "• 전 대역 절충 시사", "#faefd6"),
    ("⑥", "학습 목표 변경", "손실 함수 · 조건 정보 재설정", [
        ("손실: MAE · 차분", "MSE에 MAE · 1차 차분 오차 추가", [("structure", "m06_l3", "m06_l1", None)]),
        ("손실: 무잡음 입력 보존항", "무잡음 입력을 그대로 출력하도록 하는 손실 항", [("l6cond", "m06_l6", "m06_l1", None)]),
        ("SNR 조건화(FiLM)", "실제 입력 SNR을 FiLM으로 잔차 블록에 조건 입력", [("cond", "m06_cond", "m06_l1", None)]),
        ("보존항 + SNR 조건화", "보존항 모델 대비", [("l6cond", "m06_cond_l6", "m06_l6", None)]),
    ], "• 보존항 · SNR 조건화: 성능\n   향상(p_Holm < 0.05)\n• 두 효과는 가산되지 않음", "#e4ecf7"),
]
KEYS = (("fixed_snr_imp_scaled", "all"), ("band_15_20", "hi"))
PH = r"$p_{\mathrm{Holm}}$"                     # 그림 속 p_Holm 은 아래 첨자로


def flow_stats() -> dict:
    """그림 7 의 비교 전체(시도 × 지표)를 한 계열로 Holm 보정한다 (D-44 — 계열을 결과를 본 뒤 좁히지 않는다).

    돌려주는 값: {(arm, base, "all"|"hi"): (평균 Δ, 시드별 Δ, p, p_Holm, 유의)}. 유의 = p_Holm < 0.05.
    정규화 제거 · 가중치 EMA 는 다른 환경의 기준과 짝지었으므로 같은 환경 기준에서도 부호 · 크기가 유지될 때만
    유의로 본다(F-57).
    """
    rows = []
    for _, _, _, tries, _, _ in FLOW:
        for _, _, arms in tries:
            for sw, arm, base, bsw in arms:
                for key, short in KEYS:
                    r = MS._paired(sw, arm, base, key, bsw)
                    if r is not None:
                        rows.append(((arm, base, short), sw, bsw, key, r))
    p = np.array([r[4][2] for r in rows])
    order, m = np.argsort(p), len(p)
    ph, run = np.empty(m), 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * p[k]); ph[k] = min(1.0, run)
    out = {}
    for (k, sw, bsw, key, (mean, d, pv)), h in zip(rows, ph):
        sig = bool(h < 0.05)
        if sw == "d41" and bsw == "l6cond":
            same = MS._paired(sw, k[0], k[1], key, "d40_ref")
            if same is None or abs(same[0]) < 0.1 or np.sign(same[0]) != np.sign(mean):
                sig = False
        out[k] = (mean, d, pv, float(h), sig)
    return out

def _wrap(s: str, cm: float, pt: float) -> str:
    """글자 폭(한글 0.92 em, 그 밖 0.56 em — 나눔고딕 실측)으로 줄을 나눈다. 띄어쓰기 단위로만 끊는다."""
    em = pt / 72 * 2.54
    out, line, wl = [], "", 0.0
    for word in s.split(" "):
        ww = sum(em * 0.92 if "\uac00" <= ch <= "\ud7a3" else em * 0.56 for ch in word)
        sp = em * 0.3 if line else 0.0
        if line and wl + sp + ww > cm:
            out.append(line); line, wl = word, ww
        else:
            line, wl = (line + " " + word if line else word), wl + sp + ww
    out.append(line)
    return "\n".join(out)


def f7_flow():
    """딥러닝 개선 과정을 왼쪽 → 오른쪽 단계로 읽게 한다. 단계마다 위에서 아래로 「질문 · 목적 → 결과 → 시도 설명 → 판정」.

    가운데 그래프는 모든 단계가 **같은 세로축**(기준 모델 대비 변화량)을 쓴다. 그래서 ②~④ 의 점이 0 부근에
    머물다가 ⑤ · ⑥ 에서 위로 올라가는 것이 흐름으로 보인다. 점 옆 문자(a~n)가 아래 설명 줄과 짝이다.
    수치는 모두 같은 자(고정 평가 집합, 같은 시드끼리의 대응 차이, F-41)이다.
    ● 전 구간, ◆ 15~20 dB 구간(측정한 시도만), 채움 = p < 0.05, 세로선 = 시드 범위.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    W, H = SIZES["F7"]
    fig = plt.figure(figsize=(W * CM, H * CM))
    fx, fy = (lambda x: x / W), (lambda y: y / H)
    ln = FS_S / 72 * 2.54 * 1.22                    # 한 줄 높이 [cm]
    blue, amber, bad = MS.GROUP_C["학습 목표"], "#9a6b12", "#9b3b32"
    gap, w0 = 0.45, 4.4
    widths = [7.0, 6.2, 5.1, 5.25, 6.8]          # w0 + 합 + 간격 5 개 = 37.0 cm
    assert abs(w0 + sum(widths) + 5 * gap - W) < 1e-6
    top, bot = H - 0.1, 0.1
    c_top, c_bot = top - 1.5, top - 4.75            # 그래프 칸 [cm]
    st = flow_stats()
    pv = {f"{a}__{k}": v[3] for (a, _, k), v in st.items()}
    pv.update({f"{a}__{k}_d": v[0] for (a, _, k), v in st.items()})
    v_lines = max(v[4].count("\n") + 1 for v in FLOW)
    v_top = bot + 0.2 + v_lines * ln * 1.0         # 판정 문장(개조식) 칸의 위쪽
    YR = (-2.9, 6.0)
    a = MS._sweep("l6cond")
    v = [a[k]["band_15_20"] for k in a if k[0] == "m06_l1" and "band_15_20" in a[k]]
    base_band = float(np.mean(v)) if v else float("nan")

    def panel(x, w, col):
        fig.patches.append(FancyBboxPatch((fx(x), fy(bot)), fx(w), fy(top - bot), transform=fig.transFigure,
                                          boxstyle="round,pad=0,rounding_size=0.012", fc=col, ec="none",
                                          zorder=-5))   # 축(zorder 0)보다 먼저 그린다

    # 공통 그래프: 가로 = 판넬 cm 좌표, 세로 = dB. 축 배경은 투명하게 두어 단계 칸 색이 비친다.
    ax = fig.add_axes([fx(w0 - 0.95), fy(c_bot), fx(W - (w0 - 0.95)), fy(c_top - c_bot)])
    ax.patch.set_alpha(0)
    ax.set_xlim(w0 - 0.95, W); ax.set_ylim(*YR)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(MS.INK2)
    ax.set_xticks([]); ax.set_yticks([-2, 0, 2, 4]); ax.set_yticklabels(["-2", "0", "+2", "+4"])
    ax.tick_params(axis="y", labelsize=FS_S, colors=MS.INK2, length=3)
    ax.grid(False)
    for t in (-2, 2, 4):
        ax.plot([w0 + gap, W], [t, t], color="#d9d6ce", lw=0.7, zorder=0)
    ax.plot([w0 + gap, W], [0, 0], color=MS.INK2, lw=1.1, zorder=1)

    # ① 기준 모델 + 범례. 세로축 이름은 축 왼쪽에 가로로 쓴다(세로로 돌리면 칸 높이에 들어가지 않는다).
    panel(0.0, w0, "#f1f0ec")
    fig.text(fx(0.2), fy(top - 0.5), "① 기준 모델", fontsize=FS, fontweight="bold", va="center")
    fig.text(fx(0.2), fy(top - 1.08), "잔차 U-Net · MSE 손실", fontsize=FS_S, va="center", color=MS.INK2)
    fig.text(fx(0.2), fy(c_top - 0.05), "기준 모델\n대비 SNR\n개선량\n변화 [dB]", fontsize=FS_S, va="top",
             color=MS.INK2, linespacing=1.1)
    t0 = "• 15~20 dB 입력:\n   SNR 개선량 음수"
    fig.text(fx(0.2), fy(c_bot - 0.28), t0, fontsize=FS_S, va="top", linespacing=1.1, color=MS.INK2)
    nl = t0.count("\n") + 1
    fig.text(fx(0.55), fy(c_bot - 0.28 - (nl * 0.95 + 0.1) * ln), f"{base_band:+.1f} dB", fontsize=FS + 4,
             fontweight="bold", color=bad, va="top")
    ly = c_bot - 0.28 - (nl * 0.95 + 2.1) * ln
    fig.text(fx(0.2), fy(ly), "범례", fontsize=FS_S, fontweight="bold", va="center")
    items = [("o", True, "전 구간 평균"), ("D", True, "15~20 dB 구간"), ("o", False, "p_Holm ≥ 0.05"),
             ("|", True, "세로선: 시드별 범위")]
    for i, (mk, fill, lab) in enumerate(items):
        yy = ly - (i + 1) * ln
        fig.add_artist(Line2D([fx(0.38)], [fy(yy)], marker=mk, ms=7 if mk == "o" else 6, ls="none",
                              mfc=MS.INK2 if fill else MS.SURFACE, mec=MS.INK2, mew=1.3, transform=fig.transFigure))
        fig.text(fx(0.7), fy(yy), lab.replace("p_Holm", PH), fontsize=FS_S, va="center", color=MS.INK2)

    letters = iter("abcdefghijklmnopqrstuvwxyz")
    x = w0 + gap
    for (no, q, purpose, tries, verdict, col), w in zip(FLOW, widths):
        panel(x, w, col)
        fig.patches.append(FancyArrowPatch((fx(x - gap + 0.02), fy(top - 0.5)), (fx(x - 0.02), fy(top - 0.5)),
                                           transform=fig.transFigure, arrowstyle="-|>", mutation_scale=13,
                                           color=MS.INK2, lw=1.4))
        hc = blue if no == "⑥" else (amber if no == "⑤" else MS.INK)
        dc = blue if no == "⑥" else (amber if no == "⑤" else "#6f6d68")
        fig.text(fx(x + 0.2), fy(top - 0.5), f"{no} {q}", fontsize=FS, fontweight="bold", va="center", color=hc)
        fig.text(fx(x + 0.2), fy(top - 1.08), purpose, fontsize=FS_S, va="center", color=MS.INK2)
        # 그래프: 시도마다 칸 안에 고르게 놓는다. 한 시도에 점이 둘이면(½ · ¼ 등) 나란히 둔다.
        slots = [len(arms) for _, _, arms in tries]
        n = sum(slots)
        xs = x + 0.5 + (np.arange(n) + 0.5) * (w - 1.0) / n
        k, y = 0, c_bot - 0.28
        for name, desc, arms in tries:
            let = next(letters)
            x_let, hi_let = [], -9.0
            for arm_ in arms:
                sw, arm, base, bsw = arm_
                xc, hi = xs[k], -9.0
                for key, mk, ms, dx in (("fixed_snr_imp_scaled", "o", 46, -0.13), ("band_15_20", "D", 36, 0.13)):
                    r = st.get((arm, base, "all" if key == "fixed_snr_imp_scaled" else "hi"))
                    if r is None:
                        continue
                    m, d, _, _, sig = r
                    ax.plot([xc + dx, xc + dx], [d.min(), d.max()], color=dc, lw=1.6, alpha=0.45,
                            solid_capstyle="round", zorder=2)
                    ax.scatter([xc + dx], [m], s=ms, marker=mk, color=dc if sig else MS.SURFACE, edgecolor=dc,
                               lw=1.4, zorder=3, clip_on=False)
                    hi = max(hi, d.max())
                x_let.append(xc); hi_let = max(hi_let, hi)
                k += 1
            # 문자는 시도마다 하나 — 점이 둘이면 두 점의 가운데 위에 둔다
            ax.text(float(np.mean(x_let)), hi_let + 0.25, let, ha="center", va="bottom", fontsize=FS_S, color=hc,
                    fontweight="bold")
            # 설명 줄: 「문자 이름」(굵게) 다음 줄에 설명
            fig.text(fx(x + 0.2), fy(y), f"{let}  {name}", fontsize=FS_S, fontweight="bold", va="top",
                     color=hc if no in "⑤⑥" else MS.INK)
            dtxt = _wrap(desc, w - 0.75, FS_S)
            fig.text(fx(x + 0.55), fy(y - ln * 0.95), dtxt, fontsize=FS_S, va="top", color=MS.INK2, linespacing=1.1)
            y -= ln * (0.95 + 0.92 * (dtxt.count("\n") + 1)) + 0.04
            assert y > v_top + 0.1, f"{no} 칸의 글이 판정 문장과 겹친다 (y = {y:.2f} cm)"
        fig.text(fx(x + 0.2), fy(bot + 0.2), verdict.format(**pv).replace("p_Holm", PH), fontsize=FS_S, fontweight="bold", va="bottom",
                 linespacing=1.0, color=hc if no in "⑤⑥" else MS.INK)
        x += w + gap
    _save(fig, "F7", "F7_flow")


# ------------------------------------------------------------------ T1 SWT 임계값
def t1_swt(cache):
    """입력(왼쪽) → 세부 계수 D1~D5 와 단별 임계값(가운데) → 출력(오른쪽).

    「주파수 대역으로 나누어 대역마다 다른 임계값을 적용한 뒤 합친다」가 그림의 흐름으로 읽히게 한다.
    실험과 같은 M04 인스턴스(합성 ECG 튜닝값)의 λ_j 와 QRS 완화 창을 그대로 그린다.
    """
    import sys

    import pywt
    from matplotlib.patches import Patch

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
    det = [c[1] for c in pywt.swt(seg, cfg.wavelet, level=cfg.level)][::-1]          # D1..D5
    sigma = mad_sigma(det[1] if getattr(cfg, "sigma_source", "d1") == "d2" else det[0])
    kv = np.asarray(cfg.k, float)
    g = qrs_protection_mask(n, rp, fs, cfg.protect_ms) if cfg.protect_qrs else np.zeros(n)
    out = m04(d["y"], fs, {})[i0:i0 + n]
    t = np.arange(n) / fs
    band = ["62~125 Hz", "31~62 Hz", "16~31 Hz", "8~16 Hz", "4~8 Hz"]
    w, h = SIZES["T1"]
    fig = plt.figure(figsize=(w * CM, h * CM))
    # 칸: 입력 | 화살표 | (대역 표지) 세부 계수 | 화살표 | 출력. 대역 표지는 계수 칸의 왼쪽 여백에 놓인다.
    gs = fig.add_gridspec(5, 5, width_ratios=[1.0, 0.95, 1.5, 0.32, 1.0], left=0.02, right=0.985, top=0.88,
                          bottom=0.14, wspace=0.0, hspace=0.2)
    qrs = lambda a: [a.axvspan(t[max(r - 6, 0)], t[min(r + 6, n - 1)], color="#ddd8f0", lw=0, zorder=0)  # noqa: E731
                     for r in rp]
    ai = fig.add_subplot(gs[1:4, 0])
    ai.plot(t, seg, color=MS.NOISY, lw=0.7)
    ai.set_title("입력", fontsize=FS)
    ao = fig.add_subplot(gs[1:4, 4])
    ao.plot(t, d["x"][i0:i0 + n], color=MS.CLEAN, lw=2.4)
    ao.plot(t, out, color=MS.C["M04"], lw=1.0)
    ao.set_title("출력", fontsize=FS)
    for a in (ai, ao):
        qrs(a)
    for j in range(cfg.level):
        a = fig.add_subplot(gs[j, 2])
        lam = kv[min(j, kv.size - 1)] * sigma * np.sqrt(2 * np.log(n)) * (1 - (1 - cfg.protect_rho) * g)
        qrs(a)
        a.fill_between(t, -lam, lam, color=MS.C["M04"], alpha=0.28, lw=0)
        a.plot(t, det[j], color=MS.OUTK, lw=0.7)
        a.set_ylabel(f"D{j + 1}\n{band[j]}", rotation=0, ha="right", va="center", fontsize=FS_S - 1,
                     labelpad=4)
        if j == 0:
            a.set_title("대역별 세부 계수", fontsize=FS_S)
    for a in fig.axes:
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        for sp in a.spines.values():
            sp.set_color("#d6d3cb")
    from matplotlib.patches import FancyArrowPatch
    pi, po = ai.get_position(), ao.get_position()
    pm = fig.axes[2].get_position()
    for x0, x1 in ((pi.x1 + 0.008, pi.x1 + 0.045), (pm.x1 + 0.012, po.x0 - 0.012)):   # 분해 → 재구성 흐름
        fig.add_artist(FancyArrowPatch((x0, 0.51), (x1, 0.51), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=14, color=MS.INK2))
    fig.legend(handles=[Patch(color=MS.C["M04"], alpha=0.35, label="임계값(이하 계수 제거)"),
                        Patch(color="#ddd8f0", label="QRS 구간(임계값 완화)")],
               loc="lower center", ncol=2, frameon=False, fontsize=FS_S - 1, bbox_to_anchor=(0.5, -0.01))
    _save(fig, "T1", "T1_swt")


# ------------------------------------------------------------------ T2 칼만 모델
def t2_kalman(cache):
    """McSharry 모델의 원리: 1 주기 = 위상 θ 위 가우시안의 합(왼쪽), θ(t)는 검출된 R파마다 0 으로
    재설정되고(오른쪽 위), 그 모델로 칼만 평활기가 출력을 만든다(오른쪽 아래)."""
    from ecgdn.eval.rpeak import detect_rpeaks
    from ecgdn.methods.kalman_sameni import SameniKalman, _kernel_sum
    d = cache[("mitdb", 5.0, "mixed")]
    k = SameniKalman(smooth=True)
    out = k(d["y"], d["fs"], {})
    info, fs = k.last_info, d["fs"]
    fit = info["fit"]
    grid = np.asarray(fit.template_grid)
    w, h = SIZES["T2"]
    fig = plt.figure(figsize=(w * CM, h * CM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], left=0.035, right=0.985, top=0.9, bottom=0.17,
                          wspace=0.32, hspace=0.42)
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(grid, np.asarray(fit.template), color=MS.CLEAN, lw=2.8, label="평균 박동")
    for a_, b_, t_ in zip(np.asarray(fit.alpha), np.asarray(fit.b), np.asarray(fit.theta)):
        ax.plot(grid, _kernel_sum(grid, np.array([a_]), np.array([b_]), np.array([t_])), color=MS.OUTK,
                lw=0.7, alpha=0.6)
    ax.plot(grid, np.asarray(fit.fitted), color=MS.C["M05"], lw=1.6, label="가우시안 합")
    ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels(["-π", "0", "π"])
    ax.set_xlabel("위상 θ"); ax.set_title("1주기 모델", fontsize=FS)
    ax.legend(loc="upper left", frameon=False, fontsize=FS_S - 1, handlelength=1.2)
    i0, i1 = MS._window(d, dur=3.2)
    tt = np.arange(i0, i1) / fs
    x_in = k.fe(d["y"], fs) if k.fe is not None else d["y"]
    rp = np.asarray(detect_rpeaks(x_in, fs))
    rp = rp[(rp >= i0) & (rp < i1)]
    a1 = fig.add_subplot(gs[0, 1])
    a1.plot(tt, np.asarray(info["theta"])[i0:i1], color=MS.C["M05"], lw=1.4)
    a1.set_title("위상 θ(t), 점선: R파 검출", fontsize=FS_S)
    a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    a2.plot(tt, d["x"][i0:i1], color=MS.CLEAN, lw=2.4)
    a2.plot(tt, out[i0:i1], color=MS.C["M05"], lw=1.1)
    a2.set_title("출력", fontsize=FS_S, pad=2)
    a2.set_xlabel("시간 [s]")
    for a_ in (a1, a2):
        for r in rp:
            a_.axvline(r / fs, color=MS.PHANTOM, lw=0.9, ls="--")
    for a_ in fig.axes:
        a_.set_yticks([]); a_.grid(False)
    a1.tick_params(labelbottom=False)
    _save(fig, "T2", "T2_kalman")


# ------------------------------------------------------------------ T3 잔차 학습 U-Net
def t3_unet():
    """상자 높이 = 신호 길이(1024 → 64), 겹친 상자 수 = 채널 수가 늘어나는 정도(단마다 한 겹씩).
    점선 호 = 스킵 연결, 아래 경로 = 출력단 잔차 연결(출력 = 입력 - 추정 잡음).
    채널 수 · 신호 길이는 체크포인트에서 읽는다."""
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

    from ecgdn.methods.dl_wrapper import load_checkpoint
    model, _ = load_checkpoint("results/d1/m06_l1/best.pt")
    chs = [int(c) for c in model.chs]
    L, k = 1024, len(chs) - 1
    w, h = SIZES["T3"]
    fig = plt.figure(figsize=(w * CM, h * CM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    yc, H0, sw, off = 5.0, 4.2, 0.42, 0.09            # 중심선, 최대 높이, 상자 폭, 겹 간격 [cm]
    hgt = lambda i: H0 * (0.5 ** i) ** 0.6           # noqa: E731  — 길이가 반이 될 때마다 줄어든다
    cols = [("E", i) for i in range(k)] + [("B", k)] + [("D", i) for i in reversed(range(k))]
    x0, step = 1.55, 0.95
    pos = {}

    def stack(x, i, fc, ec):
        hh = hgt(i)
        for s_ in range(i + 1):                       # 단이 깊을수록(채널이 많을수록) 한 겹씩 더 쌓는다
            ax.add_patch(Rectangle((x + s_ * off, yc - hh / 2 + s_ * off), sw, hh, fc=fc, ec=ec, lw=0.8,
                                   zorder=10 - s_))
        return x + i * off + sw, hh

    for c, (kind, i) in enumerate(cols):
        x = x0 + c * step
        fc, ec = (("#c9d6e8", "#5b86bd") if kind != "D" else ("#cdeee1", "#3f9a77"))
        xr, hh = stack(x, i, fc, ec)
        pos[(kind, i)] = (x, xr, hh)
        if kind != "D":                                # 디코더는 같은 단의 인코더와 크기가 같아 표지를 생략한다
            ax.text((x + xr) / 2, yc - hh / 2 - 0.18, f"{chs[i]}ch\n{L >> i}", ha="center", va="top",
                    fontsize=FS_S - 2, color=MS.INK2, linespacing=1.05)
    for c in range(len(cols) - 1):                     # 흐름 화살표
        a, b = pos[cols[c]], pos[cols[c + 1]]
        ax.add_patch(FancyArrowPatch((a[1] + 0.03, yc), (b[0] - 0.03, yc), arrowstyle="-|>", mutation_scale=8,
                                     color=MS.INK2, lw=0.8, zorder=12))
    for i in range(k):                                 # 스킵 연결 — 같은 길이의 인코더 · 디코더를 잇는 점선 호
        e, dd = pos[("E", i)], pos[("D", i)]
        top = yc + hgt(i) / 2 + 0.15 + i * off
        ya = yc + H0 / 2 + 0.3 + (k - 1 - i) * 0.25
        xs = [(e[0] + e[1]) / 2, (e[0] + e[1]) / 2, (dd[0] + dd[1]) / 2, (dd[0] + dd[1]) / 2]
        ax.plot(xs, [top, ya, ya, top], color="#8f8a80", lw=0.9, ls="--")
    # 입력 · 추정 잡음 · 출력단 잔차
    ax.add_patch(FancyBboxPatch((0.25, yc - 0.55), 1.05, 1.1, boxstyle="round,pad=0.02", fc="#ece9e1",
                                ec="#8f8a80", lw=0.8))
    ax.text(0.775, yc, "입력\ny", ha="center", va="center", fontsize=FS_S)
    ax.add_patch(FancyArrowPatch((1.32, yc), (x0 - 0.04, yc), arrowstyle="-|>", mutation_scale=8, color=MS.INK2))
    xl = pos[cols[-1]][1]
    ax.add_patch(FancyBboxPatch((xl + 0.25, yc - 0.7), 1.6, 1.4, boxstyle="round,pad=0.02", fc="#f6c9c9",
                                ec="#c25b5b", lw=0.8))
    ax.text(xl + 1.05, yc + 0.3, "추정 잡음", ha="center", va="center", fontsize=FS_S - 2)
    ax.text(xl + 1.05, yc - 0.28, r"$\hat{n}$", ha="center", va="center", fontsize=FS)
    ax.add_patch(FancyArrowPatch((xl + 0.03, yc), (xl + 0.23, yc), arrowstyle="-|>", mutation_scale=8,
                                 color=MS.INK2))
    cx, cy = xl + 1.05, 1.5
    ax.add_patch(Circle((cx, cy), 0.28, fc="white", ec=MS.INK2, lw=1.0))
    ax.text(cx, cy, "-", ha="center", va="center", fontsize=FS, fontweight="bold")
    ax.add_patch(FancyArrowPatch((cx, yc - 0.72), (cx, cy + 0.3), arrowstyle="-|>", mutation_scale=8,
                                 color=MS.INK2))
    ax.plot([0.775, 0.775, cx - 0.3], [yc - 0.57, cy, cy], color=MS.INK2, lw=1.0)
    ax.add_patch(FancyArrowPatch((cx - 0.6, cy), (cx - 0.29, cy), arrowstyle="-|>", mutation_scale=8,
                                 color=MS.INK2))
    ax.text(6.2, cy + 0.12, "출력단 잔차 연결", ha="center", va="bottom", fontsize=FS_S - 1, color=MS.INK2)
    ax.text(cx, cy - 0.42, r"출력 $\hat{x} = y - \hat{n}$", ha="center", va="top", fontsize=FS_S)
    ax.text(0.25, 0.2, "상자 높이: 신호 길이 · 겹친 상자 수: 채널 수\n점선: 스킵 연결", ha="left", va="bottom",
            linespacing=1.15,
            fontsize=FS_S - 1, color=MS.INK2)
    _save(fig, "T3", "T3_unet")


def main() -> int:
    ap = argparse.ArgumentParser(description="판넬(A1) 전용 그림: 판넬 칸 크기 그대로, 그림 속 글자 14 pt 이상")
    ap.add_argument("--only", nargs="*", default=None, help="F4 F5 F6 F7 T1 T2 T3 중 일부만 생성")
    a = ap.parse_args()
    want = set(a.only or ["F4", "F5", "F6", "F7", "T1", "T2", "T3"])
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
        f7_flow()
    if "T1" in want:
        t1_swt(cache)
    if "T2" in want:
        t2_kalman(cache)
    if "T3" in want:
        t3_unet()
    save_manifest(OUT, cfg={"panel_figs": sorted(want), "sizes_cm": SIZES},
                  sources=["scripts/make_panel_figs.py", "scripts/make_slides.py"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
