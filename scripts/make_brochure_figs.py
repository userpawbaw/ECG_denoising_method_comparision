"""브로셔(학과 양식, 19.0 x 26.8 cm 한 쪽) 전용 그림 — 브로셔에 놓일 크기 그대로 그린다.

    python scripts/make_brochure_figs.py            # 전부
    python scripts/make_brochure_figs.py --only B2

산출: `results/panel/figs/B*.png` (300 dpi). `build_panel.build_brochure` 가 `SIZES` 의 폭으로 축소 없이 놓는다.

구성은 `docs/38_exhibition_story_plan.md` §6 (D-45): 깊게 판 꼭지 = 실시간 공통 전처리 + 장치 연결(B2),
보조 꼭지 = 칼만 필터의 가성 QRS군(B3), 시스템 구성도(B1). 글자는 최소로 두고 설명은 브로셔 캡션이 맡는다.

실시간 측정 화면 캡처는 사용자 본인의 심전도라 `data/arduino/captures/`(git 제외)에서 읽는다. 없으면 그 칸을 빈 틀로 둔다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np

import make_slides as MS
from ecgdn.utils import ensure_dir

OUT = Path("results/panel/figs")
CAP = Path("results/panel/captures")
LIVE = Path("data/arduino/captures")
CM = 1 / 2.54
FS = 7.5            # 그림 속 기본 글자 [pt] — 최소 7 pt (사용자 지시 2026-11-05). 축소 없이 놓으므로 인쇄 크기 그대로
FS_S = 7.0

SIZES = {
    "B1": (16.0, 3.9),     # 2. 작품 개요 — 시스템 구성도
    "B2": (16.0, 4.4),     # 3. 작품 내용 — 주 꼭지: 실시간 공통 전처리
    "B3": (7.8, 2.9),      # 3. 작품 내용 — 칼만 필터의 가성 QRS군
    "B4": (7.8, 3.3),      # 3. 작품 내용 — 실시간 처리: 인과 필터의 QRS 직후 왜곡(B2 ① 만, 사용자 결정 2026-11-05)
    "N1": (7.9, 5.2),      # 잡음 종류별 비교 — 딥러닝이 우세한 잡음(근전도 · 임펄스)
    "N2": (7.9, 5.2),      # 잡음 종류별 비교 — 고전 처리가 우세한 잡음(기저선 변동 · 전원선 간섭)
    "B5": (8.0, 4.6),      # 2. 작품 개요 — 비교 실험 구성(브로셔 폭으로 다시 그림)
}
# 잡음 종류별 실험(EXP-B)과 같은 조건 — MIT-BIH 실측 기록, 입력 SNR +10 dB (판정판 S17 과 같은 원천)
EXPB_SNR = 10.0
NOISE_KO = {"ma_synth": "근전도", "impulse": "임펄스", "bw_synth": "기저선 변동", "pli": "전원선 간섭"}
N_COLS = [("noisy", "입력"), ("M_FE", "전처리"), ("M04", "SWT"), ("M08", "딥러닝")]
FE_COL = {"오프라인 영위상": "#9a978f", "인과 o1 0.5 Hz": "#eb6834",
          "블록 영위상 0.5 s": "#2a78d6", "중앙값 200+600 ms": "#1baf7a"}
FE_LAB = {"오프라인 영위상": "오프라인 영위상", "인과 o1 0.5 Hz": "인과 필터",
          "블록 영위상 0.5 s": "블록 영위상", "중앙값 200+600 ms": "중앙값 필터"}
# 추가 지연 [ms] — 실시간 브리지 설정(보고서 표 5.49, make_slides.s12_fe_tradeoff 와 같은 값)
FE_LAT = {"오프라인 영위상": None, "인과 o1 0.5 Hz": 0, "블록 영위상 0.5 s": 548, "중앙값 200+600 ms": 424}


def _ko_font():
    """한글 글꼴(NanumGothic 등) 선택, 음수 눈금 깨짐 방지, 브로셔용 글자 크기."""
    MS.slide_style()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": FS, "axes.labelsize": FS,
                         "xtick.labelsize": FS_S, "ytick.labelsize": FS_S, "axes.titlesize": FS,
                         "legend.fontsize": FS_S, "axes.linewidth": 0.7})


def _save(fig, key, name):
    w, h = SIZES[key]
    fig.set_size_inches(w * CM, h * CM)
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  {OUT / name}.png  ({w} x {h} cm)")


def _img(path: Path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB")) if path.exists() else None


# ------------------------------------------------------------------ B1 시스템 구성도
def b1_system():
    """위 줄 = 실시간 측정 경로(전극 → AFE → 보드 → 시리얼 → PC 브리지 → 화면), 아래 줄 = 비교 실험과 전시 GUI.

    두 줄이 전시 GUI 에서 만난다(실시간 탭). AFE 는 다른 조원의 담당이라 블록 이름만 둔다(D-45 보완).
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    W, H = SIZES["B1"]
    fig = plt.figure(figsize=(W * CM, H * CM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    def box(x, y, w, h, head, sub, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h - 0.2, head, ha="center", va="top", fontsize=FS, fontweight="bold")
        ax.text(x + w / 2, y + h - 0.62, sub, ha="center", va="top", fontsize=FS_S, color=MS.INK2, linespacing=1.15)

    def arrow(x0, y0, x1, y1, text=None):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, color=MS.INK2, lw=0.8))
        if text:
            ax.text((x0 + x1) / 2, y0 + 0.08, text, ha="center", va="bottom", fontsize=FS_S - 0.6, color=MS.INK2)

    top, bh = 2.05, 1.65
    xs = [0.05, 2.15, 4.25, 6.5, 9.05]
    ws = [1.75, 1.75, 1.9, 2.2, 2.7]
    box(xs[0], top, ws[0], bh, "전극", "피부 부착 전극", "#ece9e1", "#8f8a80")
    box(xs[1], top, ws[1], bh, "아날로그 AFE", "증폭 · 대역 제한\n(조원 설계)", "#ece9e1", "#8f8a80")
    box(xs[2], top, ws[2], bh, "Arduino Uno", "ADC 10 bit\n500 Hz 표본화", "#ece9e1", "#8f8a80")
    box(xs[3], top, ws[3], bh, "시리얼 전송", "115200 baud\n순번 · 검사합", "#f3e6cf", "#b39257")
    box(xs[4], top, ws[4], bh, "PC 브리지", "250 Hz 변환\n공통 전처리 3 방식\nSWT · 칼만 · 딥러닝", "#d5e4f7", "#5b86bd")
    for i in range(4):
        arrow(xs[i] + ws[i] + 0.03, top + bh / 2, xs[i + 1] - 0.03, top + bh / 2)
    # 아래 줄: 공개 데이터 → 비교 실험 → 전시 GUI
    bot = 0.1
    box(0.05, bot, 2.9, bh, "공개 데이터", "합성 ECG · MIT-BIH\n실측 잡음(NSTDB)", "#ece9e1", "#8f8a80")
    box(3.35, bot, 3.4, bh, "비교 실험", "잡음 7종 × 입력 SNR 7단계\n기법 4계열 · 지표 4범주", "#e6e5f4", "#6f6aa8")
    box(7.15, bot, 4.6, bh, "전시 GUI", "도입부(잡음별 3D 장면) · 실험실\n전체 근거 · 실시간 측정 탭", "#cdeee1", "#3f9a77")
    arrow(2.98, bot + bh / 2, 3.32, bot + bh / 2)
    arrow(6.78, bot + bh / 2, 7.12, bot + bh / 2)
    arrow(xs[4] + ws[4] / 2, top - 0.03, xs[4] + ws[4] / 2, bot + bh + 0.03)
    ax.text(xs[4] + ws[4] / 2 + 0.08, (top + bot + bh) / 2, "실시간 스트림", fontsize=FS_S - 0.6, color=MS.INK2,
            va="center")
    # 오른쪽 끝: 두 화면 캡처
    for img, (y0, y1), lab in ((_img(LIVE / "live_02_split_t32.png"), (top, top + bh), "실시간 측정 화면"),
                               (_img(CAP / "lab_01_compare.png"), (bot, bot + bh), "실험실 화면")):
        x0, x1 = 12.05, W - 0.05
        if img is None:
            ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0", fc="#f1f0ec", ec="#c9c5bb"))
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"[{lab}]", ha="center", va="center", fontsize=FS_S)
        else:
            ih, iw = img.shape[:2]
            hh = y1 - y0; ww = min(x1 - x0, hh * iw / ih)
            ax.imshow(img, extent=(x1 - ww, x1, y0, y1), aspect="auto", zorder=2)
            ax.text(x1 - ww - 0.08, (y0 + y1) / 2, lab, ha="right", va="center", fontsize=FS_S - 0.6,
                    color=MS.INK2, rotation=90)
    _save(fig, "B1", "B1_system")


def b5_pipeline():
    """비교 실험 구성(판넬 그림 1 과 같은 구성)을 브로셔 폭 8 cm 로 그린다 — 판넬 도형을 줄이면 글자가 5 pt 아래가 된다."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    W, H = SIZES["B5"]
    fig = plt.figure(figsize=(W * CM, H * CM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, bold=False, dash=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=fc, ec=ec, lw=0.6,
                                    ls="--" if dash else "-"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=FS_S, linespacing=1.05,
                fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, color=MS.INK2, lw=0.6))

    y1, h1 = 3.35, 1.15
    box(0.05, y1, 2.45, h1, "합성 ECG\nMIT-BIH 실측 기록", "#ece9e1", "#8f8a80")
    box(2.85, y1, 2.35, h1, "잡음 7종 ×\n입력 SNR 7단계", "#f3e6cf", "#b39257")
    box(5.55, y1, 2.4, h1, "공통 전처리\n0.5~100 Hz · 노치", "#f6d6e3", "#b5718d")
    arrow(2.52, y1 + h1 / 2, 2.83, y1 + h1 / 2); arrow(5.22, y1 + h1 / 2, 5.53, y1 + h1 / 2)
    y2, h2 = 1.75, 1.0
    items = [("고전\n필터", "#d5e4f7", "#5b86bd"), ("SWT\n임계값", "#fbdccc", "#c77c55"), ("칼만\n필터", "#fbecc2", "#b8962e"),
             ("딥러닝", "#cdeee1", "#3f9a77"), ("이상적\n상한", "#e4e2dc", "#8f8a80")]
    bw = (W - 0.1 - 4 * 0.12) / 5
    for i, (t, fc, ec) in enumerate(items):
        box(0.05 + i * (bw + 0.12), y2, bw, h2, t, fc, ec, dash=(i == 4))
    arrow(6.75, y1 - 0.02, 6.75, y2 + h2 + 0.02)
    y3, h3 = 0.1, 1.15
    box(0.9, y3, 3.0, h3, "지표 4범주\n재구성 · 형태 · 검출 · 주파수", "#e6e5f4", "#6f6aa8")
    box(4.1, y3, 3.0, h3, "가성 파형 생성 검증\n박동 삭제 · 무수축 · PVC", "#e6e5f4", "#6f6aa8")
    arrow(W / 2, y2 - 0.02, W / 2, y3 + h3 + 0.02)
    _save(fig, "B5", "B5_pipeline")


# ------------------------------------------------------------------ B2 실시간 공통 전처리
def _beat_avg(v, r, fs, pre=0.25, post=0.6):
    a, b = int(pre * fs), int(post * fs)
    segs = [v[i - a:i + b] for i in r if i - a >= 0 and i + b <= v.size]
    m = np.mean(segs, axis=0)
    return np.arange(-a, b) / fs, m - np.median(m[: int(0.1 * fs)])


def b2_realtime():
    """세 칸 — ① 인과 필터가 QRS 직후에 남기는 언더슈트(MIT-BIH 평균 박동), ② 평탄도 × ST 충실도
    (합성 ECG 8 기록, 참값 기준 — 왼쪽 아래 «둘 다 좋음» 이 비어 있다), ③ 실보드 화면과 실측 지연의 분해.

    ① ② 의 값은 `audit_fe_metrics` 의 현재 구현에서 다시 계산한다(보고서 표 5.47 은 5.11.5 의 결함 수정 전 값이라
    블록 영위상 수치가 다르다 — docs/38 §10).
    """
    import pandas as pd

    import audit_fe_metrics as A
    W, H = SIZES["B2"]
    fig = plt.figure(figsize=(W * CM, H * CM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.32], left=0.065, right=0.995, top=0.86, bottom=0.2,
                          wspace=0.32)
    # ① 평균 박동
    # 평균 박동을 오프라인 영위상(실시간에는 쓸 수 없는 기준)과의 차이로 그린다 — 차이가 R파 높이의 몇 % 인가
    x, r, ramp, rec = A.take("d1")
    ax = fig.add_subplot(gs[0, 0])
    outs = {lab: _beat_avg(fn(x), r, A.FS) for lab, fn in A.CANDS}
    t, ref = outs["오프라인 영위상"]
    ax.axhline(0, color=FE_COL["오프라인 영위상"], lw=1.4)
    ax.axvspan(-0.04, 0.04, color="#e6e5f4", lw=0, zorder=0)
    ax.text(0.0, 5.3, "QRS", ha="center", fontsize=FS_S - 0.6, color=MS.INK2)
    for lab in ("인과 o1 0.5 Hz", "블록 영위상 0.5 s", "중앙값 200+600 ms"):
        ax.plot(t, (outs[lab][1] - ref) / ramp * 100, color=FE_COL[lab], lw=1.0, label=FE_LAB[lab])
    ax.set_ylim(-6, 6); ax.set_xlim(-0.2, 0.6)
    ax.set_xlabel("R파 기준 시간 [s]"); ax.set_ylabel("오프라인 대비 [%R]", labelpad=1)
    ax.set_title("① QRS 직후 파형 왜곡", fontsize=FS, loc="left", fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=FS_S - 0.6, handlelength=1.2, borderaxespad=0.1,
              labelspacing=0.25)
    ax.grid(alpha=0.3)
    # ② 평탄도 × ST
    rows, _ = A.audit_reference(n_rec=8)
    g = pd.DataFrame(rows).groupby("who")[["spread", "st_err"]].mean()
    ax = fig.add_subplot(gs[0, 1])
    ax.add_patch(plt.Rectangle((0, 0), 8, 0.6, color="#cdeee1", alpha=0.55, lw=0, zorder=0))
    ax.text(0.4, 0.3, "둘 다 좋음\n(빈 자리)", fontsize=FS_S - 0.6, color="#3f7a5f", va="center", linespacing=1.0)
    for who, row in g.iterrows():
        lat = FE_LAT.get(who)
        s = 18 if lat is None else 18 + lat / 6
        ax.scatter(row.spread, abs(row.st_err), s=s, color=FE_COL[who], edgecolor="white", lw=0.6, zorder=3,
                   marker="D" if lat is None else "o")
        if lat is None:                              # 오프라인 기준점 — 블록 영위상 표지와 겹치지 않게 오른쪽 아래로
            ax.annotate(FE_LAB[who] + "(실시간 불가)", (row.spread, abs(row.st_err)), xytext=(7, 1),
                        textcoords="offset points", fontsize=FS_S - 0.6, color=FE_COL[who], va="bottom")
        else:
            ax.annotate(FE_LAB[who] + f"\n{lat} ms", (row.spread, abs(row.st_err)), xytext=(5, 4),
                        textcoords="offset points", fontsize=FS_S - 0.6, color=FE_COL[who], linespacing=1.0)
    ax.set_xlim(0, 25); ax.set_ylim(0, 2.6)
    ax.set_xlabel("평탄도: 준위 산포 [%R]"); ax.set_ylabel("ST 준위 오차 [%R]")
    ax.set_title("② 평탄도와 ST 충실도", fontsize=FS, loc="left", fontweight="bold")
    ax.grid(alpha=0.3)
    # ③ 실보드 화면
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    img = _img(LIVE / "live_02_split_t32.png")
    if img is not None:
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, "[실보드 화면]", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("③ 실보드: 지연 644 ms = 548 + 96", fontsize=FS, loc="left", fontweight="bold")
    _save(fig, "B2", "B2_realtime")
    return {"rec": str(rec), **{f"{FE_LAB[k]}": {"spread": float(v.spread), "st_err": float(v.st_err)}
                                for k, v in g.iterrows()}}


def b4_causal():
    """B2 의 ① 만 — 실시간 공통 전처리 방식별 평균 박동과 오프라인 영위상의 차이(MIT-BIH 기록 100)."""
    import audit_fe_metrics as A
    W, H = SIZES["B4"]
    fig, ax = plt.subplots(figsize=(W * CM, H * CM))
    fig.subplots_adjust(left=0.13, right=0.70, top=0.96, bottom=0.27)
    x, r, ramp, _ = A.take("d1")
    outs = {lab: _beat_avg(fn(x), r, A.FS) for lab, fn in A.CANDS}
    t, ref = outs["오프라인 영위상"]
    ax.axhline(0, color=FE_COL["오프라인 영위상"], lw=1.4, label="오프라인\n영위상(기준)")
    ax.axvspan(-0.04, 0.04, color="#e6e5f4", lw=0, zorder=0)
    ax.text(0.0, 5.0, "QRS", ha="center", fontsize=FS_S, color=MS.INK2)
    for lab in ("인과 o1 0.5 Hz", "블록 영위상 0.5 s", "중앙값 200+600 ms"):
        ax.plot(t, (outs[lab][1] - ref) / ramp * 100, color=FE_COL[lab], lw=1.0, label=FE_LAB[lab])
    ax.set_ylim(-6, 6); ax.set_xlim(-0.2, 0.6)
    ax.set_xlabel("R파 기준 시간 [s]", labelpad=1); ax.set_ylabel("기준 대비 [%R]", labelpad=1)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=FS_S - 0.4, handlelength=1.2,
              borderaxespad=0.1, labelspacing=0.35)
    ax.grid(alpha=0.3)
    _save(fig, "B4", "B4_causal")


# ------------------------------------------------------------------ B3 가성 QRS군
def b3_phantom():
    """칼만 필터 출력 한 줄(위양성 검출 위치 음영) + 기법별 가성 QRS군 수 막대(MIT-BIH)."""
    summ = json.loads(Path("results/d1/phantom/summary.json").read_text())
    ex = np.load("results/d1/phantom/example.npz", allow_pickle=True)
    fs = float(ex["fs"])
    t = (np.arange(len(ex["clean"])) - (int(ex["center"]) - int(ex["i0"]))) / fs
    ref = np.asarray(ex["ref_peaks"]); tol = int(round(0.075 * fs))
    W, H = SIZES["B3"]
    fig = plt.figure(figsize=(W * CM, H * CM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0], left=0.02, right=0.99, top=0.85, bottom=0.08, wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, ex["clean"], color=MS.CLEAN, lw=1.6)
    ax.plot(t, ex["M05"], color=MS.OUTK, lw=0.7)
    for d in np.asarray(ex["det_M05"]):
        if ref.size == 0 or np.abs(ref - d).min() > tol:
            ax.axvspan(t[d] - 0.2, t[d] + 0.2, color=MS.PHANTOM, alpha=0.25, lw=0)
    ax.set_yticks([]); ax.set_xticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("칼만 필터 출력(음영: 박동 없는 위치)", fontsize=FS_S, loc="left")
    ax = fig.add_subplot(gs[0, 1])
    thr = str(summ["threshold"])
    ms = [("M04", "SWT"), ("M05", "칼만"), ("M06", "U-Net")]
    vals = [summ["methods"][m]["beatlike"][thr] for m, _ in ms]
    ax.barh(range(len(ms)), vals, color=[MS.PHANTOM if m == "M05" else "#8f8a80" for m, _ in ms], height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.04, i, str(v), va="center", fontsize=FS_S)
    ax.set_yticks(range(len(ms))); ax.set_yticklabels([n for _, n in ms], fontsize=FS_S)
    ax.invert_yaxis(); ax.set_xlim(0, max(vals) * 1.35); ax.set_xticks([])
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.set_title("가성 QRS군 [건]", fontsize=FS_S, loc="left")
    _save(fig, "B3", "B3_phantom")


# ------------------------------------------------------------------ N1 · N2 잡음 종류별 비교
def _expb_means():
    """EXP-B 의 SNR 개선량 평균(44 구간)과 구간 수 — `results/d1/exp_b/metrics.parquet`."""
    import pandas as pd
    df = pd.read_parquet("results/d1/exp_b/metrics.parquet")
    df = df[(df.metric == "snr_imp_scaled") & (df.snr_in_target == EXPB_SNR)]
    mean = df.groupby(["cond", "method"]).value.mean().unstack("method")
    return mean, int(df[df.method == "M_FE"].groupby("cond").size().min())


def n_pair(key, conds, name, size=None, fs=None, fs_s=None, lw=1.0, save=None):
    """잡음 두 종을 한 장에 — 잡음마다 출력 행(참값 회색 + 출력)과 잔차 행(출력 - 참값).

    출력 칸의 숫자는 이 구간 하나가 아니라 **EXP-B 44 구간 평균**이다(판정판 S17 과 같은 값). 세 처리 중 가장 높은 칸은
    바탕을 칠한다(기저선 변동은 공통 전처리 단독이 가장 높다). 잔차는 세 기법이 같은 세로축을 쓰고, 입력 칸(지울 잡음)은 크기가 수십 배라 따로 둔다.
    """
    # 판넬도 같은 그림을 쓴다(make_panel_figs.f5_pairs) — 크기 · 글자 · 선 굵기 배율 · 저장 함수만 바꾼다
    FS, FS_S = fs or globals()["FS"], fs_s or globals()["FS_S"]
    mean, n = _expb_means()
    W, H = size or SIZES[key]
    fig = plt.figure(figsize=(W * CM, H * CM))
    gs = fig.add_gridspec(4, 4, left=0.085 if size is None else 0.11, right=0.995, top=0.92, bottom=0.02, hspace=0.18,
                          wspace=0.08,
                          height_ratios=[1.3, 0.75, 1.3, 0.75])
    for k, cond in enumerate(conds):
        d = MS.prepare("mitdb", EXPB_SNR, cond, methods=("M_FE", "M04"))   # M08 은 dl_methods 로 함께 만든다
        fs, x, y = d["fs"], d["x"], d["y"]
        i0, i1 = MS._window(d, 2.0)
        t = (np.arange(i0, i1) - i0) / fs
        win = max(("M_FE", "M04", "M08"), key=lambda m: float(mean.loc[cond, m]))   # 세 처리 중 최고
        ro, rr = 2 * k, 2 * k + 1
        ax_o = [fig.add_subplot(gs[ro, c]) for c in range(4)]
        ax_r = [fig.add_subplot(gs[rr, c]) for c in range(4)]
        for c in range(1, 4):
            ax_o[c].sharey(ax_o[0])
        for c in range(2, 4):
            ax_r[c].sharey(ax_r[1])
        for c, (m, lab) in enumerate(N_COLS):
            o = y if m == "noisy" else d["outs"][m]
            col = MS.NOISY if m == "noisy" else MS.C.get(m, MS.OUTK)
            a = ax_o[c]
            a.plot(t, x[i0:i1], color=MS.CLEAN, lw=1.1 * lw)
            a.plot(t, o[i0:i1], color=col, lw=(0.4 if m == "noisy" else 0.55) * lw)
            if m == win:
                a.set_facecolor("#fbf3dc"); ax_r[c].set_facecolor("#fbf3dc")
            if m != "noisy":
                a.text(0.99, 1.02, f"{mean.loc[cond, m]:+.1f}", transform=a.transAxes, ha="right", va="top",
                       fontsize=FS_S, fontweight="bold", color=col)
            if k == 0:
                a.set_title(lab, fontsize=FS, color=MS.INK if m == "noisy" else col)
            r = ax_r[c]
            r.plot(t, ((y if m == "noisy" else o) - x)[i0:i1], color=col, lw=0.4 * lw)
            r.axhline(0, color=MS.INK2, lw=0.4)
            for aa in (a, r):
                aa.set_xticks([]); aa.set_yticks([]); aa.grid(False)
                for sp in aa.spines.values():
                    sp.set_visible(False)
        lo, hi = ax_o[0].get_ylim()                  # 숫자 띠: 위쪽 여백 — R파 봉우리와 겹치지 않게
        ax_o[0].set_ylim(lo, hi + 0.38 * (hi - lo))
        ax_o[0].set_ylabel(NOISE_KO[cond].replace(" ", "\n"), fontsize=FS_S, fontweight="bold", rotation=90, labelpad=1,
                           linespacing=1.0)
        ax_r[0].set_ylabel("잔차", fontsize=FS_S, color=MS.INK2, rotation=90, labelpad=1)
    (save or _save)(fig, key, name)
    return {c: {m: float(mean.loc[c, m]) for m in ("M_FE", "M04", "M08")} for c in conds}, n


def main() -> int:
    ap = argparse.ArgumentParser(description="브로셔 전용 그림: 시스템 구성도 · 실시간 공통 전처리 · 가성 QRS군")
    ap.add_argument("--only", nargs="*", default=None, help="B1 B2 B3 B4 B5 N1 N2 중 일부만 생성 (기본: B3 B4 B5 N1 N2 — B1 · B2 는 이전 구성)")
    a = ap.parse_args()
    want = set(a.only or ["B3", "B4", "B5", "N1", "N2"])
    _ko_font()
    ensure_dir(OUT)
    if "B1" in want:
        b1_system()
    if "B2" in want:
        vals = b2_realtime()
        (OUT / "B2_realtime.json").write_text(json.dumps(vals, ensure_ascii=False, indent=1), encoding="utf-8")
    if "B3" in want:
        b3_phantom()
    if "B4" in want:
        b4_causal()
    if "B5" in want:
        b5_pipeline()
    if "N1" in want or "N2" in want:
        vals = {}
        if "N1" in want:
            vals["N1"] = n_pair("N1", ("ma_synth", "impulse"), "N1_dl_wins")
        if "N2" in want:
            vals["N2"] = n_pair("N2", ("bw_synth", "pli"), "N2_classic_wins")
        (OUT / "N_noise_pairs.json").write_text(json.dumps(vals, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
