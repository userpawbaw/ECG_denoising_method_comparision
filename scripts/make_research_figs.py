"""연구 과정 시각화 — 시각화 제안서(`docs/96_visual_proposal.md`) P1~P11.

    python scripts/make_research_figs.py              # 전부
    python scripts/make_research_figs.py --only S22   # 일부만

산출: `results/slides/research/S22_*.png` ~ `S32_*.png`, 색인은 `docs/97_research_figs.md`.

`make_slides.py` 의 그림이 **결과**를 보여 준다면, 이 그림들은 **결과에 이르는 과정**
(측정 도구의 눈금, 원인 규명, 기각된 해석)을 보여 준다. 수치는 전부 산출물 파일에서
읽는다. 산출물로 남지 않은 「수정 전」 값(이미 고친 오류의 당시 값)만 연구 기록의
기록값을 쓰며, 그 경우 그림 안에 「기록값」이라고 적고 기록 번호를 단다.

글꼴은 `make_slides.slide_style()` 로 고른다(NanumGothic 우선). 나눔 글꼴에 U+2212 가
없으므로 그림 문자열의 빼기는 ASCII '-' 로 쓰고 `axes.unicode_minus` 를 끈다.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np

import make_slides as MS
from ecgdn.utils import ensure_dir, save_manifest

OUT = MS.OUT / "research"   # make_slides 의 manifest 를 덮지 않도록 하위 폴더에 둔다
INK, INK2, SURFACE = MS.INK, MS.INK2, MS.SURFACE
GRAY = "#8a8a8a"
LIGHT = "#d9d7d0"
BLUE = "#184f95"
BLUE2 = "#7fb2ec"
ORANGE = "#eb6834"
RED = "#d03b3b"


def _ko_font():
    """한글 글꼴(NanumGothic 등) 선택 + 음수 눈금 깨짐 방지."""
    MS.slide_style()
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "dejavusans"


def _save(fig, name):
    fig.savefig(OUT / name, dpi=175, bbox_inches="tight")
    plt.close(fig)
    print("  " + name)


def _ckpt_best(run: str) -> float:
    """체크포인트의 best_metric (로그 마지막 줄이 아니라 파일에서 읽는다)."""
    import torch
    ck = torch.load(f"results/d1/{run}/best.pt", map_location="cpu", weights_only=False)
    return float(ck["best_metric"]), int(ck["epoch"])


def _csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ P1 / S22
def s22_resolution():
    """측정 도구의 눈금(학습 반복 간 변동)과 효과 크기를 한 축에 놓는다."""
    cpu = [_ckpt_best(r)[0] for r in ("m06_l1", "m06_l1_s1", "m06_l1_s2")]
    cal = {int(r["seed"]): r for r in _csv("results/ext/calib/sweep.csv") if r["arm"] == "m06_l1"}
    t4_best = [float(cal[s]["best_metric"]) for s in (0, 1, 2)]
    t4_fix = [float(cal[s]["fixed_snr_imp_scaled"]) for s in (0, 1, 2)]
    hw = np.array(t4_best) - np.array(cpu)
    bands = [
        ("시드별 검증 잡음, 학습 3회 (F-40)", np.std(cpu, ddof=1), np.ptp(cpu)),
        ("같은 3회, 다른 환경(T4)에서 재학습 (F-41)", np.std(t4_best, ddof=1), np.ptp(t4_best)),
        ("같은 3개 모델을 고정 평가 집합으로 (F-41)", np.std(t4_fix, ddof=1), np.ptp(t4_fix)),
        ("하드웨어 재현 하한: 같은 시드의 T4 - CPU", np.std(hw, ddof=1), np.ptp(hw)),
    ]
    eff = [
        ("구조: SWT 직렬 결합", "m07_l1", GRAY), ("구조: 웨이블릿 부대역", "m08_l1", GRAY),
        ("구조: CNN-Transformer", "m09_l1", GRAY), ("구조: 시간 해상도 유지", "m10_l1", GRAY),
        ("용량: 채널 폭 1/2", "m06_l1_half", GRAY), ("용량: 채널 폭 1/4", "m06_l1_quarter", GRAY),
        ("손실: MAE와 미분 오차 추가", "m06_l3", BLUE), ("손실: 무잡음 입력 보존항 추가", "m06_l6", BLUE),
    ]
    fig, ax = plt.subplots(figsize=(12.5, 7.6))
    y = 0
    yt, yl = [], []
    for lab, sd, rng in bands:
        ax.add_patch(plt.Rectangle((-sd, y - 0.32), 2 * sd, 0.64, color=LIGHT, zorder=1))
        ax.plot([-rng / 2, rng / 2], [y, y], color=GRAY, lw=1.0, ls=":", zorder=2)
        ax.text(sd + 0.08, y, f"표준편차 {sd:.3f} dB (범위 {rng:.2f})", va="center", fontsize=9.5, color=INK2)
        yt.append(y); yl.append(lab); y += 1
    y += 0.6
    ax.axhline(y - 0.8, color=LIGHT, lw=1)
    y_eff0 = y
    floor_sd = bands[2][1]
    for lab, arm, c in eff:
        r = MS._paired("structure", arm, "m06_l1", "fixed_snr_imp_scaled")
        if r is None:
            continue
        m, d, p = r
        ax.add_patch(plt.Rectangle((-floor_sd, y - 0.32), 2 * floor_sd, 0.64, color=LIGHT, alpha=0.55, zorder=1))
        ax.plot([d.min(), d.max()], [y, y], color=c, lw=2.2, alpha=0.5, solid_capstyle="round", zorder=2)
        sig = p < 0.05
        ax.scatter([m], [y], s=95, color=c if sig else SURFACE, edgecolor=c, lw=1.8, zorder=3)
        ax.text(max(m, d.max()) + 0.1, y, f"{m:+.2f} dB" + (" (p<0.05)" if sig else ""), va="center",
                fontsize=9.5, color=INK if sig else INK2)
        yt.append(y); yl.append(lab); y += 1
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=10)
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("dB  (위: 측정 변동의 +-1 표준편차 띠와 범위(점선) / 아래: 잔차 U-Net + MSE 손실 대비 평균 차, 막대는 시드 4개의 범위)")
    ax.text(0.01, 1.01, "측정 변동", transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK2)
    ax.text(0.01, y_eff0 - 0.75, "효과 (옅은 띠: 고정 평가 집합의 +-1 표준편차)", transform=ax.get_yaxis_transform(),
            fontsize=11, fontweight="bold", color=INK2, va="bottom")
    fig.suptitle("측정 도구의 눈금을 먼저 재었다: 평가 잡음을 고정하자 변동이 "
                 f"{bands[0][1]:.2f} -> {bands[2][1]:.2f} dB로 줄었고, 그 눈금을 넘은 개선은 손실 함수 변경뿐이다\n"
                 "MIT-BIH, 잔차 U-Net. 효과는 시드 4~7 대응 비교(고정 평가 집합), 채운 점: 대응 t-검정 p < 0.05",
                 fontsize=12, y=1.02)
    _save(fig, "S22_resolution.png")


# ------------------------------------------------------------------ P2 / S23
BANDS = [("band_-5_0", "-5~0"), ("band_0_5", "0~5"), ("band_5_10", "5~10"),
         ("band_10_15", "10~15"), ("band_15_20", "15~20")]


def s23_band_specialist():
    sw = MS._sweep("band")
    arms = [("m06_l1", "전 대역 학습 모델 (-5~20 dB)", INK, 2.6),
            ("m06_l1_band_lo", "저 SNR 전용 모델 (-5~0 dB)", ORANGE, 1.6),
            ("m06_l1_band_hi", "고 SNR 전용 모델 (15~20 dB)", BLUE2, 1.6)]
    x = np.arange(len(BANDS))
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    means = {}
    for arm, lab, c, lw in arms:
        v = np.array([[sw[(arm, s)][k] for k, _ in BANDS] for s in (4, 5, 6)])
        means[arm] = v.mean(0)
        ax.fill_between(x, v.min(0), v.max(0), color=c, alpha=0.13, lw=0)
        ax.plot(x, v.mean(0), color=c, lw=lw, marker="o", ms=6, label=lab, zorder=3)
    ax.axhline(0, color=INK2, lw=1)
    base, hi = means["m06_l1"][-1], means["m06_l1_band_hi"][-1]
    film = MS._paired("cond", "m06_cond", "m06_l1", "band_15_20")
    xr = x[-1] + 1.5
    ax.plot([x[-1], xr + 0.1], [base, base], color=INK, lw=0.8, ls=":")
    ax.plot([x[-1], xr + 0.1], [hi, hi], color=BLUE2, lw=0.8, ls=":")
    ax.annotate("", xy=(xr, hi), xytext=(xr, base), arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))
    ax.text(xr + 0.07, 1.3, f"회수 가능한 상한\n+{hi - base:.2f} dB", fontsize=10, color=BLUE, va="center")
    if film:
        xf = x[-1] + 0.4
        ax.annotate("", xy=(xf, base + film[0]), xytext=(xf, base),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=2))
        ax.text(xf + 0.07, base + film[0] / 2 - 0.1, f"SNR 조건화(FiLM)로\n회수: +{film[0]:.2f} dB\n(p={film[2]:.3f})",
                fontsize=10, color=RED, va="center")
    ax.set_xticks(x); ax.set_xticklabels([b for _, b in BANDS])
    ax.set_xlim(-0.3, x[-1] + 2.45)
    ax.set_xlabel("평가한 입력 SNR 구간 [dB]")
    ax.set_ylabel("SNR 개선량(진폭 보정) [dB]")
    ax.legend(loc="lower left", frameon=False, fontsize=10)
    fig.suptitle("전 대역을 한 모델로 학습하면 저 SNR 전용 모델처럼 학습되어, 고 SNR 구간의 개선량이 음수가 된다\n"
                 "MIT-BIH, 시드 4~6 평균(띠: 시드 범위), 고정 평가 집합. FiLM 화살표는 시드 0~5 대응 차이를 "
                 "전 대역 모델의 값에서 그렸다", fontsize=12, y=1.02)
    _save(fig, "S23_band_specialist.png")


# ------------------------------------------------------------------ P3 / S24
AXIS_METRICS = [("snr_imp_scaled", "SNR 개선량(진폭 보정) [dB]", +1),
                ("cc", "상관계수", +1), ("prdn", "PRDN [%]", -1), ("psd_logdist", "PSD 로그 거리", -1)]
AXIS_METHODS = {"M06": "잔차 U-Net", "M08": "웨이블릿 부대역 U-Net", "M07": "SWT 직렬 결합 U-Net",
                "M04": "SWT 임계값 처리", "M05": "칼만 필터", "M_FE": "공통 전처리 단독"}


def _axis_gap():
    """`docs/92_axis_gap.md`(자동 생성)의 `Mxx − M01` 표를 읽는다."""
    out = {}
    cur = None
    for line in Path("docs/92_axis_gap.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"### `(\S+)` − `M01`", line)
        if m:
            cur = m.group(1); continue
        if line.startswith("## ") and "M01" not in line:
            cur = None
        if cur and line.startswith("| `"):
            c = [x.strip() for x in line.strip("|").split("|")]
            met = c[0].strip("`")
            out[(cur, met)] = (float(c[1]), "우수" in c[3] or "열세" in c[3], float(c[4]), "우수" in c[6] or "열세" in c[6])
    return out


def s24_axis_dumbbell():
    g = _axis_gap()
    meths = [m for m in AXIS_METHODS if any((m, k) in g for k, *_ in AXIS_METRICS)]
    fig, axes = plt.subplots(1, len(AXIS_METRICS), figsize=(15, 0.7 * len(meths) + 2.2), sharey=True)
    for ax, (met, lab, sgn) in zip(axes, AXIS_METRICS):
        for i, m in enumerate(meths):
            if (m, met) not in g:
                continue
            d0, s0, d1, s1 = g[(m, met)]
            a, b = sgn * d0, sgn * d1
            ax.plot([a, b], [i, i], color=LIGHT, lw=3, zorder=1)
            ax.scatter([a], [i], s=70, color=GRAY if s0 else SURFACE, edgecolor=GRAY, lw=1.6, zorder=2)
            ax.scatter([b], [i], s=80, color=BLUE if s1 else SURFACE, edgecolor=BLUE, lw=1.8, zorder=3)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_title(lab + ("" if sgn > 0 else "  (부호 반전)"), fontsize=11)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(meths)))
    axes[0].set_yticklabels([AXIS_METHODS[m] for m in meths])
    axes[0].set_ylim(len(meths) - 0.5, -0.5)
    fig.text(0.5, -0.08, "대역통과 필터(M01) 대비 차이. 오른쪽일수록 우수(PRDN · PSD 거리는 낮을수록 좋으므로 부호를 바꿔 그렸다).  "
             "회색: 합성 ECG, 파랑: MIT-BIH, 채운 점: Holm 보정 후 유의", ha="center", fontsize=10, color=INK2)
    fig.suptitle("합성 ECG의 우위는 MIT-BIH에서 SNR 개선량으로는 크게 줄지만, 파형 · 주파수 지표로는 유지된다\n"
                 "전 구간(입력 SNR -5~20 dB) 평균, 대응 Wilcoxon 검정 + Holm 보정, 22 기록",
                 fontsize=12, y=1.04)
    _save(fig, "S24_axis_dumbbell.png")


# ------------------------------------------------------------------ P4 / S25
def s25_flip_before_after():
    """측정 오류의 수정 전 · 후. 수정 전 값은 산출물이 남지 않아 연구 기록의 기록값이다."""
    main0 = {(r["metric"], r["method"]): float(r["mean"]) for r in _csv("results/d0/report/table_main.csv")}
    dist = {}
    for tag in ("d0", "d1"):
        for r in _csv(f"results/{tag}/report/table_distortion.csv"):
            dist[(tag, r["method"])] = float(r["0"])
    swt_best = max(float(r[list(r)[5]]) for r in _csv("results/d1/tune_swt/search.csv")
                   if r.get(list(r)[1]) == "d1")
    panels = [
        ("F-8 합성 데이터의 정보 누출", "SNR 개선량 [dB]",
         [("딥러닝", 26.0, main0[("snr_imp_scaled", "M06")], MS.C["M08"]),
          ("고전 신호처리", 3.0, main0[("snr_imp_scaled", "M04")], MS.C["M04"])],
         "수정 전: 예비 실행 기록값\n수정 후: 주 실험(합성 ECG)"),
        ("F-12 기준 신호의 정의", "SWT 탐색 최고 SNR 개선량 [dB]",
         [("SWT 임계값 처리", -0.49, swt_best, MS.C["M04"])],
         "수정 전: MIT-BIH 원본 기준(기록값)\n수정 후: 공통 전처리를 거친 원본 기준"),
        ("F-15 정의상 항등식", "공통 전처리 단독의 무잡음 입력 출력 SNR [dB]",
         [("합성 ECG", 34.13, dist[("d0", "M_FE")], MS.C["M_FE"])],
         "수정 전: 원신호 기준(기록값, F-11)\n수정 후: 공통 전처리 출력 기준(정의상 무한대)"),
        ("F-22 이상값이 1인 지표의 비교", "Dilated ResNet 대 잔차 U-Net 의 -log10(p)",
         [("진폭 이득 편향, 합성 ECG", -np.log10(7e-6), -np.log10(0.965), GRAY)],
         "수정 전: 부호 있는 차이(기록값)\n수정 후: 1과의 거리"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.4))
    for ax, (title, ylab, rows, note) in zip(axes, panels):
        for lab, a, b, c in rows:
            ax.plot([0, 1], [a, b], color=c, lw=2.2)
            ax.scatter([0], [a], s=80, color=SURFACE, edgecolor=RED, lw=2, zorder=3)
            ax.scatter([1], [b], s=80, color=INK, zorder=3)
            ax.text(1.06, b, f"{lab}\n{a:.2f} -> {b:.2f}", va="center", fontsize=9, color=INK2)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["수정 전", "수정 후"])
        ax.set_xlim(-0.25, 1.9)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylab)
        ax.text(0.0, -0.2, note, transform=ax.transAxes, fontsize=8.8, color=INK2, va="top")
        if "F-22" in title:
            ax.axhline(-np.log10(0.05), color=GRAY, ls="--", lw=1)
            ax.text(1.85, -np.log10(0.05), "p=0.05", fontsize=8.5, color=GRAY, ha="right", va="bottom")
    fig.suptitle("측정 정의를 고치자 결론의 크기 · 부호 · 유의성이 바뀌었다 (빈 빨간 점: 수정 전, 검은 점: 수정 후)\n"
                 "패널마다 단위가 달라 축을 공유하지 않는다. 기록값은 산출물이 남지 않은 당시 값으로, 연구 기록(F 번호)에서 옮겼다",
                 fontsize=12, y=1.05)
    fig.tight_layout()
    _save(fig, "S25_flip_before_after.png")


# ------------------------------------------------------------------ P5 / S26
def s26_l6_conditions():
    d40 = MS._sweep("d40")
    ref = MS._sweep("d40_ref")
    def vals(sw, arm):
        return np.array([v["clean_pass_snr"] for (a, s), v in sw.items() if a == arm and "clean_pass_snr" in v])
    on = [("MSE 손실", vals(ref, "m06_l1")), ("MSE + 무잡음 항(MSE)", vals(d40, "m06_l1c")),
          ("무잡음 입력 보존 손실(MAE 포함)", vals(ref, "m06_l6"))]
    off = [("MSE 손실", vals(d40, "m06_l1_nores")), ("MSE + 무잡음 항(MSE)", np.array([])),
           ("무잡음 입력 보존 손실(MAE 포함)", vals(d40, "m06_l6_nores"))]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    x = np.arange(3)
    for rows, c, ls, lab, dx in ((on, BLUE, "-", "출력단 잔차 연결 있음", -0.06), (off, GRAY, "--", "출력단 잔차 연결 없음", 0.06)):
        xs, ms = [], []
        for i, (_, v) in enumerate(rows):
            if v.size == 0:
                continue
            ax.plot([i + dx] * 2, [v.min(), v.max()], color=c, lw=2, alpha=0.5)
            ax.scatter([i + dx], [v.mean()], s=90, color=c if ls == "-" else SURFACE, edgecolor=c, lw=2, zorder=3)
            ax.text(i + dx + (0.08 if dx > 0 else -0.08), v.mean(), f"{v.mean():.1f}", fontsize=10, color=c,
                    va="center", ha="left" if dx > 0 else "right")
            xs.append(i + dx); ms.append(v.mean())
        ax.plot(xs, ms, color=c, ls=ls, lw=1.6, label=lab)
    ax.text(1.06, 25.6, "(잔차 연결 없음 +\nMSE 무잡음 항은 측정하지 않음)", fontsize=9, color=GRAY, ha="left")
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in on])
    ax.set_ylabel("무잡음 입력의 출력 SNR [dB]")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(-0.5, 2.5)
    fig.suptitle("무잡음 입력 보존 손실은 MAE 오차 측정과 출력단 잔차 연결이 함께 있을 때만 크게 작동한다\n"
                 "MIT-BIH, 점: 시드 평균, 세로 막대: 시드 범위(잔차 연결 있음의 MSE · 보존 손실은 시드 0 · 1, 나머지는 시드 0~3)",
                 fontsize=12, y=1.03)
    _save(fig, "S26_l6_conditions.png")


# ------------------------------------------------------------------ P6 / S27
def s27_cond_l6_interaction():
    sw = MS._sweep("l6cond")
    cells = {("없음", "MSE 손실"): "m06_l1", ("있음", "MSE 손실"): "m06_cond",
             ("없음", "무잡음 입력 보존 손실"): "m06_l6", ("있음", "무잡음 입력 보존 손실"): "m06_cond_l6"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, (key, title) in zip(axes, (("band_15_20", "고 SNR 구간(15~20 dB)"), ("fixed_snr_imp_scaled", "전 구간"))):
        for loss, c in (("MSE 손실", GRAY), ("무잡음 입력 보존 손실", BLUE)):
            ms = []
            for j, cond in enumerate(("없음", "있음")):
                arm = cells[(cond, loss)]
                v = np.array([sw[(arm, s)][key] for s in range(6) if (arm, s) in sw])
                ax.plot([j, j], [v.min(), v.max()], color=c, alpha=0.4, lw=2)
                ax.scatter([j], [v.mean()], s=85, color=c, zorder=3)
                ax.text(j + (0.05 if j else -0.05), v.mean(), f"{v.mean():+.2f}", fontsize=9.5, color=c,
                        va="center", ha="left" if j else "right")
                ms.append(v.mean())
            ax.plot([0, 1], ms, color=c, lw=2, label=loss)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["SNR 조건화 없음", "SNR 조건화 있음"])
        ax.set_xlim(-0.45, 1.4)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("SNR 개선량(진폭 보정) [dB]")
        ax.axhline(0, color=LIGHT, lw=1)
    axes[0].legend(frameon=False, loc="lower right")
    a = MS._paired("l6cond", "m06_cond_l6", "m06_l6", "band_15_20")
    b = MS._paired("l6cond", "m06_cond", "m06_l1", "band_15_20")
    inter = a[1] - b[1]
    se = inter.std(ddof=1) / np.sqrt(len(inter))
    from scipy import stats
    t = stats.t.ppf(0.975, len(inter) - 1)
    fig.suptitle("SNR 조건화와 무잡음 입력 보존 손실은 같은 문제를 고치므로, 함께 써도 효과가 더해지지 않는다\n"
                 f"MIT-BIH, 시드 0~5. 고 SNR 구간 상호작용 {inter.mean():+.2f} dB "
                 f"(95 % 신뢰구간 {inter.mean() - t * se:+.2f} ~ {inter.mean() + t * se:+.2f}), 세로 막대: 시드 범위",
                 fontsize=12, y=1.03)
    _save(fig, "S27_cond_l6_interaction.png")


# ------------------------------------------------------------------ P7 / S28
PERT = [("pert_m8", -8), ("pert_m5", -5), ("pert_m2", -2), (None, 0), ("pert_p2", 2), ("pert_p5", 5), ("pert_p8", 8)]


def s28_cond_error_cost():
    from scipy import stats
    rows = [r for r in _csv("results/ext/condpert8/sweep.csv") if r["arm"] == "m06_cond"]
    fig, ax = plt.subplots(figsize=(11, 6))
    for suffix, base, lab, c in (("", "fixed_snr_imp_scaled", "전 구간", BLUE), ("_band_15_20", "band_15_20", "고 SNR 구간(15~20 dB)", ORANGE)):
        xs, ms = [], []
        for key, e in PERT:
            if key is None:
                xs.append(0); ms.append(0.0); continue
            d = np.array([float(r[key + suffix]) - float(r[base]) for r in rows])
            p = stats.ttest_1samp(d, 0).pvalue
            ax.plot([e, e], [d.min(), d.max()], color=c, alpha=0.35, lw=2)
            ax.scatter([e], [d.mean()], s=80, color=c if p < 0.05 else SURFACE, edgecolor=c, lw=1.8, zorder=3)
            xs.append(e); ms.append(d.mean())
        ax.plot(xs, ms, color=c, lw=1.8, label=lab)
    m8 = np.mean([float(r["pert_m8"]) - float(r["fixed_snr_imp_scaled"]) for r in rows])
    p8 = np.mean([float(r["pert_p8"]) - float(r["fixed_snr_imp_scaled"]) for r in rows])
    ax.axhline(0, color=INK2, lw=1); ax.axvline(0, color=LIGHT, lw=1)
    ax.annotate(f"같은 8 dB 오차\n과소 추정: {m8:+.2f} dB", xy=(-8, m8), xytext=(-4.6, -1.75),
                fontsize=10, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE))
    ax.annotate(f"과대 추정: {p8:+.2f} dB\n(비 {m8 / p8:.1f} 배)", xy=(8, p8), xytext=(4.8, p8 - 0.9),
                fontsize=10, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE))
    ax.set_xticks([e for _, e in PERT])
    ax.set_xlabel("모델에 알려 준 조건 SNR 의 오차 [dB]  (음수: 실제보다 낮게 알려 줌)")
    ax.set_ylabel("정확한 조건 대비 SNR 개선량 차이 [dB]")
    ax.legend(frameon=False, loc="lower right")
    fig.suptitle("SNR 조건화 모델에 SNR을 낮게 알려 주면, 같은 크기로 높게 알려 줄 때보다 손실이 크다\n"
                 "MIT-BIH, 같은 가중치에 조건값만 바꾼 측정, 시드 0~5. 채운 점: p < 0.05(1표본 t-검정), 막대: 시드 범위",
                 fontsize=12, y=1.03)
    _save(fig, "S28_cond_error_cost.png")


# ------------------------------------------------------------------ P8 / S29
FLOOR_METRICS = [("qrs_dur_err_ms", "QRS 폭 오차"), ("rpeak_mae_ms", "R파 위치 오차"),
                 ("cc", "상관계수"), ("rmse", "RMSE")]


def s29_metric_floor():
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    for k, (tag, lab, c, dy) in enumerate((("d0", "합성 ECG", GRAY, -0.16), ("d1", "MIT-BIH", BLUE, 0.16))):
        floor = {r["metric"]: float(r["floor_p95"]) for r in _csv(f"results/{tag}/metric_floor/floor.csv")}
        st = _csv(f"results/{tag}/report/table_stats.csv")
        for i, (met, _) in enumerate(FLOOR_METRICS):
            if met not in floor or floor[met] <= 1e-9:
                if i == 1 or met in floor:
                    ax.text(1.5e-4, i + dy, f"{lab}: 판별 하한이 0이어서 비를 계산할 수 없음", fontsize=8.5,
                            color=c, va="center")
                continue
            ds = np.array([abs(float(r["delta_mean"])) for r in st if r["metric"] == met])
            if ds.size == 0:
                continue
            ratio = ds / floor[met]
            ax.scatter(ratio, [i + dy] * ratio.size, s=34, color=c, alpha=0.75, zorder=3,
                       label=lab if i == 0 else None)
    ax.axvline(1, color=RED, lw=1.4, ls="--")
    ax.text(1.08, len(FLOOR_METRICS) - 0.55, "비 = 1", color=RED, fontsize=9.5)
    ax.set_ylim(len(FLOOR_METRICS) - 0.4, -0.6)
    ax.axvspan(1e-4, 1, color="#f6e3e3", zorder=0)
    ax.set_xscale("log")
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_yticks(range(len(FLOOR_METRICS))); ax.set_yticklabels([l for _, l in FLOOR_METRICS])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("|대역통과 필터 대비 기법별 평균 차이| / 지표의 판별 하한(95 백분위)  -  로그 축")
    ax.legend(frameon=False, loc="lower right")
    fig.suptitle("같은 지표라도 데이터에 따라 판정할 수 있는지가 달라진다: 붉은 영역(비 < 1)의 차이는 p 값과 무관하게 구분할 수 없다\n"
                 "점 하나가 기법 하나. 판별 하한은 무잡음 신호에 40 dB 잡음을 더해 시드 10개로 잰 지표의 변동(3.6 절)",
                 fontsize=12, y=1.03)
    _save(fig, "S29_metric_floor.png")


# ------------------------------------------------------------------ P9 / S30
def s30_phantom_threshold():
    import pandas as pd
    names = {"M_FE": "공통 전처리 단독", "M04": "SWT 임계값 처리", "M05": "칼만 필터", "M06": "잔차 U-Net"}
    ths = (0.7, 0.8, 0.9)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)
    for ax, tag, lab in zip(axes, ("d0", "d1"), ("합성 ECG", "MIT-BIH")):
        r = pd.read_parquet(f"results/{tag}/phantom/rows.parquet")
        meths = [m for m in names if m in set(r["method"])]
        x = np.arange(len(meths)); w = 0.26
        for j, t in enumerate(ths):
            n = [int((r[(r.method == m)]["cc"] > t).sum()) for m in meths]
            bars = ax.bar(x + (j - 1) * w, n, w, color=[RED if m == "M05" else GRAY for m in meths],
                          alpha=0.45 + 0.25 * j, label=f"임계값 {t}")
            ax.bar_label(bars, fontsize=8.5, padding=2)
        ax.set_xticks(x); ax.set_xticklabels([names[m] for m in meths], fontsize=10)
        ax.set_title(lab, fontsize=11)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        ax.set_ylabel("박동 형태의 가성 QRS군 [건]")
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, title="판정 임계값\n(진할수록 엄격)", loc="upper left", fontsize=9.5)
    fig.suptitle("가성 QRS군 판정 임계값을 0.7 · 0.8 · 0.9로 바꾸어도 칼만 필터의 순위는 같다\n"
                 "위양성 검출 위치 +-200 ms의 출력과 그 출력의 평균 박동 파형의 상관계수, 입력 SNR -5 · 0 dB, 백색 · 혼합 잡음, 22 기록",
                 fontsize=12, y=1.04)
    _save(fig, "S30_phantom_threshold.png")


# ------------------------------------------------------------------ P10 / S31
def _summary_types():
    """보고서 8 장 「방법론적 요약」이 유형으로 묶은 기록을 읽는다(임의 분류를 하지 않는다)."""
    rep = Path("docs/91_report.md").read_text(encoding="utf-8")
    sec = rep[rep.index("## 방법론적 요약"):rep.index("## 실행 가능한 산출물")]
    first = sec.split("\n\n")[1]
    out = {"측정 전제의 오류": set(re.findall(r"F-(\d+)", first.split("는 모두 같은 유형이다")[0]))}
    for m in re.finditer(r"^### (.+?)\((F-[\d · F-]+)\)", sec, flags=re.M):
        out[m.group(1).strip()] = set(re.findall(r"F-(\d+)", m.group(2)))
    return out


def s31_findings_map():
    txt = Path("docs/20_findings.md").read_text(encoding="utf-8")
    ids = sorted(int(x) for x in re.findall(r"^## F-(\d+)\. ", txt, flags=re.M))
    types = _summary_types()
    rows = list(types) + ["그 외 발견 기록"]
    cols = [RED, ORANGE, BLUE, GRAY]
    fig, ax = plt.subplots(figsize=(14, 4.4))
    k_alt = 0
    for n in ids:
        hit = [k for k, v in types.items() if str(n) in v]
        y = rows.index(hit[0]) if hit else len(rows) - 1
        c = cols[y] if hit else LIGHT
        ax.scatter([n], [y], s=120 if hit else 50, color=c, edgecolor=c if hit else GRAY, lw=1.2, zorder=3)
        if hit:
            up = k_alt % 2 == 0; k_alt += 1
            ax.text(n, y - 0.28 if up else y + 0.36, f"F-{n}", fontsize=8, ha="center", color=c)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.set_ylim(len(rows) - 0.4, -0.7)
    ax.set_xlabel("발견 기록 번호 (번호 순 = 발견 순)")
    ax.grid(axis="y", visible=False)
    k = sum(len(v) for v in types.values())
    fig.suptitle(f"발견 기록 {len(ids)}건 중 보고서 8 장의 방법론적 요약이 유형으로 묶은 {k}건의 분포\n"
                 "측정 전제의 오류는 연구 초기(F-8~F-17)와 실시간 처리 연구(F-33 · F-35)에 집중되어 있다. "
                 "분류는 보고서 본문에서 읽는다", fontsize=12, y=1.06)
    _save(fig, "S31_findings_map.png")


# ------------------------------------------------------------------ P11 / S32
def s32_step_optimum():
    import yaml
    from ecgdn.data.sources import MITDB_SPLIT
    n_train = len(MITDB_SPLIT["train"])
    runs = [("m06_l1", "기준 (기록당 160창)", GRAY), ("m06_l1_data4x_eq", "데이터 4배, 학습량 고정", ORANGE),
            ("m06_l1_data4x", "데이터 4배 + 학습량 4배", BLUE)]
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    opt = []
    for run, lab, c in runs:
        cfg = yaml.safe_load(Path(f"configs/{run}.yaml").read_text(encoding="utf-8"))
        mpr = cfg["data"]["max_per_record"]; bs = cfg["train"]["batch_size"]
        spe = int(np.ceil(n_train * mpr / bs))
        h = json.loads(Path(f"results/d1/{run}/history.json").read_text())
        st = np.array([e["epoch"] * spe for e in h]); v = np.array([e["val_snr_imp_scaled"] for e in h])
        best, ep = _ckpt_best(run)
        ax.plot(st, v, color=c, lw=1.6, marker="o", ms=3, label=f"{lab}: 최고 성능 {ep * spe:,} step")
        ax.axvline(ep * spe, color=c, ls="--", lw=1.2)
        opt.append(ep * spe)
    ax.set_xlabel("학습 step 수")
    ax.set_ylabel("검증 SNR 개선량(진폭 보정) [dB]")
    ax.legend(frameon=False, loc="lower right")
    fig.suptitle(f"최고 성능 체크포인트는 데이터량 · 학습량과 무관하게 {min(opt):,}~{max(opt):,} step 에서 나타난다\n"
                 "MIT-BIH, 잔차 U-Net + MSE 손실, 시드 0. epoch 당 step = 학습 기록 수 x 기록당 창 수 / 배치 크기",
                 fontsize=12, y=1.03)
    _save(fig, "S32_step_optimum.png")


FIGS = {
    "S22": (s22_resolution, "S22_resolution.png", "P1",
            "측정 도구의 눈금. 위 네 띠는 같은 설정의 학습 반복 간 변동이 평가 잡음 고정으로 줄어드는 과정(F-40 · F-41)이고, "
            "아래 점은 구조 · 용량 · 손실 변경의 효과이다. 고정 평가 집합의 띠를 넘은 개선은 손실 함수 변경 두 건뿐이다."),
    "S23": (s23_band_specialist, "S23_band_specialist.png", "P2",
            "입력 SNR 대역 전용 모델. 전 대역 모델(검정)이 저 SNR 전용 모델(주황)과 겹쳐 내려가 고 SNR 구간에서 음수가 된다. "
            "오른쪽 화살표는 회수 가능한 상한(고 SNR 전용 모델)과 SNR 조건화로 실제 회수한 양이다(F-43 · F-45)."),
    "S24": (s24_axis_dumbbell, "S24_axis_dumbbell.png", "P3",
            "합성 ECG(회색)에서 MIT-BIH(파랑)로 옮겼을 때 대역통과 필터 대비 차이의 변화. SNR 개선량은 크게 줄고 유의성을 잃지만 "
            "상관계수 · PRDN · PSD 거리의 우위는 유지된다. 데이터는 자동 생성 문서 `docs/92_axis_gap.md`."),
    "S25": (s25_flip_before_after, "S25_flip_before_after.png", "P4",
            "측정 오류 네 건의 수정 전 · 후. 그림 S14 의 보조 그림이다. 수정 전 값 중 산출물이 남지 않은 것은 연구 기록의 기록값이다."),
    "S26": (s26_l6_conditions, "S26_l6_conditions.png", "P5",
            "무잡음 입력 보존 손실이 작동하는 세 조건(F-56). 잔차 연결이 있고 MAE 를 포함한 경우에만 무잡음 입력의 출력 SNR 이 크게 오른다."),
    "S27": (s27_cond_l6_interaction, "S27_cond_l6_interaction.png", "P6",
            "SNR 조건화와 무잡음 입력 보존 손실의 2x2 상호작용(F-47). 보존 손실 선이 거의 수평이므로 조건화를 더해도 추가 개선이 없다."),
    "S28": (s28_cond_error_cost, "S28_cond_error_cost.png", "P7",
            "조건 SNR 오차의 비대칭 비용(F-46). 과소 추정 쪽은 급하게 떨어지고 과대 추정 쪽은 완만하다. "
            "SNR 추정기가 과대 추정 쪽으로 편향되어 있으므로(F-13) 실제 비용은 작은 쪽에 해당한다."),
    "S29": (s29_metric_floor, "S29_metric_floor.png", "P8",
            "지표 분해능 대비 효과 크기. QRS 폭 오차는 MIT-BIH 에서 한 기법을 빼고 모든 기법의 차이가 판별 하한 아래에 있어 판정에 쓸 수 없다(F-16). 합성 ECG 에서는 모두 판별 하한 위에 있다."),
    "S30": (s30_phantom_threshold, "S30_phantom_threshold.png", "P9",
            "그림 S15 의 강건성 보조. 가성 QRS군 판정 임계값을 바꾸어도 칼만 필터가 가장 많다(F-58, D-42)."),
    "S31": (s31_findings_map, "S31_findings_map.png", "P10",
            "발견 기록 전체에서 보고서 8 장의 방법론적 요약이 유형으로 묶은 기록의 위치. 측정 전제의 오류 아홉 건이 연구 초기와 "
            "실시간 처리 연구에 모여 있다. 분류는 보고서 본문에서 읽으므로 보고서가 바뀌면 그림도 바뀐다."),
    "S32": (s32_step_optimum, "S32_step_optimum.png", "P11",
            "학습 곡선을 step 축으로 다시 그린 것. 데이터량과 학습량을 4 배로 늘려도 최적점의 step 수가 비슷하다(5.10.8 절의 부수 관측)."),
}


def write_index():
    md = ["# 97. 연구 과정 시각화 색인",
          "",
          "> 자동 생성: `python scripts/make_research_figs.py` (수정하지 말 것 — 스크립트를 고칠 것)",
          "",
          "시각화 제안서 `docs/96_visual_proposal.md` 의 후보 P1~P11 을 그린 것이다. 결과 그림 색인은",
          "`docs/93_slides.md` 이다. 수치는 모두 산출물 파일에서 읽으며, 수정 전 상태처럼 산출물이 남지 않은",
          "값만 연구 기록의 기록값을 쓰고 그림 안에 그렇게 적었다.",
          "",
          "| 그림 | 제안서 | 설명 |",
          "|---|---|---|"]
    for key, (_, f, pid, desc) in FIGS.items():
        cell = f"[`{f}`](../results/slides/research/{f})" if (OUT / f).exists() else f"`{f}` (생성되지 않음)"
        md.append(f"| {cell} | {pid} | {desc} |")
    Path("docs/97_research_figs.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("  docs/97_research_figs.md")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="연구 과정 시각화 S22~S32 (제안서 P1~P11)")
    ap.add_argument("--only", nargs="*", default=None, help="S22 S23 ... 일부만 생성")
    a = ap.parse_args()
    _ko_font()
    ensure_dir(OUT)
    want = a.only or list(FIGS)
    for k in want:
        FIGS[k][0]()
    save_manifest(OUT, cfg={"research_figs": want}, sources=["scripts/make_research_figs.py", "scripts/make_slides.py"])
    write_index()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
