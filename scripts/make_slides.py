#!/usr/bin/env python3
"""발표용 시각화 세트.

    python scripts/make_slides.py                 # 전부
    python scripts/make_slides.py --only S2 S5    # 일부만

산출: `results/slides/*.png`, 색인은 `docs/93_slides.md`.

**보고서 그림(`results/{tag}/report/`)과 목적이 다르다.** 보고서 그림은 모든
방법을 빠짐없이 싣는 기록용이고, 이것은 **화면에 띄워 설명하는 용도**다.
그래서 세 가지를 다르게 했다.

1. **방법을 6개로 줄였다.** 14 단은 프로젝터에서 읽히지 않는다.
2. **-5 dB 를 주력으로 쓴다.** 5 dB 에서는 출력들이 육안으로 거의 같아서
   "다 비슷하네" 로 끝난다. 차이가 보이는 곳에서 보여야 주장과 그림이 맞는다.
3. **잔차(출력 - 참조)를 나란히 그린다.** 출력만 보면 다 비슷하지만, 잔차를
   보면 무엇을 못 지웠고 무엇을 과하게 지웠는지가 바로 드러난다.

색은 눈으로 고르지 않고 검증했다 (Claude dataviz 팔레트 + 색각이상 분리도
검사). 오버레이에 쓰는 3종(M01/M04/M08)은 all-pairs 기준을 통과하고,
**`M_FE`(magenta)와 `M04`(orange)는 겹쳐 그리면 안 된다** — 정상시야 분리도가
기준 미달이라 패싯으로만 쓴다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np

from ecgdn.utils import ensure_dir, save_manifest

OUT = Path("results/slides")

# ---------------------------------------------------------------- 스타일
# dataviz 팔레트(light). 색은 **방법에 고정**한다 — 그림마다 바뀌면 안 된다.
C = {
    "M01": "#2a78d6",   # slot 1 blue    고전 bandpass
    "M04": "#eb6834",   # slot 2 orange  SWT thresholding
    "M08": "#1baf7a",   # slot 3 aqua    딥러닝 (wavelet U-Net)
    "M05": "#eda100",   # slot 4 yellow  Sameni EKS
    "M_FE": "#e87ba4",  # slot 5 magenta 공통 front-end 단독
}
# 손실 L1 -> L3 -> L6 은 **ordinal** 이다 — 순서를 바꾸면 의미가 달라진다
# (개입이 점점 커진다). 그래서 방법용 categorical 슬롯이 아니라 **단일 색조
# 램프**를 쓴다. blue 250/400/600 이고 validate_palette.js --ordinal 통과다
# (단조 L, 인접 간격 >= 0.06, 밝은 끝 2.06:1). 이 그림들에 `M01`(slot 1 blue)
# 은 등장하지 않으므로 색이 겹치지 않는다.
LOSS = {"L1": "#86b6ef", "L3": "#3987e5", "L6": "#184f95"}

# S9 는 **개입의 종류**(구조/손실) 둘만 가른다. 방법 색(C)을 쓰면 "M08 색"
# 같은 기존 의미와 충돌하므로, 여기서는 개입 종류에 색을 준다 — 구조는
# 중립 회색(결론이 "아무 일도 없다" 라 무채색이 맞다), 손실은 LOSS 램프의
# 진한 끝(L6)을 그대로 써서 S7·S8 과 이어지게 한다.
KIND = {"구조": "#8a8a8a", "손실": "#184f95"}

CLEAN = "#b8b6ae"       # 참조: 뒤에 두껍게 깔아 '목표' 로 읽히게
NOISY = "#52514e"       # 입력
INK, INK2 = "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"

ORDER = ["noisy", "M_FE", "M01", "M04", "M05", "M08"]
NAME_KO = {
    "clean": "참조 (정답)", "noisy": "입력 (잡음 섞임)",
    "M_FE": "M_FE  front-end 단독", "M01": "M01  Bandpass 0.5–40 Hz",
    "M04": "M04  SWT thresholding", "M05": "M05  Sameni EKS",
    "M08": "M08  딥러닝 (wavelet U-Net)",
}
NAME_EN = {
    "clean": "reference", "noisy": "input (noisy)",
    "M_FE": "M_FE  front-end only", "M01": "M01  bandpass 0.5-40 Hz",
    "M04": "M04  SWT thresholding", "M05": "M05  Sameni EKS",
    "M08": "M08  deep learning",
}
TXT_KO = {
    "time": "시간 [s]", "amp": "진폭 [mV]", "resid": "잔차 (출력 - 참조)",
    "output": "출력 파형", "in_snr": "입력 SNR [dB]",
    "gain": "front-end 대비 개선 [dB]",
}
TXT_EN = {
    "time": "time [s]", "amp": "amplitude [mV]", "resid": "residual (out - ref)",
    "output": "output", "in_snr": "input SNR [dB]",
    "gain": "gain over front-end [dB]",
}


def slide_style():
    """한글 폰트가 있으면 쓰고, 없으면 **영문 라벨로 자동 대체**한다.

    컨테이너에 폰트가 없으면 한글이 두부(□)로 렌더된다. 그림은 나오는데
    읽을 수 없는 상태가 가장 나쁘므로, 아예 영문으로 바꾼다.
    설치: `apt-get install -y fonts-nanum` 후 matplotlib 폰트 캐시 삭제.
    """
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    ko = next((f for f in ("NanumGothic", "NanumBarunGothic", "NanumSquare",
                           "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR")
               if f in have), None)
    plt.rcParams.update({
        "font.family": ko or "DejaVu Sans",
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#d8d6cf", "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK,
        "axes.grid": True, "grid.color": "#e8e6df", "grid.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "axes.titlesize": 12, "legend.frameon": False,
    })
    if ko is None:
        print("[slides] 한글 폰트가 없다 — 영문 라벨로 대체한다 "
              "(apt-get install -y fonts-nanum)")
    return (NAME_KO, TXT_KO) if ko else (NAME_EN, TXT_EN)


NAME, TXT = {}, {}


# ---------------------------------------------------------------- 데이터
def prepare(source: str, snr_db: float, noise: str = "mixed",
            record_idx: int = 0, methods: tuple[str, ...] = ("M_FE", "M01", "M04", "M05")):
    """지정 조건의 한 구간과 각 방법의 출력.

    **실험과 같은 경로로 만든다** — `build_eval_set` 과 `run_exp.build_methods`
    를 그대로 쓴다. 그림 전용 경로를 따로 두면 표와 그림이 어긋난다 (F-10).
    """
    sys.path.insert(0, "scripts")
    import yaml
    from run_exp import build_methods  # noqa: PLC0415

    from ecgdn.data.dataset import build_eval_set
    from ecgdn.data.nstdb import make_banks
    from ecgdn.data.sources import get_source, source_tag

    tag = source_tag(source)
    cfg = yaml.safe_load(Path("configs/exp_a.yaml").read_text())
    d = cfg.get("data", {})
    src = get_source(source, dur_s=float(d.get("dur_s", 300.0)),
                     n_test=int(d.get("n_test", 22)))
    banks = make_banks("test", d.get("nstdb_root", "data/raw/nstdb"))
    items = build_eval_set(src, "test", seg_s=float(d.get("seg_s", 60.0)),
                           snr_grid=[snr_db], noise_conditions=(noise,), banks=banks,
                           n_seg_per_record=1, seed=d.get("seed", "eval"))
    it = items[min(record_idx, len(items) - 1)]
    x, y, fs = it["x"].astype(float), it["y"].astype(float), float(it["fs"])

    mcfg = {"methods": list(methods), "frontend": True,
            "dl_methods": {"M08": {"ckpt": "results/{tag}/m08_l1/best.pt"}}}
    built = build_methods(mcfg, tag)
    outs = {}
    for mid, fn in built.items():
        ctx = {"x_clean": x} if getattr(fn, "needs_clean", False) else {}
        try:
            outs[mid] = np.asarray(fn(y, fs, ctx), dtype=float).ravel()
        except Exception as e:                                   # noqa: BLE001
            print(f"[warn] {mid} 실패: {type(e).__name__}: {e}")
    return dict(x=x, y=y, fs=fs, outs=outs, record=it["record"],
                r_peaks=np.asarray(it["r_peaks"], dtype=int), tag=tag)


def _window(d, dur=4.0, center_beat=None):
    """R-peak 에 맞춰 보기 좋은 구간을 고른다 (결정적)."""
    rp, fs = d["r_peaks"], d["fs"]
    k = len(rp) // 2 if center_beat is None else center_beat
    c = float(rp[k]) / fs if rp.size else 20.0
    t0 = max(6.0, c - dur / 2)
    i0, i1 = int(t0 * fs), int((t0 + dur) * fs)
    return i0, min(i1, len(d["x"]))


def _snr_imp(x, y, xh, fs=250.0):
    """`snr_imp_scaled` — 표와 같은 정의를 쓴다.

    직접 계산하지 않는다. `snr_db(x, err)` 는 **신호와 오차**를 받는데
    추정치를 넘기면 값이 조용히 틀린다 (실제로 그렇게 해서 모든 방법이
    +6.7 dB 로 같게 나왔다). guard band 도 평가와 같아야 한다.
    """
    from ecgdn.eval.engine import trim_guard
    from ecgdn.eval.signal_metrics import metrics_signal
    g = trim_guard(len(x), fs)
    return float(metrics_signal(x[g], y[g], xh[g])["snr_imp_scaled"])


# ---------------------------------------------------------------- S1 문제 제시
def s1_input(cache):
    """입력이 어떤 상태인가 — 축 2 × 입력 SNR 2.

    발표의 첫 장. "이 신호를 되살려야 한다" 를 보여준다.
    참조를 뒤에 회색으로 깔아 **목표가 무엇인지** 같이 보이게 했다.
    """
    # **sharex 를 쓰지 않는다.** D0 와 D1 은 서로 다른 기록의 다른 시각 구간이라
    # x 축을 공유하면 한쪽의 눈금이 다른 쪽 데이터에 붙는다. 시간은 구간
    # 시작을 0 으로 둔 상대시간으로 그린다 — 청중에게 절대 시각은 의미가 없다.
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 5.6))
    h = []
    for r, (src, axis_lab) in enumerate([("synthetic", "D0  합성 ECG"),
                                         ("mitdb", "D1  MIT-BIH 실기록")]):
        for c, snr in enumerate([5.0, -5.0]):
            d = cache[(src, snr, "mixed")]
            i0, i1 = _window(d, 4.0)
            t = (np.arange(i0, i1) - i0) / d["fs"]
            ax = axes[r, c]
            l1, = ax.plot(t, d["x"][i0:i1], color=CLEAN, lw=2.6,
                          solid_capstyle="round", label=NAME["clean"], zorder=1)
            l2, = ax.plot(t, d["y"][i0:i1], color=NOISY, lw=1.0,
                          label=NAME["noisy"], zorder=2)
            h = [l1, l2]
            ax.set_title(f"{axis_lab}   ·   입력 SNR {snr:+.0f} dB",
                         fontsize=11.5, color=INK)
            ax.set_xlim(t[0], t[-1])
            if c == 0:
                ax.set_ylabel(TXT["amp"])
            if r == 1:
                ax.set_xlabel(TXT["time"])
    fig.legend(handles=h, labels=[NAME["clean"], NAME["noisy"]],
               loc="upper right", ncol=2, fontsize=10,
               bbox_to_anchor=(0.995, 0.995))
    fig.suptitle("입력 신호 - 무엇을 되살려야 하는가\n"
                 "D1 의 큰 아래쪽 치우침이 기저선 변동(baseline wander)이다",
                 fontsize=12.5, x=0.02, ha="left", y=0.985, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.905))
    fig.savefig(OUT / "S1_input.png", dpi=170); plt.close(fig)
    print("  S1_input.png")


# ---------------------------------------------------------------- S2 방법 비교
def s2_methods(cache, src, snr=-5.0):
    """방법별 [출력 | 잔차]. **잔차 열이 이 그림의 핵심이다.**

    출력만 보면 방법들이 다 비슷해 보인다. 잔차(출력 - 참조)를 같은 y 축으로
    나란히 두면 무엇을 못 지웠는지가 한눈에 보인다.
    """
    d = cache[(src, snr, "mixed")]
    i0, i1 = _window(d, 4.0)
    t = (np.arange(i0, i1) - i0) / d["fs"]   # 상대시간 (S1 주석 참조)
    rows = [k for k in ORDER if k == "noisy" or k in d["outs"]]

    fig, axes = plt.subplots(len(rows), 2, figsize=(13.5, 1.28 * len(rows)),
                             sharex=True, sharey="col",
                             gridspec_kw={"width_ratios": [1.45, 1]})
    for ax, key in zip(axes[:, 0], rows):
        sig = d["y"] if key == "noisy" else d["outs"][key]
        col = NOISY if key == "noisy" else C[key]
        ax.plot(t, d["x"][i0:i1], color=CLEAN, lw=2.4, zorder=1)
        ax.plot(t, sig[i0:i1], color=col, lw=1.0, zorder=2)
        ax.set_ylabel(NAME[key], rotation=0, ha="right", va="center",
                      fontsize=9.5, color=INK)
        if key != "noisy":
            ax.text(0.995, 0.90, f"{_snr_imp(d['x'], d['y'], sig, d['fs']):+.1f} dB",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9.5, color=col, fontweight="bold")
    for ax, key in zip(axes[:, 1], rows):
        sig = d["y"] if key == "noisy" else d["outs"][key]
        col = NOISY if key == "noisy" else C[key]
        ax.axhline(0, color="#d8d6cf", lw=0.9, zorder=1)
        ax.plot(t, (sig - d["x"])[i0:i1], color=col, lw=0.85, zorder=2)

    axes[0, 0].set_title(TXT["output"] + "   (회색 = 참조)", fontsize=11)
    axes[0, 1].set_title(TXT["resid"] + "   — 0 에 붙을수록 좋다", fontsize=11)
    axes[-1, 0].set_xlabel(TXT["time"]); axes[-1, 1].set_xlabel(TXT["time"])
    axis_lab = "D0 합성 ECG" if src == "synthetic" else "D1 MIT-BIH 실기록"
    fig.suptitle(f"{axis_lab} · 기록 {d['record']} · 입력 SNR {snr:+.0f} dB · 혼합 잡음"
                 f"    (오른쪽 위 숫자 = SNR 개선)", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(OUT / f"S2_methods_{d['tag']}.png", dpi=170); plt.close(fig)
    print(f"  S2_methods_{d['tag']}.png")


# ---------------------------------------------------------------- S3 QRS 확대
def _beat_template(sig, r_peaks, fs, pre=0.25, post=0.35):
    """R-peak 정렬 beat 평균.

    **단일 beat 를 보여주면 안 되는 이유**: -5 dB 에서는 잔차가 대부분 잡음이라
    형태 왜곡이 묻힌다. beat 를 평균하면 잡음은 √N 으로 줄고(139 박이면 약
    21 dB) **체계적 왜곡만 남는다.** 형태 지표(`beat_cc`, `qrs_dur_err_ms`)가
    재는 것도 바로 그것이다.
    """
    a, b = int(pre * fs), int(post * fs)
    seg = [sig[r - a:r + b] for r in r_peaks
           if r - a >= 0 and r + b <= len(sig)]
    if not seg:
        return None, None
    m = np.mean(np.stack(seg), axis=0)
    t = (np.arange(-a, b) / fs) * 1000.0
    return t, m


def s3_qrs(cache, src, snr=-5.0):
    """beat 평균 템플릿 **오버레이** — 형태가 보존되는가.

    겹치는 3종(M01/M04/M08)은 all-pairs 색 검증을 통과한 조합이다.
    `M_FE`(magenta)는 `M04`(orange)와 정상시야 분리도가 기준 미달이라
    이 그림에 넣지 않는다 (패싯 그림에서만 쓴다).
    """
    d = cache[(src, snr, "mixed")]
    show = [m for m in ("M01", "M04", "M08") if m in d["outs"]]
    rp, fs = d["r_peaks"], d["fs"]

    t, ref = _beat_template(d["x"], rp, fs)
    if t is None:
        print("  [skip] S3 - beat 를 충분히 못 얻었다"); return
    tmpl = {m: _beat_template(d["outs"][m], rp, fs)[1] for m in show}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
    ax = axes[0]
    ax.plot(t, ref, color=CLEAN, lw=4.0, label=NAME["clean"], zorder=1,
            solid_capstyle="round")
    for m in show:
        ax.plot(t, tmpl[m], color=C[m], lw=1.9, label=NAME[m], zorder=3)
    ax.set_xlabel("R-peak 기준 시간 [ms]"); ax.set_ylabel(TXT["amp"])
    ax.set_title(f"beat {len(rp)} 개 평균 템플릿 - 형태가 보존되는가", fontsize=11.5)
    ax.legend(fontsize=9.5, loc="lower right")

    ax = axes[1]
    ax.axhline(0, color="#d8d6cf", lw=1.0, zorder=1)
    ax.axvspan(-50, 50, color="#efedE6", zorder=0)
    # 축 좌표로 놓는다 — 데이터 좌표는 아직 그리기 전이라 ylim 이 확정되지 않았다.
    ax.text(0.5, 0.985, "QRS 구간", transform=ax.transAxes, ha="center",
            va="top", fontsize=10, color=INK2)
    for m in show:
        ax.plot(t, tmpl[m] - ref, color=C[m], lw=1.9, zorder=3)
        j = int(np.argmax(np.abs(tmpl[m] - ref)))
        ax.annotate(m, (t[j], (tmpl[m] - ref)[j]), fontsize=10, color=C[m],
                    fontweight="bold", xytext=(5, 0),
                    textcoords="offset points", va="center")
    ax.set_xlabel("R-peak 기준 시간 [ms]")
    ax.set_ylabel("템플릿 오차 [mV]")
    ax.set_title("남은 것은 잡음이 아니라 체계적 왜곡이다", fontsize=11.5)

    axis_lab = "D0 합성 ECG" if src == "synthetic" else "D1 MIT-BIH 실기록"
    fig.suptitle(f"{axis_lab} · 기록 {d['record']} · 입력 SNR {snr:+.0f} dB"
                 f"    (beat 평균으로 잡음 진폭을 약 {np.sqrt(len(rp)):.0f} 배 줄였다)",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / f"S3_qrs_{d['tag']}.png", dpi=170); plt.close(fig)
    print(f"  S3_qrs_{d['tag']}.png")


# ---------------------------------------------------------------- S4 잡음 종류별
def s4_noise(cache, src, snr=-5.0):
    """잡음 종류가 답을 바꾼다 — 이 프로젝트의 가장 실용적인 결과.

    혼합 잡음 하나만 보여주면 이 구조가 보이지 않는다. 세 종류로 나누면
    **어느 계열이 어디에 강한지**가 드러난다: 기저선 변동은 front-end 가
    거의 다 해결하고, 임펄스는 딥러닝만 해결한다.
    """
    kinds = [("bw", "기저선 변동 (bw)"), ("ma", "근전도 (ma)"), ("impulse", "임펄스")]
    show = ["M_FE", "M04", "M08"]
    # **행 안에서 y 축을 공유한다.** 같은 조건을 비교하는 칸들의 세로 눈금이
    # 다르면 잘 지운 방법이 실제보다 좋아 보인다 (잘린 y 축과 같은 왜곡이다).
    fig, axes = plt.subplots(len(kinds), len(show) + 1,
                             figsize=(15, 2.5 * len(kinds)),
                             sharex="row", sharey="row")
    for r, (kind, klab) in enumerate(kinds):
        d = cache[(src, snr, kind)]
        i0, i1 = _window(d, 3.0)
        t = (np.arange(i0, i1) - i0) / d["fs"]
        ax = axes[r, 0]
        ax.plot(t, d["x"][i0:i1], color=CLEAN, lw=2.4)
        ax.plot(t, d["y"][i0:i1], color=NOISY, lw=0.9)
        ax.set_ylabel(klab, rotation=0, ha="right", va="center", fontsize=10, color=INK)
        if r == 0:
            ax.set_title(NAME["noisy"], fontsize=10.5)
        for c_, m in enumerate(show, start=1):
            ax = axes[r, c_]
            if m not in d["outs"]:
                ax.axis("off"); continue
            ax.plot(t, d["x"][i0:i1], color=CLEAN, lw=2.4)
            ax.plot(t, d["outs"][m][i0:i1], color=C[m], lw=1.0)
            ax.text(0.985, 0.90, f"{_snr_imp(d['x'], d['y'], d['outs'][m], d['fs']):+.1f} dB",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=10, color=C[m], fontweight="bold")
            if r == 0:
                ax.set_title(NAME[m], fontsize=10.5, color=C[m])
        for ax in axes[r]:
            ax.set_xlabel(TXT["time"] if r == len(kinds) - 1 else "")
    axis_lab = "D0 합성 ECG" if src == "synthetic" else "D1 MIT-BIH 실기록"
    fig.suptitle(f"잡음 종류가 답을 바꾼다 — {axis_lab} · 입력 SNR {snr:+.0f} dB"
                 f"    (숫자 = SNR 개선 · 행마다 같은 세로 눈금)",
                 fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    tag = cache[(src, snr, kinds[0][0])]["tag"]
    fig.savefig(OUT / f"S4_noise_{tag}.png", dpi=165); plt.close(fig)
    print(f"  S4_noise_{tag}.png")


# ---------------------------------------------------------------- S5 교차 곡선
def s5_crossover():
    """**핵심 슬라이드.** 딥러닝이 값을 하는 SNR 범위가 축마다 다르다.

    기준선은 `M_FE`(공통 front-end) 다 — 참조와 같은 대역이라 '딥러닝이
    front-end 위에 무엇을 더하는가' 를 재는 올바른 비교다. `M01` 을 기준으로
    쓰면 대역이 달라 왜곡된다 (보고서 5.8.6).
    """
    import pandas as pd

    from ecgdn.eval.stats import compare_methods

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    styles = {"d0": (C["M01"], "D0  합성 ECG", "o", "-"),
              "d1": (C["M04"], "D1  MIT-BIH 실기록", "s", "-")}
    for tag, (col, lab, mk, ls) in styles.items():
        p = Path("results") / tag / "exp_a" / "metrics.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        xs, ys, sig = [], [], []
        for s in sorted(df.snr_in_target.dropna().unique()):
            t = compare_methods(df[df.snr_in_target == s], "snr_imp_scaled", "M_FE")
            r = t[t.method == "M08"]
            if r.empty:
                continue
            xs.append(float(s)); ys.append(float(r.iloc[0].delta_mean))
            sig.append(bool(r.iloc[0].p_holm < 0.05))
        ax.plot(xs, ys, ls, color=col, lw=2.4, zorder=3, label=lab)
        xs, ys, sig = np.array(xs), np.array(ys), np.array(sig)
        ax.scatter(xs[sig], ys[sig], s=95, color=col, marker=mk, zorder=4,
                   edgecolor=SURFACE, linewidth=1.6)
        ax.scatter(xs[~sig], ys[~sig], s=95, facecolor=SURFACE, marker=mk,
                   zorder=4, edgecolor=col, linewidth=1.8)
        # 직접 라벨은 두 곡선이 가장 벌어지는 오른쪽 끝에 둔다.
        ax.annotate(lab, (xs[-1], ys[-1]), color=col, fontsize=11,
                    fontweight="bold", ha="right",
                    xytext=(-8, 10 if tag == "d0" else -18),
                    textcoords="offset points")
        # 0 을 지나는 지점 (인접 두 점 선형보간)
        for i in range(len(xs) - 1):
            if ys[i] > 0 >= ys[i + 1]:
                zx = xs[i] + (xs[i + 1] - xs[i]) * ys[i] / (ys[i] - ys[i + 1])
                ax.axvline(zx, color=col, ls=":", lw=1.6, zorder=2)
                ax.annotate(f"{zx:.0f} dB", (zx, ax.get_ylim()[0]), color=col,
                            fontsize=10.5, fontweight="bold", ha="center",
                            xytext=(0, 6), textcoords="offset points")
                break
    ax.axhline(0, color=INK2, lw=1.2, zorder=2)
    ax.set_xlabel(TXT["in_snr"]); ax.set_ylabel(TXT["gain"])
    ax.set_title("딥러닝(M08)이 공통 front-end 위에 더하는 것\n"
                 "채운 표식 = 통계적으로 유의 (paired Wilcoxon + Holm, 22 기록)",
                 fontsize=12)
    ax.text(0.015, 0.03, "0 아래 = front-end 만 쓰는 편이 낫다",
            transform=ax.transAxes, fontsize=10, color=INK2)
    # 표식의 채움 여부가 무엇을 뜻하는지는 직접 라벨로는 전할 수 없다.
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color=INK2, marker="o", ls="", markersize=9,
               label="유의 (Holm 보정 p < 0.05)"),
        Line2D([], [], color=INK2, marker="o", ls="", markersize=9,
               markerfacecolor=SURFACE, label="유의하지 않음"),
    ], loc="lower left", fontsize=9.5, bbox_to_anchor=(0.0, 0.08))
    fig.tight_layout()
    fig.savefig(OUT / "S5_crossover.png", dpi=175); plt.close(fig)
    print("  S5_crossover.png")


# ---------------------------------------------------------------- S6 안전성
def s6_safety():
    """딥러닝이 없는 파형을 지어내는가 — 통념과 반대 결과.

    beat 를 지우고 잡음을 덮은 뒤, 출력이 그 자리에 만든 에너지를 원래
    에너지로 나눈 값. **낮을수록 안전**하다.
    """
    import re
    rows = {}
    for tag, f in (("d0", "docs/07_safety_probe_d0.md"),
                   ("d1", "docs/07_safety_probe_d1.md")):
        p = Path(f)
        if not p.exists():
            continue
        txt = p.read_text()
        for probe, head in (("P1", "## P1"), ("P2", "## P2")):
            sec = txt[txt.index(head):]
            sec = sec[:sec.index("## P", 3)] if "## P" in sec[3:] else sec
            for m in re.finditer(r"^\| `(\w+)` \| ([\d.]+) \|", sec, re.M):
                rows[(tag, probe, m.group(1))] = float(m.group(2))
    if not rows:
        print("  [skip] S6 — 안전성 프로브 문서가 없다"); return

    show = ["M_FE", "M01", "M04", "M05", "M08"]
    axis_lab = {"d0": "D0  합성 ECG", "d1": "D1  MIT-BIH 실기록"}
    probes = [("P1", "beat 1 개 소실"), ("P2", "3 초 asystole")]
    tags = [t for t in ("d0", "d1") if any(k[0] == t for k in rows)]

    # **축을 alpha 로 구분하지 않는다.** 색은 이미 방법을 나타내고 있으므로
    # 거기에 밝기로 두 번째 차원을 얹으면 둘 다 읽기 어려워지고, 범례가
    # 방법 색을 축 색인 것처럼 보여준다. 패싯으로 나눈다.
    fig, axes = plt.subplots(len(probes), len(tags),
                            figsize=(5.6 * len(tags), 3.5 * len(probes)),
                            sharey="row", squeeze=False)
    for r, (probe, plab) in enumerate(probes):
        for c, tag in enumerate(tags):
            ax = axes[r][c]
            vals = [rows.get((tag, probe, m), np.nan) for m in show]
            ax.bar(np.arange(len(show)), vals, 0.62,
                   color=[C[m] for m in show], edgecolor=SURFACE,
                   linewidth=1.5, zorder=3)
            for i, v in enumerate(vals):
                if np.isfinite(v):
                    ax.text(i, v, f"{v:.3f}", ha="center", va="bottom",
                            fontsize=9.5, color=INK2, zorder=4)
            ax.set_xticks(np.arange(len(show)))
            ax.set_xticklabels(show, fontsize=10)
            ax.grid(axis="x", visible=False)
            ax.margins(y=0.18)
            if r == 0:
                ax.set_title(axis_lab[tag], fontsize=11.5, color=INK)
            if c == 0:
                ax.set_ylabel(plab, fontsize=11, color=INK)
    fig.supylabel("출력이 그 자리에 만들어낸 에너지 / 원래 에너지",
                  fontsize=10, color=INK2)
    fig.suptitle("없는 파형을 지어내는가 — 낮을수록 안전\n"
                 "딥러닝(M08)이 두 축 · 두 프로브 모두에서 가장 낮다", fontsize=12.5)
    fig.tight_layout(rect=(0.015, 0, 1, 0.95))
    fig.savefig(OUT / "S6_safety.png", dpi=175); plt.close(fig)
    print("  S6_safety.png")


# ---------------------------------------------------------------- S7 손실
def s7_loss_gap(TXT):
    """손실을 바꾸면 고 SNR 열세가 되돌아온다.

    y 는 **`M_FE` 대비 격차**다 — S5 와 같은 축이라 두 슬라이드가 이어 읽힌다.
    0 아래면 "front-end 만 쓰는 편이 낫다" 는 뜻이고, L1 -> L3 -> L6 이
    그 선을 어떻게 밀어 올리는지가 이 그림의 전부다.

    색은 categorical 슬롯이 아니라 **단일 색조 ordinal 램프**다 — L1/L3/L6 은
    순서가 의미를 갖기 때문이다(개입이 커진다). 색만으로 식별하지 않도록
    범례와 직접 라벨을 함께 둔다.
    """
    import pandas as pd
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), sharey=True)
    axis_lab = {"d0": "D0 합성", "d1": "D1 MIT-BIH"}
    for ax, tag in zip(axes, ("d0", "d1")):
        p = Path("results") / tag / "abl_loss" / "metrics.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        s = df[df.metric == "snr_imp_scaled"]
        w = s.pivot_table(index=["record", "snr_in_target"],
                          columns="method", values="value").reset_index()
        snrs = sorted(w.snr_in_target.unique())
        cross = {}
        for loss in ("L1", "L3", "L6"):
            col_m = f"M06-{loss}"
            if col_m not in w:
                continue
            ys = [w[w.snr_in_target == q][col_m].mean()
                  - w[w.snr_in_target == q]["M_FE"].mean() for q in snrs]
            ax.plot(snrs, ys, "-o", color=LOSS[loss], lw=2.4, markersize=9,
                    markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3,
                    label=loss)
            ax.annotate(loss, (snrs[-1], ys[-1]), color=LOSS[loss], fontsize=11,
                        fontweight="bold", ha="left",
                        xytext=(8, -3), textcoords="offset points")
            cross[loss] = ys
        ax.axhline(0, color=INK2, lw=1.2, zorder=2)
        # 0 을 지나는 지점 = "여기부터는 front-end 만 쓰는 편이 낫다".
        # 손실을 바꾸면 이 점이 오른쪽으로 밀리는 것이 이 슬라이드의 요지다.
        for loss, ys in cross.items():
            zx = None
            for i in range(len(snrs) - 1):
                if ys[i] > 0 >= ys[i + 1]:
                    zx = snrs[i] + (snrs[i + 1] - snrs[i]) * ys[i] / (ys[i] - ys[i + 1])
                    break
            if zx is None:
                continue
            ax.plot([zx], [0], marker="v", color=LOSS[loss], markersize=9,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)
            ax.annotate(f"{zx:.0f}", (zx, 0), color=LOSS[loss], fontsize=10,
                        fontweight="bold", ha="center", va="top",
                        xytext=(0, -12), textcoords="offset points")
        ax.set_title(axis_lab[tag], fontsize=11.5, color=INK)
        ax.set_xlabel(TXT["in_snr"]); ax.set_xticks(snrs)
        ax.margins(x=0.16)
    axes[0].set_ylabel(TXT["gain"])
    # 범례는 figure 수준에 가로로 둔다 — 축 안에 두면 주석과 겹친다.
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.845), ncol=3,
               fontsize=10, columnspacing=1.8, handletextpad=0.5)
    axes[0].text(0.03, 0.06, "0 아래 = front-end 만 쓰는 편이 낫다",
                 transform=axes[0].transAxes, fontsize=9.5, color=INK2)
    axes[1].text(0.03, 0.06, "▼ = 0 을 지나는 입력 SNR",
                 transform=axes[1].transAxes, fontsize=9.5, color=INK2)
    fig.suptitle("손실을 바꾸면 고 SNR 열세가 되돌아온다  (M06, TEST)\n"
                 "20 dB 에서 D0 는 부호가 뒤집히고(-1.6 → +3.0), "
                 "D1 은 격차의 74 %가 사라진다(-4.8 → -1.3)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / "S7_loss_gap.png", dpi=175); plt.close(fig)
    print("  S7_loss_gap.png")


# ---------------------------------------------------------------- S8 clean 보존
def s8_clean_preservation(TXT):
    """L6 의 **기제** — 깨끗한 신호를 얼마나 덜 건드리는가 (EXP-C).

    이 그림이 S7 의 "왜" 다. 손실에 "clean 을 건드리지 마라" 를 넣었더니
    바로 그 지표가 올라갔다는 것을 보인다.

    막대는 모델 × 손실의 **ordinal 쌍**이고, 참조 둘(`M04` SWT, `M00`
    front-end 몫)은 막대가 아니라 **가로선**으로 둔다 — 비교 대상이 아니라
    눈금이기 때문이다.

    **읽어야 하는 것은 막대 높이가 아니라 SWT 선까지의 거리**이므로, 각
    막대에 그 격차를 숫자로 붙인다. 초판은 붙이지 않아서 정작 요점(17 dB
    → 1.8 dB)이 그림에서 안 보였다.
    """
    import pandas as pd
    from matplotlib.transforms import blended_transform_factory as blend

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharey=False)
    axis_lab = {"d0": "D0 합성", "d1": "D1 MIT-BIH"}
    for ax, tag in zip(axes, ("d0", "d1")):
        p = Path("results") / tag / "exp_c" / "metrics.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        s = df[df.metric == "snr_out_strict"].replace([np.inf, -np.inf], np.nan)
        g = s.groupby("method")["value"].mean()
        ref = g.get("M04", np.nan)
        models, w = ["M06", "M08"], 0.34
        for i, loss in enumerate(("L1", "L6")):
            vals = [g.get(m if loss == "L1" else f"{m}-{loss}", np.nan) for m in models]
            xs = np.arange(len(models)) + (i - 0.5) * (w + 0.02)
            ax.bar(xs, vals, width=w, color=LOSS[loss], zorder=3,
                   edgecolor=SURFACE, linewidth=2, label=loss)
            for x, v in zip(xs, vals):
                if not np.isfinite(v):
                    continue
                ax.annotate(f"{v:.1f}", (x, v), ha="center", va="bottom",
                            fontsize=10.5, color=INK, zorder=4,
                            xytext=(0, 3), textcoords="offset points")
                if np.isfinite(ref):
                    # 요점은 높이가 아니라 SWT 까지 남은 거리다. 다만 막대 폭이
                    # 좁아 "SWT 까지 28.5" 는 잘린다 — 기호로 줄이고 뜻은
                    # 부제에 적는다.
                    ax.annotate(f"Δ{ref - v:.1f}", (x, v), ha="center",
                                va="top", fontsize=10, color=SURFACE, zorder=4,
                                fontweight="bold",
                                xytext=(0, -7), textcoords="offset points")
        # 참조선 — 막대 오른쪽에 **빈 띠**를 만들고 거기에 라벨을 둔다.
        # 초판은 x=len(models)-0.45 라 축 밖으로 잘렸고, 축 좌표로 옮기자
        # 이번에는 막대 값 라벨과 겹쳤다. 겹칠 자리를 아예 비우는 것이
        # 두 문제를 한 번에 없앤다.
        ax.set_xlim(-0.62, len(models) - 1 + 1.05)
        for key, lab, ls in (("M04", "M04 SWT", "--"), ("M00", "M00 무처리", ":")):
            v = g.get(key, np.nan)
            if not np.isfinite(v):
                continue
            ax.axhline(v, color=INK2, ls=ls, lw=1.5, zorder=2)
            ax.annotate(f"{lab}\n{v:.1f}", (len(models) - 1 + 0.32, v), color=INK2,
                        fontsize=9.5, ha="left", va="center", zorder=5)
        ax.set_xticks(np.arange(len(models))); ax.set_xticklabels(models, fontsize=11)
        ax.grid(axis="x", visible=False)
        ax.set_title(axis_lab[tag], fontsize=11.5, color=INK)
        ax.margins(y=0.26)
    axes[0].set_ylabel("깨끗한 신호 통과 시 출력 SNR [dB] ↑")
    axes[0].legend(title="손실", loc="upper left", fontsize=10, title_fontsize=10,
                   framealpha=0.9)
    fig.suptitle("L6 는 무엇을 고쳤나 — 깨끗한 신호를 덜 건드린다 (EXP-C)\n"
                 "D1 에서 SWT 와의 격차가 17.9 dB 에서 1.8 dB 로 줄었다\n"
                 "막대 안 Δ = SWT 선까지 남은 거리 [dB]", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(OUT / "S8_clean.png", dpi=175); plt.close(fig)
    print("  S8_clean.png")


# ---------------------------------------------------------------- S9 구조 vs 손실
def s9_structure_vs_loss(TXT):
    """이 프로젝트의 실용적 결론 한 장 — **구조를 바꿔도 안 되고 손실은 됐다.**

    네 번의 구조 변경과 손실 변경을 **같은 자(M06/M08 기준 Δ)** 로 나란히
    놓는다. 한 번의 null 로는 "그 구조가 나빴다" 와 구분되지 않으므로,
    **네 번이 모두 실린 것이 이 그림의 논거**다.

    가로 점 그림을 쓴다 — 항목이 이름이 길고 개수가 적으며, 읽어야 하는 것이
    "0 에서 얼마나 떨어졌나" 라서다. 막대는 0 이 기준선인데 여기서는 음수도
    의미가 있어 점이 낫다.
    """
    import pandas as pd
    from ecgdn.eval.stats import compare_methods

    # (라벨, 종류, 파일, 기준, 대상)
    ROWS = [
        ("M07  SWT 를 전처리로",      "구조", "exp_a",      "M06",       "M07"),
        ("M08  wavelet 표현공간",     "구조", "exp_a",      "M06",       "M08"),
        ("M10  해상도 유지",          "구조", "exp_a",      "M06",       "M10"),
        ("M09  전역 attention",       "구조", "exp_a",      "M06",       "M09"),
        ("window 4배 (16.4 s)",       "구조", "abl_window", "M06-w1024", "M06-w4096"),
        ("L3   +차분항",              "손실", "abl_loss",   "M06-L1",    "M06-L3"),
        ("L6   +clean 보존",          "손실", "abl_loss",   "M06-L1",    "M06-L6"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), sharey=True)
    for ax, tag in zip(axes, ("d0", "d1")):
        ys, xs, ps, cs = [], [], [], []
        for i, (lab, kind, exp, base, cand) in enumerate(ROWS):
            f = Path("results") / tag / exp / "metrics.parquet"
            if not f.exists():
                ys.append(i); xs.append(np.nan); ps.append(1.0); cs.append(KIND[kind]); continue
            d = pd.read_parquet(f)
            try:
                r = compare_methods(d, "snr_imp_scaled", base, unit="record")
                r = r[r.method == cand]
            except KeyError:
                r = None
            ys.append(i)
            if r is None or r.empty:
                xs.append(np.nan); ps.append(1.0)
            else:
                xs.append(float(r.iloc[0]["delta_mean"])); ps.append(float(r.iloc[0]["p_holm"]))
            cs.append(KIND[kind])
        ax.axvline(0, color=INK2, lw=1.2, zorder=2)
        for y, x, pv, c in zip(ys, xs, ps, cs):
            if not np.isfinite(x):
                continue
            sig = pv < 0.05
            ax.scatter([x], [y], s=150 if sig else 95, color=c, zorder=4,
                       edgecolor=SURFACE if sig else c, linewidth=1.8,
                       marker="o" if sig else "o", alpha=1.0 if sig else 0.45)
            ax.annotate(f"{x:+.2f}" + ("*" if sig else ""), (x, y),
                        xytext=(0, 11), textcoords="offset points",
                        ha="center", fontsize=9.5,
                        color=INK if sig else INK2,
                        fontweight="bold" if sig else "normal")
        ax.set_yticks(range(len(ROWS)))
        ax.set_ylim(-0.7, len(ROWS) - 0.3)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title({"d0": "D0 합성", "d1": "D1 MIT-BIH"}[tag], fontsize=11.5, color=INK)
        ax.set_xlabel("기준 대비 Δ  snr_imp_scaled [dB]")
        ax.margins(x=0.20)
    axes[0].set_yticklabels([r[0] for r in ROWS], fontsize=10.5)
    # 범례 — 색이 뜻하는 것은 방법이 아니라 **개입의 종류**다
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[
        Line2D([], [], marker="o", ls="", ms=9, color=KIND["구조"], label="구조 변경"),
        Line2D([], [], marker="o", ls="", ms=9, color=KIND["손실"], label="손실 변경"),
        Line2D([], [], marker="o", ls="", ms=9, color=INK2, label="* = p < 0.05 (Holm)"),
    ], loc="upper right", fontsize=9.5, framealpha=0.92,
        borderpad=0.6, labelspacing=0.4)
    fig.suptitle("구조를 네 번 바꿔도 안 됐고, 손실은 됐다  (M06 기준, TEST n=22)\n"
                 "유의한 구조 변경은 M07 하나인데 그것은 나쁜 쪽이다",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(OUT / "S9_structure_vs_loss.png", dpi=175); plt.close(fig)
    print("  S9_structure_vs_loss.png")


# ---------------------------------------------------------------- main
INDEX = [
    ("S1_input.png",
     "**도입.** 무엇을 되살려야 하는가. D1 의 큰 아래쪽 치우침이 기저선 변동이다."),
    ("S2_methods_d0.png",
     "D0 방법별 출력 + 잔차. **잔차 열이 핵심** — 출력만 보면 다 비슷하다."),
    ("S2_methods_d1.png",
     "D1 같은 그림. M08 +21.8 dB vs 나머지 +17.3~17.6 dB, 잔차가 눈에 띄게 평평하다."),
    ("S3_qrs_d0.png",
     "D0 beat 평균 템플릿 — 잡음을 평균으로 지우고 **체계적 왜곡만** 남긴다."),
    ("S3_qrs_d1.png",
     "D1 같은 그림. **M01(bandpass)이 QRS 를 가장 크게 왜곡**(±0.09 mV)하고 "
     "M04(SWT)는 거의 평평하다 — 40 Hz 절단의 대가다."),
    ("S4_noise_d0.png",
     "D0 잡음 종류별. 어느 계열이 어디에 강한가."),
    ("S4_noise_d1.png",
     "**가장 설득력 있는 그림.** 기저선 변동은 front-end 가 다 해결하고(M_FE ≈ M04 "
     "≈ +20.0 dB), 임펄스는 딥러닝만 해결한다(+6.4 → +20.1 dB). "
     "임펄스 행에서 스파이크가 M_FE·M04 에 그대로 남아 있는 것이 눈으로 보인다."),
    ("S5_crossover.png",
     "**핵심 슬라이드.** 딥러닝이 front-end 위에 더하는 값이 입력 SNR 에 따라 "
     "줄고, **D1 에서는 13 dB 부근에서 0 을 지난다.** 저 SNR(-5 dB)에서는 두 축이 "
     "거의 같다(+7.9 vs +7.1) — 합성이 딥러닝을 과대평가한 것이 아니라 "
     "**도움이 되는 SNR 범위**가 좁아진 것이다."),
    ("S6_safety.png",
     "없는 파형을 지어내는가. **딥러닝이 두 축·두 프로브 모두에서 가장 낮다** — "
     "residual 구조(출력 = 입력 - 예측잡음)의 직접적 결과이고, "
     "\"딥러닝이 파형을 지어낸다\" 는 통념과 반대다."),
    ("S7_loss_gap.png",
     "**S5 의 후속.** S5 가 보인 고 SNR 열세를 **손실만 바꿔서** 되돌린다. "
     "20 dB 에서 D0 는 부호가 뒤집히고(-1.6 → +3.0) D1 은 격차의 74 %가 "
     "사라진다(-4.8 → -1.3). 구조를 세 번 바꿔도(M07·M08·M10) 안 되던 것이 "
     "손실 한 항으로 움직였다는 것이 이 슬라이드의 요지다."),
    ("S9_structure_vs_loss.png",
     "**결론 한 장.** 구조 4연속 기각 vs 손실 성공. 한 번의 null 로는 "
     "'그 구조가 나빴다' 와 구분되지 않으므로 넷을 함께 싣는다."),
    ("S8_clean.png",
     "**S7 의 '왜'.** L6 는 \"입력이 이미 깨끗하면 건드리지 마라\" 를 손실에 "
     "넣은 것인데, **바로 그 지표(EXP-C)가 올라갔다.** D1 에서 M06 22.2 → "
     "37.4 dB, M08 23.5 → 38.4 dB 로 SWT(40.1)와의 17 dB 격차가 1.8 dB 가 "
     "된다. 이득이 우연이 아니라 **의도한 기제를 통해** 왔다는 근거다."),
    ("S10_loss_by_noise.png",
     "**S7·S8 의 범위.** L6 의 이득이 어디까지 가는가 — 잡음 7 종 × 입력 SNR "
     "3 단계(EXP-G). **결론의 축은 잡음 종류가 아니라 입력 SNR 이다**: "
     "20 dB 에서는 손해가 한 칸도 없고, 손해 다섯 칸은 전부 0 dB 다. "
     "구조가 뚜렷한 잡음(전원선 +6.3, 기저선 변동 +4.6 평균)에서 크게 벌고 "
     "광대역 백색잡음(+0.3)에서 가장 적게 번다 — 잡음이 신호와 분리 가능해야 "
     "'건드리지 않는다' 가 선택지가 되기 때문이다."),
]


def write_index():
    md = ["# 93. 발표용 그림 색인",
          "",
          "> 자동 생성: `python scripts/make_slides.py` "
          "(수정하지 말 것 — 스크립트를 고칠 것)",
          "",
          "보고서 그림(`results/{d0,d1}/report/`)은 **모든 방법을 빠짐없이 싣는 기록용**",
          "이고, 이것은 **화면에 띄워 설명하는 용도**다. 방법을 6 개로 줄이고,",
          "차이가 보이는 -5 dB 를 주력으로 쓰고, 잔차를 나란히 그린다.",
          "",
          "한글 폰트가 없는 환경에서는 라벨이 영문으로 자동 대체된다",
          "(`apt-get install -y fonts-nanum` 후 matplotlib 폰트 캐시 삭제).",
          "",
          "| 그림 | 무엇을 보여주는가 |",
          "|---|---|"]
    for f, desc in INDEX:
        md.append(f"| [`{f}`](../results/slides/{f}) | {desc} |" if (OUT / f).exists()
                  else f"| `{f}` | (생성되지 않음) — {desc} |")
    md += ["", "## 색 배정 (그림마다 바뀌지 않는다)", "",
           "| 방법 | 색 |", "|---|---|"]
    for m, col in C.items():
        md.append(f"| `{m}` | `{col}` |")
    md += ["", "참조는 회색 굵은 선으로 뒤에 깔고, 입력(잡음)은 진한 중성색이다.",
           "색은 dataviz 팔레트에서 가져와 **색각이상 분리도를 검증**했다.",
           "`M_FE`(magenta)와 `M04`(orange)는 정상시야 분리도가 기준 미달이라",
           "**겹쳐 그리지 않는다** — 패싯으로만 쓴다.", ""]
    Path("docs/93_slides.md").write_text("\n".join(md) + "\n")
    print("  docs/93_slides.md")


# -------------------------------------------------- S10 L6 의 잡음 × SNR 격자
# **형태**: 값의 일이 *극성*이다 — "L6 이 도왔나 해쳤나" 이고, 0 이 의미 있는
# 중립점이다. 그래서 diverging 히트맵이다. 잡음 7 종을 categorical 색 7 개로
# 그리면 식별이 색에 얹히는데, 여기서 묻는 것은 잡음의 정체가 아니라 **부호와
# 크기**라 색을 그쪽에 써야 한다.
#
# **색**: dataviz 규약대로 blue <-> red 두 극 + 회색 중립점(#f0efec). 두 극
# (#184f95 / #d03b3b)은 validate_palette.js --mode light 전 항목 PASS 다
# (CVD ΔE 17.2 protan, normal 31.8). 파랑을 이득 쪽에 둔 것은 S7·S8 의 손실
# 램프가 파랑이라 "손실 개입 = 파랑" 이 이어지기 때문이다.
#
# **이중 부호화**: 색만으로 읽지 않게 각 칸에 숫자를 적고, 유의한 칸만
# 굵게 + 테두리를 준다. 색맹·흑백 인쇄에서도 판정이 남는다.
DIVERGE = ["#d03b3b", "#e88080", "#f6c9c9", "#f0efec",
           "#c6dcf5", "#7fb2ec", "#3987e5", "#184f95"]


def s10_loss_by_noise(TXT):
    """L6 − L1 을 잡음 × 입력 SNR 격자로. 결론의 축이 SNR 이라는 것이 요지다."""
    import numpy as np
    import pandas as pd
    from matplotlib.colors import BoundaryNorm, ListedColormap

    from ecgdn.eval.stats import holm, paired_wilcoxon

    CONDS = ["pli", "bw_synth", "ma_synth", "mixed", "em_synth", "impulse", "awgn"]
    LAB = {"mixed": "혼합", "impulse": "임펄스", "pli": "전원선 60 Hz",
           "bw_synth": "기저선 변동", "ma_synth": "근전도", "em_synth": "전극 움직임",
           "awgn": "백색잡음"}
    # 경계는 0 을 **정확히** 가운데 두고 대칭으로 — 극성이 이 그림의 전부다.
    bounds = [-99, -2, -1, -0.001, 0.001, 1, 3, 6, 99]
    cmap = ListedColormap(DIVERGE)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    got = False
    for ax, tag in zip(axes, ("d0", "d1")):
        f = Path("results") / tag / "exp_g" / "metrics.parquet"
        if not f.exists():
            ax.axis("off"); continue
        df = pd.read_parquet(f)
        df = df[df.metric == "snr_imp_scaled"]
        snrs = sorted(df.snr_in_target.unique())
        D = np.full((len(CONDS), len(snrs)), np.nan)
        S = np.zeros_like(D, dtype=bool)
        for j, snr in enumerate(snrs):
            ps = []
            for i, cond in enumerate(CONDS):
                s = df[(df.cond == cond) & (df.snr_in_target == snr)]
                w = s.pivot_table(index=["record", "seg"], columns="method",
                                  values="value")
                if "M06L6" not in w or "M06" not in w:
                    ps.append(1.0); continue
                pr = w[["M06L6", "M06"]].dropna()
                _, pv = paired_wilcoxon(pr["M06L6"].to_numpy(), pr["M06"].to_numpy())
                D[i, j] = float((pr["M06L6"] - pr["M06"]).mean()); ps.append(pv)
            # 보정은 **한 축·한 SNR 안의 잡음 7 종**에 건다 (11_loss_by_noise 와 동일)
            S[:, j] = holm(np.asarray(ps)) < 0.05
        got = True
        ax.imshow(D, cmap=cmap, norm=norm, aspect="auto", zorder=0)
        ax.grid(False)      # rcParams 의 격자가 칸 위에 흰 줄로 얹힌다
        for i in range(len(CONDS)):
            for j in range(len(snrs)):
                if np.isnan(D[i, j]):
                    continue
                # 짙은 칸 위에서는 흰 글씨라야 읽힌다
                dark = D[i, j] >= 3 or D[i, j] <= -2
                # 유의는 **별표**로 표시한다 — 굵기와 색 말고 흑백에서도 남는
                # 세 번째 신호가 필요하다. 테두리를 써 봤는데 칸이 납작해서
                # 가로줄로 읽혔다.
                txt = f"{D[i, j]:+.1f}" + ("*" if S[i, j] else "")
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=10.5, color=("#ffffff" if dark else INK),
                        fontweight="bold" if S[i, j] else "normal")
        ax.set_xticks(range(len(snrs)))
        ax.set_xticklabels([f"{s:g} dB" for s in snrs], fontsize=10.5)
        ax.set_yticks(range(len(CONDS)))
        ax.set_yticklabels([LAB[c] for c in CONDS], fontsize=10.5)
        ax.set_xlabel(TXT["in_snr"])
        ax.set_title({"d0": "D0 합성", "d1": "D1 MIT-BIH"}[tag],
                     fontsize=11.5, color=INK)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    if not got:
        plt.close(fig); print("  S10 건너뜀 (exp_g 없음)"); return
    # 주석은 축이 아니라 figure 에 둔다 — 축에 두면 x 라벨과 겹친다.
    fig.text(0.5, 0.035, "* 와 굵은 숫자 = Holm 보정 후 유의  ·  "
             "파랑 = L6 이 이김, 빨강 = L6 이 짐  ·  값은 M06L6 - M06 [dB]",
             ha="center", fontsize=9.5, color=INK2)
    fig.suptitle("L6 의 이득은 잡음 종류가 아니라 입력 SNR 이 가른다  "
                 "(M06, TEST, 기록 단위 n=44)\n"
                 "20 dB 에서는 손해가 한 칸도 없다. 손해 다섯 칸은 전부 0 dB 이고, "
                 "구조가 뚜렷한 잡음일수록 크게 번다", fontsize=12.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.86))
    fig.savefig(OUT / "S10_loss_by_noise.png", dpi=175); plt.close(fig)
    print("  S10_loss_by_noise.png")


def main() -> int:
    import argparse
    global NAME, TXT
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="S1 S2 ... 일부만 생성")
    ap.add_argument("--snr", type=float, default=-5.0,
                    help="파형 그림의 입력 SNR (기본 -5 dB — 차이가 보이는 구간)")
    a = ap.parse_args()
    want = set(a.only) if a.only else {"S1", "S2", "S3", "S4", "S5", "S6",
                                       "S7", "S8", "S10"}

    NAME, TXT = slide_style()
    ensure_dir(OUT)

    # 필요한 조건만 준비한다 (Sameni EKS 가 느리다).
    need = set()
    if "S1" in want:
        need |= {(s, v, "mixed") for s in ("synthetic", "mitdb") for v in (5.0, a.snr)}
    if {"S2", "S3"} & want:
        need |= {(s, a.snr, "mixed") for s in ("synthetic", "mitdb")}
    if "S4" in want:
        need |= {(s, a.snr, k) for s in ("synthetic", "mitdb")
                 for k in ("bw", "ma", "impulse")}

    cache = {}
    for key in sorted(need):
        src, snr, kind = key
        print(f"[prepare] {src} {snr:+.0f} dB {kind}")
        cache[key] = prepare(src, snr, kind)

    print("[slides]")
    if "S1" in want:
        s1_input(cache)
    for src in ("synthetic", "mitdb"):
        if "S2" in want:
            s2_methods(cache, src, a.snr)
        if "S3" in want:
            s3_qrs(cache, src, a.snr)
        if "S4" in want:
            s4_noise(cache, src, a.snr)
    if "S5" in want:
        s5_crossover()
    if "S6" in want:
        s6_safety()
    if "S7" in want:
        s7_loss_gap(TXT)
    if "S8" in want:
        s8_clean_preservation(TXT)
        s9_structure_vs_loss(TXT)
    if "S10" in want:
        s10_loss_by_noise(TXT)

    save_manifest(OUT, cfg=vars(a), sources=[
        "scripts/make_slides.py", "scripts/run_exp.py", "ecgdn/data/dataset.py",
        "ecgdn/methods/frontend.py", "ecgdn/eval/engine.py",
        "ecgdn/models/losses.py"])
    write_index()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
