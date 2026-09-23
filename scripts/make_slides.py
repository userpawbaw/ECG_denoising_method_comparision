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
# 발표용 묶음. 같은 그림·같은 수치, **말만 다르다** (`--present`).
PRESENT_OUT = Path("results/slides_present")
# `_bootstrap` 이 작업 디렉터리를 저장소 루트로 옮기므로 상대경로가 곧 루트다.
ROOT = Path(".")

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

# ---------------------------------------------------- 발표용 (저장소 밖 청중)
# 저장소 안에서는 `M04` · `snr_imp_scaled` · `EXP-C` 가 옳다 — 짧고 보고서와
# 1:1 로 붙는다. **발표장에서는 아무도 모른다.** 그렇다고 보고서 그림을 바꾸면
# 보고서가 자기 표와 어긋나므로, 같은 그림을 **말만 바꿔 따로 렌더**한다.
#
# 코드를 통째로 지우지는 않는다 — 괄호로 남긴다(`SWT wavelet (M04)`). 발표 중
# 백업 슬라이드나 보고서를 펴는 사람이 두 이름을 이어야 하기 때문이다.
NAME_PRESENT = {
    "clean": "참값 (정답 신호)", "noisy": "입력 (잡음 섞임)",
    "M_FE": "공통 전처리만 (M_FE)", "M01": "Bandpass 0.5-40 Hz (M01)",
    "M04": "SWT wavelet (M04)", "M05": "Sameni 칼만필터 (M05)",
    "M08": "딥러닝 Wavelet U-Net (M08)",
}
TXT_PRESENT = {
    "time": "시간 [s]", "amp": "진폭 [mV]", "resid": "잔차 = 출력 - 참값",
    "output": "출력 파형", "in_snr": "입력 SNR [dB]  (오른쪽일수록 깨끗한 신호)",
    "gain": "공통 전처리 위에 더한 개선량 [dB]",
}

# 제목·주석은 f-string 으로 조립되는 것이 많아 리터럴마다 감쌀 수가 없다.
# 그래서 **글자가 그림에 닿는 한 지점**에서 한 번에 바꾼다(`install_present_sink`).
# 부분 치환은 조사에서 깨지므로 **긴 것부터** 적고, 스크립트에 실제로 있는
# 문자열 전부에 대해 `tests/test_slides_present.py` 가 결과를 고정한다.
#
# **손실 이름(L1 · L3 · L6)은 바꾸지 않는다.** 발표에서 그 사다리를 슬라이드
# 본문에 함께 띄우므로(docs/94 2.4 의 11 번), 그림과 본문이 같은 말을 써야 한다.
PRESENT_TERMS: list[tuple[str, str]] = [
    # ① 방법 코드 — 문맥까지 함께 갈아야 조사가 안 깨진다
    ("딥러닝(M08)", "딥러닝 (Wavelet U-Net)"),
    ("(M06, TEST, 기록 단위 n=44)", "(U-Net · 평가용 분할 · 기록 단위 n=44)"),
    ("(M06 기준, TEST n=22)", "(U-Net 기준 · 평가용 분할 n=22)"),
    ("(M06, TEST)", "(U-Net · 평가용 분할)"),
    ("M06L6 - M06", "L6 로 학습 - L1 로 학습"),
    ("SWT 와의", "SWT wavelet 과의"),
    ("SWT 선까지", "SWT wavelet 선까지"),
    # ② 데이터축 — 조사가 붙은 꼴을 통째로
    ("D0 는", "합성 데이터는"), ("D1 은", "실기록은"), ("D1 에서", "실기록에서"),
    ("— D1, 입력 SNR", "— 실기록 · 입력 SNR"),
    ("두 축 · 두 프로브", "두 데이터 · 두 시험"),
    ("합성축 참값 기준", "합성 데이터의 참값 기준"),
    # ③ 지표·실험·통계 용어
    ("기준 대비 Δ  snr_imp_scaled [dB]", "기준 대비 개선량 차이 [dB]"),
    ("snr_imp_scaled", "개선량"),
    ("(EXP-C)", "(잡음 0 입력)"),
    ("paired Wilcoxon + Holm", "짝지은 검정 + 다중비교 보정"),
    ("Holm 보정 후 유의", "다중비교 보정 후 유의"),
    # ④ 영어 약어와 줄임말. **`front-end` 만 바꾸면 안 된다** — 원문에 이미
    #    「공통 front-end」 와 「front-end 만」 이 있어서 «공통 공통 전처리» ·
    #    «공통 전처리 만» 이 된다. 눈으로 보고 잡았다(docs/17 §3).
    ("공통 front-end", "공통 전처리"),
    ("front-end 만 쓰는", "전처리만 쓰는"),
    ("front-end", "공통 전처리"),
    ("(Holm 보정 p < 0.05)", "(다중비교 보정 후 p < 0.05)"),
    ("(Holm)", "(다중비교 보정)"),
    ("M06\nresunet1d", "U-Net\n(M06)"),
    ("M08\nwavelet_unet", "Wavelet U-Net\n(M08)"),
    ("R-peak 기준 시간", "R 봉우리 기준 시간"),
    ("회색 = 참조", "회색 = 참값"),
    ("(출력 - 참조)", "(출력 - 참값)"),
]


def present_text(s: str) -> str:
    """발표용 렌더에서 그림에 닿는 모든 문자열이 지나는 자리."""
    for a, b in PRESENT_TERMS:
        s = s.replace(a, b)
    return s


def install_present_sink():
    """`Text.set_text` 하나만 감싼다.

    제목·축·범례·주석이 전부 결국 여기를 지난다. 리터럴마다 감싸면 f-string
    으로 조립된 제목의 **계산된 부분**이 그대로 빠져나가는데, 그런 제목이
    이 파일에만 여섯이다(S7 · S9 · S10 …). 눈금 숫자도 지나가지만 바꿀 말이
    없어 무해하다.
    """
    from matplotlib.text import Text
    if getattr(Text.set_text, "_present", False):
        return
    orig = Text.set_text

    def patched(self, s):
        return orig(self, present_text(s) if isinstance(s, str) else s)

    patched._present = True
    Text.set_text = patched


def slide_style(present: bool = False):
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
        # 발표용은 **한글이 전부**다. 영문으로 떨어지면 목적을 잃으므로 알린다.
        if present:
            print("[slides] 발표용(--present)인데 한글 폰트가 없다 — "
                  "영문으로 나간다. 이대로 쓰지 말 것")
        return NAME_EN, TXT_EN
    if present:
        install_present_sink()
        return NAME_PRESENT, TXT_PRESENT
    return NAME_KO, TXT_KO


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
                x_raw=np.asarray(it.get("x_raw", it["x"]), dtype=float),
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
    zeros: list[tuple[float, str]] = []
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
        # 직접 라벨은 **곡선 중간**에 둔다. 처음에는 오른쪽 끝에 뒀는데, 거기가
        # 0 교차 점선과 그 «N dB» 라벨이 모이는 자리라 글자가 겹쳤다. 중간은
        # 두 곡선이 2 dB 이상 벌어져 있어 붙일 자리가 넉넉하다.
        k = len(xs) // 2
        ax.annotate(lab, (xs[k], ys[k]), color=col, fontsize=11,
                    fontweight="bold", ha="center",
                    xytext=(0, 12 if tag == "d0" else -20),
                    textcoords="offset points")
        # 0 을 지나는 지점 (인접 두 점 선형보간)
        for i in range(len(xs) - 1):
            if ys[i] > 0 >= ys[i + 1]:
                zx = xs[i] + (xs[i + 1] - xs[i]) * ys[i] / (ys[i] - ys[i + 1])
                ax.axvline(zx, color=col, ls=":", lw=1.6, zorder=2)
                zeros.append((zx, col))
                break
    ax.axhline(0, color=INK2, lw=1.2, zorder=2)
    # **라벨은 두 곡선을 다 그린 뒤에 얹는다.** 곡선마다 바로 적으면 그때의
    # `get_ylim()` 을 쓰는데, 첫 곡선(D0)을 그릴 때는 축이 아직 -0.5 까지밖에
    # 안 내려가 있다 — 그래서 «20 dB» 가 바닥이 아니라 0 근처, 표식 위에
    # 얹혔다. 눈으로 보고 잡았다(docs/17 §3 — 「라벨 겹침」).
    y0 = ax.get_ylim()[0]
    for zx, col in zeros:
        ax.annotate(f"{zx:.0f} dB", (zx, y0), color=col,
                    fontsize=10.5, fontweight="bold", ha="center",
                    xytext=(0, 6), textcoords="offset points")
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
    seen: list[tuple[str, str, float, bool]] = []   # 부제를 여기서 계산한다
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
        for (lab, kind, *_), x, pv in zip(ROWS, xs, ps):
            if np.isfinite(x):
                seen.append((lab.split()[0], kind, x, pv < 0.05))
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
    # **부제는 점에서 계산한다.** 초판은 「유의한 구조 변경은 M07 하나」라고 손으로
    # 적혀 있었는데, D0 에서는 M10 도 유의하다 — S10 과 같은 실수다 (O-27).
    sg = [(n, x) for n, k, x, sig in seen if k == "구조" and sig]
    sl = [(n, x) for n, k, x, sig in seen if k == "손실" and sig]
    n_l = sum(1 for _, k, _, _ in seen if k == "손실")
    names = " · ".join(sorted({n for n, _ in sg})) or "없다"
    way = ("전부 나쁜 쪽이다" if sg and all(x < 0 for _, x in sg)
           else "방향이 갈린다")
    lway = ("전부 좋은 쪽" if sl and all(x > 0 for _, x in sl) else "방향이 갈린다")
    fig.suptitle("구조를 네 번 바꿔도 안 됐고, 손실은 됐다  (M06 기준, TEST n=22)\n"
                 f"유의한 구조 변경은 {names} 뿐이고 {way}. "
                 f"손실은 {len(sl)}/{n_l} 칸이 유의하고 {lway}이다",
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
     "7 단계(EXP-G), 두 축. **결론의 축은 잡음 종류가 아니라 입력 SNR 이다** — "
     "왼쪽(저 SNR)이 붉고 오른쪽으로 갈수록 파래진다. 칸 수와 판정은 "
     "**그림의 부제가 격자에서 직접 계산**하고, 해석은 보고서 5.8.10 에 있다 "
     "(같은 산문을 두 곳에 두면 한쪽이 낡는다 — O-27)."),
    ("S11_nofe_grid.png",
     "**front-end 를 아예 빼면.** 2x2 격자(모델 x 손실)로 재면 격차를 만드는 "
     "것은 **모델이 아니라 손실**이다(L1 5.05 dB vs L6 1.75 dB). 그리고 FE 를 "
     "빼면 여덟 판이 0.97 dB 안에 몰려 **무엇을 골라도 비슷해진다** — "
     "FE 판의 3.61 dB 와 견주면 띠 폭 자체가 결론이다."),
    ("S12_fe_tradeoff.png",
     "**실시간 front-end 는 무엇을 지킬 것인가의 문제다.** 평평함(가로)과 "
     "ST 충실도(세로)가 **같은 방향이 아니고**, 좌하단의 '둘 다 좋음' 자리가 "
     "비어 있다. 점 크기가 추가 지연이다 — 인과 방식만 어느 축으로도 3 위다."),
    ("S13_frontend_flip.png",
     "**F-10 의 결과 한 장.** 바꾼 것은 '딥러닝에도 같은 front-end 를 준다' "
     "하나인데 결론이 둘 뒤집혔다 — 15 dB 승자가 M04(7.79)에서 M08(8.71)로, "
     "M06 대 M01 이 n.s.(p=0.101)에서 유의(p=9e-6)로. 왜곡 하한 두 행은 "
     "**참조 정의가 달라 방향만** 읽는다."),
    ("S14_measurement_traps.png",
     "**6 장 머리말 한 장.** 측정 틀이 틀렸던 아홉 건과 각각을 잡아낸 것. "
     "아홉 중 다섯이 '참조를 무엇으로 둘 것인가' 이고, 아홉 다 p 값이 "
     "유의했다 — 통계는 전제가 틀렸다는 것을 알려주지 않는다."),
    ("S15_pli_psd.png",
     "**Q1 · 층 4 «무엇을 지웠나».** 전원선 잡음 입력과 공통 전처리 출력의 **파형 "
     "둘과 그 두 신호의 PSD** 를 한 장에. 60·120 Hz 만 도려내고 나머지 대역은 "
     "원본과 겹친다 — 그런데 남은 오차는 전부 **70 Hz**(180 Hz 고조파가 접힌 자리, "
     "notch 목록에 없다)다. 비교 기준은 평가 참값이 아니라 **원본 기록**이다 — "
     "참값은 같은 전처리를 통과한 것이라 «나머지 대역 그대로» 가 정의상 참이 된다 (F-54)."),
    ("S16_pli_swt.png",
     "**Q2 · S4 에 없던 행.** 전원선 잡음에서는 SWT 가 이긴다(실기록 · 입력 10 dB, "
     "잡음 종류별 실험과 같은 조건). 잔차 칸이 이유를 보여 준다 — 전처리가 남긴 "
     "것은 70 Hz 뿐이고 SWT 는 그것을 줄이며, 딥러닝의 오차는 대부분 **40 Hz 아래** "
     "— 이미 깨끗한 심전도를 건드린 것이다 (F-54)."),
    ("S17_noise_board.png",
     "**Q2 전체 판.** 잡음 7 조건 × (전처리만 · SWT · 딥러닝). 딥러닝이 이기는 곳과 "
     "SWT 가 이기는 곳의 수, 차이를 **표에서 계산**해 적는다. 고전이 이기는 두 곳은 "
     "전처리만으로도 딥러닝보다 높다."),
]


# ------------------------------------------------- S11 FE 없는 학습 2x2 격자
def s11_nofe_grid(TXT):
    """**말할 것 하나**: 격차를 만드는 것은 모델이 아니라 **손실**이고,
    FE 를 빼면 **선택 자체가 무의미해진다.**

    왼쪽 — 2x2 를 점 넷으로. 같은 손실끼리 이으면 **두 선이 나란히 위아래**로
    놓인다(손실이 가른다). 모델 축으로는 거의 안 움직인다.
    오른쪽 — 20 dB 에서 여덟 판을 두 띠로. **띠 폭 자체가 결론이다**
    (FE 판 3.6 dB 대 nofe 판 1.0 dB).
    """
    import pandas as pd

    f = ROOT / "results/d1/exp_nofe/metrics.parquet"
    if not f.exists():
        print("  S11 건너뜀 (exp_nofe 없음)"); return
    df = pd.read_parquet(f)
    snr = (df[df.metric == "snr_imp_scaled"]
           .pivot_table(index="method", columns="snr_in_target", values="value"))
    hi = snr.columns[-1]
    cells = [("M06", "L1"), ("M06", "L6"), ("M08", "L1"), ("M08", "L6")]
    gap = {(m, l): snr.loc[f"{m}{l}"] - snr.loc[f"{m}{l}n"] for m, l in cells}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=175)
    fig.suptitle("front-end 를 빼면 무엇이 지배하는가 — D1, 입력 SNR "
                 f"{hi:g} dB", x=0.02, ha="left", fontsize=13.5, fontweight="bold")

    # --- 왼쪽: 손실이 가르고 모델은 거의 안 가른다
    xs = [0, 1]
    for li, loss in enumerate(("L1", "L6")):
        ys = [gap[("M06", loss)][hi], gap[("M08", loss)][hi]]
        a1.plot(xs, ys, "-o", color=LOSS[loss], lw=2.6, markersize=11,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
                label=f"손실 {loss}")
        for x, y in zip(xs, ys):
            a1.annotate(f"{y:+.2f}", (x, y), color=LOSS[loss], fontsize=10.5,
                        fontweight="bold", ha="center",
                        xytext=(0, 13 if loss == "L6" else -20),
                        textcoords="offset points")
    a1.margins(y=0.26)          # 위아래로 붙인 라벨이 잘리지 않게
    a1.set_xticks(xs)
    a1.set_xticklabels(["M06\nresunet1d", "M08\nwavelet_unet"], fontsize=10.5)
    a1.set_xlim(-0.35, 1.35)
    a1.set_ylabel("front-end 를 뺀 대가 [dB]   (클수록 손해)", fontsize=10.5)
    a1.set_title("두 선이 나란하다 — 가르는 것은 손실이다", fontsize=11.5,
                 loc="left")
    a1.legend(fontsize=10, frameon=False, loc="center left")
    for sp in ("top", "right"):
        a1.spines[sp].set_visible(False)
    a1.grid(axis="y", color="#e8e8e8", lw=0.9)
    a1.set_axisbelow(True)

    # --- 오른쪽: 띠 폭이 결론
    fe = [snr.loc[f"{m}{l}"][hi] for m, l in cells]
    no = [snr.loc[f"{m}{l}n"][hi] for m, l in cells]
    for i, (vals, lab, col) in enumerate(
            ((fe, "front-end 판", "#184f95"), (no, "front-end 없는 판", "#c05010"))):
        lo, high = min(vals), max(vals)
        a2.plot([lo, high], [i, i], color=col, lw=9, alpha=0.22,
                solid_capstyle="round", zorder=1)
        a2.scatter(vals, [i] * len(vals), s=110, color=col, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        a2.annotate(f"폭 {high - lo:.2f} dB", ((lo + high) / 2, i), color=col,
                    fontsize=12, fontweight="bold", ha="center",
                    xytext=(0, 22), textcoords="offset points")
        a2.annotate(lab, (lo, i), color=col, fontsize=11, ha="right",
                    va="center", xytext=(-14, 0), textcoords="offset points")
    a2.set_ylim(-0.7, 1.7)
    a2.set_yticks([])
    a2.set_xlabel("SNR 개선 [dB]", fontsize=10.5)
    a2.set_title("FE 를 빼면 무엇을 골라도 비슷해진다", fontsize=11.5,
                 loc="left")
    for sp in ("top", "right", "left"):
        a2.spines[sp].set_visible(False)
    a2.grid(axis="x", color="#e8e8e8", lw=0.9)
    a2.set_axisbelow(True)

    fig.text(0.02, 0.015,
             "왼쪽 네 점 = 모델 2 x 손실 2. 오른쪽 여덟 점 = 그 넷의 FE 판과 "
             "nofe 판. 병목이 설계가 아니라 «앞단을 스스로 근사하는 능력» 으로 "
             "옮겨 간다.", fontsize=10, color=INK2)
    fig.tight_layout(rect=[0.02, 0.045, 1, 0.94])
    fig.savefig(OUT / "S11_nofe_grid.png", dpi=175); plt.close(fig)
    print("  S11_nofe_grid.png")


# ------------------------------------------- S12 실시간 front-end 트레이드오프
def s12_fe_tradeoff(TXT):
    """**말할 것 하나**: 평평함과 ST 충실도는 **같은 방향이 아니다.**

    좌하단이 «둘 다 좋음» 인데 **그 자리가 비어 있는 것**이 요지다. 점 크기로
    지연을 실어, 「빠른 것은 왜곡이 크다」까지 한 장에 담는다.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "audit_fe_metrics", ROOT / "scripts" / "audit_fe_metrics.py")
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
        rows = m.audit_reference(n_rec=8)[0]
    except Exception as e:                       # pragma: no cover
        print(f"  S12 건너뜀 ({type(e).__name__}: {e})"); return
    import pandas as pd
    g = pd.DataFrame(rows).groupby("who")[["spread", "st_err"]].mean()

    # 지연 [ms] — docs/13 · 30 의 실시간 브리지 설정 기준
    LAT = {"오프라인 영위상": None, "인과 o1 0.5 Hz": 0,
           "블록 영위상 0.5 s": 548, "중앙값 200+600 ms": 424}
    COL = {"오프라인 영위상": "#b8b6ae", "인과 o1 0.5 Hz": "#eb6834",
           "블록 영위상 0.5 s": "#2a78d6", "중앙값 200+600 ms": "#1baf7a"}

    fig, ax = plt.subplots(figsize=(10.6, 6.4), dpi=175)
    for who, row in g.iterrows():
        lat = LAT.get(who)
        size = 220 if lat is None else 150 + lat * 0.85
        ax.scatter(row["spread"], abs(row["st_err"]), s=size,
                   color=COL.get(who, "#666"), zorder=3,
                   edgecolor=SURFACE, linewidth=2,
                   alpha=0.55 if lat is None else 0.95)
        tag = "도달 못 함" if lat is None else f"지연 {lat} ms"
        ax.annotate(f"{who}\n{tag}", (row["spread"], abs(row["st_err"])),
                    fontsize=10.5, color=COL.get(who, "#666"),
                    fontweight="bold", ha="left", va="center",
                    xytext=(18, 0), textcoords="offset points")
    ax.set_xlabel("박동별 T-P 준위 산포 [%R]   (작을수록 평평)", fontsize=11)
    ax.set_ylabel("ST 준위 오차 [%R]   (작을수록 충실)", fontsize=11)
    ax.set_title("실시간 front-end — 평평함과 ST 충실도는 같은 방향이 아니다\n"
                 "합성축 참값 기준 8 기록 · 점 크기 = 추가 지연",
                 fontsize=12.5, loc="left")
    # 오른쪽 끝 점의 라벨이 밖으로 나가지 않게 가로 여백을 준다
    ax.set_xlim(0, g["spread"].max() * 1.42); ax.set_ylim(-0.2, None)
    ax.annotate("둘 다 좋은 자리 — 비어 있다", (0.5, 0.08),
                color=INK2, fontsize=11, style="italic")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color="#ececec", lw=0.9); ax.set_axisbelow(True)
    fig.text(0.02, 0.015,
             "인과 방식은 어느 축으로도 3 위다 — 그것만은 안 바뀐다. "
             "1·2 위는 «무엇을 지킬 것인가» 에 따라 갈린다.",
             fontsize=10, color=INK2)
    fig.tight_layout(rect=[0.02, 0.04, 1, 1])
    fig.savefig(OUT / "S12_fe_tradeoff.png", dpi=175); plt.close(fig)
    print("  S12_fe_tradeoff.png")


# ------------------------------------------- S13 front-end 를 고치자 뒤집힌 것
# 보고서 5.7 의 표를 그대로 옮긴 것이다. 초판(F-10 이전)의 값은 **재생성할 수
# 없다** — F-9 로 당시 체크포인트가 전부 무효가 됐다. 이 저장소에서 파일에서
# 못 읽고 손으로 옮기는 유일한 수치이므로, 옮겨 적은 것이 보고서와 어긋나지
# 않게 `tests/test_repo_integrity.py` 가 5.7 표와 대조한다 (F-19 · F-26).
FLIP_57 = [
    # (라벨, 초판 방법, 초판 값, 현재 방법, 현재 값, 무엇이 바뀌었나)
    ("15 dB 입력", "M04", 7.79, "M08", 8.71, "flip"),
    ("-5 dB 입력", "M07", 18.54, "M06", 17.25, "same"),
]
FLIP_57_FLOOR = [
    ("M08 왜곡 하한", 23.19, 27.04),
    ("M06 왜곡 하한", 22.33, 24.64),
]
FLIP_57_STAT = ("M06 vs M01", "n.s.  (p = 0.101)", "유의  (p = 9e-6, r = 0.99)")

FLIP_C = {"flip": "#184f95", "same": "#9a9892"}


def s13_frontend_flip(TXT):
    """**말할 것 하나**: 바뀐 것은 «딥러닝에도 같은 front-end 를 준 것» 하나인데
    결론이 둘 뒤집혔다.

    왜곡 하한 두 행은 **참조 정의가 달라 같은 자로 잰 것이 아니다.** 그림
    안에 그 사실을 적고 방향만 읽게 한다 — 안 적으면 그림이 보고서보다 강한
    주장을 하게 된다 (5.3 · O-16).
    """
    fig = plt.figure(figsize=(12.6, 7.4), dpi=175)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0],
                          left=0.075, right=0.985, top=0.80, bottom=0.30,
                          wspace=0.42)
    a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # -- 왼쪽: 최고 성능 [dB]. 같은 양(SNR 개선)이라 한 축에 놓을 수 있다.
    for lab, m0, v0, m1, v1, kind in FLIP_57:
        col = FLIP_C[kind]
        lw = 3.4 if kind == "flip" else 1.6
        al = 1.0 if kind == "flip" else 0.65
        a1.plot([0, 1], [v0, v1], color=col, lw=lw, alpha=al,
                marker="o", ms=9, zorder=3, clip_on=False)
        a1.annotate(f"{m0}  {v0:.2f}", (0, v0), xytext=(-10, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=10.5, color=col, alpha=al)
        a1.annotate(f"{m1}  {v1:.2f}", (1, v1), xytext=(10, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=10.5, color=col, alpha=al,
                    fontweight="bold" if kind == "flip" else "normal")
        # 행 이름과 «무엇이 바뀌었나» 를 한 줄로 붙인다 — 따로 두면 겹친다.
        note = ("승자가 DSP 에서 딥러닝으로 바뀐다" if kind == "flip"
                else "둘 다 딥러닝, 여기서는 안 바뀐다")
        a1.annotate(f"{lab} — {note}", (0.5, (v0 + v1) / 2),
                    xytext=(0, 13 if kind == "same" else -22),
                    textcoords="offset points", ha="center",
                    va="bottom" if kind == "same" else "top",
                    fontsize=10.5, color=col, alpha=al,
                    fontweight="bold" if kind == "flip" else "normal")
    a1.set_ylabel("최고 성능 [dB]  (SNR 개선)", fontsize=11)
    a1.set_title("성능 — 같은 자로 잰 값", fontsize=11.5, loc="left")

    # -- 오른쪽: 왜곡 하한. 참조가 바뀌었으므로 **방향만** 읽는다.
    for lab, v0, v1 in FLIP_57_FLOOR:
        a2.plot([0, 1], [v0, v1], color="#9a9892", lw=2.0, ls="--",
                marker="o", ms=8, zorder=3, clip_on=False)
        for x, v, ha, dx in ((0, v0, "right", -10), (1, v1, "left", 10)):
            a2.annotate(f"{v:.2f}", (x, v), xytext=(dx, 0),
                        textcoords="offset points", ha=ha, va="center",
                        fontsize=10.5, color=INK2)
        # 행 이름은 선 위에 얹는다 — 오른쪽 끝에 두면 축 밖으로 나간다.
        a2.annotate(lab, (0.5, (v0 + v1) / 2), xytext=(0, 12),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=10.5, color=INK2)
    a2.set_ylabel("왜곡 하한 [dB]  (EXP-C)", fontsize=11)
    a2.set_title("왜곡 하한 — 자가 바뀌었다", fontsize=11.5, loc="left")
    a2.annotate("참조를 raw 에서 FE(clean) 으로 바꾼 뒤의 값이다.\n"
                "초판 열과 같은 자로 잰 것이 아니므로 방향만 읽을 것 (5.3 · O-16).",
                (0.5, 0.03), xycoords="axes fraction", ha="center", va="bottom",
                fontsize=9.5, color=INK2, style="italic",
                bbox=dict(boxstyle="round,pad=0.45", fc="#f4f2ec", ec="#ddd9cf"))

    for ax, pad in ((a1, 0.30), (a2, 0.22)):
        ax.set_xlim(-pad, 1 + pad)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["초판\n(딥러닝만 FE 없음)", "현재\n(전 방법 동일 FE)"],
                           fontsize=10.5)
        ax.margins(y=0.30)
        ax.grid(axis="y", color="#ececec", lw=0.9); ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # -- 아래 띠: 통계 판정의 뒤집힘. 단위가 달라 축에 못 올린다.
    who, before, after = FLIP_57_STAT
    fig.text(0.075, 0.185, "통계 판정도 뒤집혔다", fontsize=11.5,
             color=FLIP_C["flip"], fontweight="bold")
    fig.text(0.075, 0.115,
             f"{who}      {before}      →      {after}",
             fontsize=12, color=INK)
    fig.text(0.075, 0.058,
             "이제 22 개 기록 전부에서 이긴다. 초판의 «딥러닝은 유의하지 않다» 는 "
             "방법의 성질이 아니라 판의 결함이었다.",
             fontsize=10, color=INK2)

    fig.suptitle("front-end 하나를 고치자 결론 둘이 뒤집혔다", x=0.075, y=0.955,
                 ha="left", fontsize=15.5, fontweight="bold")
    fig.text(0.075, 0.885,
             "바뀐 것은 «딥러닝에도 같은 front-end 를 준다» 하나다 — "
             "나머지 조건은 모두 같다 (보고서 5.7 · F-10).",
             fontsize=11, color=INK2)
    fig.savefig(OUT / "S13_frontend_flip.png", dpi=175); plt.close(fig)
    print("  S13_frontend_flip.png")


# ------------------------------------------------- S14 측정 틀이 틀렸던 아홉 건
# 색은 dataviz 슬롯 팔레트를 그대로 쓴다. 이 그림에는 **방법이 하나도
# 등장하지 않으므로** C 의 «색은 방법에 고정» 규약과 충돌하지 않는다.
TRAP_C = {
    "참조 정의": "#2a78d6",
    "조건 불일치": "#eb6834",
    "지표 분해능": "#1baf7a",
    "가정의 축 이전": "#eda100",
}
# (F 번호, 유형, 무엇이 «그럴듯한 표» 를 만들었나, 무엇이 잡았나)
# 줄바꿈은 손으로 넣는다 — 칸 폭에 맞춰 두 줄로 끊어야 아홉 행이 16:9 에 든다.
TRAPS = [
    ("F-8", "조건 불일치",
     "record 로 split 했는데 morphology 커널은 train/test 가 공유 —\n"
     "딥러닝 +26 dB 대 DSP +3 dB 로 나왔다",
     "«26 dB 가 가능한가» 라는 크기 감각. 커널을 흔드니\n"
     "+16.5 → +4~6 dB 로 주저앉았다"),
    ("F-10", "조건 불일치",
     "«전 방법 동일 front-end» 가 문서에만 있었고 딥러닝만 raw 를\n"
     "받고 있었다 — 딥러닝이 진 조건의 상당 부분이 그것이었다",
     "front-end 단독 열(M_FE)을 표에 넣자 이득의 출처가 읽혔다 —\n"
     "bw·pli 에서 SWT 의 기여는 정확히 0 dB"),
    ("F-11", "참조 정의",
     "oracle 을 «문제의 상한» 으로 인용할 뻔했다 —\n"
     "B01 · B02 · M_FE 가 소수점까지 같은 34.13 dB",
     "세 값이 같다는 것 자체가 단서였다. oracle 도 같은\n"
     "front-end 를 지난다 — 그것은 전처리의 상한이다"),
    ("F-12", "참조 정의",
     "MIT-BIH 원본을 정답으로 두니 front-end 가 지운 성분이\n"
     "전부 오차 — SWT 튜닝이 최선에서 -0.49 dB 를 냈다",
     "D0 53.2 dB 대 D1 9.3 dB 의 격차. 참조를 FE(원본) 으로\n"
     "바꾸자 같은 튜닝이 +12.47 dB 가 됐다"),
    ("F-15", "참조 정의",
     "M_FE 의 왜곡 하한 147 dB — «front-end 는 신호를 전혀\n"
     "손상시키지 않는다» 가 결론이 될 뻔했다",
     "크기 감각. 147 dB 는 성능이 아니라 SNR(FE(x), FE(x)) = ∞ 가\n"
     "float64 에서 잘린 값이었다"),
    ("F-16", "지표 분해능",
     "«D1 에서 L2 · L3 가 QRS 를 더 잘 보존한다»\n"
     "(25.995 → 22.267 ms) 를 실을 뻔했다",
     "floor 를 축마다 쟀다 — D0 0.640 ms, D1 28.07 ms 로 44 배.\n"
     "격차가 분해능 아래라 «최선» 표시를 걷었다"),
    ("F-17", "가정의 축 이전",
     "«실데이터에서는 참 파라미터를 줘도 5.91 dB» —\n"
     "모형 가정이 무너진다는 결론이 될 뻔했다",
     "천장이어야 할 값이 천장이 아니었다 — 적합 커널 11.23 dB 가\n"
     "주입 커널 5.91 을 앞섰다. 주입 커널은 D1 과 무관했다"),
    ("F-33", "참조 정의",
     "pipeline_db 로는 인과 모드가 1 위(M06 +0.89) —\n"
     "참조 FE_rt(clean) 이 모드마다 함께 움직였다",
     "파형 지표가 정반대를 말했다. 고정 참조로 재니 인과 -5.22 dB,\n"
     "블록 영위상 -3.08 로 순위가 뒤집힌다"),
    ("F-35", "참조 정의",
     "고정한 참조 FE_off(clean) 이 후보 하나가 수렴하도록\n"
     "정의된 목표였다 — 8 기록 전부 블록 영위상 승, 6.05 dB",
     "세 방식이 공유하는 부분만 참값으로 삼으니 순위가 뒤집힌다 —\n"
     "중앙값이 0.78 dB 우세, 8 중 4 승"),
]


def s14_measurement_traps(TXT):
    """**말할 것 하나**: 이 과제의 발견 중 가장 큰 계열은 방법이 아니라
    **측정**이었고, 아홉 건 모두 «그럴듯한 결과표» 를 만들어냈다.

    6 장 머리말이 이 계열을 글로 지목하는데 아홉 건이 2 000 줄에 흩어져 있어
    계열로 안 읽힌다. 한 장이면 읽힌다.
    """
    fig = plt.figure(figsize=(14.0, 8.4), dpi=170)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    XS, XA, XB = 0.145, 0.185, 0.545      # 척추 · A 칸 · B 칸
    top, bot = 0.795, 0.105
    ys = [top - i * (top - bot) / (len(TRAPS) - 1) for i in range(len(TRAPS))]

    ax.plot([XS, XS], [bot - 0.030, top + 0.030], color="#ddd9cf", lw=2,
            zorder=1)
    for i, (fid, kind, made, caught) in enumerate(TRAPS):
        y, col = ys[i], TRAP_C[kind]
        if i:
            ax.axhline((ys[i - 1] + y) / 2, xmin=0.012, xmax=0.988,
                       color="#eeece6", lw=0.8, zorder=0)
        ax.plot([XS], [y], marker="o", ms=11, color=col,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.text(XS - 0.026, y + 0.014, fid, ha="right", va="center",
                fontsize=13, color=col, fontweight="bold")
        ax.text(XS - 0.026, y - 0.020, kind, ha="right", va="center",
                fontsize=9, color=INK2)
        ax.text(XA, y, made, ha="left", va="center", fontsize=9.6, color=INK,
                linespacing=1.6)
        ax.annotate("", xy=(XB - 0.014, y), xytext=(XB - 0.048, y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.7,
                                    shrinkA=0, shrinkB=0))
        ax.text(XB, y, caught, ha="left", va="center", fontsize=9.6,
                color=INK2, linespacing=1.6)

    ax.text(XA, 0.862, "무엇이 «그럴듯한 표» 를 만들었나", ha="left",
            va="center", fontsize=11, color=INK2, fontweight="bold")
    ax.text(XB, 0.862, "무엇이 잡았나", ha="left", va="center", fontsize=11,
            color=INK2, fontweight="bold")

    for j, (kind, col) in enumerate(TRAP_C.items()):
        x = 0.560 + j * 0.112
        ax.plot([x], [0.968], marker="o", ms=8, color=col)
        ax.text(x + 0.012, 0.968, kind, ha="left", va="center", fontsize=9.5,
                color=INK2)

    ax.text(0.012, 0.972, "측정 틀이 틀렸던 아홉 건", ha="left", va="center",
            fontsize=16.5, fontweight="bold", color=INK)
    ax.text(0.012, 0.922,
            "이 과제의 발견 중 가장 큰 계열은 방법이 아니라 측정이다. "
            "아홉 중 다섯이 «참조를 무엇으로 둘 것인가» 다.",
            ha="left", va="center", fontsize=11, color=INK2)

    ax.text(0.012, 0.046,
            "아홉 건 모두 p 값이 유의했고 effect size 도 컸다 — "
            "통계는 «전제가 틀렸다» 를 알려주지 않는다.",
            ha="left", va="center", fontsize=10.5, color=INK)
    ax.text(0.012, 0.017,
            "잡아낸 것은 언제나 «이 숫자가 이렇게 클 수 있는가» 라는 크기 "
            "감각이었다 — 26 dB · 242 % · 147 dB · 44 배, 그리고 천장이 아닌 천장.",
            ha="left", va="center", fontsize=10.5, color=INK2)
    fig.savefig(OUT / "S14_measurement_traps.png", dpi=170); plt.close(fig)
    print("  S14_measurement_traps.png")


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
    grids: dict[str, tuple] = {}
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
        grids[tag] = (D, S, snrs)
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
    # **부제는 격자에서 계산한다.** 초판은 이 문장을 손으로 적었는데, EXP-G 를
    # 3 단계에서 7 단계로 넓힌 뒤에도 「손해 다섯 칸은 전부 0 dB」 이 남아 있었다
    # (O-27 — O-11 · O-12 와 같은 계열). 산출물을 설명하는 산문은 산출물에서 나와야 한다.
    n_gain = sum(int(((D > 0) & S).sum()) for D, S, _ in grids.values())
    n_loss = sum(int(((D < 0) & S).sum()) for D, S, _ in grids.values())
    n_cell = sum(int(D.size) for D, _, _ in grids.values())
    lose_at = sorted({sn for D, _, sns in grids.values()
                      for i, sn in enumerate(sns) if (D[:, i] < 0).any()})
    span = (f"손해는 전부 {max(lose_at):g} dB 이하에 있다" if lose_at
            else "손해 칸이 하나도 없다")
    fig.suptitle("L6 의 이득은 잡음 종류가 아니라 입력 SNR 이 가른다  "
                 "(M06, TEST, 기록 단위 n=44)\n"
                 f"{n_cell} 칸 · 유의 이득 {n_gain} · 유의 손해 {n_loss} — {span}. "
                 "구조가 뚜렷한 잡음일수록 크게 번다", fontsize=12.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.86))
    fig.savefig(OUT / "S10_loss_by_noise.png", dpi=175); plt.close(fig)
    print("  S10_loss_by_noise.png")


# ------------------------------------------------ S15~S17 전원선 잡음 (Q1·Q2)
# 발표의 앞 두 질문을 받치는 세 장이다(docs/94 6 장).
#
#   S15  Q1 층 4 «무엇을 지웠나» — 파형 둘 옆에 **그 두 파형의** PSD.
#        스펙트럼만 띄우면 «그래서 신호가 어떻게 됐나» 가 안 이어진다.
#   S16  Q2 «전원선에서는 고전이 이긴다» — S4 에 없던 행. S4 만 보면 딥러닝이
#        늘 이기는 것처럼 읽힌다.
#   S17  Q2 전체 판 — 잡음 7 종에서 누가 이기나, 한 장에.
#
# S16·S17 은 **잡음 종류별 실험(EXP-B)과 같은 조건**(실기록 · 입력 10 dB)이다.
# EXP-B 는 10 dB 한 점만 돌렸으므로, 다른 SNR 의 그림을 옆에 두면 표와 그림이
# 다른 조건을 말하게 된다. S4 가 -5 dB 인 것과 다르다는 사실을 제목에 적는다.
NOISE_LAB = {"pli": "전원선 60 Hz", "bw_synth": "기저선 변동", "ma_synth": "근전도",
             "em_synth": "전극 움직임", "impulse": "임펄스", "awgn": "백색잡음",
             "mixed": "혼합 (전부)"}
EXPB_SNR = 10.0
RAW_LAB = "원본 기록 (잡음 넣기 전)"
ALIAS_HZ = 70.0     # 전원선 3 고조파 180 Hz 가 fs=250 에서 접히는 자리 (F-54)


def _db_rel(p, top):
    return 10.0 * np.log10(np.maximum(p, top * 1e-12) / top)


def s15_pli_psd(cache, snr):
    """전원선 잡음 입력과 공통 전처리 출력 — 파형 둘과 그 둘의 PSD."""
    from matplotlib.gridspec import GridSpec

    from ecgdn.eval.spectral import welch_psd
    d = cache[("mitdb", snr, "pli")]
    fs, y, o = d["fs"], d["y"], d["outs"]["M_FE"]
    # **비교 기준은 원본(잡음 넣기 전, 필터 전)이다.** 평가의 참값 `x` 는 같은
    # 공통 전처리를 통과한 깨끗한 신호라(D-3), 그것과 견주면 «나머지 대역은 그대로»
    # 가 정의상 참이 된다 — 처음 그렸을 때 0.0 dB 가 나와서 알았다.
    x = d["x_raw"]
    f, px = welch_psd(x, fs)
    # 원본에는 직류 높이(-0.4 mV 안팎)가 있고 전처리의 0.5 Hz 고역통과가 그것을
    # 뺀다. 파형 칸에서 그 차이가 «R 봉우리가 달라졌다» 로 읽히므로 **그림에서만**
    # 원본과 입력을 같은 높이로 올린다. PSD 는 손대지 않는다.
    lift = float(np.median(o - x))
    _, py = welch_psd(y, fs)
    _, po = welch_psd(o, fs)
    top = float(px.max())
    dx, dy, do = _db_rel(px, top), _db_rel(py, top), _db_rel(po, top)

    # 판정은 **스펙트럼에서 계산**한다(O-27). 봉우리는 ±1.5 Hz 안의 최댓값.
    def peak(db, f0):
        m = (f >= f0 - 1.5) & (f <= f0 + 1.5)
        return float(db[m].max())
    cut = {f0: peak(dy, f0) - peak(do, f0) for f0 in (60.0, 120.0, ALIAS_HZ)}
    # 나머지 대역 = 전원선 성분 셋(60 · 70 · 120)을 뺀 1-95 Hz.
    keep = ((f >= 1.0) & (f <= 95.0) & (np.abs(f - 60.0) > 3.0)
            & (np.abs(f - ALIAS_HZ) > 3.0))
    dev = float(np.mean(np.abs(do[keep] - dx[keep])))
    # 처리 후 남은 오차 중 70 Hz 몫 — «notch 가 모르는 자리» 가 얼마나 남았나
    from ecgdn.eval.engine import trim_guard
    g = trim_guard(len(x), fs)
    fr, pr = welch_psd((o - d["x"])[g], fs)
    alias_share = float(pr[np.abs(fr - ALIAS_HZ) <= 1.5].sum() / pr.sum())

    fig = plt.figure(figsize=(15, 5.8))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.15], hspace=0.45, wspace=0.16)
    i0, i1 = _window(d, 1.2)
    t = (np.arange(i0, i1) - i0) / fs
    a0 = fig.add_subplot(gs[0, 0])
    a1 = fig.add_subplot(gs[1, 0], sharex=a0, sharey=a0)
    for ax, sig, col, lab in ((a0, y, NOISY, "입력 — 원본에 전원선 60 Hz 를 섞었다"),
                              (a1, o, C["M_FE"], f"{NAME['M_FE']} 통과 후")):
        ax.plot(t, x[i0:i1] + lift, color=CLEAN, lw=2.6, label=RAW_LAB)
        ax.plot(t, sig[i0:i1] + (lift if sig is y else 0.0), color=col, lw=0.9)
        ax.set_title(lab, fontsize=11, color=col, loc="left")
        ax.set_ylabel(TXT["amp"])
    a0.legend(loc="upper right", fontsize=9)
    a1.set_xlabel(TXT["time"] + "   (원본·입력은 직류 높이만 맞춰 그렸다)")

    ap = fig.add_subplot(gs[:, 1])
    ap.plot(f, dx, color=CLEAN, lw=4.0, label=RAW_LAB)
    ap.plot(f, dy, color=NOISY, lw=1.0, label="입력")
    ap.plot(f, do, color=C["M_FE"], lw=1.4, label="처리 후")
    for f0 in (60.0, 120.0):
        ap.axvline(f0, color="#d03b3b", lw=0.8, ls="--", zorder=0)
        ap.annotate(f"{f0:.0f} Hz\n-{cut[f0]:.0f} dB", xy=(f0, peak(dy, f0)),
                    xytext=(f0 - 2, peak(dy, f0) - 1), ha="right", va="top",
                    fontsize=10, color="#d03b3b", fontweight="bold")
    # 70 Hz 는 notch 목록에 없다 — 180 Hz 고조파가 fs=250 에서 접힌 자리다
    # (`ecgdn/data/noise.py` 의 pli 주석: «해석 시 이 점을 명시할 것»).
    ap.annotate(f"{ALIAS_HZ:.0f} Hz — 180 Hz 고조파가 접힌 자리\n"
                "notch 목록에 없어 그대로 남았다",
                xy=(ALIAS_HZ + 0.8, peak(do, ALIAS_HZ)),
                xytext=(ALIAS_HZ + 6, peak(do, ALIAS_HZ) + 1), ha="left", va="top",
                fontsize=9.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ap.axvspan(100, fs / 2, color="#e8e6df", alpha=0.6, zorder=0)
    ap.text(112.5, -93, "100 Hz 위:\n저역 필터", ha="center", fontsize=9, color=INK2)
    ap.set_xlim(0, fs / 2); ap.set_ylim(-100, 8)
    ap.set_xlabel("주파수 [Hz]"); ap.set_ylabel("PSD [dB, 원본 최대 = 0]")
    ap.legend(loc="lower left", fontsize=9.5)
    ap.set_title("같은 두 신호의 스펙트럼", fontsize=11, loc="left")
    fig.suptitle(
        f"지정한 주파수만 도려냈다 — 60 Hz 를 {cut[60.0]:.0f} dB 깎고 나머지 1-95 Hz 는 "
        f"원본과 평균 {dev:.2f} dB 차이. 그런데 남은 오차의 {alias_share * 100:.0f} % 가 "
        f"{ALIAS_HZ:.0f} Hz 다\n"
        f"D1 실기록 {d['record']} · 입력 SNR {snr:+.0f} dB · "
        f"개선량 {_snr_imp(d['x'], y, o, fs):+.1f} dB — 이 숫자 하나로는 무엇을 지웠고 "
        "무엇이 남았는지가 안 보인다",
        fontsize=12.5, y=0.995)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.83, bottom=0.10)
    fig.savefig(OUT / "S15_pli_psd.png", dpi=165); plt.close(fig)
    print("  S15_pli_psd.png")


def _expb_means(tag="d1"):
    """EXP-B(잡음 종류별) 의 개선량 평균과 구간 수. 표(`table_noise.csv`)와 같은 원천."""
    import pandas as pd
    df = pd.read_parquet(Path("results") / tag / "exp_b" / "metrics.parquet")
    df = df[(df.metric == "snr_imp_scaled") & (df.snr_in_target == EXPB_SNR)]
    mean = df.groupby(["cond", "method"]).value.mean().unstack("method")
    n = int(df[df.method == "M_FE"].groupby("cond").size().min())
    return mean, n


def _resid_share(x, o, fs):
    """잔차(출력 - 참값) 전력 중 70 Hz 접힘과 40 Hz 아래의 몫."""
    from ecgdn.eval.engine import trim_guard
    from ecgdn.eval.spectral import welch_psd
    g = trim_guard(len(x), fs)
    f, p = welch_psd((o - x)[g], fs)
    tot = float(p.sum())
    return {"alias": float(p[np.abs(f - ALIAS_HZ) <= 1.5].sum()) / tot,
            "low": float(p[f < 40.0].sum()) / tot}


def s16_pli_swt(cache):
    """전원선에서는 고전이 이긴다 — 출력과 잔차, EXP-B 와 같은 조건."""
    d = cache[("mitdb", EXPB_SNR, "pli")]
    mean, n = _expb_means()
    show = ["M_FE", "M04", "M08"]
    fs, x, y = d["fs"], d["x"], d["y"]
    i0, i1 = _window(d, 2.0)
    t = (np.arange(i0, i1) - i0) / fs
    fig, axes = plt.subplots(2, 4, figsize=(15, 5.4), sharex=True)
    # 1 행: 출력 — 행 공유. 2 행: 잔차 — **방법 셋만 공유**한다. 입력 칸의 잔차는
    # 잡음 그 자체라 크기가 수십 배라, 같이 묶으면 방법 셋이 전부 평평하게 눌린다.
    for ax in axes[0, 1:]:
        ax.sharey(axes[0, 0])
    for ax in axes[1, 2:]:
        ax.sharey(axes[1, 1])
    axes[0, 0].plot(t, x[i0:i1], color=CLEAN, lw=2.4)
    axes[0, 0].plot(t, y[i0:i1], color=NOISY, lw=0.8)
    axes[0, 0].set_title(NAME["noisy"], fontsize=10.5)
    axes[1, 0].plot(t, (y - x)[i0:i1], color=NOISY, lw=0.7)
    axes[1, 0].set_title("지울 잡음 (입력 - 참값) · 세로 눈금 다름", fontsize=9.5, color=INK2)
    axes[0, 0].set_ylabel(TXT["output"]); axes[1, 0].set_ylabel(TXT["resid"])
    for c_, m in enumerate(show, start=1):
        o = d["outs"][m]
        axes[0, c_].plot(t, x[i0:i1], color=CLEAN, lw=2.4)
        axes[0, c_].plot(t, o[i0:i1], color=C[m], lw=1.0)
        axes[0, c_].set_title(NAME[m], fontsize=10.5, color=C[m])
        # 오른쪽 위 구석은 세 번째 R 봉우리가 지나간다 — 박동 사이(1.4 s 쯤)에 둔다
        axes[0, c_].text(0.70, 0.92, f"{_snr_imp(x, y, o, fs):+.1f} dB",
                         transform=axes[0, c_].transAxes, ha="center", va="top",
                         fontsize=11, color=C[m], fontweight="bold")
        axes[1, c_].plot(t, (o - x)[i0:i1], color=C[m], lw=0.8)
        axes[1, c_].axhline(0, color=INK2, lw=0.5)
        sh = _resid_share(x, o, fs)
        axes[1, c_].text(0.985, 0.95,
                         f"오차의 {sh['alias'] * 100:.0f} % = {ALIAS_HZ:.0f} Hz 접힘\n"
                         f"오차의 {sh['low'] * 100:.0f} % = 40 Hz 아래 (심전도 대역)",
                         transform=axes[1, c_].transAxes, ha="right", va="top",
                         fontsize=9, color=INK)
    # 잔차 칸 위쪽에 글자 자리를 둔다 — 딥러닝 잔차의 봉우리가 글자를 지나갔다
    lo, hi = axes[1, 1].get_ylim()
    axes[1, 1].set_ylim(lo, hi + 0.45 * (hi - lo))
    for ax in axes[1]:
        ax.set_xlabel(TXT["time"])
    avg = " · ".join(f"{NAME[m]} {mean.loc['pli', m]:+.1f}" for m in show)
    fig.suptitle(f"전원선 잡음에서는 고전 방법이 이긴다 — 전처리가 남긴 것은 {ALIAS_HZ:.0f} Hz 뿐이고, "
                 "딥러닝은 이미 깨끗한 심전도를 건드린다\n"
                 f"D1 실기록 {d['record']} · 입력 SNR {EXPB_SNR:+.0f} dB (잡음 종류별 실험과 같은 조건) · "
                 f"숫자 = 이 구간 · 전체 평균(n={n} 구간): {avg} dB",
                 fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "S16_pli_swt.png", dpi=165); plt.close(fig)
    print("  S16_pli_swt.png")


def s17_noise_board():
    """잡음 7 종의 판정 — 딥러닝과 SWT 중 누가 이기나, 전처리만은 어디까지 가나."""
    mean, n = _expb_means()
    show = ["M_FE", "M04", "M08"]
    conds = list(NOISE_LAB)
    # 딥러닝이 이기는 폭 순으로 — 위가 딥러닝, 아래가 고전.
    margin = {c: float(mean.loc[c, "M08"] - mean.loc[c, "M04"]) for c in conds}
    conds.sort(key=lambda c: -margin[c])
    fig, ax = plt.subplots(figsize=(13, 6.2))
    h = 0.26
    for k, m in enumerate(show):
        ys = [i + (k - 1) * h for i in range(len(conds))]
        vals = [float(mean.loc[c, m]) for c in conds]
        ax.barh(ys, vals, height=h * 0.92, color=C[m], label=NAME[m])
        for yv, v in zip(ys, vals):
            ax.text(v + 0.3, yv, f"{v:.1f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([NOISE_LAB[c] for c in conds], fontsize=11.5)
    ax.invert_yaxis()
    xmax = float(mean.loc[conds, show].to_numpy().max())
    for i, c in enumerate(conds):
        dl = margin[c] > 0
        ax.text(xmax + 6.5, i, f"{'딥러닝' if dl else 'SWT'} +{abs(margin[c]):.1f} dB",
                va="center", ha="left", fontsize=11, fontweight="bold",
                color=C["M08"] if dl else C["M04"])
    ax.set_xlim(0, xmax + 14)
    # 세로축은 «전처리 위에 더한 몫» 이 아니라 **개선량 그 자체**다 — 전처리만(M_FE)
    # 막대가 함께 서야 «고전이 이기는 곳은 전처리가 이미 높다» 가 보인다.
    ax.set_xlabel("개선량 [dB]  (클수록 좋다)", labelpad=2)
    ax.grid(axis="y", visible=False)
    # 범례를 축 안에 두면 오른쪽 판정 글자(«SWT +11.1 dB»)와 겹친다 — 눈으로 보고 옮겼다
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=3, fontsize=10.5)
    k_dl = sum(margin[c] > 0 for c in conds)
    fig.suptitle(f"잡음 {len(conds)} 조건 중 딥러닝이 이기는 곳 {k_dl}, SWT 가 이기는 곳 "
                 f"{len(conds) - k_dl} — 고전이 이기는 곳은 전처리만으로도 딥러닝보다 높다\n"
                 f"D1 실기록 · 입력 SNR {EXPB_SNR:+.0f} dB · 조건마다 {n} 구간 평균 "
                 "· 오른쪽 = 딥러닝과 SWT 의 차이",
                 fontsize=12.5, y=0.985)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.87, bottom=0.2)
    fig.savefig(OUT / "S17_noise_board.png", dpi=165); plt.close(fig)
    print("  S17_noise_board.png")


def main() -> int:
    import argparse
    global NAME, TXT, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="S1 S2 ... 일부만 생성")
    ap.add_argument("--snr", type=float, default=-5.0,
                    help="파형 그림의 입력 SNR (기본 -5 dB — 차이가 보이는 구간)")
    ap.add_argument("--present", action="store_true",
                    help="발표용 말로 렌더해 results/slides_present/ 에 쓴다 "
                         "(보고서 그림은 건드리지 않는다)")
    a = ap.parse_args()
    if a.present:
        OUT = PRESENT_OUT
    want = set(a.only) if a.only else {"S1", "S2", "S3", "S4", "S5", "S6",
                                       "S7", "S8", "S9", "S10", "S11", "S12",
                                       "S13", "S14", "S15", "S16", "S17"}

    NAME, TXT = slide_style(present=a.present)
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
    if "S15" in want:
        need |= {("mitdb", a.snr, "pli")}
    if "S16" in want:
        need |= {("mitdb", EXPB_SNR, "pli")}

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
    if "S11" in want:
        s11_nofe_grid(TXT)
    if "S12" in want:
        s12_fe_tradeoff(TXT)
    if "S13" in want:
        s13_frontend_flip(TXT)
    if "S14" in want:
        s14_measurement_traps(TXT)
    if "S5" in want:
        s5_crossover()
    if "S6" in want:
        s6_safety()
    if "S7" in want:
        s7_loss_gap(TXT)
    if "S8" in want:
        s8_clean_preservation(TXT)
    if "S9" in want:
        s9_structure_vs_loss(TXT)
    if "S10" in want:
        s10_loss_by_noise(TXT)
    if "S15" in want:
        s15_pli_psd(cache, a.snr)
    if "S16" in want:
        s16_pli_swt(cache)
    if "S17" in want:
        s17_noise_board()

    save_manifest(OUT, cfg=vars(a), sources=[
        "scripts/make_slides.py", "scripts/run_exp.py", "ecgdn/data/dataset.py",
        "ecgdn/methods/frontend.py", "ecgdn/eval/engine.py",
        "ecgdn/models/losses.py"])
    # 색인(`docs/93`)은 보고서용 묶음의 것이다. 발표용이 덮어쓰면 두 묶음의
    # 설명이 한 문서에서 섞인다 — 발표용의 설계는 `docs/94_presentation.md` 다.
    if not a.present:
        write_index()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
