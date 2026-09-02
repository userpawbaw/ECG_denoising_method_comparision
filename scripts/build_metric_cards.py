#!/usr/bin/env python3
"""시연용 **지표 설명 카드** — "SNR 하나로는 왜 안 되는가" 를 여섯 장으로.

    python3 scripts/build_metric_cards.py            # 여섯 장 전부
    python3 scripts/build_metric_cards.py --only C2 C4

산출: `results/metric_cards/*.png`, 색인 `docs/32_metric_cards.md`.

왜 만드는가
----------
시연장에서 관람객이 앞에 서 있는 시간은 **1 분 남짓**이다. 지표 여섯 개의
정의를 설명할 시간이 없고, 정의를 들어도 **왜 그것이 필요했는지**는 안 남는다.

그래서 지표마다 카드 한 장을 만든다. 카드는 전부 **같은 모양**이다:

    "SNR 로는 비슷하거나 좋은데, 이 지표로는 갈린다"

여섯 번 반복되면 설명 없이도 요점이 박힌다 — **지표 하나로는 잡음 제거와 신호
보존의 충돌이 안 보인다**(보고서 3.1).

카드의 재료는 대부분 **EXP-C(잡음 없는 입력)** 다. 지울 잡음이 0 이므로 출력이
입력과 다르면 그것은 전부 **그 방법이 저지른 짓**이고, 설명이 한 문장으로
끝난다. 이 성질 때문에 EXP-C 가 시연 재료로 EXP-A/B 보다 낫다.

**말 없이 읽혀야 한다.** 각 장은 굵은 제목 한 줄 + 그림 + 숫자 둘 + 한 문장이고,
설명하는 사람이 옆에 없어도 성립해야 한다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecgdn.config import FS
from ecgdn.utils import ensure_dir, save_manifest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "metric_cards"

# ---------------------------------------------------------------- 스타일
# 색은 `make_slides.py` 와 **같은 값**을 쓴다. 두 산출물이 같은 자리에서
# 같이 보이므로 방법 색이 다르면 그것부터 혼란이 된다.
C = {"M01": "#2a78d6", "M04": "#eb6834", "M08": "#1baf7a",
     "M05": "#eda100", "M_FE": "#e87ba4", "M02": "#7b53c1", "M06": "#184f95"}
CLEAN = "#b8b6ae"       # 참값: 뒤에 두껍게 — '목표' 로 읽히게
BAD = "#d03b3b"         # 경계선·경고 (validate_palette.js light PASS)
INK = "#1b1b1b"
MUTE = "#6b6b6b"

def _ko_font() -> str:
    """한글 폰트를 고른다 — 없으면 카드가 두부(□)로 나온다.

    `make_slides.py` 와 **같은 목록·같은 순서**다. 그림이 나오는데 읽을 수
    없는 상태가 가장 나쁘므로, 없으면 경고를 띄운다.
    """
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    ko = next((f for f in ("NanumGothic", "NanumBarunGothic", "NanumSquare",
                           "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR")
               if f in have), None)
    if ko is None:
        print("[cards] 한글 폰트가 없다 — 두부로 렌더된다 "
              "(apt-get install -y fonts-nanum)", file=sys.stderr)
    return ko or "DejaVu Sans"


plt.rcParams.update({
    "font.family": _ko_font(), "axes.unicode_minus": False, "axes.grid": False,
    # 로그축 라벨(10^-3)은 mathtext 로 그려지는데 나눔 폰트에 U+2212(−)가
    # 없다. 지수가 "10¤3" 으로 나온다 — 그림은 나오는데 못 읽는 상태다.
    "mathtext.fontset": "dejavusans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cccccc", "figure.facecolor": "white",
    "axes.titlesize": 12, "axes.labelsize": 10, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "font.size": 10,
})


def card(title: str, sub: str, figsize=(11, 4.6)):
    fig = plt.figure(figsize=figsize, dpi=170)
    fig.text(0.035, 0.955, title, fontsize=16, fontweight="bold", color=INK,
             va="top")
    fig.text(0.035, 0.885, sub, fontsize=10.5, color=MUTE, va="top")
    return fig


def punch(fig, text: str) -> None:
    """카드 맨 아래 한 문장. **이 문장이 카드의 전부**다."""
    fig.text(0.035, 0.045, text, fontsize=11.5, color=INK, va="bottom",
             wrap=True)


def save(fig, name: str) -> Path:
    p = ensure_dir(OUT) / f"{name}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {p.relative_to(ROOT)}")
    return p


# ------------------------------------------------------------ 공용 재료
def clean_segment(tag: str, seconds: float = 10.0, seed: int = 0):
    """**잡음 없는** 구간 하나. EXP-C 와 같은 조건이다."""
    from ecgdn.data.dataset import ECGDenoiseDataset
    from ecgdn.data.sources import get_source
    n = int(seconds * FS)
    src = get_source("synthetic" if tag == "d0" else "mitdb")
    ds = ECGDenoiseDataset(source=src, split="test", win=n, hop=n, max_per_record=2)
    i = int(np.random.default_rng(seed).integers(len(ds)))
    d = ds.raw_item(i)
    return d["x"].astype(np.float64)      # clean 만 쓴다


def run_clean(mid: str, x: np.ndarray) -> np.ndarray:
    """깨끗한 입력을 그대로 통과시킨다 (EXP-C 의 정의)."""
    from ecgdn.methods import build
    return np.asarray(build(mid)(x, FS, {}), dtype=np.float64)


def fe(x: np.ndarray) -> np.ndarray:
    """참조는 **front-end 를 통과한 clean** 이다 (D-3). 방법들과 같은 출발선."""
    from ecgdn.methods.frontend import FrontEnd
    return np.asarray(FrontEnd()(x, FS), dtype=np.float64)


def one_beat(x: np.ndarray, pre=0.20, post=0.35):
    """R-peak 하나를 중심으로 자를 구간. 형태 차이는 확대해야 보인다."""
    from ecgdn.eval.rpeak import detect_rpeaks
    r = np.asarray(detect_rpeaks(x, FS), dtype=int)
    j = int(r[len(r) // 2]) if r.size else len(x) // 2
    return max(0, j - int(pre * FS)), min(len(x), j + int(post * FS))


def table(tag: str, exp: str) -> pd.DataFrame:
    f = ROOT / "results" / tag / exp / "metrics.parquet"
    if not f.exists():
        raise SystemExit(f"{f} 가 없다 — 그 실험을 먼저 돌릴 것")
    df = pd.read_parquet(f)
    return df.pivot_table(index="method", columns="metric", values="value",
                          aggfunc="mean")


def floor_of(tag: str, metric: str) -> float | None:
    f = ROOT / "results" / tag / "metric_floor" / "floor.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    r = d[d.metric == metric]
    return float(r.floor_p95.iloc[0]) if len(r) else None


# =============================================================== C1
def c1_gain_bias():
    """진폭만 줄어든 출력을 SNR 이 어떻게 처벌하는가."""
    x = fe(clean_segment("d1", 6.0, seed=1))
    lo, hi = one_beat(x)
    xh = 0.9 * x                                  # 파형은 완벽, 크기만 10 % 작다

    def snr_strict(a, b):
        a, b = a - a.mean(), b - b.mean()
        return 10 * np.log10((a @ a) / ((b - a) @ (b - a)))

    g = float((x - x.mean()) @ (x - x.mean()) / ((x - x.mean()) @ (xh - xh.mean())))
    fig = card("① 파형은 완벽한데 크기만 10 % 작다",
               "gain_bias · snr_out_strict vs snr_out_scaled  —  같은 신호를 0.9 배 한 것뿐이다")
    ax = fig.add_axes([0.05, 0.30, 0.52, 0.50])
    t = np.arange(hi - lo) / FS * 1000
    ax.plot(t, x[lo:hi], color=CLEAN, lw=4, label="참값")
    ax.plot(t, xh[lo:hi], color=C["M06"], lw=1.6, label="출력 (0.9 x)")
    ax.set_xlabel("ms"); ax.set_yticks([]); ax.legend(frameon=False, loc="upper right")

    ax2 = fig.add_axes([0.66, 0.30, 0.31, 0.50]); ax2.axis("off")
    rows = [("snr_out_strict", f"{snr_strict(x, xh):.1f} dB", BAD, "그대로 쓰면"),
            ("snr_out_scaled", "∞", C["M08"], "이득 보정 후"),
            ("gain_bias", f"{g:.2f}", INK, "진폭이 눌린 양")]
    for k, (name, val, col, note) in enumerate(rows):
        y = 0.82 - k * 0.33
        ax2.text(0, y, name, fontsize=10, color=MUTE, transform=ax2.transAxes)
        ax2.text(0, y - 0.13, val, fontsize=20, fontweight="bold", color=col,
                 transform=ax2.transAxes)
        ax2.text(0.42, y - 0.10, note, fontsize=9.5, color=MUTE,
                 transform=ax2.transAxes)
    punch(fig, "SNR 하나만 쓰면 «크기만 10 % 작은 출력» 이 «파형이 10 % 틀어진 출력» 과 같은 점수를 받는다.\n"
               "soft threshold · 칼만 · MSE 로 학습한 딥러닝은 구조적으로 진폭을 줄인다 — 그래서 셋을 함께 적는다.")
    return save(fig, "C1_gain_bias")


# =============================================================== C2
def c2_r_amp(tag: str = "d0"):
    """잡음이 0 인데 R-peak 를 깎는다."""
    x = clean_segment(tag, 8.0, seed=2)
    ref = fe(x)
    outs = {m: run_clean(m, x) for m in ("M02", "M04")}
    lo, hi = one_beat(ref)
    t = table(tag, "exp_c")

    fig = card("② 잡음이 하나도 없는데 R-peak 를 깎았다",
               f"r_amp_err_pct  ·  EXP-C({tag}): 입력에 잡음을 0 으로 넣고 그대로 통과시킨다")
    ax = fig.add_axes([0.05, 0.30, 0.52, 0.50])
    tt = np.arange(hi - lo) / FS * 1000
    ax.plot(tt, ref[lo:hi], color=CLEAN, lw=4.5, label="참값 (= 입력)")
    for m in ("M04", "M02"):
        ax.plot(tt, outs[m][lo:hi], color=C[m], lw=1.7, label=m)
    ax.set_xlabel("ms"); ax.set_yticks([]); ax.legend(frameon=False, loc="upper right")

    ax2 = fig.add_axes([0.66, 0.30, 0.31, 0.50]); ax2.axis("off")
    for k, m in enumerate(("M02", "M04")):
        y = 0.80 - k * 0.44
        ax2.text(0, y, f"{m}  ({'Savitzky-Golay' if m=='M02' else 'SWT + QRS 보호'})",
                 fontsize=10, color=C[m], fontweight="bold", transform=ax2.transAxes)
        ax2.text(0, y - 0.16, f"R-peak 오차 {t.loc[m,'r_amp_err_pct']:.2f} %",
                 fontsize=15, fontweight="bold",
                 color=BAD if m == "M02" else INK, transform=ax2.transAxes)
        ax2.text(0, y - 0.29, f"왜곡 하한 {t.loc[m,'snr_out_strict']:.1f} dB",
                 fontsize=10, color=MUTE, transform=ax2.transAxes)
    punch(fig, "«고주파 = 잡음» 이라고 가정하는 순간 QRS 부터 깎인다 — QRS 가 ECG 에서 가장 고주파이기 때문이다.\n"
               "SNR 만 보면 «조금 나쁘다» 인데, R-peak 진폭으로 보면 진단에 쓰는 그 값이 사라진 것이다.")
    return save(fig, "C2_r_amp_err")


# =============================================================== C3
def c3_psd(tag: str = "d0"):
    """무엇을 지웠는지는 스펙트럼에만 보인다."""
    from ecgdn.eval.spectral import welch_psd
    x = clean_segment(tag, 20.0, seed=3)
    ref = fe(x)
    outs = {m: run_clean(m, x) for m in ("M01", "M04")}
    t = table(tag, "exp_c")

    fig = card("③ SNR 은 상위권인데 스펙트럼은 최악이다",
               f"psd_logdist  ·  EXP-C({tag}): 잡음 0 입력. 없어진 것은 전부 방법이 지운 것이다")
    ax = fig.add_axes([0.05, 0.30, 0.52, 0.50])
    # **로그축 대신 dB 로 그린다.** `10^-5` 같은 지수 라벨은 (a) 한글 폰트에
    # 마이너스 글리프가 없어 깨지고 (b) 이 관객에게는 dB 가 더 익숙하다.
    # 참값의 최댓값을 0 dB 로 잡아 "얼마나 깎였나" 를 바로 읽게 한다.
    f, pr = welch_psd(ref, FS)
    top = float(np.max(pr))
    db = lambda v: 10.0 * np.log10(np.maximum(v, top * 1e-12) / top)
    ax.plot(f, db(pr), color=CLEAN, lw=4, label="참값")
    for m in ("M04", "M01"):
        f, pm = welch_psd(outs[m], FS)
        ax.plot(f, db(pm), color=C[m], lw=1.5, label=m)
    ax.axvline(40, color=BAD, lw=1, ls="--")
    ax.text(41, -3, "40 Hz", fontsize=9, color=BAD)
    ax.set_xlim(0, 100); ax.set_ylim(-72, 4)
    ax.set_xlabel("Hz"); ax.set_ylabel("PSD [dB, 참값 최대 = 0]")
    ax.legend(frameon=False, loc="lower left")

    ax2 = fig.add_axes([0.66, 0.30, 0.31, 0.50]); ax2.axis("off")
    for k, m in enumerate(("M01", "M04")):
        y = 0.80 - k * 0.44
        ax2.text(0, y, f"{m}  ({'대역통과 + notch' if m=='M01' else 'SWT + QRS 보호'})",
                 fontsize=10, color=C[m], fontweight="bold", transform=ax2.transAxes)
        ax2.text(0, y - 0.16, f"psd_logdist {t.loc[m,'psd_logdist']:.2f}",
                 fontsize=15, fontweight="bold",
                 color=BAD if m == "M01" else INK, transform=ax2.transAxes)
        ax2.text(0, y - 0.29, f"왜곡 하한 {t.loc[m,'snr_out_strict']:.1f} dB",
                 fontsize=10, color=MUTE, transform=ax2.transAxes)
    punch(fig, "M01 은 40 Hz 위를 통째로 버렸다. 그런데 SNR 은 그것을 «잡음을 잘 지웠다» 로 센다 —\n"
               "무엇을 지웠는지는 파형이 아니라 스펙트럼에만 보인다. 그래서 층 4 가 따로 있다.")
    return save(fig, "C3_psd_logdist")


# =============================================================== C4
def c4_floor(tag: str = "d1", metric: str = "qrs_dur_err_ms"):
    """지표 자체의 분해능 — 바닥 아래에는 순위가 없다."""
    t = table(tag, "exp_c")[metric].dropna().sort_values()
    fl = floor_of(tag, metric)
    if fl is None:
        raise SystemExit(f"{tag} floor.csv 가 없다")
    # 손실 변형(-L3/-L5/-L6)과 중복 방법은 뺀다 — **지표**를 설명하는
    # 카드이지 방법 목록이 아니다. 20 단이 넘으면 선 위/아래 대비가 죽는다.
    drop = ("M04np",)
    t = t[[m for m in t.index if m not in drop and "-L" not in m]]

    fig = card("④ 이 선 아래에서는 순위가 없다",
               f"qrs_dur_err_ms  ·  {tag} 분해능 바닥(floor_p95) = {fl:.2f} ms  —  지표 자체가 그만큼밖에 못 잰다",
               figsize=(11, 5.2))
    ax = fig.add_axes([0.12, 0.22, 0.84, 0.58])
    cols = [BAD if v > fl else "#c9c9c9" for v in t.values]
    ax.barh(range(len(t)), t.values, color=cols, height=0.68)
    ax.set_yticks(range(len(t))); ax.set_yticklabels(t.index, fontsize=9)
    ax.axvline(fl, color=BAD, lw=1.8)
    ax.text(fl - max(t.values) * 0.012, -0.35, f"분해능 바닥 {fl:.1f} ms →",
            color=BAD, fontsize=10, fontweight="bold", ha="right", va="center")
    ax.set_xlabel("QRS duration 오차 [ms]  (작을수록 좋다)")
    for i, v in enumerate(t.values):
        ax.text(v + max(t.values) * 0.01, i, f"{v:.1f}", va="center",
                fontsize=8.5, color=INK if v > fl else MUTE)
    n_below = int((t.values <= fl).sum())
    punch(fig, f"{len(t)} 개 방법 중 {n_below} 개가 선 아래에 있다 — 이들 사이의 순위는 «측정 잡음»이지 성능차가 아니다.\n"
               "이 바닥을 재지 않고 표를 만들면 그 표의 절반은 의미가 없다. 그래서 실험 전에 먼저 쟀다.")
    return save(fig, "C4_metric_floor")


# =============================================================== C5
def c5_halluc(tag: str = "d1"):
    """지운 자리에 무엇이 생기는가."""
    from ecgdn.data.mixer import mix_at_snr
    from ecgdn.data.noise import mixed_noise
    from ecgdn.methods import build
    from ecgdn.utils import rng

    x = clean_segment(tag, 12.0, seed=5)
    i0 = int(len(x) * 0.40)
    i1 = min(len(x), i0 + int(3.0 * FS))              # P2 — asystole 3 s
    x_mod = x.copy()
    x_mod[i0:i1] = float(np.median(x[max(0, i0 - int(0.3 * FS)):i0]))
    g = rng("card", "p2", 0)
    n, _ = mixed_noise(len(x), FS, g)
    y, _, _ = mix_at_snr(x_mod, n, 5.0)

    outs = {"M_FE": np.asarray(build("M_FE")(y, FS, {}), dtype=np.float64)}
    ck = ROOT / "results" / tag / "m06_l1" / "best.pt"
    if ck.exists():
        from ecgdn.methods.dl_wrapper import DLDenoiser
        outs["M06"] = np.asarray(DLDenoiser(ckpt=ck, name="M06")(y, FS, {}),
                                 dtype=np.float64)
    pr = pd.read_csv(ROOT / "results" / tag / "exp_e" / "probe.csv")
    he = pr[pr.probe == "p2"].groupby("method").halluc_energy.mean()

    lo, hi = max(0, i0 - int(1.0 * FS)), min(len(x), i1 + int(1.0 * FS))
    fig = card("⑤ 심장이 3 초 멈춘 구간 — 없는 파형을 만들어내는가",
               f"halluc_energy  ·  EXP-E P2({tag}): 3 초를 등전위선으로 지운 뒤 잡음을 얹어 통과시킨다")
    ax = fig.add_axes([0.05, 0.30, 0.52, 0.50])
    tt = np.arange(hi - lo) / FS
    ax.axvspan((i0 - lo) / FS, (i1 - lo) / FS, color="#f2f2f2", zorder=0)
    ax.plot(tt, y[lo:hi], color="#d8d8d8", lw=0.8, label="입력 (잡음)")
    for m, v in outs.items():
        ax.plot(tt, v[lo:hi], color=C[m], lw=1.4, label=m)
    ax.set_xlabel("s"); ax.set_yticks([]); ax.legend(frameon=False, loc="upper right",
                                                    ncol=3, fontsize=8.5)
    ax.text((i0 - lo) / FS + 0.1, ax.get_ylim()[0] * 0.92,
            "지운 구간 (여기엔 아무것도 없어야 한다)", fontsize=9, color=MUTE)
    # **정직하게 적는다.** 이 배율에서는 두 출력이 똑같이 평평해 보인다.
    # 차이는 그림이 아니라 오른쪽 숫자에만 있다 — 그것이 지표가 필요한 이유다.
    ax.text((i0 - lo) / FS + 0.1, ax.get_ylim()[0] * 0.72,
            "육안으로는 둘 다 평평하다. 차이는 오른쪽 숫자에만 있다.",
            fontsize=8.5, color=BAD)

    ax2 = fig.add_axes([0.66, 0.28, 0.31, 0.54])
    order = [m for m in ("M00", "M_FE", "M01", "M04", "M05", "B01", "M08", "M06", "M07", "M10")
             if m in he.index]
    v = he[order]
    ax2.barh(range(len(v)), v.values,
             color=[C["M06"] if m.startswith(("M06", "M07", "M08", "M10")) else "#c9c9c9"
                    for m in order], height=0.7)
    ax2.set_yticks(range(len(v))); ax2.set_yticklabels(order, fontsize=8.5)
    ax2.set_xlabel("halluc_energy  (낮을수록 안전)", fontsize=9)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    punch(fig, "«딥러닝은 없는 파형을 지어낸다» 는 걱정이 흔하다. 재보니 정반대였다 — 파랑(딥러닝)이 가장 낮다.\n"
               f"M06 {he.get('M06', float('nan')):.3f} vs front-end {he.get('M_FE', float('nan')):.3f} · identity {he.get('M00', float('nan')):.3f}. 이것을 안 재면 «아마 위험할 것» 이 결론으로 남는다.")
    return save(fig, "C5_halluc_energy")


# =============================================================== C6
def c6_downstream(tag: str = "d1"):
    """층 3 에서는 차이가 사라진다 — 그것도 결과다."""
    t = table(tag, "exp_b")
    v = t["hr_err_bpm"].dropna().sort_values()
    v = v[[m for m in v.index if m != "M04np"]]
    s = t["snr_imp_scaled"].reindex(v.index)

    fig = card("⑥ 차이가 없다는 것도 결과다",
               f"hr_err_bpm  ·  EXP-B({tag}) 잡음 7 종 10 dB  —  층 3(하류 과제)에서 방법 간 차이가 사라진다",
               figsize=(11, 5.0))
    ax = fig.add_axes([0.07, 0.55, 0.88, 0.27])
    ax.bar(range(len(s)), s.values, color="#c9c9c9", width=0.7)
    ax.set_xticks([]); ax.set_ylabel("SNR 개선 [dB]", fontsize=9)
    ax.set_ylim(0, max(s.values) * 1.32)      # 라벨이 막대에 걸리지 않게
    ax.text(0.01, 0.86, f"SNR 은 {s.min():.1f} ~ {s.max():.1f} dB 로 갈린다",
            transform=ax.transAxes, fontsize=10, color=INK)

    ax2 = fig.add_axes([0.07, 0.20, 0.88, 0.27])
    ax2.bar(range(len(v)), v.values, color=C["M08"], width=0.7)
    ax2.set_xticks(range(len(v))); ax2.set_xticklabels(v.index, fontsize=8.5, rotation=0)
    ax2.set_ylabel("심박수 오차 [bpm]", fontsize=9)
    ax2.set_ylim(0, max(v.values) * 1.5)
    ax2.text(0.01, 0.86, f"심박수 오차는 {v.min():.2f} ~ {v.max():.2f} bpm — 전부 같다",
             transform=ax2.transAxes, fontsize=10, color=INK)
    punch(fig, "심박수만 필요하면 front-end 로 충분하다. 우리가 재는 차이는 «형태가 필요할 때» 의미가 있다.\n"
               "지표를 층으로 나누지 않았다면 «SNR 이 올랐으니 좋아졌다» 로 끝났을 것이다.")
    return save(fig, "C6_downstream")


CARDS = {"C1": c1_gain_bias, "C2": c2_r_amp, "C3": c3_psd,
         "C4": c4_floor, "C5": c5_halluc, "C6": c6_downstream}

# 카드마다 (지표, 무엇을 잡나, 근거). 문서는 **이 표에서 생성**한다 — 손으로
# 쓰면 카드와 문서가 갈라지고 그것을 알아챌 방법이 없다(F-9 계열).
INDEX = [
    ("C1", "gain_bias · strict/scaled", "진폭만 눌린 출력을 SNR 이 과하게 처벌한다",
     "합성 예시 (0.9 x). 보고서 3.2.2 의 단위 테스트와 같은 상황"),
    ("C2", "r_amp_err_pct", "«고주파 = 잡음» 가정이 QRS 부터 깎는다",
     "EXP-C(d0) — 잡음 0 입력"),
    ("C3", "psd_logdist", "무엇을 지웠는지는 스펙트럼에만 보인다",
     "EXP-C(d0) — 잡음 0 입력"),
    ("C4", "qrs_dur_err_ms + floor_p95", "지표 자체의 분해능 아래에는 순위가 없다",
     "EXP-C(d1) + results/d1/metric_floor"),
    ("C5", "halluc_energy", "없는 파형을 지어내는가 — 딥러닝이 가장 안전했다",
     "EXP-E P2(d1) — 3 초를 지운 프로브"),
    ("C6", "hr_err_bpm", "층 3 에서는 방법 간 차이가 사라진다",
     "EXP-B(d1) 잡음 7 종 10 dB"),
]


def write_index(made: list[str]) -> Path:
    L = ["# 32. 시연용 지표 카드 — «SNR 하나로는 왜 안 되는가»", "",
         "> 이 문서는 `scripts/build_metric_cards.py` 가 만든다. **직접 고치지 말 것.**",
         "> 그림은 `results/metric_cards/`. 설계 논의는 `docs/31_demo_design_review.md`.",
         "",
         "시연장에서 관람객이 앞에 서 있는 시간은 **1 분 남짓**이다. 지표 여섯 개의",
         "정의를 설명할 시간이 없고, 정의를 들어도 **왜 그것이 필요했는지**는 안 남는다.",
         "그래서 지표마다 카드 한 장을 만들고, 카드를 전부 **같은 모양**으로 뒀다:",
         "",
         "> **«SNR 로는 비슷하거나 좋은데, 이 지표로는 갈린다»**",
         "",
         "여섯 번 반복되면 설명 없이도 요점이 박힌다 — **지표 하나로는 잡음 제거와",
         "신호 보존의 충돌이 안 보인다**(보고서 3.1).",
         "",
         "## 카드", "",
         "| | 지표 | 무엇을 잡나 | 근거 |", "|---|---|---|---|"]
    for cid, met, what, src in INDEX:
        mark = "" if cid in made else " *(미생성)*"
        L.append(f"| **{cid}**{mark} | `{met}` | {what} | {src} |")
    L += ["",
          "```bash",
          "python3 scripts/build_metric_cards.py            # 여섯 장",
          "python3 scripts/build_metric_cards.py --only C2 C4",
          "```",
          "",
          "## 재료가 왜 EXP-C 인가",
          "",
          "카드 넷 중 둘이 **EXP-C(잡음 없는 입력)** 를 쓴다. 지울 잡음이 0 이므로",
          "**출력이 입력과 다르면 그것은 전부 그 방법이 저지른 짓**이고, 설명이 한",
          "문장으로 끝난다. EXP-A/B 는 «잡음이 남은 것인지 신호를 깎은 것인지» 를",
          "그림만 보고 가릴 수 없어 시연 재료로는 약하다.",
          "",
          "## 넣지 않은 것", "",
          "**`beat_cc` 카드** — 값이 0.992~1.000 이라 **눈에 보이는 차이가 안 난다.**",
          "흥미로운 이야기는 정렬 기준 쪽인데(자기 R-peak 로 맞추면 타이밍 오차가",
          "상쇄돼 사라진다 — 보고서 3.3.2), 그것은 **방법 비교가 아니라 방법론**이라",
          "여섯 장의 리듬과 결이 다르다. 억지로 채우지 않았다.",
          "",
          "**화면에 지표 선택기를 다는 안** — 붙이기는 쉽지만 **아무것도 가르치지",
          "않는다.** `psd_logdist` 가 무엇인지 모르는 사람이 숫자 11 개가 바뀌는 것을",
          "본다고 알게 되지 않는다. 선택기는 카드를 이미 본 사람에게만 쓸모가 있다.",
          "",
          "## C5 가 스스로 밝히는 한계", "",
          "지운 구간에서 `M_FE` 와 `M06` 은 **육안으로 똑같이 평평하다.** 차이는",
          "그림이 아니라 숫자에만 있고, 카드가 그 사실을 그림 안에 적어 둔다.",
          "**그것이 지표가 필요한 이유 자체**이므로 숨기지 않는다.", ""]
    p = ROOT / "docs" / "32_metric_cards.md"
    p.write_text("\n".join(L))
    print(f"  -> {p.relative_to(ROOT)}")
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, choices=sorted(CARDS))
    a = ap.parse_args()
    made = []
    for k in (a.only or sorted(CARDS)):
        print(f"[{k}]")
        made.append(CARDS[k]())
    save_manifest(OUT, cfg={"cards": [p.stem for p in made]}, sources=[__file__])
    if a.only is None:                    # 일부만 만들었으면 색인을 덮지 않는다
        write_index([p.stem.split("_")[0] for p in made])
    print(f"\n{len(made)} 장 -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
