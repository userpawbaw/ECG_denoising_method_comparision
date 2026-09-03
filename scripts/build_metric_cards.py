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
    """R-peak 하나를 중심으로 자를 구간. 형태 차이는 확대해야 보인다.

    **교육용 카드에는 `pick_teaching_beat` 를 쓸 것.** 이 함수는 «가운데 박동»
    을 그냥 집으므로 T 파가 뒤집힌 박동에 걸릴 수 있다 (실제로 걸렸다).
    """
    from ecgdn.eval.rpeak import detect_rpeaks
    r = np.asarray(detect_rpeaks(x, FS), dtype=int)
    j = int(r[len(r) // 2]) if r.size else len(x) // 2
    return max(0, j - int(pre * FS)), min(len(x), j + int(post * FS))


# ----------------------------------------------------- 교육용 박동 고르기
# 카드는 **심전도를 처음 보는 사람**에게 «이것이 P-QRS-T 다» 를 먼저 가르쳐야
# 한다. 그런데 무작위로 집은 박동은 그 일을 못 할 수 있다 — 첫 판에서 실제로
# **T 파가 뒤집힌 박동**이 뽑혔다. 합성기의 `jitter_kernel` 이 12 % 확률로 T 를
# 뒤집는 것은 **의도된 변이**라 버그가 아니지만, 교육용으로는 최악이다.
#
# 그래서 «가르치기 좋은 박동» 을 점수로 정의해 고른다. 점수 항목은 전부
# **교과서적 정상 박동**의 조건이다.
BEAT_W = dict(t_pos=1.0, p_pos=0.8, r_dom=1.0, tmpl=1.2)


def _beat_score(v: np.ndarray, j: int, rr: int, tmpl: np.ndarray | None):
    """박동 하나가 «가르치기 좋은가». 0~1 로 정규화한 항목의 가중합.

    반환 `(점수, 항목별 값)`. 항목을 함께 돌려주는 것은 **왜 그 박동이 뽑혔는지
    카드 만들 때 확인할 수 있어야** 하기 때문이다.
    """
    i0, i1 = j - int(0.25 * FS), j + int(0.45 * FS)
    if i0 < 0 or i1 > v.size:
        return -1.0, {}
    seg = v[i0:i1]
    base = float(np.median(v[j + int(0.42 * rr / FS * FS):min(v.size, j + rr)])) \
        if j + rr <= v.size else float(np.median(seg))
    r_amp = float(v[j]) - base
    if r_amp <= 0:
        return -1.0, {}
    # T 파: R+150~400 ms 최고점이 기저선 위로 얼마나 (R 진폭 대비)
    t = (float(np.max(v[j + int(0.15 * FS): j + int(0.40 * FS)])) - base) / r_amp
    # P 파: R−250~−100 ms 최고점
    p = (float(np.max(v[j - int(0.25 * FS): j - int(0.10 * FS)])) - base) / r_amp
    # R 지배: 박동 안에서 R 이 가장 큰 편위인가 (아래로 큰 것이 있으면 감점)
    dom = r_amp / max(r_amp, float(np.max(np.abs(seg - base))))
    it = dict(t_pos=float(np.clip(t / 0.25, 0, 1)),      # T 가 R 의 25 % 면 만점
              p_pos=float(np.clip(p / 0.15, 0, 1)),      # P 는 15 %
              r_dom=float(np.clip(dom, 0, 1)))
    if tmpl is not None and tmpl.size == seg.size:
        a, b = seg - seg.mean(), tmpl - tmpl.mean()
        den = float(np.sqrt((a @ a) * (b @ b)))
        it["tmpl"] = float(np.clip((a @ b) / den, 0, 1)) if den > 0 else 0.0
    else:
        it["tmpl"] = 0.0
    return sum(BEAT_W[k] * it[k] for k in BEAT_W) / sum(BEAT_W.values()), it


def pick_teaching_beat(tag: str, pad_s: float = 2.5, max_records: int = 8):
    """교육용 박동 하나를 고른다 — **P 양수 · T 양수 · R 지배 · PVC 아님 ·
    자기 기록의 템플릿에 가까움.**

    반환 `(x_raw, lo, hi, info)`. `x_raw` 는 **front-end 를 통과하기 전** 구간
    이고(방법을 통과시켜야 하므로), `lo:hi` 는 그 안에서 그릴 박동의 범위다.
    앞뒤로 `pad_s` 를 붙인다 — 0.5 Hz 고역통과는 수 초간 울리므로 여유가 없으면
    잘라낸 자리가 전부 트랜지언트가 된다.
    """
    from ecgdn.data.sources import get_source
    src = get_source("synthetic" if tag == "d0" else "mitdb")
    best = None
    for name in src.records("test")[:max_records]:
        rec = src.get(name)
        x_raw = np.asarray(rec.x, dtype=np.float64)
        v = fe(x_raw)
        rp = np.asarray(rec.r_peaks, dtype=int)
        sym = np.asarray(rec.symbols)
        keep = sym == "N"                       # **PVC·기타는 후보에서 뺀다**
        if keep.sum() < 8:
            continue
        rr = int(np.median(np.diff(rp))) if rp.size > 2 else int(0.8 * FS)
        # 템플릿 = 정상 beat 들의 중앙값 파형
        w0, w1 = int(0.25 * FS), int(0.45 * FS)
        stack = [v[j - w0: j + w1] for j in rp[keep]
                 if j - w0 >= 0 and j + w1 <= v.size]
        tmpl = np.median(np.stack(stack), axis=0) if len(stack) >= 5 else None
        for j in rp[keep]:
            pad = int(pad_s * FS)
            if j - pad - w0 < 0 or j + pad + w1 > v.size:
                continue                        # 여유가 안 나오는 자리는 뺀다
            sc, it = _beat_score(v, int(j), rr, tmpl)
            if best is None or sc > best[0]:
                best = (sc, str(name), int(j), it)
    if best is None:
        raise SystemExit(f"{tag}: 교육용 박동을 못 골랐다 — 조건을 확인할 것")
    sc, name, j, it = best
    rec = src.get(name)
    pad = int(pad_s * FS)
    a, b = j - pad - int(0.25 * FS), j + pad + int(0.45 * FS)
    x_raw = np.asarray(rec.x, dtype=np.float64)[a:b]
    lo = (j - a) - int(0.20 * FS)
    hi = (j - a) + int(0.35 * FS)
    info = dict(record=name, score=round(sc, 3),
                **{k: round(vv, 2) for k, vv in it.items()})
    return x_raw, lo, hi, info


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
    x_raw, lo, hi, beat = pick_teaching_beat("d1")
    x = fe(x_raw)
    xh = 0.9 * x                                  # 파형은 완벽, 크기만 10 % 작다

    def snr_strict(a, b):
        a, b = a - a.mean(), b - b.mean()
        return 10 * np.log10((a @ a) / ((b - a) @ (b - a)))

    g = float((x - x.mean()) @ (x - x.mean()) / ((x - x.mean()) @ (xh - xh.mean())))
    fig = card("① 파형은 완벽한데 크기만 10 % 작다",
               "gain_bias · snr_out_strict vs snr_out_scaled  —  같은 신호를 0.9 배 한 것뿐이다"
               f"   (d1 기록 {beat['record']})")
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
    x, lo, hi, beat = pick_teaching_beat(tag)
    ref = fe(x)
    outs = {m: run_clean(m, x) for m in ("M02", "M04")}
    t = table(tag, "exp_c")

    fig = card("② 잡음이 하나도 없는데 R-peak 를 깎았다",
               f"r_amp_err_pct  ·  EXP-C({tag}): 입력에 잡음을 0 으로 넣고 그대로 통과시킨다"
               f"   ({tag} 기록 {beat['record']})")
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
# **C6 을 여기에 흡수했다.** 둘 다 «차이가 없다» 를 말해 중복이었고, 다른 것은
# **왜 없는가** 뿐이다 — C4 는 «지표가 못 잰다», C6 은 «실제로 없다». 한 장에
# 나란히 두면 그 구분 자체가 카드의 내용이 된다.
C4_METHODS = ("M00", "M_FE", "M01", "M04", "M05", "M06", "M08")


def c4_floor(tag: str = "d1", metric: str = "qrs_dur_err_ms"):
    """지표의 분해능(왼쪽)과 하류 과제의 무차별(오른쪽) — «차이가 없다» 두 종류."""
    t = table(tag, "exp_c")[metric].dropna()
    fl = floor_of(tag, metric)
    if fl is None:
        raise SystemExit(f"{tag} floor.csv 가 없다")
    # **방법 목록이 아니라 지표를 설명하는 카드다.** 15 단이 넘으면 선 위/아래
    # 대비가 죽으므로 대표 일곱만 남긴다.
    t = t[[m for m in C4_METHODS if m in t.index]].sort_values()

    b = table(tag, "exp_b")
    hr = b["hr_err_bpm"].dropna()
    hr = hr[[m for m in C4_METHODS if m in hr.index]]
    sn = b["snr_imp_scaled"].reindex(hr.index).dropna()

    fig = card("④ «차이가 없다» 에는 두 종류가 있다",
               f"{metric} + floor_p95  ·  hr_err_bpm  —  «지표가 못 재는 것» 과 «실제로 없는 것»",
               figsize=(12.5, 5.0))

    ax = fig.add_axes([0.10, 0.30, 0.40, 0.50])
    cols = [BAD if v > fl else "#c9c9c9" for v in t.values]
    ax.barh(range(len(t)), t.values, color=cols, height=0.66)
    ax.set_yticks(range(len(t))); ax.set_yticklabels(t.index, fontsize=9.5)
    ax.axvline(fl, color=BAD, lw=1.8)
    ax.set_xlabel("QRS duration 오차 [ms]  (작을수록 좋다)", fontsize=9.5)
    ax.set_title(f"못 재는 것 — 분해능 바닥 {fl:.1f} ms", fontsize=10.5,
                 loc="left", color=INK)
    for i, v in enumerate(t.values):
        ax.text(v + max(t.values) * 0.015, i, f"{v:.1f}", va="center",
                fontsize=8.5, color=INK if v > fl else MUTE)
    n_below = int((t.values <= fl).sum())

    ax2 = fig.add_axes([0.60, 0.55, 0.37, 0.25])
    ax2.bar(range(len(sn)), sn.values, color="#c9c9c9", width=0.68)
    ax2.set_xticks([]); ax2.set_ylabel("SNR 개선 [dB]", fontsize=9)
    ax2.set_ylim(0, max(sn.values) * 1.35)
    ax2.set_title(f"실제로 없는 것 — SNR 은 {sn.min():.1f}~{sn.max():.1f} dB 로 갈리는데",
                  fontsize=10.5, loc="left", color=INK)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    ax3 = fig.add_axes([0.60, 0.22, 0.37, 0.25])
    ax3.bar(range(len(hr)), hr.values, color=C["M08"], width=0.68)
    ax3.set_xticks(range(len(hr))); ax3.set_xticklabels(hr.index, fontsize=8.5)
    ax3.set_ylabel("심박수 오차 [bpm]", fontsize=9)
    ax3.set_ylim(0, max(hr.values) * 1.5)
    ax3.text(0.01, 0.84, f"심박수 오차는 {hr.min():.2f}~{hr.max():.2f} bpm — 전부 같다",
             transform=ax3.transAxes, fontsize=9.5, color=INK)
    for sp in ("top", "right"):
        ax3.spines[sp].set_visible(False)

    punch(fig, f"왼쪽: {len(t)} 개 중 {n_below} 개가 선 아래다 — 이들 사이의 순위는 «측정 잡음»이지 성능차가 아니다.\n"
               "오른쪽: 심박수만 필요하면 front-end 로 충분하다. 우리가 재는 차이는 «형태가 필요할 때» 의미가 있다.")
    return save(fig, "C4_metric_floor")


# =============================================================== C5
def c5_pvc():
    """**합성에서만 보이던 위험이 실데이터에서 사라졌다** — EXP-E P3.

    처음 판은 P2(3 초 지움)로 만들었는데, 그 배율에서는 모든 출력이 똑같이
    평평해 **그림이 일을 안 했다.** P3 로 바꾸면 훼손이 파형에 그대로 보인다.

    그리고 **한 축만 그리면 안 된다.** `M05` 의 PVC 훼손은 **D0 에서만** 나타나고
    D1 에서는 부호가 뒤집힌다(F-28). 두 축을 나란히 두는 것이 이 카드의 내용이다 —
    「합성에서 보인 것은 실데이터에서 다시 물어야 한다」(F-8)를 한 장으로 말한다.
    """
    from ecgdn.data.mixer import mix_at_snr
    from ecgdn.data.noise import mixed_noise
    from ecgdn.data.nstdb import make_banks
    from ecgdn.data.sources import get_source
    from ecgdn.eval.morphology import beat_matrix
    from ecgdn.methods import build as build_method
    from ecgdn.utils import rng

    banks = make_banks("test", "data/raw/nstdb")
    m05 = build_method("M05")
    panes = []
    for tag in ("d0", "d1"):
        src = get_source("synthetic" if tag == "d0" else "mitdb")
        got = None
        for name in src.records("test"):
            rec = src.get(name)
            sym = np.asarray(rec.symbols)
            rp = np.asarray(rec.r_peaks, dtype=int)
            v = np.flatnonzero(sym == "V")
            if v.size < 2:
                continue
            # P3 와 **같은 조건**: 60 s 구간, mixed 잡음, 5 dB
            n_seg = int(60.0 * FS)
            j = int(rp[v[v.size // 2]])
            st = max(0, min(int(rec.x.size) - n_seg, j - n_seg // 2))
            x = np.asarray(rec.x, dtype=np.float64)[st:st + n_seg]
            m = (rp >= st) & (rp < st + n_seg)
            rp_s, sym_s = rp[m] - st, sym[m]
            if (sym_s == "V").sum() < 2 or (sym_s == "N").sum() < 4:
                continue
            g = rng("probe", "p3", str(name), 0)
            nz, _ = mixed_noise(x.size, FS, g, banks=banks)
            y, _, _ = mix_at_snr(x, nz, 5.0)
            out = np.asarray(m05(y, FS, {}), dtype=np.float64)
            got = (str(name), x, out, rp_s, sym_s)
            break
        if got is None:
            raise SystemExit(f"{tag}: PVC 가 있는 구간을 못 찾았다")
        name, x, out, rp_s, sym_s = got
        cc = {}
        for want in ("N", "V"):
            k = sym_s == want
            if k.sum() < 2:
                continue
            br, _ = beat_matrix(x, rp_s[k], FS)
            bh, _ = beat_matrix(out, rp_s[k], FS)
            if br.shape != bh.shape or br.size == 0:
                continue
            aa, bb = br - br.mean(1, keepdims=True), bh - bh.mean(1, keepdims=True)
            den = np.sqrt((aa * aa).sum(1) * (bb * bb).sum(1))
            with np.errstate(invalid="ignore", divide="ignore"):
                cc[want] = float(np.nanmean((aa * bb).sum(1) / den))
        jv = int(rp_s[np.flatnonzero(sym_s == "V")[0]])
        lo, hi = max(0, jv - int(0.35 * FS)), min(x.size, jv + int(0.45 * FS))
        # **축 평균을 산출물에서 읽어 함께 적는다.** 카드가 기록 하나만 보이면
        # 그 숫자가 보고서(5.5 · 5.8.4)의 축 평균과 달라 보인다 — 축을 안 적은
        # 인용이 F-28 을 만들었다. 두 숫자를 같이 두면 어긋날 수 없다.
        pr = pd.read_csv(ROOT / "results" / tag / "exp_e" / "probe.csv")
        p3 = pr[(pr.probe == "p3") & (pr.method == "M05")]
        piv = p3.pivot_table(index="method", columns="beat_type", values="beat_cc")
        axis_vn = float(piv["V"].iloc[0] - piv["N"].iloc[0])
        panes.append(dict(tag=tag, name=name, x=x, out=out, lo=lo, hi=hi, cc=cc,
                          axis_vn=axis_vn))

    fig = card("⑤ 합성에서만 보이던 위험 — 실데이터에서는 없었다",
               "beat_cc(V) - beat_cc(N)  ·  EXP-E P3: 부정맥(PVC) 박동의 형태를 "
               "정상 박동만큼 지키는가", figsize=(12.5, 5.0))
    for k, pn in enumerate(panes):
        ax = fig.add_axes([0.06 + k * 0.47, 0.30, 0.38, 0.50])
        tt = np.arange(pn["hi"] - pn["lo"]) / FS * 1000
        ax.plot(tt, pn["x"][pn["lo"]:pn["hi"]], color=CLEAN, lw=4.0, label="참값 (PVC)")
        ax.plot(tt, pn["out"][pn["lo"]:pn["hi"]], color=C["M05"], lw=1.6, label="M05 출력")
        ax.set_xlabel("ms"); ax.set_yticks([])
        d = pn["cc"].get("V", float("nan")) - pn["cc"].get("N", float("nan"))
        ax.set_title(f"{pn['tag']} {'합성' if pn['tag']=='d0' else 'MIT-BIH'} "
                     f"· 기록 {pn['name']}", fontsize=10.5, loc="left", color=INK)
        ax.text(.98, .93, f"V - N = {d:+.3f}", transform=ax.transAxes, ha="right",
                fontsize=13, fontweight="bold",
                color=BAD if d < 0 else INK)
        ax.text(.98, .855, f"이 기록 · 축 평균 {pn['axis_vn']:+.3f}",
                transform=ax.transAxes, ha="right", fontsize=9, color=MUTE)
        if k == 0:
            ax.legend(frameon=False, loc="lower right", fontsize=9)
    punch(fig, "왼쪽(합성): M05 의 위상 템플릿이 PVC 를 «정상 쪽으로» 끌어당긴다 — 유일하게 음수다.\n"
               "오른쪽(실데이터): 같은 실험에서 부호가 뒤집힌다. 합성에서는 정상 박동이 모형에 정확히 맞아 "
               "PVC 만 튀지만, 실기록은 정상 박동도 모형을 벗어나 PVC 가 특별히 불리하지 않다.")
    return save(fig, "C5_pvc_damage")


# **다섯 장이다.** C6(하류 과제)은 C4 에 흡수했다 — 둘 다 «차이가 없다» 를
# 말해 중복이었고, 합치자 «못 재는 것 / 실제로 없는 것» 이라는 구분이 생겼다.
CARDS = {"C1": c1_gain_bias, "C2": c2_r_amp, "C3": c3_psd,
         "C4": c4_floor, "C5": c5_pvc}

# 카드마다 (지표, 무엇을 잡나, 근거). 문서는 **이 표에서 생성**한다 — 손으로
# 쓰면 카드와 문서가 갈라지고 그것을 알아챌 방법이 없다(F-9 계열).
INDEX = [
    ("C1", "gain_bias · strict/scaled", "진폭만 눌린 출력을 SNR 이 과하게 처벌한다",
     "합성 예시 (0.9 x). 보고서 3.2.2 의 단위 테스트와 같은 상황"),
    ("C2", "r_amp_err_pct", "«고주파 = 잡음» 가정이 QRS 부터 깎는다",
     "EXP-C(d0) — 잡음 0 입력"),
    ("C3", "psd_logdist", "무엇을 지웠는지는 스펙트럼에만 보인다",
     "EXP-C(d0) — 잡음 0 입력"),
    ("C4", "qrs_dur_err_ms + floor_p95 · hr_err_bpm",
     "«차이가 없다» 의 두 종류 — 지표가 못 재는 것과 실제로 없는 것",
     "EXP-C(d1) + results/d1/metric_floor + EXP-B(d1)"),
    ("C5", "beat_cc(V) − beat_cc(N)",
     "합성에서만 보이던 위험 — 실데이터에서는 재현되지 않았다 (F-8 · F-28)",
     "EXP-E P3(d0 · d1) — PVC 형태 보존"),
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
