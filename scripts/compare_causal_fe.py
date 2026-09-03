#!/usr/bin/env python3
"""인과 front-end 설계 비교 — **두 대가를 같은 단위로 잰다** (F-27 · D-19).

    python3 scripts/compare_causal_fe.py            # 표 + 그림
    python3 scripts/compare_causal_fe.py --axis d1

산출: `docs/12_causal_fe.md`, `results/fig/causal_fe.png`

왜 다시 재는가
-------------
D-19 를 처음 적을 때 두 대가를 **서로 다른 단위**로 적었다 — 기울기는
"R 진폭 대비 %/s", 기저선 잔여는 "PSD 파워 비". 0.0022 대 0.0096 을 보고
"5 배지만 둘 다 소수점 셋째 자리" 로 읽으면 **아무것도 결정할 수 없다.**
그래서 둘을 **같은 단위(R 진폭의 %)** 로 다시 잰다.

두 지표의 정의
-------------
**(1) 박동당 기저선 이동** `drift_per_beat`

    T-P 구간 = [R + 420 ms, 다음 R − 280 ms]      (T 파 뒤 ~ P 파 앞)
    그 구간에 1 차 다항식을 맞춰 기울기 k [mV/s] 를 얻는다
    박동당 이동 = |k| x RR                          [mV]
    지표        = 100 x 박동당 이동 / R 진폭        [%R]

    **"한 박동이 지나는 동안 기저선이 R 파 높이의 몇 % 만큼 흘렀는가."**
    박동마다 구해 중앙값을 쓴다. 평탄해야 할 자리이므로 이상적으로 0 이다.

**(2) 남은 기저선 변동** `wander_left`

    알려진 크기(R 의 50 %)의 기저선 변동을 **주입**하고 front-end 를 통과시킨 뒤,
    0.5 Hz 이하 성분의 p-p 를 잰다(측정용 저역통과는 영위상이다).
    지표 = 100 x p-p / R 진폭                       [%R]

    **"넣은 기저선 변동 중 얼마가 화면까지 살아남는가."**
    주입량이 50 %R 이므로 이 값이 50 에 가까우면 **거의 못 걸렀다**는 뜻이다.

**두 지표는 같은 방향으로 못 간다.** 고역통과를 세게 걸수록(차수↑ 또는
차단↑) 기저선은 잘 지워지지만 QRS 마다 남는 처짐이 커진다. 이 파일이
그 교환을 숫자와 그림으로 보인다.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sps

from ecgdn.config import DEFAULT_FE, FS
from ecgdn.data.mixer import mix_at_snr  # noqa: F401  (형태 유지)
from ecgdn.data.noise import make_noise
from ecgdn.data.sources import get_source
from ecgdn.eval.rpeak import detect_rpeaks
from ecgdn.methods.frontend import FrontEnd
from ecgdn.realtime.frontend_stream import StreamingFrontEnd
from ecgdn.utils import ensure_dir, rng, save_manifest

ROOT = Path(__file__).resolve().parents[1]

# (라벨, cfg). cfg=None 은 오프라인 영위상 — 비교의 기준선이다.
CFGS = [
    ("영위상 o4 0.5 Hz (모드 B 기준)", None),
    ("인과 o4 0.5 Hz", replace(DEFAULT_FE, order=4, hp_hz=0.5)),
    ("인과 o2 0.5 Hz", replace(DEFAULT_FE, order=2, hp_hz=0.5)),
    ("인과 o1 0.5 Hz", replace(DEFAULT_FE, order=1, hp_hz=0.5)),
    ("인과 o4 0.05 Hz", replace(DEFAULT_FE, order=4, hp_hz=0.05)),
    ("인과 o2 0.05 Hz", replace(DEFAULT_FE, order=2, hp_hz=0.05)),
    ("인과 o1 0.05 Hz", replace(DEFAULT_FE, order=1, hp_hz=0.05)),
]
PICK = "인과 o1 0.5 Hz"                 # 지금 쓰는 것 (D-19)
INJECT_FRAC = 0.5                       # 주입 기저선 변동 = R 의 50 %

def _ko_font() -> str:
    """한글 폰트 — 없으면 그림이 두부(□)로 나온다 (`make_slides.py` 와 같은 목록)."""
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    return next((f for f in ("NanumGothic", "NanumBarunGothic", "NanumSquare",
                             "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR")
                 if f in have), "DejaVu Sans")


plt.rcParams.update({"font.family": _ko_font(), "axes.unicode_minus": False,
                     "mathtext.fontset": "dejavusans"})

C = {"ref": "#b8b6ae", "o4": "#eb6834", "o1": "#2a78d6", "low": "#1baf7a",
     "ink": "#1b1b1b", "mute": "#6b6b6b"}


def causal(x, cfg, block=25):
    fe = StreamingFrontEnd(FS, cfg)
    return np.concatenate([fe.push(x[i:i + block]) for i in range(0, x.size, block)])


def apply(x, cfg):
    return np.asarray(FrontEnd()(x, FS)) if cfg is None else causal(x, cfg)


def drift_per_beat(v, r, r_amp):
    """한 박동 동안 기저선이 R 진폭의 몇 % 흐르는가. 정의는 모듈 docstring."""
    out = []
    for a, b in zip(r[:-1], r[1:]):
        i0, i1 = a + int(0.42 * FS), b - int(0.28 * FS)
        if i1 - i0 < 10:
            continue
        seg = v[i0:i1]
        k = np.polyfit(np.arange(seg.size) / FS, seg, 1)[0]      # mV/s
        out.append(abs(k) * (b - a) / FS)                        # mV / beat
    return 100 * float(np.median(out)) / r_amp if out else float("nan")


def wander_left(x, cfg, r_amp, inj_pp):
    """주입한 기저선 변동 중 얼마가 살아남는가 [%R]."""
    g = rng("fe", "bw")
    n = make_noise("bw_synth", x.size, FS, g)
    n = n / (np.percentile(n, 97.5) - np.percentile(n, 2.5)) * inj_pp
    v = apply(x + n, cfg)
    sos = sps.butter(4, 0.5 / (FS / 2), btype="lowpass", output="sos")
    low = sps.sosfiltfilt(sos, v)                                # 측정용은 영위상
    return 100 * float(np.percentile(low, 97.5) - np.percentile(low, 2.5)) / r_amp


def load(tag: str, seconds: float = 60.0):
    src = get_source("synthetic" if tag == "d0" else "mitdb")
    rec = src.get(src.records("test")[0])
    x = np.asarray(rec.x)[: int(seconds * FS)]
    r = np.asarray(detect_rpeaks(x, FS), dtype=int)
    return x, r, float(np.median(x[r]) - np.median(x)), str(src.records("test")[0])


def figure(tag: str, out: Path):
    """**숫자로는 감이 안 온다** — 같은 구간을 세 설정으로 통과시켜 나란히 본다."""
    x, r, r_amp, name = load(tag, 30.0)
    g = rng("fe", "bw")
    n = make_noise("bw_synth", x.size, FS, g)
    n = n / (np.percentile(n, 97.5) - np.percentile(n, 2.5)) * (INJECT_FRAC * r_amp)
    y = x + n
    picks = [("영위상 (모드 B)", None, C["ref"]),
             ("인과 o4 · 0.5 Hz", replace(DEFAULT_FE, order=4, hp_hz=0.5), C["o4"]),
             ("인과 o1 · 0.5 Hz  [채택]", replace(DEFAULT_FE, order=1, hp_hz=0.5), C["o1"]),
             ("인과 o1 · 0.05 Hz", replace(DEFAULT_FE, order=1, hp_hz=0.05), C["low"])]
    j = int(r[len(r) // 3])
    lo, hi = j - int(0.5 * FS), j + int(4.5 * FS)        # 박동 5~6 개

    fig, axes = plt.subplots(len(picks) + 1, 2, figsize=(13, 9.5), dpi=150,
                             gridspec_kw={"width_ratios": [1, 1]})
    fig.suptitle(f"인과 front-end 설계의 두 대가 — {tag} 기록 {name}",
                 x=0.02, ha="left", fontsize=14, fontweight="bold")
    axes[0, 0].set_title("① 기저선 변동을 안 넣었을 때 — T-P 가 평평한가",
                         fontsize=11, loc="left")
    axes[0, 1].set_title(f"② 기저선 변동을 R 의 {INJECT_FRAC*100:.0f} % 넣었을 때 — 얼마나 지우나",
                         fontsize=11, loc="left")
    t = np.arange(hi - lo) / FS
    for k, (lab, cfg, col) in enumerate([("입력 (필터 전)", "raw", "#8a8a8a")] + picks):
        for c, src_sig in ((0, x), (1, y)):
            ax = axes[k, c]
            v = src_sig if cfg == "raw" else apply(src_sig, cfg)
            ax.plot(t, v[lo:hi] - np.median(v[lo:hi]), color=col, lw=1.1)
            ax.set_ylim(-1.1 * r_amp, 1.5 * r_amp)
            ax.set_yticks([]); ax.set_xticks([] if k < len(picks) else [0, 1, 2, 3, 4])
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.axhline(0, color="#dddddd", lw=0.8, zorder=0)
        axes[k, 0].set_ylabel(lab, rotation=0, ha="right", va="center",
                              fontsize=10, labelpad=8)
        if cfg != "raw":
            d = drift_per_beat(apply(x, cfg), r, r_amp)
            w = wander_left(x, cfg, r_amp, INJECT_FRAC * r_amp)
            axes[k, 0].text(.985, .90, f"박동당 이동 {d:.0f} %R", transform=axes[k, 0].transAxes,
                            ha="right", fontsize=9.5, color=C["mute"])
            axes[k, 1].text(.985, .90, f"남은 변동 {w:.0f} %R", transform=axes[k, 1].transAxes,
                            ha="right", fontsize=9.5, color=C["mute"])
    axes[-1, 0].set_xlabel("s"); axes[-1, 1].set_xlabel("s")
    fig.text(0.02, 0.015,
             "왼쪽 열이 0 에 가까울수록 «T-P 가 평평하다», 오른쪽 열이 0 에 가까울수록 "
             "«기저선 변동을 잘 지웠다». 둘은 같은 방향으로 못 간다.",
             fontsize=10, color=C["ink"])
    fig.tight_layout(rect=[0.02, 0.03, 1, 0.95])
    ensure_dir(out.parent); fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", nargs="*", default=["d0", "d1"])
    a = ap.parse_args()

    L = ["# 12. 인과 front-end 설계 — **두 대가를 같은 단위로**", "",
         "> 이 문서는 `scripts/compare_causal_fe.py` 가 만든다. **직접 고치지 말 것.**",
         "> 경위는 **F-27**, 결정은 **D-19**. 그림은 `results/fig/causal_fe.png`.", "",
         "영위상 필터는 앞뒤로 두 번 걸어 비대칭 처짐이 상쇄되지만 **인과 필터는",
         "그럴 수 없다.** 고역통과를 세게 걸수록(차수↑ 또는 차단↑) 기저선은 잘",
         "지워지지만 **QRS 마다 남는 처짐이 커진다.** 두 대가를 같은 단위로 잰다.", "",
         "## 두 지표의 정의", "",
         "**① 박동당 기저선 이동** — «한 박동이 지나는 동안 기저선이 R 파 높이의 몇 % 흘렀나»", "",
         "```",
         "T-P 구간 = [R + 420 ms, 다음 R − 280 ms]     (T 파 뒤 ~ P 파 앞)",
         "그 구간에 1 차 다항식을 맞춰 기울기 k [mV/s]",
         "지표 = 100 × |k| × RR / R진폭   [%R]        박동마다 구해 **중앙값**",
         "```", "",
         "**② 남은 기저선 변동** — «넣은 기저선 변동 중 얼마가 화면까지 살아남나»", "",
         "```",
         f"기저선 변동을 R 진폭의 {INJECT_FRAC*100:.0f} % 크기(p-p)로 **주입**한 뒤 통과",
         "지표 = 100 × (출력의 0.5 Hz 이하 성분 p-p) / R진폭   [%R]",
         f"      -> {INJECT_FRAC*100:.0f} 에 가까우면 **거의 못 걸렀다**는 뜻",
         "```", ""]

    figs = []
    for tag in a.axis:
        x, r, r_amp, name = load(tag)
        L += [f"## {tag} (기록 {name} · R 진폭 {r_amp:.3f} mV)", "",
              "| 설정 | ① 박동당 이동 | ② 남은 기저선 변동 |", "|---|---|---|",
              f"| (원본 · 필터 전) | {drift_per_beat(x, r, r_amp):.1f} %R | — |"]
        for lab, cfg in CFGS:
            d = drift_per_beat(apply(x, cfg), r, r_amp)
            w = wander_left(x, cfg, r_amp, INJECT_FRAC * r_amp)
            mark = " **← 채택 (D-19)**" if lab == PICK else ""
            L.append(f"| {lab}{mark} | {d:.1f} %R | {w:.1f} %R |")
        L.append("")
        figs.append(figure(tag, ROOT / "results" / "fig" / f"causal_fe_{tag}.png"))

    L += ["## 읽는 법", "",
          "- **0.05 Hz 는 기저선 변동을 거의 못 지운다.** 주입량이 50 %R 인데 남는 것이",
          "  45~60 %R 다 — «5 배 남는다» 가 아니라 **«거의 그대로 통과»** 다.",
          "  처음에 이것을 PSD 파워 비(0.0022 vs 0.0096)로 적어 «둘 다 소수점 셋째",
          "  자리» 로 보이게 만들었다. **단위를 잘못 고르면 결정을 못 한다.**",
          "- **0.05 Hz 에서는 차수가 거의 무의미하다**(60.5 / 60.0 / 59.9 %R) — 차단이",
          "  기저선 대역보다 한참 아래라 차수를 올려도 그 대역에 도달하지 못한다.",
          "- **0.5 Hz 에서는 차수가 둘을 정면으로 맞바꾼다.** o4 는 이동 66 %R / 잔여",
          "  9 %R, o1 은 이동 23 %R / 잔여 22 %R 이다.",
          "- **실기록(D1)은 원본부터 이미 16 %R 흐른다.** 필터가 만드는 몫은 그 위에",
          "  얹히는 것이고, 그래서 D1 에서는 설정을 바꿔도 화면 인상이 D0 만큼 크게",
          "  달라지지 않는다.", ""]
    p = ROOT / "docs" / "12_causal_fe.md"
    p.write_text("\n".join(L))
    save_manifest(ROOT / "results" / "fig", cfg={"inject_frac": INJECT_FRAC,
                  "cfgs": [c[0] for c in CFGS]}, sources=[__file__])
    print(f"-> {p.relative_to(ROOT)}")
    for f in figs:
        print(f"-> {f.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
