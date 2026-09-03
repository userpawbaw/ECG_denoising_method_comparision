#!/usr/bin/env python3
"""중앙값 캐스케이드 vs 블록 영위상 — **무엇이 다른가** (F-29 후속).

    python3 scripts/compare_median_vs_zerophase.py

산출: `docs/14_median_vs_zerophase.md`,
      `results/fig/median_{subtract,hr,clean}.png`

왜 이 문서가 따로 있는가
----------------------
`docs/13_lookahead_fe.md` 의 표에서 **중앙값 200+600 ms** 가 T-P 를 가장 평평하게
만든다. 화면으로 보면 «원본이 중심선을 벗어나 있는데 통과시킨 파형은 중심선에
딱 붙는» 것처럼 보이고, **그 인상이 맞다.** 두 방식은 하는 일의 **종류**가 다르다:

    블록 영위상   선형 필터다. «0.5 Hz 아래를 지운다» 를 앞뒤로 두 번 해서
                  위상을 상쇄한다. 지우는 것은 **주파수로 정의된 대역**이고,
                  그 대역 밖의 기저선 변동은 **그대로 남는다.**

    중앙값        필터가 아니다. 매 표본에서 «여기 국소 준위가 얼마인가» 를
                  **비선형으로 추정**해 빼기만 한다. 추정 대상에 주파수 제약이
                  없으므로 **아무 모양의 느린 곡선이든 따라가서 지운다.**

그래서 중앙값이 더 평평하다. **공짜가 아니고, 대가가 심박수에 걸려 있다** —
그것을 재는 것이 이 파일의 목적이다.

왜 QRS 는 안 상하는가
--------------------
창 200 ms 안에서 QRS(약 80 ms)는 **절반이 안 된다.** 중앙값은 정의상 표본의
과반이 결정하므로 QRS 는 중앙값이 될 수 없다 — 추정된 기저선에 QRS 가 안 들어가고,
그래서 빼도 QRS 가 안 깎인다. **이것이 중앙값을 쓰는 이유 전부다.**

그리고 그 논리가 **깨지는 지점**이 곧 위험 지점이다: 창 안에서 어떤 파형이
과반을 차지하면 그것은 기저선으로 오인되어 깎인다. `100+300 ms` 판이 T 를
16 %R 깎은 것(F-29)이 이 현상이다.

**여기서 예측을 틀렸다.** 이 파일을 짜면서 "창 600 ms 에 T 파가 들어앉으려면
RR 이 짧아야 하니 **심박수가 올라가면** T 파가 위험하다" 고 적었는데,
재보니 **정반대**였다 `[측정]` — `100+300` 의 T 오차는 50 bpm 12.5 %R 에서
140 bpm 0.0 %R 로 **심박이 빠를수록 좋아진다.**

옳은 기제는 이것이다: **창은 「보존하려는 가장 넓은 파형」보다 넓어야 한다.**
T 파는 안정 시 약 200 ms 다. 창 300 ms 안에서는 T 가 과반을 차지할 수 있어
기저선으로 오인된다. 창 600 ms 는 T 보다 충분히 넓어 안전하다. 그리고
**심박이 빠르면 T 가 짧아지고(QT 단축) 창 하나에 여러 박동이 들어와** T 가
과반이 될 수 없다 — 그래서 빠를수록 안전해진다. **RR 이 아니라 파형 폭과
창 길이의 비**가 결정한다.
"""
from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np

from ecgdn.config import FS
from ecgdn.data.noise import make_noise
from ecgdn.data.synthetic import synth_ecg
from ecgdn.eval.rpeak import detect_rpeaks
from ecgdn.utils import ensure_dir, rng, save_manifest
from explore_lookahead_fe import (C, INJECT_FRAC, _ko_font, f_lookahead, f_median,
                                  f_offline, load, s_depth, t_amp, tp_noise, tp_spread)

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.family": _ko_font(), "axes.unicode_minus": False,
                     "mathtext.fontset": "dejavusans"})

MED = lambda x: f_median(x, 0.2, 0.6)        # noqa: E731
MED_S = lambda x: f_median(x, 0.1, 0.3)      # noqa: E731
LA = lambda x: f_lookahead(x, 0.5)           # noqa: E731

HR_GRID = (50, 60, 70, 85, 100, 120, 140)


def _wander(x, r_amp):
    g = rng("fe", "bw")
    n = make_noise("bw_synth", x.size, FS, g)
    return n / (np.percentile(n, 97.5) - np.percentile(n, 2.5)) * (INJECT_FRAC * r_amp)


# ============================================================ ① 무엇을 빼는가
def fig_subtract(tag: str, out: Path):
    """**두 방식이 «제거한 것» 을 직접 그린다.** 이것이 차이의 전부다."""
    x, r, r_amp, name = load(tag, 30.0)
    y = x + _wander(x, r_amp)
    j = int(r[len(r) // 3])
    lo, hi = j - int(0.5 * FS), j + int(7.5 * FS)
    t = np.arange(hi - lo) / FS

    v_med, v_la = MED(y), LA(y)
    # «제거한 것» = 입력 − 출력. 중앙값은 이것이 곧 기저선 추정치다.
    rm_med, rm_la = y - v_med, y - v_la

    fig, axes = plt.subplots(3, 1, figsize=(13, 8.6), dpi=150, sharex=True)
    fig.suptitle(f"두 방식이 «무엇을 빼는가» — {tag} 기록 {name}",
                 x=0.02, ha="left", fontsize=14, fontweight="bold")

    for ax, rm, col, lab in ((axes[0], rm_med, C["med"], "중앙값 200+600 ms 가 뺀 것"),
                             (axes[1], rm_la, C["look"], "블록 영위상 0.5 s 가 뺀 것")):
        ax.axhline(0, color="#dcdcdc", lw=0.9, zorder=0)
        ax.plot(t, y[lo:hi] - np.median(y[lo:hi]), color="#c9c7c0", lw=1.0,
                label="입력 (기저선 변동 주입)", zorder=1)
        ax.plot(t, rm[lo:hi] - np.median(y[lo:hi]), color=col, lw=2.6,
                label=lab, zorder=3)
        ax.legend(frameon=False, loc="upper right", fontsize=9.5, ncol=2)
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)

    axes[0].set_title("① 중앙값은 「기저선 자체」를 추정한다 — 굵은 선이 입력의 굽이를 그대로 탄다",
                      fontsize=11, loc="left")
    axes[1].set_title("② 블록 영위상은 「0.5 Hz 아래」만 뺀다 — 굵은 선이 훨씬 밋밋하다 "
                      "(그 대역 밖의 변동은 출력에 남는다)", fontsize=11, loc="left")

    ax = axes[2]
    ax.axhline(0, color="#dcdcdc", lw=0.9, zorder=0)
    ax.plot(t, v_la[lo:hi] - np.median(v_la[lo:hi]), color=C["look"], lw=1.3,
            label="블록 영위상 0.5 s")
    ax.plot(t, v_med[lo:hi] - np.median(v_med[lo:hi]), color=C["med"], lw=1.3,
            label="중앙값 200+600 ms")
    ax.legend(frameon=False, loc="upper right", fontsize=9.5, ncol=2)
    ax.set_title("③ 그래서 출력이 이렇게 다르다 — 초록이 중심선에 더 붙는다",
                 fontsize=11, loc="left")
    ax.set_yticks([]); ax.set_xlabel("s")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    fig.text(0.02, 0.015,
             "핵심: 중앙값은 주파수로 정의된 대역이 아니라 «국소 준위» 를 뺀다. "
             "그래서 아무 모양의 느린 굽이든 따라가 지운다 — 그것이 더 평평한 이유이자, 대가의 출처다.",
             fontsize=10, color=C["ink"])
    fig.tight_layout(rect=[0.02, 0.035, 1, 0.945])
    ensure_dir(out.parent); fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ====================================================== ② 심박수 의존성 (핵심)
def measure_hr():
    """창 길이와 파형 폭의 비가 T 파의 안전을 정한다 — 심박수로 그 비를 훑는다.

    합성축으로만 잰다 — 실기록은 심박수를 우리가 못 정한다.
    """
    rows = []
    for hr in HR_GRID:
        s = synth_ecg(duration_s=60.0, fs=FS, hr_bpm=float(hr), seed=7)
        x = np.asarray(s.x, dtype=np.float64)
        r = np.asarray(detect_rpeaks(x, FS), dtype=int)
        r_amp = float(np.median(x[r]) - np.median(x))
        ref = f_offline(x)
        t_ref, s_ref = t_amp(ref, r, r_amp), s_depth(ref, r, r_amp)
        row = dict(hr=hr, rr_ms=60000.0 / hr)
        for key, fn in (("med", MED), ("med_s", MED_S), ("la", LA)):
            v = fn(x)
            row[f"t_{key}"] = abs(t_amp(v, r, r_amp) - t_ref)
            row[f"s_{key}"] = abs(s_depth(v, r, r_amp) - s_ref)
            row[f"b_{key}"] = tp_spread(v, r, r_amp)
        rows.append(row)
    return rows


def fig_hr(rows, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)
    fig.suptitle("심박수를 올리면 무엇이 먼저 무너지는가 — 합성축, 잡음 없음",
                 x=0.02, ha="left", fontsize=13, fontweight="bold")
    hr = [r["hr"] for r in rows]
    series = (("med", "중앙값 200+600 ms", C["med"], "-o"),
              ("med_s", "중앙값 100+300 ms", "#7fcfae", "--s"),
              ("la", "블록 영위상 0.5 s", C["look"], "-o"))
    for ax, key, ttl in ((axes[0], "t", "T 진폭 오차 [%R]  (T 파를 깎는가)"),
                         (axes[1], "b", "T-P 준위 산포 [%R]  (박동마다 같은 높이인가)")):
        for k, lab, col, st in series:
            ax.plot(hr, [r[f"{key}_{k}"] for r in rows], st, color=col, label=lab,
                    lw=1.8, ms=5)
        ax.set_xlabel("심박수 [bpm]"); ax.set_title(ttl, fontsize=11, loc="left")
        ax.grid(alpha=.25, lw=.6); ax.legend(frameon=False, fontsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].axhline(5, color="#cccccc", lw=1.0, ls=":")
    axes[0].text(HR_GRID[0], 5.4, "5 %R", fontsize=8.5, color=C["mute"])
    fig.text(0.02, 0.02,
             "창은 「보존하려는 가장 넓은 파형」보다 넓어야 한다 — T 파(안정 시 약 200 ms)가 "
             "창 300 ms 안에서 과반이 되면 기저선으로 오인돼 깎인다. 그래서 짧은 창은 "
             "«느린 심박에서» 무너진다. 블록 영위상은 심박수와 무관하다(선형 필터라 박동 간격을 안 본다).",
             fontsize=9.5, color=C["ink"])
    fig.tight_layout(rect=[0.02, 0.05, 1, 0.92])
    ensure_dir(out.parent); fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# =================================================== ③ 비선형의 값 — 중첩이 깨진다
def measure_superposition(tag: str, snr_db: float = 5.0):
    """**선형 필터는 `FE(신호+잡음) = FE(신호) + FE(잡음)` 이다. 중앙값은 아니다.**

    이것이 왜 이 과제에 중요한가: 우리 평가는 참조를 `FE(clean)` 로 두고
    `출력 − 참조` 를 오차로 센다(D-3). 그 정의는 **front-end 가 신호와 잡음을
    같은 방식으로 다룬다** 는 가정 위에 있다. 비선형이면 **잡음 수준이
    신호가 받는 처리를 바꾸므로**, 참조 자체가 입력 SNR 에 따라 흔들린다.

    재는 법: `FE(x+n) − FE(x)` 와 `FE(n)` 을 비교한다. 선형이면 정확히 같다.
    남는 차이를 신호 크기 대비 dB 로 적는다 — **클수록 비선형이 세다.**
    """
    from ecgdn.data.mixer import mix_at_snr
    x, r, r_amp, name = load(tag, 30.0)
    g = rng("fe", "super")
    n = make_noise("mixed" if False else "bw_synth", x.size, FS, g)
    y, _, _ = mix_at_snr(x, n, snr_db)
    n_only = y - x
    out = []
    for key, lab, fn in (("la", "블록 영위상 0.5 s", LA),
                         ("med", "중앙값 200+600 ms", MED)):
        d = (fn(y) - fn(x)) - fn(n_only)
        # 신호 자체의 크기 대비로 적는다 (dB). 선형이면 -inf 여야 한다.
        num = float(np.sum(fn(x) ** 2))
        den = float(np.sum(d ** 2))
        out.append(dict(key=key, label=lab,
                        err_db=10 * np.log10(den / num) if den > 0 else -np.inf,
                        err_pct_r=100 * float(np.percentile(np.abs(d), 99)) / r_amp))
    return out, name


def measure_clean(tag: str):
    """잡음 0 입력에 해를 끼치는가 — EXP-C 와 같은 질문, 참조는 영위상."""
    x, r, r_amp, name = load(tag, 30.0)
    ref = f_offline(x)
    t_ref, s_ref = t_amp(ref, r, r_amp), s_depth(ref, r, r_amp)
    out = []
    for key, lab, fn in (("la", "블록 영위상 0.5 s", LA),
                         ("med", "중앙값 200+600 ms", MED),
                         ("med_s", "중앙값 100+300 ms", MED_S)):
        v = fn(x)
        a, b = ref - ref.mean(), v - v.mean()
        out.append(dict(key=key, label=lab,
                        floor_db=10 * np.log10((a @ a) / ((b - a) @ (b - a))),
                        t_err=abs(t_amp(v, r, r_amp) - t_ref),
                        s_err=abs(s_depth(v, r, r_amp) - s_ref),
                        band=tp_spread(v, r, r_amp),
                        noise=tp_noise(v, r, r_amp)))
    return out, name


def fig_clean(per_axis, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=150)
    fig.suptitle("잡음 0 입력에 해를 끼치는가 — 참조는 오프라인 영위상",
                 x=0.02, ha="left", fontsize=13, fontweight="bold")
    labs = [r["label"] for r in per_axis["d0"][0]]
    cols = [C["look"], C["med"], "#7fcfae"]
    w, idx = 0.36, np.arange(len(labs))
    for ax, key, ttl in ((axes[0], "floor_db",
                          "왜곡 하한 [dB]  (높을수록 영위상과 같다)"),
                         (axes[1], "band", "T-P 준위 산포 [%R]  (낮을수록 평평)")):
        for k, tag in enumerate(("d0", "d1")):
            v = [r[key] for r in per_axis[tag][0]]
            ax.bar(idx + (k - 0.5) * w, v, w, color=cols,
                   alpha=1.0 if k == 0 else 0.55,
                   label="d0 합성" if k == 0 else "d1 MIT-BIH")
        ax.set_xticks(idx); ax.set_xticklabels([l.replace(" ", "\n", 1) for l in labs],
                                               fontsize=8.5)
        ax.set_title(ttl, fontsize=11, loc="left"); ax.grid(alpha=.2, axis="y", lw=.6)
        ax.legend(frameon=False, fontsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.9])
    ensure_dir(out.parent); fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ------------------------------------------------------------------ 문서
def write_doc(hr_rows, clean, supers, made) -> Path:
    L = ["# 14. 중앙값 캐스케이드 vs 블록 영위상 — 무엇이 다른가", "",
         "> 이 문서는 `scripts/compare_median_vs_zerophase.py` 가 만든다. **직접 고치지 말 것.**",
         "> 후보 전체 표는 `docs/13_lookahead_fe.md`.",
         "",
         "## 한 줄", "",
         "**블록 영위상은 «어떤 주파수 대역» 을 빼고, 중앙값은 «국소 준위가 얼마인가» 를 빼다.**",
         "전자는 선형 필터이고 후자는 비선형 추정기다. 그래서 중앙값이 더 평평하고,",
         "그 능력의 대가가 **심박수**와 **중첩**에 걸려 있다.", "",
         "## 중앙값 캐스케이드는 무엇을 하는가", "",
         "```",
         "기저선 추정 = 이동중앙값( 이동중앙값( x, 창 200 ms ), 창 600 ms )",
         "출력        = x − 기저선 추정",
         "```", "",
         "「이동중앙값(창 w)」은 각 표본에서 **주변 w 안의 표본을 크기순으로 줄 세워 가운데 값**을",
         "고르는 것이다. 평균과 달리 **큰 값 몇 개가 결과를 못 끌어당긴다.**", "",
         "**왜 QRS 가 안 깎이나.** 창 200 ms 안에서 QRS(약 80 ms)는 절반이 안 된다.",
         "중앙값은 정의상 **과반이 결정**하므로 QRS 는 중앙값이 될 수 없다 —",
         "추정된 기저선에 QRS 가 안 들어가고, 그래서 빼도 QRS 가 안 상한다.", "",
         "**왜 더 평평한가.** 고역통과는 «0.5 Hz 아래» 라는 **주파수로 정의된 대역**만",
         "지우므로 그 대역 밖의 기저선 변동은 출력에 남는다. 중앙값이 추정하는 것은",
         "대역이 아니라 **국소 준위**여서 **아무 모양의 느린 굽이든 따라가 지운다.**",
         "화면에서 «원본은 중심선을 벗어났는데 통과시킨 것은 딱 붙는» 인상이 이것이다.", "",
         "**그리고 같은 논리가 위험을 정한다.** 창 안에서 **어떤 파형이 과반을 차지하면**",
         "그것은 기저선으로 오인되어 깎인다. 그러므로 규칙은 하나다 —",
         "**창은 「보존하려는 가장 넓은 파형」보다 넓어야 한다.** T 파는 안정 시 약 200 ms 이므로",
         "창 300 ms 는 위험하고 600 ms 는 안전하다.", "",
         "## 대가 (1) — 심박수 `[측정]`", "",
         "> **여기서 예측을 틀렸다.** 처음에는 \"창 600 ms 에 T 파가 들어앉으려면 RR 이",
         "> 짧아야 하니 **심박수가 올라가면** 위험하다\" 고 적었다. 재보니 **정반대**다 —",
         "> `100+300` 의 T 오차가 50 bpm 12.5 %R 에서 140 bpm 0.0 %R 로 **빠를수록 좋아진다.**",
         "> 심박이 빠르면 T 파가 짧아지고(QT 단축) 창 하나에 여러 박동이 들어와 T 가 과반이",
         "> 될 수 없기 때문이다. **RR 이 아니라 「파형 폭 : 창 길이」 비**가 결정한다.", "",
         "합성축, 잡음 없음. 영위상 대비 오차다.", "",
         "| 심박수 | RR | 중앙값 200+600 T 오차 | 중앙값 100+300 T 오차 | 블록 영위상 T 오차 |",
         "|---:|---:|---:|---:|---:|"]
    for r in hr_rows:
        L.append(f"| {r['hr']} bpm | {r['rr_ms']:.0f} ms | **{r['t_med']:.1f} %R** | "
                 f"{r['t_med_s']:.1f} %R | {r['t_la']:.1f} %R |")
    L += ["",
          "**블록 영위상은 심박수와 무관하다** — 선형 필터라 박동 간격을 안 본다.",
          "**중앙값 200+600 도 이 범위 전체에서 안전하다**(T 오차 ≤ 1.1 %R).",
          "위험한 것은 창이 짧을 때이고, 그것도 **느린 심박에서** 나타난다.", "",
          "## 대가 (2) — 중첩이 깨진다 `[측정]`", "",
          "선형 필터는 `FE(신호+잡음) = FE(신호) + FE(잡음)` 이다. 중앙값은 아니다.",
          "우리 평가는 참조를 `FE(clean)` 으로 두고 `출력 − 참조` 를 오차로 센다(D-3).",
          "그 정의는 **front-end 가 신호와 잡음을 같은 방식으로 다룬다** 는 가정 위에 있다.", "",
          "`FE(x+n) − FE(x)` 와 `FE(n)` 의 차 (선형이면 0):", "",
          "| 축 | 방식 | 남은 차 [dB] | p99 [%R] |", "|---|---|---:|---:|"]
    for tag, (rows, name) in supers.items():
        for r in rows:
            db = "−∞ (정확히 선형)" if not np.isfinite(r["err_db"]) else f"{r['err_db']:.1f}"
            L.append(f"| {tag} | {r['label']} | {db} | {r['err_pct_r']:.2f} |")
    L += ["",
          "## 대가 (3) — 잡음 0 입력 `[측정]`", "", "| 축 | 방식 | 왜곡 하한 [dB] | T 오차 | S 오차 | 준위 산포 | *구간내 잡음* |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for tag, (rows, name) in clean.items():
        for r in rows:
            L.append(f"| {tag} | {r['label']} | **{r['floor_db']:.1f}** | {r['t_err']:.1f} %R | "
                     f"{r['s_err']:.1f} %R | **{r['band']:.1f} %R** | *{r['noise']:.1f} %R* |")
    L += ["", "## 그림", ""]
    L += [f"![{p.stem}](../{p.relative_to(ROOT)})" for p in made] + [""]
    p = ROOT / "docs" / "14_median_vs_zerophase.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p


def main() -> int:
    made = [fig_subtract("d1", ROOT / "results" / "fig" / "median_subtract.png")]
    print("심박수 훑는 중 …")
    hr_rows = measure_hr()
    for r in hr_rows:
        print(f"  {r['hr']:3d} bpm (RR {r['rr_ms']:.0f} ms)  T오차 중앙값 {r['t_med']:5.1f}"
              f" / 짧은창 {r['t_med_s']:5.1f} / 블록영위상 {r['t_la']:5.1f} %R")
    made.append(fig_hr(hr_rows, ROOT / "results" / "fig" / "median_hr.png"))
    clean = {t: measure_clean(t) for t in ("d0", "d1")}
    supers = {t: measure_superposition(t) for t in ("d0", "d1")}
    for t, (rows, _) in supers.items():
        for r in rows:
            print(f"  [{t}] 중첩 잔차 {r['label']:<18} {r['err_db']:6.1f} dB")
    made.append(fig_clean(clean, ROOT / "results" / "fig" / "median_clean.png"))
    doc = write_doc(hr_rows, clean, supers, made)
    save_manifest(ROOT / "results" / "fig",
                  cfg={"script": "compare_median_vs_zerophase", "hr_grid": list(HR_GRID)},
                  sources=[__file__])
    print(f"-> {doc.relative_to(ROOT)}")
    for p in made:
        print(f"-> {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
