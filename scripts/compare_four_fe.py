#!/usr/bin/env python3
"""네 방식만 나란히 — **재중심화 없이** 본다.

    python3 scripts/compare_four_fe.py

산출: `results/fig/four_fe_{d0,d1}.png`

왜 따로 만드는가
---------------
`docs/13` 의 그림은 후보를 여럿 올려 한 장이 빽빽하고, 무엇보다 **패널마다
그 창의 중앙값을 빼서** 그렸다. 그러면 «평활 구간이 중심선에 붙어 있는가» 는
**정의상 안 보인다** — 빼는 순간 모든 패널이 0 근처로 옮겨지기 때문이다.

그래서 여기서는:

  * 후보를 **넷**으로 줄인다 (입력 · 오프라인 영위상 · 블록 영위상 · 중앙값)
  * front-end 출력은 **손대지 않고 그대로** 그린다. 0 선이 진짜 0 이다
  * 입력만은 예외로 중앙값을 뺀다 — ADC 오프셋이 화면 밖으로 나가기 때문이고,
    **그 사실을 패널에 적는다**

같은 기록·같은 구간·같은 세로 배율을 네 행이 공유한다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np

from ecgdn.config import FS
from ecgdn.data.noise import make_noise
from ecgdn.utils import ensure_dir, rng, save_manifest
from explore_lookahead_fe import (INJECT_FRAC, _ko_font, drift_per_beat, f_lookahead,
                                  f_median, f_offline, load, s_depth, tp_off,
                                  tp_spread, wander_left)

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.family": _ko_font(), "axes.unicode_minus": False,
                     "mathtext.fontset": "dejavusans"})

INK, MUTE = "#1b1b1b", "#6b6b6b"
ROWS = [
    ("입력 (front-end 전)", None, "#8a8a8a"),
    ("오프라인 영위상 (도달 못 하는 기준)", f_offline, "#b8b6ae"),
    ("블록 영위상 · 미리보기 0.5 s", lambda x: f_lookahead(x, 0.5), "#2a78d6"),
    ("중앙값 200+600 ms", lambda x: f_median(x, 0.2, 0.6), "#1baf7a"),
]


def figure(tag: str, out: Path, seconds: float = 30.0, show_s: float = 6.0):
    x, r, r_amp, name = load(tag, seconds)
    g = rng("fe", "bw")
    n = make_noise("bw_synth", x.size, FS, g)
    n = n / (np.percentile(n, 97.5) - np.percentile(n, 2.5)) * (INJECT_FRAC * r_amp)
    y = x + n

    j = int(r[len(r) // 3])
    lo = max(0, j - int(0.5 * FS))
    hi = min(x.size, lo + int(show_s * FS))
    t = np.arange(hi - lo) / FS

    fig, axes = plt.subplots(len(ROWS), 2, figsize=(14, 9.0), dpi=150,
                             sharex=True, sharey=True)
    axis_ko = "합성 (D0)" if tag == "d0" else "MIT-BIH (D1)"
    fig.suptitle(f"네 방식만 — 재중심화 없이 본다   ·   {axis_ko} 기록 {name}",
                 x=0.02, ha="left", fontsize=14, fontweight="bold")
    axes[0, 0].set_title("원본 입력", fontsize=11.5, loc="left")
    axes[0, 1].set_title(f"기저선 변동 R 의 {INJECT_FRAC*100:.0f} % 주입",
                         fontsize=11.5, loc="left")

    for k, (lab, fn, col) in enumerate(ROWS):
        for c, src in ((0, x), (1, y)):
            ax = axes[k, c]
            if fn is None:
                # 입력만 중앙값을 뺀다 — ADC 오프셋이 화면 밖이라. **적어 둔다.**
                v = src - np.median(src)
            else:
                v = fn(src)                      # **그대로.** 0 선이 진짜 0 이다
            ax.axhline(0, color="#c8c8c8", lw=1.0, zorder=1)
            ax.plot(t, v[lo:hi], color=col, lw=1.15, zorder=2)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.set_yticks([])
            if fn is not None:
                ax.text(.995, .06,
                        f"준위 산포 {tp_spread(v, r, r_amp):.1f} · "
                        f"이탈 {tp_off(v, r, r_amp):.1f} %R",
                        transform=ax.transAxes, ha="right", fontsize=9, color=MUTE)
        axes[k, 0].set_ylabel(lab, rotation=0, ha="right", va="center",
                              fontsize=10.5, labelpad=10)
        if fn is None:
            axes[k, 0].text(.008, .84, "※ 이 행만 중앙값을 뺐다 (ADC 오프셋)",
                            transform=axes[k, 0].transAxes, fontsize=9, color="#c05010")
        else:
            axes[k, 1].text(.008, .84,
                            f"남은 변동 {wander_left(fn, x, r_amp, INJECT_FRAC*r_amp):.0f} %R",
                            transform=axes[k, 1].transAxes, fontsize=9, color=MUTE)
            axes[k, 0].text(.008, .84,
                            f"박동당 이동 {drift_per_beat(fn(x), r, r_amp):.0f} %R · "
                            f"S 깊이 {s_depth(fn(x), r, r_amp):.0f} %R",
                            transform=axes[k, 0].transAxes, fontsize=9, color=MUTE)
    axes[0, 0].set_ylim(-1.0 * r_amp, 1.5 * r_amp)
    axes[-1, 0].set_xlabel("s"); axes[-1, 1].set_xlabel("s")
    fig.text(0.02, 0.015,
             "회색 가로선이 «진짜 0» 이다. front-end 출력은 손대지 않았으므로, "
             "평활 구간이 그 선에 붙는지가 그대로 보인다.",
             fontsize=10, color=INK)
    fig.tight_layout(rect=[0.02, 0.035, 1, 0.945])
    ensure_dir(out.parent); fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", nargs="*", default=["d1", "d0"])
    a = ap.parse_args()
    made = [figure(t, ROOT / "results" / "fig" / f"four_fe_{t}.png") for t in a.axis]
    save_manifest(ROOT / "results" / "fig", cfg={"script": "compare_four_fe",
                                                 "axes": a.axis},
                  sources=[__file__])
    for p in made:
        print(f"-> {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
