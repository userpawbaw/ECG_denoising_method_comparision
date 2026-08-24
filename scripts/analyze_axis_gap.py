#!/usr/bin/env python3
"""D0/D1 대조 분석 — 합성 벤치마크가 실데이터를 얼마나 예측하는가.

    python scripts/analyze_axis_gap.py

산출: docs/92_axis_gap.md

**왜 별도 스크립트인가.** `90_results_{d0,d1}.md` 는 각 축을 따로 보고한다.
두 축을 나란히 놓는 순간 다른 질문이 생기고(어떤 결론이 옮겨가고 어떤 것이
옮겨가지 않는가), 그 질문에는 축을 가로지르는 검정이 필요하다.

이 스크립트가 지키는 규칙 세 가지 — 셋 다 P-8 해석 중에 실제로 틀릴 뻔했다.

1. **전 구간 평균과 구간별 검정을 함께 낸다.** `snr_imp_scaled` 의 전 구간
   평균은 D1 에서 유의하지 않은데, 구간별로 보면 저 SNR 에서는 크게 유의하고
   고 SNR 에서 상쇄된다. 평균만 보면 "딥러닝이 실데이터에서 안 된다" 로 읽힌다.

2. **기준선을 둘 다 쓴다.** `M01`(bandpass 0.5–40 Hz)은 참조 대역(0.5–100 Hz)과
   **대역이 다르다.** 그래서 `psd_logdist` 에서 딥러닝이 `M01` 을 18 dB 이기는데,
   그것은 잡음 제거 성능이 아니라 40 Hz 절단의 결과다. 같은 대역인 `M_FE` 와
   비교하면 두 축 모두 유의하지 않다.

3. **오차 크기 지표(rmse/prdn)의 우세를 SNR 과 독립된 증거로 쓰지 않는다.**
   선형 오차는 저 SNR 조건이 압도적으로 무겁게 실린다(D1 −5 dB 에서 0.138 mV,
   20 dB 에서 0.006 mV — 23 배). 구간별로 보면 SNR 지표와 완전히 일치한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecgdn.eval.stats import compare_methods  # noqa: E402

DL = ("M06", "M08")
BASELINES = ("M01", "M_FE")
# 방향: 낮을수록 좋은 지표
LOWER = {"rmse", "prdn", "psd_logdist", "r_amp_err_pct", "hr_err_bpm",
         "qrs_dur_err_ms", "rpeak_mae_ms"}
# 1 에 가까울수록 좋은 지표
TO_ONE = {"gain_bias"}
METRICS = ["snr_imp_scaled", "snr_imp_strict", "rmse", "prdn", "cc",
           "gain_bias", "ppv", "f1", "psd_logdist", "hr_err_bpm"]


def verdict(m: str, delta: float, p: float, a_mean: float, b_mean: float) -> str:
    if not np.isfinite(p) or p >= 0.05:
        return "n.s."
    if m in TO_ONE:
        better = abs(a_mean - 1.0) < abs(b_mean - 1.0)
    else:
        better = (delta < 0) if m in LOWER else (delta > 0)
    return "**우수**" if better else "**열등**"


def one_row(d: pd.DataFrame, m: str, meth: str, base: str):
    t = compare_methods(d, m, base)
    r = t[t.method == meth]
    if r.empty:
        return None
    r = r.iloc[0]
    return dict(delta=r.delta_mean, p=r.p_holm, r=r.effect_r,
                v=verdict(m, r.delta_mean, r.p_holm, r["mean"], r.baseline_mean))


def main() -> int:
    md = ["# 92. D0 / D1 대조 — 합성 벤치마크는 실데이터를 예측하는가",
          "",
          "> 자동 생성: `python scripts/analyze_axis_gap.py`  "
          "(수정하지 말 것 — 스크립트를 고칠 것)",
          "",
          "F-8 은 합성 데이터가 딥러닝을 12 dB 과대평가하고 있었음을 보였다.",
          "그 결함을 고친 뒤에도 **합성축의 결론이 실데이터로 옮겨가는가** 는",
          "따로 물어야 하는 질문이다. 이 문서가 그 답이다.",
          ""]

    dfs = {}
    for tag in ("d0", "d1"):
        p = Path("results") / tag / "exp_a" / "metrics.parquet"
        if not p.exists():
            print(f"[error] {p} 가 없다. 두 축 모두 EXP-A 를 먼저 돌릴 것.")
            return 2
        dfs[tag] = pd.read_parquet(p)

    for base in BASELINES:
        note = ("고전 DSP 대표. **참조(0.5–100 Hz)와 대역이 다르다**(0.5–40 Hz) — "
                "`psd_logdist` 비교는 이 때문에 왜곡된다."
                if base == "M01" else
                "공통 front-end 만 적용한 기준선. **참조와 같은 대역**이므로 "
                "'딥러닝이 front-end 위에 무엇을 더하는가' 를 재는 데 적절하다.")
        md += [f"## 기준선 `{base}` 대비", "", f"> {note}", "",
               "전 구간(입력 SNR −5~20 dB) 평균, paired Wilcoxon + Holm, 22 기록.", ""]
        for meth in DL:
            md += [f"### `{meth}` − `{base}`", "",
                   "| metric | D0 Δ | D0 p(Holm) | D0 판정 | D1 Δ | D1 p(Holm) | D1 판정 |",
                   "|---|---|---|---|---|---|---|"]
            for m in METRICS:
                cells = []
                for tag in ("d0", "d1"):
                    if m not in set(dfs[tag].metric.unique()):
                        cells.append(("—", "—", "—")); continue
                    r = one_row(dfs[tag], m, meth, base)
                    cells.append(("—", "—", "—") if r is None
                                 else (f"{r['delta']:+.4g}", f"{r['p']:.1e}", r["v"]))
                md.append(f"| `{m}` | " + " | ".join(c for cell in cells for c in cell) + " |")
            md.append("")

    # ---- 구간별: 전 구간 평균이 지우는 것
    md += ["## 입력 SNR 구간별 — 전 구간 평균이 지우는 것", "",
           "전 구간 평균에서 `snr_imp_scaled` 가 D1 에서 유의하지 않은 것은",
           "**딥러닝이 실데이터에서 안 된다** 는 뜻이 아니다. 구간별로 보면",
           "저 SNR 에서 크게 유의하고 고 SNR 에서 부호가 뒤집혀 상쇄된다.", ""]
    for base in BASELINES:
        md += [f"### `M08` − `{base}`, `snr_imp_scaled` (Holm 은 각 구간 안에서 보정)", "",
               "| 입력 SNR | D0 Δ | D0 p | D0 판정 | D1 Δ | D1 p | D1 판정 |",
               "|---|---|---|---|---|---|---|"]
        snrs = sorted(dfs["d1"].snr_in_target.dropna().unique())
        for s in snrs:
            cells = []
            for tag in ("d0", "d1"):
                sub = dfs[tag][dfs[tag].snr_in_target == s]
                r = one_row(sub, "snr_imp_scaled", "M08", base) if not sub.empty else None
                cells.append(("—", "—", "—") if r is None
                             else (f"{r['delta']:+.3f}", f"{r['p']:.1e}", r["v"]))
            md.append(f"| {s:g} dB | " + " | ".join(c for cell in cells for c in cell) + " |")
        md.append("")

    # ---- 오차 크기 지표가 왜 독립 증거가 아닌가
    d1 = dfs["d1"]
    md += ["## 왜 `rmse` 의 우세를 별도 증거로 쓰지 않는가", "",
           "`rmse` 는 D1 전 구간 평균에서 유의하고 `snr_imp_scaled` 는 아니다.",
           "지표가 서로 다른 것을 재는 것처럼 보이지만, **구간별로는 완전히 일치한다.**",
           "차이는 평균 방식에 있다 — 선형 오차는 저 SNR 조건이 압도적으로 무겁다.", "",
           "| 입력 SNR | `M08` rmse | `M01` rmse | 그 구간의 rmse 크기 |",
           "|---|---|---|---|"]
    piv = (d1[d1.metric == "rmse"]
           .pivot_table(index="snr_in_target", columns="method", values="value"))
    ref = piv["M01"].max()
    for s, row in piv.iterrows():
        md.append(f"| {s:g} dB | {row['M08']:.4f} | {row['M01']:.4f} | "
                  f"{row['M01'] / ref * 100:.0f} % |")
    md += ["",
           "가장 어려운 −5 dB 조건의 오차가 가장 쉬운 20 dB 조건의 "
           f"**{piv['M01'].max() / piv['M01'].min():.0f} 배**다. "
           "전 구간 rmse 평균은 사실상 저 SNR 성능이다.",
           "반면 dB 평균은 여섯 구간에 같은 무게를 준다.", ""]

    Path("docs/92_axis_gap.md").write_text("\n".join(md) + "\n")
    print("[ok] docs/92_axis_gap.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
