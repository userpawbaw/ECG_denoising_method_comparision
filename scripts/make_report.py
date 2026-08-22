"""STEP 27: 모든 표·그림을 한 번에 재생성한다.

    python scripts/make_report.py

`results/` 의 산출물만 읽는다 (원본 데이터 재접근 불필요).
산출: results/report/*.{png,csv,md}, docs/90_results.md
"""
import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import ecgdn.methods  # noqa: F401
from ecgdn.eval.stats import compare_methods, summarize
from ecgdn.registry import available, meta
from ecgdn.utils import ensure_dir
from ecgdn.viz import plots

MAIN_METRICS = [
    ("snr_imp_scaled", "SNR improvement [dB]", "↑"),
    ("rmse", "RMSE [mV]", "↓"),
    ("cc", "Pearson CC", "↑"),
    ("rpeak_mae_ms", "R-peak MAE [ms]", "↓"),
    ("qrs_dur_err_ms", "QRS duration error [ms]", "↓"),
]
AUX_METRICS = [
    ("snr_imp_strict", "SNR imp. (no gain corr.) [dB]", "↑"),
    ("gain_bias", "gain bias α*", "→1"),
    ("prdn", "PRDN [%]", "↓"),
    ("r_amp_err_pct", "R amplitude error [%]", "↓"),
    ("beat_cc", "beat template CC", "↑"),
    ("hr_err_bpm", "HR error [bpm]", "↓"),
    ("psd_logdist", "PSD log-distance [dB]", "↓"),
    ("rtf", "RTF (latency / signal)", "↓"),
]


def load_exp(name: str):
    p = Path("results") / name / "metrics.parquet"
    return pd.read_parquet(p) if p.exists() else None


def load_floor() -> dict[str, float]:
    p = Path("results/metric_floor/floor.csv")
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    return dict(zip(d["metric"], d["floor_p95"]))


def per_record(df, metric):
    """record 단위 집계 (통계의 기본 단위)."""
    s = df[df.metric == metric]
    return s.pivot_table(index="record", columns="method", values="value", aggfunc="mean")


def table_main(df, floor: dict[str, float]) -> pd.DataFrame:
    rows = []
    for m, label, direction in MAIN_METRICS + AUX_METRICS:
        w = per_record(df, m)
        if w.empty:
            continue
        for meth in w.columns:
            s = summarize(w[meth].to_numpy())
            rows.append(dict(metric=m, label=label, dir=direction, method=meth,
                             **s, floor_p95=floor.get(m, np.nan)))
    return pd.DataFrame(rows)


def fmt_table(t: pd.DataFrame, metrics, floor) -> list[str]:
    methods = plots._order(sorted(t.method.unique()))
    lines = ["| method | " + " | ".join(f"{lab} {d}" for _, lab, d in metrics) + " |",
             "|---" * (len(metrics) + 1) + "|"]
    for meth in methods:
        cells = []
        for m, _, _ in metrics:
            r = t[(t.metric == m) & (t.method == meth)]
            if r.empty or not np.isfinite(r["mean"].iloc[0]):
                cells.append("—"); continue
            mu, sd = float(r["mean"].iloc[0]), float(r["std"].iloc[0])
            fl = floor.get(m, np.nan)
            mark = " *" if (np.isfinite(fl) and abs(mu) < fl) else ""
            digits = 4 if m in ("cc", "beat_cc", "rmse", "rtf", "gain_bias") else 2
            cells.append(f"{mu:.{digits}f} ± {sd:.{digits}f}{mark}")
        lines.append(f"| `{meth}` | " + " | ".join(cells) + " |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-snr", type=float, default=None,
                    help="실측 장비 SNR 추정치 [dB]. F4 에 수직선으로 표시.")
    args = ap.parse_args()

    out = ensure_dir("results/report")
    floor = load_floor()
    md: list[str] = ["# 90. 실험 결과", "",
                     "> 자동 생성: `python scripts/make_report.py`  "
                     "(수정하지 말 것 — 스크립트를 고칠 것)", ""]

    a, b, c = load_exp("exp_a"), load_exp("exp_b"), load_exp("exp_c")

    # ---------------- 방법 목록
    md += ["## 비교 대상", "", "| ID | 분류 | 설명 |", "|---|---|---|"]
    for mid in plots.METHOD_ORDER:
        if mid in available():
            mm = meta(mid)
            md.append(f"| `{mid}` | {mm.get('family','')} | {mm.get('label','')} |")
    md.append("")

    # ---------------- EXP-A
    if a is not None:
        t = table_main(a, floor)
        t.to_csv(out / "table_main.csv", index=False)
        md += ["## EXP-A — 입력 SNR sweep (RQ1)", "",
               f"평가 구간 {a.record.nunique()} 기록 × "
               f"{a[a.metric=='snr_imp_scaled'].groupby('record').size().median():.0f} 조건, "
               "집계 단위는 **record**.", "",
               "### T1. 주 지표 (mean ± std, record 단위)", ""]
        md += fmt_table(t, MAIN_METRICS, floor)
        md += ["", "`*` 표시는 `docs/03_metric_floor.md` 의 지표 분해능(floor p95)보다 "
               "작은 값 = **구분 불가**.", "",
               "### T1b. 보조 지표", ""]
        md += fmt_table(t, AUX_METRICS, floor)
        md.append("")

        plots.snr_curve(a, "snr_imp_scaled", out / "F4_snr_curve.png",
                        real_snr=args.real_snr,
                        title="EXP-A: SNR improvement vs input SNR")
        plots.snr_curve(a, "qrs_dur_err_ms", out / "F4b_qrs_curve.png",
                        real_snr=args.real_snr, ylabel="QRS duration error [ms]",
                        title="EXP-A: QRS duration error vs input SNR")
        md += ["![F4](../results/report/F4_snr_curve.png)", "",
               "![F4b](../results/report/F4b_qrs_curve.png)", ""]

        # 통계
        stats = []
        for m, _, _ in MAIN_METRICS:
            try:
                s = compare_methods(a, m, baseline="M01")
                if not s.empty:
                    stats.append(s)
            except Exception:
                pass
        if stats:
            st = pd.concat(stats, ignore_index=True)
            st.to_csv(out / "table_stats.csv", index=False)
            md += ["### T2. 통계 검정 (baseline = `M01`, paired Wilcoxon + Holm)", "",
                   "| metric | method | Δmean | p | p(Holm) | effect r |",
                   "|---|---|---|---|---|---|"]
            for _, r in st.iterrows():
                if r["metric"] not in ("snr_imp_scaled", "qrs_dur_err_ms"):
                    continue
                p = "—" if not np.isfinite(r["p"]) else f"{r['p']:.2e}"
                ph = "—" if not np.isfinite(r["p_holm"]) else f"{r['p_holm']:.2e}"
                md.append(f"| `{r['metric']}` | `{r['method']}` | {r['delta_mean']:+.3f} "
                          f"| {p} | {ph} | {r['effect_r']:+.3f} |")
            md.append("")

    # ---------------- EXP-C + Pareto
    if c is not None:
        cd = per_record(c, "snr_out_strict").mean(axis=0)
        cd = cd.replace([np.inf, -np.inf], np.nan)
        cd.to_csv(out / "table_distortion.csv")
        md += ["## EXP-C — distortion floor (RQ3)", "",
               "**잡음이 전혀 없는 clean 신호**를 각 방법에 그대로 통과시켰을 때의 출력 SNR.",
               "낮을수록 그 방법은 '잡음이 없어도 신호를 망가뜨린다'. "
               "출력 SNR 은 원리적으로 이 값을 넘을 수 없으므로 **성능의 천장**이기도 하다.", "",
               "| method | distortion floor [dB] ↑ |", "|---|---|"]
        for meth in plots._order(list(cd.index)):
            v = cd[meth]
            md.append(f"| `{meth}` | {'∞ (identity)' if not np.isfinite(v) else f'{v:.2f}'} |")
        md.append("")
        if a is not None:
            rem = per_record(a, "snr_imp_scaled").mean(axis=0).to_dict()
            pres = {k: v for k, v in cd.to_dict().items() if np.isfinite(v)}
            plots.pareto_scatter(rem, pres, out / "F5_pareto.png")
            md += ["![F5](../results/report/F5_pareto.png)", "",
                   "오른쪽 위로 갈수록 좋다. 오른쪽 아래는 '잡음은 잘 지우지만 파형을 망가뜨리는' 방법이다.", ""]

    # ---------------- EXP-B
    if b is not None:
        plots.noise_heatmap(b, "snr_imp_scaled", out / "F6_noise_heatmap.png",
                            title="EXP-B: snr_imp_scaled by noise type (input 10 dB)")
        md += ["## EXP-B — 잡음 종류별 강약 (RQ2)", "",
               "![F6](../results/report/F6_noise_heatmap.png)", ""]
        piv = b[b.metric == "snr_imp_scaled"].pivot_table(index="method", columns="cond",
                                                          values="value", aggfunc="mean")
        piv.round(2).to_csv(out / "table_noise.csv")
        md += ["| method | " + " | ".join(piv.columns) + " |",
               "|---" * (len(piv.columns) + 1) + "|"]
        for meth in plots._order(list(piv.index)):
            md.append(f"| `{meth}` | " + " | ".join(f"{v:.2f}" for v in piv.loc[meth]) + " |")
        md.append("")

    # ---------------- 학습 곡선 / 비용
    runs = {}
    for p in sorted(Path("results").glob("m0*/log.csv")):
        try:
            runs[p.parent.name] = pd.read_csv(p)
        except Exception:
            pass
    if runs:
        plots.training_curves(runs, out / "F8_training.png")
        md += ["## 학습 곡선", "", "![F8](../results/report/F8_training.png)", "",
               "| run | best epoch | best val snr_imp_scaled [dB] |", "|---|---|---|"]
        for nm, d in runs.items():
            i = int(d["val_snr_imp_scaled"].idxmax())
            md.append(f"| `{nm}` | {int(d['epoch'].iloc[i])} | "
                      f"{d['val_snr_imp_scaled'].iloc[i]:.2f} |")
        md.append("")

    if a is not None:
        cost = per_record(a, "rtf").mean(axis=0)
        md += ["### T3. 계산 비용", "", "| method | RTF (latency / signal duration) |",
               "|---|---|"]
        for meth in plots._order(list(cost.index)):
            md.append(f"| `{meth}` | {cost[meth]:.4f} |")
        md += ["", "RTF < 1 이면 실시간 처리 가능. (단일 CPU 스레드, 60 s 구간 기준)", ""]

    doc = ensure_dir("docs") / "90_results.md"
    doc.write_text("\n".join(md) + "\n")
    print(f"-> {doc}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
