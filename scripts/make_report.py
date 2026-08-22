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
from ecgdn.registry import available, build, meta
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


def make_waveform_figures(out: Path, cfg_path: str = "configs/exp_a.yaml",
                          snr_db: float = 5.0, record_idx: int = 0):
    """F1/F2/F3 — 대표 구간의 파형 스택, QRS 확대, PSD.

    숫자 표만으로는 morphology 보존을 보여줄 수 없다. 이 그림들이 결과의 핵심이다.
    """
    import yaml

    from ecgdn.data.dataset import build_eval_set
    from ecgdn.data.nstdb import make_banks
    from ecgdn.data.sources import get_source
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    d = cfg.get("data", {})
    src = get_source(d.get("source", "auto"), dur_s=float(d.get("dur_s", 300.0)),
                     n_test=int(d.get("n_test", 22)))
    banks = make_banks("test", d.get("nstdb_root", "data/raw/nstdb"))
    items = build_eval_set(src, "test", seg_s=float(d.get("seg_s", 60.0)),
                           snr_grid=[snr_db], noise_conditions=("mixed",), banks=banks,
                           n_seg_per_record=1, seed=d.get("seed", "eval"))
    if not items:
        return None
    it = items[min(record_idx, len(items) - 1)]
    x, y, fs = it["x"].astype(float), it["y"].astype(float), it["fs"]

    methods = {}
    for mid in cfg.get("methods", []):
        try:
            methods[mid] = build(mid)
        except Exception as e:
            print(f"[warn] build({mid}) failed: {type(e).__name__}: {e}")
    for mid, spec in (cfg.get("dl_methods") or {}).items():
        spec = {"ckpt": spec} if isinstance(spec, str) else spec
        if Path(spec["ckpt"]).exists():
            from ecgdn.methods.dl_wrapper import DLDenoiser
            methods[mid] = DLDenoiser(ckpt=spec["ckpt"], name=mid, pre=spec.get("pre"),
                                      batch=32)
    outs = {}
    for mid, fn in methods.items():
        ctx = {"x_clean": x} if getattr(fn, "needs_clean", False) else {}
        try:
            outs[mid] = fn(y, fs, ctx)
        except Exception as e:
            print(f"[warn] {mid} failed on figure segment: {type(e).__name__}: {e}")

    rp = np.asarray(it["r_peaks"], dtype=int)
    mid_t = float(rp[len(rp) // 2]) / fs if rp.size else 20.0
    plots.waveform_stack(x, y, outs, fs, t0=max(6.0, mid_t - 2.0), dur=4.0,
                         path=out / "F1_waveforms.png",
                         title=f"Representative segment (record {it['record']}, "
                               f"input SNR {snr_db:.0f} dB)")
    plots.waveform_stack(x, y, outs, fs, t0=max(6.0, mid_t - 0.25), dur=0.6,
                         path=out / "F2_qrs_zoom.png",
                         title="QRS zoom (same segment)")
    plots.psd_compare(x, y, outs, fs, path=out / "F3_psd.png",
                      title=f"PSD (record {it['record']}, input SNR {snr_db:.0f} dB)")
    return it["record"]


def _pareto_front(rem: dict[str, float], pres: dict[str, float]) -> list[str]:
    """두 축 모두에서 다른 방법에 지배당하지 않는 방법들."""
    keys = [k for k in rem if k in pres and np.isfinite(rem[k]) and np.isfinite(pres[k])]
    front = []
    for k in keys:
        dominated = any(rem[o] >= rem[k] and pres[o] >= pres[k]
                        and (rem[o] > rem[k] or pres[o] > pres[k]) for o in keys if o != k)
        if not dominated:
            front.append(k)
    return sorted(front)


def _conclusions(a, c) -> list[str]:
    """결과표에서 직접 도출되는 결론만 쓴다 (사람이 손으로 쓰지 않는다)."""
    PRACTICAL = [m for m in plots.METHOD_ORDER
                 if m in a.method.unique() and not m.startswith("B") and m != "M00"]
    piv = a[a.metric == "snr_imp_scaled"].pivot_table(
        index="snr_in_target", columns="method", values="value", aggfunc="mean")

    md = ["## 결론 (표에서 직접 도출)", "",
          "### 1) 입력 SNR 구간별 최적 기법 (RQ1)", "",
          "| 입력 SNR [dB] | 최적(실용) | 그 값 | 2위 | oracle `B01` 대비 부족분 |",
          "|---|---|---|---|---|"]
    for snr in piv.index:
        row = piv.loc[snr, [m for m in PRACTICAL if m in piv.columns]].sort_values(ascending=False)
        b01 = piv.loc[snr, "B01"] if "B01" in piv.columns else np.nan
        gap = f"{b01 - row.iloc[0]:+.2f} dB" if np.isfinite(b01) else "—"
        md.append(f"| {snr:g} | **`{row.index[0]}`** | {row.iloc[0]:+.2f} dB "
                  f"| `{row.index[1]}` ({row.iloc[1]:+.2f}) | {gap} |")
    md.append("")

    # 유해 구간
    harmful = []
    for m in PRACTICAL:
        if m not in piv.columns:
            continue
        neg = [f"{s:g}" for s in piv.index if piv.loc[s, m] < 0]
        if neg:
            harmful.append(f"`{m}` (입력 {', '.join(neg)} dB)")
    if harmful:
        md += ["### 2) 오히려 신호를 나쁘게 만드는 구간", "",
               "아래 조건에서는 **아무 처리도 하지 않는 편이 낫다** (`snr_imp_scaled < 0`):", "",
               "- " + "\n- ".join(harmful), ""]

    # Pareto
    if c is not None:
        cd = per_record(c, "snr_out_strict").mean(axis=0).replace([np.inf, -np.inf], np.nan)
        rem = per_record(a, "snr_imp_scaled").mean(axis=0).to_dict()
        pres = {k: v for k, v in cd.to_dict().items() if np.isfinite(v)}
        front = _pareto_front({k: v for k, v in rem.items() if k in PRACTICAL},
                              {k: v for k, v in pres.items() if k in PRACTICAL})
        dom = [m for m in PRACTICAL if m in pres and m not in front]
        md += ["### 3) 잡음 제거 vs 신호 보존의 Pareto front (RQ3)", "",
               "가로축 = 잡음을 얼마나 제거했는가, 세로축 = 깨끗한 신호를 얼마나 안 망가뜨리는가.", "",
               f"- **Pareto front (실용 기법)**: {', '.join('`'+m+'`' for m in front)}",
               f"- **지배당함 (두 축 모두에서 더 나은 대안이 있음)**: "
               f"{', '.join('`'+m+'`' for m in dom) if dom else '없음'}", "",
               "> `distortion floor` 는 출력 SNR 의 **천장**이기도 하다. "
               "예를 들어 floor 가 15 dB 인 방법은 입력이 아무리 깨끗해도 출력이 15 dB 를 넘지 못한다.", ""]

    # 지표의 한계
    rp = per_record(a, "rpeak_mae_ms").mean(axis=0)
    if "M00" in rp:
        spread = float(np.nanmax(rp) - np.nanmin(rp))
        md += ["### 4) 이 실험에서 판별력이 없었던 지표", "",
               f"- `rpeak_mae_ms` 는 무처리(`M00`)에서도 {rp['M00']:.2f} ms 다. "
               "이는 방법의 성능이 아니라 **검출기(xqrs)의 체계적 위치 편향**이며 모든 방법에 공통이다. "
               f"방법 간 편차는 {spread:.2f} ms 로 좁아 이 조건에서는 판별력이 없다. "
               "MIT-BIH 처럼 형태 변이가 큰 데이터에서 다시 볼 것.", ""]

    md += ["### 5) 이 결과의 적용 범위", "",
           "- **합성 ECG(D0) 기준이다.** MIT-BIH(D1) 로 반드시 재검증해야 한다 "
           "(`docs/02_procedure.md` STEP 14-15).",
           "- 딥러닝은 이 합성 분포에서 학습했다. 다른 morphology 분포에서 얼마나 유지되는지는 "
           "별도 검증 대상이다 (F-8 참조).",
           "- 실측 Arduino 신호의 SNR 을 추정한 뒤(`STEP 29`), 위 1) 표에서 해당 구간의 행을 보면 "
           "**그 장비에 어떤 기법을 써야 하는지** 바로 읽을 수 있다.", ""]
    return md


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-snr", type=float, default=None,
                    help="실측 장비 SNR 추정치 [dB]. F4 에 수직선으로 표시.")
    ap.add_argument("--fig-snr", type=float, default=5.0,
                    help="대표 파형 그림(F1-F3)의 입력 SNR [dB].")
    ap.add_argument("--no-waveforms", action="store_true")
    args = ap.parse_args()

    out = ensure_dir("results/report")
    floor = load_floor()
    md: list[str] = ["# 90. 실험 결과", "",
                     "> 자동 생성: `python scripts/make_report.py`  "
                     "(수정하지 말 것 — 스크립트를 고칠 것)", ""]

    a, b, c = load_exp("exp_a"), load_exp("exp_b"), load_exp("exp_c")

    # ---------------- 대표 파형 (숫자표가 못 보여주는 것)
    if not args.no_waveforms:
        try:
            rec = make_waveform_figures(out, snr_db=args.fig_snr)
            if rec:
                md += ["## 대표 파형", "",
                       f"기록 `{rec}`, 입력 SNR {args.fig_snr:.0f} dB, 동일 y축.", "",
                       "![F1](../results/report/F1_waveforms.png)", "",
                       "### QRS 확대", "",
                       "![F2](../results/report/F2_qrs_zoom.png)", "",
                       "### 주파수 스펙트럼", "",
                       "![F3](../results/report/F3_psd.png)", "",
                       "PSD 는 시간영역에서 보이지 않는 것을 보여준다: "
                       "어떤 방법이 60 Hz 를 지웠는지, 어떤 방법이 QRS 의 고주파 성분까지 "
                       "함께 잘라냈는지.", ""]
        except Exception as e:
            print(f"[warn] waveform figures skipped: {type(e).__name__}: {e}")

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

    # ---------------- EXP-D (RQ4): wavelet 을 어디에 쓸 것인가
    if a is not None:
        trio = ["M06", "M07", "M08"]
        have = [m for m in trio if m in a.method.unique()]
        if len(have) >= 2:
            md += ["## EXP-D — wavelet 을 어디에 쓸 것인가 (RQ4)", "",
                   "같은 backbone(Residual U-Net)·같은 손실·같은 데이터에서 **wavelet 의 위치만** 바꾼다.",
                   "",
                   "| ID | wavelet 의 역할 |", "|---|---|",
                   "| `M06` | 쓰지 않음 (raw 파형을 그대로 입력) |",
                   "| `M07` | **전처리 필터** — SWT thresholding 으로 1차 제거한 뒤 U-Net |",
                   "| `M08` | **입력 표현공간** — SWT subband 를 채널로 주고 U-Net 이 대역별 잡음을 학습 |",
                   ""]
            rows = []
            for m, _, _ in MAIN_METRICS:
                w = per_record(a, m)
                if w.empty:
                    continue
                rows.append([m] + [f"{w[x].mean():.4g}" if x in w else "—" for x in have])
            md += ["| metric | " + " | ".join(have) + " |",
                   "|---" * (len(have) + 1) + "|"]
            for r in rows:
                md.append("| `" + r[0] + "` | " + " | ".join(r[1:]) + " |")
            md += ["",
                   "이 표가 답하는 것: **DSP 를 AI 앞단에 두는 것**과 "
                   "**DSP 가 정의한 표현공간을 AI 에게 주는 것** 중 무엇이 효과적인가.",
                   "차이가 크지 않다면 그것도 결과다 — 'wavelet 표현이 U-Net 이 스스로 배우는 것 이상을 주지 못한다'.",
                   ""]

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

    # ---------------- 자동 결론 (데이터에서만 도출)
    if a is not None:
        md += _conclusions(a, c)

    doc = ensure_dir("docs") / "90_results.md"
    doc.write_text("\n".join(md) + "\n")
    print(f"-> {doc}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
