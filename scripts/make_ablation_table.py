#!/usr/bin/env python3
"""STEP 19 DoD — loss ablation 표 생성.

    python scripts/run_exp.py -c configs/abl_loss.yaml
    python scripts/make_ablation_table.py

`results/abl_loss/metrics.parquet` (long format) 을 읽어
`results/ablation_loss.csv` 와 `docs/10_loss_ablation.md` 를 만든다.

표에 **체크포인트의 git hash 와 frontend 플래그를 함께 싣는다.** F-9 에서
겪었듯, 파이프라인 앞단을 고치면 그 뒤로 학습된 체크포인트가 한꺼번에 무효가
되는데 표만 보면 그 사실이 드러나지 않기 때문이다. 학습 조건이 섞여 있으면
이 스크립트가 **경고를 내고 비교를 거부**한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# DoD 가 지정한 지표. SNR 하나만 보면 "매끄럽게 뭉갠" 출력이 이기므로
# morphology 지표를 반드시 같이 본다.
METRICS = ["snr_imp_scaled", "qrs_dur_err_ms", "beat_cc"]
LOWER_BETTER = {"qrs_dur_err_ms"}


def ckpt_provenance(run: str) -> dict:
    """체크포인트가 '언제, 어떤 조건에서' 학습됐는지."""
    d = Path("results") / run
    out = {"run": run, "git": "?", "frontend": "?", "loss": "?", "model": "?"}
    mf = d / "manifest.json"
    if mf.exists():
        m = json.loads(mf.read_text())
        out["git"] = m.get("env", {}).get("git", "?")
        out["model"] = m.get("model", "?")
        ex = m.get("extra", m)
        out["frontend"] = ex.get("frontend", "?")
        out["loss"] = ex.get("loss", "?")
    return out


def main() -> int:
    src = Path("results/abl_loss/metrics.parquet")
    if not src.exists():
        print(f"[!] {src} 없음. 먼저 실행: python scripts/run_exp.py -c configs/abl_loss.yaml")
        return 1
    d = pd.read_parquet(src)
    d = d[d["metric"].isin(METRICS)]

    # method 이름 'M06-L3' -> 체크포인트 디렉터리 'm06_l3'
    runs = {m: m.lower().replace("-", "_") for m in d["method"].unique()
            if m.upper().startswith("M0") and "-L" in m.upper()}
    prov = {m: ckpt_provenance(r) for m, r in runs.items()}

    # --- 학습 조건이 섞였는지 먼저 확인한다 (F-9 재발 방지) -------------
    fes = {str(p["frontend"]) for p in prov.values()}
    mixed = len(fes) > 1
    if mixed:
        print("[!] 체크포인트의 front-end 설정이 섞여 있다:", sorted(fes))
        print("    이 상태의 비교는 loss 효과가 아니라 front-end 효과를 잰다 (F-9).")
        print("    전부 같은 조건으로 재학습한 뒤 다시 실행할 것.")

    # 기록 단위로 먼저 접은 뒤 요약한다 (구간 수가 많은 기록이 평균을 지배하지 않도록)
    per_rec = (d.groupby(["method", "snr_in_target", "metric", "record"])["value"]
                 .mean().reset_index())
    agg = (per_rec.groupby(["method", "snr_in_target", "metric"])["value"]
                  .agg(["mean", "std", "count"]).reset_index())

    rows = []
    for _, r in agg.iterrows():
        p = prov.get(r["method"], {})
        rows.append(dict(method=r["method"], snr_in=r["snr_in_target"],
                         metric=r["metric"], mean=r["mean"], std=r["std"],
                         n_records=int(r["count"]),
                         loss=p.get("loss", ""), model=p.get("model", ""),
                         frontend=p.get("frontend", ""), ckpt_git=p.get("git", "")))
    out = pd.DataFrame(rows).sort_values(["metric", "snr_in", "method"])
    Path("results").mkdir(exist_ok=True)
    out.to_csv("results/ablation_loss.csv", index=False)
    print(f"[ok] results/ablation_loss.csv  ({len(out)} 행)")

    # --- 사람이 읽는 표 -------------------------------------------------
    md = ["# STEP 19 — loss ablation 결과", "",
          "`configs/abl_loss.yaml` 로 **TEST split** 에서 평가한 결과다.",
          "학습 로그의 VAL 값이 아니다 (F-9 참조).", ""]
    if mixed:
        md += ["> ⚠️ **이 표는 아직 유효하지 않다.** 체크포인트의 front-end 설정이 "
               f"섞여 있다({sorted(fes)}). 손실 항의 효과와 front-end 의 효과가 "
               "분리되지 않는다.", ""]

    md += ["## 체크포인트 출처", "",
           "| 방법 | 모델 | loss | front-end | 학습 시점 git |", "|---|---|---|---|---|"]
    for m in sorted(prov):
        p = prov[m]
        md.append(f"| `{m}` | {p['model']} | {p['loss']} | `{p['frontend']}` | `{p['git']}` |")
    md.append("")

    for met in METRICS:
        sub = out[out["metric"] == met]
        if sub.empty:
            continue
        arrow = "낮을수록 좋음" if met in LOWER_BETTER else "높을수록 좋음"
        piv = sub.pivot_table(index="method", columns="snr_in", values="mean")
        md += [f"## `{met}` ({arrow})", "",
               "| 방법 | " + " | ".join(f"{c:g} dB" for c in piv.columns) + " |",
               "|---" * (len(piv.columns) + 1) + "|"]
        best = piv.min() if met in LOWER_BETTER else piv.max()
        for m, row in piv.iterrows():
            cells = []
            for c in piv.columns:
                v = row[c]
                cells.append(f"**{v:.3f}**" if np.isclose(v, best[c]) else f"{v:.3f}")
            md.append(f"| `{m}` | " + " | ".join(cells) + " |")
        md.append("")

    md += ["## 읽는 법", "",
           "- `snr_imp_scaled` 만 보고 결론 내리지 않는다. 출력을 매끄럽게 뭉개면",
           "  SNR 은 오르지만 QRS 가 뭉툭해진다. 그래서 `qrs_dur_err_ms` 를 같이 본다.",
           "- `M_FE` 는 공통 front-end 만 적용한 기준선이다. loss 간 차이가 이 기준선",
           "  대비 얼마나 되는지가 실질적 크기다.",
           "- 차이가 지표의 분해능(`docs/03_metric_floor.md`) 미만이면 차이가 아니다.", ""]
    Path("docs/10_loss_ablation.md").write_text("\n".join(md) + "\n")
    print("[ok] docs/10_loss_ablation.md")
    return 2 if mixed else 0


if __name__ == "__main__":
    raise SystemExit(main())
