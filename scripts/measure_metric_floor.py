"""STEP 07: metric noise floor 측정 (docs/00_review.md A-8).

지표 자체의 오차를 먼저 재둔다. 방법 간 차이가 이 값보다 작으면 "구분 불가" 다.

절차
----
  clean x 에 아주 약한 교란(기본 40 dB AWGN)을 여러 seed 로 주고,
  각 지표가 '이상값(perfect)' 에서 얼마나 흔들리는지 측정한다.
  QRS delineation 처럼 알고리즘적 불안정성이 큰 지표는 여기서 드러난다.

산출: docs/03_metric_floor.md
"""
import _bootstrap  # noqa: F401

import argparse

import numpy as np
import pandas as pd

from ecgdn.config import EVAL_GUARD_S, FS
from ecgdn.data.mixer import mix_at_snr
from ecgdn.data.noise import awgn
from ecgdn.data.sources import real_clean_segments
from ecgdn.data.synthetic import synth_ecg
from ecgdn.eval.engine import evaluate
from ecgdn.utils import ensure_dir, rng, save_manifest

# 방향: 이상값이 무엇인지 (표시용)
IDEAL = {
    "snr_out_strict": np.inf, "snr_out_scaled": np.inf, "gain_bias": 1.0,
    "rmse": 0.0, "mae": 0.0, "prdn": 0.0, "cc": 1.0,
    "se": 1.0, "ppv": 1.0, "f1": 1.0, "rpeak_mae_ms": 0.0, "rpeak_bias_ms": 0.0,
    "hr_err_bpm": 0.0, "rr_mae_ms": 0.0,
    "r_amp_err_pct": 0.0, "beat_cc": 1.0, "beat_cc_median": 1.0,
    "qrs_dur_err_ms": 0.0, "psd_logdist": 0.0,
}
REPORT = ["rmse", "prdn", "cc", "beat_cc", "beat_cc_median", "r_amp_err_pct",
          "rpeak_mae_ms", "rpeak_bias_ms", "hr_err_bpm", "rr_mae_ms",
          "qrs_dur_err_ms", "delineate_success_rate", "psd_logdist",
          "snr_out_strict", "gain_bias"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-snr", type=float, default=40.0,
                    help="교란 강도 [dB]. 클수록 약한 교란.")
    ap.add_argument("--n-seed", type=int, default=10)
    ap.add_argument("--dur", type=float, default=120.0)
    ap.add_argument("--n-record", type=int, default=5)
    ap.add_argument("--source", default="synthetic", choices=("synthetic", "mitdb"),
                    help="config 없이 도는 보조 스크립트다. synthetic 이 기본 — D0 결과(이미 보고서에 인용된 값)와의 연속성을 지킨다")
    args = ap.parse_args()

    tag = "d0" if args.source == "synthetic" else "d1"
    suffix = "" if tag == "d0" else "_d1"

    if args.source == "mitdb":
        # TRAIN split 을 쓴다 — 지표의 분해능을 재는 일이 TEST 에 새어들면 안 된다.
        segs = real_clean_segments(args.n_record, args.dur, fs=FS, split="train")
    else:
        segs = [synth_ecg(args.dur, fs=FS, hr_bpm=60 + 8 * rec, seed=100 + rec)
                for rec in range(args.n_record)]

    rows = []
    for rec, s in enumerate(segs):
        base = evaluate(s.x, None, s.x.copy(), s.fs, r_peaks_ref=s.r_peaks)
        for k in REPORT:
            rows.append(dict(record=rec, seed=-1, kind="perfect", metric=k,
                             value=base.get(k, np.nan)))
        for sd in range(args.n_seed):
            _xr = getattr(s, "x_raw", s.x)
            y, _, _ = mix_at_snr(_xr, awgn(len(_xr), s.fs, rng("floor", rec, sd)),
                                 args.probe_snr)      # 잡음은 원본에 (F-12)
            m = evaluate(s.x, y, y, s.fs, r_peaks_ref=s.r_peaks)
            for k in REPORT:
                rows.append(dict(record=rec, seed=sd, kind="probe", metric=k,
                                 value=m.get(k, np.nan)))

    df = pd.DataFrame(rows)
    out_dir = ensure_dir(f"results/{tag}/metric_floor")
    df.to_csv(out_dir / "raw.csv", index=False)

    # floor = |probe - perfect| 의 평균 및 95 퍼센타일
    lines = []
    for k in REPORT:
        pf = df[(df.metric == k) & (df.kind == "perfect")].set_index("record")["value"]
        pb = df[(df.metric == k) & (df.kind == "probe")]
        if pb.empty:
            continue
        d = pb.apply(lambda r: abs(r["value"] - pf.get(r["record"], np.nan)), axis=1)
        d = d[np.isfinite(d)]
        if d.empty:
            continue
        lines.append(dict(metric=k, ideal=IDEAL.get(k, np.nan),
                          floor_mean=float(d.mean()), floor_p95=float(np.percentile(d, 95)),
                          floor_max=float(d.max())))
    fl = pd.DataFrame(lines)
    fl.to_csv(out_dir / "floor.csv", index=False)

    # ---- 문서 생성
    md = ["# 03. Metric noise floor",
          "",
          "> 자동 생성: `python scripts/measure_metric_floor.py`  "
          "(수정하지 말 것 — 스크립트를 고칠 것)",
          "",
          f"합성 ECG {args.n_record} 기록 × {args.dur:.0f} s 에 **{args.probe_snr:.0f} dB "
          f"AWGN**(거의 무시할 수준의 교란)을 {args.n_seed} 개 seed 로 준 뒤,",
          "각 지표가 이상값에서 얼마나 흔들리는지 측정했다. "
          f"평가 guard = {EVAL_GUARD_S:.0f} s.",
          "",
          "**결과표에서 방법 간 차이가 `floor_p95` 보다 작으면 `n.s.`(구분 불가)로 표기한다.**",
          "",
          "| metric | 이상값 | floor (mean) | floor (p95) | floor (max) |",
          "|---|---|---|---|---|"]
    for _, r in fl.iterrows():
        ideal = "∞" if np.isinf(r["ideal"]) else (f"{r['ideal']:g}" if np.isfinite(r["ideal"]) else "—")
        md.append(f"| `{r['metric']}` | {ideal} | {r['floor_mean']:.4g} | "
                  f"{r['floor_p95']:.4g} | {r['floor_max']:.4g} |")

    q = fl[fl.metric == "qrs_dur_err_ms"]
    md += ["", "## 해석", ""]
    if not q.empty:
        md.append(f"- **`qrs_dur_err_ms` 의 floor 는 p95 기준 {float(q.floor_p95.iloc[0]):.1f} ms** 다. "
                  "즉 두 방법의 QRS duration 오차가 이보다 작게 차이 나면 "
                  "그것은 denoising 성능 차이가 아니라 **delineator 자체의 불안정성**이다. "
                  "(docs/00_review.md A-8 에서 예고한 항목)")
    pl = fl[fl.metric == "psd_logdist"]
    md += [
        "- `rpeak_mae_ms` / `hr_err_bpm` / `rr_mae_ms` 는 floor 가 사실상 0 이다. "
        "타이밍 계열 지표는 해상도가 높아 신뢰할 수 있다.",
        "- `beat_cc`, `cc` 는 1 에 붙어 있어 소수점 4~5 자리까지 봐야 한다. "
        "표에는 `1 - cc` 형태로 적는 것을 권한다.",
    ]
    if not pl.empty:
        md.append(f"- **`psd_logdist` 의 floor 가 {float(pl.floor_p95.iloc[0]):.1f} dB 로 매우 크다.** "
                  "ECG 파워가 거의 없는 주파수 구간에서 log-PSD 차이가 폭발하기 때문이다. "
                  "따라서 `psd_logdist` 는 **절대값으로 해석하지 말고 방법 간 상대 비교로만** 쓴다. "
                  "PSD 는 그림(F3)으로 보이는 것이 주 용도다.")
    md += [
        "",
        "## 이 표의 한계",
        "",
        "- 여기서 쓴 것은 **합성 ECG** 다. 파형이 실제보다 규칙적이라 "
        "`qrs_dur_err_ms` 의 floor 가 낙관적으로 나올 수 있다.",
        "  MIT-BIH 를 확보한 뒤 (STEP 15) **동일 스크립트를 `--source mitdb` 로 다시 돌려** "
        "실데이터 floor 로 갱신할 것.",
        "- floor 의 정의는 '이상값 대비, 40 dB 교란에서의 편차' 다. "
        "즉 **지표의 실효 분해능**이며, 알고리즘 자체의 불안정성(delineator 실패 등)과 "
        "교란에 대한 정상적 민감도가 합쳐진 값이다.",
        "- `snr_out_*` 는 이상값이 무한대라 이 표에 포함되지 않는다 "
        "(대신 `gain_bias` 로 안정성을 본다).",
        "",
        "## 재현", "", "```bash", "python scripts/measure_metric_floor.py", "```",
    ]
    doc = ensure_dir("docs") / f"03_metric_floor{suffix}.md"
    doc.write_text("\n".join(md) + "\n")
    save_manifest(out_dir, cfg=vars(args))
    print(fl.to_string(index=False))
    print(f"\n-> {doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
