"""STEP 29: 실측 Arduino 신호의 SNR 추정 — **"정말 15 dB 인가?" 에 답한다.**

    python scripts/estimate_real_snr.py --in data/arduino --out results/real_snr.csv

docs/04_snr_estimator_calibration.md 의 편향표와 함께 해석해야 한다.
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ecgdn.config import FS
from ecgdn.data.arduino import NOISE_ONLY, load_dir
from ecgdn.eval.rpeak import detect_rpeaks
from ecgdn.eval.snr_estimation import SNR_CEILING_BY_AXIS, estimate_snr_all

# 실측 장비 신호는 **실데이터**다. 합성에서 정한 천장(22 dB)을 쓰면 추정치가
# 이미 천장인데도 경고가 뜨지 않는다 (F-18). 실기록에서 잰 값을 쓴다.
CEILING_DB = SNR_CEILING_BY_AXIS["d1"]
from ecgdn.eval.spectral import pli_ratio, welch_psd
from ecgdn.methods.frontend import apply_frontend
from ecgdn.utils import ensure_dir, power, save_manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/arduino")
    ap.add_argument("--out", default="results/real_snr")
    ap.add_argument("--seg-s", type=float, default=60.0,
                    help="구간 길이. 여러 구간으로 나눠 분포를 본다.")
    args = ap.parse_args()

    recs = load_dir(args.inp, fs_out=FS)
    if not recs:
        print(f"[!] {args.inp} 에 CSV 가 없다.")
        print("    docs/08_acquisition.md 의 수집 프로토콜(S1~S6)을 먼저 수행할 것.")
        print("    스키마: 헤더 주석 3줄 (# fs_hz=..., # adc_bits=..., # session=...)")
        return 1

    out = ensure_dir(args.out)
    rows = []
    for r in recs:
        n_seg = int(round(args.seg_s * r.fs))
        n_chunk = max(1, r.x.size // n_seg)
        for ci in range(n_chunk):
            seg = r.x[ci * n_seg:(ci + 1) * n_seg]
            if seg.size < int(20 * r.fs):
                continue
            fe = apply_frontend(seg, r.fs)
            base = dict(file=r.name, session=r.session, chunk=ci,
                        dur_s=seg.size / r.fs, unit=r.unit,
                        pli_ratio_60=pli_ratio(seg, r.fs, 60.0),
                        pli_ratio_50=pli_ratio(seg, r.fs, 50.0))
            if r.is_noise_only:
                # 잡음 전용 세션: SNR 이 아니라 잡음 파워/스펙트럼을 기록
                f, p = welch_psd(fe, r.fs)
                rows.append({**base, "kind": "noise_only",
                             "noise_rms": float(np.sqrt(power(fe))),
                             "n_beats": 0.0})
                continue
            est_raw = estimate_snr_all(seg, r.fs, ceiling_db=CEILING_DB)
            est_fe = estimate_snr_all(fe, r.fs, ceiling_db=CEILING_DB)
            rows.append({**base, "kind": "ecg",
                         "n_beats": est_fe["n_beats"],
                         **{f"raw_{k}": v for k, v in est_raw.items() if k != "n_beats"},
                         **{f"fe_{k}": v for k, v in est_fe.items() if k != "n_beats"}})

    df = pd.DataFrame(rows)
    df.to_csv(Path(out) / "real_snr.csv", index=False)
    save_manifest(out, cfg=vars(args), sources=[
        "scripts/estimate_real_snr.py", "ecgdn/eval/snr_estimation.py"])

    ecg = df[df.kind == "ecg"] if "kind" in df else df
    md = ["# 08b. 실측 신호 SNR 추정 결과", "",
          "> 자동 생성: `python scripts/estimate_real_snr.py`", "",
          "**해석은 `docs/04_snr_estimator_calibration.md` 의 편향표와 함께 해야 한다.**",
          "beat 평균 계열 추정기는 잡음 종류에 따라 체계적으로 치우친다.", ""]
    if not ecg.empty:
        cols = ["fe_snr_beat_residual_db", "fe_snr_hsc_db", "fe_snr_isoelectric_db",
                "fe_snr_median_db", "fe_snr_spread_db", "fe_beat_resid_lag1_corr",
                "fe_artifact_beat_frac"]
        cols = [c for c in cols if c in ecg.columns]
        g = ecg.groupby("session")[cols].mean().round(2)
        md += ["## 세션별 요약 (front-end 적용 후)", "", g.to_markdown(), "",
               "| 항목 | 값 |", "|---|---|",
               f"| 전체 중앙값 SNR | **{ecg['fe_snr_median_db'].median():.1f} dB** |",
               f"| 추정기 간 최대 불일치 (평균) | {ecg['fe_snr_spread_db'].mean():.1f} dB |",
               f"| 상관잡음 경고 비율 | {ecg.get('fe_correlated_noise_warning', pd.Series([0])).mean()*100:.0f} % |",
               f"| 포화 경고 비율 (≥ {CEILING_DB:.0f} dB) | "
               f"{ecg.get('fe_ceiling_warning', pd.Series([0])).mean()*100:.0f} % |",
               f"| 60 Hz PLI ratio (중앙값) | {ecg['pli_ratio_60'].median():.1f} "
               "(≥ 10 이면 notch 적용 대상) |", ""]
        md += ["## 다음 단계", "",
               "1. 위 중앙값 SNR 을 `scripts/make_report.py --real-snr <값>` 에 넣으면",
               "   EXP-A 곡선(F4)에 **실측 조건 위치**가 수직선으로 표시된다.",
               "2. 그 위치에서 어떤 기법이 가장 유리한지가 곧 RQ1 의 답이다.", ""]
    noise = df[df.kind == "noise_only"] if "kind" in df else pd.DataFrame()
    if not noise.empty:
        md += ["## 잡음 전용 세션 (S2~S5)", "",
               noise.groupby("session")["noise_rms"].agg(["count", "mean", "std"]).round(5).to_markdown(),
               "", "이 구간들은 딥러닝 fine-tuning 용 **장비 고유 잡음 데이터셋**이 된다 (EXP-F).", ""]
    doc = ensure_dir("docs") / "08b_real_snr.md"
    doc.write_text("\n".join(md) + "\n")
    print(df.to_string(max_cols=12))
    print(f"\n-> {doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
