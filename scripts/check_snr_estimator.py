"""STEP 08: ground-truth 없는 SNR 추정기의 교정/검증 (docs/00_review.md B-3).

**목적**: 실측 Arduino 신호에 대해 "이게 정말 15 dB 인가?" 라고 물으려면,
추정기가 잡음 종류별로 어느 방향으로 얼마나 치우치는지 먼저 알아야 한다.

방법: 참 SNR 을 아는 합성 데이터에 각 잡음을 넣고 추정기를 돌려 bias 를 표로 만든다.
front-end(HPF) 적용 전/후를 모두 낸다 — 실무에서는 front-end 이후에 추정하기 때문.

산출: docs/04_snr_estimator_calibration.md
"""
import _bootstrap  # noqa: F401

import argparse

import numpy as np
import pandas as pd

from ecgdn.config import FS
from ecgdn.data.mixer import mix_at_snr
from ecgdn.data.noise import (awgn, baseline_synth, emg_synth, impulse,
                              mixed_noise, motion_synth, pli)
from ecgdn.data.sources import real_clean_segments
from ecgdn.data.synthetic import synth_ecg
from ecgdn.eval.engine import trim_guard
from ecgdn.eval.signal_metrics import snr_db
from ecgdn.eval.snr_estimation import estimate_snr_all
from ecgdn.methods.frontend import apply_frontend
from ecgdn.utils import ensure_dir, rng, save_manifest

NOISES = {
    "awgn": awgn, "pli": pli, "bw": baseline_synth,
    "ma(emg)": emg_synth, "em(motion)": motion_synth, "impulse": impulse,
}
EST = ["snr_beat_residual_db", "snr_hsc_db", "snr_isoelectric_db"]
SHORT = {"snr_beat_residual_db": "(A) beat-resid",
         "snr_hsc_db": "(B) half-sample",
         "snr_isoelectric_db": "(C) isoelectric"}


def run(dur, snrs, n_rep, fe, source="synthetic", split="train"):
    if source == "mitdb":
        segs = real_clean_segments(n_rep, dur, fs=FS, split=split)
    else:
        segs = [synth_ecg(dur, fs=FS, hr_bpm=62 + 7 * rep, seed=300 + rep)
                for rep in range(n_rep)]
    rows = []
    for rep, s in enumerate(segs):
        for nname in list(NOISES) + ["mixed"]:
            for snr in snrs:
                g = rng("cal", nname, snr, rep)
                if nname == "mixed":
                    n, _ = mixed_noise(len(s.x), s.fs, g)
                else:
                    n = NOISES[nname](len(s.x), s.fs, g)
                y, _, _ = mix_at_snr(s.x, n, snr)
                if fe:
                    # front-end 는 신호와 잡음을 모두 바꾼다. 따라서 참조 SNR 도
                    # 필터 후 기준으로 다시 계산해야 한다. 그렇지 않으면
                    # 'front-end 가 잡음을 제거한 효과' 가 '추정기 편향' 으로 잘못 집계된다.
                    y_fe = apply_frontend(y, s.fs)
                    x_fe = apply_frontend(s.x, s.fs)
                    sl = trim_guard(len(y_fe), s.fs)
                    true_snr = snr_db(x_fe[sl], y_fe[sl] - x_fe[sl])
                    y = y_fe
                else:
                    true_snr = float(snr)
                est = estimate_snr_all(y, s.fs)
                for k in EST:
                    rows.append(dict(rep=rep, noise=nname, true_snr=snr,
                                     true_snr_eff=true_snr, fe=fe,
                                     estimator=k, est=est[k], err=est[k] - true_snr,
                                     lag1=est["beat_resid_lag1_corr"],
                                     artf=est["artifact_beat_frac"],
                                     warn_corr=est["correlated_noise_warning"]))
    return pd.DataFrame(rows)


CEILING = 20.0


def bias_table(df, fe):
    """포화 영역(참 SNR >= CEILING)은 제외한 편향표 + 포화 비율."""
    d = df[(df.fe == fe) & (df.true_snr_eff < CEILING)]
    t = d.pivot_table(index="noise", columns="estimator", values="err", aggfunc="mean")
    t = t.reindex(columns=EST)
    t.columns = [SHORT[c] for c in t.columns]
    sat = (df[df.fe == fe].groupby("noise")["true_snr_eff"]
           .apply(lambda v: float((v >= CEILING).mean())))
    t["포화비율"] = sat.reindex(t.index).round(2)
    return t.round(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dur", type=float, default=120.0)
    ap.add_argument("--n-rep", type=int, default=3)
    ap.add_argument("--snrs", type=int, nargs="+", default=[0, 5, 10, 15, 20])
    ap.add_argument("--source", default="synthetic", choices=("synthetic", "mitdb"),
                    help="synthetic 이 기본 — D0 결과(보고서 인용값)와의 연속성을 지킨다")
    args = ap.parse_args()

    tag = "d0" if args.source == "synthetic" else "d1"
    suffix = "" if tag == "d0" else "_d1"

    df = pd.concat([run(args.dur, args.snrs, args.n_rep, fe, args.source)
                    for fe in (False, True)], ignore_index=True)
    out = ensure_dir(f"results/{tag}/snr_calibration")
    df.to_csv(out / "raw.csv", index=False)
    save_manifest(out, cfg=vars(args))

    b_off, b_on = bias_table(df, False), bias_table(df, True)

    # 진단 지표가 실제로 편향을 잡아내는지
    diag = df[(df.fe == False) & (df.estimator == "snr_beat_residual_db")] \
        .groupby("noise").agg(mean_err=("err", "mean"), lag1=("lag1", "mean"),
                              artf=("artf", "mean"), warn=("warn_corr", "mean")).round(2)

    md = ["# 04. SNR 추정기 교정표 (ground truth 없는 실측 신호용)",
          "",
          "> 자동 생성: `python scripts/check_snr_estimator.py`",
          "",
          "## 왜 필요한가",
          "",
          "실측 Arduino ECG 에는 clean reference 가 없어 SNR 을 직접 계산할 수 없다.",
          "따라서 **beat 반복성**을 이용한 추정기를 쓰는데, 이 추정기들은 잡음 종류에 따라",
          "체계적으로 치우친다. 그 치우침의 크기를 **참 SNR 을 아는 합성 데이터로 먼저 측정**해 둔다.",
          "그래야 실측값 \"약 15 dB\" 를 정직하게 해석할 수 있다.",
          "",
          f"조건: 합성 ECG {args.n_rep} 기록 × {args.dur:.0f} s, "
          f"참 SNR ∈ {args.snrs} dB, 잡음 {len(NOISES)+1} 종.",
          "",
          "## 추정기",
          "",
          "| 기호 | 방법 | 핵심 가정 |",
          "|---|---|---|",
          "| (A) | beat 평균 잔차 | beat 형태 반복 + **잡음이 beat 간 무상관** |",
          "| (B) | half-sample consistency (홀/짝 beat 평균의 상관) | 상동 (잡음 스펙트럼 가정은 없음) |",
          "| (C) | 등전위(TP) 구간 분산 | TP 구간이 평탄 + **잡음이 정상(stationary)** |",
          "",
          "> **필수 전처리**: 4배 업샘플 후 template 교차상관으로 **sub-sample 정렬**한다.",
          "> fs=250 Hz 에서 1 샘플(4 ms) 정렬 오차만으로 추정치가 6 dB 이상 낮아진다(실측).",
          "",
          "## 편향 (추정치 − 참값), 단위 dB — **front-end 미적용**",
          "",
          f"> 참 SNR 이 {CEILING:.0f} dB 이상인 조건은 추정기의 생리학적 포화 영역이므로 "
          "편향 계산에서 제외했다. `포화비율` 열이 그 비중이다 (1.0 이면 그 잡음 조건에서는 "
          "front-end 가 잡음을 거의 다 제거해 SNR 추정 자체가 의미를 잃는다는 뜻).",
          "",
          b_off.to_markdown(),
          "",
          "## 편향 — **front-end(0.5 Hz HPF + 100 Hz LPF + 자동 notch) 적용 후**",
          "",
          "> 주의: front-end 는 잡음을 실제로 제거하므로, 여기서는 참조 SNR 을 "
          "**필터 후 기준으로 다시 계산**해서 뺐다. 그렇게 하지 않으면 "
          "'front-end 의 성능' 이 '추정기 편향' 으로 잘못 집계된다.",
          "",
          b_on.to_markdown(),
          "",
          "## 진단 지표가 편향을 잡아내는가 (front-end 미적용, 추정기 A)",
          "",
          diag.to_markdown(),
          "",
          "## 사용 지침 (실측 데이터에 적용할 때)",
          "",
          "1. **front-end 를 먼저 적용한 뒤 추정한다.** baseline/motion 계열의 상관 잡음이 제거되어",
          "   (A),(B) 의 편향이 크게 줄어든다.",
          "2. 세 추정치를 모두 보고한다. `snr_spread_db` 가 5 dB 를 넘으면 단일 값으로 인용하지 않는다.",
          "3. `beat_resid_lag1_corr > 0.15` → 잡음이 beat 시간척도에서 상관을 가짐.",
          "   (A),(B) 는 **낙관적**이므로 (C) 를 함께 본다.",
          "4. `artifact_beat_frac` 이 크면 burst/임펄스형 잡음이다. 이때는 SNR 을 단일 스칼라로",
          "   말하는 것 자체가 부정확하므로 **`snr_beat_p50_db` 와 `snr_beat_p10_db` 를 함께** 적는다.",
          "5. 추정치가 22 dB 이상이면 `ceiling_warning` 이 뜬다. 생리학적 beat 변동이 바닥을 만들기",
          "   때문에 그 이상은 구분되지 않는다. \"22 dB 이상\" 으로만 해석한다.",
          "",
          "## 재현", "", "```bash", "python scripts/check_snr_estimator.py", "```",
          ]
    doc = ensure_dir("docs") / f"04_snr_estimator_calibration{suffix}.md"
    doc.write_text("\n".join(md) + "\n")

    print("=== bias (front-end OFF) ==="); print(b_off.to_string())
    print("\n=== bias (front-end ON) ==="); print(b_on.to_string())
    print("\n=== diagnostics ==="); print(diag.to_string())
    print(f"\n-> {doc}")

    # DoD: 포화 영역(참 SNR >= 20 dB) 을 뺀 AWGN 조건에서 |bias| < 1.5 dB
    aw = df[(df.noise == "awgn") & (df.true_snr < 20)
            & (df.estimator != "snr_isoelectric_db")]
    v = float(aw.err.abs().mean())
    ok = v < 1.5
    print(f"\n[{'PASS' if ok else 'FAIL'}] AWGN (true SNR < 20 dB) mean |bias| = {v:.2f} dB (< 1.5)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
