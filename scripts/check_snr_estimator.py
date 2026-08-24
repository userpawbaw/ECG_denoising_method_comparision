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


def load_segments(dur, n_rep, source, split="train"):
    if source == "mitdb":
        return real_clean_segments(n_rep, dur, fs=FS, split=split)
    return [synth_ecg(dur, fs=FS, hr_bpm=62 + 7 * rep, seed=300 + rep)
            for rep in range(n_rep)]


def ceiling_table(dur, n_rep, source, split="train"):
    """**잡음을 전혀 넣지 않은 신호**에 추정기를 돌린다 = 추정기의 천장.

    세 추정기 모두 "beat 가 반복되고 그 잔차가 잡음" 이라는 가정 위에 서 있다.
    실제 ECG 는 beat 마다 형태가 달라(호흡, 축 변화, 부정맥) 그 변동이 전부
    잡음으로 계산된다. 그래서 **잡음이 하나도 없어도 추정치가 무한대가 아니다.**

    그 유한한 값이 이 추정기가 그 데이터에서 보고할 수 있는 최대치다.
    참 SNR 이 그 위에 있으면 추정기는 천장을 반환하며, 그것을 실제 SNR 로
    읽으면 신호를 실제보다 나쁘게 판정하게 된다.

    F-13 의 -7.6 dB 편향이 이것으로 설명된다 (그 항목은 원인을 지목했지만
    크기를 재지는 않았다).
    """
    segs = load_segments(dur, n_rep, source, split)
    rows = []
    for rep, s in enumerate(segs):
        xr = getattr(s, "x_raw", s.x)
        for fe in (False, True):
            xx = apply_frontend(xr, s.fs) if fe else xr
            d = estimate_snr_all(xx, s.fs)
            rows.append(dict(rep=rep, fe=fe, name=getattr(s, "name", str(rep)),
                             **{k: d.get(k) for k in EST},
                             n_beats=d.get("n_beats")))
    return pd.DataFrame(rows)


def run(dur, snrs, n_rep, fe, source="synthetic", split="train"):
    segs = load_segments(dur, n_rep, source, split)
    rows = []
    for rep, s in enumerate(segs):
        for nname in list(NOISES) + ["mixed"]:
            for snr in snrs:
                g = rng("cal", nname, snr, rep)
                xr = getattr(s, "x_raw", s.x)
                if nname == "mixed":
                    n, _ = mixed_noise(len(xr), s.fs, g)
                else:
                    n = NOISES[nname](len(xr), s.fs, g)
                y, _, _ = mix_at_snr(xr, n, snr)      # 잡음은 원본에 (F-12)
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
    save_manifest(out, cfg=vars(args), sources=[
        "scripts/check_snr_estimator.py", "ecgdn/eval/snr_estimation.py"])

    b_off, b_on = bias_table(df, False), bias_table(df, True)

    ceil = ceiling_table(args.dur, args.n_rep, args.source)
    ceil_tbl = (ceil.groupby("fe")[EST].median().round(1)
                .rename(index={False: "front-end 미적용", True: "front-end 적용"})
                .rename(columns=SHORT))

    # 진단 지표가 실제로 편향을 잡아내는지
    diag = df[(df.fe == False) & (df.estimator == "snr_beat_residual_db")] \
        .groupby("noise").agg(mean_err=("err", "mean"), lag1=("lag1", "mean"),
                              artf=("artf", "mean"), warn=("warn_corr", "mean")).round(2)

    src_name = "합성 ECG" if args.source == "synthetic" else "MIT-BIH 실기록"
    from ecgdn.eval.snr_estimation import SNR_CEILING_BY_AXIS
    ceil_use = SNR_CEILING_BY_AXIS[tag]
    md = [f"# 04. SNR 추정기 교정표 ({tag.upper()}) — ground truth 없는 실측 신호용",
          "",
          "> 자동 생성: "
          f"`python scripts/check_snr_estimator.py --source {args.source}`",
          "",
          f"> **데이터축: {tag.upper()} — {src_name}**",
          "",
          "## 왜 필요한가",
          "",
          "실측 Arduino ECG 에는 clean reference 가 없어 SNR 을 직접 계산할 수 없다.",
          "따라서 **beat 반복성**을 이용한 추정기를 쓰는데, 이 추정기들은 잡음 종류에 따라",
          "체계적으로 치우친다. 그 치우침의 크기를 **참 SNR 을 아는 데이터로 먼저 측정**해 둔다.",
          "그래야 실측값을 정직하게 해석할 수 있다.",
          "",
          "> **교정표는 적용할 데이터와 같은 축에서 만들어야 한다** (F-13).",
          "> 합성으로 교정한 추정기를 실데이터에 쓰면 편향이 그대로 남는다.",
          "",
          f"조건: {src_name} {args.n_rep} 기록 × {args.dur:.0f} s, "
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
          "## 추정기의 천장 — 잡음을 전혀 넣지 않았을 때의 추정치",
          "",
          "세 추정기 모두 **\"beat 가 반복되고 그 잔차가 잡음\"** 이라는 가정 위에",
          "서 있다. 실제 ECG 는 beat 마다 형태가 달라(호흡, 축 변화, 부정맥) 그",
          "변동이 전부 잡음으로 계산된다. **그래서 잡음이 하나도 없어도 추정치가",
          "무한대가 아니다.** 그 유한한 값이 이 추정기가 이 데이터에서 보고할 수",
          "있는 최대치다.",
          "",
          ceil_tbl.to_markdown(),
          "",
          f"> 단위 dB, {args.n_rep} 기록의 중앙값. 참 SNR 이 이 값보다 높으면 추정기는",
          "> 천장을 반환하고, 그것을 실제 SNR 로 읽으면 **신호를 실제보다 나쁘게**",
          "> 판정하게 된다.",
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
          f"5. 추정치가 **{ceil_use:.0f} dB** 이상이면 `ceiling_warning` 이 뜬다 "
          f"(`SNR_CEILING_BY_AXIS[\"{tag}\"]`).",
          "   생리학적 beat 변동이 바닥을 만들기 때문에 그 이상은 구분되지 않는다.",
          f"   \"{ceil_use:.0f} dB 이상\" 으로만 해석한다. 이 임계값은 **위 '추정기의 천장' "
          "표에서 실측한 값**이고, 축마다 다르다 (F-18).",
          "",
          "### 실기록에 적용할 때 — 어느 추정기를 쓸 것인가",
          "",
          "실기록에서는 **(A),(B) 가 (C) 보다 나쁘다.** 셋 다 beat 반복성에 기대지만",
          "(A),(B) 는 그것이 유일한 근거이고, (C) 는 등전위 구간의 분산이라 beat 형태가",
          "달라도 흔들리지 않는다. 위 천장 표에서 D1 의 (A),(B) 가 D0 대비 8 dB 떨어지는",
          "동안 (C) 는 1.2 dB 만 떨어진다.",
          "",
          "그렇다고 (C) 가 항상 낫지는 않다. **잡음 종류가 갈림길이다.**",
          "",
          "| 잡음 성격 | 권장 | 이유 |",
          "|---|---|---|",
          "| 광대역 (awgn, emg) | **(C)** | D1 awgn 편향 -1.6 dB vs (A) -5.8 |",
          "| burst/임펄스 (motion, impulse) | (A),(B) 를 함께 | (C) 가 burst 를 등전위 구간에서 놓쳐 SNR 을 +4~+7 dB 과대평가 |",
          "| 협대역 (pli, bw) | front-end 로 먼저 제거 | 셋 다 -6 dB 이상 치우친다 |",
          "",
          "`artifact_beat_frac` 과 `beat_resid_lag1_corr` 로 어느 경우인지 판정한다.",
          "**단일 추정기로 실측 SNR 을 인용하지 않는다.**",
          "",
          "## 재현", "", "```bash",
          f"python scripts/check_snr_estimator.py --source {args.source}", "```",
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
