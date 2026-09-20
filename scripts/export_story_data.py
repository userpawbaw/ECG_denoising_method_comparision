"""발표·시연용 **외부 시각화 재료**를 산출물에서 직접 뽑는다.

    python scripts/export_story_data.py

산출: `results/story/*.csv` — Flourish 등 저장소 밖 도구에 올릴 표다.
설계와 쓰임은 `docs/94_presentation.md`.

왜 스크립트로 두나
------------------
저장소 밖 도구에 올리는 값은 **손으로 옮기기 가장 쉬운 자리**다. 한 번
옮겨 적으면 그 뒤로는 원본이 바뀌어도 아무도 모른다 — `docs/17_checklists.md`
§6 이 막으려는 것이 정확히 이것이고, F-26 은 같은 문서 안에서 9 배 어긋난
사례다. 그래서 CSV 는 전부 `results/**/metrics.parquet` 에서 계산하고,
검정은 **보고서가 쓰는 함수를 그대로** 부른다. 같은 값이 나오지 않으면
둘 중 하나가 틀린 것이고, 그 사실이 드러나야 한다.

`analyze_loss_by_noise.table()` 을 재구현하지 않고 그대로 부르는 것도 같은
이유다 — Holm 보정의 묶음 범위(한 축·한 SNR 안의 잡음 7 종)가 여기서 달라지면
그림과 보고서가 다른 것을 말하게 된다.
"""
import _bootstrap  # noqa: F401

from pathlib import Path

import pandas as pd

from analyze_loss_by_noise import table
from ecgdn.eval.stats import compare_methods

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "story"
METRIC = "snr_imp_scaled"

AXIS_LABEL = {"d0": "D0 합성", "d1": "D1 실데이터"}
COND_LABEL = {
    "mixed": "혼합", "impulse": "임펄스", "pli": "전원선 60Hz",
    "bw_synth": "기저선 변동", "ma_synth": "근전도(MA)",
    "em_synth": "전극 움직임(EM)", "awgn": "백색잡음",
}
# 히트맵의 행 순서. 값 크기가 아니라 **부호가 뒤집히는 SNR** 순으로 둔다 —
# 그래야 «경계가 가로로 서 있다» 가 그림에서 계단으로 읽힌다.
NOISE_ORDER = ["전극 움직임(EM)", "임펄스", "혼합", "백색잡음",
               "근전도(MA)", "기저선 변동", "전원선 60Hz"]


def crossover() -> pd.DataFrame:
    """딥러닝이 공통 front-end 위에 더하는 것 (보고서 5.8.2 · `S5_crossover`).

    기준선은 `M_FE` 다 — 참조와 같은 대역이라 '딥러닝이 front-end 위에
    무엇을 더하는가' 를 재는 올바른 비교다 (5.8.6).
    """
    rows = []
    for tag in ("d0", "d1"):
        f = ROOT / "results" / tag / "exp_a" / "metrics.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        for s in sorted(df.snr_in_target.dropna().unique()):
            t = compare_methods(df[df.snr_in_target == s], METRIC, "M_FE")
            for m in ("M06", "M08"):
                r = t[t.method == m]
                if r.empty:
                    continue
                r = r.iloc[0]
                rows.append(dict(axis=AXIS_LABEL[tag], axis_tag=tag, method=m,
                                 snr_db=int(s), gain_db=round(float(r.delta_mean), 3),
                                 p_holm=float(r.p_holm),
                                 verdict="유의" if r.p_holm < 0.05 else "n.s."))
    return pd.DataFrame(rows)


def loss_grid() -> pd.DataFrame:
    """`L6` 의 이득이 잡음 종류별로도 유지되는가 (5.8.10 · EXP-G · `S10`)."""
    rows = []
    for tag in ("d0", "d1"):
        f = ROOT / "results" / tag / "exp_g" / "metrics.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        df = df[df.metric == METRIC]
        for base, var in (("M06", "M06L6"), ("M08", "M08L6")):
            for _, r in table(df, base, var).iterrows():
                verdict = ("이득" if r.delta > 0 else "손해") if r.p_holm < 0.05 else "n.s."
                rows.append(dict(axis=AXIS_LABEL[tag], axis_tag=tag, model=base,
                                 noise=COND_LABEL.get(r.cond, r.cond),
                                 snr_db=int(r.snr), delta_db=round(float(r.delta), 3),
                                 p_holm=float(r.p_holm), verdict=verdict))
    return pd.DataFrame(rows)


def noise_by_method() -> pd.DataFrame:
    """잡음 종류가 답을 바꾼다 (5.2 · `S4_noise_*`). EXP-B 는 10 dB 한 칸이다."""
    rows = []
    for tag in ("d0", "d1"):
        f = ROOT / "results" / tag / "exp_b" / "metrics.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        df = df[df.metric == METRIC]
        for (cond, method), v in df.groupby(["cond", "method"])["value"].mean().items():
            rows.append(dict(axis=AXIS_LABEL[tag], axis_tag=tag,
                             noise=COND_LABEL.get(cond, cond), method=method,
                             snr_in_db=int(df.snr_in_target.iloc[0]),
                             snr_imp_db=round(float(v), 3)))
    return pd.DataFrame(rows)


def _write(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)
    print(f"  {name}  ({len(df)} 행)")


def main() -> None:
    print(f"results/story/ 에 쓴다 — 근거는 results/**/metrics.parquet")

    cx = crossover()
    _write(cx, "01_crossover.csv")
    # 발표에서 바로 올리는 판: 행 = 입력 SNR, 열 = 축 (M08)
    m08 = cx[cx.method == "M08"].pivot(index="snr_db", columns="axis",
                                       values="gain_db")
    m08.index.name = "입력 SNR (dB)"
    _write(m08.reset_index(), "01_crossover_wide.csv")

    lg = loss_grid()
    _write(lg, "02_loss_grid.csv")
    counts = lg.verdict.value_counts().to_dict()
    print(f"    전 격자 {len(lg)} 칸 — {counts}   (보고서 5.8.10 과 대조할 것)")
    d1 = lg[(lg.axis_tag == "d1") & (lg.model == "M06")].copy()
    d1["noise"] = pd.Categorical(d1.noise, NOISE_ORDER, ordered=True)
    d1 = d1.sort_values(["noise", "snr_db"])
    heat = d1[["snr_db", "noise", "delta_db", "verdict"]].copy()
    heat["snr_db"] = heat.snr_db.astype(str) + " dB"
    heat.columns = ["입력 SNR", "잡음 종류", "L6 이득 (dB)", "판정"]
    _write(heat, "02_loss_grid_d1_m06_heatmap.csv")

    _write(noise_by_method(), "03_noise_by_method.csv")


if __name__ == "__main__":
    main()
