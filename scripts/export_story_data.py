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

import numpy as np
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


# ── 4. front-end 가 이미 가져간 몫과, 그 위에 딥러닝이 더하는 몫 ────
CLASSICAL = ["M01", "M02", "M03", "M04", "M05"]
DEEP = ["M06", "M08"]


def frontend_share(tag: str = "d1") -> pd.DataFrame:
    """잡음마다 «front-end 가 이미 한 몫» 과 «그 위에 더 붙는 몫» 을 가른다.

    보고서의 중심 틀(«공통 전처리가 이미 해결한 몫을 빼고 물어야 한다», 8 장)을
    잡음 축에서 직접 재는 표다. 본문은 이것을 산문으로만 말하고 그림이 없다.

    EXP-B 는 입력 10 dB 한 칸이므로 **한 동작점의 관측**이고, 칸이 잡음 7 종뿐
    이라 표본이 작다. 순위 상관은 그래서 정확 순열 검정으로 낸다(근사는 n=7 에서
    p 를 과소평가한다).
    """
    f = ROOT / "results" / tag / "exp_b" / "metrics.parquet"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_parquet(f)
    df = df[df.metric == METRIC]
    snr_in = int(df.snr_in_target.iloc[0])
    w = df.groupby(["cond", "method"])["value"].mean().unstack()
    rows = []
    for cond in w.index:
        fe = float(w.loc[cond, "M_FE"])
        cls = w.loc[cond, [c for c in CLASSICAL if c in w.columns]]
        dl = w.loc[cond, [c for c in DEEP if c in w.columns]]
        rows.append(dict(
            axis=AXIS_LABEL[tag], noise=COND_LABEL.get(cond, cond), snr_in_db=snr_in,
            frontend_db=round(fe, 3),
            classical_adds_db=round(float(cls.max()) - fe, 3), classical_best=cls.idxmax(),
            deep_adds_db=round(float(dl.max()) - fe, 3), deep_best=dl.idxmax()))
    return pd.DataFrame(rows).sort_values("frontend_db", ascending=False)


def _exact_spearman(x, y):
    """n 이 작을 때 정확 양측 p. scipy 의 t 근사는 n=7 에서 너무 낙관적이다."""
    from itertools import permutations

    from scipy import stats as st
    obs = st.spearmanr(x, y).statistic
    y = np.asarray(y)
    hit = sum(1 for perm in permutations(range(len(y)))
              if abs(st.spearmanr(x, y[list(perm)]).statistic) >= abs(obs) - 1e-12)
    n_perm = 1
    for k in range(2, len(y) + 1):
        n_perm *= k
    return obs, hit / n_perm


# ── 5. 평균이 숨기는 것 — 기록·구간마다의 산포 ──────────────────────
def record_spread(tag: str = "d1") -> pd.DataFrame:
    """`M08 − M_FE` 를 **기록·구간 단위**로 편다.

    5.8.2 는 이것의 평균만 싣는다. 평균 하나를 결론으로 읽으면 «20 dB 에서
    딥러닝이 진다» 가 «모든 기록에서 진다» 로 읽힌다 — 실제로는 부호가 갈린다.
    D-17 이 시연 구간을 고를 때 같은 산포에 걸렸다.
    """
    f = ROOT / "results" / tag / "exp_a" / "metrics.parquet"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_parquet(f)
    df = df[df.metric == METRIC]
    rows = []
    for s in sorted(df.snr_in_target.dropna().unique()):
        w = df[df.snr_in_target == s].pivot_table(
            index=["record", "seg"], columns="method", values="value", aggfunc="mean")
        if "M08" not in w or "M_FE" not in w:
            continue
        for (rec, seg), v in (w["M08"] - w["M_FE"]).dropna().items():
            rows.append(dict(axis=AXIS_LABEL[tag], snr_db=int(s), record=str(rec),
                             seg=int(seg), contrast_db=round(float(v), 3),
                             sign="딥러닝 우세" if v > 0 else "front-end 우세"))
    return pd.DataFrame(rows)


# ── 6. D0 의 판정 중 무엇이 D1 에서 살아남았나 ───────────────────────
SURVIVAL_METRICS = ["snr_imp_scaled", "rmse", "prdn", "cc", "gain_bias",
                    "ppv", "f1", "psd_logdist", "hr_err_bpm"]


def axis_survival(var: str = "M08", base: str = "M01") -> pd.DataFrame:
    """지표 아홉 개의 판정을 두 축에서 나란히 놓는다 (92_axis_gap 과 같은 자).

    `gain_bias` 는 이상값이 1 이라 **부호 있는 차이로 비교하면 안 된다** —
    `compare_methods` 가 `use_ideal` 로 |값 − 1| 을 쓴다(F-22). 그래서 여기서도
    직접 빼지 않고 그 함수를 부른다. 음수 = 이상값에 더 가깝다.
    """
    rows = []
    for tag in ("d0", "d1"):
        f = ROOT / "results" / tag / "exp_a" / "metrics.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        for met in SURVIVAL_METRICS:
            sub = df[df.metric == met]
            if sub.empty:
                continue
            t = compare_methods(sub, met, base)
            r = t[t.method == var]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append(dict(axis=AXIS_LABEL[tag], axis_tag=tag, metric=met,
                             delta=round(float(r.delta_mean), 5), p_holm=float(r.p_holm),
                             verdict="유의" if r.p_holm < 0.05 else "n.s."))
    return pd.DataFrame(rows)


# ── 7. 잡음마다 순위가 뒤집힌다 ─────────────────────────────────────
def rank_flip(tag: str = "d1") -> pd.DataFrame:
    """잡음 × 방법의 **순위**. 값이 아니라 순위로 보면 폭이 결론이 된다."""
    f = ROOT / "results" / tag / "exp_b" / "metrics.parquet"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_parquet(f)
    df = df[df.metric == METRIC]
    w = df.groupby(["cond", "method"])["value"].mean().unstack()
    cols = [c for c in ["M_FE", *CLASSICAL, *DEEP] if c in w.columns]
    rk = w[cols].rank(axis=1, ascending=False)
    rows = []
    for cond in rk.index:
        for m in cols:
            rows.append(dict(axis=AXIS_LABEL[tag], noise=COND_LABEL.get(cond, cond),
                             method=m, rank=int(rk.loc[cond, m]),
                             snr_imp_db=round(float(w.loc[cond, m]), 3)))
    return pd.DataFrame(rows)


# ── 8. 잡음 7 조건 판정판 (Q2 · `S17_noise_board`) ─────────────────
BOARD = {"M_FE": "공통 전처리만", "M04": "SWT wavelet", "M08": "딥러닝 Wavelet U-Net"}


def noise_board(tag: str = "d1") -> pd.DataFrame:
    """행 = 잡음, 열 = 방법 셋 + 판정. `noise_by_method()` 를 그대로 접는다."""
    nb = noise_by_method()
    nb = nb[(nb.axis_tag == tag) & nb.method.isin(BOARD)]
    w = nb.pivot(index="noise", columns="method", values="snr_imp_db")
    w = w[list(BOARD)].rename(columns=BOARD)
    gap = w["딥러닝 Wavelet U-Net"] - w["SWT wavelet"]
    w["판정"] = [f"{'딥러닝' if g > 0 else 'SWT'} +{abs(g):.1f} dB" for g in gap]
    w = w.loc[gap.sort_values(ascending=False).index]
    w.index.name = "잡음 종류"
    return w.reset_index()


# ── 9. 전원선 잡음의 PSD (Q1 층 4 · `S15_pli_psd`) ────────────────────
def pli_psd(snr: float = -5.0) -> pd.DataFrame:
    """원본 · 입력 · 처리 후 셋의 PSD [dB, 원본 최대 = 0]. 그림과 **같은 경로**.

    `make_slides.prepare` 를 그대로 부른다 — 여기서 잡음을 따로 만들면 그림과
    다른 구간이 된다(F-10). 기록 100 · 입력 -5 dB 한 구간이다.
    """
    import make_slides as ms
    from ecgdn.eval.spectral import welch_psd
    d = ms.prepare("mitdb", snr, "pli")
    fs = d["fs"]
    f, pr = welch_psd(d["x_raw"], fs)
    top = float(pr.max())
    col = {"원본 기록": d["x_raw"], "입력 (전원선 섞임)": d["y"]}
    col.update({f"{BOARD[m]} 출력": d["outs"][m] for m in BOARD})
    out = {"주파수 (Hz)": np.round(f, 3)}
    for k, v in col.items():
        _, p = welch_psd(v, fs)
        out[k] = np.round(10 * np.log10(np.maximum(p, top * 1e-12) / top), 2)
    return pd.DataFrame(out)


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
    _write(noise_board(), "08_noise_board_d1.csv")
    _write(pli_psd(), "09_pli_psd_d1.csv")

    fs = frontend_share()
    _write(fs, "04_frontend_share.csv")
    if len(fs) > 2:
        rho, pexact = _exact_spearman(fs.frontend_db.to_numpy(), fs.deep_adds_db.to_numpy())
        print(f"    front-end 몫 vs 딥러닝 추가분 — Spearman rho {rho:+.3f}, "
              f"정확 양측 p {pexact:.5f} (n={len(fs)})")

    rs = record_spread()
    _write(rs, "05_record_spread.csv")
    for s in (-5, 10, 20):
        c = rs[rs.snr_db == s].contrast_db
        if len(c):
            flip = int((c < 0).sum()) if c.mean() > 0 else int((c > 0).sum())
            print(f"    {s:>3} dB — 평균 {c.mean():+.2f} dB · 폭 {c.min():+.2f}~{c.max():+.2f}"
                  f" · 평균과 부호가 반대인 구간 {flip}/{len(c)}")

    sv = axis_survival()
    _write(sv, "06_axis_survival.csv")
    kept = sv.pivot(index="metric", columns="axis_tag", values="verdict")
    both = int(((kept.get("d0") == "유의") & (kept.get("d1") == "유의")).sum())
    print(f"    M08 − M01 — 지표 {len(kept)} 개 중 양축 유의 {both} 개")

    _write(rank_flip(), "07_rank_flip.csv")


if __name__ == "__main__":
    main()
