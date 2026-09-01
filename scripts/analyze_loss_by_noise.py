"""EXP-G 분석 — **`L6` 의 이득이 잡음 종류별로도 유지되는가.**

    python scripts/analyze_loss_by_noise.py            # 두 축
    python scripts/analyze_loss_by_noise.py --axis d1

산출: `docs/11_loss_by_noise.md`

왜 필요한가
----------
5.8.9 는 `L6`(clean 보존 항)이 `M06` 에서 **양축 유의**라고 말한다(D0 +2.426,
D1 +1.420). 그런데 그 근거인 `abl_loss` 는 `conditions: [mixed]` 하나만 돌았다.
혼합 잡음은 여러 성분의 평균이라, **어떤 성분에서 벌고 어떤 성분에서 잃는지**
평균 하나로는 알 수 없다.

시연용으로 뽑은 구간 하나에서 부호가 갈렸다 — D1 임펄스 10 dB 에서 `M06L6`
18.64 vs `M06` 13.43(+5.2 dB), PLI 10 dB 에서 9.13 vs 16.12(**−7.0 dB**).
구간 하나라 결론이 못 되지만 크기가 우연으로 보기 어려웠다. EXP-G 가 그것을
기록 22 개 × 구간 2 개로 다시 잰다.

읽는 법
------
* 짝지어 비교한다 — 같은 기록·같은 구간·같은 잡음 실현에서 `L1` 과 `L6` 을
  맞댄다. `snr_imp_scaled` 는 floor 가 없는 지표라 분해능 열이 NaN 이다
  (**누락이 아니라 부재**).
* Holm 보정은 **한 축 · 한 SNR 안의 잡음 7 종**에 대해 건다. 축과 SNR 을
  섞어 보정하면 "어느 잡음에서" 라는 질문 자체가 흐려진다.
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ecgdn.eval.stats import holm, paired_wilcoxon, rank_biserial

ROOT = Path(__file__).resolve().parents[1]
METRIC = "snr_imp_scaled"
PAIRS = [("M06", "M06L6"), ("M08", "M08L6")]
COND_LABEL = {
    "mixed": "혼합", "impulse": "임펄스", "pli": "전원선 60 Hz",
    "bw_synth": "기저선 변동", "ma_synth": "근전도(MA)",
    "em_synth": "전극 움직임(EM)", "awgn": "백색잡음",
}


def table(df: pd.DataFrame, base: str, var: str) -> pd.DataFrame:
    """(cond, snr) 칸마다 `var − base` 의 짝지은 검정."""
    rows = []
    for snr in sorted(df.snr_in_target.unique()):
        block = []
        for cond in df.cond.unique():
            sub = df[(df.cond == cond) & (df.snr_in_target == snr)]
            wide = sub.pivot_table(index=["record", "seg"], columns="method",
                                   values="value", aggfunc="mean")
            if base not in wide or var not in wide:
                continue
            pair = wide[[var, base]].dropna()
            if pair.empty:
                continue
            a, b = pair[var].to_numpy(), pair[base].to_numpy()
            stat, p = paired_wilcoxon(a, b)
            block.append(dict(cond=cond, snr=float(snr), n=len(a),
                              base_mean=float(np.mean(b)), var_mean=float(np.mean(a)),
                              delta=float(np.mean(a - b)), p=p,
                              effect_r=rank_biserial(a, b)))
        if block:
            bl = pd.DataFrame(block)
            # 보정은 **한 축·한 SNR 안의 잡음 7 종**에 대해서만 건다
            bl["p_holm"] = holm(bl["p"].to_numpy())
            rows.append(bl)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def render(tag: str, out: list[str]) -> bool:
    f = ROOT / "results" / tag / "exp_g" / "metrics.parquet"
    if not f.exists():
        out.append(f"## {tag}\n\n`{f.relative_to(ROOT)}` 가 없다 — EXP-G 미실행.\n")
        return False
    df = pd.read_parquet(f)
    df = df[df.metric == METRIC]
    out.append(f"## {tag}\n")
    any_row = False
    for base, var in PAIRS:
        t = table(df, base, var)
        if t.empty:
            out.append(f"### {var} − {base}\n\n두 방법이 함께 있는 칸이 없다.\n")
            continue
        any_row = True
        out.append(f"### {var} − {base}  (`{METRIC}`, dB)\n")
        out.append("| 잡음 | " + " | ".join(
            f"{s:g} dB" for s in sorted(t.snr.unique())) + " |")
        out.append("|---|" + "---|" * t.snr.nunique())
        for cond in COND_LABEL:
            if cond not in set(t.cond):
                continue
            cells = []
            for snr in sorted(t.snr.unique()):
                r = t[(t.cond == cond) & (t.snr == snr)]
                if r.empty:
                    cells.append("—"); continue
                r = r.iloc[0]
                mark = "**" if r.p_holm < 0.05 else ""
                sig = "" if r.p_holm < 0.05 else " n.s."
                cells.append(f"{mark}{r.delta:+.2f}{mark}{sig} "
                             f"(p={r.p_holm:.3g}, r={r.effect_r:+.2f})")
            out.append(f"| {COND_LABEL[cond]} | " + " | ".join(cells) + " |")
        out.append("")
        win = t[(t.p_holm < 0.05) & (t.delta > 0)]
        lose = t[(t.p_holm < 0.05) & (t.delta < 0)]
        out.append(f"유의하게 **이득** {len(win)} 칸, **손해** {len(lose)} 칸, "
                   f"나머지 {len(t) - len(win) - len(lose)} 칸 n.s. "
                   f"(전체 {len(t)} 칸, 기록 단위 n={int(t.n.iloc[0])})\n")
        if len(lose):
            worst = lose.loc[lose.delta.idxmin()]
            out.append(f"가장 큰 손해: **{COND_LABEL[worst.cond]} {worst.snr:g} dB "
                       f"{worst.delta:+.2f} dB** (p={worst.p_holm:.3g})\n")
    return any_row


def _grid_line() -> str:
    """격자 크기를 `configs/exp_g.yaml` 에서 읽어 쓴다.

    처음에는 "7 종 × 3 단계" 를 문자열로 박아 뒀는데, 격자를 7 단계로 넓힌 뒤에도
    그 문장만 3 단계로 남았다. **자동 생성 문서 안의 손으로 쓴 숫자**가 가장
    먼저 낡는다.
    """
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "exp_g.yaml").read_text())
    d = cfg.get("data", {}) or {}
    conds = len(d.get("conditions", []) or [])
    grid = d.get("snr_grid", []) or []
    if not conds or not grid:
        return "잡음 여러 종 × 여러 입력 SNR"      # 못 읽으면 숫자를 지어내지 않는다
    return (f"잡음 {conds} 종 × 입력 SNR {len(grid)} 단계"
            f"({min(grid):g}~{max(grid):g} dB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", nargs="*", default=["d0", "d1"])
    ap.add_argument("--out", default="docs/11_loss_by_noise.md")
    args = ap.parse_args()

    out = [
        "# 11. 손실 변형의 잡음 종류별 효과 (EXP-G)",
        "",
        "> 이 문서는 `scripts/analyze_loss_by_noise.py` 가 만든다. **직접 고치지 말 것.**",
        "> 근거: `results/{d0,d1}/exp_g/metrics.parquet`, 설정 `configs/exp_g.yaml`.",
        "",
        "`abl_loss` 는 혼합 잡음 하나만 다룬다. 혼합은 여러 성분의 평균이라",
        "**어디서 벌고 어디서 잃는지**를 평균 하나로는 알 수 없다. 여기서",
        f"{_grid_line()} 로 다시 잰다.",
        "",
        "짝지은 비교다 — 같은 기록·같은 구간·같은 잡음 실현에서 `L1` 과 `L6` 을",
        "맞댄다. Holm 보정은 **한 축·한 SNR 안의 잡음 7 종**에 걸었다.",
        "`snr_imp_scaled` 는 floor 가 정의되지 않은 지표라 분해능 열이 없다 —",
        "**누락이 아니라 부재**다.",
        "",
    ]
    got = False
    for tag in args.axis:
        got |= render(tag, out)
        out.append("")
    p = ROOT / args.out
    p.write_text("\n".join(out))
    print(f"-> {p.relative_to(ROOT)}")
    if not got:
        print("[warn] 비교할 칸이 하나도 없다 — EXP-G 가 아직 안 끝났을 수 있다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
