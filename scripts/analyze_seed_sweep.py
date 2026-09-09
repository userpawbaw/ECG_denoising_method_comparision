"""`run_seed_sweep.py` 가 만든 표를 읽어 **판정**한다.

    python scripts/analyze_seed_sweep.py results/ext/structure
    python scripts/analyze_seed_sweep.py results/ext/structure --base m06_l1

네 가지를 낸다.

1. **캘리브레이션** — 외부 환경의 `m06_l1` seed 0/1/2 가 이 저장소의 CPU 값
   (3.437 / 4.864 / 4.493)을 재현하는가. 재현하지 않으면 두 집합은 **다른 계보**라
   한 표에 섞을 수 없다. 그래도 외부 집합 **안에서의** 비교는 유효하다.

2. **산포** — arm 별 n · 평균 · 표본 sd, 그리고 sd 의 95 % 신뢰구간(χ²).
   F-40 은 n=3 이라 그 구간이 0.39~4.66 dB 였다.

3. **짝지은 비교** — base arm 대비 seed 를 맞춰 뺀 차이. 짝짓기가 얼마나 이득인지
   상관 ρ 와 `sd_paired / (sd·√2)` 로 함께 보여 준다.

4. **필요한 판 수** — 측정된 sd 로 다시 계산한다. F-40 이 쓴 0.741 은 추정치였다.

`best_metric` 과 `fixed_snr_imp_scaled` 를 **둘 다** 낸다. 앞은 seed 마다 다른 val
잡음 뽑기 위에서 고른 값이고, 뒤는 seed 와 무관한 고정 뽑기로 다시 잰 값이다.
두 산포의 차이가 「모델이 달라서」와 「val 뽑기가 운이 좋아서」를 가른다.
"""
import _bootstrap  # noqa: F401

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent

# 이 저장소(CPU · fp32)에서 나온 값. 캘리브레이션의 기준이다.
LOCAL_CPU = {("m06_l1", 0): 3.437, ("m06_l1", 1): 4.864, ("m06_l1", 2): 4.493,
             ("m09_l1", 0): 3.440, ("m09_l1", 1): 4.288, ("m09_l1", 2): 4.169}

METRICS = [("best_metric", "best val (seed 별 잡음 뽑기)"),
           ("fixed_snr_imp_scaled", "고정 평가 (공통 잡음 뽑기)")]


def sd_ci(sd: float, n: int, conf: float = 0.95) -> tuple[float, float]:
    """표본 표준편차의 신뢰구간 (χ² 기반). n 이 작으면 **매우** 넓다."""
    if n < 2:
        return (float("nan"), float("nan"))
    df = n - 1
    a = (1 - conf) / 2
    lo = sd * math.sqrt(df / stats.chi2.ppf(1 - a, df))
    hi = sd * math.sqrt(df / stats.chi2.ppf(a, df))
    return lo, hi


def need_n(sd: float, delta: float, power: float = 0.80, alpha: float = 0.05) -> int:
    """두 팔을 가르는 데 필요한 팔당 판 수 (정규 근사)."""
    if sd <= 0 or delta <= 0:
        return 0
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return max(2, math.ceil(2 * (z * sd / delta) ** 2))


def need_n_paired(sd_d: float, delta: float, power: float = 0.80,
                  alpha: float = 0.05) -> int:
    if sd_d <= 0 or delta <= 0:
        return 0
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return max(2, math.ceil((z * sd_d / delta) ** 2))


def load(dirpath: Path) -> list[dict]:
    p = dirpath / "sweep.csv" if dirpath.is_dir() else dirpath
    if not p.exists():
        raise SystemExit(f"[오류] {p} 가 없다")
    with p.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("seed", "n_epochs", "best_epoch", "n_params"):
            if r.get(k):
                r[k] = int(float(r[k]))
        for k, _ in METRICS:
            r[k] = float(r[k]) if r.get(k) not in (None, "") else float("nan")
    return rows


def section(title: str) -> None:
    print(f"\n{'='*72}\n{title}\n{'='*72}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="results/ext/<stage> 또는 sweep.csv 경로")
    ap.add_argument("--base", default=None, help="짝지어 비교할 기준 arm")
    ap.add_argument("--targets", default="1.4,0.5,0.25",
                    help="필요 판 수를 계산할 차이 [dB]")
    args = ap.parse_args()

    rows = load(Path(args.path))
    if not rows:
        raise SystemExit("[오류] 표가 비어 있다")

    by_arm: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        by_arm[r["arm"]][r["seed"]] = r
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    base = args.base or arms[0]

    env = {k[4:]: rows[0][k] for k in rows[0] if k.startswith("env_")}
    section("환경")
    for k in ("device", "amp", "gpu", "torch", "cuda", "git_commit"):
        if env.get(k):
            print(f"  {k:<12} {env[k]}")
    spe = [float(r["sec_per_epoch"]) for r in rows if r.get("sec_per_epoch")]
    if spe:
        r = 66.0 / max(float(np.mean(spe)), 1e-9)
        how = f"{r:.1f} 배 빠르다" if r >= 1 else f"{1/r:.1f} 배 느리다"
        print(f"  {'epoch 당':<12} {np.mean(spe):.1f} s  "
              f"(이 저장소 CPU 는 약 66 s — {how})")

    # ---------------------------------------------------------------- 1. 캘리브레이션
    cal = [(f"{a} s{s}", by_arm[a][s]["best_metric"], v)
           for (a, s), v in sorted(LOCAL_CPU.items())
           if s in by_arm.get(a, {})]
    if cal:
        section("1. 캘리브레이션 — 이 환경이 CPU 결과를 재현하는가")
        print(f"  {'판':<12}  {'외부':>8}  {'CPU(fp32)':>10}  {'차이':>8}")
        for nm, ext, cpu in cal:
            print(f"  {nm:<12}  {ext:>8.3f}  {cpu:>10.3f}  {ext-cpu:>+8.3f}")
        d = np.array([e - c for _, e, c in cal])
        print(f"\n  평균 차이 {d.mean():+.3f} dB, 최대 |차이| {np.abs(d).max():.3f} dB")
        if np.abs(d).max() < 0.05:
            print("  → 재현된다. 두 집합을 **한 표에 합쳐도** 된다.")
        else:
            print("  → 재현되지 않는다. 커널·수치정밀도가 다르므로 두 집합은")
            print("     **다른 계보**다. 외부 집합 안에서만 비교한다 (그래도 유효하다).")

    # ---------------------------------------------------------------- 2. 산포
    for key, label in METRICS:
        vals = {a: np.array([by_arm[a][s][key] for s in sorted(by_arm[a])])
                for a in arms}
        if all(np.all(np.isnan(v)) for v in vals.values()):
            continue
        section(f"2. arm 별 산포 — {label}")
        print(f"  {'arm':<18} {'n':>3} {'평균':>8} {'sd':>7} "
              f"{'sd 95% 구간':>18} {'최소':>8} {'최대':>8}")
        for a in arms:
            v = vals[a][~np.isnan(vals[a])]
            if v.size == 0:
                continue
            sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")
            lo, hi = sd_ci(sd, v.size)
            ci = f"{lo:.3f} ~ {hi:.3f}" if v.size > 1 else "—"
            print(f"  {a:<18} {v.size:>3} {v.mean():>8.3f} {sd:>7.3f} "
                  f"{ci:>18} {v.min():>8.3f} {v.max():>8.3f}")

    # 두 지표의 산포 대조 — val 뽑기 운의 몫
    b = by_arm.get(base, {})
    bv = np.array([b[s]["best_metric"] for s in sorted(b)])
    fv = np.array([b[s].get("fixed_snr_imp_scaled", np.nan) for s in sorted(b)])
    if bv.size > 2 and not np.all(np.isnan(fv)):
        m = ~np.isnan(fv)
        if m.sum() > 2:
            sb, sf = bv[m].std(ddof=1), fv[m].std(ddof=1)
            section("2b. 산포를 가른다 — 모델이 달라서 vs val 뽑기가 달라서")
            print(f"  best val   sd = {sb:.3f} dB   (seed 마다 다른 val 잡음)")
            print(f"  고정 평가  sd = {sf:.3f} dB   (모든 seed 공통 val 잡음)")
            print(f"\n  고정 평가 산포가 {'작다' if sf < sb else '작지 않다'} — "
                  f"best val 산포의 {sf/max(sb,1e-9):.0%}.")
            if sf < sb:
                print("  → 남은 몫이 **val 뽑기 운**이다. 「모델이 얼마나 흔들리나」는")
                print("     고정 평가 쪽 숫자로 말해야 한다.")
            else:
                print("  → val 뽑기는 산포의 원인이 아니다. 흔들리는 것은 학습 자체다.")

    # ---------------------------------------------------------------- 3. 짝지은 비교
    others = [a for a in arms if a != base]
    if others:
        for key, label in METRICS:
            section(f"3. {base} 대비 짝지은 비교 — {label}")
            print(f"  {'arm':<18} {'n':>3} {'Δ평균':>8} {'sd(Δ)':>8} "
                  f"{'ρ':>6} {'짝 이득':>8} {'p(paired t)':>12}  판정")
            for a in others:
                common = sorted(set(by_arm[base]) & set(by_arm[a]))
                x = np.array([by_arm[base][s][key] for s in common])
                y = np.array([by_arm[a][s][key] for s in common])
                m = ~(np.isnan(x) | np.isnan(y))
                x, y = x[m], y[m]
                if x.size < 2:
                    print(f"  {a:<18} {x.size:>3}  (판이 모자라다)")
                    continue
                d = y - x
                sd_d = float(d.std(ddof=1))
                sd_pool = math.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
                rho = float(np.corrcoef(x, y)[0, 1]) if x.size > 2 else float("nan")
                gain = sd_d / (sd_pool * math.sqrt(2)) if sd_pool > 0 else float("nan")
                t = stats.ttest_rel(y, x)
                verdict = ("유의" if t.pvalue < 0.05 else
                           f"못 잼 (|Δ|<{2.0*sd_d/math.sqrt(x.size):.2f} 해상도)")
                print(f"  {a:<18} {x.size:>3} {d.mean():>+8.3f} {sd_d:>8.3f} "
                      f"{rho:>6.2f} {gain:>8.2f} {t.pvalue:>12.2e}  {verdict}")
            print("\n  ρ = 두 arm 이 seed 를 따라 함께 움직이는 정도. "
                  "짝 이득 < 1 이면 짝짓기가 이득이다\n"
                  "  (1.00 이면 짝지어도 독립 표본과 같다).")

    # ---------------------------------------------------------------- 4. 필요한 판 수
    key = "fixed_snr_imp_scaled"
    bb = np.array([by_arm[base][s].get(key, np.nan) for s in sorted(by_arm[base])])
    bb = bb[~np.isnan(bb)]
    if bb.size < 2:
        bb = np.array([by_arm[base][s]["best_metric"] for s in sorted(by_arm[base])])
        key = "best_metric"
    if bb.size >= 2:
        sd = float(bb.std(ddof=1))
        section(f"4. 필요한 판 수 — 측정된 sd = {sd:.3f} dB ({key}, n={bb.size})")
        sd_d_map = {}
        for a in others:
            common = sorted(set(by_arm[base]) & set(by_arm[a]))
            x = np.array([by_arm[base][s].get(key, np.nan) for s in common])
            y = np.array([by_arm[a][s].get(key, np.nan) for s in common])
            m = ~(np.isnan(x) | np.isnan(y))
            if m.sum() > 1:
                sd_d_map[a] = float((y[m] - x[m]).std(ddof=1))
        sd_d = float(np.median(list(sd_d_map.values()))) if sd_d_map else sd * math.sqrt(2)
        print(f"  짝지은 차이의 sd(중앙값) = {sd_d:.3f} dB\n")
        print(f"  {'재려는 Δ':>10}  {'독립 표본':>10}  {'짝지은 설계':>12}")
        for t in [float(x) for x in args.targets.split(",")]:
            print(f"  {t:>10.2f}  {need_n(sd, t):>10d}  {need_n_paired(sd_d, t):>12d}")
        if spe:
            per = np.mean(spe) * 45 / 60      # 45 epoch 가정, 분
            print(f"\n  (이 환경 한 판 ≈ {per:.1f} 분 — 45 epoch 기준)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
