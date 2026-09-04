#!/usr/bin/env python3
"""FE 판 대 FE 없는 판 — **같은 분모로** 견준다 (docs/15 §11).

    python3 scripts/analyze_nofe.py

`results/d1/exp_nofe/metrics.parquet` 을 읽어 표를 만들고, 「왜 그런가」를
가르는 **기전 측정**을 따로 한다.

**학습 로그의 `val snr_imp` 로 둘을 견주면 안 된다.** 그 값은 「그 학습이
받은 입력」 대비 개선이라 분모가 서로 다르다 — FE 판의 입력은 이미 기저선
변동이 빠진 것이고, nofe 판의 입력은 날것이라 변동 제거분까지 자기 공로로
계산된다. 그래서 nofe 가 +9.6 dB, FE 판이 +2.6 dB 로 보이는데 이것은
**성능 차가 아니라 분모 차다** (F-10 의 덫).

평가 세트는 이미 공정하다 (`dataset.build_eval_set`):

    y (입력) = 원본 clean + 잡음      <- 두 방식이 **같은 것**을 받는다
    x (참조) = FE_off(clean)           <- 두 방식이 **같은 것**과 비교된다

FE 는 방법 **안쪽**에 있다. `M06` 은 자기 안에서 걸고 `M06n` 은 안 건다.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecgdn.config import FS                                    # noqa: E402
from ecgdn.data.dataset import build_eval_set                  # noqa: E402
from ecgdn.data.nstdb import make_banks                        # noqa: E402
from ecgdn.data.sources import get_source                      # noqa: E402
from ecgdn.eval.rpeak import detect_rpeaks                     # noqa: E402
from ecgdn.utils import ensure_dir, save_manifest              # noqa: E402

EXP = ROOT / "results" / "d1" / "exp_nofe" / "metrics.parquet"
DOC = ROOT / "docs" / "15_reference_and_training.md"
OUT = ROOT / "results" / "d1" / "exp_nofe"

# 표에 실을 방법과 읽는 이름. **순서가 곧 표의 순서다.**
NAMES = {"M00": "아무것도 안 함", "M_FE": "FE 만",
         "M06": "M06 (FE 판)", "M06n": "M06n (FE 없는 판)"}


def _fe_metrics():
    """평활도 지표는 `explore_lookahead_fe.py` 에 **한 벌만** 둔다.

    여기에 다시 구현하면 정의가 둘로 갈리고, 갈린 줄도 모르게 된다 —
    T-P 창이 그렇게 한 번 어긋났다 (O-22).
    """
    sys.path.insert(0, str(ROOT / "scripts"))     # 그쪽이 _bootstrap 을 읽는다
    spec = importlib.util.spec_from_file_location(
        "explore_lookahead_fe", ROOT / "scripts" / "explore_lookahead_fe.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ 표
def table(df: pd.DataFrame, key: str, agg: str = "mean") -> pd.DataFrame:
    p = (df[df.metric == key]
         .pivot_table(index="method", columns="snr_in_target", values="value",
                      aggfunc=agg))
    return p.reindex([m for m in NAMES if m in p.index])


def md(p: pd.DataFrame, fmt: str = "{:+.2f}", first: str = "방법") -> str:
    cols = [f"{c:g}" for c in p.columns]
    out = ["| " + first + " | " + " | ".join(cols) + " |",
           "|" + "---|" * (len(cols) + 1)]
    for mid, row in p.iterrows():
        cells = [fmt.format(v) if np.isfinite(v) else "—" for v in row]
        out.append(f"| {NAMES.get(mid, mid)} | " + " | ".join(cells) + " |")
    return "\n".join(out)


# ------------------------------------------- 기전: 모델이 앞단을 배웠는가
# 오차를 어디에 놓을지. **59~61 Hz 를 따로 뺀다** — 전원선 잡음이고, 그것을
# 없애는 것은 FE 의 notch 다. 이 칸이 크면 «모델이 notch 를 안 배웠다» 는 뜻이다.
ERR_BANDS = (("<0.5 Hz 기저선", 0.0, 0.5), ("0.5~40 Hz 신호대역", 0.5, 40.0),
             ("59~61 Hz 전원선", 59.0, 61.0), (">61 Hz 나머지", 61.0, 125.0))


def err_budget(e: np.ndarray, fs: float) -> dict[str, float]:
    """오차 `xhat - x` 의 파워를 대역별로 나눈다 [전체의 %].

    dB 한 숫자는 «얼마나 나쁜가» 만 말하고 «어디가 나쁜가» 는 말하지 않는다.
    두 모델의 차가 어느 대역에서 나오는지가 곧 «무엇을 못 배웠는가» 다.
    """
    E = np.fft.rfft(e - e.mean())
    f = np.fft.rfftfreq(e.size, 1 / fs)
    p = np.abs(E) ** 2
    tot = float(p.sum()) or 1.0
    return {name: 100 * float(p[(f >= lo) & (f < hi)].sum()) / tot
            for name, lo, hi in ERR_BANDS}


def mechanism(n_rec: int = 6, snrs=(0.0, 10.0)) -> pd.DataFrame:
    """**어디서 지는가.** 두 가지를 잰다.

    1. **평활도** — §7 에서 «창 1024 표본 = 4.096 s 인데 기저선 변동은 주기
       2~20 s 라 절대 준위를 원리적으로 못 본다» 고 `[미측정]` 으로 적어 둔
       가설을 확인한다.
    2. **오차의 대역 분포** — dB 한 숫자로는 원인을 못 가른다.
    """
    from ecgdn.methods.dl_wrapper import DLDenoiser

    fe = _fe_metrics()
    src = get_source("mitdb", n_train=18, n_val=4, n_test=22)
    banks = make_banks("test", "data/raw/nstdb")
    items = build_eval_set(src, "test", seg_s=60.0, snr_grid=list(snrs),
                           noise_conditions=("mixed",), banks=banks,
                           n_seg_per_record=1, seed="eval_a")
    recs = sorted({it["record"] for it in items})[:n_rec]
    items = [it for it in items if it["record"] in recs]

    dl = {"M06": DLDenoiser(ckpt=ROOT / "results/d1/m06_l1/best.pt", name="M06"),
          "M06n": DLDenoiser(ckpt=ROOT / "results/d1/m06_l1_nofe/best.pt", name="M06n")}
    rows = []
    for it in items:
        x, y, fs = it["x"].astype(float), it["y"].astype(float), it["fs"]
        r = detect_rpeaks(x, fs)
        if r.size < 5:
            continue
        r_amp = float(np.percentile(np.abs(x[r]), 75)) or 1.0
        outs = {"참조 FE_off(clean)": x, "입력 (원본+잡음)": y}
        for mid, fn in dl.items():
            outs[NAMES[mid]] = fn(y, fs, {})
        for label, v in outs.items():
            n = min(v.size, x.size)
            row = dict(record=it["record"], snr=it["snr"], who=label,
                       남은변동=fe.tp_off(v[:n], r[r < n], r_amp),
                       준위산포=fe.tp_spread(v[:n], r[r < n], r_amp),
                       박동당이동=fe.drift_per_beat(v[:n], r[r < n], r_amp))
            row.update(err_budget(v[:n] - x[:n], fs))
            # 오차의 절대 크기도 함께 — 비중만 보면 «작은 오차의 큰 비중» 에
            # 속는다.
            row["오차파워 [%R^2]"] = 100 * float(np.mean((v[:n] - x[:n]) ** 2)) / r_amp ** 2
            rows.append(row)
    return pd.DataFrame(rows)


def val_best(run: str) -> tuple[float, int]:
    """그 학습이 **모델 선택에 쓴** 값과 epoch. 체크포인트에서 읽는다.

    처음에는 `log.csv` 의 끝 몇 줄을 눈으로 읽고 손으로 옮겨 적었다가
    **틀렸다** — 마지막 epoch 은 best 가 아니다(m06_l1 은 ep 34 가 best 인데
    ep 44~46 을 보고 +2.6 이라고 적었다. 실제는 +3.437). 손으로 옮기지 않는다.
    """
    import torch
    ck = torch.load(ROOT / "results" / "d1" / run / "best.pt",
                    map_location="cpu", weights_only=False)
    return float(ck["best_metric"]), int(ck["epoch"])


def main() -> int:
    if not EXP.exists():
        print(f"{EXP} 가 없다. 먼저:\n"
              f"  python3 scripts/run_exp.py -c configs/exp_nofe.yaml --source mitdb")
        return 2
    df = pd.read_parquet(EXP)
    ensure_dir(OUT)

    snr = table(df, "snr_imp_scaled")
    prdn = table(df, "prdn")
    bcc = table(df, "beat_cc")
    qrs = table(df, "qrs_dur_err_ms")
    ramp = table(df, "r_amp_err_pct")
    f1 = table(df, "f1")

    mech = mechanism()
    mech.to_csv(OUT / "mechanism.csv", index=False)
    vals = [c for c in mech.columns if c not in ("record", "snr", "who")]
    piv = (mech.pivot_table(index="who", values=vals, aggfunc="median")
           .reindex(["참조 FE_off(clean)", "입력 (원본+잡음)",
                     NAMES["M06"], NAMES["M06n"]]))

    print("\n=== SNR 개선 (scaled, dB) ===\n", snr.round(2))
    print("\n=== 기전 (중앙값, %R) ===\n", piv.round(1))

    # **비율 자체는 평균도 중앙값도 오해를 부른다** — 평균은 소수의 큰 값에
    # 끌려가고, 중앙값은 그 소수를 통째로 숨긴다. 판정 기준(10)을 넘는
    # **항목의 비율**로 본다.
    d = df[df.metric == "pli_ratio_hat"]
    pli = (d.assign(hit=d.value >= 10.0)
           .pivot_table(index="method", columns="snr_in_target", values="hit",
                        aggfunc="mean") * 100)
    pli = pli.reindex([m for m in NAMES if m in pli.index])
    n_pli_rec = int(d[(d.method == "M06n") & (d.value >= 10.0)].record.nunique())
    write_doc(snr, prdn, bcc, qrs, ramp, f1, piv, mech, pli, n_pli_rec)
    save_manifest(OUT, cfg={"analysis": "nofe"},
                  extra={"n_mech_items": int(len(mech))},
                  sources=["scripts/analyze_nofe.py", "configs/exp_nofe.yaml"])
    return 0


def write_doc(snr, prdn, bcc, qrs, ramp, f1, piv, mech, pli, n_pli_rec) -> None:
    """docs/15 의 §11 을 **숫자에서** 다시 쓴다. 손으로 옮겨 적지 않는다."""
    def g(t, mid, col=0.0):
        return float(t.loc[mid, col])

    def w(k, who):
        return float(piv.loc[who, k])

    lo, hi = snr.columns[0], snr.columns[-1]
    gap = snr.loc["M06"] - snr.loc["M06n"]
    ref, inp = "참조 FE_off(clean)", "입력 (원본+잡음)"
    a, b = NAMES["M06"], NAMES["M06n"]
    # 대역 비중은 %, 오차 파워는 %R^2 — 곱해야 «절대 크기» 가 된다.
    ib = lambda who: w("0.5~40 Hz 신호대역", who) / 100 * w("오차파워 [%R^2]", who)  # noqa: E731
    # f-string 안에서는 중괄호를 못 쓴다 — 표는 밖에서 만든다.
    pli_md = md(pli, fmt="{:.1f}", first="방법 (판정 비율 %)")
    (vfe, efe), (vno, eno) = val_best("m06_l1"), val_best("m06_l1_nofe")

    body = f"""## 11. FE 없는 판을 학습했다 — **FE 가 이긴다. 다만 내가 예상한 이유는 아니었다** `[측정]`

`m06_l1_nofe`: 입력에만 FE 를 빼고(`frontend: false`), **목표는 그대로**
`FE_off(clean)` 로 뒀다(`ref_frontend: true`). 목표까지 날것으로 두면 D1 의
참조가 「MIT-BIH 원본」이 되어 **F-12 를 재현**한다 — 묻는 것은 «앞단을 스스로
배울 수 있는가» 이지 «참조를 바꾸면 어떻게 되는가» 가 아니다.

60 에폭 완주, best epoch {eno}.

### 먼저 — **학습 로그의 숫자로 견주면 안 된다**

| | val `snr_imp` (모델 선택에 쓴 값) |
|---|---|
| `m06_l1` (FE 판) | {vfe:+.2f} dB (ep {efe}) |
| `m06_l1_nofe` | **{vno:+.2f} dB** (ep {eno}) |

{vno/vfe:.1f} 배 좋아 보이는데 **이것은 성능 차가 아니라 분모 차다.** `snr_imp` 는
「그 학습이 받은 입력」 대비 개선이다. FE 판의 입력은 이미 기저선 변동이
빠진 것이라 개선할 여지가 적고, nofe 판의 입력은 날것이라 **변동 제거분까지
자기 공로로 계산된다.** F-10 이 정확히 이 덫이었다.

### 같은 분모로 재면 — `results/d1/exp_nofe`

평가 세트가 이미 공정하다 (`dataset.build_eval_set`): 입력 `y = 원본 + 잡음`,
참조 `x = FE_off(clean)`. 둘 다 방식과 무관하고 FE 는 방법 **안쪽**에 있다.
`M06` 은 자기 안에서 FE 를 걸고 `M06n` 은 안 건다.

**SNR 개선 [dB]** (`snr_imp_scaled`, 22 기록 x 2 구간 x 6 동작점)

{md(snr)}

**FE 판이 모든 동작점에서 이긴다** — 격차 {gap.min():+.1f} ~ {gap.max():+.1f} dB.
그리고 격차가 **입력이 깨끗할수록 벌어진다** ({lo:g} dB 에서 {gap[lo]:+.1f},
{hi:g} dB 에서 {gap[hi]:+.1f}). 잡음을 못 지워서가 아니라 **체계적인 차**라는
뜻이다 — 잡음이 없어져도 남는다.

파형도 같은 방향이다 (입력 SNR 0 dB, 나머지 동작점도 순서가 같다).

| | PRD_N (낮을수록) | 박동 상관 | QRS 폭 오차 [ms] | R 진폭 오차 [%] | R 검출 F1 |
|---|---|---|---|---|---|
| M06 (FE 판) | {g(prdn,'M06'):.1f} | {g(bcc,'M06'):.3f} | {g(qrs,'M06'):.1f} | {g(ramp,'M06'):.1f} | {g(f1,'M06'):.3f} |
| M06n (FE 없는 판) | {g(prdn,'M06n'):.1f} | {g(bcc,'M06n'):.3f} | {g(qrs,'M06n'):.1f} | {g(ramp,'M06n'):.1f} | {g(f1,'M06n'):.3f} |

### **§7 에서 세운 가설은 틀렸다** — 기저선이 아니다

§7 에 이렇게 적고 `[미측정]` 을 달아 뒀다:

> 창 1024 표본 = **4.096 s** 인데 기저선 변동 0.05~0.5 Hz 는 주기 **2~20 s** 다.
> 0.05 Hz 성분은 창 안에서 DC + 직선으로만 보이고 절대 준위는 **원리적으로
> 관측 불가**다.

재 보니 **아니었다.** T-P 평활 구간 [%R, 중앙값]:

| | 준위 이탈 | 준위 산포 | 박동당 이동 |
|---|---|---|---|
| 참조 `FE_off(clean)` | {w('남은변동',ref):.1f} | {w('준위산포',ref):.1f} | {w('박동당이동',ref):.1f} |
| 입력 (원본+잡음) | {w('남은변동',inp):.1f} | {w('준위산포',inp):.1f} | {w('박동당이동',inp):.1f} |
| M06 (FE 판) | {w('남은변동',a):.1f} | {w('준위산포',a):.1f} | {w('박동당이동',a):.1f} |
| M06n (FE 없는 판) | {w('남은변동',b):.1f} | {w('준위산포',b):.1f} | {w('박동당이동',b):.1f} |

**nofe 판의 준위 산포가 오히려 더 작다** ({w('준위산포',b):.1f} 대
{w('준위산포',a):.1f}, 참조 자신이 {w('준위산포',ref):.1f}). 입력의
{w('준위산포',inp):.1f} 에서 여기까지 내렸다. 창이 짧아 절대 준위를 못
본다는 예측은 **관측을 못 이겼다** — 창마다 자기 DC 를 빼고 겹침-합으로
이으면, 결과적으로 평활 대역이 0 에 붙는다.

### 그럼 어디서 지는가 — **신호 대역 안에서**

오차 `xhat - x` 의 파워를 대역별로 갈랐다 (`{OUT.relative_to(ROOT)}/mechanism.csv`).

| | 오차 파워 [%R²] | <0.5 Hz 기저선 | 0.5~40 Hz 신호대역 | 59~61 Hz 전원선 | >61 Hz |
|---|---|---|---|---|---|
| 입력 (원본+잡음) | {w('오차파워 [%R^2]',inp):.2f} | {w('<0.5 Hz 기저선',inp):.0f} % | {w('0.5~40 Hz 신호대역',inp):.0f} % | {w('59~61 Hz 전원선',inp):.0f} % | {w('>61 Hz 나머지',inp):.0f} % |
| M06 (FE 판) | {w('오차파워 [%R^2]',a):.2f} | {w('<0.5 Hz 기저선',a):.0f} % | {w('0.5~40 Hz 신호대역',a):.0f} % | {w('59~61 Hz 전원선',a):.1f} % | {w('>61 Hz 나머지',a):.0f} % |
| M06n (FE 없는 판) | {w('오차파워 [%R^2]',b):.2f} | {w('<0.5 Hz 기저선',b):.0f} % | {w('0.5~40 Hz 신호대역',b):.0f} % | {w('59~61 Hz 전원선',b):.1f} % | {w('>61 Hz 나머지',b):.0f} % |

nofe 판의 오차 파워가 **{w('오차파워 [%R^2]',b)/w('오차파워 [%R^2]',a):.1f} 배**
크고(= {10*np.log10(w('오차파워 [%R^2]',b)/w('오차파워 [%R^2]',a)):.1f} dB, 표의 격차와 맞는다), 그 초과분이
**신호 대역 안**에 있다 (절대량 {ib(b):.3f} 대 {ib(a):.3f} %R²).
기저선도 전원선도 아니다.

읽는 방법은 이렇다. **용량은 같은데 할 일이 늘었다.** FE 판의 모델은
「대역제한된 신호에서 잡음 지우기」 하나만 하면 되는데, nofe 판은 같은
97 만 파라미터로 **대역제한 + 노치 + 잡음 제거**를 한꺼번에 해야 한다.
그래서 파형 재현이 나빠진다 — 박동 상관 {g(bcc,'M06n'):.3f} 대 {g(bcc,'M06'):.3f},
QRS 폭 오차 {g(qrs,'M06n'):.1f} 대 {g(qrs,'M06'):.1f} ms 가 그 결과다.

### 곁다리 하나 — **노치는 실제로 안 배웠다**

오차 파워에서는 작지만, 화면에서는 바로 보이는 것이 있다. 60 Hz 봉우리가
이웃 대역 배경의 몇 배인가(`pli_ratio_hat`, 10 이상이면 «PLI 있음» 판정):

{pli_md}

**nofe 판은 구간의 {pli.loc['M06n'].min():.0f}~{pli.loc['M06n'].max():.0f} % 에서 «PLI 있음» 판정에 걸린다** —
FE 판은 {pli.loc['M06'].max():.1f} %, FE 만 걸어도 {pli.loc['M_FE'].max():.0f} % 다.
22 기록 중 {n_pli_rec} 기록에서 나온다.

다만 **모든 구간이 그런 것은 아니다.** 중앙값으로 보면 nofe 판도 배경의
2 배 남짓이라 멀쩡해 보인다 — 소수의 구간에서만 무너지는데, 평균은 그
소수에 끌려가고 중앙값은 그것을 통째로 숨긴다. 그래서 **판정 비율**로 적었다.

오차 에너지로는 {w('59~61 Hz 전원선',b):.1f} % 뿐이라 dB 를 거의 안 움직인다.
그런데 **파형 위에서는 60 Hz 잔물결로 얹혀 바로 보인다.** 지표가 «작다» 고
말하는 것과 사람이 «깨끗하다» 고 느끼는 것이 갈리는 자리다.

### 그래서 어떻게 하는가

| 목적 | 결론 |
|---|---|
| **A. 어느 방법이 우월한가** (이 과제의 본체) | FE 유지. 공정성 규약이 요구했고, 이제 **성능 근거도 생겼다** |
| **B. 낼 수 있는 최선** | 그것도 FE 유지다. FE 없는 판이 **모든 동작점에서 진다** |

§7 에서 목적 B 를 «정당하고 매력도 있다» 고 적었다. **판단은 옳았고(재 볼
값이 있었다) 예측은 틀렸다.** 매력의 근거로 든 셋 중 앞의 둘(위상 왜곡 소멸,
F-25 취약성 소멸)은 성립한다. 셋째 «자기 디코더에 맞는 앞단을 학습한다» 가
성립하지 않는데, **이유는 내가 든 「창이 짧다」가 아니라 「용량이 나뉜다」였다.**

> 그래서 값싼 결론이 하나 나온다. **손으로 짠 4 차 필터 하나가 모델 용량을
> 아껴 준다.** FE 를 없애 얻는 것(위상 왜곡 소멸)은 블록 영위상 0.5 s 로 이미
> 대부분 얻고 있고(§8.5), 잃는 것은 {gap.min():.1f}~{gap.max():.1f} dB 다.

**이 체크포인트는 EXP-A~G 의 표에 넣지 않는다** — 공정성 규약 4.2 를 일부러
깨는 조건이다. 곁가지(`configs/exp_nofe.yaml`, `results/d1/exp_nofe`)에만 산다.
"""
    s = DOC.read_text()
    head = "## 11. FE 없는 판을 학습했다"
    if head in s:
        i = s.index(head)
        j = s.find("\n## ", i + 1)
        s = s[:i] + body + (s[j + 1:] if j != -1 else "")
    else:
        s = s.rstrip() + "\n\n" + body
    DOC.write_text(s)
    print(f"  -> {DOC.relative_to(ROOT)} §11")


if __name__ == "__main__":
    raise SystemExit(main())
