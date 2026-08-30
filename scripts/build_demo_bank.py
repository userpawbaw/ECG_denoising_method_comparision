"""R-1: 모드 B(기존 데이터) 시연용 파형 은행을 **미리** 만든다.

    python scripts/build_demo_bank.py                 # 두 축 전부
    python scripts/build_demo_bank.py --axis d1       # 한 축만
    python scripts/build_demo_bank.py --scenes mixed0 # 한 장면만 (확인용)

산출: demo/demo_bank.js  (브라우저가 <script src> 로 읽는다)

왜 사전 계산인가
---------------
모드 B 의 입력은 **녹음본**이라 실시간 제약이 아예 없다(30_realtime_demo 1절).
미리 계산해 두면 (a) 시연 중 계산이 실패할 수 없고 (b) 선택이 즉시 반영되며
(c) 파이썬 환경 없이 브라우저만으로 돌릴 수 있다.

이 스크립트가 지키는 것 셋
-------------------------
**(1) 구간을 성능으로 고르지 않는다 — 대표성으로 고른다.**
장면마다 후보(기록 × 1 구간) 전부에 전 방법을 돌리고, **방법별 값이 축 평균과 가장
가까운** 구간을 쓴다. 가장 잘 나온 구간을 고르면 시연이
보고서보다 좋아 보인다 — 실제로 D1 `mixed` 0 dB 에서 딥러닝-고전 대비가
기록마다 −9.3~+13.3 dB 로 벌어진다. 산출물에 `contrast_rank` 를 함께 적어
**고른 구간이 최상위가 아님을 확인할 수 있게** 했다.

그리고 그 구간의 값 옆에 **축 전체 평균(`ref_mean`)을 같이 담는다.** 구간
하나만 보이면 그것이 결론으로 읽힌다.

**(2) 화면에 보이는 구간 = 지표를 잰 구간.**
평가는 양끝 `EVAL_GUARD_S`(5 s)를 버린다. 그래서 20 s 를 처리하고 **가운데
10 s 만** 저장한다. 방법은 20 s 를 다 보므로 가장자리 효과는 화면 밖에 있다.

**(3) 한 장면의 모든 파형이 같은 스케일로 저장된다.**
장면마다 양자화 스케일 하나를 공유한다. 행마다 y 축이 다르면 차이가 없는
방법도 좋아 보인다 — S4 슬라이드를 만들 때 실제로 그런 일이 있었다
(30_realtime_demo 4.1).

형식
----
int16 양자화 + base64. 한 파형 2500 샘플 = 5 kB → base64 6.7 kB.
브라우저에서 `atob` → `Int16Array` → `× scale` 이면 끝이다.
"""
import _bootstrap  # noqa: F401

import argparse
import base64
import json
import time
from pathlib import Path

import numpy as np

import ecgdn.methods  # noqa: F401  레지스트리 등록
from ecgdn.config import EVAL_GUARD_S
from ecgdn.data.dataset import build_eval_set
from ecgdn.data.nstdb import make_banks
from ecgdn.data.sources import get_source, source_tag
from ecgdn.eval.engine import trim_guard
from ecgdn.eval.signal_metrics import metrics_signal
from ecgdn.utils import ensure_dir, save_manifest
from run_exp import build_methods


ROOT = Path(__file__).resolve().parents[1]

SEG_S = 20.0            # 처리 길이. guard 5 s × 2 를 빼면 화면 10 s
N_CAND = 1              # 기록당 후보 구간 수 (기록이 22 개라 이것으로 충분)


# ---------------------------------------------------------------- 장면 정의
#
# 여섯 장면은 **보고서의 여섯 결론을 하나씩** 보인다. 근거는 5.1(SNR별)과
# 5.2(잡음별) 표이고, 아래 `근거` 는 D1 `exp_b`/`exp_a` 의 실측 평균이다.
SCENES = [
    dict(id="mixed0", cond="mixed", snr=0.0, ref="exp_a",
         title="혼합 잡음 0 dB — 대표 조건",
         claim="딥러닝이 앞선다. 다만 격차는 5 dB 정도다.",
         evidence="D1 exp_a 0 dB 평균: M06 16.74 / M04 11.49 / M_FE 9.77 dB"),
    dict(id="impulse10", cond="impulse", snr=10.0, ref="exp_b",
         title="임펄스 잡음 10 dB — 딥러닝만 잡는다",
         claim="고전 방법은 거의 손을 못 댄다. 격차가 가장 큰 장면이다.",
         evidence="D1 exp_b 평균: M06 13.65 / M08 13.69 vs M04 3.56 / M01 5.15 dB"),
    dict(id="pli10", cond="pli", snr=10.0, ref="exp_b",
         title="전원선(60 Hz) 잡음 10 dB — front-end 가 이미 해결",
         claim="딥러닝이 **진다**. 좁은 대역 잡음은 notch 하나가 최선이다.",
         evidence="D1 exp_b 평균: M04 28.76 / M_FE 23.43 vs M06 16.12 dB"),
    dict(id="bw10", cond="bw_synth", snr=10.0, ref="exp_b",
         title="기저선 변동 10 dB — front-end 가 상한에 닿는다",
         claim="M_FE 가 oracle(B01)과 같은 값이다. 더 할 것이 없다.",
         evidence="D1 exp_b 평균: M_FE 24.70 = B01 24.70 vs M06 15.24 dB"),
    dict(id="mixed20", cond="mixed", snr=20.0, ref="exp_a",
         title="혼합 잡음 20 dB — 깨끗한 입력에서는 딥러닝이 진다",
         claim="원래 깨끗한 신호를 딥러닝이 건드려 손해를 본다. L6 은 이것을 겨냥한 손실이다.",
         evidence="D1 exp_a 20 dB 평균: M_FE 17.43 / M04 17.08 vs M06 11.72 dB"),
    dict(id="em10", cond="em_synth", snr=10.0, ref="exp_b",
         title="전극 움직임 10 dB — 차이가 거의 없다",
         claim="모든 방법이 1.3 dB 안에 모인다. **차이가 안 나는 조건도 보여야 정직하다.**",
         evidence="D1 exp_b 평균: M06 11.35 / M04 10.09 / M_FE 10.12 dB"),
]

# 화면에 실을 후보. 무엇을 보일지는 UI(R-2)가 고른다 — 은행은 넉넉히 담는다.
# `M06L6` 은 5.8.9 의 성공한 손실 변경이다. 표에 쓰인 것은 전부 L1 이므로
# **둘을 나란히 담아** 손실 변경의 효과를 시연에서 그대로 보일 수 있게 한다.
DISPLAY_CFG = {
    "methods": ["M_FE", "M01", "M02", "M03", "M04", "M05", "B01"],
    "dl_methods": {
        "M06": {"ckpt": "results/{tag}/m06_l1/best.pt"},
        "M06L6": {"ckpt": "results/{tag}/m06_l6/best.pt"},
        "M08": {"ckpt": "results/{tag}/m08_l1/best.pt"},
        "M09": {"ckpt": "results/{tag}/m09_l1/best.pt"},
    },
    "frontend": True,
}
# `[?]` 버튼이 띄울 설명. **한 곳에서만 쓴다** — 브라우저 쪽에 따로 적으면
# 보고서와 갈라지고, 갈라진 것을 알아챌 방법이 없다(F-9 계열).
# 이름표(label)는 레지스트리에서 그대로 가져온다.
# 딥러닝 방법은 레지스트리에 등록되지 않는다(체크포인트를 직접 지정해 만든다).
# 그래서 이름표만 여기에 둔다 — 고전 방법의 이름표는 레지스트리가 정본이다.
DL_LABELS = {
    "M06": "ResUNet1D (L1 손실)",
    "M06L6": "ResUNet1D (L6 손실 — clean 보존)",
    "M08": "Wavelet subband U-Net",
    "M09": "CNN + Transformer",
}

NOTES = {
    "M_FE": "모든 방법이 공통으로 통과하는 front-end 다(0.5~40 Hz + 자동 notch). "
            "표의 이득 중 얼마가 front-end 에서 온 것인지 여기서 읽는다. "
            "좁은 대역 잡음(PLI)과 기저선 변동은 사실상 이 단계에서 끝난다.",
    "M01": "대역통과 + 자동 notch. 가장 단순한 기준선인데, 입력이 이미 깨끗하면"
           "(20 dB) 딥러닝을 이긴다.",
    "M02": "Savitzky-Golay — 국소 다항 적합으로 매끄럽게 만든다. 잡음이 크면 "
           "QRS 의 뾰족한 부분까지 함께 뭉갠다.",
    "M03": "DWT soft threshold. 이동 불변이 아니라서 임계 처리 자리에 따라 "
           "인공물이 남는다. M04 가 그 약점을 고친 것이다.",
    "M04": "SWT(이동 불변 wavelet) + QRS 보호. 고전 방법 중 가장 강하다. "
           "파라미터는 데이터축마다 따로 튜닝했다 — D0 값으로 D1 을 돌리면 "
           "부당하게 약해진다(D-9).",
    "M05": "Sameni 의 EKF/EKS — ECG 파형을 상태공간 모델로 두고 추정한다. "
           "커널(파형 모델)을 아는 조건에서는 매우 강하지만, 실제 파이프라인에서는 "
           "그 전제가 깨져 단순 대역통과 수준으로 내려간다(F-14 · F-17).",
    "B01": "**방법이 아니라 한계선이다.** 참값을 보고 계수를 최적으로 고른 "
           "oracle 이므로 그 계열이 낼 수 있는 상한을 뜻한다. 어떤 방법이 이것을 "
           "넘으면 측정이 틀린 것이다.",
    "M06": "ResUNet1D — 이 프로젝트의 주 딥러닝 모델. 표의 딥러닝 값은 특별한 "
           "표시가 없으면 전부 이 모델(L1 손실)이다.",
    "M06L6": "같은 M06 을 **L6 손실**로 학습한 것. 깨끗한 입력을 덜 건드리도록 "
             "'clean 보존' 항을 넣었다. 20 dB 에서 L3 의 두 배 이득을 냈다(5.8.9). "
             "다만 손실 절제 실험이 혼합 잡음만 다뤄서 **잡음 종류별 축 평균이 "
             "없다** — 이 화면에서도 평균 칸이 비어 있다.",
    "M08": "wavelet 부분대역마다 U-Net 을 두는 구조. M06 과 유의한 차이가 없었다 "
           "— 구조를 바꿔 얻은 것이 없다는 이 프로젝트의 결론 중 하나다.",
    "M09": "CNN + Transformer(전역 attention). 긴 문맥을 보게 했지만 8 개 비교 "
           "칸 전부에서 유의한 차이가 없었다(5.10).",
}


def _q(sig: np.ndarray, scale: float) -> str:
    """int16 양자화 + base64. 스케일은 **장면 전체가 공유**한다."""
    q = np.clip(np.round(sig / scale), -32768, 32767).astype("<i2")
    return base64.b64encode(q.tobytes()).decode("ascii")


def _imp(x, y, xhat) -> float:
    return float(metrics_signal(x, y, xhat)["snr_imp_scaled"])


def reference_means(axis: str, scene: dict, methods: list[str]) -> dict[str, float]:
    """축 전체 평균(`exp_a`/`exp_b` 의 `snr_imp_scaled`).

    시연은 **구간 하나**를 보이는데, 구간 하나는 평균과 크게 다를 수 있다.
    실제로 D1 `mixed` 0 dB 에서 딥러닝-고전 대비가 기록마다 −9.3~+13.3 dB 로
    벌어진다. 그래서 화면에 **그 구간의 값과 축 평균을 나란히** 띄우려고
    여기서 평균을 함께 담는다. 하나만 보이면 구간 하나가 결론으로 읽힌다.
    """
    import pandas as pd

    p = ROOT / "results" / axis / scene["ref"] / "metrics.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    sub = df[(df.metric == "snr_imp_scaled") & (df.cond == scene["cond"])]
    if scene["ref"] == "exp_a":
        sub = sub[sub.snr_in_target == scene["snr"]]
    g = sub.groupby("method")["value"].mean()
    # M06L6 은 실험표에 없다(표는 전부 L1). 평균을 지어내지 않고 **빼 둔다** —
    # 브라우저는 값이 없으면 '평균 없음' 으로 그린다.
    return {m: round(float(g[m]), 2) for m in methods if m in g.index}


def pick_segment(items, methods, ref) -> tuple[int, list, dict]:
    """**축 평균과 방법별 값이 가장 가까운** 구간을 고른다.

    기준을 두 번 바꿨다.

    1. *딥러닝−고전 대비가 중앙값인 구간*. 시험해 보니 고른 구간에서
       `M05`(Sameni)가 oracle 보다 높게 나왔다 — 축 평균에서는 `M01` 보다도
       낮은 방법이다. **대비 하나만 통제하면 나머지 방법의 순위는 통제되지
       않는다.**
    2. *방법 순위의 Spearman 상관이 최대인 구간*. 순위는 맞았지만
       D1 `mixed` 0 dB 에서 **대비가 22 개 중 1 위인 구간**이 뽑혔다. 순위는
       크기를 보지 않으므로, 격차가 가장 크게 벌어진 구간이 순위를 가장
       깨끗하게 재현한다. 시연이 보고서보다 좋아 보이게 된다.
    3. **채택** — 방법별 `snr_imp` 벡터와 축 평균 벡터의 평균 절대차가
       최소인 구간. 순서와 크기를 **한 숫자로** 함께 통제한다.

    "가장 좋은 구간" 이 아니라 **"가장 평범한 구간"** 을 고르는 것이다.
    산출물의 `contrast_rank` 로 그것을 확인할 수 있다.
    """
    per_item = []
    for it in items:
        x, y, fs = it["x"].astype(np.float64), it["y"].astype(np.float64), it["fs"]
        sl = trim_guard(x.size, fs, EVAL_GUARD_S)
        traces, imps = {}, {}
        for mid, fn in methods.items():
            ctx = {"x_clean": x} if getattr(fn, "needs_clean", False) else {}
            t0 = time.perf_counter()
            h = np.asarray(fn(y, fs, ctx), dtype=np.float64)
            rtf = (time.perf_counter() - t0) / (y.size / fs)
            m = metrics_signal(x[sl], y[sl], h[sl])
            traces[mid] = h[sl]
            imps[mid] = {"snr_imp": round(m["snr_imp_scaled"], 2),
                         "snr_out": round(m["snr_out_scaled"], 2),
                         "cc": round(m["cc"], 4), "rtf": round(rtf, 4)}
        per_item.append(dict(traces=traces, metrics=imps, x=x[sl], y=y[sl]))

    def contrast(mets):
        dl = max(mets[m]["snr_imp"] for m in ("M06", "M08", "M09") if m in mets)
        cl = max(mets[m]["snr_imp"] for m in ("M_FE", "M01", "M02", "M03", "M04")
                 if m in mets)
        return round(dl - cl, 2)

    con = [contrast(d["metrics"]) for d in per_item]
    have = [m for m in methods if m in ref]
    if len(have) >= 4:
        dist = [float(np.mean([abs(d["metrics"][m]["snr_imp"] - ref[m]) for m in have]))
                for d in per_item]
        k = int(np.argmin(dist))
    else:                       # 참조 평균이 없으면(산출물 미생성) 대표성을 못 본다
        print("  [warn] 축 평균이 없어 첫 구간을 쓴다 — 대표성 미보장")
        dist, k = [float("nan")] * len(per_item), 0
    rank = int(np.sum(np.asarray(con) > con[k])) + 1
    return k, per_item, dict(mean_abs_dev=round(dist[k], 3),
                             dev_range=[round(min(dist), 3), round(max(dist), 3)],
                             contrast_db=con[k],
                             contrast_rank=f"{rank}/{len(con)}",
                             contrast_range=[min(con), max(con)])


def build_scene(scene: dict, tag: str, src, banks, methods) -> dict:
    items = build_eval_set(src, "test", seg_s=SEG_S, snr_grid=[scene["snr"]],
                           noise_conditions=(scene["cond"],), banks=banks,
                           n_seg_per_record=N_CAND, seed=f"demo_{scene['id']}")
    if not items:
        raise RuntimeError(f"{scene['id']}: 평가 항목이 비었다")
    ref = reference_means(tag, scene, list(methods))
    k, per_item, sel = pick_segment(items, methods, ref)
    it, got = items[k], per_item[k]

    traces = {"clean": got["x"], "input": got["y"], **got["traces"]}
    # 장면 전체가 스케일 하나를 공유한다 — y 축을 맞추는 것을 데이터에서 강제한다.
    peak = max(float(np.max(np.abs(v))) for v in traces.values())
    scale = peak / 32000.0 if peak > 0 else 1.0
    return dict(
        id=f"{tag}-{scene['id']}", axis=tag, cond=scene["cond"], snr=scene["snr"],
        ref_exp=scene["ref"], title=scene["title"], claim=scene["claim"],
        evidence=scene["evidence"],
        record=str(it["record"]), seg=int(it["seg"]), n_candidates=len(items),
        selection=sel,
        ylim=round(float(np.max(np.abs(traces["clean"]))) * 1.3, 6),
        scale=scale, metrics=got["metrics"], ref_mean=ref,
        # 축 평균이 **없는** 방법을 명시한다. 브라우저는 이 목록의 방법 옆에
        # 평균 칸을 비우고 "이 구간 값" 이라고만 적어야 한다 — 빈칸을 그럴듯한
        # 숫자로 메우지 않기 위해서다. `M06L6` 은 손실 절제(`abl_loss`)가
        # `mixed` 만 다뤄서 잡음 종류별 평균이 존재하지 않는다.
        no_ref=[m for m in methods if m not in ref],
        traces={k2: _q(v, scale) for k2, v in traces.items()},
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", nargs="*", default=["d0", "d1"], choices=("d0", "d1"))
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="장면 id 부분집합 (확인용)")
    ap.add_argument("--out", default="demo/demo_bank.js")
    args = ap.parse_args()

    want = [s for s in SCENES if args.scenes is None or s["id"] in args.scenes]
    if not want:
        raise SystemExit(f"알 수 없는 장면: {args.scenes}")

    scenes, fs_seen = [], set()
    for axis in args.axis:
        kind = "synthetic" if axis == "d0" else "mitdb"
        tag = source_tag(kind)
        assert tag == axis, f"{kind} -> {tag}, 기대 {axis}"
        src = get_source(kind, dur_s=300.0, n_train=18, n_val=4, n_test=22)
        banks = make_banks("test", "data/raw/nstdb")
        methods = build_methods(DISPLAY_CFG, tag)
        if "M06" not in methods:
            raise SystemExit(f"{axis}: M06 체크포인트가 없다")
        for sc in want:
            t0 = time.perf_counter()
            out = build_scene(sc, tag, src, banks, methods)
            fs_seen.add(round(float(src.get(src.records("test")[0]).fs), 3))
            scenes.append(out)
            s = out["selection"]
            print(f"[{out['id']}] {out['record']}/{out['seg']}  "
                  f"평균편차 {s['mean_abs_dev']:.2f} dB "
                  f"(후보 {s['dev_range'][0]:.2f}~{s['dev_range'][1]:.2f})  "
                  f"대비 {s['contrast_db']:+.2f} dB [{s['contrast_rank']} 위, "
                  f"{s['contrast_range'][0]:+.1f}~{s['contrast_range'][1]:+.1f}]  "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)

    if len(fs_seen) != 1:
        raise RuntimeError(f"축마다 fs 가 다르다: {fs_seen} — 시간축을 공유할 수 없다")
    fs = fs_seen.pop()
    n = scenes[0]["metrics"] and len(base64.b64decode(scenes[0]["traces"]["clean"])) // 2
    from ecgdn.registry import available, meta
    mids = list(scenes[0]["metrics"])
    info = {}
    for m in mids:
        base = "M06" if m.startswith("M06") else m
        md = (meta(base) if base in available()
              else {"label": DL_LABELS.get(m, m), "family": "deep"})
        note = NOTES.get(m)
        if note is None:            # 설명 없는 방법을 조용히 내보내지 않는다
            raise SystemExit(f"NOTES 에 '{m}' 설명이 없다 — 빈 [?] 버튼을 만들지 않는다")
        info[m] = {"label": md["label"], "family": md["family"], "note": note}
    bank = dict(fs=fs, n=n, guard_s=EVAL_GUARD_S, seg_s=SEG_S,
                methods=mids, method_info=info, scenes=scenes)
    # 브라우저는 file:// 에서 `fetch` 를 못 한다. 그래서 JSON 파일이 아니라
    # **전역 변수를 대입하는 `.js`** 로 쓴다 — 시연 노트북에 서버가 필요 없다.
    p = Path(args.out)
    ensure_dir(p.parent)
    body = json.dumps(bank, ensure_ascii=False, separators=(",", ":"))
    p.write_text("window.DEMO_BANK = " + body + ";\n")
    save_manifest(p.parent / "demo_bank", cfg=DISPLAY_CFG,
                  extra={"scenes": [s["id"] for s in scenes], "n": n, "fs": fs},
                  sources=["scripts/build_demo_bank.py", "ecgdn/data/dataset.py"])
    print(f"\n{p}  {p.stat().st_size / 1e6:.2f} MB  "
          f"장면 {len(scenes)} × 파형 {len(scenes[0]['traces'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
