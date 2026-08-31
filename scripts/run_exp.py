"""STEP 24-25: 실험 실행기 (EXP-A / EXP-B / EXP-C).

    python scripts/run_exp.py -c configs/exp_a.yaml --source synthetic
    python scripts/run_exp.py -c configs/exp_a.yaml --source mitdb

공정성 규약
----------
  * 모든 방법이 **완전히 동일한 y** 를 받는다 (평가 세트를 먼저 만들어 공유).
  * 모든 방법의 출력에 **동일한 R-peak 검출기**와 **동일한 guard band** 를 적용한다.
  * oracle(B01/B02)만 ctx['x_clean'] 을 받는다.
  * `mode: distortion` 이면 잡음 없이 clean 을 그대로 통과시킨다 (EXP-C).

산출: results/{tag}/{exp_id}/metrics.parquet (long format), manifest.json
  tag = d0(합성) / d1(MIT-BIH). 데이터축이 다른 결과가 서로를 덮어쓰지 않도록
  경로를 분리한다. config 의 체크포인트 경로에 쓴 `{tag}` 도 같은 값으로 치환된다.
"""
import _bootstrap  # noqa: F401

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import ecgdn.methods  # noqa: F401  레지스트리 등록
from ecgdn.data.dataset import build_eval_set
from ecgdn.data.nstdb import make_banks
from ecgdn.data.sources import get_source, resolve_source_kind, source_tag
from ecgdn.eval.engine import evaluate, make_ref_cache
from ecgdn.registry import available, build, meta
from ecgdn.utils import ensure_dir, save_manifest


# front-end 를 끌 수 없는 방법: 그 필터 자체가 방법의 정의다
FE_INTRINSIC = {"M_FE", "M01", "M01d"}

# 축별 SWT 튜닝 결과를 주입할 방법. M04s 는 **교과서 기본 설정** 이 그 정의이므로
# 제외한다 (튜닝의 이득을 보이는 대조군이다).
SWT_TUNED = {"M04", "M04np", "B01"}


def load_swt_tuning(tag: str) -> dict | None:
    """`results/{tag}/tune_swt/best.json` 을 SWTCfg 인자로 읽는다.

    축마다 최적점이 다르다 — D0 는 sigma=d2/garrote/protect=True/k=(2.5,...),
    D1 은 sigma=d1/hard/protect=False/k=(0.6,...). D0 파라미터로 D1 을 돌리면
    M04 가 부당하게 약해지고, 그 상태의 비교는 F-6 에서 겪은 실수(약한
    baseline 과의 비교)의 반복이다 (docs/21_decisions.md D-9).
    """
    from ecgdn.config import SWTCfg

    p = Path("results") / tag / "tune_swt" / "best.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    fields = set(SWTCfg.__dataclass_fields__)
    return {k: (tuple(v) if isinstance(v, list) else v)
            for k, v in d.items() if k in fields}


def _build_with_fe(mid: str, frontend: bool, swt_kw: dict | None = None):
    """use_frontend 와 축별 SWT 파라미터를 받는 팩토리에만 전달한다."""
    import inspect

    from ecgdn.registry import _REGISTRY  # noqa: PLC2701

    kw = dict(swt_kw) if (swt_kw and mid in SWT_TUNED) else {}
    # M04np 는 'QRS 보호만 끈 조건' 이 정의다. 튜닝값이 무엇이든 이 축은 고정한다.
    if mid == "M04np":
        kw["protect_qrs"] = False
    if not (frontend or mid in FE_INTRINSIC):
        fn = _REGISTRY[mid]
        try:
            params = inspect.signature(fn).parameters
            if "kw" in params or "use_frontend" in params:
                kw["use_frontend"] = False
        except (TypeError, ValueError):
            pass
    return build(mid, **kw) if kw else build(mid)


def build_methods(cfg: dict, tag: str = "") -> dict:
    frontend = bool(cfg.get("frontend", True))
    swt_kw = load_swt_tuning(tag) if cfg.get("swt_from_tuning", True) else None
    if swt_kw:
        print(f"[swt] {tag} 튜닝값 적용 -> {swt_kw}")
    else:
        print(f"[swt] {tag} 튜닝값 없음 — config.py 기본값 사용")
    ms: dict = {}
    for mid in cfg.get("methods", []):
        ms[mid] = _build_with_fe(mid, frontend, swt_kw)
    for mid, spec in (cfg.get("dl_methods") or {}).items():
        from ecgdn.methods.dl_wrapper import DLDenoiser
        if isinstance(spec, str):
            spec = {"ckpt": spec}
        # config 는 `results/{tag}/m06_l1/best.pt` 처럼 쓴다. 실행 시 고른 데이터축의
        # 체크포인트가 자동으로 선택되므로, D0 모델을 D1 평가에 쓰는 사고가 막힌다.
        ck = Path(str(spec["ckpt"]).replace("{tag}", tag))
        if not ck.exists():
            print(f"[skip] {mid}: checkpoint not found -> {ck}")
            continue
        ms[mid] = DLDenoiser(ckpt=ck, name=mid, pre=spec.get("pre"),
                             batch=int(spec.get("batch", 32)),
                             # 명시하지 않으면 체크포인트의 학습 설정을 따른다
                             frontend=spec.get("frontend"))
    return ms


# ---------------------------------------------------------------- 실험 내부 재개
#
# 학습에는 `--resume` 이 있는데 실험에는 **항목 단위 재개가 없었다.** 컨테이너가
# 실험 도중에 재시작되면(실제로 자주 그런다 — O-2 · O-15 · O-17) 그 실험은
# 처음부터 다시 돈다. EXP-G(2156 항목, 축당 약 80 분)를 돌리다 두 번 연속
# 220/2156 과 70/2156 에서 죽었고, **재개가 없으니 진행이 0 이었다.**
#
# `run_all_experiments.sh` 의 재개는 **실험 단위**라 여기까지 못 막는다.
#
# 방식: 항목 `CHUNK` 개마다 부분 결과를 `_partial/` 에 떨군다. 다시 시작하면
# 거기 있는 것을 읽고 **그 다음 항목부터** 잇는다. 끝나면 합쳐서
# `metrics.parquet` 을 쓰고 `_partial/` 을 지운다.
#
# **설정이 바뀌면 부분 결과를 버린다.** 평가 세트가 달라졌는데 이어 붙이면
# 서로 다른 조건의 행이 한 표에 섞인다 — 그것이 F-9 계열의 사고다. 그래서
# 부분 결과에 설정 지문을 함께 적고 다르면 처음부터 돈다.
CHUNK = 25


def _fingerprint(cfg: dict, n_items: int, methods: list[str]) -> str:
    """이 부분 결과가 **같은 실험의 것인가**. 설정·항목 수·방법 목록으로 만든다."""
    import hashlib
    blob = json.dumps({"data": cfg.get("data"), "mode": cfg.get("mode"),
                       "frontend": cfg.get("frontend"), "eval": cfg.get("eval"),
                       "n": n_items, "m": sorted(methods)},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_partial(out: Path, fp: str) -> tuple[list[dict], int]:
    """저장된 부분 결과와 **다음에 처리할 항목 인덱스**."""
    d = out / "_partial"
    meta = d / "fingerprint.txt"
    if not d.exists() or not meta.exists():
        return [], 0
    if meta.read_text().strip() != fp:
        print(f"[resume] 설정이 달라 부분 결과를 버린다 -> {d}")
        shutil.rmtree(d, ignore_errors=True)
        return [], 0
    rows, done = [], 0
    for f in sorted(d.glob("rows-*.parquet")):
        rows.extend(pd.read_parquet(f).to_dict("records"))
        done = max(done, int(f.stem.split("-")[1]))
    if rows:
        print(f"[resume] 부분 결과 {len(rows)} 행 · 항목 {done} 개까지 완료 -> 이어서 돈다")
    return rows, done


def save_partial(out: Path, fp: str, new_rows: list[dict], upto: int) -> None:
    """**이번 조각에서 새로 생긴 행만** 쓴다.

    처음에는 `rows` 전체를 매번 썼는데, `load_partial` 이 조각 파일을 전부
    이어 붙이므로 **같은 행이 조각 수만큼 중복**된다(50 항목에서 25200 행 —
    맞는 값은 16800). 표는 여전히 그럴듯해 보이고 평균도 안 변한다(같은 행이
    똑같이 늘어나니까). 그래서 눈으로는 못 잡는다 — 행 수를 세야 잡힌다.
    """
    if not new_rows:
        return
    d = ensure_dir(out / "_partial")
    (d / "fingerprint.txt").write_text(fp)
    pd.DataFrame(new_rows).to_parquet(d / f"rows-{upto:05d}.parquet", index=False)



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="평가 항목 수 상한 (연습용)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--source", default=None, choices=("auto", "synthetic", "mitdb"),
                    help="config 의 data.source 를 덮어쓴다. 재현성을 위해 명시를 권한다")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.source is not None:
        cfg.setdefault("data", {})["source"] = args.source
    exp_id = cfg.get("exp_id", Path(args.config).stem)
    d = cfg.get("data", {})
    mode = cfg.get("mode", "snr")

    requested = d.get("source", "auto")
    kind = resolve_source_kind(requested)
    tag = source_tag(requested)
    if requested == "auto":
        print(f"[{exp_id}] source=auto -> {kind!r} 로 해석됨. "
              f"재현성을 위해 --source {kind} 를 명시할 것.")
    out = ensure_dir(args.out or f"results/{tag}/{exp_id}")

    src = get_source(d.get("source", "auto"), dur_s=float(d.get("dur_s", 300.0)),
                     n_train=int(d.get("n_train", 18)), n_val=int(d.get("n_val", 4)),
                     n_test=int(d.get("n_test", 22)))
    banks = make_banks(d.get("noise_split", "test"), d.get("nstdb_root", "data/raw/nstdb"))

    snr_grid = [0.0] if mode == "distortion" else [float(v) for v in d.get("snr_grid",
                                                                           [-5, 0, 5, 10, 15, 20])]
    items = build_eval_set(src, d.get("split", "test"),
                           seg_s=float(d.get("seg_s", 60.0)),
                           snr_grid=snr_grid,
                           noise_conditions=tuple(d.get("conditions", ["mixed"])),
                           banks=banks,
                           n_seg_per_record=int(d.get("n_seg_per_record", 2)),
                           seed=d.get("seed", "eval"))
    if mode == "distortion":
        # 잡음 없이 **원본** 을 그대로 입력한다. 참조(it["x"])는 front-end 를 통과한
        # 신호이므로 그것을 입력으로 주면 front-end 가 두 번 걸린다.
        for it in items:
            it["y"] = it.get("x_raw", it["x"]).copy()
            it["snr"] = float("inf")
    if args.limit:
        items = items[: args.limit]

    methods = build_methods(cfg, tag)
    ev = cfg.get("eval", {})
    do_morph = bool(ev.get("do_morph", True))
    do_spec = bool(ev.get("do_spectral", True))

    print(f"[{exp_id}] mode={mode} source={src.kind} tag={tag} items={len(items)} "
          f"frontend={cfg.get('frontend', True)} methods={list(methods)}")
    fp = _fingerprint(cfg, len(items), list(methods))
    rows, done = load_partial(out, fp)
    saved = len(rows)                       # 이미 파일에 들어간 행 수
    t0 = time.perf_counter()
    for i, it in enumerate(items, 1):
        if i <= done:                       # 이미 끝난 항목 — 건너뛴다
            continue
        x, y, fs = it["x"].astype(np.float64), it["y"].astype(np.float64), it["fs"]
        # reference 쪽 계산(특히 느린 delineation)은 구간당 1회만 한다
        cache = make_ref_cache(x, fs, it["r_peaks"], do_morph=do_morph)
        for mid, fn in methods.items():
            ctx = {"x_clean": x} if getattr(fn, "needs_clean", False) else {}
            ts = time.perf_counter()
            try:
                xhat = fn(y, fs, ctx)
            except Exception as e:
                print(f"  [err] {mid} on {it['record']}/{it['seg']}: {type(e).__name__}: {e}")
                continue
            lat = time.perf_counter() - ts
            m = evaluate(x, y, xhat, fs, r_peaks_ref=it["r_peaks"],
                         do_morph=do_morph, do_spectral=do_spec, cache=cache)
            m["latency_s"] = lat
            m["rtf"] = lat / (len(y) / fs)
            fam = meta(mid).get("family", "") if mid in available() else "deep"
            base = dict(exp=exp_id, record=it["record"], seg=it["seg"],
                        cond=it["cond"], snr_in_target=it["snr"], method=mid,
                        family=fam)
            for k, v in m.items():
                rows.append({**base, "metric": k, "value": float(v)})
        if i % CHUNK == 0 or i == len(items):
            save_partial(out, fp, rows[saved:], i)   # 재시작해도 여기까지는 안 잃는다
            saved = len(rows)
        if i % 10 == 0 or i == len(items):
            el = time.perf_counter() - t0
            n_new = max(1, i - done)
            print(f"  {i}/{len(items)}  {el:.0f}s  "
                  f"(eta {el / n_new * (len(items) - i):.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(out / "metrics.parquet", index=False)
    shutil.rmtree(out / "_partial", ignore_errors=True)   # 완주했으면 부분 결과는 필요 없다
    cfg = dict(cfg, _swt_applied=load_swt_tuning(tag))
    save_manifest(out, cfg=cfg, extra={"n_items": len(items), "methods": list(methods)},
                  sources=["scripts/run_exp.py", "ecgdn/data/dataset.py",
                           "ecgdn/methods/frontend.py", "ecgdn/eval/engine.py", "ecgdn/eval/signal_metrics.py", "ecgdn/eval/morphology.py", "ecgdn/eval/rpeak.py"])
    print(f"\nrows={len(df)}  -> {out/'metrics.parquet'}")

    key = "snr_imp_scaled" if mode != "distortion" else "snr_out_strict"
    piv = (df[df.metric == key].pivot_table(index="method", columns="snr_in_target",
                                            values="value", aggfunc="mean").round(2))
    print(f"\n=== {key} (mean) ===")
    print(piv.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
