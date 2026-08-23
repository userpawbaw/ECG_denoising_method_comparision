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


def _build_with_fe(mid: str, frontend: bool):
    """use_frontend 를 받는 팩토리에만 전달한다."""
    import inspect

    from ecgdn.registry import _REGISTRY  # noqa: PLC2701

    if frontend or mid in FE_INTRINSIC:
        return build(mid)
    fn = _REGISTRY[mid]
    try:
        params = inspect.signature(fn).parameters
        if "kw" in params or "use_frontend" in params:
            return build(mid, use_frontend=False)
    except (TypeError, ValueError):
        pass
    return build(mid)


def build_methods(cfg: dict, tag: str = "") -> dict:
    frontend = bool(cfg.get("frontend", True))
    ms: dict = {}
    for mid in cfg.get("methods", []):
        ms[mid] = _build_with_fe(mid, frontend)
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
        for it in items:                       # 잡음 없이 clean 을 그대로 입력
            it["y"] = it["x"].copy()
            it["snr"] = float("inf")
    if args.limit:
        items = items[: args.limit]

    methods = build_methods(cfg, tag)
    ev = cfg.get("eval", {})
    do_morph = bool(ev.get("do_morph", True))
    do_spec = bool(ev.get("do_spectral", True))

    print(f"[{exp_id}] mode={mode} source={src.kind} tag={tag} items={len(items)} "
          f"frontend={cfg.get('frontend', True)} methods={list(methods)}")
    rows = []
    t0 = time.perf_counter()
    for i, it in enumerate(items, 1):
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
        if i % 10 == 0 or i == len(items):
            el = time.perf_counter() - t0
            print(f"  {i}/{len(items)}  {el:.0f}s  (eta {el / i * (len(items) - i):.0f}s)",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(out / "metrics.parquet", index=False)
    save_manifest(out, cfg=cfg, extra={"n_items": len(items), "methods": list(methods)})
    print(f"\nrows={len(df)}  -> {out/'metrics.parquet'}")

    key = "snr_imp_scaled" if mode != "distortion" else "snr_out_strict"
    piv = (df[df.metric == key].pivot_table(index="method", columns="snr_in_target",
                                            values="value", aggfunc="mean").round(2))
    print(f"\n=== {key} (mean) ===")
    print(piv.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
