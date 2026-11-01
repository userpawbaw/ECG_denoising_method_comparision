#!/usr/bin/env python3
"""D-42 — 오검출 자리의 출력이 **심박처럼 생겼는가** (없는 심박을 그리는가).

    python scripts/probe_phantom_beats.py --source mitdb
    python scripts/probe_phantom_beats.py --source synthetic

산출: `results/{tag}/phantom/`
    rows.parquet   오검출 하나당 한 줄 (기록 · 잡음 · SNR · 방법 · 위치 · 상관)
    counts.parquet 구간·방법마다 참 검출 · 오검출 수
    summary.json   방법별 요약 (임계 0.7 / 0.8 / 0.9, 기록 집중도)
    example.npz    그림용 발췌 하나 — **`M05` 의 심박 모양 가짜가 가장 많이 몰린 4 초**
                   (같으면 상관 평균이 높은 쪽). 「상관 최고 한 점」으로 고르면 빈자리
                   가짜가 아니라 **진짜 beat 를 옮겨 그린** 경우가 뽑힌다 — 처음 그렇게
                   골랐다가 바꿨다. 어느 쪽이든 고른 규칙을 그림에 적는다 (F-24).

왜 에너지가 아니라 모양인가
-------------------------
안전성 프로브의 `halluc_energy` 와 EXP-G 의 `fp` 는 **잡음이 남은 것**과
**심박을 그린 것**을 구별하지 못한다. 칼만(`M05`)은 위상을 잡음 섞인 입력에서
검출한 R-peak 로 매기고 그 위상 위에 template 를 그리므로, 검출이 틀리면 그 자리에
**깨끗한 QRS** 를 그릴 수 있다 — 개수로는 front-end 와 같아도 모양이 다르다.

기준 template 은 **그 출력 자신의 참 beat 평균**이다(참값 template 이 아니다).
방법마다 진폭·형태가 달라 「이 출력에서 진짜 beat 는 어떻게 생겼나」가 기준이어야
한다 (D-42 선택지 B).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

import ecgdn.methods  # noqa: F401
from ecgdn.data.dataset import build_eval_set
from ecgdn.data.nstdb import make_banks
from ecgdn.data.sources import get_source, source_tag
from ecgdn.eval.rpeak import detect_rpeaks
from ecgdn.methods.dl_wrapper import DLDenoiser
from ecgdn.registry import build
from ecgdn.utils import ensure_dir, save_manifest

HALF_S = 0.20          # 비교 창 ±200 ms
MATCH_S = 0.075        # 참 beat 와의 매칭 허용 (평가와 같다)
THRESHOLDS = (0.7, 0.8, 0.9)
MAIN_THR = 0.8


def _methods(tag: str) -> dict:
    ms = {m: build(m) for m in ("M_FE", "M04", "M05")}
    ms["M06"] = DLDenoiser(ckpt=f"results/{tag}/m06_l1/best.pt", name="M06", batch=32)
    return ms


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _write_example(out: Path, tag: str, df: pd.DataFrame, a) -> None:
    """그림용 발췌 — `M05` 의 심박 모양 가짜가 **가장 많이 몰린 4 초** 창."""
    b = df[(df.method == "M05") & (df.cc > MAIN_THR)]
    if b.empty:
        return
    best = None                                  # (개수, 상관 평균, 기록, 잡음, SNR, 중심)
    for (rec, cond, snr), g in b.groupby(["record", "cond", "snr"]):
        idx = np.sort(g.idx.to_numpy())
        for c0 in idx:
            m = np.abs(idx - c0) <= 250          # ±1 s 안의 가짜 (fs 250 기준, 창은 ±2 s)
            key = (int(m.sum()), float(g.cc.to_numpy()[np.argsort(g.idx.to_numpy())][m].mean()))
            cand = (*key, rec, cond, snr, int(np.round(idx[m].mean())))
            if best is None or cand[:2] > best[:2]:
                best = cand
    n_in, ccm, rec, cond, snr, center = best
    src = get_source(a.source, dur_s=300.0, n_test=22)
    banks = make_banks("test", "data/raw/nstdb")
    items = build_eval_set(src, "test", seg_s=60.0, snr_grid=[snr], noise_conditions=(cond,),
                           banks=banks, n_seg_per_record=1, seed="phantom")
    it = next(i for i in items if str(i["record"]) == rec)
    y, x, fs = it["y"].astype(float), it["x"].astype(float), float(it["fs"])
    i0, i1 = max(0, center - int(2 * fs)), min(len(y), center + int(2 * fs))
    ref = np.asarray(it["r_peaks"], dtype=int)
    ex = dict(fs=fs, i0=i0, center=center, n_in_window=n_in, cc=ccm, record=rec, cond=cond,
              snr=snr, clean=x[i0:i1], noisy=y[i0:i1], ref_peaks=ref[(ref >= i0) & (ref < i1)] - i0,
              rule="M05 심박 모양 가짜가 가장 많이 몰린 4 초 (같으면 상관 평균이 높은 쪽)")
    for mid, fn in _methods(tag).items():
        xh = np.asarray(fn(y, fs, {}), dtype=float)
        det = np.asarray(detect_rpeaks(xh, fs), dtype=int)
        ex[mid] = xh[i0:i1]
        ex[f"det_{mid}"] = det[(det >= i0) & (det < i1)] - i0
    np.savez(out / "example.npz", **ex)
    print(f"  example: {rec} {cond} {snr:+.0f} dB, 가짜 {n_in} 개 / 4 s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", required=True, choices=("synthetic", "mitdb"),
                    help="데이터축 — synthetic(D0) 또는 mitdb(D1). 명시해야 한다")
    ap.add_argument("--snrs", type=float, nargs="+", default=[-5.0, 0.0],
                    help="입력 SNR 격자 [dB]. 오검출이 생기는 저 SNR 이 기본이다 (D-42)")
    ap.add_argument("--conds", nargs="+", default=["awgn", "mixed"],
                    help="잡음 조건 (configs 의 잡음 이름)")
    ap.add_argument("--example-only", action="store_true",
                    help="측정은 건너뛰고 기존 rows.parquet 에서 그림용 발췌(example.npz)만 다시 고른다")
    ap.add_argument("--limit", type=int, default=None,
                    help="앞에서 N 개 구간만 (빠른 확인용 — 산출물을 덮어쓰니 주의)")
    a = ap.parse_args()

    tag = source_tag(a.source)
    out = ensure_dir(Path("results") / tag / "phantom")
    if a.example_only:
        df = pd.read_parquet(out / "rows.parquet")
        _write_example(out, tag, df, a)
        return 0
    src = get_source(a.source, dur_s=300.0, n_test=22)
    banks = make_banks("test", "data/raw/nstdb")
    items = build_eval_set(src, "test", seg_s=60.0, snr_grid=a.snrs,
                           noise_conditions=tuple(a.conds), banks=banks,
                           n_seg_per_record=1, seed="phantom")
    if a.limit:
        items = items[: a.limit]
    ms = _methods(tag)

    rows, counts = [], []
    for it in items:
        y, x, fs = it["y"].astype(float), it["x"].astype(float), float(it["fs"])
        ref = np.asarray(it["r_peaks"], dtype=int)
        h, tol = int(round(HALF_S * fs)), int(round(MATCH_S * fs))
        inner = ref[(ref > h) & (ref < len(y) - h)]
        for mid, fn in ms.items():
            xh = np.asarray(fn(y, fs, {}), dtype=float)
            det = np.asarray(detect_rpeaks(xh, fs), dtype=int)
            det = det[(det > h) & (det < len(xh) - h)]
            dist = (np.abs(det[:, None] - ref[None, :]).min(1) if ref.size
                    else np.full(det.size, 10 ** 9))
            fp = det[dist > tol]
            counts.append(dict(record=str(it["record"]), cond=it["cond"], snr=it["snr"],
                               method=mid, n_det=int(det.size), n_fp=int(fp.size),
                               n_tp=int((dist <= tol).sum()), n_ref=int(inner.size)))
            if inner.size < 3:
                continue
            tmpl = np.stack([xh[r - h:r + h] for r in inner]).mean(0)
            for f in fp:
                c = _corr(xh[f - h:f + h], tmpl)
                rows.append(dict(record=str(it["record"]), cond=it["cond"], snr=it["snr"],
                                 method=mid, idx=int(f), cc=c))
        print(f"  {it['record']} {it['cond']} {it['snr']:+.0f} dB", flush=True)

    df = pd.DataFrame(rows)
    cn = pd.DataFrame(counts)
    df.to_parquet(out / "rows.parquet", index=False)
    cn.to_parquet(out / "counts.parquet", index=False)

    summ = {}
    for mid in ms:
        d = df[df.method == mid]
        c = cn[cn.method == mid]
        beat = d[d.cc > MAIN_THR]
        per_rec = beat.groupby("record").size().sort_values(ascending=False)
        summ[mid] = dict(
            n_fp=int(c.n_fp.sum()), n_tp=int(c.n_tp.sum()), n_ref=int(c.n_ref.sum()),
            cc_median=float(d.cc.median()) if len(d) else None,
            beatlike={str(t): int((d.cc > t).sum()) for t in THRESHOLDS},
            beatlike_frac=float((d.cc > MAIN_THR).mean()) if len(d) else None,
            beatlike_records=int(per_rec.size),
            beatlike_top2_share=(float(per_rec.iloc[:2].sum() / per_rec.sum())
                                 if per_rec.sum() else None),
        )
    json.dump(dict(axis=tag, snrs=a.snrs, conds=a.conds, n_segments=len(items),
                   threshold=MAIN_THR, methods=summ),
              open(out / "summary.json", "w"), indent=1, ensure_ascii=False)

    _write_example(out, tag, df, a)

    save_manifest(out, cfg=vars(a), sources=[
        "scripts/probe_phantom_beats.py", "ecgdn/methods/kalman_sameni.py",
        "ecgdn/eval/rpeak.py", "ecgdn/data/dataset.py"])
    print(json.dumps(summ, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
