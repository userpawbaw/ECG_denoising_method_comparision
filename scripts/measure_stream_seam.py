#!/usr/bin/env python3
"""스트리밍 재생의 이음매와 품질 — 오프라인 경로와 직접 대조한다.

    python scripts/measure_stream_seam.py --axis d1

**무엇이 다른가.** 오프라인 경로(`dl_wrapper`)는 window 를 50 % 겹쳐
`Hann²` overlap-add 로 합친다. 그래서 한 샘플이 **두 window 의 평균**으로
나오고, 그것이 이음매를 지운다.

스트리밍에서는 그럴 수 없다. 시각 `t` 에 쓸 수 있는 window 는 **오른쪽 끝이
`t` 이하**인 것뿐이다. 샘플 `p` 를 지연 `d` 로 내보내면 `t = p + d` 이고,
그때까지 도착한 window 는 시작점이 `[p−W+1, p+d−W+1]` 인 것들뿐이다 —
**hop 이 크면 그 구간에 격자점이 하나밖에 없다.**

    W = 1024, hop = 512, d = 12  ->  겹치는 window 사실상 1 개

즉 **지연을 줄이면 overlap-add 가 사라진다.** 그러면 window 마다 독립적으로
추정한 조각을 이어 붙이게 되고 이음매가 보인다. 이 스크립트는 그 대가를
`(hop, d)` 격자에서 잰다.

재는 것 둘
----------
- **품질**: 오프라인 대비 출력 SNR 손실 [dB]
- **이음매**: 새 window 가 도착하는 지점에서의 1 차 차분이 그 주변보다
  얼마나 큰가 (배수). 1.0 이면 티가 안 나고, 크면 눈에 보인다.

정지 화면에서는 안 보이고 **움직일 때 보이는** 결함이라 수치로 재야 한다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
FS = 250.0


def _method(axis: str, ckpt: str = "m06_l1"):
    """**실제 오프라인 경로를 그대로 쓴다.** 재구현하면 다른 것을 재게 된다.

    처음엔 프레이밍과 overlap-add 를 손으로 다시 짰는데, FrontEnd·reflect
    패딩·window 별 `robust_scale` 을 빠뜨려 오프라인이 10.66 dB 로 나왔다
    (스트리밍이 14 dB 더 좋다는 불가능한 결과). 비교 대상은 **보고서가 실제로
    쓴 코드**여야 한다 — F-9 와 같은 계열이다.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from ecgdn.methods.dl_wrapper import DLDenoiser
    return DLDenoiser(ckpt=ROOT / "results" / axis / ckpt / "best.pt", name="M06")


def _segment(axis: str, src: str, seconds: float, seed: int):
    import sys
    sys.path.insert(0, str(ROOT))
    from ecgdn.data.dataset import ECGDenoiseDataset
    from ecgdn.data.sources import get_source
    n = int(seconds * FS)
    ds = ECGDenoiseDataset(source=get_source(src), split="test",
                           win=n, hop=n, max_per_record=2)
    i = int(np.random.default_rng(seed).integers(len(ds)))
    d = ds.raw_item(i)
    return d["y"].astype(np.float64), d["x"].astype(np.float64)


def streaming(meth, y: np.ndarray, d: int, H: int) -> tuple[np.ndarray, list[int]]:
    """시각 `t` 에 오른쪽 끝이 `t` 이하인 window 만 쓴다. 샘플 `p` 는 `t = p+d` 에 확정.

    **오프라인과 같은 부품을 쓴다** — 같은 FrontEnd, 같은 window 별 정규화,
    같은 Hann² 가중. 다른 것은 *언제 무엇을 쓸 수 있는가* 뿐이다.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from ecgdn.data.windows import analysis_window
    from ecgdn.utils import robust_scale

    W = meth.win
    yf = meth._fe(y, FS) if meth._fe is not None else y      # FE 는 버퍼 전체에
    w2 = analysis_window(W) ** 2
    acc = np.zeros(len(y)); wsq = np.zeros(len(y))
    out = np.zeros(len(y)); done = 0
    seams: list[int] = []
    for s in range(0, len(y) - W + 1, H):
        buf = yf[s:s + W]
        sc = robust_scale(buf) if meth.normalize else 1.0
        with torch.no_grad():
            tt = torch.from_numpy((buf / sc)[None, None].astype(np.float32))
            est = meth.model(tt)
            if isinstance(est, tuple):
                est = est[0]
            est = est[0, 0].numpy().astype(np.float64) * sc
        acc[s:s + W] += est * w2
        wsq[s:s + W] += w2
        upto = min(s + W - 1 - d + 1, len(y))
        if upto > done:
            seg = slice(done, upto)
            out[seg] = np.where(wsq[seg] > 1e-12, acc[seg] / np.maximum(wsq[seg], 1e-12), 0.0)
            if done > 0:
                seams.append(done)
            done = upto
    return out[:done], seams


def seam_stats(sig: np.ndarray, seams: list[int], half: int = 25) -> tuple[float, float]:
    """이음매가 **눈에 보이는 크기인가.**

    배수(주변 중앙값 대비)만으로는 판단이 안 된다 — ECG 는 QRS 에서 정상적으로
    큰 차분을 내므로, 주변 중앙값의 3 배여도 신호에 흔한 크기일 수 있다.
    그래서 **분포상 위치**를 함께 낸다: 이음매의 1 차 차분이 신호 전체 차분
    분포에서 몇 백분위인가. 99.9 이면 신호에 없던 크기이고, 80 이면 흔하다.

    반환 `(배수, 백분위)`.
    """
    dif = np.abs(np.diff(sig))
    if not seams or len(dif) < 4 * half:
        return float("nan"), float("nan")
    at, around = [], []
    for p in seams:
        if half < p < len(dif) - half:
            at.append(dif[p - 1])
            around.append(np.median(np.r_[dif[p-half:p-1], dif[p+1:p+half]]))
    if not at:
        return float("nan"), float("nan")
    ratio = float(np.mean(at) / max(np.mean(around), 1e-12))
    pct = float(np.mean([(dif < v).mean() * 100 for v in at]))
    return ratio, pct


def snr(x: np.ndarray, xh: np.ndarray) -> float:
    n = min(len(x), len(xh))
    x, xh = x[:n] - x[:n].mean(), xh[:n] - xh[:n].mean()
    a = float(x @ xh) / max(float(xh @ xh), 1e-20)      # 이득 보정 (scaled 규약)
    e = x - a * xh
    return 10 * np.log10(float(x @ x) / max(float(e @ e), 1e-20))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", default="d1")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/stream_seam.json")
    a = ap.parse_args()
    src = {"d0": "synthetic", "d1": "mitdb"}[a.axis]

    meth = _method(a.axis)
    W = meth.win
    grid = [(512, 12), (512, 256), (256, 12), (128, 12), (64, 12), (32, 12), (12, 12)]

    # **비교 구간을 모든 설정에 대해 고정한다.** 처음엔 행마다 `len(st)` 로
    # 잘랐는데, 그러면 오프라인 값이 행마다 달라져(23.68 / 22.58 / 21.68 …)
    # 비교가 성립하지 않는다. 실제로 그 탓에 "스트리밍이 오프라인보다 2 dB
    # 좋다" 는 불가능한 표가 나왔다.
    rows: dict = {}
    off_db: list[float] = []
    for seed in range(a.seeds):
        y, x = _segment(a.axis, src, a.seconds, seed)
        off = meth(y, FS, {})                       # 보고서가 쓰는 바로 그 경로
        sts = {}
        for H, d in grid:
            sts[(H, d)] = streaming(meth, y, d, H)
        lo = W                                       # 버퍼가 차는 구간은 제외
        hi = min(len(s) for s, _ in sts.values())
        if hi - lo < FS * 2:
            continue
        off_db.append(snr(x[lo:hi], off[lo:hi]))
        for (H, d), (st, seams) in sts.items():
            rows.setdefault((H, d), []).append(
                (snr(x[lo:hi], st[lo:hi]),
                 *seam_stats(st[lo:hi], [p - lo for p in seams if lo < p < hi])))

    print(f"=== {a.axis.upper()}  M06, {a.seconds:.0f}s x {a.seeds} 구간, "
          f"W={W} (앞 {W} 샘플 제외) ===")
    base = float(np.mean(off_db))
    print(f"오프라인(전체를 보고 처리) {base:.2f} dB — 기준\n")
    print(f"{'hop':>5s} {'지연 d':>8s} {'스트리밍':>10s} {'오프라인 대비':>13s} "
          f"{'이음매 배수':>9s} {'백분위':>10s} {'추론/초':>8s}")
    out = {"window": W, "offline_db": base, "grid": []}
    for H, d in grid:
        if (H, d) not in rows:
            continue
        r = np.array(rows[(H, d)])
        print(f"{H:5d} {d/FS*1000:7.0f}ms {r[:,0].mean():9.2f} dB "
              f"{r[:,0].mean()-base:+12.2f} dB {np.nanmean(r[:,1]):9.2f}x "
              f"{np.nanmean(r[:,2]):10.1f} {FS/H:7.1f}")
        out["grid"].append(dict(hop=H, d=d, delay_ms=d / FS * 1000,
                                streaming_db=float(r[:, 0].mean()),
                                vs_offline_db=float(r[:, 0].mean() - base),
                                seam_ratio=float(np.nanmean(r[:, 1])),
                                seam_percentile=float(np.nanmean(r[:, 2])),
                                inferences_per_sec=float(FS / H)))
    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
