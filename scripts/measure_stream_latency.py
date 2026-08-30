#!/usr/bin/env python3
"""스트리밍 출력 지연 대 품질 — **재학습 없이** 기존 체크포인트로 잰다.

    python scripts/measure_stream_latency.py            # 양축
    python scripts/measure_stream_latency.py --axis d1

**왜 이 곡선이 필요한가.** 실시간 시연에서 시각 `t` 에 가진 것은
`[t-W+1, t]` 뿐이다. 모델은 window 전체(W 개)에 대한 추정을 내지만 **끝에
가까운 샘플일수록 미래 문맥이 없다.** 샘플 `t-d` 를 내보내면

    출력 지연 = d / fs,   그 샘플이 가진 미래 문맥 = d 샘플

이 된다. 즉 이 곡선이 **"지연을 얼마나 주면 품질이 회복되는가"** 를 직접
답하고, 그것이 시연 구조(실시간 표시 vs 지연 표시)를 정한다.

**window 길이는 지연이 아니다.** 4.096 s 는 *문맥* 이고, 버퍼가 한 번 차고
나면 정상 상태 지연은 `d + hop/2 + 추론시간` 이다. 처음에 이것을 window 길이와
혼동했는데(대화 기록), 그러면 4 초 지연이라는 잘못된 결론이 나온다.

측정 주의
---------
- **위치별 오차는 한 샘플만 보면 분산이 지배한다.** 인접 ±`half` 샘플을
  평균한다. 이때 **경계에서 0 패딩을 하면 안 된다** — `np.convolve(mode="same")`
  은 끝에서 0 을 섞어 오차를 과소평가하고, 그러면 정작 궁금한 가장자리가
  실제보다 좋아 보인다. 유효 표본만 평균한다.
- seed 를 바꿔 재현성을 함께 본다. D0 는 표준편차가 1.9 dB 라 단일 점을
  과하게 읽으면 안 된다.
- 공통 front-end 의 가장자리 효과(F-4)도 이 숫자에 **포함**돼 있다 —
  데이터셋이 window 마다 FE 를 적용하기 때문이다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
FS = 250.0
DS = [0, 5, 12, 25, 50, 100, 200, 400]


def curve(axis: str, src: str, ckpt: str, name: str, *, n=512, seed=0, half=8):
    import sys
    sys.path.insert(0, str(ROOT))
    from ecgdn.data.dataset import ECGDenoiseDataset
    from ecgdn.data.sources import get_source
    from ecgdn.models import build_model

    sd = torch.load(ROOT / "results" / axis / ckpt / "best.pt",
                    map_location="cpu", weights_only=False)
    m = build_model(name=name); m.load_state_dict(sd.get("model", sd)); m.eval()
    ds = ECGDenoiseDataset(source=get_source(src), split="test", max_per_record=60)
    n = min(n, len(ds))
    idx = np.random.default_rng(seed).choice(len(ds), n, replace=False)

    errs, sig_sum, cnt = [], 0.0, 0
    for s0 in range(0, n, 32):
        Y, X = [], []
        for i in idx[s0:s0 + 32]:
            d = ds.raw_item(int(i)); Y.append(d["y"][None]); X.append(d["x"][None])
        y = torch.from_numpy(np.stack(Y)).float()
        x = torch.from_numpy(np.stack(X)).float()
        with torch.no_grad():
            xh = m(y)
        errs.append(((xh - x) ** 2).squeeze(1).numpy())
        sig_sum += float((x ** 2).mean()) * len(Y); cnt += len(Y)

    err = np.concatenate(errs, 0)
    sig = sig_sum / cnt
    W = err.shape[1]
    per_pos = err.mean(axis=0)

    # 경계를 0 으로 채우지 않는 이동평균 (위 독스트링 참조)
    c = np.cumsum(np.concatenate([[0.0], per_pos]))
    lo = np.maximum(np.arange(W) - half, 0)
    hi = np.minimum(np.arange(W) + half + 1, W)
    sm = (c[hi] - c[lo]) / (hi - lo)
    return W, 10 * np.log10(sig / sm)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=("d0", "d1", "both"), default="both")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/stream_latency.json")
    a = ap.parse_args()

    axes = [("d0", "synthetic"), ("d1", "mitdb")]
    if a.axis != "both":
        axes = [x for x in axes if x[0] == a.axis]

    out = {}
    for axis, src in axes:
        rows = []
        for seed in range(a.seeds):
            W, s = curve(axis, src, "m06_l1", "resunet1d", seed=seed)
            mid = float(np.median(s[W // 4:3 * W // 4]))
            rows.append([float(s[W - 1 - d] - mid) for d in DS] + [mid])
        arr = np.array(rows)
        out[axis] = dict(window=W, ds=DS, delay_ms=[d / FS * 1000 for d in DS],
                         rel_mean=arr[:, :-1].mean(0).tolist(),
                         rel_std=arr[:, :-1].std(0).tolist(),
                         mid_db=float(arr[:, -1].mean()))
        print(f"\n=== {axis.upper()}  M06, window {W} ({W/FS:.2f} s), "
              f"seed {a.seeds}개 ===")
        print(f"{'끝에서 d':>9s} {'지연':>9s} {'중앙부 대비':>12s} {'표준편차':>9s}")
        for i, d in enumerate(DS):
            print(f"{d:9d} {d/FS*1000:8.0f}ms {arr[:, i].mean():+11.2f} dB "
                  f"{arr[:, i].std():8.2f}")
        print(f"{'중앙부':>9s} {'':>9s} {out[axis]['mid_db']:11.2f} dB (절대)")

    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
