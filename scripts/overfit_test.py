"""STEP 17 DoD: 배치 1개를 과적합시켜 모델/학습 루프의 버그를 잡는다.

대규모 학습 전에 반드시 통과해야 한다. 통과하지 못하면 모델이나 루프에 버그가 있다.
"""
import _bootstrap  # noqa: F401

import argparse

import numpy as np
import torch

from ecgdn.data.dataset import ECGDenoiseDataset
from ecgdn.data.sources import SyntheticSource
from ecgdn.models import build_model, make_loss
from ecgdn.train import snr_metrics_torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resunet1d")
    ap.add_argument("--loss", default="L1")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--target", type=float, default=1e-4)
    args = ap.parse_args()

    torch.manual_seed(0)
    ds = ECGDenoiseDataset(SyntheticSource(n_train=2, n_val=1, n_test=1, dur_s=60.0),
                           "train", salt=0)
    idx = list(range(args.batch))
    y = torch.from_numpy(np.stack([ds[i][0] for i in idx]))
    x = torch.from_numpy(np.stack([ds[i][1] for i in idx]))

    model = build_model(args.model)
    loss_fn = make_loss(args.loss)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def fwd(v):
        if hasattr(model, "swt"):
            return model(v, return_bands=True)
        return model(v), None, None

    model.train()
    first = None
    for i in range(args.steps):
        opt.zero_grad(set_to_none=True)
        xhat, s, sh = fwd(y)
        loss, _ = loss_fn(xhat, x, sh, s)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if first is None:
            first = float(loss)
        if (i + 1) % max(1, args.steps // 10) == 0:
            print(f"  step {i+1:4d}  loss {float(loss):.3e}")

    model.eval()
    with torch.no_grad():
        xhat, *_ = fwd(y)
        m = snr_metrics_torch(x[:, 0], y[:, 0], xhat[:, 0])
    final = float(loss)
    ok = final < args.target
    print(f"\nmodel={args.model} loss={args.loss}  first={first:.3e} -> final={final:.3e} "
          f"(target < {args.target:.0e})")
    print(f"snr_imp_scaled = {float(m['snr_imp_scaled'].mean()):+.2f} dB, "
          f"gain_bias = {float(m['gain_bias'].mean()):.4f}")
    print("[PASS]" if ok else "[FAIL] — 모델 또는 학습 루프에 버그가 있다.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
