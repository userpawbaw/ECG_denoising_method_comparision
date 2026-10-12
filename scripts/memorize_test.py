"""D-38 E1: 암기 시험 — 용량을 **쓸 수 있는가**를 잰다.

    python scripts/memorize_test.py -c configs/m06_l1.yaml --out results/ext/d38_e1

정상 학습에서는 학습 손실이 폭과 무관하게 0.0012~0.0013 에서 멈췄다(F-42).
용량이 모자라서인지, 최적화가 용량을 못 써서인지, 과제에 예측할 수 없는 바닥이
있어서인지를 이것만으로는 못 가른다. 여기서는 **외울 수 있는 과제**를 준다 —
창 256 개와 **한 번 뽑아 고정한 잡음**. 그러면 입력이 목표를 완전히 정하므로
바닥이 없다. 큰 모델이 이것을 외우면 (가) 「최적화가 막혀 있다」가 기각된다.

100 step 마다 세 가지를 잰다 (`docs/21_decisions.md` D-38):

  ① fixed  — 외우는 그 256 개 (같은 잡음)
  ② fresh  — **같은 ECG 창에 새 잡음** (`set_epoch` 으로 다른 뽑기)
  ③ val    — val split 창 256 개

① 만 내려가고 ② 가 안 내려가면 잡음을 외운 것이고, ② 도 내려가면 ECG 를 외운
것이다. ③ 은 그 외움이 일반화를 해치는지 본다.
"""
import _bootstrap  # noqa: F401

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from ecgdn.models import build_model, make_loss
from ecgdn.train import snr_metrics_torch
from train import build_datasets, load_cfg

# 새 잡음 뽑기에 쓰는 epoch. 정상 학습(0..59)과 겹치지 않으면 무엇이든 된다.
FRESH_EPOCH = 10_007


def pick(n_total: int, n: int, seed: int) -> list[int]:
    """고정 rng 로 창 n 개를 고른다 — 기록 전체에 고르게 퍼지게 정렬해 둔다."""
    g = np.random.default_rng(seed)
    return sorted(int(i) for i in g.choice(n_total, size=min(n, n_total), replace=False))


def stack(ds, idx: list[int]) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """(noisy, 참조, scale). 둘 다 **noisy 의 robust scale 로 나눈** 값이다 —
    그래서 잡음을 다시 뽑으면 정규화된 참조도 조금 바뀐다. 참조 자체가 같은지는
    `x * scale` 로 본다."""
    ys, xs, sc = [], [], []
    for i in idx:
        y, x, meta = ds[i]
        ys.append(y)
        xs.append(x)
        sc.append(float(meta["scale"]))
    return torch.from_numpy(np.stack(ys)), torch.from_numpy(np.stack(xs)), np.asarray(sc)


@torch.no_grad()
def evaluate(model, y: torch.Tensor, x: torch.Tensor, bs: int = 64) -> dict:
    model.eval()
    se, imp = [], []
    for s in range(0, y.shape[0], bs):
        yh = model(y[s:s + bs])
        se.append(((yh - x[s:s + bs]) ** 2).mean(dim=(1, 2)))
        imp.append(snr_metrics_torch(x[s:s + bs, 0], y[s:s + bs, 0], yh[:, 0])["snr_imp_scaled"])
    model.train()
    return dict(mse=float(torch.cat(se).mean()), snr_imp=float(torch.cat(imp).mean()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="arm 의 yaml")
    ap.add_argument("--out", default="results/ext/d38_e1", help="결과 폴더")
    ap.add_argument("--source", default="mitdb", choices=("mitdb", "synthetic"),
                    help="데이터축. D-38 E1 은 D1(mitdb) 에서 잰다")
    ap.add_argument("--n", type=int, default=256, help="외울 창 수")
    ap.add_argument("--steps", type=int, default=2000,
                    help="학습 step 수. early stop 은 없다 — 외울 때까지 돈다")
    ap.add_argument("--batch", type=int, default=64, help="배치 크기 (정상 학습과 같다)")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="warmup 뒤 고정 학습률 (정상 학습의 최고값과 같다)")
    ap.add_argument("--warmup", type=int, default=100, help="선형 warmup step 수")
    ap.add_argument("--every", type=int, default=100, help="평가 간격 (step)")
    ap.add_argument("--seed", type=int, default=0,
                    help="창 고르기·배치 순서·초기화의 seed")
    ap.add_argument("--threads", type=int, default=4, help="torch CPU 스레드 수")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    cfg = load_cfg(args.config)
    cfg.setdefault("data", {})["source"] = args.source
    arm = Path(args.config).stem
    out = Path(args.out) / arm
    if (out / "summary.json").exists():
        print(f"[{arm}] 이미 끝났다 — {out / 'summary.json'}")
        return 0
    out.mkdir(parents=True, exist_ok=True)

    _, tr, va = build_datasets(cfg)
    idx = pick(len(tr), args.n, args.seed)
    # ① 고정 잡음: 데이터셋을 만든 그대로(set_epoch 안 부름) 한 번 뽑는다
    y_fix, x_fix, s_fix = stack(tr, idx)
    # ② 같은 ECG, 새 잡음
    tr.set_epoch(FRESH_EPOCH)
    y_new, x_new, s_new = stack(tr, idx)
    # ③ val
    y_val, x_val, _ = stack(va, pick(len(va), args.n, args.seed))
    raw_fix = x_fix.numpy() * s_fix[:, None, None]
    raw_new = x_new.numpy() * s_new[:, None, None]
    assert np.allclose(raw_fix, raw_new, rtol=1e-5, atol=1e-6), \
        "참조(물리 스케일)가 잡음 뽑기에 따라 바뀌었다 — ①②가 같은 ECG 라는 전제가 무너진다"
    assert not torch.equal(y_fix, y_new), "새 잡음이 이전 잡음과 같다 — ② 가 ① 의 복사본이다"

    mcfg = cfg.get("model", {})
    model = build_model(mcfg.get("name", "resunet1d"), **(mcfg.get("kwargs") or {}))
    n_params = sum(p.numel() for p in model.parameters())
    loss_fn = make_loss(cfg.get("loss", "L1"))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(args.seed)
    print(f"[{arm}] params={n_params:,} n={len(idx)} steps={args.steps} "
          f"source={args.source}")

    rows, t0 = [], time.time()
    log_path = out / "log.csv"
    with log_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "lr", "train_loss", "fixed_mse", "fresh_mse", "val_mse",
                    "fixed_snr_imp", "fresh_snr_imp", "val_snr_imp", "wall_s"])
        model.train()
        run = 0.0
        for step in range(1, args.steps + 1):
            lr = args.lr * min(1.0, step / max(args.warmup, 1))
            for pg in opt.param_groups:
                pg["lr"] = lr
            b = torch.randint(0, y_fix.shape[0], (args.batch,), generator=g)
            opt.zero_grad(set_to_none=True)
            loss, _ = loss_fn(model(y_fix[b]), x_fix[b], None, None)
            if not torch.isfinite(loss):
                print(f"[{arm}] step {step}: 손실이 유한하지 않다 — 중단 (D-38 되돌림 조건)")
                return 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += float(loss.detach())
            if step % args.every == 0 or step == args.steps:
                ef, en, ev = (evaluate(model, y_fix, x_fix), evaluate(model, y_new, x_new),
                              evaluate(model, y_val, x_val))
                r = dict(step=step, lr=lr, train_loss=run / args.every,
                         fixed_mse=ef["mse"], fresh_mse=en["mse"], val_mse=ev["mse"],
                         fixed_snr_imp=ef["snr_imp"], fresh_snr_imp=en["snr_imp"],
                         val_snr_imp=ev["snr_imp"], wall_s=time.time() - t0)
                rows.append(r)
                w.writerow([r[k] for k in ["step", "lr", "train_loss", "fixed_mse",
                                           "fresh_mse", "val_mse", "fixed_snr_imp",
                                           "fresh_snr_imp", "val_snr_imp", "wall_s"]])
                f.flush()
                run = 0.0
                print(f"  step {step:5d}  fixed {ef['mse']:.5f}  fresh {en['mse']:.5f}  "
                      f"val {ev['mse']:.5f}  | imp {ef['snr_imp']:+.2f} / "
                      f"{en['snr_imp']:+.2f} / {ev['snr_imp']:+.2f} dB  "
                      f"{time.time() - t0:.0f}s", flush=True)

    last = rows[-1]
    summary = dict(arm=arm, n_params=n_params, n=len(idx), steps=args.steps,
                   lr=args.lr, weight_decay=0.0, batch=args.batch, source=args.source,
                   torch=torch.__version__, seed=args.seed,
                   final=last, min_fixed_mse=min(r["fixed_mse"] for r in rows),
                   wall_s=time.time() - t0)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[{arm}] 끝 — fixed {last['fixed_mse']:.5f} · fresh {last['fresh_mse']:.5f} · "
          f"val {last['val_mse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
