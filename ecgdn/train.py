"""학습 루프 (docs/02_procedure.md STEP 18).

설계 결정
--------
  * **best 체크포인트는 val loss 가 아니라 `snr_imp_scaled` 로 고른다.**
    loss 최소가 우리가 원하는 성능의 최소는 아니다 (특히 복합 손실에서).
  * 배치 안에서 window 별 SNR 이 다르므로 지표는 window 단위로 계산해 평균낸다.
  * 재현성: dataset 이 (salt, record, window) 로 난수를 유도하므로 epoch 을 고정하면
    같은 배치가 재현된다.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import TrainCfg
from .utils import ensure_dir, save_manifest

__all__ = ["snr_metrics_torch", "Trainer", "TrainState"]


# ------------------------------------------------------------------ 지표 (torch)
def snr_metrics_torch(x: torch.Tensor, y: torch.Tensor, xhat: torch.Tensor
                      ) -> dict[str, torch.Tensor]:
    """window 단위 SNR 지표. eval/signal_metrics.py 와 동일 정의(평균 제거 + 최적 이득)."""
    def dm(v):
        return v - v.mean(dim=-1, keepdim=True)

    x0, y0, h0 = dm(x), dm(y), dm(xhat)
    eps = 1e-12
    px = (x0 ** 2).mean(-1)
    snr_in = 10 * torch.log10(px / ((y0 - x0) ** 2).mean(-1).clamp_min(eps))
    snr_st = 10 * torch.log10(px / ((h0 - x0) ** 2).mean(-1).clamp_min(eps))
    a = (x0 * h0).sum(-1) / (h0 * h0).sum(-1).clamp_min(eps)
    snr_sc = 10 * torch.log10(px / ((a[..., None] * h0 - x0) ** 2).mean(-1).clamp_min(eps))
    return {"snr_in": snr_in, "snr_imp_strict": snr_st - snr_in,
            "snr_imp_scaled": snr_sc - snr_in, "gain_bias": a,
            "rmse": ((h0 - x0) ** 2).mean(-1).sqrt()}


@dataclass
class TrainState:
    epoch: int = 0
    best_metric: float = -math.inf
    best_epoch: int = -1
    history: list[dict[str, float]] = None

    def __post_init__(self):
        if self.history is None:
            self.history = []


class Trainer:
    def __init__(self, model: torch.nn.Module, loss_fn, train_ds, val_ds,
                 cfg: TrainCfg = TrainCfg(), out_dir: str | Path = "results/run",
                 device: str | None = None, num_workers: int = 0,
                 model_name: str = "model", extra_manifest: dict | None = None):
        self.model = model
        self.loss_fn = loss_fn
        self.train_ds, self.val_ds = train_ds, val_ds
        self.cfg = cfg
        self.out = ensure_dir(out_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_workers = num_workers
        self.model_name = model_name
        self.state = TrainState()
        self.model.to(self.device)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=(self.device.type == "cuda"))
        save_manifest(self.out, cfg=asdict(cfg),
                      extra={"model": model_name,
                             "n_params": sum(p.numel() for p in model.parameters()),
                             **(extra_manifest or {})})

    # ---------------- 내부
    def _loader(self, ds, shuffle: bool):
        return DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, drop_last=shuffle,
                          collate_fn=_collate)

    def _forward(self, y):
        if hasattr(self.model, "swt"):
            return self.model(y, return_bands=True)
        return self.model(y), None, None

    def _lr_at(self, step: int, total: int) -> float:
        warm = max(1, int(self.cfg.warmup_frac * total))
        if step < warm:
            return self.cfg.lr * step / warm
        p = (step - warm) / max(1, total - warm)
        return self.cfg.lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    # ---------------- 공개
    def evaluate(self, ds) -> dict[str, float]:
        self.model.eval()
        acc: dict[str, list[float]] = {}
        with torch.no_grad():
            for y, x, meta in self._loader(ds, shuffle=False):
                y, x = y.to(self.device), x.to(self.device)
                xhat, s, s_hat = self._forward(y)
                _, parts = self.loss_fn(xhat, x, s_hat, s)
                m = snr_metrics_torch(x[:, 0], y[:, 0], xhat[:, 0])
                for k, v in m.items():
                    acc.setdefault(k, []).extend(v.detach().cpu().numpy().tolist())
                acc.setdefault("loss", []).append(float(parts["total"]))
        return {k: float(np.nanmean(v)) for k, v in acc.items()}

    def fit(self) -> TrainState:
        train_loader = self._loader(self.train_ds, shuffle=True)
        total_steps = max(1, self.cfg.epochs * len(train_loader))
        step = 0
        log_path = self.out / "log.csv"
        log_path.write_text("epoch,lr,train_loss,val_loss,val_snr_imp_scaled,"
                            "val_snr_imp_strict,val_gain_bias,secs\n")

        for ep in range(1, self.cfg.epochs + 1):
            if hasattr(self.train_ds, "set_epoch"):
                self.train_ds.set_epoch(ep)
            self.model.train()
            t0, tl, nb = time.perf_counter(), 0.0, 0
            for y, x, meta in train_loader:
                lr = self._lr_at(step, total_steps)
                for g in self.opt.param_groups:
                    g["lr"] = lr
                y, x = y.to(self.device), x.to(self.device)
                self.opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(self.device.type,
                                        enabled=(self.device.type == "cuda")):
                    xhat, s, s_hat = self._forward(y)
                    loss, _ = self.loss_fn(xhat, x, s_hat, s)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
                tl += float(loss.detach()); nb += 1; step += 1

            vm = self.evaluate(self.val_ds)
            secs = time.perf_counter() - t0
            row = dict(epoch=ep, lr=lr, train_loss=tl / max(nb, 1),
                       val_loss=vm["loss"], val_snr_imp_scaled=vm["snr_imp_scaled"],
                       val_snr_imp_strict=vm["snr_imp_strict"],
                       val_gain_bias=vm["gain_bias"], secs=secs)
            self.state.history.append(row)
            with log_path.open("a") as f:
                f.write(",".join(f"{row[k]:.6f}" if isinstance(row[k], float) else str(row[k])
                                 for k in ("epoch", "lr", "train_loss", "val_loss",
                                           "val_snr_imp_scaled", "val_snr_imp_strict",
                                           "val_gain_bias", "secs")) + "\n")
            print(f"ep {ep:3d}  train {row['train_loss']:.5f}  val {row['val_loss']:.5f}  "
                  f"snr_imp {row['val_snr_imp_scaled']:+6.2f} dB  gain {row['val_gain_bias']:.3f}  "
                  f"{secs:.1f}s", flush=True)

            # best 선택은 **지표** 기준
            if vm["snr_imp_scaled"] > self.state.best_metric:
                self.state.best_metric = vm["snr_imp_scaled"]
                self.state.best_epoch = ep
                self.save("best.pt")
            self.save("last.pt")
            self.state.epoch = ep

            if ep - self.state.best_epoch >= self.cfg.patience:
                print(f"early stop at epoch {ep} (best {self.state.best_epoch}: "
                      f"{self.state.best_metric:+.2f} dB)")
                break

        (self.out / "history.json").write_text(json.dumps(self.state.history, indent=2))
        return self.state

    def save(self, name: str) -> Path:
        p = self.out / name
        torch.save({"model": self.model.state_dict(), "model_name": self.model_name,
                    "epoch": self.state.epoch, "best_metric": self.state.best_metric,
                    "cfg": asdict(self.cfg)}, p)
        return p


def _collate(batch):
    ys = torch.from_numpy(np.stack([b[0] for b in batch]))
    xs = torch.from_numpy(np.stack([b[1] for b in batch]))
    meta = {k: np.array([b[2][k] for b in batch]) for k in batch[0][2]}
    return ys, xs, meta
