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
    step: int = 0                      # LR 스케줄 위치. 재개 시 복원해야 한다
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
        step = self.state.step
        start_ep = self.state.epoch + 1
        log_path = self.out / "log.csv"
        if start_ep == 1 or not log_path.exists():
            log_path.write_text("epoch,lr,train_loss,val_loss,val_snr_imp_scaled,"
                                "val_snr_imp_strict,val_gain_bias,secs\n")
        if start_ep > self.cfg.epochs:
            print(f"이미 {self.state.epoch} epoch 까지 끝났다 (목표 {self.cfg.epochs}). "
                  f"더 돌리려면 --epochs 를 늘릴 것.")
            return self.state

        lr = self.cfg.lr           # 재개 직후 첫 로그 줄을 위한 초기값
        for ep in range(start_ep, self.cfg.epochs + 1):
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

            # 체크포인트에 기록될 epoch/step 을 **저장 전에** 갱신한다.
            # (이전에는 save 뒤에 갱신해서 두 체크포인트의 epoch 이 1 작았다)
            self.state.epoch = ep
            self.state.step = step

            # best 선택은 **지표** 기준
            if vm["snr_imp_scaled"] > self.state.best_metric:
                self.state.best_metric = vm["snr_imp_scaled"]
                self.state.best_epoch = ep
                self.save("best.pt")
            self.save("last.pt")

            if ep - self.state.best_epoch >= self.cfg.patience:
                print(f"early stop at epoch {ep} (best {self.state.best_epoch}: "
                      f"{self.state.best_metric:+.2f} dB)")
                break

        (self.out / "history.json").write_text(json.dumps(self.state.history, indent=2))
        return self.state

    def save(self, name: str) -> Path:
        """`last.pt` 에는 **재개에 필요한 전부**를 담는다.

        `best.pt` 는 추론용이라 모델 가중치만 넣는다 — optimizer 상태까지 넣으면
        파일이 두 배가 되는데 추론에서는 쓰이지 않는다.
        """
        p = self.out / name
        ck = {"model": self.model.state_dict(), "model_name": self.model_name,
              "epoch": self.state.epoch, "best_metric": self.state.best_metric,
              "cfg": asdict(self.cfg)}
        if name == "last.pt":
            ck["opt"] = self.opt.state_dict()
            ck["scaler"] = self.scaler.state_dict()
            ck["state"] = {"best_epoch": self.state.best_epoch,
                           "step": self.state.step,
                           "history": self.state.history}
        torch.save(ck, p)
        return p

    def _recover_from_log(self) -> tuple[int, list[dict]]:
        """`log.csv` 에서 history 와 best epoch 을 되살린다.

        체크포인트가 무엇을 담고 있든 `log.csv` 는 매 epoch append 되므로
        학습 이력의 진실 원천이다.
        """
        p = self.out / "log.csv"
        if not p.exists():
            return -1, []
        import csv
        try:
            rows = [{k: float(v) for k, v in r.items()}
                    for r in csv.DictReader(p.open())]
        except (ValueError, KeyError):
            return -1, []
        if not rows:
            return -1, []
        for r in rows:
            r["epoch"] = int(r["epoch"])
        best = max(rows, key=lambda r: r["val_snr_imp_scaled"])
        print(f"[resume] log.csv 에서 이력 복원: {len(rows)} epoch, "
              f"best {best['epoch']} ({best['val_snr_imp_scaled']:+.3f} dB)")
        return best["epoch"], rows

    def try_resume(self, path: str | Path | None = None) -> bool:
        """`last.pt` 에서 이어서 학습한다. 없으면 False 를 돌려주고 처음부터 간다.

        모델 가중치만 복원하는 것으로는 부족하다. optimizer 의 모멘텀,
        LR 스케줄의 위치(`step`), early stopping 이 보는 `best_epoch` 까지
        되돌리지 않으면 '이어서' 가 아니라 '다른 학습' 이 된다.

        **한계**: LR 스케줄은 `total_steps = epochs x len(loader)` 로 정규화되므로,
        재개할 때 `epochs` 를 바꾸면 남은 구간의 LR 곡선이 원래와 달라진다.
        중단분을 그대로 이어붙이려면 `epochs` 를 바꾸지 말 것.
        """
        p = Path(path) if path else (self.out / "last.pt")
        if not p.exists():
            return False
        ck = torch.load(p, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        if "opt" in ck:
            self.opt.load_state_dict(ck["opt"])
        if "scaler" in ck:
            self.scaler.load_state_dict(ck["scaler"])
        st = ck.get("state") or {}
        self.state.epoch = int(ck.get("epoch", 0))
        self.state.best_metric = float(ck.get("best_metric", -math.inf))
        self.state.step = int(st.get("step", 0))
        self.state.history = list(st.get("history") or [])

        if "best_epoch" in st:
            self.state.best_epoch = int(st["best_epoch"])
        else:
            # 구버전 체크포인트에는 best_epoch 가 없다. epoch 으로 대신하면
            # early stopping 이 그 지점부터 다시 세기 시작해 이미 정체된 학습을
            # patience 만큼 더 돌린다 (실제로 m06_l1 이 12 epoch 을 낭비했다).
            # log.csv 는 매 epoch append 되므로 언제나 정확하다 — 거기서 되살린다.
            self.state.best_epoch, self.state.history = self._recover_from_log()
            if self.state.best_epoch < 0:
                self.state.best_epoch = self.state.epoch
                print("[resume] best_epoch 을 복원할 수 없다 (log.csv 없음). "
                      "early stopping 이 현재 epoch 부터 다시 센다.")

        if "opt" not in ck:
            print(f"[resume] {p} 에 optimizer 상태가 없다 (구버전 체크포인트). "
                  f"가중치만 복원하므로 모멘텀과 LR 위치는 초기화된다.")
        print(f"[resume] epoch {self.state.epoch} 까지 완료된 상태에서 재개 "
              f"(best {self.state.best_epoch}: {self.state.best_metric:+.3f} dB)")
        return True


def _collate(batch):
    ys = torch.from_numpy(np.stack([b[0] for b in batch]))
    xs = torch.from_numpy(np.stack([b[1] for b in batch]))
    meta = {k: np.array([b[2][k] for b in batch]) for k in batch[0][2]}
    return ys, xs, meta
