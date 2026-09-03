"""STEP 18: 학습 진입점. 실험 1개 = yaml 1개.

    python scripts/train.py -c configs/m06_l1.yaml --source synthetic
    python scripts/train.py -c configs/m06_l1.yaml --source mitdb
    python scripts/train.py -c configs/m06_l1.yaml --epochs 5      # 빠른 확인
    python scripts/train.py -c configs/m06_l1.yaml --source mitdb --resume   # 중단분 이어서

산출물은 **데이터축별로 분리**된다: `results/d0/<exp_id>` (합성),
`results/d1/<exp_id>` (MIT-BIH). 같은 경로를 쓰면 D0/D1 결과가 서로를 덮어쓰고,
나중에 표에서 어느 쪽 숫자인지 구분할 수 없다.
"""
import _bootstrap  # noqa: F401

import argparse
import copy
from pathlib import Path

import torch
import yaml

from ecgdn.config import TrainCfg
from ecgdn.data.dataset import ECGDenoiseDataset
from ecgdn.data.nstdb import make_banks
from ecgdn.data.sources import get_source, resolve_source_kind, source_tag
from ecgdn.models import build_model, make_loss
from ecgdn.train import Trainer
from ecgdn.utils import ensure_dir


def load_cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def build_datasets(cfg: dict):
    d = cfg.get("data", {})
    src = get_source(d.get("source", "auto"),
                     dur_s=float(d.get("dur_s", 300.0)),
                     n_train=int(d.get("n_train", 18)),
                     n_val=int(d.get("n_val", 4)),
                     n_test=int(d.get("n_test", 22)))
    nstdb_root = d.get("nstdb_root", "data/raw/nstdb")
    kw = dict(win=int(d.get("win", 1024)), hop=int(d.get("hop", 512)),
              snr_range=tuple(d.get("snr_range", (-5.0, 20.0))),
              max_per_record=d.get("max_per_record"),
              pre_denoise=d.get("pre_denoise"),
              frontend=bool(d.get("frontend", True)),
              # **안 적으면 예전과 같다** — `frontend` 를 따라간다.
              # 명시하면 «입력은 날것, 목표는 FE 통과» 를 만들 수 있다.
              ref_frontend=bool(d.get("ref_frontend", d.get("frontend", True))))
    tr = ECGDenoiseDataset(src, "train", banks=make_banks("train", nstdb_root),
                           salt=("train", cfg.get("seed", 0)), **kw)
    va = ECGDenoiseDataset(src, "val", banks=make_banks("val", nstdb_root),
                           salt=("val", cfg.get("seed", 0)), **kw)
    return src, tr, va


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--source", default=None, choices=("auto", "synthetic", "mitdb"),
                    help="config 의 data.source 를 덮어쓴다. 재현성을 위해 명시를 권한다")
    ap.add_argument("--resume", action="store_true",
                    help="출력 디렉터리의 last.pt 에서 이어서 학습한다")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.source is not None:
        cfg.setdefault("data", {})["source"] = args.source
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.threads:
        torch.set_num_threads(args.threads)

    torch.manual_seed(int(cfg.get("seed", 0)))
    exp_id = cfg.get("exp_id", Path(args.config).stem)

    # 데이터축을 **적재 전에** 확정한다. auto 로 두면 data/raw 의 상태에 따라
    # 조용히 D0/D1 이 바뀌므로, 무엇이 선택됐는지 경로와 로그에 함께 남긴다.
    requested = cfg.get("data", {}).get("source", "auto")
    kind = resolve_source_kind(requested)
    tag = source_tag(requested)
    if requested == "auto":
        print(f"[{exp_id}] source=auto -> {kind!r} 로 해석됨. "
              f"재현성을 위해 --source {kind} 를 명시할 것.")
    out = ensure_dir(args.out or f"results/{tag}/{exp_id}")
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False,
                                                    allow_unicode=True))

    src, tr, va = build_datasets(cfg)
    mcfg = cfg.get("model", {})
    model = build_model(mcfg.get("name", "resunet1d"), **(mcfg.get("kwargs") or {}))
    loss_fn = make_loss(cfg.get("loss", "L1"))
    tcfg = TrainCfg(**{k: v for k, v in (cfg.get("train") or {}).items()
                       if k in TrainCfg.__dataclass_fields__})

    print(f"[{exp_id}] source={src.kind} train_windows={len(tr)} val_windows={len(va)} "
          f"model={mcfg.get('name')} params={model.n_params():,} loss={cfg.get('loss')}")
    trainer = Trainer(model, loss_fn, tr, va, tcfg, out_dir=out,
                       device=args.device, num_workers=args.workers,
                       model_name=mcfg.get("name", "resunet1d"),
                       extra_manifest={"exp_id": exp_id, "source": src.kind,
                                       "source_requested": requested, "tag": tag,
                                       "loss": cfg.get("loss"),
                                       "pre_denoise": cfg.get("data", {}).get("pre_denoise"),
                                       "frontend": cfg.get("data", {}).get("frontend", True),
                                       "model_kwargs": mcfg.get("kwargs") or {}})
    if args.resume and not trainer.try_resume():
        print("[resume] last.pt 가 없다. 처음부터 학습한다.")
    tr_state = trainer.fit()
    # 체크포인트에 model_kwargs 를 남겨 두어야 나중에 복원 가능
    for nm in ("best.pt", "last.pt"):
        p = out / nm
        if p.exists():
            ck = torch.load(p, map_location="cpu", weights_only=False)
            ck["model_kwargs"] = mcfg.get("kwargs") or {}
            ck["frontend"] = bool(cfg.get("data", {}).get("frontend", True))
            ck["pre_denoise"] = cfg.get("data", {}).get("pre_denoise")
            torch.save(ck, p)
    print(f"best epoch {tr_state.best_epoch}: {tr_state.best_metric:+.3f} dB  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
