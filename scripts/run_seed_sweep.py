"""외부 컴퓨팅 자원(Colab · Kaggle · Lightning · 로컬 GPU)에서 **학습 런 간 산포**를 재는 판.

F-40 이 seed 3 판으로 sd 0.741 dB(df=2)를 얻었지만, 그 sd 의 95 % 구간이
0.39~4.66 dB 라 「0.74 근처」 이상은 말할 수 없었다. 그리고 그 산포가 5.9·5.10 의
구조 효과(0.1~0.4 dB)보다 커서 **네 번의 「기각」이 전부 「못 잼」으로 내려앉았다.**
이 스크립트는 그 자를 제대로 재고, 나아가 **구조·용량 비교를 seed 를 짝지어**
다시 판정하기 위한 것이다.

    # 1) 캘리브레이션 — 이 환경이 CPU 결과를 재현하는가 (약 10 분)
    python scripts/run_seed_sweep.py --stage calib

    # 2) 산포 — M06 을 seed 0..9 로
    python scripts/run_seed_sweep.py --arms m06_l1 --seeds 0-9

    # 3) 구조 재판정 — 다섯 구조를 **같은 seed 집합**에서 (짝지어 분석한다)
    python scripts/run_seed_sweep.py --stage structure --seeds 0-9

    # 4) 용량 재판정
    python scripts/run_seed_sweep.py --stage capacity --seeds 0-9

**설계상 중요한 두 가지**

1. **짝지은 설계.** 모든 arm 을 같은 seed 집합에서 돌린다. 학습 잡음의 공통
   성분이 차감되므로 독립 표본보다 훨씬 적은 판으로 같은 분해능이 나온다.
   (얼마나 이득인지는 `analyze_seed_sweep.py` 가 상관 ρ 로 알려준다.)

2. **고정 평가.** `best_metric` 은 **seed 에 따라 달라지는 val 잡음 뽑기** 위에서
   고른 값이다(`salt=("val", seed)`). 즉 산포 안에 「모델이 달라서」와
   「val 뽑기가 운이 좋아서」가 섞여 있다. 그래서 학습이 끝나면 best 체크포인트를
   **seed 와 무관한 고정 잡음 뽑기**로 한 번 더 재서 두 값을 같이 남긴다.
   둘의 산포를 비교하면 그 두 성분이 갈린다.

**산출물**: 실행마다 `<out>/<run>/summary.json` + `history.json` + `log.csv`,
그리고 최상위 `sweep.csv` 한 장. 기본적으로 **체크포인트는 남기지 않는다**
(`--keep-ckpt` 로 켠다) — 업로드해야 하는 것은 수 MB 로 끝나야 한다.

중단되면 그냥 같은 명령을 다시 실행하면 된다. `summary.json` 이 있는 판은 건너뛴다.
"""
import _bootstrap  # noqa: F401

import argparse
import copy
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

from ecgdn.config import TrainCfg
from ecgdn.data.dataset import ECGDenoiseDataset
from ecgdn.data.nstdb import make_banks
from ecgdn.data.sources import get_source, resolve_source_kind
from ecgdn.models import build_model, make_loss
from ecgdn.train import Trainer
from ecgdn.utils import ensure_dir

ROOT = Path(__file__).resolve().parent.parent

# 단계별 arm 묶음. **base arm 을 맨 앞에 둔다** — 분석기가 그것을 기준으로 짝을 짓는다.
STAGES = {
    # 이 환경이 CPU 결과를 재현하는지부터 본다. seed 0 하나, 재현되면 3.437 근처.
    "calib":     ["m06_l1"],
    # F-40 의 자를 df=2 에서 df=n−1 로 올린다.
    "spread":    ["m06_l1"],
    # 5.9 · 5.10 의 네 번의 기각을 다시 판정한다.
    "structure": ["m06_l1", "m07_l1", "m08_l1", "m09_l1", "m10_l1"],
    # D-23 B 의 용량 곡선을 다시 판정한다.
    "capacity":  ["m06_l1_quarter", "m06_l1_half", "m06_l1"],
    # 5.8.9 의 손실 효과가 정말 산포보다 큰지 확인한다.
    "loss":      ["m06_l1", "m06_l3", "m06_l6"],
    # D-27 등파라미터 폭<->깊이.
    "depth":     ["m06_l1", "m06_l1_deep2", "m06_l1_deep3", "m06_l1_deep4"],
    # D-28 2 번 — 참 SNR 조건화(FiLM). 등파라미터.
    "cond":      ["m06_l1", "m06_cond"],
}

# 고정 평가의 salt. seed 와 무관해야 의미가 있으므로 **상수**다. 바꾸면 예전 값과
# 비교할 수 없게 되므로 바꾸지 않는다.
FIXED_EVAL_SALT = ("fixed_eval", 20260909)

# **SNR 대역별로도 따로 잰다.** 총합 하나는 동작점에 따라 부호까지 바뀌는 효과를
# 가린다 — D1 EXP-A 에서 `M06` 은 −5 dB 에서 오라클에 0.90 dB 차로 붙고 20 dB
# 에서는 7.95 dB 뒤진다. 좁은 대역으로 학습한 모델과 넓은 대역으로 학습한 모델을
# 견주려면 **같은 대역에서** 재야 하므로, 이 다섯 칸이 그 공통 자가 된다.
EVAL_BANDS = [(-5.0, 0.0), (0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0)]


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def env_fingerprint(device: torch.device, amp: bool) -> dict:
    """숫자가 어느 환경에서 나왔는지 남긴다.

    F-9 의 교훈이다 — 체크포인트만 봐서는 어느 파이프라인의 산물인지 알 수 없다.
    환경이 다르면 값도 다를 수 있고, **다르다는 사실 자체가 결과**이므로 남긴다.
    """
    fp = {
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": device.type,
        "amp": bool(amp),
        "threads": torch.get_num_threads(),
    }
    if device.type == "cuda":
        fp["gpu"] = torch.cuda.get_device_name(0)
        fp["cuda"] = torch.version.cuda
        fp["cudnn"] = torch.backends.cudnn.version()
    return fp


def build_datasets(cfg: dict, source: str):
    """train.py 의 것과 **같은 방식**이어야 한다. 여기서 갈리면 비교가 무의미해진다."""
    d = dict(cfg.get("data", {}))
    d["source"] = source
    src = get_source(source,
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
              ref_frontend=bool(d.get("ref_frontend", d.get("frontend", True))))
    seed = int(cfg.get("seed", 0))
    tr = ECGDenoiseDataset(src, "train", banks=make_banks("train", nstdb_root),
                           salt=("train", seed), **kw)
    va = ECGDenoiseDataset(src, "val", banks=make_banks("val", nstdb_root),
                           salt=("val", seed), **kw)
    # 고정 평가용 — 같은 val 기록, **seed 와 무관한 잡음 뽑기**.
    fx = ECGDenoiseDataset(src, "val", banks=make_banks("val", nstdb_root),
                           salt=FIXED_EVAL_SALT, **kw)
    # 대역별 고정 평가. `snr_range` 만 좁히고 나머지는 같다.
    bands = {}
    for lo, hi in EVAL_BANDS:
        bkw = dict(kw, snr_range=(lo, hi))
        bands[f"{lo:g}_{hi:g}"] = ECGDenoiseDataset(
            src, "val", banks=make_banks("val", nstdb_root),
            salt=(FIXED_EVAL_SALT, lo, hi), **bkw)
    return src, tr, va, fx, bands


def partial_epoch(out: Path) -> int | None:
    """이 판이 **몇 epoch 까지 갔다가 끊겼나**. 안 끊겼으면 None.

    `last.pt` 는 epoch 마다 저장되므로 그 안의 `epoch` 이 곧 진행분이다.
    `summary.json` 이 있으면 끝난 판이라 진행분이 아니다.
    """
    if (out / "summary.json").exists() or not (out / "last.pt").exists():
        return None
    try:
        return int(torch.load(out / "last.pt", map_location="cpu",
                              weights_only=False).get("epoch", 0)) or None
    except Exception:
        return None


def run_one(arm: str, seed: int, out_root: Path, source: str, device: str | None,
            amp: bool | None, epochs: int | None, keep_ckpt: bool,
            workers: int, resume: bool = True) -> dict:
    run_id = f"{arm}__s{seed}"
    out = ensure_dir(out_root / run_id)
    summary_p = out / "summary.json"
    if summary_p.exists():
        return json.loads(summary_p.read_text())

    cfg = yaml.safe_load((ROOT / "configs" / f"{arm}.yaml").read_text())
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = seed
    cfg["exp_id"] = run_id
    if epochs is not None:
        cfg.setdefault("train", {})["epochs"] = epochs
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False,
                                                    allow_unicode=True))

    torch.manual_seed(seed)                      # train.py 와 동일
    src, tr, va, fx, bands = build_datasets(cfg, source)
    mcfg = cfg.get("model", {})
    model = build_model(mcfg.get("name", "resunet1d"), **(mcfg.get("kwargs") or {}))
    loss_fn = make_loss(cfg.get("loss", "L1"))
    tcfg = TrainCfg(**{k: v for k, v in (cfg.get("train") or {}).items()
                       if k in TrainCfg.__dataclass_fields__})

    trainer = Trainer(model, loss_fn, tr, va, tcfg, out_dir=out, device=device,
                      num_workers=workers, amp=amp,
                      model_name=mcfg.get("name", "resunet1d"),
                      extra_manifest={"exp_id": run_id, "arm": arm, "seed": seed,
                                      "source": src.kind, "loss": cfg.get("loss"),
                                      "sweep": True})
    n_params = model.n_params()
    print(f"  [{run_id}] params={n_params:,} loss={cfg.get('loss')} "
          f"device={trainer.device.type} amp={trainer.amp}", flush=True)

    # **판 하나가 중간에 끊겨도 이어서 간다.** `last.pt` 에 optimizer·스케줄
    # 위치·`best_epoch` 까지 들어 있어 「이어서」가 「다른 학습」이 되지 않는다.
    # Colab 처럼 세션이 자주 끊기는 환경에서 이것이 없으면 한 판을 통째로 다시
    # 돌린다 — 실제로 그렇게 잃었다.
    if resume and not trainer.try_resume():
        pass                                   # 없으면 그냥 처음부터

    t0 = time.perf_counter()
    st = trainer.fit()
    wall = time.perf_counter() - t0

    # 고정 평가 — best 체크포인트를 seed 와 무관한 잡음 뽑기로 다시 잰다.
    fixed: dict[str, float] = {}
    bp = out / "best.pt"
    if bp.exists():
        ck = torch.load(bp, map_location=trainer.device, weights_only=False)
        trainer.model.load_state_dict(ck["model"])
        fixed = {f"fixed_{k}": float(v) for k, v in trainer.evaluate(fx).items()}
        # 대역별 — `snr_imp_scaled` 하나만 남긴다 (열이 너무 늘지 않게).
        for name, bds in bands.items():
            fixed[f"band_{name}"] = float(trainer.evaluate(bds)["snr_imp_scaled"])

    rec = {
        "run_id": run_id, "arm": arm, "seed": seed,
        "n_params": int(n_params), "loss": cfg.get("loss"),
        "best_metric": float(st.best_metric), "best_epoch": int(st.best_epoch),
        "n_epochs": int(st.epoch), "wall_s": round(wall, 1),
        "sec_per_epoch": round(wall / max(st.epoch, 1), 2),
        "source": src.kind,
        **fixed,
        "env": env_fingerprint(trainer.device, trainer.amp),
    }
    summary_p.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    if not keep_ckpt:
        for nm in ("best.pt", "last.pt"):
            (out / nm).unlink(missing_ok=True)
    print(f"  [{run_id}] best {st.best_metric:+.3f} dB (ep {st.best_epoch}) · "
          f"fixed {rec.get('fixed_snr_imp_scaled', float('nan')):+.3f} dB · "
          f"{wall/60:.1f} min", flush=True)
    return rec


def rel(p: Path) -> str:
    """저장소 밖 경로도 받으므로 relative_to 를 그냥 쓰면 터진다."""
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def write_csv(out_root: Path) -> Path:
    rows = []
    for p in sorted(out_root.glob("*/summary.json")):
        r = json.loads(p.read_text())
        env = r.pop("env", {})
        r.update({f"env_{k}": v for k, v in env.items()})
        rows.append(r)
    if not rows:
        return out_root / "sweep.csv"
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    p = out_root / "sweep.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=sorted(STAGES), default=None,
                    help="미리 정의된 arm 묶음. --arms 와 함께 쓰면 --arms 가 이긴다")
    ap.add_argument("--arms", default=None, help="쉼표로 구분한 config stem")
    ap.add_argument("--seeds", default=None, help="예: 0-9  또는  0,1,2,5")
    ap.add_argument("--out", default=None, help="기본 results/ext/<stage 또는 custom>")
    ap.add_argument("--source", default="mitdb", choices=("mitdb", "synthetic"))
    ap.add_argument("--device", default=None, help="cuda | cpu (기본: 있으면 cuda)")
    ap.add_argument("--amp", choices=("on", "off"), default="off",
                    help="기본 off — CPU(fp32) 결과와 같은 계보로 두기 위해서다")
    ap.add_argument("--epochs", type=int, default=None, help="빠른 확인용 축소")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=None,
                    help="torch 스레드 수. 이 저장소의 학습은 4 로 돌았다 — "
                         "맞춰 두면 부동소수점 축약 순서까지 같아진다")
    ap.add_argument("--no-resume", action="store_true",
                    help="끊긴 판을 처음부터 다시 돌린다 (기본은 last.pt 에서 재개)")
    ap.add_argument("--keep-ckpt", action="store_true",
                    help="best.pt/last.pt 를 남긴다 (업로드가 커진다)")
    ap.add_argument("--dry-run", action="store_true", help="무엇을 돌릴지만 출력")
    args = ap.parse_args()

    if args.arms:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    elif args.stage:
        arms = list(STAGES[args.stage])
    else:
        ap.error("--stage 또는 --arms 중 하나는 필요하다")

    seeds = parse_seeds(args.seeds) if args.seeds else ([0] if args.stage == "calib"
                                                       else [0, 1, 2])
    out_root = ensure_dir(Path(args.out) if args.out
                          else ROOT / "results" / "ext" / (args.stage or "custom"))

    missing = [a for a in arms if not (ROOT / "configs" / f"{a}.yaml").exists()]
    if missing:
        print(f"[오류] 없는 config: {', '.join(missing)}", file=sys.stderr)
        return 2
    if resolve_source_kind(args.source) != args.source:
        print(f"[오류] source={args.source} 를 쓸 수 없다 "
              f"(data/raw 확인)", file=sys.stderr)
        return 2

    # **seed 우선**으로 돈다 (arm 우선이 아니라). 짝지은 분석은 «한 seed 의 모든
    # arm» 이 있어야 성립하는데, arm 우선이면 중간에 끊겼을 때 arm 몇 개만 전 seed
    # 를 갖고 나머지는 하나도 없다 — 짝이 하나도 안 맞는다. seed 우선이면 끊긴
    # 지점까지의 seed 가 통째로 완성돼 그대로 분석에 들어간다.
    todo = [(a, s) for s in seeds for a in arms]
    done = sum(1 for a, s in todo if (out_root / f"{a}__s{s}" / "summary.json").exists())
    print(f"[sweep] arms={arms} seeds={seeds} → {len(todo)} 판 "
          f"(완료 {done}, 남은 {len(todo)-done})")
    print(f"[sweep] out={rel(out_root)} source={args.source} "
          f"amp={args.amp}")
    if args.dry_run:
        for a, s in todo:
            d = out_root / f"{a}__s{s}"
            if (d / "summary.json").exists():
                print(f"  ✓ {a}__s{s}")
            elif (ep := partial_epoch(d)) is not None:
                print(f"  ↻ {a}__s{s}   epoch {ep} 까지 갔다가 끊겼다 — 거기서 재개한다")
            else:
                print(f"    {a}__s{s}")
        return 0

    amp = {"on": True, "off": False}[args.amp]
    if args.threads:
        torch.set_num_threads(args.threads)
    t_start = time.perf_counter()
    n_ran = 0
    for i, (arm, seed) in enumerate(todo, 1):
        d = out_root / f"{arm}__s{seed}"
        pre = (d / "summary.json").exists()
        part = partial_epoch(d)
        tag = ("  (이미 있음, 건너뜀)" if pre else
               f"  (epoch {part} 에서 재개)" if part and not args.no_resume else "")
        print(f"[sweep] {i}/{len(todo)}  {arm} seed={seed}{tag}", flush=True)
        try:
            run_one(arm, seed, out_root, args.source, args.device, amp,
                    args.epochs, args.keep_ckpt, args.workers,
                    resume=not args.no_resume)
        except Exception as e:                      # 한 판이 죽어도 큐는 계속
            print(f"  [실패] {arm} seed={seed}: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            continue
        if not pre:
            n_ran += 1
            el = time.perf_counter() - t_start
            left = len(todo) - i
            print(f"  ...누적 {el/60:.1f} 분, 실행한 판 {n_ran}, "
                  f"남은 {left} 판 ≈ {left*el/max(n_ran,1)/60:.0f} 분", flush=True)
        write_csv(out_root)

    p = write_csv(out_root)
    print(f"\n[sweep] 끝. 표 → {rel(p)}")
    print(f"[sweep] 업로드할 것: {rel(out_root)} 폴더 전체 "
          f"({sum(f.stat().st_size for f in out_root.rglob('*') if f.is_file())/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
