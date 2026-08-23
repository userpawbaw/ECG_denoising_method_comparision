#!/usr/bin/env python3
"""실험 실행 전 체크포인트 게이트.

    python scripts/check_ckpts.py --source mitdb

`configs/*.yaml` 의 `dl_methods` 가 요구하는 체크포인트가 해당 데이터축에
**전부** 있는지 확인하고, 하나라도 없으면 종료코드 2 로 중단한다.

왜 필요한가: `run_exp.py` 는 체크포인트가 없으면 `[skip]` 한 줄만 찍고
그 방법을 빼고 계속 간다. 실행 로그를 `tail` 로만 보면 그 줄이 스크롤에
묻히고, **결과 표는 정상적으로 생성된다** — 다만 행이 하나 없을 뿐이다.
F-9 에서 겪은 것과 같은 함정이다(표만 봐서는 무엇이 빠졌는지 드러나지 않는다).
학습이 중단된 채로 실험을 돌리는 사고가 실제로 가능하므로, 실험 러너가
시작하기 전에 여기서 먼저 막는다.

또한 `manifest.json` 의 `frontend` 플래그가 런들 사이에서 갈리면 경고한다.
비교가 성립하지 않는 조건이기 때문이다(F-9). 이 경우는 중단까지는 하지
않는다 — 최종 판정은 `make_ablation_table.py` 가 한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 실험 러너가 실제로 도는 순서. 여기 없는 config 는 게이트 대상이 아니다.
DEFAULT_CONFIGS = ("exp_c", "exp_a", "exp_b", "abl_loss", "exp_e")


def training_in_progress() -> str | None:
    """학습 러너가 지금 돌고 있으면 사유 문자열, 아니면 None.

    체크포인트가 '있다'는 것과 '학습이 끝났다'는 것은 다르다. `best.pt` 는
    epoch 마다 갱신되므로, 학습 도중에도 파일은 존재한다. 그 상태로 실험을
    돌리면 **덜 학습된 모델이 표에 들어간다** — 파일이 있으니 게이트도
    통과하고, 결과도 그럴듯하게 나온다.
    """
    lock = Path("results/.train.lock")
    if not lock.exists():
        return None
    for pd in Path("/proc").iterdir():
        if not pd.name.isdigit():
            continue
        try:
            cmd = (pd / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace")
        except OSError:
            continue
        if "run_all_training.sh" in cmd or "scripts/train.py" in cmd:
            return f"학습 러너가 실행 중이다 (pid {pd.name})"
    return ("results/.train.lock 이 남아 있는데 학습 프로세스가 없다. "
            "컨테이너가 강제 종료되어 락만 남은 상태로 보인다 "
            "(rmdir results/.train.lock 후 재개할 것)")


def required_ckpts(cfg_paths: list[Path], tag: str) -> dict[Path, list[str]]:
    """{체크포인트 경로: [요구하는 config:method, ...]}"""
    need: dict[Path, list[str]] = {}
    for p in cfg_paths:
        cfg = yaml.safe_load(p.read_text()) or {}
        for mid, spec in (cfg.get("dl_methods") or {}).items():
            if isinstance(spec, str):
                spec = {"ckpt": spec}
            ck = Path(str(spec["ckpt"]).replace("{tag}", tag))
            need.setdefault(ck, []).append(f"{p.stem}:{mid}")
    return need


def frontend_flags(ckpts: list[Path]) -> dict[Path, object]:
    out: dict[Path, object] = {}
    for ck in ckpts:
        mf = ck.parent / "manifest.json"
        if not mf.exists():
            out[ck] = "?"
            continue
        m = json.loads(mf.read_text())
        out[ck] = m.get("extra", m).get("frontend", "?")
    return out


def main() -> int:
    import argparse
    from ecgdn.data.sources import source_tag

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="auto",
                    choices=("auto", "synthetic", "mitdb"))
    ap.add_argument("-c", "--configs", nargs="*", default=list(DEFAULT_CONFIGS),
                    help="config 이름 또는 경로 (기본: 실험 러너가 도는 전부)")
    ap.add_argument("--ignore-lock", action="store_true",
                    help="학습이 돌고 있어도 통과시킨다 (의도적 부분 실행용)")
    a = ap.parse_args()

    tag = source_tag(a.source)

    busy = training_in_progress()
    if busy and not a.ignore_lock:
        print(f"[ckpt-gate] {busy}")
        print("  학습이 끝나기 전에는 실험을 돌리지 않는다. best.pt 는 epoch 마다")
        print("  갱신되므로 지금 돌리면 덜 학습된 모델이 표에 들어간다.")
        print("  그래도 강행하려면 --ignore-lock 을 줄 것.")
        return 2

    paths = []
    for c in a.configs:
        p = Path(c) if str(c).endswith(".yaml") else Path("configs") / f"{c}.yaml"
        if not p.exists():
            print(f"[error] config 없음: {p}")
            return 2
        paths.append(p)

    need = required_ckpts(paths, tag)
    missing = {ck: who for ck, who in need.items() if not ck.exists()}
    present = [ck for ck in need if ck.exists()]

    print(f"[ckpt-gate] source={a.source} tag={tag} "
          f"필요 {len(need)}개 / 존재 {len(present)}개")

    if missing:
        print(f"[ckpt-gate] 체크포인트 {len(missing)}개가 없다. 실험을 중단한다.")
        for ck, who in sorted(missing.items()):
            print(f"  MISSING  {ck}   <- {', '.join(who)}")
        runs = sorted({ck.parent.name for ck in missing})
        print("\n  이대로 실험을 돌리면 run_exp.py 가 해당 방법만 빼고 표를"
              "\n  정상적으로 만들어 낸다. 먼저 학습을 끝낼 것:")
        print(f"    bash scripts/run_all_training.sh {tag_to_source(tag)} "
              f"{' '.join(runs)}")
        return 2

    flags = frontend_flags(present)
    distinct = set(flags.values())
    if len(distinct) > 1:
        print("[warn] 체크포인트들의 frontend 플래그가 갈린다 — "
              "같은 조건의 비교가 아니다 (F-9).")
        for ck, f in sorted(flags.items()):
            print(f"  frontend={f}  {ck}")

    print("[ckpt-gate] 통과.")
    return 0


def tag_to_source(tag: str) -> str:
    return {"d0": "synthetic", "d1": "mitdb"}.get(tag, "auto")


if __name__ == "__main__":
    raise SystemExit(main())
