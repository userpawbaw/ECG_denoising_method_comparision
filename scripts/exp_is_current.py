#!/usr/bin/env python3
"""이 실험 산출물이 **지금 설정 그대로** 만들어진 것인가.

    python scripts/exp_is_current.py abl_loss d1   # 종료코드 0=최신, 1=재실행

**시각 비교로는 안 된다.** 처음에는 "산출물이 체크포인트·config 보다 새로우면
건너뛴다" 로 짰는데, 실험이 **config 변경 전에 시작해 변경 후에 끝나면**
산출물 시각이 config 보다 새로우면서도 낡은 설정으로 만들어진다. 실제로
`exp_b` 가 그렇게 M09 없이 완주했고, 그대로 뒀으면 **M09 만 빠진 표가
조용히 살아남았다.**

그래서 시각이 아니라 **내용**을 본다:

1. manifest 의 `methods` 가 config 가 요구하는 방법을 전부 담고 있는가
2. 산출물이 그 축의 최신 체크포인트보다 새로운가 (F-9 — 체크포인트가 바뀌면
   산출물도 다시 만들어야 한다)

둘 다 참이어야 건너뛴다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def is_current(exp: str, tag: str) -> tuple[bool, str]:
    out = ROOT / "results" / tag / exp
    man = out / "manifest.json"
    art = next((p for p in (out / "metrics.parquet", out / "probe.csv")
                if p.exists()), None)
    if art is None or not man.exists():
        return False, "산출물 없음"

    cfg = yaml.safe_load((ROOT / "configs" / f"{exp}.yaml").read_text()) or {}
    want = set(cfg.get("methods") or []) | set((cfg.get("dl_methods") or {}))
    try:
        have = set(json.loads(man.read_text()).get("methods") or [])
    except (json.JSONDecodeError, OSError):
        return False, "manifest 읽기 실패"
    missing = want - have
    if missing:
        return False, f"방법 누락: {sorted(missing)}"

    ckpts = list((ROOT / "results" / tag).glob("*/best.pt"))
    if ckpts:
        newest = max(p.stat().st_mtime for p in ckpts)
        if art.stat().st_mtime <= newest:
            return False, "체크포인트가 산출물보다 새롭다 (F-9)"
    return True, "최신"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__); return 2
    ok, why = is_current(sys.argv[1], sys.argv[2])
    print(why)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
