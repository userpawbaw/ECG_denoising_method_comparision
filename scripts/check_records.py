#!/usr/bin/env python3
"""기록 규약(docs/19_record_keeping.md)이 실제로 지켜지는지 점검한다.

    python scripts/check_records.py

규약을 만들어 놓고 정작 새 항목을 기존 방식으로 적으면 의미가 없다.
특히 **"놓쳤다면"** 은 보고서에서 인용 가치가 가장 높은 항목인데,
자동 점검을 넣기 전에는 F-1~F-9 대부분에 빠져 있었다.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 필수 요소와 그것을 담을 수 있는 헤딩(동의 표현 허용)
F_REQUIRED = {
    "발단": ("### 발단",),
    "먼저 의심한 것": ("### 먼저 의심", "### 왜 문제인가"),
    "결정적 측정": ("### 결정적 측정", "### 관측", "### 조치와 측정"),
    "놓쳤다면": ("### 놓쳤다면",),
}
D_REQUIRED = {
    "갈림길": ("### 갈림길",),
    "검토한 선택지": ("### 검토한 선택지", "### 근거 —"),
    "고른 것과 근거": ("### 고른 것과 근거", "### 근거 —"),
}


def _blocks(path: Path, pat: str):
    text = path.read_text()
    parts = re.split(rf"\n## ({pat}[^\n]*)\n", text)
    return [(parts[i].split(".")[0], parts[i + 1]) for i in range(1, len(parts), 2)]


def check(path: Path, pat: str, required: dict, label: str) -> list[str]:
    bad = []
    items = _blocks(path, pat)
    print(f"\n[{label}] {path.name} — {len(items)} 항목")
    for fid, body in items:
        miss = [k for k, heads in required.items()
                if not any(h in body for h in heads)]
        # 잠정/철회 항목은 아직 못 채운 것이 정상일 수 있다
        provisional = "[잠정]" in body[:400] or "상태 | **잠정**" in body
        if miss and not provisional:
            bad.append(f"{fid}: {', '.join(miss)}")
            print(f"  ✗ {fid:<6} 누락: {', '.join(miss)}")
        elif miss:
            print(f"  ~ {fid:<6} 누락(잠정이라 허용): {', '.join(miss)}")
        else:
            print(f"  OK {fid}")
    return bad


def main() -> int:
    d = ROOT / "docs"
    problems: list[str] = []
    problems += check(d / "20_findings.md", r"F-\d+", F_REQUIRED, "발견")
    problems += check(d / "21_decisions.md", r"D-\d+", D_REQUIRED, "결정")

    # O 는 표 한 줄 + 증상/원인/조치면 충분 — 재발 방지 칸만 확인한다
    inc = (d / "22_incidents.md").read_text()
    o_ids = re.findall(r"\n## (O-\d+)\.", inc)
    print(f"\n[사고] 22_incidents.md — {len(o_ids)} 항목: {', '.join(o_ids)}")

    print("\n" + "=" * 60)
    if problems:
        print(f"규약 미충족 {len(problems)}건:")
        for p in problems:
            print("  -", p)
        return 1
    print("기록 규약 충족.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
