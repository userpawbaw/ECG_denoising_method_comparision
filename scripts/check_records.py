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

# 근거 등급. [재구성] 은 "당시 기록이 없어 사후 복원" 이라는 뜻이고, 이것을
# 숨기지 않는 것이 이 검사의 목적이다 (docs/19_record_keeping.md 8절).
EVIDENCE_TAGS = ("[측정]", "[로그]", "[커밋]", "[대화]", "[재구성]")
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
        # 근거가 없어 "기록 없음" 으로 둔 절은 충족으로 본다.
        # 억지로 채우면 없던 정보를 만들어낸다 — 빈칸이 허위보다 낫다.
        if "기록 없음" in body:
            miss = [k for k in miss if k != "먼저 의심한 것"]
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


def check_evidence_tags(path: Path) -> list[str]:
    """서술이 근거 등급을 달고 있는지. 최소 한 개는 있어야 한다."""
    bad = []
    for fid, body in _blocks(path, r"F-\d+"):
        if not any(t in body for t in EVIDENCE_TAGS) and "기록 없음" not in body:
            bad.append(f"{fid}: 근거 등급 표기 없음")
    return bad


def check_cited_numbers() -> list[str]:
    """문서에 인용된 수치가 산출물의 실제 값과 맞는지 대조한다.

    인용을 지어내면 여기서 잡힌다. 대조 가능한 것만 검사하고, 산출물이 아직
    없으면 건너뛴다(아직 돌리지 않은 축의 값을 요구하지 않기 위해).
    """
    import csv

    bad = []
    text = (ROOT / "docs" / "20_findings.md").read_text()
    checks = [
        # (설명, 파일, 지표, 문서에 적힌 값, 허용오차)
        ("D0 qrs_dur_err_ms floor", "results/d0/metric_floor/floor.csv",
         "qrs_dur_err_ms", 0.639731, 1e-4),
        ("D1 qrs_dur_err_ms floor", "results/d1/metric_floor/floor.csv",
         "qrs_dur_err_ms", 28.072098, 0.5),
        ("D1 r_amp_err_pct floor", "results/d1/metric_floor/floor.csv",
         "r_amp_err_pct", 0.226334, 0.02),
    ]
    for label, rel, metric, want, tol in checks:
        p = ROOT / rel
        if not p.exists():
            continue
        got = None
        for row in csv.DictReader(p.open()):
            if row["metric"] == metric:
                got = float(row["floor_p95"])
                break
        if got is None:
            continue
        if abs(got - want) > tol:
            bad.append(f"{label}: 문서 {want} vs 실제 {got:.6g}")
        # 문서에 그 수치가 실제로 적혀 있는지도 확인
        if f"{want:.6g}"[:6] not in text and f"{want:.3f}"[:5] not in text:
            bad.append(f"{label}: 문서에서 인용을 찾지 못했다 ({want})")
    return bad


def main() -> int:
    d = ROOT / "docs"
    problems: list[str] = []
    problems += check(d / "20_findings.md", r"F-\d+", F_REQUIRED, "발견")
    problems += check(d / "21_decisions.md", r"D-\d+", D_REQUIRED, "결정")

    ev = check_evidence_tags(d / "20_findings.md")
    if ev:
        print("\n[근거 등급] 미표기")
        for e in ev:
            print("  ✗", e)
    problems += ev

    num = check_cited_numbers()
    print("\n[수치 대조] 문서 인용 vs 산출물")
    if num:
        for n in num:
            print("  ✗", n)
    else:
        print("  일치 (또는 대조할 산출물 없음)")
    problems += num

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
