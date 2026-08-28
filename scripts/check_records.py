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
EVIDENCE_TAGS = ("[측정]", "[로그]", "[커밋]", "[대화]",
                 "[코드]", "[문헌]", "[추론]", "[재구성]")
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



# ---------------------------------------------------------------- 누락 검사
#
# 위의 검사들은 전부 **존재하는 기록이 잘 쓰였는가** 를 본다. 그것만으로는
# 기록이 **아예 없는** 경우를 잡을 수 없고, 실제로 그렇게 빠졌다
# (L5·L6 설계 근거가 D/F/보고서 어디에도 0 건이었다 — 19_record_keeping.md 5절).
#
# 아래 검사들은 **코드·산출물에서 의무를 유도한다.** 대장(ledger)을 따로 두면
# 대장에 적는 것을 잊는 같은 실패가 생기므로, 잊을 수 없는 것(레지스트리,
# preset 목록)에서 뽑는다.

# 보고서 6장 요약을 면제하는 F 와 그 이유. **빈 면제는 두지 않는다** —
# 이유를 적게 해서 "귀찮아서 면제" 를 막는다.
F_SUMMARY_EXEMPT = {
    "F-7": "4장 '(2) SNR sweep에서 잡음 실현을 고정' 에 설계 규칙으로 실려 있다"
           " — 틀렸다가 고친 이야기가 아니라 방법 규약이라 6장이 아니다",
}


def check_design_choices_have_a_decision() -> list[str]:
    """모델 레지스트리와 손실 preset 은 전부 D 나 설계 문서에 이름이 있어야 한다.

    새 모델·새 손실을 코드에 넣는 것은 **설계 결정**이다. 그런데 그 결정이
    실험 결과와 함께 기록되는 습관이 있어서, 결과가 늦으면 기록도 늦고
    **기각한 후보는 결과가 없으므로 영영 안 실린다.**
    """
    bad = []
    dec = (ROOT / "docs" / "21_decisions.md").read_text()
    des = (ROOT / "docs" / "01_design.md").read_text()
    hay = dec + des

    try:
        sys.path.insert(0, str(ROOT))
        from ecgdn.models import MODELS
        from ecgdn.models.losses import LOSS_NAMES, _PRESETS
    except Exception as e:                      # torch 없는 환경
        print(f"  (건너뜀: {e})")
        return []

    for key, ctor in MODELS.items():
        name = getattr(ctor, "__name__", "")
        if key not in hay and name not in hay:
            bad.append(f"모델 '{key}' ({name}): D 기록도 01_design 도 언급하지 않는다")
    for name in LOSS_NAMES:
        # 'L1' 같은 짧은 이름은 오탐이 나기 쉬우므로 백틱 표기까지 본다
        if name not in hay and f"`{name}`" not in hay:
            bad.append(f"손실 '{name}': D 기록도 01_design 도 언급하지 않는다")
    _ = _PRESETS
    return bad


def check_findings_reach_the_report() -> list[str]:
    """모든 F 는 보고서 6장에 요약이 있거나, 이유를 적은 면제 목록에 있어야 한다."""
    bad = []
    fnd = (ROOT / "docs" / "20_findings.md").read_text()
    rep = (ROOT / "docs" / "91_report.md").read_text()
    for fid in re.findall(r"\n## (F-\d+)\.", fnd):
        if re.search(rf"\n## {fid}\.", rep):
            continue
        if fid in F_SUMMARY_EXEMPT:
            continue
        bad.append(f"{fid}: 보고서 6장에 요약이 없고 면제 사유도 없다")
    for fid in F_SUMMARY_EXEMPT:
        if not re.search(rf"\n## {fid}\.", fnd):
            bad.append(f"{fid}: 면제 목록에 있는데 F 기록이 없다 (오래된 면제)")
    return bad


def check_handoff_sections_name_a_destination() -> list[str]:
    """인수인계 절의 각 소절은 **영구 기록의 목적지**를 적어야 한다.

    `99_status.md` 의 인수인계 절은 "여기 적힌 작업이 끝나면 지운다" 는
    임시 영역이다. 그런데 지울 때 **그 안에만 있던 내용이 같이 사라진다.**
    실제로 L5/L6 의 기각 후보 표가 이 절에만 있었다.

    그래서 각 소절이 `-> D-13` / `-> F-21` / `-> 5.8.8` 처럼 목적지를 적게
    하고, 적힌 목적지가 실제로 존재하는지 확인한다. 아직 못 정했으면
    `[미정]` 을 적는다 — **빈칸과 '미정' 은 다르다.**
    """
    bad = []
    p = ROOT / "docs" / "99_status.md"
    if not p.exists():
        return []
    text = p.read_text()
    m = re.search(r"\n## \d+\. 세션 인수인계[^\n]*\n(.*)\Z", text, re.S)
    if not m:
        return []                                # 인수인계 절이 없으면 검사 대상 없음
    body = m.group(1)
    dec = (ROOT / "docs" / "21_decisions.md").read_text()
    fnd = (ROOT / "docs" / "20_findings.md").read_text()
    rep = (ROOT / "docs" / "91_report.md").read_text()
    for head, sub in re.findall(r"\n### ([^\n]+)\n(.*?)(?=\n### |\Z)", body, re.S):
        if "[미정]" in sub:
            continue
        dests = re.findall(r"\*\*(D-\d+|F-\d+|O-\d+)\*\*|\*\*(\d+\.\d+(?:\.\d+)?)\*\*", sub)
        flat = [a or b for a, b in dests]
        if not flat:
            bad.append(f"인수인계 '{head[:40]}': 목적지 표기가 없다 "
                       f"(-> **D-n** / **F-n** / **5.8.8** 또는 [미정])")
            continue
        for d in flat:
            if d.startswith("D-") and f"\n## {d}." not in dec:
                bad.append(f"인수인계 '{head[:30]}' 가 가리키는 {d} 이 없다")
            elif d.startswith("F-") and f"\n## {d}." not in fnd:
                bad.append(f"인수인계 '{head[:30]}' 가 가리키는 {d} 이 없다")
            elif d[0].isdigit() and f"{d} " not in rep:
                bad.append(f"인수인계 '{head[:30]}' 가 가리키는 보고서 {d} 절이 없다")
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

    print("\n[누락] 설계 결정이 D 기록에 있는가")
    dc = check_design_choices_have_a_decision()
    for b in dc:
        print("  ✗", b)
    if not dc:
        print("  모든 모델·손실이 D 또는 01_design 에 있다")
    problems += dc

    print("\n[누락] F 가 보고서에 닿는가")
    fr = check_findings_reach_the_report()
    for b in fr:
        print("  ✗", b)
    if not fr:
        print(f"  전부 도달 (면제 {len(F_SUMMARY_EXEMPT)}건, 사유 기재됨)")
    problems += fr

    print("\n[누락] 인수인계 절이 영구 목적지를 적었는가")
    hs = check_handoff_sections_name_a_destination()
    for b in hs:
        print("  ✗", b)
    if not hs:
        print("  전부 목적지 명시 (또는 인수인계 절 없음)")
    problems += hs

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

    # R 은 '재사용 규칙' 이 있어야 R 로 남길 가치가 있다
    rp = d / "23_ai_review.md"
    if rp.exists():
        r_items = _blocks(rp, r"R-\d+")
        r_bad = [f"{rid}: 재사용 규칙 없음" for rid, body in r_items
                 if "### 재사용 규칙" not in body]
        print("\n[AI 검토] 23_ai_review.md — %d 항목" % len(r_items))
        for b in r_bad:
            print("  ✗", b)
        problems += r_bad

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
