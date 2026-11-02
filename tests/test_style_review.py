"""학술 서술 검토 규칙판(`scripts/style_rules.py`) · 추출기(`scripts/style_review.py`) — D-43 보완.

규칙판은 개방 코딩 → T1 → 대조 → F 순서로 만들었다(`docs/95_academic_style.md` §6). 여기서는
① 규칙의 형식이 갖춰졌는지, ② 규칙이 자기 예시와 맞는지(수정 전은 걸리고 수정 후는 안 걸린다),
③ 개방 코딩의 모든 코드가 범주로 옮겨지는지, ④ 저장된 산출물이 지금 규칙으로 다시 뽑은 것과 같은지를 본다.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import style_review as sr  # noqa: E402
from style_rules import F_CONTEXT, GROUPS, OPEN_CODEBOOK, open_category, ruleset  # noqa: E402

OUT = ROOT / "results" / "style_review"


@pytest.mark.parametrize("version", ["T1", "F"])
def test_rule_form_is_complete(version):
    rules = ruleset(version)
    assert len({r.rid for r in rules}) == len(rules)
    for r in rules:
        assert r.group in GROUPS
        assert r.define and r.include and r.guide, r.rid
        assert r.mode in ("자동", "자동+수동", "수동")
        assert r.tier in (1, 2)
        assert r.patterns or r.func, f"{r.rid} 에 탐지 신호가 없다"


def test_final_rule_has_same_shape_and_records_changes():
    t1 = {r.rid: r for r in ruleset("T1")}
    f = {r.rid: r for r in ruleset("F")}
    assert t1.keys() == f.keys(), "최종 규칙은 T1 과 같은 범주 체계여야 한다"
    for rid, r in f.items():
        changed = (r.patterns != t1[rid].patterns or r.func is not t1[rid].func or r.mode != t1[rid].mode
                   or r.raw != t1[rid].raw)
        if changed:
            assert r.note, f"{rid} 는 T1 에서 바뀌었는데 바뀐 이유(note)가 없다"


def test_final_rules_agree_with_their_own_examples():
    """교정 지침의 예 — 수정 전은 그 범주에 걸리고, 수정 후는 걸리지 않아야 한다."""
    for r in ruleset("F"):
        for before, after in r.examples:
            ctx = dict(F_CONTEXT, kind="제목" if r.rid == "B8" else "", abbrev_scope="sentence",
                       coined_scope="sentence")
            assert r.hits(before, dict(ctx, plain=re.sub(r"`[^`]*`", " ", before))), (r.rid, before)
            assert not r.hits(after, dict(ctx, plain=re.sub(r"`[^`]*`", " ", after))), (r.rid, after)


@pytest.mark.parametrize("rid,text", [
    ("A2", "seed 만 바꿔 세 판 돌렸다"),
    ("A3", "천장은 모델이 아니라 무엇을 맞추라고 시키느냐에 있었다"),
    ("C7", "그 산포를 뒤늦게 쟀다 `[측정]`."),
    ("E1", "처음에는 두 쌍만 보고 그렇게 읽었다"),
    ("C2", "5.10.7 에서 좁혀진다"),
])
def test_final_rules_catch_known_cases(rid, text):
    r = {x.rid: x for x in ruleset("F")}[rid]
    assert r.hits(text, dict(F_CONTEXT, plain=re.sub(r"`[^`]*`", " ", text)))


@pytest.mark.parametrize("rid,text", [
    ("C2", "p=2.4e-05 이고 +7.59 에서 대략 −0.10 이다"),           # 소수는 절 번호가 아니다
    ("C2", "다른 소프트웨어 환경(torch 2.13)에서 학습되었다"),
    ("C9", "가우시안 함수 합으로 표현된다"),                         # 「시안」 아님
    ("E1", "잔차 학습 신경망은 가성 파형 생성이 가장 적었다"),       # 적다(少)
    ("A2", "이득 보정판과는 0.5 dB 이내로 다르다"),                   # gain
    ("C4", "QRS 가 실리는 대역(D3·D4)에 2 배 무게를 준다"),           # 웨이블릿 단계
    ("B10", "심장 위상 θ 위의 가우시안 함수 합"),
])
def test_final_rules_do_not_repeat_fixed_false_positives(rid, text):
    r = {x.rid: x for x in ruleset("F")}[rid]
    assert not r.hits(text, dict(F_CONTEXT, plain=text, kind=""))


def test_sentence_split_keeps_abbreviation_periods():
    assert sr._split_sentences("나머지 3종은 n.s. 다. 다음 문장이다.") == ["나머지 3종은 n.s. 다", "다음 문장이다."]


def test_every_open_code_maps_to_a_category():
    unknown = set()
    with open(OUT / "open_coding.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            for c in filter(None, row["codes"].split(" | ")):
                code = re.match(r"([^()]+?)\s*(?:\((.*)\))?$", c).group(1).strip()
                if open_category(code, "", "F") is None and code not in OPEN_CODEBOOK:
                    unknown.add(code)
    assert not unknown, f"코드표에 없는 개방 코드: {sorted(unknown)}"


@pytest.mark.parametrize("version", ["T1", "F"])
def test_saved_candidates_match_current_rules(version):
    """규칙을 고치고 산출물을 다시 안 뽑으면 여기서 걸린다 (O-35 와 같은 이유)."""
    rows = sr.review(version)
    oc = sr.load_open(OUT / "open_coding.csv", version)
    with open(OUT / f"candidates_{version}.csv", encoding="utf-8-sig") as f:
        saved = list(csv.DictReader(f))
    fresh_auto = {(r["where"], r["text"]): r["rules"] for r in rows if r["rules"]}
    saved_auto = {(r["where"], r["text"]): r["auto"] for r in saved if r["auto"]}
    assert fresh_auto == saved_auto
    summ = json.loads((OUT / f"summary_{version}.json").read_text(encoding="utf-8"))
    assert summ["units"] == len(rows) and summ["candidates"] == len(saved)
    assert len(oc) == len({(r["where"], r["text"]) for r in rows})      # 같은 줄의 같은 문장은 한 키


def test_examples_point_at_real_candidates():
    md = (OUT / "examples_F.md").read_text(encoding="utf-8")
    with open(OUT / "candidates_F.csv", encoding="utf-8-sig") as f:
        texts = {r["text"] for r in csv.DictReader(f)}
    origs = re.findall(r"^- \*\*원문\*\* — (.*)$", md, re.M)
    assert len(origs) >= 30
    assert all(o in texts for o in origs)
