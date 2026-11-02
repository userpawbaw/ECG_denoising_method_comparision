"""대외 글의 학술 서술 검사 (D-43, `docs/95_academic_style.md`).

판넬 · 브로셔 문안(`scripts/build_panel.py`)과 판넬에 실리는 그림의 한글 문자열에
금지 어휘가 없어야 한다. 금지 어휘의 원천은 `scripts/check_academic_style.py` 의
`LEXICON` 하나다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_academic_style as cas  # noqa: E402


def test_panel_text_has_no_banned_expressions():
    v = cas.check_targets()
    assert not v, "대외 글에 금지 표현이 있다 (docs/95_academic_style.md):\n  " + "\n  ".join(v[:30])


def test_checker_actually_sees_the_panel_text():
    """검사 대상이 비면 «위반 0» 은 아무 뜻이 없다 — 대상 수를 확인한다."""
    t = cas.targets()
    files = {f for f, _, _ in t}
    assert {"build_panel.py", "make_slides.py"} <= files
    assert len(t) > 80


# 사용자가 짚은 세 표현(D-43 의 출발점)은 반드시 걸려야 한다
@pytest.mark.parametrize("bad", [
    "심박 모양 가짜 213 개",
    "칼만은 없는 자리에 깨끗한 심박을 그린다",
    "실데이터(D1) 튜닝은 hard · QRS 보호 끔을 골랐다",
    "M05 가 front-end 와 같은 자로 재면 못 가림",
])
def test_known_bad_phrases_are_caught(bad: str):
    assert cas.check_text(bad), f"검사기가 못 잡는다: {bad}"


# 학술 문장은 걸리면 안 된다 — 오탐이 잦으면 규칙을 피하려고 문장을 비틀게 된다
@pytest.mark.parametrize("good", [
    "확장 칼만 평활기는 위양성 검출 위치에서 가성 QRS군을 생성하였다.",
    "실측 기록에서는 임계값 이하 계수 제거(경임계값 처리)와 QRS 구간 완화 미적용이 최적 설정이었다.",
    "웨이블릿을 전처리로 직렬 결합하면 SNR 개선량이 1.83 dB 감소하였다(p = 0.002).",
    "본 연구의 판넬과 판정 기준은 공통 전처리를 전제로 한다.",
    "모든 조건에서 최적인 단일 기법은 관찰되지 않았다.",
])
def test_academic_sentences_pass(good: str):
    assert not cas.check_text(good), cas.check_text(good)
