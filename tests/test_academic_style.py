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
    "확장 칼만 필터는 위양성 검출 위치에서 가성 QRS군을 생성하였다.",
    "실측 기록에서는 임계값 이하 계수 제거(경임계값 처리)와 QRS 구간 완화 미적용이 최적 설정이었다.",
    "웨이블릿을 전처리로 직렬 결합하면 SNR 개선량이 1.83 dB 감소하였다(p = 0.002).",
    "본 연구의 판넬과 판정 기준은 공통 전처리를 전제로 한다.",
    "모든 조건에서 최적인 단일 기법은 관찰되지 않았다.",
])
def test_academic_sentences_pass(good: str):
    assert not cas.check_text(good), cas.check_text(good)


def test_term_registry_rows_have_sources():
    """용어 대조 기록(docs/95 §3.1)의 모든 줄에 출처와 확인일이 있다 (A6).

    「특징 지도」처럼 출처 없이 직역한 말이 대외 글에 들어간 일이 있었다(2026-11-04 사용자 지적).
    표에 올리는 순간 출처를 적게 하여, 확인하지 않은 용어가 기준표로 굳는 것을 막는다.
    """
    import re
    from pathlib import Path
    doc = (Path(__file__).resolve().parents[1] / "docs" / "95_academic_style.md").read_text(encoding="utf-8")
    sec = doc.split("### 3.1 용어 대조 기록", 1)[1].split("\n## ", 1)[0]
    rows = [r for r in sec.splitlines() if r.startswith("|") and not r.startswith("|---")][1:]
    assert rows, "용어 대조 기록 표가 비었다"
    for r in rows:
        cells = [c.strip() for c in r.strip("|").split("|")]
        assert len(cells) == 5, f"칸 수가 다르다: {r}"
        assert len(cells[3]) >= 8, f"출처가 비었다: {cells[0]}"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", cells[4]), f"확인일 형식: {cells[0]}"


def test_ui_lines_keeps_visible_text_and_drops_code():
    """화면 파일 검사(docs/39): 주석 · 삼항 연산자의 «?» · 빈 값 «—» 는 빼고, 보이는 글 · 툴팁 · 문자열 상수는 본다."""
    from check_academic_style import check_text, ui_lines
    html = (
        "<!-- 개발 메모: 녹음본 -->\n"
        "<style>/* 그린다 */ .a{}</style>\n"
        '<div title="툴팁에서 지어낸 말"><span id="lat">—</span></div>\n'
        "<p>전극 신호를 처리한다</p>\n"
        "<script>\n"
        "// 주석: 손해\n"
        "const s = ok ? '연결됨' : '끊김 — 다시 붙는 중';\n"
        "</script>\n"
    )
    lines = dict(ui_lines(html, ".html"))
    assert 1 not in lines and 2 not in lines            # 주석 · style
    assert "지어낸" in lines[3] and "—" not in lines[3]  # 툴팁은 보고, 빈 값 표시는 뺀다
    assert lines[4].strip() == "전극 신호를 처리한다"
    v = [x for ln, t in ui_lines(html, ".html") for x in check_text(t, f"{ln} ")]
    assert not any("[D5]" in x for x in v)              # 삼항 연산자는 의문문이 아니다
    assert any(x.startswith("7 ") and "[B3]" in x for x in v)
    assert not any("손해" in x or "녹음" in x or "그린다" in x for x in v)
    js = "const a = 1; // 주석은 녹음\nel.textContent = `${n} 로 바꿨다`;\n"
    assert [ln for ln, _ in ui_lines(js, ".js")] == [2]
