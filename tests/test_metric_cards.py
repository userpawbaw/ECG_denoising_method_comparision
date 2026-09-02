"""시연용 지표 카드 (`scripts/build_metric_cards.py`).

카드는 **말 없이 읽혀야** 하는 산출물이라, 눈으로 확인하지 않으면 못 잡는
결함이 둘 있다. 둘 다 실제로 한 번씩 냈다.

1. **matplotlib 은 마크다운을 모른다.** `**강조**` 를 쓰면 별표가 그대로
   찍힌다. 문서에서 그대로 복사해 오다 걸렸다.
2. **한글 폰트에 U+2212(−) 가 없다.** 로그축 지수가 `10¤3` 으로 나온다 —
   그림은 나오는데 못 읽는 상태가 가장 나쁘다.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "scripts" / "build_metric_cards.py"


def _mod():
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("build_metric_cards", SRC)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as e:                       # pragma: no cover
        pytest.skip(f"build_metric_cards 를 못 읽었다: {e}")
    return m


def test_no_markdown_emphasis_reaches_a_figure():
    """`**...**` 는 matplotlib 에서 별표 그대로 찍힌다."""
    bad = []
    for ln, line in enumerate(SRC.read_text().splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue                              # 주석은 그림에 안 들어간다
        for call in ("punch(", "ax.text(", "ax2.text(", "fig.text(", "card("):
            if call in line and "**" in line:
                bad.append(f"{ln}: {line.strip()[:70]}")
    assert not bad, "그림 문자열에 마크다운 강조가 있다:\n" + "\n".join(bad)


def test_a_korean_font_is_selected_and_math_falls_back():
    """한글 폰트 + mathtext 폴백이 둘 다 걸려 있어야 한다."""
    src = SRC.read_text()
    assert "_ko_font()" in src, "한글 폰트 선택이 사라졌다"
    assert '"mathtext.fontset": "dejavusans"' in src, \
        "로그축 지수의 마이너스가 깨진다 (나눔 폰트에 U+2212 가 없다)"


def test_the_index_document_is_generated_not_handwritten():
    """카드와 문서가 갈라지면 알아챌 방법이 없다 (F-9 계열)."""
    doc = ROOT / "docs" / "32_metric_cards.md"
    assert doc.exists(), "색인 문서가 없다 — 스크립트를 한 번 돌릴 것"
    assert "직접 고치지 말 것" in doc.read_text()
    m = _mod()
    ids = {c[0] for c in m.INDEX}
    assert ids == set(m.CARDS), f"INDEX 와 CARDS 가 어긋났다: {ids ^ set(m.CARDS)}"


def test_every_indexed_card_names_its_evidence():
    """근거 칸이 비면 «어디서 나온 숫자인가» 를 물을 수 없다."""
    m = _mod()
    for cid, met, what, src in m.INDEX:
        assert met and what and src, f"{cid}: 빈 칸이 있다"
        assert any(k in src for k in ("EXP-", "results/", "합성")), \
            f"{cid}: 근거가 실험·산출물·합성 중 하나를 가리켜야 한다 — {src!r}"


def test_cards_read_their_numbers_from_the_tables():
    """숫자를 손으로 박으면 실험을 다시 돌렸을 때 조용히 낡는다."""
    src = SRC.read_text()
    body = src[src.index("def c2_r_amp"):]
    assert "t.loc[" in body, "카드가 표에서 값을 읽지 않는다"
    # 카드 본문에 dB/%/ms 를 붙인 하드코딩 숫자가 없어야 한다
    hard = re.findall(r'"[^"]*?\d+\.\d+\s*(?:dB|%|ms)[^"]*?"', body)
    assert not hard, f"그림 문자열에 박아 넣은 수치가 있다: {hard}"
