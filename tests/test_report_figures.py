"""보고서의 **그림 링크가 실재하는가**.

2 500 줄 문서에서 깨진 이미지 하나는 **눈으로 안 잡힌다** — 렌더링해 보기
전까지 조용하고, 렌더링해도 «그림이 하나 없네» 로 넘어가기 쉽다. 그런데
보고서에서 그림은 장식이 아니라 **근거**라, 빠지면 그 절의 주장이 근거를 잃는다.

그리고 그림은 `results/` 아래에 있고 그것을 만드는 스크립트가 따로 있으므로,
**스크립트 이름이 바뀌거나 산출 경로가 바뀌면 링크가 조용히 끊긴다.**
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
IMG = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def _docs_with_images():
    for p in sorted(DOCS.glob("*.md")):
        if IMG.search(p.read_text()):
            yield p


@pytest.mark.parametrize("doc", list(_docs_with_images()), ids=lambda p: p.name)
def test_every_figure_link_resolves(doc):
    """문서가 가리키는 그림이 저장소에 실제로 있어야 한다."""
    missing = []
    for rel in IMG.findall(doc.read_text()):
        if rel.startswith(("http://", "https://")):
            continue
        if not (doc.parent / rel).resolve().exists():
            missing.append(rel)
    assert not missing, f"{doc.name}: 없는 그림을 가리킨다 — {missing}"


def test_the_report_actually_shows_its_evidence():
    """**보고서에 그림이 있어야 한다.**

    한동안 `91_report.md` 는 2 500 줄에 그림이 **0 장**이었다. 표와 글은
    충실했지만, «곡선이 교차한다» 나 «계단이 블록 경계에 맞는다» 같은 것은
    **문장으로 대신할 수 없다.** 최소한 결과 장과 지표 장에는 있어야 한다.
    """
    txt = (DOCS / "91_report.md").read_text()
    n = len(IMG.findall(txt))
    assert n >= 12, f"보고서의 그림이 {n} 장뿐이다 — 근거를 글로만 말하고 있다"

    # 장별로 최소 한 장씩. 그림이 한 장에 몰리면 나머지 장은 여전히 글뿐이다.
    for head, name in (("# 3. 평가지표", "지표"), ("# 5. 결과와 해석", "결과")):
        i = txt.index(head)
        j = txt.find("\n# ", i + 1)
        assert IMG.search(txt[i: j if j != -1 else len(txt)]), \
            f"{name} 장에 그림이 하나도 없다"
