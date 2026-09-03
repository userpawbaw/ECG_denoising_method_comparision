"""그림에 들어가는 문자열의 함정 — **스크립트 전체**를 본다.

`tests/test_metric_cards.py` 에 같은 검사가 있지만 `build_metric_cards.py`
**한 파일에만** 걸려 있었다. 그래서 `compare_median_vs_zerophase.py` 를 새로
짜면서 `**강조**` 가 그림에 별표로 그대로 찍혔다 — **가드의 범위가 좁으면
같은 실수가 새 파일에서 그대로 재발한다.**

검사 대상을 고르는 법이 이 파일의 핵심이다. 「스크립트 안의 모든 한글
문자열」로 잡으면 **`print` 문과 생성되는 마크다운 문서까지** 걸린다 —
문서에서는 `**강조**` 가 정상이다. 그래서 **matplotlib 의 글자 넣는 호출에
실제로 들어가는 문자열만** AST 로 골라낸다.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# 글자를 그림에 넣는 matplotlib 호출. 여기 들어가는 문자열만 검사 대상이다.
TEXT_CALLS = {
    "text", "figtext", "annotate", "set_title", "suptitle", "title",
    "set_xlabel", "set_ylabel", "xlabel", "ylabel", "set_label",
    "set_xticklabels", "set_yticklabels", "bar_label", "legend",
    # 이 저장소의 카드 헬퍼들
    "punch", "card",
}
LABEL_KWARGS = {"label", "title", "xlabel", "ylabel", "suptitle"}


def _figure_scripts() -> list[tuple[Path, str]]:
    out = []
    for p in sorted(SCRIPTS.glob("*.py")):
        src = p.read_text(encoding="utf-8")
        if "matplotlib" in src and "savefig" in src:
            out.append((p, src))
    return out


def _call_name(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def _figure_strings(src: str) -> list[tuple[int, str]]:
    """그림에 실제로 들어가는 문자열 (줄번호, 내용)."""
    out: list[tuple[int, str]] = []

    def collect(node, lineno):
        for n in ast.walk(node):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                out.append((getattr(n, "lineno", lineno), n.value))

    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node) in TEXT_CALLS:
            for a in node.args:
                collect(a, node.lineno)
        for kw in node.keywords:
            if kw.arg in LABEL_KWARGS:
                collect(kw.value, node.lineno)
    return out


def _has_hangul(s: str) -> bool:
    return any("가" <= c <= "힣" for c in s)


def _korean_figure_scripts():
    return [(p, src) for p, src in _figure_scripts()
            if any(_has_hangul(v) for _, v in _figure_strings(src))]


def test_no_markdown_emphasis_reaches_any_figure():
    """`**...**` 는 matplotlib 에서 별표 그대로 찍힌다."""
    bad = []
    for p, src in _figure_scripts():
        for ln, v in _figure_strings(src):
            if "**" in v:
                bad.append(f"{p.name}:{ln}: {v.strip()[:64]}")
    assert not bad, ("그림에 들어가는 문자열에 마크다운 강조가 있다 — "
                     "matplotlib 은 이것을 별표로 찍는다:\n" + "\n".join(bad))


def test_scripts_that_draw_korean_select_a_korean_font():
    """한글 라벨을 쓰면서 폰트를 안 고르면 전부 두부(□)로 나온다."""
    # 직접 목록을 갖거나, 목록을 가진 헬퍼를 들여왔으면 된다.
    bad = [p.name for p, src in _korean_figure_scripts()
           if "NanumGothic" not in src and "_ko_font" not in src]
    assert not bad, f"그림에 한글을 넣는데 폰트 선택이 없다: {bad}"


def test_scripts_that_draw_korean_disable_unicode_minus():
    """나눔 폰트에 U+2212(−) 가 없다 — 음수 눈금이 깨진다."""
    bad = [p.name for p, src in _korean_figure_scripts()
           if "unicode_minus" not in src and "_ko_font" not in src]
    assert not bad, f"음수 눈금이 깨진다 — `axes.unicode_minus: False` 가 없다: {bad}"


def test_korean_scripts_with_log_axes_fall_back_for_mathtext():
    """로그축 지수는 mathtext 로 그려진다 — 한글 폰트로 그리면 `10¤3` 이 된다."""
    marks = ("set_xscale", "set_yscale", "semilogx", "semilogy", "loglog")
    bad = [p.name for p, src in _korean_figure_scripts()
           if any(m in src for m in marks) and "mathtext.fontset" not in src]
    assert not bad, ("로그축 지수의 마이너스가 깨진다 — "
                     f"`mathtext.fontset` 폴백이 없다: {bad}")
