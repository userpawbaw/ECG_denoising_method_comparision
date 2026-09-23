"""발표용 렌더(`--present`)가 실제로 말을 바꾸는가.

왜 따로 검사하나
----------------
`tests/test_figure_text.py` 는 **글꼴로 깨지는 글자**(마크다운 별표 · U+2212)를
본다. 여기서 보는 것은 다르다 — **읽을 수는 있는데 뜻을 모르는 글자**다.
`M04` 는 두부로 안 나오고 폰트 검사를 통과하지만, 저장소를 안 본 사람에게는
아무 뜻이 없다(`docs/94_presentation.md` 2.3).

바꾸는 자리가 `PRESENT_TERMS` 표 하나뿐이라, 표가 낡으면 **그림이 조용히 코드를
그대로 내보낸다.** 제목은 f-string 으로 조립되는 것이 많아 눈으로는 안 잡힌다.
그래서 세 가지를 고정한다.

1. 발표용 이름·축 라벨이 보고서용과 **같은 칸**을 가진다 (빠진 방법이 없다).
2. 스크립트가 그림에 넣는 문자열을 `present_text()` 에 통과시키면 **남는 코드가
   없다.**
3. 표의 항목이 전부 **실제로 쓰이는 문자열에 걸린다** — 안 걸리는 항목은 원문이
   바뀌었다는 뜻이고, 그러면 바꾸려던 그 자리가 안 바뀐 채로 남는다.
"""
from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_slides.py"

# 발표장에서 뜻이 통하지 않는 말. 괄호 안에 남겨 둔 코드(`SWT wavelet (M04)`)는
# 일부러 둔 것이라 뺀다 — 백업 슬라이드·보고서와 이으라고 붙인 꼬리다.
BANNED = re.compile(
    r"(?<!\()\b(M0\d|M10|M_FE|B0\d|M06L6|M08L6)\b"       # 괄호 밖의 방법 코드
    r"|\bsnr_imp\w*|\bEXP-[A-G]\b|\bWilcoxon\b|\bHolm\b|front-end|\bTEST\b")

# 라벨을 담아 두는 이름들. S5 의 `styles` 처럼 dict 에 label 을 넣는 자리가 있어
# matplotlib 호출만 봐서는 놓친다. **보고서용 사전(`NAME_KO` · `NAME_EN` …)은
# 일부러 뺀다** — 거기서는 코드가 옳고, 발표용 렌더는 그것을 쓰지 않는다.
LABEL_HOLDERS = re.compile(r"^(NAME_PRESENT|TXT_PRESENT|styles)$")

# `label=` 를 받는 호출이 글자 호출보다 넓다. 실제로 범례의
# `Line2D(..., label="유의 (Holm 보정 p < 0.05)")` 이 빠져 있어서 `Holm` 이
# 그대로 나갔다 — 렌더해서 눈으로 보고 잡았다.
TEXT_CALLS = {
    "text", "figtext", "annotate", "set_title", "suptitle", "title",
    "set_xlabel", "set_ylabel", "xlabel", "ylabel", "set_label",
    "set_xticklabels", "set_yticklabels", "bar_label", "legend",
    "Line2D", "Patch", "plot", "scatter", "bar", "barh", "step",
    "axhline", "axvline", "fill_between", "errorbar", "hist",
}
LABEL_KWARGS = {"label", "title", "xlabel", "ylabel", "suptitle"}


@pytest.fixture(scope="module")
def mod():
    """`make_slides` 를 **불러오기만** 한다 (그림은 안 그린다)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("make_slides_under_test", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _call_name(node: ast.Call) -> str:
    f = node.func
    return f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")


def _figure_strings() -> list[tuple[int, str]]:
    """그림에 들어갈 수 있는 문자열 (줄번호, 내용).

    matplotlib 의 글자 호출 + 라벨을 담는 이름에 대입된 리터럴. `print` 와
    독스트링, 파일 경로는 빠진다 — 그림에 안 들어간다.
    """
    src = SCRIPT.read_text(encoding="utf-8")
    out: list[tuple[int, str]] = []

    def collect(node, lineno):
        # dict 의 **키**와 `C["M01"]` 같은 **첨자**는 그림에 안 들어간다 —
        # 세면 「M01 이 남아 있다」는 거짓 경보가 난다.
        skip = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Dict):
                skip.update(id(k) for k in n.keys if k is not None)
            elif isinstance(n, ast.Subscript):
                skip.add(id(n.slice))
        for n in ast.walk(node):
            if (isinstance(n, ast.Constant) and isinstance(n.value, str)
                    and id(n) not in skip):
                out.append((getattr(n, "lineno", lineno), n.value))

    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and _call_name(node) in TEXT_CALLS:
            for a in node.args:
                collect(a, node.lineno)
            for kw in node.keywords:
                if kw.arg in LABEL_KWARGS:
                    collect(kw.value, node.lineno)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                name = tgt.id if isinstance(tgt, ast.Name) else ""
                if name and LABEL_HOLDERS.match(name):
                    collect(node.value, node.lineno)
    return out


def test_present_register_covers_every_method(mod):
    """발표용 라벨에 빠진 방법이 있으면 그 칸만 코드로 남는다."""
    assert set(mod.NAME_PRESENT) == set(mod.NAME_KO), (
        "발표용 이름표의 칸이 보고서용과 다르다: "
        f"{set(mod.NAME_KO) ^ set(mod.NAME_PRESENT)}")
    assert set(mod.TXT_PRESENT) == set(mod.TXT_KO), (
        "발표용 축 라벨의 칸이 보고서용과 다르다: "
        f"{set(mod.TXT_KO) ^ set(mod.TXT_PRESENT)}")
    # 이름표 자체에 코드만 덩그러니 남아 있으면 안 된다 (괄호 꼬리는 허용).
    bad = [f"{k}: {v}" for k, v in mod.NAME_PRESENT.items() if BANNED.search(v)]
    assert not bad, f"발표용 이름표에 코드가 그대로다: {bad}"


def test_no_jargon_survives_the_present_pass(mod):
    """그림에 들어가는 문자열을 전부 통과시켜 **남는 코드가 없어야** 한다."""
    bad = []
    for ln, v in _figure_strings():
        after = mod.present_text(v)
        if BANNED.search(after):
            hits = sorted({m.group(0) for m in BANNED.finditer(after)})
            bad.append(f"make_slides.py:{ln}: {hits} — {after.strip()[:70]}")
    assert not bad, (
        "발표용으로 바꾼 뒤에도 코드가 남는다 — `PRESENT_TERMS` 에 항목을 "
        "더할 것 (긴 것부터, 조사까지 함께):\n" + "\n".join(bad))


def test_every_term_still_matches_something(mod):
    """표의 항목이 원문에 안 걸리면, 바꾸려던 자리가 **안 바뀐 채로** 남는다."""
    pool = [v for _, v in _figure_strings()]
    pool += list(mod.NAME_KO.values()) + list(mod.TXT_KO.values())
    blob = "\n".join(pool)
    dead = [a for a, _ in mod.PRESENT_TERMS if a not in blob]
    assert not dead, (
        "`PRESENT_TERMS` 의 이 항목이 어디에도 안 걸린다 — 원문이 바뀌었거나 "
        f"오타다. 그 자리는 지금 코드 그대로 나간다: {dead}")


def test_present_terms_are_ordered_longest_first(mod):
    """짧은 항목이 먼저 걸리면 긴 항목이 영영 안 걸린다.

    `snr_imp_scaled` 가 `기준 대비 Δ  snr_imp_scaled [dB]` 보다 **앞**에 있으면,
    긴 쪽은 영영 안 걸리고 축 라벨이 반만 바뀐 채로 나간다.
    """
    terms = [a for a, _ in mod.PRESENT_TERMS]
    bad = [(a, b) for i, a in enumerate(terms) for b in terms[i + 1:] if a in b]
    assert not bad, (
        "앞 항목이 뒤 항목의 **부분 문자열**이다 — 앞이 먼저 먹어서 뒤가 영영 "
        f"안 걸린다. 긴 쪽을 위로 올릴 것: {bad}")


# ---------------------------------------------------------------- 산출물
PRESENT = ROOT / "results" / "slides_present"
REPORT = ROOT / "results" / "slides"


def test_present_set_matches_the_report_set():
    """발표용 묶음이 보고서용과 **같은 그림 목록**이어야 한다.

    한쪽에만 있는 그림이 생기면 발표 자료가 보고서에 없는 것을 말하거나,
    반대로 발표에서 한 장이 조용히 빠진다.
    """
    if not PRESENT.exists():
        pytest.skip("발표용 묶음이 아직 없다 — `make_slides.py --present`")
    a = {f.name for f in PRESENT.glob("*.png")}
    b = {f.name for f in REPORT.glob("*.png")}
    assert a == b, f"두 묶음의 그림이 다르다: 발표용만 {a - b}, 보고서용만 {b - a}"
