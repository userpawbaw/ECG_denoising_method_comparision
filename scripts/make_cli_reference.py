#!/usr/bin/env python3
"""`docs/24_cli_reference.md` 를 만든다 — **실행 스크립트의 옵션 전부.**

    python3 scripts/make_cli_reference.py          # 재생성
    python3 scripts/make_cli_reference.py --check  # 낡았는지만 본다 (CI·테스트용)

왜 손으로 안 쓰나
----------------
옵션이 167 개다. 손으로 적으면 **즉시 낡고**, 낡은 것을 알아챌 방법이 없다.
그래서 각 스크립트의 `ArgumentParser` 를 **실제로 만들어** 거기서 읽는다 —
`--help` 를 파싱하는 것과 달리 기본값·선택지·타입이 원본 그대로 나온다.

어떻게 얻나
----------
스크립트는 `main()` 안에서 parser 를 만들고 바로 `parse_args()` 를 부른다.
그래서 `parse_args` 를 잠시 가로채 **그 시점의 parser 를 들고 빠져나온다.**
스크립트를 실행하지 않으므로 학습이 돌거나 파일이 써지지 않는다.

`help=` 가 비어 있으면 표에 **`(설명 없음)`** 으로 남는다 — 지우지 말 것.
그것이 「여기를 채워라」는 표시이고, `--check` 는 그것 때문에 실패하지 않는다.

규약
----
이 문서는 **자동 생성**이다. 고칠 것이 있으면 문서가 아니라
**스크립트의 `help=`** 를 고치고 다시 생성한다 (`docs/17_checklists.md` §4).
"""
import _bootstrap  # noqa: F401

import argparse
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "24_cli_reference.md"
SELF = Path(__file__).name

# 접두사로 묶는다. **직접 실행하는 것이 위로** 오게 한 순서다.
GROUPS = [
    ("시연 — 이 둘만 직접 띄운다", "실시간 시연(모드 A)의 실행 파일. 절차는 "
     "`docs/30_realtime_demo.md` 6.2~6.3.",
     ("fake_arduino", "serial_bridge")),
    ("데이터 준비", "외부 데이터를 받고, 아두이노에서 기록한다.",
     ("download_data", "log_arduino")),
    ("학습 · 실험", "체크포인트와 실험 결과를 만든다. **무인 실행 규칙**이 "
     "붙는다 — `docs/17_checklists.md` §2.",
     ("train", "run_exp", "run_seed_sweep", "overfit_test", "tune_swt",
      "run_safety_probe")),
    ("산출물 생성", "보고서·그림·슬라이드·시연 카드. 자동 생성 문서 규칙이 "
     "붙는다 — §4.",
     ("make_report", "make_slides", "make_ablation_table", "make_cli_reference",
      "build_demo_bank", "build_card_bank", "build_metric_cards")),
    ("검증 · 분석", "숫자를 새로 만들지 않고 **이미 있는 것을 확인하거나 재는** "
     "것들.", ()),
]


def collect_parser(path: Path):
    """스크립트를 실행하지 않고 그 `ArgumentParser` 만 꺼낸다."""

    class _Caught(Exception):
        def __init__(self, parser):
            self.parser = parser

    real = argparse.ArgumentParser.parse_args

    def fake(self, *a, **kw):
        raise _Caught(self)

    argparse.ArgumentParser.parse_args = fake
    try:
        spec = importlib.util.spec_from_file_location("cli_" + path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            spec.loader.exec_module(mod)
            mod.main()
        return None, "parse_args 까지 안 갔다"
    except _Caught as c:
        return c.parser, None
    except BaseException as e:                               # pragma: no cover
        return None, f"{type(e).__name__}: {str(e)[:80]}"
    finally:
        argparse.ArgumentParser.parse_args = real


def summary_of(path: Path) -> str:
    """모듈 docstring 의 첫 줄 — 「이 스크립트가 무엇인가」.

    따옴표를 손으로 찾으면 **shebang 에 걸린다** (`#!/usr/bin/env python3` 가
    먼저 온다). 그래서 `ast` 에게 묻는다.
    """
    import ast

    try:
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    except SyntaxError:                                      # pragma: no cover
        return ""
    for line in (doc or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def _cell(s: object) -> str:
    """표 한 칸. `|` 가 들어가면 표가 깨지므로 막는다."""
    return str(s).replace("|", "\\|").replace("\n", " ")


def describe(action) -> tuple[str, str, str]:
    """(플래그, 값, 기본값) 세 칸."""
    flags = " · ".join(f"`{o}`" for o in action.option_strings)
    if isinstance(action, argparse._StoreTrueAction):
        value = "**켬/끔**"
    elif action.choices:
        value = " · ".join(f"`{c}`" for c in action.choices)
    elif action.type is not None:
        value = {int: "정수", float: "실수", str: "문자열"}.get(
            action.type, getattr(action.type, "__name__", "값"))
    else:
        value = "문자열"
    if isinstance(action, argparse._StoreTrueAction):
        default = "끔"
    elif action.default is None:
        default = "—"
    else:
        default = f"`{action.default}`"
    if action.required:
        default = "**필수**"
    return flags, value, default


def render(scripts: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# 24. 실행 스크립트 옵션 — **전부**")
    A("")
    A("> **이 문서는 자동 생성이다.** 고칠 것이 있으면 문서가 아니라")
    A("> **스크립트의 `help=`** 를 고치고 다시 만든다:")
    A(">")
    A("> ```bash")
    A("> python3 scripts/make_cli_reference.py")
    A("> ```")
    A(">")
    A("> `tests/test_repo_integrity.py` 가 이 파일과 스크립트를 대조한다 —")
    A("> 옵션을 늘리고 재생성을 안 하면 거기서 걸린다 (O-31).")
    A("")
    A("**왜 이 문서가 있나.** 매뉴얼은 「이 절차를 따라 하면 된다」까지만")
    A("데려다준다. 스스로 여러 판을 돌려보려면 **무엇을 고를 수 있는지**가")
    A("한자리에 있어야 한다 — 실제로 `--record`·`--noise` 를 못 찾아")
    A("예시 한 조합만 돌려본 일이 있었다(O-31).")
    A("")
    A("각 절차서는 자주 쓰는 몇 개만 설명하고 **전부는 여기를 가리킨다.**")
    A("")

    total = sum(len(v["actions"]) for v in scripts.values())
    A(f"| | |")
    A("|---|---|")
    A(f"| 스크립트 | **{len(scripts)} 개** |")
    A(f"| 옵션 | **{total} 개** |")
    A(f"| 설명이 빈 옵션 | **{sum(1 for v in scripts.values() for a in v['actions'] if not a.help)} 개** (`(설명 없음)` 으로 표시된다) |")
    A("")
    A("---")
    A("")

    named = {n for _, _, names in GROUPS for n in names}
    rest = sorted(set(scripts) - named)
    for title, note, names in GROUPS:
        members = [n for n in names if n in scripts] if names else rest
        if not members:
            continue
        A(f"## {title}")
        A("")
        A(note)
        A("")
        for name in members:
            info = scripts[name]
            A(f"### `scripts/{name}.py`")
            A("")
            if info["summary"]:
                A(f"{_cell(info['summary'])}")
                A("")
            if not info["actions"]:
                A("옵션 없음.")
                A("")
                continue
            A("| 옵션 | 값 | 기본 | 설명 |")
            A("|---|---|---|---|")
            for a in info["actions"]:
                flags, value, default = describe(a)
                help_s = _cell(a.help) if a.help else "**(설명 없음)**"
                A(f"| {flags} | {value} | {default} | {help_s} |")
            A("")
    return "\n".join(L) + "\n"


def gather() -> tuple[dict, list]:
    scripts, failed = {}, []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        if path.name.startswith("_"):
            continue
        src = path.read_text(encoding="utf-8")
        if "add_argument" not in src and "ArgumentParser" not in src:
            continue
        parser, err = collect_parser(path)
        if parser is None:
            failed.append((path.name, err))
            continue
        actions = [a for a in parser._actions
                   if a.option_strings and not isinstance(a, argparse._HelpAction)]
        scripts[path.stem] = {"summary": summary_of(path), "actions": actions}
    return scripts, failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="다시 만들지 않고 **낡았는지만** 본다. 낡았으면 1 을 낸다")
    ap.add_argument("--out", default=str(OUT),
                    help="쓸 파일. 기본은 docs/24_cli_reference.md")
    args = ap.parse_args()

    scripts, failed = gather()
    if failed:
        for name, err in failed:
            print(f"[warn] {name} 에서 parser 를 못 꺼냈다: {err}")
        print("  -> 그 스크립트의 옵션은 문서에 안 실린다. main() 이 "
              "parse_args() 를 부르는지 확인할 것")
    body = render(scripts)
    out = Path(args.out)
    if args.check:
        old = out.read_text(encoding="utf-8") if out.exists() else ""
        if old == body:
            print(f"최신이다 — 스크립트 {len(scripts)} 개")
            return 0
        print(f"{out} 가 낡았다. `python3 scripts/{SELF}` 로 다시 만들 것")
        return 1
    out.write_text(body, encoding="utf-8")
    n_opt = sum(len(v["actions"]) for v in scripts.values())
    n_empty = sum(1 for v in scripts.values() for a in v["actions"] if not a.help)
    print(f"-> {out}  (스크립트 {len(scripts)} · 옵션 {n_opt} · "
          f"설명 없는 것 {n_empty})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
