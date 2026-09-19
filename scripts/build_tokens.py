#!/usr/bin/env python3
"""`ui/palette.json` 하나에서 **CSS 와 Python 을 함께 만든다**.

## 왜

색이 네 파일에 각각 하드코딩돼 있었고, 그래서 같은 방법이 파일마다 다른 색이었다
(U-2). 방법 색만의 문제가 아니었다 — 역할 색 `INK` 도 슬라이드 `#0b0b0b` · 카드
`#1b1b1b` 로 갈려 있었다.

원본을 하나로 두고 **생성**하면 어긋날 자리가 없어진다.

    ui/palette.json
       ├→ demo/ui/tokens.css     (demo/*.html 이 읽는다)
       └→ scripts/_palette.py    (make_slides · build_metric_cards 가 읽는다)

생성물은 **git 에 넣는다.** 시연 화면이 `file://` 로 열려야 하므로 빌드 단계를
전제할 수 없다(UD-1 3.1). 대신 `--check` 로 「원본과 생성물이 맞는지」를 검사에
걸어, 원본만 고치고 생성을 잊는 사고를 막는다 — 자동 생성 문서에 쓰는 것과 같은
장치다(`docs/17_checklists.md` §4).

## 쓰기

    python3 scripts/build_tokens.py           # 생성
    python3 scripts/build_tokens.py --check   # 최신인지만 확인 (어긋나면 종료 1)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "ui" / "palette.json"
CSS_OUT = ROOT / "demo" / "ui" / "tokens.css"
PY_OUT = ROOT / "scripts" / "_palette.py"

BANNER = "ui/palette.json 에서 생성됨 — 직접 고치지 말 것 (scripts/build_tokens.py)"


def _load() -> dict:
    return json.loads(SRC.read_text(encoding="utf-8"))


def _pairs(node: dict) -> list[tuple[str, str, str | None]]:
    """`_` 주석 키를 빼고 (이름, light, dark) 를 뽑는다."""
    out = []
    for k, v in node.items():
        if k == "_":
            continue
        if isinstance(v, str):                      # kinds 처럼 값이 곧 색인 경우
            out.append((k, v, None))
        elif isinstance(v, dict) and "light" in v:
            out.append((k, v["light"], v.get("dark")))
    return out


def render_css(p: dict) -> str:
    L: list[str] = [f"/* {BANNER} */", ""]

    # 라이트를 bare :root 에 **전부** 깐다. 다크 블록에는 **다른 값만** 둔다 —
    # 어떤 색의 유일한 정의가 다크 블록 안에 있으면, 테마를 안 고른 사용자에게는
    # 그 색이 아예 적용되지 않는다.
    # **@font-face 가 먼저다.** 스택에 이름만 적어 두면 그 서체가 없는 기계에서는
    # 대체 서체로 그려지고, 한글 대체 서체에는 굵은 판이 없어 위계가 통째로
    # 사라진다 — 그것을 못 보고 여러 커밋을 지났다 (UF-6).
    faces = p["typography"].get("faces")
    if faces:
        for f in faces["list"]:
            L += [
                "@font-face{",
                f"  font-family: \"{f['family']}\";",
                f"  src: url(\"{faces['dir']}{f['file']}\") format(\"woff2\");",
                f"  font-weight: {f['weight']};",
                "  font-style: normal;",
                "  font-display: swap;",
                "}",
            ]
        L.append("")

    L.append(":root{")
    groups = [("method", p["methods"]), ("family", p["families"]),
              ("role", p["roles"]), ("kind", p["kinds"])]
    for prefix, node in groups:
        L.append(f"  /* {prefix} */")
        for name, light, _dark in _pairs(node):
            L.append(f"  --{prefix}-{name.lower().replace('_', '-')}: {light};")
    L.append("  /* ramp — ordinal, 순서가 의미를 가진다 */")
    for ramp, steps in p["ramps"].items():
        if ramp == "_":
            continue
        for step, hexv in steps.items():
            L.append(f"  --ramp-{ramp}-{step.lower()}: {hexv};")
    L.append("  /* type */")
    for fam, stack in p["typography"]["families"].items():
        L.append(f"  --font-{fam}: {stack};")
    L.append("}")
    L.append("")

    darks: list[str] = []
    for prefix, node in groups:
        for name, _light, dark in _pairs(node):
            if dark:
                darks.append(f"  --{prefix}-{name.lower().replace('_', '-')}: {dark};")
    if darks:
        L += [
            "/* 다크는 **자동 반전이 아니라 따로 고른 값**이다. 아직 계열과 표면만",
            "   있고 방법 색은 [미정] — 다크 표면 기준으로 다시 골라야 한다",
            "   (지금 시안의 다크 4 색은 명도대를 넷 다 벗어난다). docs/ui/01_system.md 7 절. */",
            "@media (prefers-color-scheme: dark){ :root:not([data-theme=\"light\"]){",
            *darks,
            "}}",
            "",
            ":root[data-theme=\"dark\"]{",
            *darks,
            "}",
            "",
        ]

    vp = p["viewport"]
    L += [f"/* 표시 장치: {vp['target']['w']}×{vp['target']['h']} · "
          f"최소 {vp['min']['w']}×{vp['min']['h']} · 검사 폭 "
          f"{', '.join(str(w) for w in vp['test_widths'])} */"]
    return "\n".join(L) + "\n"


def render_py(p: dict) -> str:
    def lit(node, key="light"):
        return {n: l for n, l, _ in _pairs(node)}

    methods = lit(p["methods"])
    families = lit(p["families"])
    roles = lit(p["roles"])
    kinds = {k: v for k, v in p["kinds"].items() if k != "_"}
    loss = {k: v for k, v in p["ramps"]["loss"].items() if k != "_"}

    def fmt(d: dict, indent: str = "    ") -> str:
        return "\n".join(f'{indent}{k!r}: {v!r},' for k, v in d.items())

    return f'''"""{BANNER}

`scripts/make_slides.py` 와 `scripts/build_metric_cards.py` 가 이것을 읽는다.
값을 고칠 곳은 `ui/palette.json` 이고, 고친 뒤 `python3 scripts/build_tokens.py`.
"""

# 방법 → 색. **색은 방법에 고정한다** (docs/33).
METHODS = {{
{fmt(methods)}
}}

# 계열 → 색. 시연 화면처럼 방법이 많을 때 쓴다.
FAMILIES = {{
{fmt(families)}
}}

# 역할 색.
ROLES = {{
{fmt(roles)}
}}

# 개입의 종류 (구조 / 손실).
KINDS = {{
{fmt(kinds)}
}}

# 손실 램프 — ordinal 이다. 순서를 바꾸면 의미가 달라진다.
LOSS = {{
{fmt(loss)}
}}

# ---- 기존 이름 (호출부를 덜 고치려고 그대로 둔다) ----
C = METHODS
KIND = KINDS
CLEAN = ROLES["clean"]
NOISY = ROLES["noisy"]
INK = ROLES["ink"]
INK2 = ROLES["ink_2"]
MUTE = ROLES["ink_2"]
SURFACE = ROLES["surface"]
BAD = ROLES["bad"]
'''


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="팔레트 원본에서 토큰을 생성한다")
    ap.add_argument("--check", action="store_true",
                    help="생성하지 않고 **최신인지만** 확인한다 (어긋나면 종료 1)")
    a = ap.parse_args(argv)

    p = _load()
    want = {CSS_OUT: render_css(p), PY_OUT: render_py(p)}

    if a.check:
        stale = [f for f, text in want.items()
                 if not f.exists() or f.read_text(encoding="utf-8") != text]
        if stale:
            for f in stale:
                print(f"낡음: {f.relative_to(ROOT)}", file=sys.stderr)
            print("→ python3 scripts/build_tokens.py 를 돌릴 것", file=sys.stderr)
            return 1
        print(f"최신 — {len(want)} 개 생성물이 ui/palette.json 과 맞는다")
        return 0

    for f, text in want.items():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")
        print(f"  {f.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
