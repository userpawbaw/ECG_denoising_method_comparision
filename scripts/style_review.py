#!/usr/bin/env python3
"""학술 서술 **검토 후보 추출기** — 규칙판(T1 · F)으로 전체 서술에서 수정 후보 문장을 넓게 추린다.

    python scripts/style_review.py --rules T1 --open results/style_review/open_coding.csv   # 임시 규칙 1 + 대조
    python scripts/style_review.py --rules F  --open results/style_review/open_coding.csv   # 최종 규칙 + 대조

규칙의 내용(범주 정의 · 판정 기준 · 교정 지침 · 탐지 신호)은 `scripts/style_rules.py` 에 있다.
규칙판은 **개방 코딩 → T1 → 대조 → F** 순서로 만들었다(D-43 보완, `docs/95_academic_style.md` §7).
규칙부터 쓰고 추리면 규칙이 놓친 유형을 찾을 방법이 없기 때문이다.

산출: `results/style_review/`
    candidates_<판>.csv   후보 문장 전체 (출처 · 위치 · 문장 · 걸린 범주 · 걸린 표현 · 판정 방식)
    summary_<판>.json     범주별 · 출처별 집계
    rules_<판>.md         그 판의 규칙 전문
    compare_<판>.md       개방 코딩과의 대조 (범주별 재현율 · 규칙만 잡은 문장 · 규칙이 놓친 문장)

`check_academic_style.py` 와 역할이 다르다. 그쪽은 판넬에 **들어가면 안 되는 것**을 막는 엄격한
관문(위반 0 이어야 통과)이고, 이쪽은 **사람이 판단할 후보**를 넓게 모으는 도구다 — 「애매하면 포함」이
원칙이라 오탐이 많은 것이 정상이다.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from typing import NamedTuple
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "style_review"
HANGUL = re.compile(r"[가-힣]")
sys.path.insert(0, str(ROOT / "scripts"))
from style_rules import F_CONTEXT, GROUPS, open_category, ruleset  # noqa: E402

PANEL_SIDE = {"판넬", "판넬 그림", "브로셔"}


# =====================================================================  말뭉치
class Unit(NamedTuple):
    source: str
    where: str
    text: str
    chapter: str = ""            # 보고서의 장(## 제목) — 「처음 나올 때」를 세는 단위


_ABBR_END = re.compile(r"(?:n\.s\.|e\.g\.|i\.e\.|vs\.|cf\.|al\.|approx\.)$")


def _split_sentences(par: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[다음함됨임요])\.\s+|(?<=[.!?])\s+(?=[가-힣A-Z`「(])", par) if p.strip()]
    out: list[str] = []
    for p in parts:                       # 「n.s. 다」처럼 약어 마침표에서 끊긴 것을 다시 붙인다
        if out and _ABBR_END.search(out[-1]):
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out


def _md_units(path: Path, source: str, start: str | None = None, stop: str | None = None) -> list[Unit]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out, fence, par, par_ln, on = [], False, [], 0, start is None
    chap = [""]

    def flush():
        nonlocal par
        if par:
            for s in _split_sentences(" ".join(par)):
                if HANGUL.search(s):
                    out.append(Unit(source, f"{path.name}:{par_ln}", s, chap[0]))
        par = []

    for i, raw in enumerate(lines, 1):
        if not on:
            if start and raw.startswith(start):
                on = True
            else:
                continue
        elif stop and raw.startswith(stop) and i > 1:
            flush(); break
        line = raw.rstrip()
        if line.startswith("```"):
            flush(); fence = not fence; continue
        if fence:
            continue
        s = line.lstrip("> ").strip()
        if not s or s.startswith("![") or re.match(r"^\|?\s*-{3}", s) or s.startswith("<!--") or s == "---":
            flush(); continue
        if s.startswith("|"):
            flush()
            for cell in s.strip("|").split("|"):
                c = cell.strip()
                if HANGUL.search(c):
                    out.append(Unit(source, f"{path.name}:{i} (표)", c, chap[0]))
            continue
        if s.startswith("#"):
            flush()
            if re.match(r"^##\s", s):
                chap[0] = s.lstrip("# ").strip()[:40]
            out.append(Unit(source, f"{path.name}:{i} (제목)", s.lstrip("# ").strip(), chap[0]))
            continue
        m = re.match(r"^(?:[-*]|\d+\.)\s+(.*)", s)
        if m:
            flush(); par_ln = i; par = [m.group(1)]; flush(); continue
        if not par:
            par_ln = i
        par.append(s)
    flush()
    return out


def _py_units(path: Path, funcs: dict[str, str], consts: dict[str, str]) -> list[Unit]:
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_academic_style import _strings
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in tree.body:
        src = None
        if isinstance(node, ast.FunctionDef) and node.name in funcs:
            src = funcs[node.name]
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in consts for t in node.targets):
            src = next(consts[t.id] for t in node.targets if isinstance(t, ast.Name) and t.id in consts)
        if src:
            for ln, s in _strings(node):
                for piece in s.split("\n"):
                    for sent in (_split_sentences(piece) if SPLIT_PY[0] else [piece]):
                        if HANGUL.search(sent):
                            out.append(Unit(src, f"{path.name}:{ln}", sent.strip()))
    return out


SPLIT_PY = [True]


def corpus() -> list[Unit]:
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_academic_style import PANEL_CONSTS, PANEL_FIGS
    u = _py_units(ROOT / "scripts" / "build_panel.py", {"build_panel": "판넬", "build_brochure": "브로셔"}, {})
    u += _py_units(ROOT / "scripts" / "make_slides.py", {f: "판넬 그림" for f in PANEL_FIGS},
                   {c: "판넬 그림" for c in PANEL_CONSTS})
    u += _md_units(ROOT / "docs" / "91_report.md", "보고서")
    u += _md_units(ROOT / "docs" / "94_panel_plan.md", "해설 노트", start="## 10. 발표자 해설 노트")
    return u


# =====================================================================  실행
def _kind(where: str) -> str:
    m = re.search(r"\((표|제목)\)$", where)
    return m.group(1) if m else ""


def _doc(u: Unit) -> str:
    """약어 · 고유 용어의 「처음 나올 때」를 세는 단위 — 판넬 · 브로셔 · 판넬 그림은 지면 하나, 보고서는 장(章)."""
    if u.source in PANEL_SIDE:
        return u.source
    return f"{u.source}:{u.chapter}"


def review(version: str) -> list[dict]:
    rules = ruleset(version)
    scope = "document" if version != "T1" else "sentence"
    ctxs: dict[str, dict] = {}
    rows = []
    for u in corpus():
        doc = ctxs.setdefault(_doc(u), {"abbrev_scope": scope, "coined_scope": scope})
        ctx = dict(doc, kind=_kind(u.where), plain=re.sub(r"`[^`]*`", " ", u.text), source=u.source)
        if version != "T1":
            ctx.update(F_CONTEXT)
            if u.source in PANEL_SIDE:
                ctx["abbrev_set"] = F_CONTEXT["abbrev_set_panel"]
        hits = []
        for r in rules:
            for h in r.hits(u.text, ctx):
                hits.append((r.rid, h))
        for k in ("abbrev_seen", "coined_seen"):          # 문서 단위 상태는 문서 맥락에 되돌린다
            if k in ctx:
                doc[k] = ctx[k]
        rids = sorted({h[0] for h in hits}, key=_order)
        rows.append(dict(source=u.source, where=u.where, kind=_kind(u.where), text=u.text,
                         rules=" ".join(rids), matches=" | ".join(f"{a}:{b}" for a, b in hits[:16]),
                         tier=min((RULE_TIER[version][r] for r in rids), default=0)))
    return rows


RULE_TIER = {v: {r.rid: r.tier for r in ruleset(v)} for v in ("T1", "F")}


def _order(rid: str) -> tuple:
    return (rid[0], int(re.sub(r"\D", "", rid) or 0))


def load_open(path: Path, version: str) -> dict[tuple[str, str], set[str]]:
    """개방 코딩 — (위치, 문장) → 범주 집합. 원래 코드를 규칙판의 코드표(`open_category`)로 옮긴다.
    판정 제외(X)는 뺀다."""
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cats = set()
            for c in filter(None, row["codes"].split(" | ")):
                m = re.match(r"([^()]+?)\s*(?:\((.*)\))?$", c)
                code, detail = m.group(1).strip(), m.group(2) or ""
                for d in (detail.split("·") if detail else [""]):
                    cat = open_category(code, d.strip(), version)
                    if cat:
                        cats.add(cat)
            out[(row["where"], row["text"])] = cats - {"X"}
    return out


def compare(version: str, rows: list[dict], open_codes: dict) -> tuple[str, dict]:
    rules = ruleset(version)
    auto = {(r["where"], r["text"]): set(r["rules"].split()) for r in rows}
    stats = {}
    lines = [f"# 개방 코딩 ↔ {version} 대조", "",
             "개방 코딩(규칙 없이 사람이 표시한 범주)을 기준으로, 규칙판의 자동 탐지가 무엇을 잡고 무엇을 놓쳤는지 센다.",
             "", "| 범주 | 개방 코딩 문장 | 자동 탐지 문장 | 둘 다 | 재현율 | 규칙만 | 방식 |", "|---|---:|---:|---:|---:|---:|---|"]
    miss_ex, new_ex = defaultdict(list), defaultdict(list)
    for r in rules:
        man = {k for k, v in open_codes.items() if r.rid in v}
        det = {k for k, v in auto.items() if r.rid in v}
        both = man & det
        rec = len(both) / len(man) if man else float("nan")
        stats[r.rid] = dict(open=len(man), auto=len(det), both=len(both), recall=round(rec, 3) if man else None,
                            auto_only=len(det - man))
        lines.append(f"| {r.rid} {r.name} | {len(man)} | {len(det)} | {len(both)} | "
                     f"{'—' if not man else f'{rec:.0%}'} | {len(det - man)} | {r.mode} |")
        miss_ex[r.rid] = sorted(man - det)[:12]
        new_ex[r.rid] = sorted(det - man)[:8]
    union_open = {k for k, v in open_codes.items() if v}
    union_auto = {k for k, v in auto.items() if v}
    lines += ["", f"- 개방 코딩 후보 {len(union_open)} 문장, 자동 후보 {len(union_auto)} 문장, "
              f"둘 다 {len(union_open & union_auto)}, 개방 코딩만 {len(union_open - union_auto)}, 자동만 {len(union_auto - union_open)}.", ""]
    lines += ["## 규칙이 놓친 문장 (개방 코딩에만 있음, 범주별 최대 12)", ""]
    for rid, ex in miss_ex.items():
        if ex:
            lines.append(f"### {rid}")
            lines += [f"- `{w}` {t[:160]}" for w, t in ex]
            lines.append("")
    lines += ["## 규칙만 잡은 문장 (새 후보, 범주별 최대 8)", ""]
    for rid, ex in new_ex.items():
        if ex:
            lines.append(f"### {rid}")
            lines += [f"- `{w}` {t[:160]}" for w, t in ex]
            lines.append("")
    return "\n".join(lines), stats


def rules_md(version: str) -> str:
    md = [f"# 학술 서술 검토 규칙 — {version}", "",
          "형식: 범주마다 **정의 · 판정 기준(포함 · 제외) · 교정 지침 · 예시 · 판정 방식**. "
          "등급 1 = 뜻 · 독자 이해에 영향, 등급 2 = 표기 · 문체 다듬기.", ""]
    for g, gname in GROUPS.items():
        md += [f"## {g}. {gname}", ""]
        for r in ruleset(version):
            if r.group != g:
                continue
            md += [f"### {r.rid} {r.name}  · 등급 {r.tier} · {r.mode}", "",
                   f"- **정의** — {r.define}", f"- **포함** — {r.include}"]
            if r.exclude:
                md.append(f"- **제외** — {r.exclude}")
            md.append(f"- **교정 지침** — {r.guide}")
            for a, b in r.examples:
                md.append(f"- **예** — {a} → {b}")
            if r.note:
                md.append(f"- **T1 → {version} 변경** — {r.note}")
            md.append("")
    return "\n".join(md)


def write(version: str, rows: list[dict], open_codes: dict | None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    cand = []
    for r in rows:
        oc = open_codes.get((r["where"], r["text"]), set()) if open_codes else set()
        auto = set(r["rules"].split())
        manual_only = {c for c in oc - auto if MODE[version].get(c) in ("수동", "자동+수동")}
        missed = oc - auto - manual_only
        if not (auto or manual_only or missed):
            continue
        cats = sorted(auto | manual_only | missed, key=_order)
        how = "자동" if auto else "개방 코딩"
        if auto and (manual_only or missed):
            how = "자동+개방 코딩"
        cand.append(dict(source=r["source"], where=r["where"], kind=r["kind"] or "본문", text=r["text"],
                         rules=" ".join(cats), auto=r["rules"], open=" ".join(sorted(oc, key=_order)),
                         how=how, matches=r["matches"],
                         tier=min((RULE_TIER[version].get(c, 2) for c in cats), default=2)))
    keys = ["source", "where", "kind", "tier", "rules", "auto", "open", "how", "matches", "text"]
    with open(OUT / f"candidates_{version}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(cand)
    tot_src = Counter(r["source"] for r in rows)
    summ = dict(version=version, units=len(rows), candidates=len(cand),
                by_rule=dict(sorted(Counter(c for x in cand for c in x["rules"].split()).items(), key=lambda kv: _order(kv[0]))),
                by_how=dict(Counter(x["how"] for x in cand)), by_tier=dict(Counter(x["tier"] for x in cand)),
                by_source={s: [sum(1 for x in cand if x["source"] == s), tot_src[s]] for s in tot_src})
    if open_codes:
        md, stats = compare(version, rows, open_codes)
        (OUT / f"compare_{version}.md").write_text(md, encoding="utf-8")
        summ["compare"] = stats
    (OUT / f"summary_{version}.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / f"rules_{version}.md").write_text(rules_md(version), encoding="utf-8")
    return summ


MODE = {v: {r.rid: r.mode for r in ruleset(v)} for v in ("T1", "F")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rules", default="F", choices=("T1", "F"),
                    help="어느 규칙판으로 추출하나 (T1 = 임시 규칙 1, F = 대조를 거친 최종 규칙)")
    ap.add_argument("--open", default=None, metavar="CSV",
                    help="개방 코딩 CSV — 주면 대조표(compare_<판>.md)를 쓰고 후보에 개방 코딩 범주를 함께 싣는다")
    a = ap.parse_args()
    rows = review(a.rules)
    oc = load_open(Path(a.open), a.rules) if a.open else None
    summ = write(a.rules, rows, oc)
    print(json.dumps({k: v for k, v in summ.items() if k != "compare"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
