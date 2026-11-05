#!/usr/bin/env python3
"""대외 글(판넬 · 브로셔 · 판넬 그림 · 논문)의 **학술 서술 검사** — D-43, `docs/95_academic_style.md`.

    python scripts/check_academic_style.py                 # 판넬 · 브로셔 · 판넬 그림의 글
    python scripts/check_academic_style.py --file 초안.txt  # 임의의 원고
    python scripts/check_academic_style.py --lexicon        # 어휘표를 마크다운으로 출력

**이 파일의 `LEXICON` 이 금지 표현의 유일한 원천이다.** 문서(95)는 원칙과 대표 예만
싣고 전체 목록은 `--lexicon` 출력을 가리킨다 — 같은 목록을 두 곳에 적으면 한쪽이
낡는다(O-27).

**무엇을 검사하나** — 한글이 들어간 문자열만 본다. 경로 · 열 이름 · 파일명 같은
코드 문자열은 독자에게 안 보이므로 대상이 아니다. 독스트링과 `print` · `help=` 도
뺀다(개발자용 글이다).

**무엇을 못 잡나** — 정의 없이 처음 등장하는 용어(규칙 C1), 두괄식 구성(D4),
관찰과 해석의 구분(D3)은 기계로 못 가린다. 그것은 `docs/95` 의 점검표로 사람이 본다.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HANGUL = re.compile(r"[가-힣]")

# (정규식, 규칙, 왜 안 되나, 대신 쓸 것)
LEXICON: list[tuple[str, str, str, str]] = [
    # ---- A. 용어: 조어 · 직역 · 내부 식별자 · 은유
    (r"심박 ?모양 ?가짜|가짜 ?(심박|박동|beat)", "A1", "직역 조어",
     "가성 QRS군(spurious QRS complex) · 위양성 검출(false positive)"),
    (r"가짜", "A1", "구어 — 학술어가 아니다", "가성(假性) · 위양성 · 허위"),
    (r"front-?end|프런트엔드|프론트엔드", "A3", "내부 별칭 · 영어 혼용", "공통 전처리"),
    (r"\bM_FE\b|\bM0\d\b|\bM1\d\b|\bB0[12]\b", "A4", "내부 기법 코드", "기법의 이름(예: SWT 임계값 처리)"),
    (r"(?<![A-Za-z])L[1-6](?![0-9A-Za-z])", "A4", "내부 손실 코드", "손실 구성의 서술(예: 무잡음 입력 보존항을 추가한 손실)"),
    (r"(?<![A-Za-z])D[0-3](?![0-9A-Za-z])", "A4", "내부 데이터축 코드", "합성 ECG · MIT-BIH 실측 기록"),
    (r"\b[FDOR]-\d+\b|EXP-[A-Z]", "A4", "내부 기록 · 실험 번호", "삭제하거나 참고문헌 · 부록으로"),
    (r"snr_imp|beat_cc|gain_bias|best\.pt|\.parquet", "A4", "코드 식별자", "지표의 이름과 정의"),
    (r"같은 자\b|자를 (다시 )?세", "A5", "은유", "동일 평가 기준 · 평가 절차 정비"),
    (r"조준|고장|천장|균형추|여정|공짜|몫", "A5", "은유 · 구어", "학습 목표 설정 · 성능 저하 · 상한 · 기여분"),
    (r"(?<![가-힣])판(?![가-힣])", "A5", "구어(「한 판」)", "실험 · 학습 반복"),
    # ---- B. 의인화 · 구어 · 강조 수사
    (r"그린다|그려(?:진|졌|넣|낸)|그리는|그렸", "B1", "의인화(신호를 「그린다」)", "생성한다 · 출력한다"),
    (r"지운다|지우(?:고|는|지|면)|지웠", "B1", "구어", "제거한다"),
    (r"건드리|건드린|건드려", "B1", "의인화", "변형한다 · 왜곡한다"),
    (r"지어내|지어낸", "B1", "의인화", "생성한다(가성 파형)"),
    (r"속는|속지", "B1", "구어", "오인한다"),
    (r"버티|버틴", "B1", "구어", "유지된다 · 재현된다"),
    (r"깨끗한|깨끗하", "B2", "감각어", "무잡음 · 잡음이 제거된"),
    (r"없는 자리|자리에|자리를|자리의", "B2", "구어(「자리」)", "위치 · 구간"),
    (r"몰린|몰려|몰리", "B2", "구어", "집중된다"),
    (r"갈린다|갈렸|가른다|갈라", "B2", "구어", "구분된다 · 달라진다"),
    (r"뒤집", "B2", "구어", "역전된다 · 부호가 바뀐다"),
    (r"손해", "B2", "구어", "성능 저하"),
    (r"(?:^|\s)(?:이긴다|진다|졌다|이겼다)", "B2", "승부 은유", "우수하다 · 열등하다"),
    (r"못 ?(?:잼|잰|가림|가린|가려)", "B2", "구어 판정", "통계적으로 유의한 차이가 관찰되지 않았다(p 값 · n 병기)"),
    (r"(?<![가-힣])값을 (?:한다|못)", "B2", "구어", "효과가 있다 · 없다"),
    (r"우리(?:는|가|의|를)?(?![가-힣])", "D2", "1 인칭", "본 연구"),
    # ---- A6. 출처 없이 직역한 용어 · 일상어 (2026-11-04 사용자 지적, docs/95 §3.1)
    (r"특징 ?지도", "A6", "직역 조어(feature map) — 사용 사례 없음", "특징 맵(feature map) · 채널 수"),
    (r"읽는 ?법", "A6", "일상어", "범례"),
    (r"처방", "A6", "의학 은유 — 선행 연구의 목적이 드러나지 않는다", "기법의 목적 서술(예: 일반화 개선 목적)"),
    (r"알린다|알려 ?준다|알려 ?주", "A6", "일상어 — 조건 입력의 방식이 드러나지 않는다",
     "조건 입력한다(FiLM) · 조건 정보로 사용한다"),
    (r"칼만 ?평활기", "A3", "한 개념 한 용어 — 대외 글은 「칼만 필터」로 고정(사용자 결정 2026-11-04)",
     "칼만 필터 · 확장 칼만 필터"),
    (r"녹음", "A6", "소리에 쓰는 말 — 심전도는 녹음하지 않는다", "기록 · 저장 신호"),

    (r"했다", "D1", "구어체 과거형", "하였다"),
    (r"「|」|«|»", "B3", "강조용 인용부호", "삭제 — 용어 정의 시에만 작은따옴표"),
    (r"—", "B3", "줄표로 잇는 구어 연결", "쌍점 · 쉼표 · 괄호 · 문장 분리"),
    (r"\*\*", "B3", "마크다운 강조", "삭제"),
    (r"이 아니라 .{0,30}(?:이다|다\b)", "B3", "대조 수사(「A 가 아니라 B」)", "B 를 직접 진술"),
]
_COMPILED = [(re.compile(p), r, why, alt) for p, r, why, alt in LEXICON]

# 판넬 · 브로셔에 실리는 글이 있는 곳
PANEL_SCRIPT = ROOT / "scripts" / "build_panel.py"
PANEL_FUNCS = {"build_panel", "build_brochure"}
SLIDES_SCRIPT = ROOT / "scripts" / "make_slides.py"
# 판넬 전용 그림 — 판넬에 실리는 그림의 글은 여기서 나온다. 처음에는 대상에서 빠져 있어서
# 그림 7 의 「특징 지도」가 검사 없이 판넬에 실렸다(2026-11-04).
PANEL_FIGS_SCRIPT = ROOT / "scripts" / "make_panel_figs.py"
PANEL_FIG_FUNCS = {"f4_crossover", "f5_phantom", "f6_noise", "f7_flow", "t1_swt", "t2_kalman", "t3_unet"}
PANEL_FIG_CONSTS = {"FLOW"}
# 브로셔 전용 그림
BROCHURE_FIGS_SCRIPT = ROOT / "scripts" / "make_brochure_figs.py"
BROCHURE_FIG_FUNCS = {"b1_system", "b2_realtime", "b3_phantom"}
BROCHURE_FIG_CONSTS = {"FE_LAB"}
# 판넬에 실리는 그림 — `docs/94_panel_plan.md` §4 의 배치와 같아야 한다
PANEL_FIGS = {"s4_noise", "s5_crossover", "s15_kalman_phantom", "s16_wavelet_placement",
              "s17_dl_journey", "s18_pipeline", "s19_swt", "s20_kalman_model", "s21_resunet"}
PANEL_CONSTS = {"NAME_KO", "TXT_KO", "JOURNEY", "GROUP_C", "MODE_KO", "COND_KO"}


def check_text(text: str, where: str = "") -> list[str]:
    out = []
    for rx, rule, why, alt in _COMPILED:
        for m in rx.finditer(text):
            out.append(f"{where}[{rule}] «{m.group(0)}» — {why} → {alt}")
    return out


def _strings(node: ast.AST) -> list[tuple[int, str]]:
    """노드 안의 한글 문자열 — 독스트링 · print · help= 는 뺀다."""
    skip: set[int] = set()
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.Module)) and n.body and \
                isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant):
            skip.add(id(n.body[0].value))
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name in ("print", "add_argument", "ArgumentParser"):
                for m in ast.walk(n):
                    skip.add(id(m))
            for kw in n.keywords:
                if kw.arg == "help":
                    for m in ast.walk(kw.value):
                        skip.add(id(m))
    return [(n.lineno, n.value) for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in skip and HANGUL.search(n.value)]


def targets() -> list[tuple[str, int, str]]:
    """(파일, 줄, 문자열) — 판넬 · 브로셔 · 판넬 그림에 실리는 글."""
    out = []
    for path, funcs, consts in ((PANEL_SCRIPT, PANEL_FUNCS, {"STEPS"}),
                                (PANEL_FIGS_SCRIPT, PANEL_FIG_FUNCS, PANEL_FIG_CONSTS),
                                (BROCHURE_FIGS_SCRIPT, BROCHURE_FIG_FUNCS, BROCHURE_FIG_CONSTS),
                                (SLIDES_SCRIPT, PANEL_FIGS, PANEL_CONSTS)):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in funcs:
                out += [(path.name, ln, s) for ln, s in _strings(node)]
            elif isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id in consts for t in node.targets):
                out += [(path.name, ln, s) for ln, s in _strings(node)]
    return out


def check_targets() -> list[str]:
    return [v for f, ln, s in targets() for v in check_text(s, f"{f}:{ln} ")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--file", type=Path, default=None,
                    help="이 원고 파일 하나만 검사한다 (UTF-8 텍스트 · 마크다운)")
    ap.add_argument("--lexicon", action="store_true",
                    help="금지 표현 어휘표 전체를 마크다운 표로 출력하고 끝낸다")
    a = ap.parse_args()
    if a.lexicon:
        print("| 규칙 | 걸리는 표현(정규식) | 왜 | 대신 |\n|---|---|---|---|")
        for p, r, why, alt in LEXICON:
            print(f"| {r} | `{p}` | {why} | {alt} |")
        return 0
    if a.file:
        v = []
        for i, line in enumerate(a.file.read_text(encoding="utf-8").splitlines(), 1):
            v += check_text(line, f"{a.file.name}:{i} ")
    else:
        v = check_targets()
    for x in v:
        print(x)
    print(f"[academic] 위반 {len(v)} 건")
    return 1 if v else 0


if __name__ == "__main__":
    sys.exit(main())
