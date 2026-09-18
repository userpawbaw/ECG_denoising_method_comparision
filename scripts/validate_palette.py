#!/usr/bin/env python3
"""색 팔레트 검증 — **눈으로 고르지 않고 잰다**.

## 왜 이 파일이 저장소 안에 있나

`docs/33_card_design_samples.md` 는 「`--pairs all` 로 검증했고」
「`validate_palette.js --mode light` 전 항목 PASS」라고 적어 두었는데, **그 도구가
저장소에 없었다.** 다른 세션의 것이었고 사라졌다. 그래서 팔레트를 손대려는 순간
**같은 자로 다시 잴 수가 없었다** — `docs/19_record_keeping.md` 6 절(측정 근거는
재현 가능하게) 위반이다. UD-1 이 이것을 1 단계 첫 항목으로 올린 이유다.

**도구가 또 사라져도 자는 남아야 하므로** 기준값과 알고리즘을 여기 함께 적는다.
전부 공개 표준이다 — OKLab(Ottosson), CVD 시뮬레이션(Machado·Oliveira·Fernandes
2009, severity 1.0), WCAG 2 상대휘도.

## 무엇을 재나

| 검사 | 무엇 | 문턱 |
|---|---|---|
| 명도대 | OKLCH `L` 이 모드의 띠 안에 | light 0.43~0.77 · dark 0.48~0.67 |
| 채도 바닥 | OKLCH `C` | >= 0.10 (아래로 가면 색상이 회색으로 읽힌다) |
| CVD 분리 | OKLab ΔE×100, `min(protan, deutan)` | 목표 8 · 바닥 6 (6~8 은 **직접 라벨 등 보조 부호가 있을 때만**) |
| 정상시야 바닥 | 시뮬레이션 없는 ΔE×100, 최악 쌍 | **15 — 이건 보조 부호로도 면제되지 않는다** |
| 표면 대비 | WCAG 대비비 | 3:1 (미만이면 보이는 라벨이나 표 뷰가 의무) |

## 이 저장소에만 있는 것 — `--scope`

`docs/ui/01_system.md` 2.1 이 정한 두 자다. 우리 화면은 **방법마다 레인이 따로**라
(`demo/cards/card_core.js` 의 `drawLane()`) 한 레인 안에 동시에 보이는 색은
[출력 1 + 참값 회색] 둘뿐이고, 여러 색이 나란히 오는 곳은 **범례뿐**이다.

- `--scope lane` (기본): 한 화면에서 서로 섞일 수 있는 색들 — **엄격**하게 all-pairs
- `--scope legend`: 범례처럼 **직접 라벨이 붙는** 자리 — 인접 쌍만 보되 정상시야
  바닥은 그대로 적용한다 (색만으로 신원을 알리지 않으므로 CVD 는 완화, 그러나
  「색이 아예 같아 보이는 것」은 라벨이 있어도 문제다)

사용:

    python3 scripts/validate_palette.py "#2a78d6,#eb6834,#1baf7a" --mode light
    python3 scripts/validate_palette.py "#e0705f,#7bb0f0" --mode dark --scope legend
    python3 scripts/validate_palette.py "#..." --json

FAIL 이 하나라도 있으면 종료 코드 1. WARN 은 0 이다 — 다만 WARN 은 **보조 부호를
의무로 만든다**(직접 라벨·간격·질감). 면제가 아니다.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys

# ---------------------------------------------------------------- 문턱값
# 여기 숫자를 바꾸면 과거 기록과 비교가 끊긴다. 바꿀 일이 생기면 D 항목을 먼저 적는다.
BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}   # OKLCH L
CHROMA_FLOOR = 0.10                                     # OKLCH C
CVD_TARGET, CVD_FLOOR = 8.0, 6.0                        # OKLab ΔE×100
NORMAL_FLOOR = 15.0                                     # 시뮬레이션 없는 ΔE×100
CONTRAST_MIN = 3.0                                      # WCAG vs 표면
DEFAULT_SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}

# Machado, Oliveira & Fernandes (2009), severity 1.0, **선형 RGB 에 적용한다**.
# 시뮬레이션 모델이 문턱값 교정의 일부다 — 다른 모델(Viénot 1999 등)로 바꾸면
# 경계 쌍이 움직이므로 위 문턱을 다시 교정해야 한다.
MACHADO = {
    "protan": ((0.152286, 1.052583, -0.204868),
               (0.114503, 0.786281, 0.099216),
               (-0.003882, -0.048116, 1.051998)),
    "deutan": ((0.367322, 0.860646, -0.227968),
               (0.280085, 0.672501, 0.047413),
               (-0.011820, 0.042940, 0.968881)),
    "tritan": ((1.255528, -0.076749, -0.178779),
               (-0.078411, 0.930809, 0.147602),
               (0.004733, 0.691367, 0.303900)),
}

# 붙여 넣은 hex 목록에는 눈에 안 보이는 공백이 섞여 온다(NBSP·em space).
_WS = (" \t\n\v\f\r         "
       "        　")
_HEX = re.compile(r"#?[0-9a-fA-F]{6}\Z")


# ---------------------------------------------------------------- 색 변환
def _srgb(hex_str: str) -> tuple[float, float, float]:
    h = hex_str.strip(_WS).lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_rgb(hex_str: str) -> tuple[float, float, float]:
    return tuple(_to_linear(c) for c in _srgb(hex_str))


def _oklab(lin: tuple[float, float, float]) -> tuple[float, float, float]:
    """선형 RGB → OKLab (Ottosson). 세제곱근 앞의 음수는 부호를 살려 다룬다."""
    r, g, b = lin
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = (math.copysign(abs(v) ** (1 / 3), v) for v in (l, m, s))
    return (0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_)


def _oklch(hex_str: str) -> tuple[float, float]:
    """(L, C) 만 돌려준다 — 색상각은 이 검사들이 안 쓴다."""
    L, a, b = _oklab(_linear_rgb(hex_str))
    return L, math.hypot(a, b)


def _simulate(lin: tuple[float, float, float], kind: str) -> tuple[float, float, float]:
    """Machado 행렬을 선형 RGB 에 적용하고 **[0,1] 로 자른다**.

    자르는 것이 중요하다 — `tritan` 행렬은 계수가 커서(1.256 · 0.691) 결과가 색역을
    자주 벗어나고, 자르지 않으면 표시할 수 없는 색끼리의 거리를 재게 된다. 실제로
    이 한 줄이 없을 때 tritan 만 스킬 구현과 값이 어긋났다(protan·deutan 은 계수가
    작아 그대로 일치했다).
    """
    m = MACHADO[kind]
    return tuple(max(0.0, min(1.0, sum(m[i][j] * lin[j] for j in range(3))))
                 for i in range(3))


def _delta_e(lin_a, lin_b, kind: str | None = None) -> float:
    """OKLab 유클리드 거리 ×100. `kind` 가 있으면 그 색각으로 시뮬레이션한 뒤 잰다."""
    if kind:
        lin_a, lin_b = _simulate(lin_a, kind), _simulate(lin_b, kind)
    pa, pb = _oklab(lin_a), _oklab(lin_b)
    return 100 * math.dist(pa, pb)


def _luminance(hex_str: str) -> float:
    r, g, b = _linear_rgb(hex_str)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(hex_a: str, hex_b: str) -> float:
    la, lb = _luminance(hex_a), _luminance(hex_b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


# ---------------------------------------------------------------- 검사
def _pairs(n: int, scope: str):
    """`lane` 은 모든 쌍, `legend` 는 인접 쌍만 — docs/37 2.1."""
    if scope == "legend":
        return [(i, i + 1) for i in range(n - 1)]
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def validate(colors: list[str], mode: str = "light", surface: str | None = None,
             scope: str = "lane") -> dict:
    """검사 다섯을 돌려 판정 딕셔너리를 만든다. 표시는 하지 않는다."""
    bad = [c for c in colors if not _HEX.match(c.strip(_WS))]
    if bad:
        raise ValueError(f"hex 가 아닌 값: {bad}")
    if mode not in BAND:
        raise ValueError(f"mode 는 light|dark: {mode}")
    if scope not in ("lane", "legend"):
        raise ValueError(f"scope 는 lane|legend: {scope}")
    if len(colors) < 2:
        raise ValueError("두 색 이상이어야 쌍을 잰다")

    surface = surface or DEFAULT_SURFACE[mode]
    lo, hi = BAND[mode]
    lin = [_linear_rgb(c) for c in colors]
    checks: list[dict] = []

    out_of_band = [(c, round(_oklch(c)[0], 3)) for c in colors
                   if not (lo <= _oklch(c)[0] <= hi)]
    checks.append({"name": "명도대", "status": "FAIL" if out_of_band else "PASS",
                   "detail": out_of_band or f"{len(colors)} 개 전부 L {lo}~{hi} 안"})

    low_chroma = [(c, round(_oklch(c)[1], 3)) for c in colors
                  if _oklch(c)[1] < CHROMA_FLOOR]
    checks.append({"name": "채도 바닥", "status": "FAIL" if low_chroma else "PASS",
                   "detail": low_chroma or f"{len(colors)} 개 전부 >= {CHROMA_FLOOR}"})

    pairs = _pairs(len(colors), scope)
    worst_cvd, worst_normal = None, None
    # tritan 은 protan·deutan 과 **다른 쌍**에서 최악이 나오므로 따로 센다.
    worst_tritan = min(_delta_e(lin[i], lin[j], "tritan") for i, j in pairs)
    for i, j in pairs:
        cvd = min(_delta_e(lin[i], lin[j], "protan"),
                  _delta_e(lin[i], lin[j], "deutan"))
        if worst_cvd is None or cvd < worst_cvd[0]:
            worst_cvd = (cvd, colors[i], colors[j], worst_tritan)
        normal = _delta_e(lin[i], lin[j])
        if worst_normal is None or normal < worst_normal[0]:
            worst_normal = (normal, colors[i], colors[j])

    cvd_status = ("PASS" if worst_cvd[0] >= CVD_TARGET
                  else "WARN" if worst_cvd[0] >= CVD_FLOOR else "FAIL")
    checks.append({"name": "CVD 분리", "status": cvd_status,
                   "detail": f"최악 {worst_cvd[1]}↔{worst_cvd[2]} "
                             f"ΔE {worst_cvd[0]:.1f} · tritan {worst_cvd[3]:.1f}"})

    normal_status = "PASS" if worst_normal[0] >= NORMAL_FLOOR else "FAIL"
    checks.append({"name": "정상시야 바닥", "status": normal_status,
                   "detail": f"최악 {worst_normal[1]}↔{worst_normal[2]} "
                             f"ΔE {worst_normal[0]:.1f}"
                             + ("" if normal_status == "PASS"
                                else f" — {NORMAL_FLOOR} 미만, 색각이 정상이어도 구분이 어렵다")})

    weak = [(c, round(_contrast(c, surface), 2)) for c in colors
            if _contrast(c, surface) < CONTRAST_MIN]
    checks.append({"name": "표면 대비", "status": "WARN" if weak else "PASS",
                   "detail": weak or f"{len(colors)} 개 전부 >= {CONTRAST_MIN}:1"})

    failed = [c for c in checks if c["status"] == "FAIL"]
    return {"mode": mode, "surface": surface, "scope": scope,
            "colors": colors, "checks": checks, "ok": not failed,
            "worst_cvd": round(worst_cvd[0], 1),
            "worst_normal": round(worst_normal[0], 1)}


def render(result: dict) -> str:
    head = (f"팔레트 {len(result['colors'])} 색 · {result['mode']} "
            f"· 표면 {result['surface']} · scope {result['scope']}")
    rows = [f"  [{c['status']:4}] {c['name']:<8} {c['detail']}"
            for c in result["checks"]]
    tail = ("  → 전 항목 통과" if result["ok"] else "  → FAIL — 표시된 검사를 고칠 것")
    note = ("     (WARN 은 면제가 아니라 **보조 부호 의무**다 — 직접 라벨 · 간격 · 질감)"
            if any(c["status"] == "WARN" for c in result["checks"]) else "")
    return "\n".join([head, *rows, tail, note]).rstrip() + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="색 팔레트 검증 (docs/ui/01_system.md · UD-1)")
    ap.add_argument("colors", help="쉼표로 나눈 hex 목록")
    ap.add_argument("--mode", default="light", choices=sorted(BAND),
                    help="검사 기준이 되는 표면 모드. 명도대·기본 표면색이 모드마다 다르다 (--surface 로 표면만 따로 줄 수 있다)")
    ap.add_argument("--surface", default=None, help="차트 표면색 (기본은 모드별)")
    ap.add_argument("--scope", default="lane", choices=("lane", "legend"),
                    help="lane=모든 쌍(엄격) · legend=인접 쌍 (docs/ui 01_system 2.1)")
    ap.add_argument("--json", action="store_true",
                    help="사람이 읽는 표 대신 JSON 으로 낸다 — 검사·스크립트가 쓴다 (tests/test_palette.py)")
    a = ap.parse_args(argv)

    cols = [c for c in (s.strip(_WS) for s in a.colors.split(",")) if c]
    try:
        res = validate(cols, a.mode, a.surface, a.scope)
    except ValueError as e:
        print(f"입력 오류: {e}", file=sys.stderr)
        return 2
    print(json.dumps(res, ensure_ascii=False, indent=2) if a.json else render(res),
          end="" if not a.json else "\n")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
