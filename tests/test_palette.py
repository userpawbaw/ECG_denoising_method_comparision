"""색 검증기 — **문서가 적은 값을 검사가 붙들어 둔다**.

`docs/33_card_design_samples.md` 는 색 검증 결과를 숫자로 적어 두었는데, 그것을
잰 도구가 저장소에 없어 **몇 달간 아무도 다시 재지 못했다**(UD-1). 도구를 다시
만들었으니, 이번에는 **문서의 숫자와 코드를 검사로 묶는다.**

여기 있는 기대값은 전부 `dataviz` 스킬의 `validate_palette.js` 와 **다섯 팔레트에서
전부 일치**하는 것을 확인한 값이다. 값이 바뀌면 둘 중 하나다 — 구현이 깨졌거나,
문턱값을 바꿨는데 D 항목을 안 적었거나.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_palette import (  # noqa: E402
    CVD_FLOOR, NORMAL_FLOOR, _delta_e, _linear_rgb, _oklch, _contrast, validate,
)

# 팔레트 값은 **원본에서 읽는다** — 여기 적어 두면 색을 바꿀 때 검사가 안 따라온다
# (CLAUDE.md 「수치 옮기기: 로그가 아니라 파일에서 읽는다」와 같은 이유).
# 원본은 `ui/palette.json` 하나이고 `scripts/build_tokens.py` 가 CSS 와 Python 을 만든다.
PALETTE_SRC = ROOT / "ui" / "palette.json"


def _palette() -> dict:
    return json.loads(PALETTE_SRC.read_text(encoding="utf-8"))


def _method_colors() -> dict[str, str]:
    return {k: v["light"] for k, v in _palette()["methods"].items() if k != "_"}


# **그림이 실제로 쓰는 조합.** 팔레트 전체를 한 번에 재는 것은 과잉이었다 — 7 색이
# 한 그림에 함께 나오는 일이 없고, 그렇게 재면 쓰이지 않는 조합까지 실패로 센다
# (UF-1). 출처는 `scripts/make_slides.py` 의 `show = [...]` 세 줄이다.
COMBOS = {
    "겹쳐 그림 + 범례": ["M01", "M04", "M08"],
    "facet 잡음 비교": ["M_FE", "M04", "M08"],
    "막대 5 색": ["M_FE", "M01", "M04", "M05", "M08"],
}
# 카드가 한 그림에 놓는 조합 (`scripts/build_metric_cards.py`)
CARD_COMBOS = {
    "C3 스펙트럼": ["M02", "M04"],
    "C4 PSD": ["M01", "M04"],
    "이득 보정": ["M06", "M08"],
}
CARD3 = ["#2a78d6", "#eb6834", "#1baf7a"]        # docs/33 카드 3 안


def _worst_normal(colors: list[str]) -> float:
    lin = [_linear_rgb(c) for c in colors]
    return min(_delta_e(lin[i], lin[j])
               for i in range(len(colors)) for j in range(i + 1, len(colors)))


# ------------------------------------------------- 문서가 적은 값의 재현
def test_docs33_records_the_mfe_m04_failure():
    """`docs/33` 의 「`M_FE` magenta ↔ `M04` orange 가 ΔE 12.9 로 15 미만 FAIL」.

    이 한 줄이 **검증기가 사라지기 전 마지막으로 남은 측정**이었다. 새 구현이
    같은 값을 내야 「같은 자」라고 말할 수 있다. **당시 값**으로 고정한다 —
    `M_FE` 는 그 뒤 UD-2 로 옮겼으므로 현재 팔레트에서는 이 쌍이 안 나온다.
    """
    d = _delta_e(_linear_rgb("#e87ba4"), _linear_rgb("#eb6834"))
    assert round(d, 1) == 12.9, f"docs/33 이 적은 12.9 와 다르다: {d:.1f}"
    assert d < NORMAL_FLOOR


def test_docs33_claim_card_three_colors_pass():
    """「슬롯 앞 세 개만 쓴다」는 `docs/33` 의 결론이 실제로 통과한다."""
    r = validate(CARD3, "light")
    assert r["ok"], r["checks"]
    assert r["worst_cvd"] == 9.2
    assert r["worst_normal"] == 24.0


# ------------------------------------------------- 실제 조합이 전부 통과하는가
@pytest.mark.parametrize("label", sorted(COMBOS))
def test_every_slide_combination_passes(label):
    """**슬라이드가 한 그림에 함께 그리는 색들**이 검사를 통과해야 한다 (UD-2).

    색을 바꾸면 이 검사가 값을 코드에서 다시 읽으므로 자동으로 따라온다.
    """
    pal = _method_colors()
    cols = [pal[m] for m in COMBOS[label]]
    r = validate(cols, "light")
    assert r["ok"], f"{label}: {[c for c in r['checks'] if c['status'] == 'FAIL']}"


@pytest.mark.parametrize("label", sorted(CARD_COMBOS))
def test_every_card_combination_passes(label):
    """지표 카드가 한 그림에 놓는 조합도 같은 자로 잰다."""
    pal = _method_colors()
    cols = [pal[m] for m in CARD_COMBOS[label]]
    r = validate(cols, "light")
    assert r["ok"], f"{label}: {[c for c in r['checks'] if c['status'] == 'FAIL']}"


def test_scripts_read_the_palette_instead_of_defining_one():
    """두 산출물이 **같은 방법에 같은 색**을 쓰는 유일한 방법은 원본을 함께 읽는 것이다.

    예전에는 각자 `C = {...}` 를 들고 있어서 같은 방법이 보고서와 카드에서 다른
    색이었다 (U-2). 역할 색도 갈려 있었다 — `INK` 가 `#1b1b1b` / `#0b0b0b`.

    **같은 hex 를 다른 축에 재사용하는 것은 여기서 잡지 않는다.** 슬롯 색을
    front-end 모드나 측정 함정 종류에 다시 쓰는 그림들이 있고, 그 그림에는 방법이
    등장하지 않아 충돌하지 않는다. 슬롯을 이름 있는 축으로 올릴지는 **[미정]**
    (`ui/palette.json` 의 `methods._`).
    """
    for name in ("make_slides.py", "build_metric_cards.py"):
        src = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "from _palette import" in src, f"{name} 이 팔레트 원본을 안 읽는다"
        code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
        assert "\nC = {" not in code, (
            f"{name} 이 방법 색 딕셔너리를 다시 정의한다 — ui/palette.json 에서 읽을 것")


def test_generated_tokens_are_up_to_date():
    """`ui/palette.json` 을 고치고 **생성을 잊으면** 이 검사가 잡는다.

    생성물을 git 에 넣는 이유는 시연 화면이 `file://` 로 열려야 해서 빌드 단계를
    전제할 수 없기 때문이다 (UD-1 3.1). 그 대가가 「원본과 생성물이 어긋날 수
    있다」이고, 이것이 그 대가를 갚는 장치다.
    """
    r = subprocess.run([sys.executable, str(ROOT / "scripts/build_tokens.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_generated_python_matches_the_source():
    """생성된 모듈의 방법 색이 원본과 같아야 한다 (생성기 자체의 검사)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    import _palette
    importlib.reload(_palette)
    assert _palette.METHODS == _method_colors()
    assert _palette.CLEAN == _palette_roles()["clean"]


def _palette_roles() -> dict[str, str]:
    return {k: v["light"] for k, v in _palette()["roles"].items() if k != "_"}


def test_the_two_moved_colors_stayed_close_to_their_originals():
    """UD-2 의 요점은 **색 정체성을 지키면서** 통과시킨 것이다.

    분홍이 보라가 되거나 노랑이 갈색이 되면(후보 탐색에서 실제로 그런 답이 먼저
    나왔다 — ΔE 21.6) 보고서 그림의 인상이 바뀐다.
    """
    pal = _method_colors()
    for method, was in (("M_FE", "#e87ba4"), ("M05", "#eda100")):
        moved = _delta_e(_linear_rgb(pal[method]), _linear_rgb(was))
        assert moved < 5.0, f"{method} 가 원래 색에서 ΔE {moved:.1f} 나 움직였다"


# ------------------------------------------------- scope 두 자 (docs/37 2.1)
def test_legend_scope_is_looser_than_lane_scope():
    """레인은 모든 쌍, 범례는 인접 쌍만 — 그래서 범례 쪽이 더 너그럽거나 같다."""
    seven = list(_method_colors().values())
    lane = validate(seven, "light", scope="lane")
    legend = validate(seven, "light", scope="legend")
    assert legend["worst_cvd"] >= lane["worst_cvd"]
    assert legend["worst_normal"] >= lane["worst_normal"]


def test_normal_floor_applies_in_both_scopes():
    """정상시야 바닥은 **직접 라벨로도 면제되지 않는다** — 두 scope 모두에 건다."""
    pair = ["#e87ba4", "#eb6834"]            # 12.9
    for scope in ("lane", "legend"):
        r = validate(pair, "light", scope=scope)
        assert not r["ok"], scope


# ------------------------------------------------- 변환 자체
def test_color_conversions_hit_known_anchors():
    assert round(_oklch("#ffffff")[0], 3) == 1.0
    assert round(_oklch("#000000")[0], 3) == 0.0
    assert round(_oklch("#808080")[1], 3) == 0.0          # 무채색은 채도 0
    assert round(_contrast("#ffffff", "#000000"), 1) == 21.0


def test_cvd_simulation_is_clamped_to_gamut():
    """시뮬레이션 결과를 자르지 않으면 **표시할 수 없는 색끼리의 거리**를 재게 된다.

    `tritan` 행렬은 계수가 커서 색역을 자주 벗어난다 — 이 검사가 없을 때 실제로
    tritan 만 참조 구현과 값이 어긋났다.
    """
    from validate_palette import _simulate
    for kind in ("protan", "deutan", "tritan"):
        for hex_str in ("#ffffff", "#00ff00", "#ff00ff"):
            assert all(0.0 <= v <= 1.0
                       for v in _simulate(_linear_rgb(hex_str), kind))
    # 자르지 않았을 때 어긋났던 그 값
    assert round(min(_delta_e(_linear_rgb(a), _linear_rgb(b), "tritan")
                     for a, b in [("#2a78d6", "#eb6834"), ("#2a78d6", "#1baf7a"),
                                  ("#eb6834", "#1baf7a")]), 1) == 9.6


# ------------------------------------------------- 입력 경계
@pytest.mark.parametrize("bad", [
    (["#2a78d6"], "light", "두 색 이상"),
    (["#2a78d6", "nothex"], "light", "hex"),
    (["#2a78d6", "#eb6834"], "twilight", "mode"),
])
def test_bad_input_raises_rather_than_passing_quietly(bad):
    colors, mode, _ = bad
    with pytest.raises(ValueError):
        validate(colors, mode)


def test_unknown_scope_raises():
    with pytest.raises(ValueError):
        validate(CARD3, "light", scope="dashboard")


# ------------------------------------------------- CLI
def test_cli_exit_code_follows_the_verdict():
    """통과면 0, FAIL 이면 1 — 검사 파이프라인에 그대로 걸 수 있어야 한다."""
    ok = subprocess.run([sys.executable, str(ROOT / "scripts/validate_palette.py"),
                         ",".join(CARD3), "--mode", "light"],
                        capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr

    bad = subprocess.run([sys.executable, str(ROOT / "scripts/validate_palette.py"),
                          "#e87ba4,#eb6834", "--mode", "light"],
                         capture_output=True, text=True)
    assert bad.returncode == 1
    assert "정상시야" in bad.stdout


def test_cli_json_mode_is_machine_readable():
    import json
    r = subprocess.run([sys.executable, str(ROOT / "scripts/validate_palette.py"),
                        ",".join(CARD3), "--mode", "light", "--json"],
                       capture_output=True, text=True)
    payload = json.loads(r.stdout)
    assert payload["ok"] is True
    assert payload["worst_normal"] == 24.0
    assert len(payload["checks"]) == 5
