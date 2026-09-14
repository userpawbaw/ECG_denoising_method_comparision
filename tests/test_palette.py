"""색 검증기 — **문서가 적은 값을 검사가 붙들어 둔다**.

`docs/33_card_design_samples.md` 는 색 검증 결과를 숫자로 적어 두었는데, 그것을
잰 도구가 저장소에 없어 **몇 달간 아무도 다시 재지 못했다**(D-29). 도구를 다시
만들었으니, 이번에는 **문서의 숫자와 코드를 검사로 묶는다.**

여기 있는 기대값은 전부 `dataviz` 스킬의 `validate_palette.js` 와 **다섯 팔레트에서
전부 일치**하는 것을 확인한 값이다. 값이 바뀌면 둘 중 하나다 — 구현이 깨졌거나,
문턱값을 바꿨는데 D 항목을 안 적었거나.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_palette import (  # noqa: E402
    CVD_FLOOR, NORMAL_FLOOR, _delta_e, _linear_rgb, _oklch, _contrast, validate,
)

# 이 저장소에서 실제로 쓰이는 팔레트들 (docs/37 2 절의 표와 같다)
CARD3 = ["#2a78d6", "#eb6834", "#1baf7a"]                      # docs/33 카드 3 안
REPORT7 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#7b53c1", "#184f95"]                    # make_slides · build_metric_cards
EXPO4 = ["#b0453a", "#c07a1f", "#184f95", "#2f7d4f"]           # demo/mockup_expo.html
EXPO4_DARK = ["#e0705f", "#e0a445", "#7bb0f0", "#5fbe86"]      # 같은 파일 다크 스텝


def _worst_normal(colors: list[str]) -> float:
    lin = [_linear_rgb(c) for c in colors]
    return min(_delta_e(lin[i], lin[j])
               for i in range(len(colors)) for j in range(i + 1, len(colors)))


# ------------------------------------------------- 문서가 적은 값의 재현
def test_docs33_records_the_mfe_m04_failure():
    """`docs/33` 의 「`M_FE` magenta ↔ `M04` orange 가 ΔE 12.9 로 15 미만 FAIL」.

    이 한 줄이 **검증기가 사라지기 전 마지막으로 남은 측정**이었다. 새 구현이
    같은 값을 내야 「같은 자」라고 말할 수 있다.
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


# ------------------------------------------------- 현 상태의 실패를 기록으로 고정
# D-29: 「기존 값을 먼저 그대로 담고 **현 상태의 실패를 먼저 기록한다**」.
# 고치기 전에 무엇이 왜 실패했는지가 남아 있어야 나중에 비교가 된다.
@pytest.mark.parametrize("colors,mode,worst_pair,worst_normal", [
    (REPORT7, "light", {"#2a78d6", "#7b53c1"}, 12.3),      # M01 파랑 ↔ M02 보라
    (EXPO4, "light", {"#b0453a", "#c07a1f"}, 14.0),        # 입력 적갈 ↔ 고전 황토
    (EXPO4_DARK, "dark", {"#e0705f", "#e0a445"}, 13.7),
])
def test_current_palettes_fail_the_normal_vision_floor(colors, mode, worst_pair,
                                                       worst_normal):
    """**지금 쓰는 팔레트 셋은 정상시야 바닥을 못 넘는다.**

    `docs/33` 이 검증한 것은 카드 3 색뿐이었고, 보고서 그림과 시연 시안은 검증을
    통과한 적이 없다. 고치면 이 검사가 먼저 깨지므로, 그때 이 표를 함께 고친다.
    """
    r = validate(colors, mode)
    assert not r["ok"]
    assert r["worst_normal"] == worst_normal
    names = [c for c in r["checks"] if c["name"] == "정상시야 바닥"][0]["detail"]
    assert all(c in names for c in worst_pair), names


def test_expo_dark_steps_leave_the_lightness_band():
    """시안의 다크 스텝은 **명도대를 넷 다 벗어난다** — 자동 반전의 전형적 결과다.

    `docs/33` 은 카드의 다크 열을 「따로 고른 값」이라 적었는데, 그것은 카드 3 색의
    이야기이고 `mockup_expo` 의 다크 4 색은 다른 값이다.
    """
    r = validate(EXPO4_DARK, "dark")
    band = [c for c in r["checks"] if c["name"] == "명도대"][0]
    assert band["status"] == "FAIL"
    assert len(band["detail"]) == 4, band["detail"]


# ------------------------------------------------- scope 두 자 (docs/37 2.1)
def test_legend_scope_is_looser_than_lane_scope():
    """레인은 모든 쌍, 범례는 인접 쌍만 — 그래서 범례 쪽이 더 너그럽거나 같다."""
    lane = validate(REPORT7, "light", scope="lane")
    legend = validate(REPORT7, "light", scope="legend")
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
                          ",".join(REPORT7), "--mode", "light"],
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
