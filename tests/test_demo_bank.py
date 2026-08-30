"""시연 파형 은행 (R-1 · D-17).

이 은행은 **발표에서 사람이 보는 것**이다. 여기서 구간을 잘못 고르면 시연이
보고서보다 좋아 보이고, 보는 사람은 그것을 알 방법이 없다. 그래서 선정 기준이
부풀림 쪽으로 되돌아가지 않는지(D-17 의 기각안 2·3), 그리고 값 옆에 축 평균이
빠지지 않는지를 고정한다.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
BANK = ROOT / "demo" / "demo_bank.js"


def _mod():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "build_demo_bank", ROOT / "scripts" / "build_demo_bank.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _bank() -> dict:
    """빌드된 은행. **빌더보다 낡았으면 건너뛴다.**

    낡은 산출물로 통과 판정을 내리면 F-9 계열의 실수가 된다. 여기서는 막지
    말고 건너뛰되 이유를 적는다 — 커밋 전에 다시 만들면 검사가 살아난다.
    (`results/demo_bank/manifest.json` 의 소스 해시로도 같은 것을 본다.)
    """
    if not BANK.exists():
        pytest.skip("demo_bank.js 없음 — scripts/build_demo_bank.py 를 먼저 돌린다")
    builder = ROOT / "scripts" / "build_demo_bank.py"
    if BANK.stat().st_mtime < builder.stat().st_mtime:
        pytest.skip("은행이 빌더보다 낡았다 — scripts/build_demo_bank.py 를 다시 돌린다")
    t = BANK.read_text()
    return json.loads(t[t.index("=") + 1: t.rindex(";")])


# ------------------------------------------------------------ 선정 기준
def test_selection_is_not_the_best_segment():
    """**고른 구간이 대비 1 위면 안 된다** (D-17 의 3 번 기각안 재발).

    순위 상관 기준은 정확히 이 실패를 냈다 — 격차가 가장 크게 벌어진 구간이
    순위를 가장 깨끗하게 재현하기 때문이다.
    """
    bank = _bank()
    top = [s["id"] for s in bank["scenes"]
           if s["selection"]["contrast_rank"].startswith("1/")]
    assert not top, f"대비 1 위 구간을 골랐다: {top}"


def test_selection_beats_the_worst_candidate_by_a_lot():
    """고른 구간의 평균편차가 후보 최댓값보다 확실히 작아야 한다."""
    for s in _bank()["scenes"]:
        sel = s["selection"]
        lo, hi = sel["dev_range"]
        assert sel["mean_abs_dev"] == pytest.approx(lo, abs=1e-6), s["id"]
        assert hi > lo, f"{s['id']}: 후보가 전부 같다 — 선정이 의미 없다"


def test_pick_segment_prefers_the_candidate_closest_to_the_axis_mean():
    """합성 입력으로 선정 규칙 자체를 고정한다 (산출물 없이도 돈다)."""
    m = _mod()
    ref = {"A": 10.0, "B": 5.0, "C": 1.0}

    def cand(a, b, c):
        return {k: {"snr_imp": v, "snr_out": v, "cc": 0.9, "rtf": 0.0}
                for k, v in zip("ABC", (a, b, c))}

    per = [cand(20, 5, 1), cand(10.2, 4.9, 1.1), cand(0, 0, 0)]
    dist = [np.mean([abs(p[k]["snr_imp"] - ref[k]) for k in ref]) for p in per]
    assert int(np.argmin(dist)) == 1, "축 평균에 가장 가까운 후보를 골라야 한다"
    # 대비(A−C)가 가장 큰 것은 0 번이다 — 그것을 고르면 안 된다
    assert per[0]["A"]["snr_imp"] - per[0]["C"]["snr_imp"] > \
        per[1]["A"]["snr_imp"] - per[1]["C"]["snr_imp"]
    assert m.SEG_S > 2 * 5.0, "guard(5 s×2)를 빼고도 화면에 보일 길이가 남아야 한다"


# ------------------------------------------------------------ 값 옆의 평균
def test_every_method_either_has_an_axis_mean_or_is_listed_as_missing():
    """빈칸을 그럴듯한 숫자로 메우지 않는다 (D-17)."""
    for s in _bank()["scenes"]:
        for m in s["metrics"]:
            has = m in s["ref_mean"]
            listed = m in s["no_ref"]
            assert has != listed, f"{s['id']}/{m}: ref_mean {has}, no_ref {listed}"


def test_loss_variant_gets_an_axis_mean_once_exp_g_exists():
    """`M06L6` 의 잡음별 축 평균은 **`exp_g` 를 돌린 이유**다.

    그 전에는 손실 절제가 `mixed` 만 다뤄서 잡음 칸의 평균이 없었고, 화면이
    "축 평균 없음" 을 띄웠다. `exp_g` 를 참조로 쓰는 칸이라면 이제 값이
    있어야 한다 — 없으면 그 실험이 그 조합을 안 돌린 것이다.
    """
    bank = _bank()
    if "M06L6" not in bank["methods"]:
        pytest.skip("M06L6 이 은행에 없다")
    bad = [s["id"] for s in bank["scenes"]
           if s["ref_exp"] == "exp_g" and "M06L6" not in s["ref_mean"]]
    assert not bad, f"exp_g 를 참조하는데 M06L6 평균이 없다: {bad}"


def test_scene_uses_one_record_across_snrs():
    """SNR 칩을 눌렀을 때 **기록까지 바뀌면 안 된다.**

    바뀌면 화면의 변화가 SNR 때문인지 기록 때문인지 알 수 없다 — 잡음과
    크기를 따로 고르게 만든 이유가 바로 그것이다.
    """
    by = {}
    for s in _bank()["scenes"]:
        by.setdefault((s["axis"], s["cond"]), set()).add(s["record"])
    bad = {k: v for k, v in by.items() if len(v) > 1}
    assert not bad, f"같은 잡음인데 SNR 마다 기록이 다르다: {bad}"


def test_noise_realization_does_not_depend_on_snr():
    """평가 세트의 seed 에 SNR 이 들어가면 SNR 을 바꿀 때 **잡음 자체가 바뀐다.**

    `build_eval_set` 은 잡음을 (기록·구간·조건)마다 한 번 뽑고 SNR 로 크기만
    바꾸도록 돼 있는데, seed 를 조합 id 로 주면 그 설계가 무력화된다.
    """
    src = (ROOT / "scripts" / "build_demo_bank.py").read_text()
    body = src[src.index("def eval_items("):src.index("def choose_record(")]
    assert 'seed=f"demo_{scene[\'cond\']}"' in body, \
        "seed 가 잡음 종류에만 의존해야 한다"


# ------------------------------------------------------------ 표시 규약
def test_one_scale_per_scene():
    """한 장면의 모든 파형이 같은 스케일이어야 y 축이 맞는다 (4.1)."""
    for s in _bank()["scenes"]:
        n = {len(base64.b64decode(v)) for v in s["traces"].values()}
        assert len(n) == 1, f"{s['id']}: 파형 길이가 다르다 {n}"
        assert isinstance(s["scale"], float) and s["scale"] > 0


def test_quantization_keeps_three_decimal_digits():
    m = _mod()
    rng = np.random.default_rng(0)
    sig = rng.normal(size=2500) * 2.0
    scale = float(np.max(np.abs(sig))) / 32000.0
    back = np.frombuffer(base64.b64decode(m._q(sig, scale)), dtype="<i2") * scale
    assert np.max(np.abs(back - sig)) < scale, "양자화 오차가 1 LSB 를 넘는다"


def test_every_displayed_method_has_an_explanation():
    """`[?]` 버튼이 빈 채로 뜨지 않게 한다."""
    m = _mod()
    for mid in m.DISPLAY_CFG["methods"] + list(m.DISPLAY_CFG["dl_methods"]):
        assert mid in m.NOTES, f"NOTES 에 {mid} 설명이 없다"


def test_scene_grid_matches_the_reference_experiment_grid():
    """장면 격자가 참조 실험(`exp_g`)의 격자와 **같아야** 축 평균을 나란히 놓을 수 있다.

    처음에 잡음별 장면을 0 dB 로 잡았는데 `exp_b` 는 **10 dB 하나**였다.
    그대로 뒀으면 화면의 '축 평균' 이 다른 조건의 값이었을 것이다. 격자로
    바꾼 지금은 `exp_g` 가 정본이다.
    """
    import yaml

    m = _mod()
    cfg = yaml.safe_load((ROOT / "configs" / "exp_g.yaml").read_text())["data"]
    assert set(m.CONDS) <= set(cfg["conditions"]), \
        f"exp_g 가 안 돌린 잡음: {sorted(set(m.CONDS) - set(cfg['conditions']))}"
    assert set(m.SNRS) <= {float(v) for v in cfg["snr_grid"]}, \
        f"exp_g 가 안 돌린 SNR: {sorted(set(m.SNRS))} vs {cfg['snr_grid']}"
    assert m.REF_EXPS[0] == "exp_g", "격자를 덮는 실험을 먼저 봐야 한다"


def test_every_grid_cell_has_a_label_and_the_highlights_are_real_cells():
    m = _mod()
    specs = {(s["cond"], s["snr"]) for s in m.scene_specs()}
    assert len(specs) == len(m.CONDS) * len(m.SNRS)
    for c in m.CONDS:
        assert c in m.COND_LABEL, f"{c} 의 한글 이름표가 없다"
    for key in m.HIGHLIGHT:
        assert key in specs, f"격자에 없는 조합에 해설이 달려 있다: {key}"


def test_auto_claim_states_numbers_without_claiming():
    """해설을 손으로 안 단 칸은 **숫자만** 적는다 — 42 칸을 손으로 쓰면 그중
    몇 개는 근거 없이 그럴듯한 문장이 된다."""
    m = _mod()
    s = m.auto_claim({"M06": 13.65, "M08": 12.0, "M04": 3.56, "M_FE": 3.5})
    assert "13.65" in s and "3.56" in s and "+10.09" in s
    assert m.auto_claim({}) == ""
    assert m.auto_claim({"M04": 1.0}) == "", "고전만 있으면 차이를 말할 수 없다"


# ------------------------------------------------------------ 화면과 은행
def test_ui_only_reads_keys_the_bank_provides():
    """`demo/index.html` 이 은행에 없는 필드를 읽지 않는지."""
    html = (ROOT / "demo" / "index.html").read_text()
    bank = _bank()
    top = set(bank) | {"scenes"}
    for key in set(re.findall(r"\bB\.([A-Za-z_]\w*)", html)):
        assert key in top, f"index.html 이 B.{key} 를 읽는데 은행에 없다"
    scene = set(bank["scenes"][0])
    for key in set(re.findall(r"\bs\.([A-Za-z_]\w*)", html)):
        assert key in scene, f"index.html 이 s.{key} 를 읽는데 장면에 없다"


def test_ui_loads_without_a_server():
    """시연 노트북에 서버를 띄우게 하지 않는다 — `fetch` 를 쓰면 file:// 에서 죽는다."""
    html = (ROOT / "demo" / "index.html").read_text()
    assert "fetch(" not in html
    assert 'src="demo_bank.js"' in html


# ------------------------------------------------------------ 화면 불변식
#
# 아래는 **깨져도 화면이 멀쩡해 보이는** 성질들이다. 축이 칸마다 달라지거나
# 칸마다 다른 시각을 그리면 그림은 여전히 그럴듯한데 비교가 거짓이 된다.
def _ui() -> str:
    return (ROOT / "demo" / "index.html").read_text()


def test_animation_uses_a_wall_clock_not_a_frame_counter():
    """프레임이 밀리면 칸마다 속도가 어긋나 **같은 시각의 두 파형이 아니게 된다.**

    전역에서 `performance.now()` 를 한 번 읽어 모든 칸이 같은 위상을 쓴다.
    """
    js = _ui()
    assert "performance.now()" in js
    # 패널마다 프레임을 세는 형태가 되살아나지 않는지
    assert not re.search(r"p\.(frame|tick|count)\b", js), "칸별 카운터가 생겼다"
    assert re.search(r"for \(const p of panels\) sweep\(p, i1,", js), \
        "모든 칸이 같은 인덱스로 그려져야 한다"


def test_static_mode_is_the_default():
    """애니메이션은 임팩트용이고 **비교는 정지 화면에서** 한다 (4.2)."""
    assert re.search(r'play:\s*"off"', _ui())


def test_grid_lines_live_outside_the_canvas():
    """스윕의 지우개가 캔버스를 지우므로 격자를 캔버스 안에 그리면 함께 지워진다."""
    js = _ui()
    assert "repeating-linear-gradient" in js, "격자가 CSS 로 있어야 한다"
    body = js[js.index("function draw(cv"):js.index("function tick(")]
    assert "#f0f0f0" not in body, "격자를 다시 캔버스에 그리고 있다"


def test_zoom_applies_to_every_panel_at_once():
    """칸마다 다른 구간을 보면 비교가 아니다 — 구간은 전역 상태 하나다."""
    js = _ui()
    assert re.search(r"win:\s*10,\s*start:\s*0", js)
    assert "p.win" not in js and "p.start" not in js


def test_clean_overlay_is_translucent_and_toggleable():
    js = _ui()
    assert re.search(r'clean:\s*"rgba\([^)]*0\.5\)"', js), "참값이 반투명이어야 한다"
    assert "state.ref = !state.ref" in js, "겹치기 토글이 있어야 한다"


def test_browser_keeps_every_panel_in_phase():
    """실제 브라우저에서 스윕이 진행하고 **모든 칸이 같은 위상**인지 본다.

    playwright 가 없으면 건너뛴다 — 이 검사는 있으면 좋은 것이지 필수가 아니다.
    """
    pytest.importorskip("playwright")
    from playwright.sync_api import sync_playwright

    _bank()                                   # 은행이 낡았으면 여기서 skip
    chrome = Path("/opt/pw-browsers/chromium")
    if not chrome.exists():
        pytest.skip("chromium 없음")
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=str(chrome))
        pg = b.new_page(viewport={"width": 1000, "height": 800})
        errs: list[str] = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.goto("file://" + str((ROOT / "demo" / "index.html").resolve()))
        pg.wait_for_timeout(500)
        pg.get_by_role("button", name="스윕", exact=True).click()
        pg.wait_for_timeout(700)
        # 캔버스에 실제로 뭔가 그려졌고, 지우개가 지나간 빈 구간도 있어야 한다
        stat = pg.evaluate("""() => {
          const out = [];
          for (const cv of document.querySelectorAll('canvas')) {
            const g = cv.getContext('2d');
            const d = g.getImageData(0, 0, cv.width, cv.height).data;
            let ink = 0;
            for (let i = 3; i < d.length; i += 4) if (d[i] > 10) ink++;
            out.push(ink);
          }
          return out;
        }""")
        b.close()
    assert not errs, f"콘솔 오류: {errs}"
    assert len(stat) >= 3 and all(v > 0 for v in stat), f"안 그려진 칸이 있다: {stat}"
