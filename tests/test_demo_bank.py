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
    if not BANK.exists():
        pytest.skip("demo_bank.js 없음 — scripts/build_demo_bank.py 를 먼저 돌린다")
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


def test_loss_variant_has_no_axis_mean_on_noise_scenes():
    """`M06L6` 은 손실 절제가 `mixed` 만 다뤄 잡음별 평균이 **없다.**

    있는 척하게 되면 시연이 없는 근거를 주장한다.
    """
    bank = _bank()
    if "M06L6" not in bank["methods"]:
        pytest.skip("M06L6 이 은행에 없다")
    for s in bank["scenes"]:
        assert "M06L6" in s["no_ref"], f"{s['id']}: M06L6 축 평균이 생겼다 — 출처 확인"


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


def test_scene_snr_matches_its_reference_experiment():
    """장면의 SNR 이 참조 실험의 격자에 있어야 축 평균이 같은 조건이다.

    처음에 잡음별 장면을 0 dB 로 잡았는데 `exp_b` 는 **10 dB 하나**다.
    그대로 뒀으면 화면의 '축 평균' 이 다른 조건의 값이었을 것이다.
    """
    m = _mod()
    grid = {"exp_a": {-5.0, 0.0, 5.0, 10.0, 15.0, 20.0}, "exp_b": {10.0}}
    for sc in m.SCENES:
        assert sc["snr"] in grid[sc["ref"]], f"{sc['id']}: {sc['ref']} 격자에 없다"
        if sc["ref"] == "exp_b":
            assert sc["cond"] != "mixed" or sc["snr"] == 10.0


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
