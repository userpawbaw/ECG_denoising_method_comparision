"""실험 내부 재개 (`run_exp.py` 의 `_partial/`).

컨테이너가 실험 도중 재시작되면(자주 그런다 — O-2 · O-15 · O-17) 그 실험은
처음부터 다시 돌았다. EXP-G(2156 항목)를 돌리다 220 과 70 항목에서 연속으로
죽어 **진행이 0** 이었다.

여기서 고정하는 것은 두 가지다.
  1. 이어 붙인 결과가 **한 번에 돈 결과와 같아야** 한다 (행이 늘지도 줄지도
     않는다).
  2. **설정이 바뀌면 부분 결과를 버려야** 한다 — 이어 붙이면 서로 다른 조건의
     행이 한 표에 섞인다(F-9 계열).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("run_exp", ROOT / "scripts" / "run_exp.py")
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as e:                      # torch 없는 환경
        pytest.skip(f"run_exp 를 못 읽었다: {e}")
    return m


def _rows(a: int, b: int) -> list[dict]:
    return [{"item": i, "metric": "snr", "value": float(i)} for i in range(a, b)]


def test_chunks_hold_only_new_rows_so_resume_does_not_duplicate(tmp_path):
    """조각마다 **새 행만** 써야 한다.

    처음 구현은 매번 `rows` 전체를 썼고, 읽을 때 조각을 다 이어 붙이므로 같은
    행이 조각 수만큼 늘었다(50 항목에서 25200 행 — 맞는 값은 16800).
    **평균은 안 변해서** 표를 봐도 모른다. 행 수를 세야 잡힌다.
    """
    m = _mod()
    fp = "abc123"
    m.save_partial(tmp_path, fp, _rows(0, 25), 25)
    m.save_partial(tmp_path, fp, _rows(25, 50), 50)
    rows, done = m.load_partial(tmp_path, fp)
    assert done == 50
    assert len(rows) == 50, f"중복이다: {len(rows)}"
    assert [r["item"] for r in rows] == list(range(50))


def test_resumed_table_equals_the_uninterrupted_one(tmp_path):
    """끊었다 이은 결과가 한 번에 돈 결과와 같아야 한다."""
    m = _mod()
    fp = "abc123"
    m.save_partial(tmp_path, fp, _rows(0, 25), 25)
    resumed, done = m.load_partial(tmp_path, fp)
    resumed = resumed + _rows(done, 40)          # 재시작 후 이어서 처리한 몫
    straight = _rows(0, 40)
    assert pd.DataFrame(resumed).equals(pd.DataFrame(straight))


def test_a_changed_config_throws_the_partial_away(tmp_path):
    """설정이 바뀌었는데 이어 붙이면 다른 조건의 행이 한 표에 섞인다."""
    m = _mod()
    m.save_partial(tmp_path, "old-fp", _rows(0, 25), 25)
    rows, done = m.load_partial(tmp_path, "new-fp")
    assert (rows, done) == ([], 0)
    assert not (tmp_path / "_partial").exists(), "낡은 부분 결과를 안 지웠다"


def test_fingerprint_reacts_to_the_things_that_change_the_eval_set():
    """지문이 평가 세트를 바꾸는 항목에 반응해야 한다 — 아니면 1 번 검사가 헛돈다."""
    m = _mod()
    base = dict(data={"snr_grid": [0, 10]}, mode="snr", frontend=True, eval={})
    fp = m._fingerprint(base, 100, ["M04"])
    assert fp != m._fingerprint(dict(base, data={"snr_grid": [0, 10, 20]}), 100, ["M04"])
    assert fp != m._fingerprint(base, 200, ["M04"])
    assert fp != m._fingerprint(base, 100, ["M04", "M06"])
    assert fp == m._fingerprint(base, 100, ["M04"]), "같은 설정인데 지문이 다르다"


def test_completed_run_leaves_no_partial_behind():
    """완주하면 `_partial/` 이 없어야 한다 — 남아 있으면 '안 끝났다' 는 신호다."""
    src = (ROOT / "scripts" / "run_exp.py").read_text()
    assert 'shutil.rmtree(out / "_partial"' in src
    assert "results/**/_partial/" in (ROOT / ".gitignore").read_text(), \
        "진행 상태를 추적하면 커밋 잡음이 된다"
