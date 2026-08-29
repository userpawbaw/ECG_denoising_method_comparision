"""실험 이어받기 판정 (D-10 계열).

**시각 비교로는 안 된다.** 실험이 config 변경 **전에 시작해 변경 후에
끝나면** 산출물 시각이 config 보다 새로우면서도 낡은 설정으로 만들어진다.
`exp_b` 가 실제로 그렇게 M09 없이 완주했고, 시각 규칙이었으면 그대로
건너뛰어 **M09 만 빠진 표가 조용히 살아남았다.**
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location(
        "exp_is_current", ROOT / "scripts" / "exp_is_current.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _setup(tmp, exp="e1", tag="d9", methods=("M00",), dl=("M06",),
           man_methods=("M00", "M06")):
    (tmp / "configs").mkdir(parents=True, exist_ok=True)
    (tmp / "configs" / f"{exp}.yaml").write_text(yaml.safe_dump(
        {"methods": list(methods), "dl_methods": {k: {} for k in dl}}))
    out = tmp / "results" / tag / exp
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.parquet").write_bytes(b"x")
    (out / "manifest.json").write_text(json.dumps({"methods": list(man_methods)}))
    return out


def test_missing_method_forces_rerun(tmp_path, monkeypatch):
    """config 에 방법을 추가했으면 산출물이 아무리 새로워도 다시 돌아야 한다."""
    m = _mod(); monkeypatch.setattr(m, "ROOT", tmp_path)
    _setup(tmp_path, dl=("M06", "M09"), man_methods=("M00", "M06"))
    ok, why = m.is_current("e1", "d9")
    assert not ok and "M09" in why, why


def test_complete_run_is_skipped(tmp_path, monkeypatch):
    m = _mod(); monkeypatch.setattr(m, "ROOT", tmp_path)
    _setup(tmp_path, dl=("M06",), man_methods=("M00", "M06"))
    ok, why = m.is_current("e1", "d9")
    assert ok, why


def test_newer_checkpoint_forces_rerun(tmp_path, monkeypatch):
    """체크포인트가 바뀌면 산출물도 다시 만들어야 한다 (F-9)."""
    import os, time
    m = _mod(); monkeypatch.setattr(m, "ROOT", tmp_path)
    out = _setup(tmp_path, dl=("M06",), man_methods=("M00", "M06"))
    ck = tmp_path / "results" / "d9" / "m06_l1"; ck.mkdir(parents=True)
    (ck / "best.pt").write_bytes(b"w")
    os.utime(ck / "best.pt", (time.time() + 100, time.time() + 100))
    ok, why = m.is_current("e1", "d9")
    assert not ok and "체크포인트" in why, why


def test_missing_artifact_is_not_current(tmp_path, monkeypatch):
    m = _mod(); monkeypatch.setattr(m, "ROOT", tmp_path)
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "configs" / "e1.yaml").write_text(yaml.safe_dump({"methods": ["M00"]}))
    ok, why = m.is_current("e1", "d9")
    assert not ok and "없음" in why


def test_runner_uses_the_content_check_not_mtimes():
    """러너가 시각 비교로 되돌아가면 exp_b 사고가 재현된다."""
    src = (ROOT / "scripts" / "run_all_experiments.sh").read_text()
    assert "exp_is_current.py" in src, "러너가 내용 비교를 쓰지 않는다"
    assert "newest_ckpt=" not in src, "옛 시각 비교 잔재가 남아 있다"
