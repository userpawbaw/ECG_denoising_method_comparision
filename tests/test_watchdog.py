"""watchdog 의 판정 (O-2 · O-7 · O-17).

이 스크립트가 잘못 판정하면 **두 학습이 같은 경로에 쓴다** — O-7 이 그렇게
일어났고 서로의 결과를 지웠다. 그래서 "죽었다" 판정은 세 신호가 모두
맞아야 하고, 그중 하나라도 살아 있으면 재시작하지 않는다.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location(
        "watchdog", ROOT / "scripts" / "watchdog.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_argv_matching_ignores_mere_mentions():
    """명령줄에 'train.py' 문자열만 담은 프로세스는 학습이 아니다 (O-17)."""
    m = _mod()
    assert not m._is(os.getpid(), ("train.py", "run_all_training.sh")), \
        "테스트 프로세스 자신이 학습으로 잡힌다 — 부분 문자열 매칭이다"


def test_argv_matching_is_by_element_basename():
    """경로가 붙어 있어도 basename 으로 잡아야 한다."""
    m = _mod()
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
    try:
        assert not m._is(p.pid, ("train.py",))
    finally:
        p.kill(); p.wait()


def test_dead_pid_has_no_argv():
    """죽은 PID 는 argv 가 비어 있어야 한다 (좀비·재사용 방어의 근거)."""
    m = _mod()
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait(); time.sleep(0.2)
    assert m._argv(p.pid) == [] or not m._is(p.pid, ("train.py",))


def test_orphan_trainer_is_not_declared_stalled(tmp_path, monkeypatch):
    """**러너 셸이 죽고 학습 자식만 살아 있으면 재시작하면 안 된다.**

    락 PID 만 보면 '죽음' 으로 보이지만 학습은 돌고 있다. 여기서 재시작하면
    두 학습이 같은 경로에 쓴다 — O-7.
    """
    m = _mod()
    monkeypatch.setattr(m, "LOCK", tmp_path / ".train.lock")
    m.LOCK.mkdir()
    (m.LOCK / "pid").write_text("999999")            # 죽은 러너 PID
    monkeypatch.setattr(m, "_pids", lambda names: [12345] if "train.py" in names else [])
    monkeypatch.setattr(m, "_newest_log_age", lambda: 99999.0)
    st = m.assess()
    assert st["state"] == "orphan_trainer", st
    assert m.restart(st, ["true"]) == 0, "고아 상태에서 재시작하면 안 된다"


def test_stalled_requires_all_three_signals(tmp_path, monkeypatch):
    m = _mod()
    monkeypatch.setattr(m, "LOCK", tmp_path / ".train.lock")
    m.LOCK.mkdir(); (m.LOCK / "pid").write_text("999999")
    monkeypatch.setattr(m, "_pids", lambda names: [])
    # 로그가 방금 움직였으면 아직 stalled 가 아니다
    monkeypatch.setattr(m, "_newest_log_age", lambda: 5.0)
    assert m.assess()["state"] == "settling"
    # 오래 조용해야 stalled
    monkeypatch.setattr(m, "_newest_log_age", lambda: m.STALE_LOG_S + 1)
    assert m.assess()["state"] == "stalled"


def test_restart_is_capped(tmp_path, monkeypatch):
    """반복 실패를 재시작으로 가리지 않는다."""
    m = _mod()
    monkeypatch.setattr(m, "STATE", tmp_path / "wd")
    monkeypatch.setattr(m, "EVENTS", tmp_path / "wd" / "events.jsonl")
    monkeypatch.setattr(m, "LOCK", tmp_path / ".train.lock")
    (tmp_path / "wd").mkdir()
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (tmp_path / "wd" / "events.jsonl").write_text(
        "".join(f'{{"ts":"{now}","action":"restart"}}\n' for _ in range(m.MAX_RESTARTS)))
    st = dict(ts=now, state="stalled")
    assert m.restart(st, ["true"]) == 2, "상한에 걸려야 한다"


def test_lock_is_moved_not_deleted(tmp_path, monkeypatch):
    """락을 지우면 증거가 사라지고, 옛 러너의 rmdir 이 새 락을 지울 수 있다."""
    m = _mod()
    monkeypatch.setattr(m, "STATE", tmp_path / "wd")
    monkeypatch.setattr(m, "EVENTS", tmp_path / "wd" / "events.jsonl")
    monkeypatch.setattr(m, "LOCK", tmp_path / ".train.lock")
    monkeypatch.setattr(m, "ROOT", tmp_path)
    m.LOCK.mkdir(); (m.LOCK / "pid").write_text("999999")
    (tmp_path / "results" / "logs").mkdir(parents=True)
    from datetime import datetime, timezone
    st = dict(ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
              state="stalled")
    m.restart(st, ["true"])
    assert not m.LOCK.exists(), "락이 치워져야 새 러너가 시작할 수 있다"
    stale = list(tmp_path.glob(".train.lock.stale-*"))
    assert stale, "락이 **지워지지 않고** 보존돼야 한다"
    assert (stale[0] / "pid").exists(), "증거(옛 PID)가 남아야 한다"
