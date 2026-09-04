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


# ---------------------------------------------------------------- 실험 단계
def test_experiment_stage_is_watched_too(tmp_path, monkeypatch):
    """감시가 학습에서 끝나면 파이프라인 절반이 무방비다.

    실제로 컨테이너 재시작이 **실험 도중**(D1 exp_c 30/44)에 터졌고,
    `.exp.lock` 이 고아로 남아 **재개 자체를 막았다.** 학습은 `--resume` 이
    있어 이어지지만 실험은 그렇지 않다.
    """
    m = _mod()
    monkeypatch.setattr(m, "EXP_LOCK", tmp_path / ".exp.lock")

    monkeypatch.setattr(m, "_pids", lambda names: [])
    assert m.assess_exp()["state"] == "idle", "락도 프로세스도 없으면 idle"

    (tmp_path / ".exp.lock").mkdir()
    assert m.assess_exp()["state"] == "stalled", "고아 락은 stalled 여야 한다"

    monkeypatch.setattr(m, "_pids", lambda names: [111] if "run_exp.py" in names else [])
    assert m.assess_exp()["state"] == "running"


def test_exit_code_flags_a_stalled_experiment(tmp_path, monkeypatch):
    """학습이 멀쩡해도 실험 락이 고아면 종료코드가 이상을 알려야 한다."""
    m = _mod()
    monkeypatch.setattr(m, "LOCK", tmp_path / ".train.lock")
    monkeypatch.setattr(m, "EXP_LOCK", tmp_path / ".exp.lock")
    monkeypatch.setattr(m, "STATE", tmp_path / "wd")
    monkeypatch.setattr(m, "EVENTS", tmp_path / "wd" / "events.jsonl")
    (tmp_path / ".exp.lock").mkdir()
    monkeypatch.setattr(m, "_pids", lambda names: [])
    monkeypatch.setattr(m, "_newest_log_age", lambda: 1.0)
    monkeypatch.setattr(sys, "argv", ["watchdog.py"])
    assert m.main() == 1, "실험 고아 락을 정상으로 보고하면 안 된다"


def test_events_log_only_records_changes(tmp_path, monkeypatch):
    """점검마다 한 줄씩 쌓으면 개입 기록이 no-op 에 묻히고 저장소가 더러워진다."""
    m = _mod()
    monkeypatch.setattr(m, "STATE", tmp_path / "wd")
    monkeypatch.setattr(m, "EVENTS", tmp_path / "wd" / "events.jsonl")
    ev = dict(ts="t1", state="running", exp=dict(state="running"))
    m._record(ev)
    m._record(dict(ev, ts="t2"))          # 같은 상태 — 안 적혀야 한다
    m._record(dict(ev, ts="t3"))
    n = len((tmp_path / "wd" / "events.jsonl").read_text().strip().splitlines())
    assert n == 1, f"같은 상태를 {n} 줄 적었다"

    m._record(dict(ts="t4", state="stalled", exp=dict(state="running")))
    m._record(dict(ts="t5", state="stalled", action="restart"), force=True)
    n = len((tmp_path / "wd" / "events.jsonl").read_text().strip().splitlines())
    assert n == 3, f"상태 변화와 개입은 적혀야 한다 ({n} 줄)"


# ------------------------------------------- 일회성 학습도 감시 안에 있어야 한다
def test_single_run_runner_counts_as_a_runner():
    """`run_one_training.sh` 가 러너 목록에 없으면 그 실행은 **감시 밖이다**(O-24).

    락 PID 가 살아 있어도 `runner_alive` 가 False 가 되어 `orphan_trainer` 로
    잘못 분류되고, 죽은 뒤에도 판정이 흐려진다.
    """
    m = _mod()
    assert "run_one_training.sh" in m.RUNNER_NAMES
    assert "run_all_training.sh" in m.RUNNER_NAMES


def test_a_trainer_without_a_lock_is_reported_as_a_fault(tmp_path, monkeypatch):
    """**락 없이 도는 학습은 정상이 아니다.**

    감시자는 락으로만 "돌아야 할 학습이 있다" 를 안다. 락이 없으면 그 실행이
    죽어도 `stalled` 로 갈 수 없고 영원히 `idle` 이다 — 45 분을 그렇게 잃었다
    (O-24). 그래서 종료코드가 이상을 알려야 한다.
    """
    m = _mod()
    monkeypatch.setattr(m, "LOCK", tmp_path / ".train.lock")
    monkeypatch.setattr(m, "EXP_LOCK", tmp_path / ".exp.lock")
    monkeypatch.setattr(m, "STATE", tmp_path / "wd")
    monkeypatch.setattr(m, "EVENTS", tmp_path / "wd" / "events.jsonl")
    monkeypatch.setattr(m, "_pids", lambda names: [4242] if "train.py" in names else [])
    monkeypatch.setattr(m, "_newest_log_age", lambda: 1.0)
    assert m.assess()["state"] == "trainer_without_lock"
    monkeypatch.setattr(sys, "argv", ["watchdog.py"])
    assert m.main() == 1, "감시 밖의 학습을 정상으로 보고하면 안 된다"


def test_restart_uses_the_command_the_lock_recorded(tmp_path, monkeypatch):
    """일회성 학습을 러너 전체로 되살리면 **엉뚱한 학습이 대신 돈다**(O-24).

    `run_all_training.sh` 의 기본 목록에 `m06_l1_nofe` 는 없다. 무엇을 띄워야
    하는지는 그 실행이 락에 적어 둔 것뿐이다.
    """
    m = _mod()
    monkeypatch.setattr(m, "LOCK", tmp_path / ".train.lock")
    monkeypatch.setattr(m, "EXP_LOCK", tmp_path / ".exp.lock")
    monkeypatch.setattr(m, "STATE", tmp_path / "wd")
    monkeypatch.setattr(m, "EVENTS", tmp_path / "wd" / "events.jsonl")
    monkeypatch.setattr(m, "ROOT", tmp_path)
    m.LOCK.mkdir()
    (m.LOCK / "pid").write_text("999999")
    want = ["bash", "scripts/run_one_training.sh", "mitdb", "m06_l1_nofe"]
    (m.LOCK / "cmd").write_bytes(b"\0".join(a.encode() for a in want) + b"\0")
    monkeypatch.setattr(m, "_pids", lambda names: [])
    monkeypatch.setattr(m, "_newest_log_age", lambda: 1e9)

    st = m.assess()
    assert st["state"] == "stalled"
    assert st["lock_cmd"] == want, "락이 적어 둔 재개 명령을 못 읽었다"

    seen: list = []
    monkeypatch.setattr(m.subprocess, "Popen",
                        lambda cmd, **kw: seen.append(cmd) or _FakeProc())
    monkeypatch.setattr(sys, "argv", ["watchdog.py", "--restart"])
    (tmp_path / "results" / "logs").mkdir(parents=True)
    m.main()
    assert seen and seen[0] == want, (
        f"락의 명령 대신 {seen} 을 띄웠다 — 목록에 없는 학습은 되살아나지 않는다")


class _FakeProc:
    pid = 12345
