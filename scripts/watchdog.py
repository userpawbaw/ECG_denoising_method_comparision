#!/usr/bin/env python3
"""학습 러너가 살아 있는지 판정하고, 죽었으면 **증거를 남기고** 재개한다.

    python scripts/watchdog.py                 # 판정만 (기본)
    python scripts/watchdog.py --restart       # 죽었으면 재개까지
    python scripts/watchdog.py --report        # 그동안 무슨 일이 있었나

**왜 필요한가** — 이 프로젝트에서 학습이 네 번 죽었고(O-2 ×2, O-15, O-17)
**네 번 다 사람이 물어서 발견했다.** 합계 10 시간을 잃었다. 기존 장치 중
어느 것도 "학습이 죽었다" 를 알리지 않는다:

    --resume        복구를 싸게 만든다 (감지는 안 한다)
    락 + 게이트     덜 학습된 모델이 표에 드는 것을 막는다 (감지는 안 한다)
    check_freshness 산출물 낡음만 본다

**컨테이너 안에서 도는 감시자는 쓸모가 없다** — 주 실패 모드가 컨테이너
재시작이고, 감시자도 함께 죽는다(O-17 의 드라이버는 `setsid` 였는데도 죽었다).
그래서 이 스크립트는 **밖에서 주기적으로 호출되는 것을 전제**로 한다.

판정을 3 중으로 하는 이유
--------------------------
락 PID 의 생존만 보면 구멍이 셋이다.

1. **PID 재사용** — 재부팅하면 번호가 낮은 것부터 다시 배정된다. 무관한
   프로세스가 옛 PID 를 받을 수 있다. → argv 를 함께 본다.
2. **좀비** — `/proc/PID` 는 있는데 실체가 없다. → argv 가 비어 걸러진다.
3. **고아 자식** — 러너 셸은 죽고 `train.py` 자식만 살아남는 경우.
   **락 PID 는 죽었는데 학습은 계속 돈다.** 이때 재시작하면 두 학습이 같은
   경로에 쓴다 — O-7 이 그렇게 일어났다(두 러너가 서로의 결과를 지웠다).
   → `train.py` 프로세스를 따로 찾고, **로그가 전진하는지**도 본다.

셋을 다 보지 않으면 "정말 죽음" 과 "고아가 돌고 있음" 을 가를 수 없다.

재시작이 오판이었을 때
----------------------
락은 **지우지 않고 옆으로 치운다**(`.train.lock.stale-{ts}`). 증거가 남고,
옛 러너가 `trap EXIT` 으로 `rmdir` 을 시도해도 무해하게 실패한다. 그리고
재시작 **전에** 현재 산출물을 `archive_run` 으로 보존하므로, 옛 러너가
사실은 끝까지 갔더라도 **두 결과가 모두 남는다.** 무슨 일이 있었는지는
`--report` 로 읽는다 — 사용자가 세션을 다시 열었을 때 "옛 러너는 끝났고 새
러너는 N 시간째" 를 보고 **버릴지 기다릴지 고를 수 있게** 하기 위한 것이다.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "results" / ".train.lock"
STATE = ROOT / "results" / ".watchdog"
EVENTS = STATE / "events.jsonl"

# 로그가 이만큼 조용하면 '전진하지 않는다' 로 본다. 한 epoch 이 D1 M09 기준
# 약 2 분이므로 넉넉히 잡는다 — 짧게 잡으면 느린 epoch 을 죽음으로 오판한다.
STALE_LOG_S = 900
# 같은 이유로 재시작은 몇 번까지만. OOM 처럼 매번 죽는 학습을 무한히 되살리면
# 로그에 '재시작' 만 쌓이고 원인은 가려진다.
MAX_RESTARTS = 3


def _argv(pid: int) -> list[str]:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except OSError:
        return []
    return [a for a in raw.decode("utf-8", "replace").split("\0") if a]


def _is(pid: int, names: tuple[str, ...]) -> bool:
    """argv 의 **원소** basename 이 names 에 있는가. 부분 문자열이 아니다(O-17)."""
    return any(a.rsplit("/", 1)[-1] in names for a in _argv(pid))


def _pids(names: tuple[str, ...]) -> list[int]:
    out = []
    for pd in Path("/proc").iterdir():
        if pd.name.isdigit() and _is(int(pd.name), names):
            out.append(int(pd.name))
    return out


def _newest_log_age() -> float:
    logs = list((ROOT / "results" / "logs").glob("*.log"))
    logs += list((ROOT / "results").glob("d*/*/log.csv"))
    if not logs:
        return float("inf")
    return time.time() - max(p.stat().st_mtime for p in logs)


def assess() -> dict:
    """상태를 판정한다. 부작용 없음."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lock_pid = None
    if (LOCK / "pid").exists():
        try:
            lock_pid = int((LOCK / "pid").read_text().strip())
        except (ValueError, OSError):
            lock_pid = None

    runner_alive = lock_pid is not None and _is(lock_pid, ("run_all_training.sh",))
    trainers = _pids(("train.py",))
    log_age = _newest_log_age()

    if not LOCK.exists():
        state = "idle" if not trainers else "trainer_without_lock"
    elif runner_alive:
        state = "running"
    elif trainers:
        # 러너 셸은 죽었는데 학습 자식이 살아 있다. **재시작하면 안 된다.**
        state = "orphan_trainer"
    elif log_age < STALE_LOG_S:
        # 락 PID 도 학습도 없는데 로그는 방금 움직였다 — 전환 중일 수 있다.
        state = "settling"
    else:
        state = "stalled"

    return dict(ts=now, state=state, lock=LOCK.exists(), lock_pid=lock_pid,
                runner_alive=runner_alive, trainers=trainers,
                log_age_s=round(log_age, 1))


def _record(ev: dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def _restart_count_recent(hours: float = 6.0) -> int:
    if not EVENTS.exists():
        return 0
    cut = time.time() - hours * 3600
    n = 0
    for line in EVENTS.read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("action") != "restart":
            continue
        try:
            t = datetime.fromisoformat(ev["ts"]).timestamp()
        except (KeyError, ValueError):
            continue
        if t >= cut:
            n += 1
    return n


def restart(st: dict, cmd: list[str]) -> int:
    """죽었다고 판정된 학습을 재개한다. **증거를 먼저 남긴다.**"""
    if st["state"] != "stalled":
        print(f"[watchdog] 재시작하지 않는다 (상태 {st['state']}).")
        return 0
    n = _restart_count_recent()
    if n >= MAX_RESTARTS:
        print(f"[watchdog] 최근 6 시간 재시작 {n} 회 — 상한({MAX_RESTARTS})에 걸렸다. "
              "반복 실패를 재시작으로 가리지 않는다. 로그를 볼 것.")
        _record(dict(**st, action="restart_blocked", restarts_recent=n))
        return 2

    sys.path.insert(0, str(ROOT))
    from ecgdn.utils import archive_run

    archived = []
    for d in sorted((ROOT / "results").glob("d*/*/")):
        if (d / "last.pt").exists():
            try:
                archived.append(str(archive_run(d).relative_to(ROOT)))
            except Exception as e:            # 보존 실패가 재개를 막지는 않는다
                print(f"[watchdog] 보존 실패 {d}: {e}")

    # 락은 **지우지 않고 옆으로 치운다** — 증거가 남고, 옛 러너의 rmdir 도
    # 무해하게 실패한다.
    moved = None
    if LOCK.exists():
        moved = LOCK.with_name(
            f".train.lock.stale-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
        LOCK.rename(moved)

    log = ROOT / "results" / "logs" / "watchdog_restart.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(f"\n===== {st['ts']} watchdog 재시작 =====\n")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log.open("a"),
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)
    _record(dict(**st, action="restart", pid=proc.pid, cmd=cmd,
                 archived=archived, stale_lock=str(moved) if moved else None,
                 restarts_recent=n + 1))
    print(f"[watchdog] 재개했다 (pid {proc.pid}). 보존 {len(archived)} 건, "
          f"낡은 락 -> {moved.name if moved else '없음'}")
    return 1


def report() -> int:
    """세션을 다시 열었을 때 읽는 요약. **무슨 일이 있었는지**를 먼저 말한다."""
    st = assess()
    print(f"[watchdog] 지금 상태: {st['state']}  "
          f"(락={st['lock']}, 러너={st['runner_alive']}, "
          f"학습 프로세스={len(st['trainers'])}, 로그 나이={st['log_age_s']}s)")
    if not EVENTS.exists():
        print("  기록된 사건 없음.")
        return 0
    evs = [json.loads(l) for l in EVENTS.read_text().splitlines() if l.strip()]
    acts = [e for e in evs if e.get("action")]
    if not acts:
        print("  개입한 적 없음 (판정만 했다).")
        return 0
    print(f"  개입 {len(acts)} 건:")
    for e in acts[-10:]:
        print(f"    {e['ts']}  {e['action']}  "
              f"(직전 상태 {e.get('state')}, 보존 {len(e.get('archived') or [])} 건)")
    print("\n  **두 결과가 함께 남아 있을 수 있다.** 옛 러너가 사실은 끝까지 갔다면")
    print("  그 산출물은 results/d*/*/runs/ 에 보존돼 있다. 둘 다 실험에 넣어")
    print("  비교하는 편이 싸다 — 재학습보다 실험이 훨씬 저렴하다.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart", action="store_true",
                    help="stalled 로 판정되면 재개까지 한다")
    ap.add_argument("--report", action="store_true", help="그동안의 사건 요약")
    ap.add_argument("--cmd", nargs=argparse.REMAINDER,
                    help="재개에 쓸 명령 (기본: run_all_training.sh 전체)")
    a = ap.parse_args()

    if a.report:
        return report()

    st = assess()
    _record(st)
    print(f"[watchdog] {st['state']}  락={st['lock']} pid={st['lock_pid']} "
          f"러너생존={st['runner_alive']} 학습프로세스={st['trainers']} "
          f"로그나이={st['log_age_s']}s")

    if st["state"] == "orphan_trainer":
        print("  러너 셸은 죽었지만 **학습 프로세스가 살아 있다.** 재시작하지 "
              "않는다 — 같은 경로에 두 학습이 쓰면 O-7 이 재현된다.")
    elif st["state"] == "stalled":
        print("  **학습이 죽었다.** --restart 를 주면 보존 후 재개한다.")

    if a.restart:
        cmd = a.cmd or ["bash", "scripts/run_all_training.sh", "auto"]
        return restart(st, cmd)
    return 0 if st["state"] in ("running", "orphan_trainer", "idle", "settling") else 1


if __name__ == "__main__":
    raise SystemExit(main())
