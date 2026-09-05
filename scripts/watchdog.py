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
EXP_LOCK = ROOT / "results" / ".exp.lock"
STATE = ROOT / "results" / ".watchdog"
EVENTS = STATE / "events.jsonl"

# 로그가 이만큼 조용하면 '전진하지 않는다' 로 본다. 한 epoch 이 D1 M09 기준
# 약 2 분이므로 넉넉히 잡는다 — 짧게 잡으면 느린 epoch 을 죽음으로 오판한다.
STALE_LOG_S = 900
# **러너도 학습도 없는데 로그만 최근일 때** 봐주는 창 (O-25).
#
# 이 상태는 「전환 중」이 아니라 **「방금 죽었다」** 다. 진짜 전환 중이라면
# 러너 셸이 살아 있어서 `running` 으로 잡힌다. 여기에 `STALE_LOG_S`(15 분)를
# 쓰면 **죽은 학습을 15 분 동안 «정상» 이라고 보고한다** — 실제로 그랬다.
# 락을 막 만들고 러너가 아직 exec 전인 찰나만 봐주면 되므로 짧게 잡는다.
SETTLING_S = 60
# 같은 이유로 재시작은 몇 번까지만. OOM 처럼 매번 죽는 학습을 무한히 되살리면
# 로그에 '재시작' 만 쌓이고 원인은 가려진다.
MAX_RESTARTS = 3
# 락을 잡는 러너들. **일회성 러너를 빠뜨리면 그 실행은 감시 밖이다**(O-24).
RUNNER_NAMES = ("run_all_training.sh", "run_one_training.sh")


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


def assess_exp() -> dict:
    """실험 단계도 같은 방식으로 본다.

    초판은 학습만 봤는데, 실제로 컨테이너 재시작이 **실험 도중**(D1 exp_c
    30/44)에 터졌다. 학습은 `--resume` 이 있어 이어지지만 실험은 그렇지
    않고, `.exp.lock` 이 고아로 남아 **재개 자체를 막았다.** 감시가 학습에서
    끝나면 파이프라인의 절반이 무방비다.
    """
    running = bool(_pids(("run_exp.py",)))
    lock = EXP_LOCK.exists()
    if running:
        state = "running"
    elif lock:
        state = "stalled"          # 락만 남았다 — 고아 락이 재개를 막는다
    else:
        state = "idle"
    return dict(stage="experiment", state=state, lock=lock, running=running)


def assess() -> dict:
    """상태를 판정한다. 부작용 없음."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lock_pid = None
    if (LOCK / "pid").exists():
        try:
            lock_pid = int((LOCK / "pid").read_text().strip())
        except (ValueError, OSError):
            lock_pid = None

    lock_cmd = _lock_cmd()
    runner_alive = lock_pid is not None and _is(lock_pid, RUNNER_NAMES)
    trainers = _pids(("train.py",))
    log_age = _newest_log_age()

    if not LOCK.exists():
        # **락 없이 도는 학습은 감시 밖이다.** 락이 없으면 감시자는 "돌아야 할
        # 학습이 있다" 를 알 수 없고, 그것이 죽어도 stalled 로 갈 수 없다 —
        # 영원히 idle 이다. 그래서 이것 자체를 이상으로 보고한다 (O-24).
        state = "idle" if not trainers else "trainer_without_lock"
    elif runner_alive:
        state = "running"
    elif trainers:
        # 러너 셸은 죽었는데 학습 자식이 살아 있다. **재시작하면 안 된다.**
        state = "orphan_trainer"
    elif log_age < SETTLING_S:
        # 락 PID 도 학습도 없는데 로그가 **방금** 움직였다. 러너가 exec 하기
        # 직전의 찰나일 수 있으므로 그 폭만 봐준다 — 그보다 길면 죽은 것이다
        # (O-25). 전환 중이라면 러너 셸이 살아 있어 위에서 `running` 이 된다.
        state = "settling"
    else:
        state = "stalled"

    return dict(ts=now, state=state, lock=LOCK.exists(), lock_pid=lock_pid,
                runner_alive=runner_alive, trainers=trainers,
                log_age_s=round(log_age, 1), lock_cmd=lock_cmd)


def _lock_cmd() -> list[str] | None:
    """락이 적어 둔 **재개 명령.**

    러너 전체를 기본값으로 되살리면, 목록에 없는 일회성 학습은 **되살아나지
    않고 엉뚱한 학습이 대신 돈다.** 무엇을 띄워야 하는지는 그 실행만 안다.
    """
    f = LOCK / "cmd"
    try:
        raw = f.read_bytes()
    except OSError:
        return None
    argv = [a for a in raw.decode("utf-8", "replace").split("\0") if a]
    return argv or None


def _last_recorded() -> dict | None:
    if not EVENTS.exists():
        return None
    lines = [l for l in EVENTS.read_text().splitlines() if l.strip()]
    for l in reversed(lines):
        try:
            return json.loads(l)
        except json.JSONDecodeError:
            continue
    return None


def _record(ev: dict, *, force: bool = False) -> None:
    """**바뀐 것만** 적는다.

    초판은 점검마다 한 줄씩 남겼다. 그러면 개입 기록이어야 할 파일이 상태
    스냅샷 로그가 되고, 60 분마다 저장소가 더러워진다(그리고 정작 중요한
    개입 줄이 no-op 사이에 묻힌다).

    개입(`action` 이 있는 것)은 항상, 그 외에는 **상태가 직전과 다를 때만**
    적는다.
    """
    if not force and not ev.get("action"):
        prev = _last_recorded()
        if prev is not None:
            same = (prev.get("state") == ev.get("state")
                    and (prev.get("exp") or {}).get("state")
                        == (ev.get("exp") or {}).get("state"))
            if same:
                return
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
        _record(dict(**st, action="restart_blocked", restarts_recent=n), force=True)
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
                 restarts_recent=n + 1), force=True)
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
    ex = assess_exp()
    st["exp"] = ex
    _record(st)
    print(f"[watchdog] 학습 {st['state']}  락={st['lock']} pid={st['lock_pid']} "
          f"러너생존={st['runner_alive']} 학습프로세스={st['trainers']} "
          f"로그나이={st['log_age_s']}s")
    print(f"[watchdog] 실험 {ex['state']}  락={ex['lock']} 실행중={ex['running']}")
    if ex["state"] == "stalled":
        print("  **실험 락만 남았다.** 재개하려면 먼저 치워야 한다:\n"
              "    rm -rf results/.exp.lock")

    if st["state"] == "trainer_without_lock":
        print("  **락 없이 학습이 돌고 있다 — 감시 밖이다.** 이 실행이 죽으면\n"
              "  감시자는 idle 로 보고할 뿐 되살리지 못한다 (O-24). 일회성\n"
              "  학습도 락을 잡는 러너로 띄울 것:\n"
              "    bash scripts/run_one_training.sh <축> <설정>")
    elif st["state"] == "orphan_trainer":
        print("  러너 셸은 죽었지만 **학습 프로세스가 살아 있다.** 재시작하지 "
              "않는다 — 같은 경로에 두 학습이 쓰면 O-7 이 재현된다.")
    elif st["state"] == "stalled":
        print("  **학습이 죽었다.** --restart 를 주면 보존 후 재개한다.")

    if a.restart:
        # **락이 적어 둔 명령이 기본값보다 우선한다.** 러너 전체를 되살리면
        # 목록에 없는 일회성 학습은 되살아나지 않는다 (O-24).
        cmd = a.cmd or st.get("lock_cmd") or [
            "bash", "scripts/run_all_training.sh", "auto"]
        return restart(st, cmd)
    healthy = st["state"] in ("running", "orphan_trainer", "idle", "settling")
    return 0 if healthy and ex["state"] != "stalled" else 1


if __name__ == "__main__":
    raise SystemExit(main())
