"""락을 잡는 스크립트가 **전부** 감시자의 목록에 있는가.

O-24 가 정확히 이 빠뜨림이었다 — 일회성 러너가 목록에 없어서, 락은 잡혔는데
감시자는 러너를 못 알아봤다. 새 러너를 만들 때마다 사람이 목록을 고쳐야 하는
구조라 또 빠진다. 그래서 **`results/.train.lock` 을 만드는 스크립트를 저장소에서
찾아** 목록과 대조한다.
"""
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _watchdog():
    """`scripts/watchdog.py` 를 모듈로 읽는다 (패키지가 아니라 스크립트다)."""
    spec = importlib.util.spec_from_file_location(
        "_wd", ROOT / "scripts" / "watchdog.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scripts_that_take_the_training_lock() -> set[str]:
    out = set()
    for p in sorted((ROOT / "scripts").glob("*.sh")):
        t = p.read_text()
        # `LOCK=results/.train.lock` 를 두고 `mkdir "$LOCK"` 로 잡는 형태
        if re.search(r"LOCK=results/\.train\.lock", t) and 'mkdir "$LOCK"' in t:
            out.add(p.name)
    return out


def test_every_lock_taking_runner_is_known_to_the_watchdog():
    RUNNER_NAMES = _watchdog().RUNNER_NAMES
    takers = _scripts_that_take_the_training_lock()
    assert takers, "락을 잡는 스크립트를 하나도 못 찾았다 — 이 검사가 무력해졌다"
    missing = sorted(takers - set(RUNNER_NAMES))
    assert not missing, (
        f"감시자가 모르는 러너: {missing} — scripts/watchdog.py 의 RUNNER_NAMES "
        "에 추가할 것. 없으면 락은 보이는데 러너는 안 보여 멀쩡한 학습이 "
        "stalled 로 판정돼 재시작된다 (O-24).")


def test_watchdog_counts_the_sweep_as_a_trainer():
    """`train.py` 만 세면 sweep 이 학습 중인데도 «학습 프로세스 없음» 이 된다."""
    TRAINER_NAMES = _watchdog().TRAINER_NAMES
    assert "train.py" in TRAINER_NAMES
    assert "run_seed_sweep.py" in TRAINER_NAMES
