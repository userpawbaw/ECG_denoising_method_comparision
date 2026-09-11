#!/usr/bin/env bash
# seed sweep(`run_seed_sweep.py`)을 **학습 락을 잡고** 돌린다.
#
#   bash scripts/run_sweep_locked.sh structure 4-7
#   bash scripts/run_sweep_locked.sh --arms m06_l3,m06_l6 4-7
#
# **왜 따로 있는가** — `run_seed_sweep.py` 를 직접 띄우면 락을 안 잡아서
# 감시자(`watchdog.py`)가 "돌아야 할 학습이 있다" 는 사실을 모른다. 죽어도
# `idle` 로만 보고하고 아무도 되살리지 않는다. 실제로 그렇게 잃었다(O-24).
#
# 락 안에 **sweep 을 재개하는 명령**을 적는다 — 판 하나가 아니라 sweep 전체다.
# `run_seed_sweep.py` 는 `summary.json` 이 있는 판을 건너뛰므로 그대로 이어진다.
set -u
cd "$(dirname "$0")/.."

# 첫 인자가 `--arms` 면 arm 목록을 직접 받는다. 같은 `--out` 에 계속 쌓아
# 기준 arm(`m06_l1`)을 여러 비교가 **공유**하게 하려는 것이다 — 다시 돌릴
# 이유가 없다.
MODE=stage
if [ "${1:-}" = "--arms" ]; then MODE=arms; shift; fi
STAGE="${1:-}"
SEEDS="${2:-}"
[ -n "$STAGE" ] && [ -n "$SEEDS" ] || {
  echo "사용법: bash scripts/run_sweep_locked.sh <stage> <seeds>"
  echo "        bash scripts/run_sweep_locked.sh --arms <a,b,c> <seeds>"
  echo "  예:  bash scripts/run_sweep_locked.sh structure 4-7"; exit 2; }
shift 2
if [ "$MODE" = arms ]; then SEL=(--arms "$STAGE"); else SEL=(--stage "$STAGE"); fi

LOCK=results/.train.lock
mkdir -p results results/logs
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another training run holds $LOCK; abort"; exit 1
fi
echo $$ > "$LOCK/pid"
if [ "$MODE" = arms ]; then
  printf '%s\0' bash scripts/run_sweep_locked.sh --arms "$STAGE" "$SEEDS" "$@" > "$LOCK/cmd"
else
  printf '%s\0' bash scripts/run_sweep_locked.sh "$STAGE" "$SEEDS" "$@" > "$LOCK/cmd"
fi
trap 'rm -f "$LOCK/pid" "$LOCK/cmd"; rmdir "$LOCK" 2>/dev/null' EXIT

LOG="results/logs/sweep_$(echo "$STAGE" | tr ',' '-')_$(echo "$SEEDS" | tr ',' '_').log"
echo "[sweep-locked] $MODE=$STAGE seeds=$SEEDS  $(date -u +%FT%TZ)"
python3 scripts/run_seed_sweep.py "${SEL[@]}" --seeds "$SEEDS" \
    --source mitdb --device cpu --workers 2 --threads 4 "$@" 2>&1 | tee -a "$LOG"
echo "[sweep-locked] 끝 $(date -u +%FT%TZ)"
