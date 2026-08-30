#!/usr/bin/env bash
# EXP-G (잡음 × SNR 격자) 만 돌린다.
#
#     bash scripts/run_exp_g.sh              # 두 축
#     bash scripts/run_exp_g.sh mitdb        # 한 축
#
# `run_all_experiments.sh` 에 넣지 않는다 — 전체 실행이 매번 100 분 길어진다.
# 대신 **같은 락**(`results/.exp.lock`)을 잡아 `scripts/watchdog.py` 가 이
# 실행을 볼 수 있게 한다. 락이 없으면 watchdog 은 idle 로 보고 아무것도
# 지키지 않는다.
#
# 이 파일이 저장소 안에 있는 이유: 처음에는 스크래치패드에 두고 돌렸는데,
# 컨테이너가 재시작되자 **누구도 같은 실행을 재개할 수 없는 상태**가 됐다
# (그 세션 밖에서는 명령을 알 수 없다). 재개 절차는 저장소에 있어야 한다.
set -u
cd "$(dirname "$0")/.."
LOCK=results/.exp.lock
mkdir -p results results/logs
if ! mkdir "$LOCK" 2>/dev/null; then echo "another run holds $LOCK; abort"; exit 1; fi
echo $$ > "$LOCK/pid"
trap 'rm -f "$LOCK/pid"; rmdir "$LOCK" 2>/dev/null' EXIT

for SRC in "${@:-mitdb synthetic}"; do
  echo "=== EXP-G $SRC  $(date -u +%H:%M:%S) ==="
  python3 scripts/run_exp.py -c configs/exp_g.yaml --source "$SRC" || exit 3
done
echo "DONE $(date -u +%H:%M:%S)"
