#!/usr/bin/env bash
# D-39 큐 — E2′(D0 + 참조 잡음) 8 판 → E3′(L6 아래 ¼) 4 판 → E5(nb4 LR) 6 판.
#
#   bash scripts/run_d39_queue.sh
#
# 락 안의 재개 명령은 **이 큐 전체**다 (O-28). sweep 은 끝난 판을 건너뛰고 끊긴 판은
# `last.pt` 에서 잇는다 — 하네스가 약 30 분에 끊어도(O-37) 같은 명령을 다시 실행하면 된다.
set -u
cd "$(dirname "$0")/.."

LOCK=results/.train.lock
mkdir -p results results/logs
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another training run holds $LOCK; abort"; exit 1
fi
echo $$ > "$LOCK/pid"
printf '%s\0' bash scripts/run_d39_queue.sh > "$LOCK/cmd"
trap 'rm -f "$LOCK/pid" "$LOCK/cmd"; rmdir "$LOCK" 2>/dev/null' EXIT

LOG=results/logs/d39_queue.log
SW=(python3 scripts/run_seed_sweep.py --device cpu --workers 2 --threads 4)
echo "[d39] 시작 $(date -u +%FT%TZ)" | tee -a "$LOG"

# E2′ — 참조 순도만 바꾼다 (D0, E2 와 같은 seed)
"${SW[@]}" --arms m06_l1_quarter_rn20,m06_l1_rn20 --seeds 4-7 --source synthetic \
    --out results/ext/d39_e2p 2>&1 | tee -a "$LOG"
# E3′ — L6 아래 ¼ (full 은 results/ext/structure 의 m06_l6)
"${SW[@]}" --arms m06_l6_quarter --seeds 4-7 --source mitdb \
    --out results/ext/d39_e3p 2>&1 | tee -a "$LOG"
# E5 — nb4 의 LR (1e-3 은 results/ext/blocks)
"${SW[@]}" --arms m06_l1_nb4_lr03,m06_l1_nb4_lr3 --seeds 3-5 --source mitdb \
    --out results/ext/d39_e5 2>&1 | tee -a "$LOG"

echo "[d39] 끝 $(date -u +%FT%TZ)" | tee -a "$LOG"
