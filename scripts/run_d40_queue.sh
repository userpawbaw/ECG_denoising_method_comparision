#!/usr/bin/env bash
# D-39 + D-40 큐 — D-39 세 단계를 마저 돌고, 이어서 D-40 세 단계를 돈다.
#
#   bash scripts/run_d40_queue.sh
#
# `run_d39_queue.sh` 의 **상위 집합**이다. 돌고 있는 셸 스크립트를 고치면 bash 가
# 읽던 자리가 어긋나므로 그 파일은 그대로 두고, 다음 재개부터 이것을 띄운다.
# sweep 은 끝난 판을 건너뛰고 끊긴 판은 `last.pt` 에서 잇는다 (O-37 · O-28).
set -u
cd "$(dirname "$0")/.."

LOCK=results/.train.lock
mkdir -p results results/logs
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another training run holds $LOCK; abort"; exit 1
fi
echo $$ > "$LOCK/pid"
printf '%s\0' bash scripts/run_d40_queue.sh > "$LOCK/cmd"
trap 'rm -f "$LOCK/pid" "$LOCK/cmd"; rmdir "$LOCK" 2>/dev/null' EXIT

LOG=results/logs/d40_queue.log
SW=(python3 scripts/run_seed_sweep.py --device cpu --workers 2 --threads 4)
echo "[d40] 시작 $(date -u +%FT%TZ)" | tee -a "$LOG"

# ---- D-39 (이미 끝난 판은 건너뛴다)
"${SW[@]}" --arms m06_l1_quarter_rn20,m06_l1_rn20 --seeds 4-7 --source synthetic \
    --out results/ext/d39_e2p 2>&1 | tee -a "$LOG"
"${SW[@]}" --arms m06_l6_quarter --seeds 4-7 --source mitdb \
    --out results/ext/d39_e3p 2>&1 | tee -a "$LOG"
"${SW[@]}" --arms m06_l1_nb4_lr03,m06_l1_nb4_lr3 --seeds 3-5 --source mitdb \
    --out results/ext/d39_e5 2>&1 | tee -a "$LOG"

# ---- D-40
# 참조 — clean 통과 지표의 기준값 + 환경 재현 확인 (l6cond 와 같은 seed)
"${SW[@]}" --arms m06_l1,m06_l6 --seeds 0-1 --source mitdb --clean-pass \
    --out results/ext/d40_ref 2>&1 | tee -a "$LOG"
# E6 (L1C) · E7 (잔차 끔) — 잔차를 켠 L1 · L6 칸은 results/ext/l6cond
"${SW[@]}" --arms m06_l1c,m06_l1_nores,m06_l6_nores --seeds 0-3 --source mitdb \
    --clean-pass --out results/ext/d40 2>&1 | tee -a "$LOG"

echo "[d40] 끝 $(date -u +%FT%TZ)" | tee -a "$LOG"
