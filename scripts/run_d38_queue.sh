#!/usr/bin/env bash
# D-38 큐 — E1(암기 시험) 세 판 → E2(D0 용량 sweep) 여덟 판을 **한 락 안에서** 돈다.
#
#   bash scripts/run_d38_queue.sh
#
# 락 안의 재개 명령은 **이 큐 전체**다 (O-28). 두 단계 모두 끝난 판을 건너뛰므로
# 같은 명령을 다시 실행하면 이어 간다 — E1 은 `summary.json`, E2 는 sweep 이 본다.
set -u
cd "$(dirname "$0")/.."

LOCK=results/.train.lock
mkdir -p results results/logs
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another training run holds $LOCK; abort"; exit 1
fi
echo $$ > "$LOCK/pid"
printf '%s\0' bash scripts/run_d38_queue.sh > "$LOCK/cmd"
trap 'rm -f "$LOCK/pid" "$LOCK/cmd"; rmdir "$LOCK" 2>/dev/null' EXIT

LOG=results/logs/d38_queue.log
echo "[d38] 시작 $(date -u +%FT%TZ)" | tee -a "$LOG"

# E1 — 작은 것부터. 하나가 터져도 나머지는 돈다.
for arm in m06_l1_quarter m06_l1 m06_l1_nb4; do
  python3 scripts/memorize_test.py -c "configs/$arm.yaml" --out results/ext/d38_e1 \
      --source mitdb --threads 4 2>&1 | tee -a "$LOG"
done

# E2 — D0(합성) 에서 ¼ vs full, D1 structure 판과 같은 seed
python3 scripts/run_seed_sweep.py --arms m06_l1_quarter,m06_l1 --seeds 4-7 \
    --source synthetic --out results/ext/d38_e2_d0 --device cpu --workers 2 \
    --threads 4 2>&1 | tee -a "$LOG"

echo "[d38] 끝 $(date -u +%FT%TZ)" | tee -a "$LOG"
