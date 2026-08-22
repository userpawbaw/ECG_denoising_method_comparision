#!/usr/bin/env bash
# RQ4 비교 세트 + loss ablation. 순차 실행 (CPU 경합 방지).
# 중복 실행 방지 락을 건다 — 두 러너가 같은 results/ 를 건드리면 서로를 망가뜨린다.
set -u
cd "$(dirname "$0")/.."
LOCK=results/.train.lock
mkdir -p results
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another training run holds $LOCK; abort"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

RUNS="${*:-m06_l1 m08_l1 m07_l1 m06_l3 m08_l4}"
for c in $RUNS; do
  echo "=============== $c ==============="
  python3 scripts/train.py -c "configs/$c.yaml" --workers 2 --threads 4 \
      2>&1 | tee "results/train_$c.log" | grep -E "^\[|^ep|^best|^early" || echo "FAILED: $c"
done
echo "ALL DONE"
