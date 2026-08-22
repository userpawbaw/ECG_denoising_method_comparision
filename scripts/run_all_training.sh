#!/usr/bin/env bash
# RQ4 비교 세트 + loss ablation. 순차 실행 (CPU 4코어 경합 방지).
set -u
cd "$(dirname "$0")/.."
for c in m06_l1 m08_l1 m07_l1 m06_l3 m08_l4; do
  echo "=============== $c ==============="
  python3 scripts/train.py -c "configs/$c.yaml" --workers 2 --threads 4 || echo "FAILED: $c"
done
echo "ALL DONE"
