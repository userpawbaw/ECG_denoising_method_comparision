#!/usr/bin/env bash
# 전체 학습. 순차 실행 (CPU 경합 방지).
#
#   bash scripts/run_all_training.sh synthetic            # D0 전체
#   bash scripts/run_all_training.sh mitdb                # D1 전체
#   bash scripts/run_all_training.sh synthetic m07_l1     # 일부만
#
# **데이터축을 첫 인자로 반드시 준다.** 생략하면 auto 가 되는데, auto 는
# data/raw 의 상태에 따라 D0/D1 이 조용히 바뀐다 (docs/99_status.md 2.1).
set -u
cd "$(dirname "$0")/.."

SOURCE="${1:-auto}"
case "$SOURCE" in
  synthetic|mitdb|auto) shift || true ;;
  *) echo "첫 인자는 데이터축이다: synthetic | mitdb | auto"; exit 2 ;;
esac

LOCK=results/.train.lock
mkdir -p results
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another training run holds $LOCK; abort"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

# 기본값 = 실험이 실제로 쓰는 체크포인트 전부.
# loss ablation(configs/abl_loss.yaml)은 **모든 loss 를 같은 조건에서** 학습해야
# 성립하므로, l2/l3/l4 를 빠뜨리면 표가 조용히 거짓이 된다 (docs/02 F-9).
RUNS="${*:-m06_l1 m08_l1 m07_l1 m06_l2 m06_l3 m08_l3 m08_l4}"
mkdir -p results/logs
for c in $RUNS; do
  echo "=============== $c  (source=$SOURCE) ==============="
  python3 scripts/train.py -c "configs/$c.yaml" --source "$SOURCE" \
      --workers 2 --threads 4 \
      2>&1 | tee "results/logs/train_${SOURCE}_$c.log" \
           | grep -E "^\[|^ep|^best|^early" || echo "FAILED: $c"
done
echo "ALL DONE"
