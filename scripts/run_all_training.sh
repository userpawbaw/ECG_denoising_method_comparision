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
RUNS="${*:-m06_l1 m08_l1 m07_l1 m10_l1 m06_l2 m06_l3 m08_l3 m08_l4 m06_l5 m06_l6 m08_l5 m08_l6}"

# 기본은 **재개 켬**이다. 이 러너를 다시 돌리는 상황은 대부분 중단 복구이고,
# 이미 목표 epoch 까지 끝난 학습은 train.py 가 알아서 건너뛴다.
# 처음부터 다시 학습하려면 RESUME=0 으로 준다.
RESUME_FLAG=""
[ "${RESUME:-1}" = "1" ] && RESUME_FLAG="--resume"

mkdir -p results/logs
for c in $RUNS; do
  echo "=============== $c  (source=$SOURCE) ==============="
  python3 scripts/train.py -c "configs/$c.yaml" --source "$SOURCE" $RESUME_FLAG \
      --workers 2 --threads 4 \
      2>&1 | tee "results/logs/train_${SOURCE}_$c.log" \
           | grep --line-buffered -E "^\[|^ep|^best|^early|^이미" || echo "FAILED: $c"
  # --line-buffered 가 없으면 grep 이 4KB 블록을 채울 때까지 출력을 붙들고 있어,
  # 학습이 정상인데도 진행 상황이 전혀 보이지 않는다 (tee 가 쓰는 로그만 갱신됨).
done
echo "ALL DONE"
