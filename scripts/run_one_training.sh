#!/usr/bin/env bash
# 설정 **하나**만 학습한다 — 러너 전체를 돌리지 않는 일회성 실행용.
#
#   bash scripts/run_one_training.sh mitdb m06_l1_nofe
#
# **왜 따로 있는가** — `run_all_training.sh` 의 기본 목록에 없는 설정(예:
# `m06_l1_nofe`)을 `python3 scripts/train.py ...` 로 직접 띄우면 **학습 락을
# 잡지 않는다.** 그러면 감시자(`watchdog.py`)가 "돌아야 할 학습이 있다" 는
# 사실 자체를 알 수 없어 `idle` 로 보고하고, 그 실행이 죽어도 아무도 모른다.
# 실제로 그렇게 잃었다 (O-24). 일회성 실행에도 **락은 필요하다.**
set -u
cd "$(dirname "$0")/.."

SOURCE="${1:-}"
case "$SOURCE" in
  synthetic|mitdb|auto) shift ;;
  *) echo "첫 인자는 데이터축이다: synthetic | mitdb | auto"; exit 2 ;;
esac
CFG="${1:-}"
[ -n "$CFG" ] || { echo "둘째 인자는 설정 이름이다 (configs/<이름>.yaml)"; exit 2; }
shift
[ -f "configs/$CFG.yaml" ] || { echo "configs/$CFG.yaml 이 없다"; exit 2; }

LOCK=results/.train.lock
mkdir -p results results/logs
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another training run holds $LOCK; abort"; exit 1
fi
echo $$ > "$LOCK/pid"
# **재개 명령을 락 안에 적어 둔다.** 감시자가 이 실행을 되살릴 때 무엇을
# 띄워야 하는지는 여기밖에 없다 — 러너 전체를 대신 띄우면 목록에 없는 이
# 설정은 학습되지 않고, 목록에 있는 다른 학습이 대신 돈다.
printf '%s\0' bash scripts/run_one_training.sh "$SOURCE" "$CFG" "$@" > "$LOCK/cmd"
trap 'rm -f "$LOCK/pid" "$LOCK/cmd"; rmdir "$LOCK" 2>/dev/null' EXIT

LOG="results/logs/train_${SOURCE}_$CFG.log"
echo "=============== $CFG  (source=$SOURCE) ==============="
# 기본은 **재개 켬** — 이 스크립트를 다시 부르는 상황은 대부분 중단 복구다.
python3 scripts/train.py -c "configs/$CFG.yaml" --source "$SOURCE" --resume \
    --workers 2 --threads 4 "$@" \
    2>&1 | tee "$LOG" \
         | grep --line-buffered -E "^\[|^ep|^best|^early|^이미" || echo "FAILED: $CFG"
