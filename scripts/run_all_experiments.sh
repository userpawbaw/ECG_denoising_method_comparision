#!/usr/bin/env bash
# 학습 완료 후 전체 실험 + 리포트. 순차 실행.
#
#   bash scripts/run_all_experiments.sh synthetic
#   bash scripts/run_all_experiments.sh mitdb
#
# 데이터축을 첫 인자로 반드시 준다 (docs/99_status.md 2.1).
set -u
cd "$(dirname "$0")/.."

SOURCE="${1:-auto}"
case "$SOURCE" in
  synthetic|mitdb|auto) ;;
  *) echo "첫 인자는 데이터축이다: synthetic | mitdb | auto"; exit 2 ;;
esac

LOCK=results/.exp.lock
mkdir -p results results/logs
if ! mkdir "$LOCK" 2>/dev/null; then echo "another run holds $LOCK; abort"; exit 1; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

# 학습이 끝났는지 먼저 확인한다. run_exp.py 는 체크포인트가 없으면 `[skip]` 한
# 줄만 찍고 그 방법을 뺀 채 표를 정상적으로 만들어 내므로, 학습이 덜 끝난 상태로
# 여기까지 오면 결과가 조용히 짧아진다 (F-9 와 같은 계열).
python3 scripts/check_ckpts.py --source "$SOURCE" || exit 2

for c in exp_c exp_a exp_b abl_loss; do
  echo "=============== $c  (source=$SOURCE) ==============="
  python3 scripts/run_exp.py -c "configs/$c.yaml" --source "$SOURCE" \
      2>&1 | tee "results/logs/run_${SOURCE}_$c.log" | tail -20
done
echo "=============== loss ablation 표 (STEP 19 DoD) ==============="
# 종료코드 2 = 체크포인트의 학습 조건이 섞여 비교가 성립하지 않음 (F-9)
python3 scripts/make_ablation_table.py --source "$SOURCE" 2>&1 | tail -5
echo "=============== exp_e (safety probe) ==============="
python3 scripts/run_safety_probe.py --source "$SOURCE" \
    2>&1 | tee "results/logs/run_${SOURCE}_exp_e.log" | tail -5
echo "=============== report ==============="
python3 scripts/make_report.py --source "$SOURCE" --fig-snr 5 2>&1 | tail -5
echo "ALL EXPERIMENTS DONE"
