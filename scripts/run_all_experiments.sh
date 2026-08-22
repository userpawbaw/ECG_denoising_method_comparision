#!/usr/bin/env bash
# 학습 완료 후 전체 실험 + 리포트. 순차 실행.
set -u
cd "$(dirname "$0")/.."
LOCK=results/.exp.lock
mkdir -p results
if ! mkdir "$LOCK" 2>/dev/null; then echo "another run holds $LOCK; abort"; exit 1; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

for c in exp_c exp_a exp_b; do
  echo "=============== $c ==============="
  python3 scripts/run_exp.py -c "configs/$c.yaml" 2>&1 | tee "results/run_$c.log" | tail -20
done
echo "=============== exp_e (safety probe) ==============="
python3 scripts/run_safety_probe.py 2>&1 | tee results/run_exp_e.log | tail -5
echo "=============== SNR 추정기 교정표 재생성 ==============="
python3 scripts/check_snr_estimator.py 2>&1 | tail -5
echo "=============== report ==============="
python3 scripts/make_report.py --fig-snr 5 2>&1 | tail -5
echo "ALL EXPERIMENTS DONE"
