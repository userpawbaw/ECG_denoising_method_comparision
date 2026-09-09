#!/usr/bin/env bash
# D-23 의 여덟 판을 **B -> A -> C 순서로** 하나씩 돌린다.
#
#   bash scripts/run_d23_queue.sh
#
# **왜 큐 스크립트가 따로 있는가** — 여덟 판을 `&&` 로 이으면 하나가 실패할 때
# 나머지가 통째로 날아가고, 컨테이너가 죽으면 감시자가 **그 순간 도는 한 판만**
# 되살린다 (락에 적힌 재개 명령이 그 한 판이다). 큐를 파일로 두고 **매번 처음부터
# 훑되 끝난 판은 건너뛰게** 하면, 어디서 끊겨도 이 스크립트를 다시 부르는 것만으로
# 이어진다 — 「재개는 가장 긴 단일 실행보다 촘촘해야 한다」 (O-19).
#
# 끝난 판을 건너뛰는 것은 `train.py --resume` 이 스스로 한다: early stopping
# 조건을 이미 만족했거나 epoch 상한에 닿았으면 «재학습하지 않는다» 하고 빠진다.
# 그래서 별도 상태 파일을 두지 않는다 — 상태를 두 곳에 두면 한쪽이 낡는다.
set -u
cd "$(dirname "$0")/.."

# (단계, 설정) — 순서가 곧 D-23 의 근거다. B 가 40 분에 「키우는 쪽」을 닫고,
# A 가 B·C 를 읽는 자를 만들고, C 가 데이터를 겨눈다.
QUEUE=(
  "B m06_l1_quarter"
  "B m06_l1_half"
  "A m06_l1_s1"
  "A m06_l1_s2"
  "A m09_l1_s1"
  "A m09_l1_s2"
  "C m06_l1_data4x_eq"
  "C m06_l1_data4x"
)

mkdir -p results/logs
STAMP=$(date +%m%d_%H%M)
SUM="results/logs/d23_queue_$STAMP.log"
echo "[d23] 큐 ${#QUEUE[@]} 판 시작 $(date -u +%FT%TZ)" | tee -a "$SUM"

for item in "${QUEUE[@]}"; do
  STAGE="${item%% *}"; CFG="${item#* }"
  echo "[d23] --- $STAGE · $CFG  $(date -u +%FT%TZ)" | tee -a "$SUM"
  # 락에 **큐 재개 명령**을 적게 한다 — 감시자가 판 하나만 되살리면 나머지가
  # 사라진다. 실제로 그렇게 잃었다 (O-28).
  if RESUME_CMD="bash scripts/run_d23_queue.sh" \
       bash scripts/run_one_training.sh mitdb "$CFG" >>"$SUM" 2>&1; then
    echo "[d23] ok   $CFG" | tee -a "$SUM"
  else
    # 실패해도 **큐를 멈추지 않는다** — 한 판이 죽었다고 나머지를 잃으면
    # 여덟 판을 다시 재는 꼴이 된다. 실패는 로그에 남고 마지막에 센다.
    echo "[d23] FAIL $CFG (계속 진행)" | tee -a "$SUM"
  fi
done

echo "[d23] 큐 종료 $(date -u +%FT%TZ)" | tee -a "$SUM"
grep -c '^\[d23\] FAIL' "$SUM" | xargs -I{} echo "[d23] 실패 {} 건" | tee -a "$SUM"
echo "[d23] DONE" | tee -a "$SUM"
