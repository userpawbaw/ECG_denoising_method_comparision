# 11. 손실 변형의 잡음 종류별 효과 (EXP-G)

> 이 문서는 `scripts/analyze_loss_by_noise.py` 가 만든다. **직접 고치지 말 것.**
> 근거: `results/{d0,d1}/exp_g/metrics.parquet`, 설정 `configs/exp_g.yaml`.

`abl_loss` 는 혼합 잡음 하나만 다룬다. 혼합은 여러 성분의 평균이라
**어디서 벌고 어디서 잃는지**를 평균 하나로는 알 수 없다. 여기서 잡음
7 종 × 입력 SNR 3 단계로 다시 잰다.

짝지은 비교다 — 같은 기록·같은 구간·같은 잡음 실현에서 `L1` 과 `L6` 을
맞댄다. Holm 보정은 **한 축·한 SNR 안의 잡음 7 종**에 걸었다.
`snr_imp_scaled` 는 floor 가 정의되지 않은 지표라 분해능 열이 없다 —
**누락이 아니라 부재**다.

## d0

`results/d0/exp_g/metrics.parquet` 가 없다 — EXP-G 미실행.


## d1

`results/d1/exp_g/metrics.parquet` 가 없다 — EXP-G 미실행.

