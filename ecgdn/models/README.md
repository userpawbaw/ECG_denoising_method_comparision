# 모델 설계 근거

> `python scripts/model_summary.py` 로 재생성 가능한 수치를 요약한다.

## 수용영역(receptive field) 계산

출력 한 샘플이 실제로 참조하는 입력 범위:

```
RF = 1 + Σ_layers (k - 1) * Π(이전 stride)
```

`fs = 250 Hz`, window `1024 samples = 4.096 s` 기준 설계 목표는
**0.5 s 이상, 이상적으로 1~2 s** 였다 (docs/02_procedure.md STEP 17).

| 모델 | chs | k | params | RF (samples) | RF (s) |
|---|---|---|---|---|---|
| `M06` ResUNet1D | (24,32,48,64,96) | 9 (stem 15) | 976,489 | 887 | **3.55** |
| `M08` WaveletSubbandUNet | 동일 backbone + SWT 전단 | 9 | ~1.0 M | 887 + SWT 지원폭 | > 3.55 |
| `M10` DilatedResNet1D | 95 (일정) | 9 (stem 15) | 979,926 | 895 | 3.58 |

> **이 RF 는 컨볼루션 경로의 수용영역이다.** 세 모델 모두 GroupNorm 을 쓰는데,
> GroupNorm 은 시간축 전체의 평균·분산을 쓰므로 **통계를 통한 약한 결합은
> window 전체에 걸친다.** 실측으로 gradient 의존 범위를 재면 어느 모델이든
> 1024(=window)가 나온다. 설계 수치로 의미가 있는 것은 컨볼루션 경로 쪽이고,
> 그것이 위 표의 값이다 (`tests/test_dilated_resnet.py` 가 계산식을 검증한다).

**`M10` 은 `M06` 과 params·RF 를 1 % 안에서 맞췄다.** 차이는 시간 해상도
하나뿐이다 — `M06` 은 1/2 씩 4회 줄였다 복원하고 `M10` 은 끝까지 유지한다.
그 대가로 추론이 **2.25 배** 비싸다 (full resolution 유지).

RF 가 3.55 s 로 목표를 넘는다. 이는 window(4.096 s) 거의 전체를 보므로
**여러 beat 의 리듬 맥락**까지 사용할 수 있다는 뜻이다. 다만 window 를 넘지는 않으므로
경계에서 정보가 잘리지 않는다 — 이 균형이 window 1024 를 고른 이유다.

## 왜 residual (잡음 예측) 인가

네트워크는 clean ECG 가 아니라 잡음 `n̂` 을 예측하고 `x̂ = y − n̂` 을 만든다.

1. ECG 전체를 다시 그릴 필요가 없어 학습이 쉽다.
2. 진폭 편향(`gain_bias`, docs/00_review.md A-1)이 잘 생기지 않는다 — 출력 대부분이 입력에서 온다.
3. **hallucination 경향이 줄어든다.** head 를 0 으로 초기화해 학습 시작 시점의 출력이 정확히 `x̂ = y`
   (identity) 가 되도록 했다.

## 왜 GroupNorm 인가

ECG window 는 환자/구간마다 진폭 분포가 크게 다르다. BatchNorm 의 batch 통계는
그 때문에 불안정하고, 추론 시 batch 크기에도 의존하게 된다. GroupNorm 은 샘플별로
정규화하므로 이 문제가 없다.

## 왜 stride-conv 다운샘플 / 보간 업샘플인가

- max-pool 은 위치 정보를 버린다. R-peak timing 보존이 핵심 지표이므로 stride-conv 를 쓴다.
- transposed conv 는 체커보드 아티팩트를 만든다. 선형 보간 + conv 가 파형에 안전하다.
