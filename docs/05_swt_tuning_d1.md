# 05. SWT thresholding 파라미터 튜닝

> 자동 생성: `python scripts/tune_swt.py`

## 방법론

- 탐색: 합성 ECG seed [0, 1, 2, 3]
- 검증: seed [4, 5, 6, 7] (탐색에 쓰지 않음)
- 조건: 잡음 7 종 × SNR [5, 10, 15] dB, 지표 `snr_imp_scaled`
- **MIT-BIH 확보 후에는 TRAIN split 으로 다시 돌려야 한다.**

## 결과

| 설정 | 값 |
|---|---|
| `sigma_source` | `d1` |
| `mode` | `soft` |
| `protect_qrs` | `True` |
| `k` (D1..D5) | `(2.5, 2.5, 0.6, 0.2, 0.1)` |

| 조건 | mean `snr_imp_scaled` [dB] | worst-case [dB] |
|---|---|---|
| 튜닝 세트 | -0.49 | -11.00 |
| **검증 세트(미사용 seed)** | **-0.55** | -12.11 |
| 교과서 기본값 (level MAD + soft + k=1) | -4.67 | -13.59 |

## 해석

- 검증 세트에서 -0.55 dB vs 교과서 기본값 -4.67 dB → **차이 +4.12 dB**.
- `sigma_source` 가 가장 큰 요인이다. level 별 MAD 는 D3~D5 에서 ECG 자체를 재기 때문에
  sigma 를 2~5 배 과대추정하고 QRS 대역을 잘라낸다.
- `k` 가 D1/D2 에서 크고 D3~D5 에서 작게 수렴한 것은 실측 band SNR
  (D1 −13 dB, D2 +0.5, D3 +12, D4 +17, D5 +18)과 정확히 일치하는 방향이다.
- soft 보다 garrote/hard 가 낫다. soft 는 살아남은 계수에서도 λ 만큼을 빼서
  진폭이 체계적으로 줄어들기 때문이다(`gain_bias` 로 확인 가능).

## 재현

```bash
python scripts/tune_swt.py
```
