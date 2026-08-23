# 03. Metric noise floor

> 자동 생성: `python scripts/measure_metric_floor.py`  (수정하지 말 것 — 스크립트를 고칠 것)

합성 ECG 5 기록 × 120 s 에 **40 dB AWGN**(거의 무시할 수준의 교란)을 10 개 seed 로 준 뒤,
각 지표가 이상값에서 얼마나 흔들리는지 측정했다. 평가 guard = 5 s.

**결과표에서 방법 간 차이가 `floor_p95` 보다 작으면 `n.s.`(구분 불가)로 표기한다.**

| metric | 이상값 | floor (mean) | floor (p95) | floor (max) |
|---|---|---|---|---|
| `rmse` | 0 | 0.001499 | 0.001504 | 0.001505 |
| `prdn` | 0 | 1 | 1.004 | 1.005 |
| `cc` | 1 | 5.004e-05 | 5.041e-05 | 5.048e-05 |
| `beat_cc` | 1 | 3.122e-07 | 3.634e-07 | 3.905e-07 |
| `beat_cc_median` | 1 | 4.217e-05 | 4.965e-05 | 5.005e-05 |
| `r_amp_err_pct` | 0 | 0.1293 | 0.1432 | 0.1522 |
| `rpeak_mae_ms` | 0 | 0.0009467 | 0 | 0.02367 |
| `rpeak_bias_ms` | 0 | 0.0009467 | 0 | 0.02367 |
| `hr_err_bpm` | 0 | 0 | 0 | 0 |
| `rr_mae_ms` | 0 | 0.001905 | 0 | 0.04762 |
| `qrs_dur_err_ms` | 0 | 0.4105 | 0.6397 | 0.7194 |
| `delineate_success_rate` | — | 0 | 0 | 0 |
| `psd_logdist` | 0 | 3.632 | 6.328 | 6.379 |
| `gain_bias` | 1 | 0.0001022 | 0.0002037 | 0.0002352 |

## 해석

- **`qrs_dur_err_ms` 의 floor 는 p95 기준 0.6 ms** 다. 즉 두 방법의 QRS duration 오차가 이보다 작게 차이 나면 그것은 denoising 성능 차이가 아니라 **delineator 자체의 불안정성**이다. (docs/00_review.md A-8 에서 예고한 항목)
- `rpeak_mae_ms` / `hr_err_bpm` / `rr_mae_ms` 는 floor 가 사실상 0 이다. 타이밍 계열 지표는 해상도가 높아 신뢰할 수 있다.
- `beat_cc`, `cc` 는 1 에 붙어 있어 소수점 4~5 자리까지 봐야 한다. 표에는 `1 - cc` 형태로 적는 것을 권한다.
- **`psd_logdist` 의 floor 가 6.3 dB 로 매우 크다.** ECG 파워가 거의 없는 주파수 구간에서 log-PSD 차이가 폭발하기 때문이다. 따라서 `psd_logdist` 는 **절대값으로 해석하지 말고 방법 간 상대 비교로만** 쓴다. PSD 는 그림(F3)으로 보이는 것이 주 용도다.

## 이 표의 한계

- 여기서 쓴 것은 **합성 ECG** 다. 파형이 실제보다 규칙적이라 `qrs_dur_err_ms` 의 floor 가 낙관적으로 나올 수 있다.
  MIT-BIH 를 확보한 뒤 (STEP 15) **동일 스크립트를 `--source mitdb` 로 다시 돌려** 실데이터 floor 로 갱신할 것.
- floor 의 정의는 '이상값 대비, 40 dB 교란에서의 편차' 다. 즉 **지표의 실효 분해능**이며, 알고리즘 자체의 불안정성(delineator 실패 등)과 교란에 대한 정상적 민감도가 합쳐진 값이다.
- `snr_out_*` 는 이상값이 무한대라 이 표에 포함되지 않는다 (대신 `gain_bias` 로 안정성을 본다).

## 재현

```bash
python scripts/measure_metric_floor.py
```
