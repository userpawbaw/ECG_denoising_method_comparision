# 90. 실험 결과

> 자동 생성: `python scripts/make_report.py`  (수정하지 말 것 — 스크립트를 고칠 것)

## 대표 파형

기록 `S022`, 입력 SNR 5 dB, 동일 y축.

![F1](../results/report/F1_waveforms.png)

### QRS 확대

![F2](../results/report/F2_qrs_zoom.png)

### 주파수 스펙트럼

![F3](../results/report/F3_psd.png)

PSD 는 시간영역에서 보이지 않는 것을 보여준다: 어떤 방법이 60 Hz 를 지웠는지, 어떤 방법이 QRS 의 고주파 성분까지 함께 잘라냈는지.

## 비교 대상

| ID | 분류 | 설명 |
|---|---|---|
| `M00` | baseline | Identity (no-op) |
| `M01` | classical | Bandpass 0.5-40 Hz + auto notch |
| `M02` | classical | Savitzky-Golay (40 ms, order 3) |
| `M03` | timefreq | DWT soft threshold (sym4, L5) |
| `M04` | timefreq | SWT adaptive threshold (level-k + QRS protect, garrote) |
| `M05f` | model | Sameni EKF (forward only) |
| `M05` | model | Sameni EKS (EKF + RTS smoother) |
| `B01` | bound | Oracle wavelet threshold (wavelet upper bound) |
| `B02` | bound | Oracle Wiener (LTI upper bound) |

## 학습 곡선

![F8](../results/report/F8_training.png)

| run | best epoch | best val snr_imp_scaled [dB] |
|---|---|---|
| `m06_l1` | 21 | 18.60 |

