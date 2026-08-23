# ECG Denoising Method Comparison

아날로그로 취득한 ECG의 디지털 잡음 제거 기법 비교 연구.
**고전 DSP · 모델 기반(Kalman) · 딥러닝 · 하이브리드**를 동일 조건에서 정량 비교한다.

> **범위**: ECG morphology를 보존하면서 acquisition noise를 줄이는 **연구용 신호처리 시스템**이다.
> 임상 진단용 의료기기가 아니며, 진단 목적으로 사용해서는 안 된다.

---

## 문서 (읽는 순서)

| 문서 | 내용 |
|---|---|
| [`docs/00_review.md`](docs/00_review.md) | 착수 전 제안에 대한 검토 — 지적 11 / 발전 6 / 추가 8 항목 |
| [`docs/01_design.md`](docs/01_design.md) | **확정 설계안** — 신호 규격, 데이터, 기법 11종, 평가 체계, 실험 매트릭스 |
| [`docs/02_procedure.md`](docs/02_procedure.md) | **구현 절차서** — STEP 00~30, 각 단계에 DoD와 검증 명령 |
| [`docs/03_metric_floor.md`](docs/03_metric_floor.md) | 지표 분해능 (결과표의 "구분 불가" 판정 기준) |
| [`docs/04_snr_estimator_calibration.md`](docs/04_snr_estimator_calibration.md) | ground-truth 없는 SNR 추정기의 편향표 |
| [`docs/05_swt_tuning.md`](docs/05_swt_tuning.md) | SWT threshold 파라미터 탐색 (탐색/검증 분리) |
| [`docs/06_sameni_diagnosis.md`](docs/06_sameni_diagnosis.md) | Sameni EKF/EKS 자가진단 6항목 |
| [`docs/07_safety_probe_d0.md`](docs/07_safety_probe_d0.md) | hallucination / 부정맥 훼손 검증 |
| [`docs/08_acquisition.md`](docs/08_acquisition.md) | 실측 데이터 수집 프로토콜 (S1~S6) |
| [`docs/90_results_d0.md`](docs/90_results_d0.md) | 실험 결과 (자동 생성) |
| [`docs/91_report.md`](docs/91_report.md) | **종합 보고서** — 설계 논리, 지표 선택 근거, 결과 해석 |

---

## 빠른 시작

```bash
pip install -r requirements.txt

# 외부 데이터 없이 전부 검증된다 (합성 ECG 사용)
make check-nodata
```

`make check-nodata` 가 하는 일:

| 명령 | 검증 내용 |
|---|---|
| `pytest` | 122개 단위/회귀 테스트 |
| `check_synthetic.py` | 합성 ECG: ODE vs 위상영역 일치, R-peak 100 % |
| `check_noise.py` | 잡음 6종의 PSD 대역 |
| `check_snr_estimator.py` | SNR 추정기 교정표 생성 |
| `diagnose_sameni.py` | Sameni 구현 6항목 자가진단 |

### 실제 데이터로 진행

```bash
# 1) MIT-BIH + NSTDB (로컬에서 실행 — 원격 환경에서는 physionet 이 막힐 수 있다)
python scripts/download_data.py --db mitdb --db nstdb

# 2) SWT 재탐색 (TRAIN split 기준)
python scripts/tune_swt.py

# 3) 딥러닝 학습
bash scripts/run_all_training.sh

# 4) 실험
python scripts/run_exp.py -c configs/exp_a.yaml     # SNR sweep
python scripts/run_exp.py -c configs/exp_b.yaml     # 잡음 종류별
python scripts/run_exp.py -c configs/exp_c.yaml     # distortion floor
python scripts/run_safety_probe.py                  # 안전성 프로브

# 5) 표/그림 생성
python scripts/make_report.py
```

---

## 비교 대상

| ID | 분류 | 방법 |
|---|---|---|
| `M00` | — | Identity (하한선) |
| `M01`/`M01d` | classical | Bandpass 0.5–40 / 0.5–100 Hz + 자동 notch |
| `M02` | classical | Savitzky-Golay |
| `M03` | time-freq | DWT soft threshold (대조군) |
| `M04` | time-freq | **SWT adaptive threshold** (level별 k + QRS 보호 + garrote) |
| `M05`/`M05f` | model-based | **Sameni EKS / EKF** |
| `M06` | deep | **Residual 1D U-Net** |
| `M07` | hybrid | SWT → Residual U-Net (순차) |
| `M08` | hybrid | **Wavelet-subband Residual U-Net** (표현공간) |
| `B01` | bound | Oracle wavelet threshold — wavelet 계열의 **상한** |
| `B02` | bound | Oracle Wiener — 선형 시불변 필터의 **상한** |

상한(B01/B02)이 있어야 "M04가 더 튜닝될 여지가 있는가", "비선형 처리가 실제로 필요한가"에
정량적으로 답할 수 있다.

---

## 이 프로젝트가 다르게 하는 것

1. **평가를 기법보다 먼저 만든다.** 나중에 만들면 무의식적으로 자기 기법에 유리한 평가를 만들게 된다.
2. **SNR을 세 가지로 나눠 본다.** `strict`(그대로 써도 되는가) / `scaled`(파형 구조가 맞는가) /
   `gain_bias`(진폭이 얼마나 눌렸는가). soft-threshold와 MSE 학습은 구조적으로 진폭을 줄이므로,
   이 분리 없이는 방법 간 순위가 뒤집힌다.
3. **지표 자체의 분해능을 먼저 잰다** (`docs/03_metric_floor.md`).
   그 값보다 작은 차이는 결과표에 `*`로 표시해 "구분 불가"임을 명시한다.
4. **성능 상한을 함께 그린다.** oracle wavelet / oracle Wiener.
5. **"해를 끼치는가"를 따로 잰다** (EXP-C distortion floor): 잡음이 없는 신호를 통과시켰을 때의 출력 SNR.
6. **hallucination을 실측한다** (EXP-E): beat를 지우고 모델이 만들어내는지, 부정맥을 훼손하는지.
7. **baseline을 약하게 두지 않는다.** Sameni EKF/EKS는 6항목 자가진단을 통과해야 비교에 들어간다.
   (실제로 이 절차가 성능을 10 dB 이상 바꾸는 구현 버그 2건을 찾아냈다.)

---

## 구조

```
ecgdn/
├─ config.py            신호 규격 단일 진실 원천 (fs=250, win=1024, hop=512, ...)
├─ utils.py             결정론적 seed 유도, 파워 정의 단일화
├─ data/                synthetic · noise · mixer · windows · mitdb · nstdb · arduino · dataset
├─ methods/             base(계약) · frontend · bandpass · savgol · wavelet · kalman_sameni ·
│                       oracle(bound) · dl_wrapper
├─ models/              blocks · resunet1d · swt_torch(미분가능) · wavelet_unet · losses
├─ eval/                signal_metrics · rpeak · morphology · spectral · snr_estimation ·
│                       engine · stats
└─ viz/plots.py

scripts/                실행 진입점 (모든 산출물은 여기서만 생성)
configs/                실험 1개 = yaml 1개
hardware/               Arduino 스케치
tests/                  pytest
```

**모든 denoiser는 하나의 계약을 따른다**:

```python
x_hat = denoiser(y, fs, ctx=None)      # 길이·스케일 보존
```

덕분에 기법을 추가해도 실험 스크립트를 고칠 필요가 없다.
`ctx['x_clean']`은 oracle 계열만 받을 수 있고, 그 경우 이름이 `oracle_`로 시작해야 한다
(`BaseDenoiser`가 강제 검사한다).
