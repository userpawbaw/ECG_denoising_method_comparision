# 01. 확정 설계안

> 본 문서는 `00_review.md` 의 검토 결과를 반영한 **최종 확정안**이다.
> 이후 모든 코드는 이 문서를 규격으로 삼는다. 변경이 필요하면 이 문서를 먼저 고친다.

---

## 1. 프로젝트 정의

### 1.1 제목
**아날로그 취득 ECG의 디지털 잡음 제거: 고전 DSP · 모델 기반 · 딥러닝 · 하이브리드 기법의 비교 연구**

### 1.2 목표 (범위를 명확히 제한한다)

> **ECG morphology를 보존하면서 acquisition noise를 감소시키는 디지털 후처리 기법들을
> 동일한 조건에서 정량 비교하고, 실제 자체 제작 취득 시스템의 잡음 조건에서 어떤 기법이 적합한지 규명한다.**

**의도적으로 하지 않는 주장:**
- "임상 진단 가능한 병원급 ECG를 복원한다" (X) — 검증 불가능하고 규제 대상 주장이다.
- "의료기기를 개발했다" (X) — 본 과제는 **연구용 신호처리 시스템**이다.

### 1.3 연구 질문 (RQ)

| RQ | 질문 | 답하는 실험 |
|---|---|---|
| RQ1 | 입력 SNR에 따라 어떤 기법이 유리한가? 특히 **실측 조건(≈15 dB 부근)** 에서는? | EXP-A |
| RQ2 | 잡음 **종류별**(BW/MA/EM/PLI/AWGN)로 기법의 강약이 갈리는가? | EXP-B |
| RQ3 | 각 기법은 **깨끗한 신호에 해를 끼치는가**? (noise 제거 vs morphology 보존의 trade-off) | EXP-C |
| RQ4 | wavelet을 **전처리로 쓰는 것**과 **신경망의 표현공간으로 쓰는 것** 중 무엇이 효과적인가? | EXP-D |
| RQ5 | 딥러닝이 **없는 파형을 만들어내지는(hallucination) 않는가**? 부정맥 beat를 훼손하지 않는가? | EXP-E |
| RQ6 | 공개 데이터로 학습한 모델이 **자체 취득 장비**에 전이되는가? device-adaptation이 필요한가? | EXP-F |

RQ3·RQ5는 대화 원안에 없던 것이며, 이 과제를 단순 벤치마크에서 분리시키는 핵심이다.

---

## 2. 신호 규격 (전 파이프라인 공통)

| 항목 | 값 | 근거 |
|---|---|---|
| 샘플링 주파수 `fs` | **250 Hz** | 125 Hz Nyquist. ECG 진단 대역(≤150 Hz) 전체는 못 담지만, 취득계가 이미 그 이상을 담지 못한다. 계산량 대비 최적 |
| window 길이 | **1024 samples = 4.096 s** | 정상 심박에서 4~6 beat 포함. `2^10` 이라 SWT level 5 안전 |
| hop | **512 samples (50 % overlap)** | Hann√ COLA 재구성 |
| 진폭 단위 | **mV** (MIT-BIH는 gain/baseline으로 물리단위 변환) | R-peak amplitude 지표를 물리적으로 해석 가능 |
| dtype | `float32` (저장), `float64` (metric 계산) | metric에서 정밀도 손실 방지 |

### 2.0 평가 구간 규약 (guard band) — **실측으로 확정**

0.5 Hz zero-phase HPF는 경계에서 수 초간 링잉하고, Kalman은 수렴 시간이 필요하며,
wavelet/OLA는 패딩 경계 효과가 있다. 이를 방치하면 **깨끗한 신호에 front-end만 걸어도
왜곡 11.9 dB**가 나온다(측정값).

| guard [s] | front-end distortion [dB] |
|---|---|
| 0 | 11.5 |
| 1 | 25.9 |
| 2 | 35.2 |
| 3 | 40.9 |
| **5** | **41.9** (포화) |
| 10 | 41.7 |

→ **확정**: 평가 단위 구간 `EVAL_SEG_S = 60 s`, 양끝 `EVAL_GUARD_S = 5 s`를 지표 계산에서 제외.
**방법에는 전체 구간을 주고, 지표만 내부 구간에서 계산한다.** 모든 방법에 동일 적용.

### 2.1 공통 front-end (모든 방법에 동일 적용)

```
raw
 └─ [FE1] zero-phase Butterworth HPF, 4차, fc = 0.5 Hz      (baseline wander)
 └─ [FE2] zero-phase Butterworth LPF, 4차, fc = 100 Hz      (anti-alias 여유 / 대역 제한)
 └─ [FE3] IIR notch, Q = 30, f0 = 60 Hz + 120 Hz            (조건부: PSD 검사 통과 시에만)
```
- `filtfilt` (zero-phase) 사용 → 위상 왜곡 없음. R-peak timing이 밀리지 않는다.
- **[FE3]은 무조건 적용하지 않는다.** `psd_has_pli()` 로 60 Hz 피크의 존재를 판정한 뒤 적용한다.
  판정 기준: 59–61 Hz 대역 power가 인접 배경대역(55–58, 62–65 Hz) 중앙값의 **10 배 이상**.

> **적용 방식 (오해하기 쉬운 지점)**: 별도의 전역 front-end 단계가 파이프라인에 존재하는 것이 아니다.
> **각 방법이 자기 안에서 front-end를 정확히 1회 적용**한다 (`use_frontend=True`).
> `M01`/`M01d`/`M_FE`는 front-end 그 자체가 방법의 정의이므로 항상 켜져 있다.
> 따라서 notch가 중복 적용되는 경로는 없다.
>
> **딥러닝도 동일하게 front-end를 받는다.** 학습 데이터셋과 추론 래퍼 양쪽에 적용된다.
> 이 경로가 없으면 딥러닝만 기저선 제거를 처음부터 학습해야 해 비교가 불공정해진다
> (실측: baseline wander 조건에서 front-end 단독 **+20.3 dB** vs front-end 없는 U-Net **+6.3 dB**).
> 학습 시에는 창 양쪽에 **실제 이웃 구간을 `EVAL_GUARD_S` 만큼 붙여** 필터링한 뒤 가운데만 잘라낸다 —
> 0.5 Hz HPF는 수 초간 링잉하므로 4.096 s 창에 그대로 걸면 창 전체가 트랜지언트가 된다.

### 2.1.1 `frontend` 스위치와 주/부 조건

| 조건 | 설정 | 역할 |
|---|---|---|
| **`frontend: true`** (주) | 모든 방법 + 딥러닝에 front-end 적용 | **실사용 시나리오.** 실측 장비에서는 front-end를 당연히 쓴다 |
| `frontend: false` (부록) | 끌 수 있는 모든 방법에서 해제 | 알고리즘 자체의 비교 |

`configs/exp_a.yaml`(주) / `configs/exp_a_nofe.yaml`(부록). **주 결과표는 `frontend: true`다.**

**`M_FE`(front-end만)를 모든 결과표의 필수 행으로 넣는다.** 그래야 각 방법의 이득 중
얼마가 front-end에서 온 것인지 표에서 바로 읽힌다. (이 행이 없어서
"`M04`가 baseline wander에서 oracle과 동일"이라는 오독이 실제로 발생했다 — 실은
`M04` 20.26 dB = `M_FE` 20.26 dB로, SWT thresholding의 기여가 정확히 0이었다.
baseline wander는 A5에 있고 우리는 A5를 thresholding하지 않기 때문이다.)

---

## 3. 데이터셋

### 3.1 D0 — Synthetic ECG (McSharry ODE)  *[의존성 없음, 항상 사용 가능]*

- 용도: **파이프라인 검증**, **Sameni EKF 정답 검증**, 단위테스트
- 생성 파라미터를 알고 있으므로 EKF가 이론 최적에 도달하는지 확인 가능
- HR, HRV, beat 형태를 통제 가능

### 3.2 D1 — MIT-BIH Arrhythmia (`mitdb`) + Noise Stress Test (`nstdb`)  *[주 학습·평가]*

| | |
|---|---|
| 원본 | 48 records × 30 min, 2 ch, 360 Hz, 11-bit |
| 사용 채널 | **MLII (lead II)** 우선. 없는 record는 첫 번째 채널 |
| 리샘플 | 360 → 250 Hz (polyphase, `scipy.signal.resample_poly(25, 36)`) |
| 잡음 | `nstdb` 의 `bw`, `ma`, `em` (각 30 min, 360 Hz → 250 Hz) + 합성 PLI + AWGN |

**record split (고정, 변경 금지) — SSOT 는 `ecgdn/data/splits.py`:**

DS1/DS2 는 de Chazal 등이 제안해 널리 인용되는 **inter-patient** 분할이다.
paced beat 기록(102, 104, 107, 217)은 morphology 가 근본적으로 달라 주 실험에서 분리하고
보조 분석에만 쓴다.

```
TRAIN (18): 101 106 108 109 112 114 115 116 118 119 122 124 201 203 205 207 209 215
VAL   (4): 208 220 223 230
TEST  (22): 100 103 105 111 113 117 121 123 200 202 210 212 213 214 219 221 222 228 231 232 233 234
PACED (4): 102 104 107 217     <- 주 실험 제외
```

`splits.check_split()` 이 import 시점에 교집합·누락을 자동 검증한다.

**noise split (시간축 disjoint, A-7):**
```
bw/ma/em 각각:  [0%,60%) TRAIN   [60%,75%) VAL   [75%,100%] TEST
```

### 3.3 D2 — Arduino noise-only + MIT-BIH clean  *[device adaptation]*

`B-4` 프로토콜(S2~S5)로 수집한 실측 잡음을 MIT-BIH clean ECG와 합성.

> **leakage 규약 (강제)**: fine-tuning에 쓸 수 있는 clean ECG는 **TRAIN split뿐**이다.
> TEST record의 morphology를 adaptation에 쓰면 F-8과 같은 종류의 leakage다.
> `splits.assert_adaptation_records()` 가 코드에서 검사한다.

### 3.4 D3 — Arduino 실측 ECG  *[최종 real-world 평가, ground truth 없음]*

`B-4` 프로토콜 S1, S6.

### 3.5 SNR 통제 합성 규약

```
x : clean (mV),  n : noise
n' = n * sqrt( P(x) / (P(n) * 10^(SNR/10)) )
y  = x + n'
```
- `P(·)` 는 **평균 제거 후** 파워 (`var`).
- SNR 그리드: **`{-5, 0, 5, 10, 15, 20} dB`**
- 학습 시에는 `U(-5, 20) dB` 연속 샘플링 (on-the-fly)
- 평가 시에는 **고정 시드**로 생성해 모든 방법이 **완전히 동일한 `y`** 를 받는다.
- **SNR sweep 규약**: 하나의 (record, segment, noise condition)에 대해 잡음 실현 `n`을
  **한 번만** 뽑고 SNR로 **스케일만** 바꾼다. SNR마다 다른 잡음을 뽑으면 성능-입력SNR 곡선이
  '잡음 조성 변화'와 뒤섞여 해석할 수 없게 된다.

---

## 4. 비교 대상 기법 (9개 + 2 bound)

| ID | 이름 | 분류 | 역할 |
|---|---|---|---|
| `M00` | Identity (no-op) | — | **하한선** |
| `M01` | Bandpass + Notch | Classical | 최소 baseline |
| `M02` | Savitzky-Golay | Classical | 저비용 morphology 보존형 |
| `M03` | DWT soft threshold | Time-freq | SWT 대조군 |
| `M04` | **SWT adaptive threshold** | Time-freq | ★ classical 주력 |
| `M05` | **Sameni EKF / EKS** | Model-based | ★ model-based 주력 |
| `M06` | **Residual 1D U-Net** | Deep | ★ DL 주력 |
| `M07` | SWT → Residual U-Net | Hybrid (순차) | DSP 전처리 + DL |
| `M08` | **Wavelet-subband Residual U-Net** | Hybrid (구조) | ★ 최종 후보 |
| `M09` | CNN + Transformer | Deep (확장) | 긴 문맥의 효용 검증 |
| `B01` | **Oracle wavelet threshold** | bound | wavelet 계열 **상한** |
| `B02` | Wiener (oracle PSD) | bound | 선형 필터 **상한** |

`M09`는 `M08`까지 완료된 후에만 착수한다. 미완료여도 과제는 성립한다.

**`M09` 구조** (구현 완료, 768 K params)
```
y (B,1,1024)
 └─ CNN stem + 2단 다운샘플        -> (B, 96, 256)   토큰 256개
 └─ 학습형 positional embedding
 └─ TransformerEncoder × 3 (nhead 4, pre-norm, GELU)
 └─ 2단 업샘플 + skip
 └─ head(0 초기화) -> 예측 잡음 n̂,  x̂ = y − n̂
```
attention 이 전 토큰을 보므로 실효 수용영역은 window 전체(4.096 s)다.
`M06`(RF 3.55 s)과의 차이가 곧 "**명시적 장거리 attention 이 추가 이득을 주는가**" 의 답이 된다.

### 4.1 딥러닝 모델 규격

**`M06` Residual 1D U-Net**
```
in (B,1,1024)
 stem: Conv1d(1, 32, k=15, p=7) + GN + SiLU
 enc1: ResBlock(32)  -> down(/2) -> 64        # 512
 enc2: ResBlock(64)  -> down(/2) -> 96        # 256
 enc3: ResBlock(96)  -> down(/2) -> 128       # 128
 enc4: ResBlock(128) -> down(/2) -> 160       #  64
 bott: ResBlock(160) x2
 dec4..dec1: up(x2) + skip concat + ResBlock
 head: Conv1d(32, 1, k=1)  ->  n̂  (예측 잡음)
 출력: x̂ = y - n̂            # residual learning (A-1의 gain bias 완화에도 유리)
```
- `ResBlock` = Conv(k=9) → GN → SiLU → Conv(k=9) → GN → (+skip) → SiLU
- **GroupNorm** 사용 (BatchNorm 아님): window마다 진폭 분포가 달라 batch 통계가 불안정
- 파라미터 목표: **< 1 M**
- 수용영역 계산 결과를 `models/README.md`에 기록 (설계 근거로 필요)

**`M08` Wavelet-subband Residual U-Net**
```
y (B,1,1024)
 └─ TorchSWT(level=5, wavelet='sym4')  -> s (B,6,1024)   [A5,D5,D4,D3,D2,D1]
 └─ band-wise Conv1d(6 -> 6*8, k=9, groups=6) + GN + SiLU     # 대역 독립 처리
 └─ fuse: Conv1d(48 -> 32, k=1)
 └─ Residual U-Net (M06과 동일 backbone, in_ch=32)
 └─ head: Conv1d(32 -> 6, k=1)  -> n̂_band
 └─ ŝ = s - n̂_band
 └─ TorchISWT  -> x̂ (B,1,1024)
```
- `TorchSWT`/`TorchISWT` 는 미분 가능해야 하며 **`pywt` 와 수치 일치(≤1e-6) 단위테스트 통과가 착수 조건**.

### 4.2 손실 함수 (단계적 도입 — 순서를 지킨다)

```
L1 :  L_mse
L2 :  L_mse + 0.5 * L_mae
L3 :  L2   + λd * L_diff        # λd = 0.3
L4 :  L3   + λw * L_band        # λw = 0.2, D3/D4 가중 2배  (M08 전용)
```
```
L_diff  = mean( (Δx̂ - Δx)² ),  Δ = 1차 차분
L_band  = Σ_j w_j * mean( (ŝ_j - s_j)² )
```
**L1 → L2 → L3 → L4 순서로 각각 학습해 ablation 표를 만든다.** 한 번에 다 넣지 않는다.

### 4.3 학습 하이퍼파라미터 (초기값)

| | |
|---|---|
| optimizer | AdamW, `wd = 1e-4` |
| lr | `1e-3`, cosine decay, warmup 5 % |
| batch | 64 |
| epoch | 100, early stop patience 15 (val loss) |
| AMP | 사용 (GPU 있을 때) |
| seed | 3개 (0, 1, 2) 반복 → **평균 ± std 리포트** |

**lr 탐색은 `{1e-3, 3e-4, 1e-4}` 3점만.** 그 이상은 과제 범위 밖.

---

## 5. 평가 체계

### 5.1 주 지표 (5) — 결과표 본문

| 지표 | 방향 | 정의 요약 |
|---|---|---|
| `snr_imp_scaled` | ↑ | `SNR_out_scaled − SNR_in` (dB) |
| `rmse` | ↓ | `sqrt(mean((x−x̂)²))` (mV) |
| `cc` | ↑ | Pearson (평균 제거 후) |
| `rpeak_mae_ms` | ↓ | 매칭된 R-peak의 시간 오차 평균 |
| `qrs_dur_err_ms` | ↓ | QRS onset~offset 폭 오차 (**metric noise floor 병기**) |

> **DC 규약 (전 지표 공통)**: 모든 지표는 `x` 와 `x̂` 양쪽의 **평균을 제거한 뒤** 계산한다.
> ECG의 절대 전위 기준선은 임의값이고, 공통 front-end HPF가 이미 DC를 제거하기 때문이다.
> 이렇게 해야 `SNR`/`RMSE`/`PRDN`/`CC` 의 정의가 서로 모순되지 않는다.
> 느린 baseline wander 잔차는 평균 제거로 숨지 않으므로 평가력은 유지된다.

### 5.2 보조 지표 (7) — 부록표

`snr_imp_strict`, `gain_bias(α*)`, `prdn`, `r_amp_err_pct`, `beat_cc`, `hr_mae_bpm`, `psd_logdist`

`psd_logdist = mean( |10log10 PSD_x̂ − 10log10 PSD_x| )` (1–100 Hz)

### 5.3 특수 실험 지표

| 지표 | 실험 | 의미 |
|---|---|---|
| `snr_distortion` | EXP-C | clean 입력 시 출력 SNR = **왜곡 상한** |
| `oracle_gap` | EXP-A | `B01` 대비 부족분 (dB). **`M03`/`M04` 등 wavelet thresholding 계열에만 적용한다** — `B01` 은 그 계열의 상한이지 모든 방법의 상한이 아니다. 실제로 `M07` 은 저 SNR 에서 `B01` 을 넘는다 |
| `halluc_energy` | EXP-E | 삭제 구간(P1/P2)에서 출력이 만들어낸 에너지 |
| `pvc_beat_cc` | EXP-E | PVC beat만의 template correlation |
| `hsc` | 실측 | half-sample consistency (ground truth 불필요) |
| `rtf` | 전체 | `latency / 4.096 s` |

### 5.4 R-peak / delineation 공통 규약

- **검출기: `wfdb.processing.xqrs_detect` 하나로 통일.** 모든 방법의 출력에 동일 적용.
- 매칭 허용오차: **±75 ms**
- `Se = TP/(TP+FN)`, `PPV = TP/(TP+FP)`
- delineation: `neurokit2.ecg_delineate(method="dwt")`, 실패 beat는 결측 처리 후 **성공률도 함께 리포트**
- **beat template 정렬은 항상 clean reference의 R-peak 기준** (A-9)

### 5.5 통계

- 집계 단위 **record**
- paired Wilcoxon signed-rank + Holm-Bonferroni
- effect size: rank-biserial
- 표기: `mean ± std` / `median [IQR]`

---

## 6. 실험 매트릭스

| ID | 이름 | 데이터 | 조건 | 산출물 |
|---|---|---|---|---|
| **EXP-A** | SNR sweep | D1 test | SNR ∈ {-5,0,5,10,15,20}, mixed noise | 성능-입력SNR 곡선, 주 결과표 |
| **EXP-B** | Noise type | D1 test | {bw, ma, em, pli, awgn, mixed} × SNR=10 | 잡음종류별 히트맵 |
| **EXP-C** | Distortion floor | D1 test | noise 없음 | `snr_distortion` 표 + Pareto 산점도 |
| **EXP-D** | Wavelet 위치 ablation | D1 test | M06 / M07 / M08 | RQ4 답변 |
| **EXP-E** | Safety probe | D1 test + annotation | P1/P2/P3 | hallucination 표 |
| **EXP-F** | Device transfer | D2, D3 | pretrain-only vs fine-tuned | 전이 성능표 + 실측 파형 |
| **EXP-G** | 비용 | — | 전 기법 | params/MACs/latency/RTF |

---

## 7. 저장소 구조

```
ECG_denoising_method_comparision/
├─ docs/                      00_review, 01_design, 02_procedure, 03_datasheet, 90_results
├─ ecgdn/                     파이썬 패키지 (모든 로직)
│  ├─ config.py  utils.py  registry.py
│  ├─ data/      synthetic, mitdb, nstdb, noise, mixer, windows, splits, dataset, download
│  ├─ methods/   base, identity, bandpass, savgol, wavelet, kalman_sameni, oracle, dl_wrapper
│  ├─ models/    blocks, resunet1d, swt_torch, wavelet_unet, cnn_transformer, losses
│  ├─ eval/      signal_metrics, rpeak, morphology, spectral, snr_estimation, engine, stats
│  └─ viz/       plots
├─ scripts/                   실행 진입점 (CLI)
├─ configs/                   실험 1개 = yaml 1개
├─ tests/                     pytest
├─ results/                   (git-ignored) metrics/figures
└─ data/                      (git-ignored) 원본 데이터
```

---

## 8. 진행 단계 요약 (상세는 `02_procedure.md`)

| Phase | 내용 | 산출물 | 외부 데이터 필요? |
|---|---|---|---|
| P0 | 저장소 · 규격 · 테스트 골격 | 패키지 스켈레톤 | X |
| P1 | 합성 ECG + 잡음 + 믹서 + 윈도잉 | D0 파이프라인 | X |
| P2 | 평가 엔진 (5 주지표 + 보조) | `eval/` 전체 | X |
| P3 | 고전 기법 M00~M04 + bound B01/B02 | `methods/` | X |
| P4 | Sameni EKF/EKS + 6항목 자가진단 | `kalman_sameni.py` | X |
| P5 | MIT-BIH/NSTDB 적재 (로컬 실행) | D1 파이프라인 | **O** |
| P6 | M06 Residual U-Net 학습 | 체크포인트 + ablation | O |
| P7 | TorchSWT 검증 → M07 / M08 | 체크포인트 | O |
| P8 | EXP-A~E 전체 실행 + 통계 + 그림 | 결과표/그림 | O |
| P9 | Arduino 수집 → SNR 추정 → EXP-F | 실측 결과 | 실측 |

**P0~P4, P2는 외부 데이터 없이 완결된다.** 본 세션에서 여기까지 실제로 만든다.
