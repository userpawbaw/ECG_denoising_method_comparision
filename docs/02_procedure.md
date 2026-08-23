# 02. 구현 진행 절차서

> **사용법**: 위에서부터 STEP 순서대로 진행한다. 각 STEP은
> `목적 / 산출 파일 / 구현 사양 / 완료 판정(DoD) / 검증 명령` 5개로 구성된다.
> **DoD를 통과하지 못한 STEP은 다음으로 넘어가지 않는다.**
>
> 표기: `[NO-DATA]` = 외부 데이터 없이 수행 가능 / `[NEEDS-MITDB]` = PhysioNet 데이터 필요 / `[NEEDS-HW]` = 실측 장비 필요

---

## 진행 현황 (자동 아님 — STEP 완료 시 갱신할 것)

> **현재 상태·다음 절차의 단일 진실 원천은 `docs/99_status.md` 다.**
> 이 표는 STEP 단위 요약이고, 산출물이 어느 데이터(D0/D1)·어느 커밋 기준인지와
> 미해결 문제 목록은 그쪽에 있다.

| STEP | 상태 | 산출물 / 통과한 DoD | 비고 |
|---|---|---|---|
| 00 환경 | ✅ | `requirements.txt`, `Makefile` | |
| 01 규격 | ✅ | `config.py`, `utils.py`, `registry.py` | `pytest tests/test_utils.py` |
| 02 합성 ECG | ✅ | ODE↔위상영역 오차 5.9e-11, R-peak 100 % | **PVC 커널을 zero-phase-mean 으로 정규화**(아래 F-2) |
| 03 잡음 | ✅ | 6종 PSD 대역 검증 | **degenerate 잡음 버그 수정**(아래 F-1) |
| 04 믹서/OLA | ✅ | 재구성 오차 < 1e-10, SNR 오차 < 1e-6 | |
| 05 신호 지표 | ✅ | 0.5배 축소가 strict 6.02 dB / scaled ∞ 로 분리됨 | DC 규약 통일 |
| 06 R-peak 지표 | ✅ | 1:1 매칭, +8 ms 편향 검출 확인 | |
| 07 morphology + floor | ✅ | `docs/03_metric_floor.md` | **sub-sample beat 정렬 필수**(아래 F-3) |
| 08 스펙트럼 + SNR 추정 | ✅ | `docs/04_snr_estimator_calibration.md` | AWGN 편향 0.9 dB |
| 09 평가 엔진 | ✅ | `evaluate`, `evaluate_many`, `stats` | **guard band 5 s 확정**(아래 F-4) |
| 10 M00~M02 | ✅ | R-peak 편향 < 0.2 ms | |
| 11 M03/M04 + B01 | ✅ | `docs/05_swt_tuning.md`, 검증 세트 +11.4 dB | **σ 는 전역 추정**(00_review B-1 정정) |
| 12 B02 | ✅ | M04/M05 가 B02 를 넘음 → 비선형 처리 필요 | |
| 13 Sameni | ✅ | `docs/06_sameni_diagnosis.md` 7/7 | **버그 2건 수정, +5.9 → +16.2 dB** |
| 14 다운로드 | ✅ | mitdb 48기록 90 MB + nstdb bw/ma/em 31 MB | 2026-08-23 도착. `check_realdata_path.py` 로 경로 검증 |
| 15 적재/split | ✅ | `mitdb.py`, `nstdb.py`, `splits.py` (DS1/DS2) | 데이터 확보 후 통합 테스트 |
| 16 데이터셋 | ✅ | 결정론적 재현, 역정규화 검증 | 합성 소스로 MIT-BIH 없이도 동작 |
| 17 M06 | ✅ | overfit 2.2e-5, 976K params, RF 3.55 s | |
| 18 학습 루프 | ✅ | best 선택을 지표 기준으로 | |
| 19 loss ablation | ✅ | `results/d0/ablation_loss.csv`, `docs/10_loss_ablation_d0.md` | TEST split 측정. M06 은 L2/L3 가 L1 을 지배, M08 의 L4 는 beat_cc 손실 |
| 20 DL 래퍼 | ✅ | 임의 길이, 경계 불연속 없음 | |
| 21 TorchSWT | ✅ | pywt 대비 < 1e-6, gradcheck 통과 | |
| 22 M07 | ✅ | D0+FE 학습 완료 (best ep 8) | 전처리형 hybrid 는 M06 보다도 낮다 — SWT 가 신경망 도달 전에 정보를 없앤다 |
| 23 M08 | ✅ | D0+FE, L1/L3/L4 전부 학습 | 전체 평균 최고(11.97 dB). 표현공간형이 전처리형을 이긴다 |
| 24 EXP-A | ✅ | `results/d0/exp_a` (173,712 행) | 15 dB 최고가 M04 → **M08** 로 바뀜. M06/M08 이 M01 대비 유의(r≈1.0) |
| 25 EXP-B/C | ✅ | `results/d0/exp_b`(202,664), `exp_c`(28,952) | **F-11**: oracle=M_FE=34.13 dB. wavelet 은 bw/em/impulse 에서 기여 0 |
| 26 EXP-E | ✅ | `docs/07_safety_probe_d0.md` | DL 이 **가장 안전**(asystole 0.044). M05 만 PVC 훼손(−0.065) |
| 27 리포트 | ✅ | `docs/90_results_d0.md`, `results/d0/report/` | 91_report 5장 전면 재작성, 경고 배너 해제 |
| 28 실측 수집 | ⬜ | `docs/08_acquisition.md`, 스케치, 로거 | **하드웨어 필요** |
| 29 실측 SNR | ⬜ | `scripts/estimate_real_snr.py` | 28 이후 |
| 30 EXP-F | ⬜ | | 28 이후 |

범례: ✅ 완료 / 🔄 진행 중 / ⬜ 외부 조건 대기

### 구현 중 실제로 발견한 것 — 인덱스

> **전문은 `docs/20_findings.md` 에 있다.** 각 항목의 발단(무엇이 이상해
> 보였나), 먼저 의심했다가 배제한 가설, 결정적 측정, 놓쳤다면 무엇이 보고서에
> 실렸을지가 거기 기록돼 있다. 기록 형식은 `docs/19_record_keeping.md`.
>
> 결과를 안 바꾼 운영 사고는 `docs/22_incidents.md`(O),
> 갈림길에서 고른 것은 `docs/21_decisions.md`(D).

| ID | 한 줄 | 영향 |
|---|---|---|
| **F-12** | MIT-BIH 를 '정답' 으로 두면 front-end 를 쓰는 방법이 전부 부당하게 진다 | D1 실험 전체. 이대로면 "무처리가 최선" 이 나왔다 |
| **F-11** | oracle 도 front-end 의 천장(34.13 dB)에 걸려 있다 | "SWT 가 oracle 수준" 서술의 근거 붕괴 |
| **F-10** | 딥러닝만 공통 front-end 를 받지 못했다 | **RQ1·RQ4 결론 뒤집힘.** 전 모델 재학습 |
| **F-9** | 앞단을 고치면 뒤에 학습된 체크포인트가 동시에 무효 | STEP 19 결론 철회 |
| **F-8** | 합성 벤치마크가 딥러닝을 12 dB 과대평가 | 생성기 전면 수정 |
| F-7 | SNR sweep 은 잡음 실현을 1회만 뽑아야 한다 | EXP-A 의 전제 |
| F-6 | Sameni 구현 버그 2건 — 방법이 아니라 구현이 틀렸다 | `M05` +5.9 → +16.2 dB |
| F-5 | SWT 의 σ 를 level 별 MAD 로 추정하면 안 된다 | `M04` −3.5 → +11.4 dB |
| F-4 | `filtfilt` padlen 으로 front-end 자체가 11.9 dB 왜곡 | guard band 5 s 확정 |
| F-3 | 1 샘플(4 ms) 정렬 오차가 SNR 추정을 6 dB 왜곡 | sub-sample 정렬 필수화 |
| F-2 | PVC 커널 위상 평균 차이로 beat 마다 DC 계단 | 왜곡 하한 40 → 13 dB 붕괴 |
| F-1 | `motion_synth` 가 드물게 전(全) 0 잡음 생성 | 학습이 매번 에폭 6 에서 죽음 |

**앞의 다섯(F-12·F-11·F-10·F-9·F-8)은 성격이 같다 — 방법이 아니라 측정 틀이
틀렸던 경우이고, 다섯 다 그럴듯한 결과표를 만들어냈다.**


## PHASE 0 — 기반

### STEP 00. 환경 구축 `[NO-DATA]`

**목적**: 모든 STEP이 같은 환경에서 돌도록 고정.

**산출 파일**: `requirements.txt`, `pyproject.toml`, `.gitignore`, `Makefile`

**구현 사양**
```
numpy scipy matplotlib pandas pyarrow PyWavelets wfdb neurokit2 tqdm pyyaml pytest torch
```
- 가상환경 사용을 권장: `python -m venv .venv && source .venv/bin/activate`
- `.gitignore` 에 `data/`, `results/`, `*.pt`, `.venv/` 포함 (원본 데이터·체크포인트는 커밋하지 않는다)

**DoD**: 아래 명령이 전부 성공.

**검증**
```bash
pip install -r requirements.txt
python -c "import numpy,scipy,pywt,wfdb,neurokit2,torch;print('env ok')"
```

---

### STEP 01. 패키지 골격 + 공통 규격 `[NO-DATA]`

**목적**: `01_design.md` 2장의 신호 규격을 **코드 상수**로 못박는다. 이후 매직넘버 금지.

**산출 파일**: `ecgdn/__init__.py`, `ecgdn/config.py`, `ecgdn/utils.py`, `ecgdn/registry.py`

**구현 사양**
- `config.py`:
  ```python
  FS = 250.0
  WIN = 1024           # 4.096 s
  HOP = 512
  SNR_GRID = (-5, 0, 5, 10, 15, 20)
  RPEAK_TOL_MS = 75.0
  QRS_PROTECT_MS = 60.0
  @dataclass(frozen=True) class FrontEndCfg: hp_hz=0.5; lp_hz=100.0; notch_hz=(60.,120.); notch_q=30.; auto_notch=True
  @dataclass(frozen=True) class SWTCfg: wavelet='sym4'; level=5; k=(1.0,1.0,0.4,0.4,0.6); mode='garrote'; protect=0.3
  ```
- `utils.py`:
  - `derive_seed(*parts) -> int` : 문자열/정수 튜플의 blake2b 해시 → 32-bit. **전역 seed에 의존하지 않는 결정론적 난수**
  - `rng(*parts) -> np.random.Generator`
  - `power(x)` : `np.var(x)` (평균 제거 파워). **모든 SNR 계산이 이 함수 하나만 쓴다.**
  - `git_hash()`, `save_manifest(path, cfg)`
- `registry.py`: `@register_method("M04")` 데코레이터로 이름→생성자 매핑. 실험 스크립트가 문자열만 알면 되도록.

**DoD**
- `derive_seed("a", 1) == derive_seed("a", 1)` 이고 `!= derive_seed("a", 2)`
- `power(x)` 가 상수 offset에 불변

**검증**: `pytest tests/test_utils.py -q`

---

## PHASE 1 — 데이터 파이프라인 (외부 데이터 없이 완결)

### STEP 02. 합성 ECG 생성기 (McSharry ODE) `[NO-DATA]`

**목적**: PhysioNet 없이 전체 파이프라인을 개발·검증하고, **Sameni EKF의 정답 케이스**를 확보한다.

**산출 파일**: `ecgdn/data/synthetic.py`

**구현 사양**
```
상태: (x, y, z) 또는 극좌표 (θ, z)
θ̇ = ω = 2π/RR(t)
ż = -Σ_i a_i Δθ_i exp(-Δθ_i²/(2 b_i²)) - (z - z0(t))
Δθ_i = ((θ - θ_i + π) mod 2π) - π          # [-π, π] 로 wrap. 이 wrap을 틀리면 파형이 깨진다.
```
- 기본 kernel 5개 `(P, Q, R, S, T)`; `θ_i` 는 rad, `a_i` 는 진폭, `b_i` 는 폭
- 적분: `scipy.integrate.solve_ivp(method="RK45", max_step=1/(4*fs))` 후 `t_eval=1/fs` 격자
- HRV: `RR_k = RR0 * (1 + hrv_std * η_k)` 로 beat 단위 변조
- **반환값에 `r_peak_idx`(정확한 R 위치)와 `params`(a,b,θ)를 함께 반환** — EKF 검증에 필요
- 출력 스케일: R-peak 진폭이 약 `1.0 mV` 가 되도록 정규화

**DoD**
1. 생성 신호에 `xqrs_detect` 를 돌린 R-peak가 **반환된 `r_peak_idx` 와 ±20 ms 이내로 일치** (≥ 99 %)
2. 생성 신호의 HR이 지정한 HR과 ±1 bpm 이내
3. 파형 그림(`results/fig/synth_check.png`)에서 P-QRS-T가 육안 확인됨

**검증**: `python scripts/check_synthetic.py`

---

### STEP 03. 잡음 모델 `[NO-DATA]`

**목적**: SNR 통제 합성에 쓸 잡음 소스를 전부 갖춘다.

**산출 파일**: `ecgdn/data/noise.py`

**구현 사양** — 각 함수는 `(n_samples, fs, rng) -> np.ndarray`(단위분산)
| 이름 | 구현 |
|---|---|
| `awgn` | 표준정규 |
| `pli` | `Σ_h A_h sin(2π h f0 t + φ_h)`, `f0 = 60 ± 0.2 Hz` 랜덤 드리프트, harmonic 3개, 진폭비 `1 : 0.3 : 0.1` |
| `baseline_synth` | 0.05–0.5 Hz 대역제한 잡음 + 호흡성 정현파 (0.2–0.35 Hz) |
| `emg_synth` | 백색잡음 → 20–150 Hz 대역통과 → **랜덤 burst 포락선** (근수축 모사) |
| `motion_synth` | 저주파 대진폭 램프/스텝 + 지수 감쇠 (전극 접촉 변화 모사) |
| `impulse` | 포아송 시점에 sinc/지수 감쇠 임펄스 (사진에서 보인 큰 spike 모사) |
| `NSTDBNoise` | `bw/ma/em` 실제 기록 로더 (STEP 12에서 연결). **시간축 disjoint split 강제** |

- 모든 합성 잡음은 반환 전 `x = (x - mean) / std` 로 단위분산 정규화. (믹서가 SNR로 스케일링)
- `impulse` 는 대화에서 언급된 "사진 속 큰 spike" 를 재현하기 위한 것. **EXP-B에 별도 조건으로 넣는다.**

**DoD**
- 각 잡음의 PSD를 그렸을 때 의도한 대역에 에너지가 몰림 (`results/fig/noise_psd.png`)
- `pli` 의 PSD에 60/120/180 Hz 피크가 보임 (fs=250이므로 180 Hz는 aliasing → **주석으로 명시**)

**검증**: `python scripts/check_noise.py`

---

### STEP 04. SNR 믹서 + 윈도잉/OLA `[NO-DATA]`

**목적**: `01_design.md` 3.5 규약을 구현하고, **A-5 overlap-add**를 확정한다.

**산출 파일**: `ecgdn/data/mixer.py`, `ecgdn/data/windows.py`

**구현 사양**
```python
def mix_at_snr(x, n, snr_db, rng=None) -> (y, n_scaled, actual_snr_db)
    # n' = n * sqrt(var(x) / (var(n) * 10**(snr/10)))
    # 반환된 actual_snr_db 를 assert 로 검증 (|actual - target| < 1e-6)

def frame(x, win=1024, hop=512, pad="reflect") -> (frames, n_pad_left, n_pad_right)
def overlap_add(frames, hop, total_len, n_pad_left) -> x
    # 창: w = sqrt(hann(win, sym=False));  분석·합성 양쪽에 곱함
    # 50% overlap 에서 sum(w^2) = 1 (COLA) → 정확 재구성
```
**DoD (매우 중요 — 여기서 틀리면 이후 전부 오염)**
1. `overlap_add(frame(x)) == x` 를 만족 (`max|err| < 1e-10`), 임의 길이 `N`에 대해
2. `mix_at_snr` 로 만든 `y` 를 다시 측정한 SNR이 목표와 `< 1e-6` 일치

**검증**: `pytest tests/test_windows.py tests/test_mixer.py -q`

---

## PHASE 2 — 평가 엔진 (기법보다 **먼저** 만든다)

> 평가를 나중에 만들면 "기법에 유리한 평가"를 무의식적으로 만들게 된다. 순서를 지킨다.

### STEP 05. 신호 지표 `[NO-DATA]`

**산출 파일**: `ecgdn/eval/signal_metrics.py`

**구현 사양**
```python
def snr_db(x, err):            return 10*log10(var(x)/var(err))
def optimal_gain(x, xhat):     return dot(x0, xhat0)/dot(xhat0, xhat0)      # 0 = 평균 제거
def snr_out_strict(x, xhat):   return snr_db(x, xhat - x)
def snr_out_scaled(x, xhat):   a = optimal_gain(...); return snr_db(x, a*xhat - x)
def rmse(x, xhat)
def prdn(x, xhat):             return 100*sqrt(sum((x-xhat)**2)/sum((x-mean(x))**2))
def pearson_cc(x, xhat)
def metrics_signal(x, y, xhat) -> dict   # snr_in, snr_out_strict, snr_out_scaled,
                                          # snr_imp_strict, snr_imp_scaled, gain_bias,
                                          # rmse, prdn, cc
```
- **모든 계산은 `float64`.**
- `gain_bias = optimal_gain(x, xhat)` — 1보다 크면 출력이 작다는 뜻. 표에 그대로 싣는다.

**DoD**
1. `xhat == x` 일 때 `snr_out_* = inf`, `rmse = 0`, `cc = 1`, `gain_bias = 1`
2. `xhat = 0.5 * x` 일 때 `snr_out_strict = 10log10(4) ≈ 6.02 dB`, **`snr_out_scaled = inf`**, `gain_bias = 2.0`
   → A-1의 scale bias 분리가 실제로 작동함을 증명하는 테스트
3. `xhat = x + c` (상수 오프셋)일 때 지표가 불변

**검증**: `pytest tests/test_signal_metrics.py -q`

---

### STEP 06. R-peak 지표 `[NO-DATA]`

**산출 파일**: `ecgdn/eval/rpeak.py`

**구현 사양**
```python
def detect_rpeaks(x, fs) -> np.ndarray          # wfdb xqrs_detect 단일화. 실패 시 빈 배열
def match_peaks(ref, test, fs, tol_ms=75) -> (matched_pairs, tp, fp, fn)
    # 그리디가 아니라 최소비용 1:1 매칭 (scipy.optimize.linear_sum_assignment) 사용
def metrics_rpeak(x_ref, xhat, fs) -> dict      # se, ppv, f1, rpeak_mae_ms, rpeak_bias_ms,
                                                 # hr_ref_bpm, hr_hat_bpm, hr_mae_bpm, rr_mae_ms
```
- **매칭은 반드시 1:1.** 그리디로 하면 tolerance 안에 2개가 들어올 때 지표가 부풀려진다.
- `hr` 은 RR 중앙값 기반 (평균은 이상치에 취약)

**DoD**
- 합성 ECG에서 `ref == test` 일 때 `se = ppv = 1`, `rpeak_mae_ms = 0`
- `test` 를 인위적으로 `+8 ms` 밀었을 때 `rpeak_bias_ms ≈ +8`

**검증**: `pytest tests/test_rpeak.py -q`

---

### STEP 07. Morphology 지표 + **metric noise floor** `[NO-DATA]`

**산출 파일**: `ecgdn/eval/morphology.py`, `scripts/measure_metric_floor.py`

**구현 사양**
```python
def beat_matrix(x, r_peaks, fs, pre_ms=250, post_ms=400) -> (n_beats, L)   # 경계 beat 제외
def beat_template(x, r_peaks, fs) -> (L,)
def metrics_morph(x_ref, xhat, fs, r_peaks_ref) -> dict
    # r_amp_err_pct, beat_cc, qrs_dur_ref_ms, qrs_dur_hat_ms, qrs_dur_err_ms,
    # delineate_success_rate
def half_sample_consistency(x, r_peaks, fs) -> float     # C-5, ground truth 불필요
```
- **beat 잘라내기는 항상 `r_peaks_ref`(clean 기준)로 양쪽 모두** (A-9)
- QRS onset/offset: `neurokit2.ecg_delineate(method="dwt")`. 실패는 `NaN` 으로 두고 성공률을 함께 반환

**metric noise floor 측정 (A-8) — 이 STEP의 핵심 산출물**
```
clean x → 지표 계산                       ... v0
clean x + 40 dB AWGN (10회, 다른 seed) → ... v1..v10
floor = mean_i |v_i - v0|
```
결과를 `docs/03_metric_floor.md` 에 표로 박아 둔다.
**이후 모든 결과표에서 `floor` 미만의 차이는 "구분 불가(n.s.)"로 표기한다.**

**DoD**
- `qrs_dur_err_ms` 의 floor가 측정되어 문서화됨
- `half_sample_consistency` 가 clean에서 ≥ 0.99, SNR 5 dB 잡음에서 뚜렷하게 낮아짐

**검증**: `python scripts/measure_metric_floor.py && cat docs/03_metric_floor.md`

---

### STEP 08. 스펙트럼 지표 + SNR 추정기 `[NO-DATA]`

**산출 파일**: `ecgdn/eval/spectral.py`, `ecgdn/eval/snr_estimation.py`

**구현 사양**
- `welch_psd(x, fs, nperseg=1024)`
- `psd_logdist(x, xhat, fs, band=(1,100))`
- `band_power_err(x, xhat, fs, bands=[(0.5,4),(4,15),(15,40),(40,100)])`
- `has_pli(x, fs, f0=60) -> bool` — 01_design 2.1의 10배 기준
- **`estimate_snr_beat_averaged(x, fs)`** (B-3-a):
  ```
  R-peak 검출 → beat matrix → template correlation > 0.9 인 beat만
  m = mean(beats);  r_i = beat_i - m
  Pn = mean_i var(r_i)
  Ps = var(m) - Pn/N          # N-beat 평균의 잡음 저감 보정. 음수면 0으로 클립
  return 10*log10(Ps/Pn), 진단정보 dict
  ```
- `estimate_snr_isoelectric(x, fs)` (B-3-b)

**DoD (정확도 검증)**
합성 ECG에 알려진 SNR로 AWGN을 넣고 `estimate_snr_beat_averaged` 를 돌렸을 때
**`{0,5,10,15,20} dB` 전 구간에서 추정오차 < 1.5 dB**

**검증**: `python scripts/check_snr_estimator.py`
→ 이 STEP을 통과하면 **"실측 신호가 정말 15 dB인가"에 답할 도구가 준비된 것**이다.

---

### STEP 09. 평가 엔진 통합 `[NO-DATA]`

**산출 파일**: `ecgdn/eval/engine.py`, `ecgdn/eval/stats.py`

**구현 사양**
```python
def evaluate(x, y, xhat, fs, *, r_peaks_ref=None, do_morph=True) -> dict[str, float]
def evaluate_batch(records, methods, ...) -> pd.DataFrame     # long format
# 컬럼: record, window, snr_in_target, noise_type, method, metric, value
```
- 결과는 **long format DataFrame** → parquet. 표/그림은 전부 여기서 pivot.
- `stats.py`: `paired_wilcoxon_holm(df, metric, baseline_method) -> DataFrame(p, p_holm, rank_biserial)`

**DoD**: `M00`(identity)을 넣으면 `snr_imp ≈ 0`, `xhat=x` 를 넣으면 모든 지표가 이상적

**검증**: `pytest tests/test_engine.py -q`

---

## PHASE 3 — 고전 기법

### STEP 10. M00~M02 + front-end `[NO-DATA]`

**산출 파일**: `ecgdn/methods/base.py`, `identity.py`, `frontend.py`, `bandpass.py`, `savgol.py`

**구현 사양**
- `base.py`: `Denoiser` 프로토콜 (B-5 계약). `name`, `__call__(y, fs, ctx=None) -> np.ndarray`
- `frontend.py`: `FrontEnd(cfg)` — `filtfilt` HPF/LPF + 조건부 notch. `has_pli()` 로 자동 판정
- `bandpass.py`: `M01 = FrontEnd + 40 Hz LPF` (monitoring 대역) / 파라미터로 diagnostic 대역도
- `savgol.py`: `M02` — `scipy.signal.savgol_filter`. `window_length`, `polyorder` 를 ablation 대상으로

**DoD**
- 모든 방법이 **입력과 같은 길이**를 반환
- `M00` 이 입력을 그대로 반환 (`array_equal`)
- `filtfilt` 사용으로 R-peak bias가 `|bias| < 2 ms`

**검증**: `pytest tests/test_methods_classical.py -q`

---

### STEP 11. M03/M04 wavelet + B01 oracle `[NO-DATA]`

**산출 파일**: `ecgdn/methods/wavelet.py`, `ecgdn/methods/oracle.py`

**구현 사양** (B-1 전체)
```python
def mad_sigma(d):  return np.median(np.abs(d - np.median(d))) / 0.6745

class SWTDenoiser:                       # M04
    # 1) 길이를 2^level 배수로 pad (edge/reflect) → pywt.swt(trim_approx=False, norm=True)
    # 2) level별 σ_j = mad_sigma(D_j)
    # 3) λ_j = k_j * σ_j * sqrt(2*ln(N))
    # 4) QRS 보호: r_peaks(ctx 또는 자체검출) 로 g(t) 생성 → λ_j(t) = λ_j*(1-(1-ρ)*g(t))
    # 5) soft 또는 garrote 적용.  A5(근사계수)는 건드리지 않음
    # 6) iswt → unpad
class DWTDenoiser: ...                   # M03 (대조군, wavedec/waverec)
class OracleWaveletDenoiser:             # B01 — ctx['x_clean'] 필수, name 은 반드시 'oracle_'로 시작
    # keep_j[n] = 1 if s_clean_j[n]**2 > sigma_j**2 else 0
```
**DoD**
1. `k_j = 0` 이면 출력 == 입력 (완전 재구성) — `max|err| < 1e-9`
2. `pad → swt → iswt → unpad` 왕복이 무손실
3. `B01`(oracle)이 `M04`보다 항상 `snr_imp` 가 높다 (아니면 구현 버그)
4. 합성 ECG SNR 5 dB에서 `M04` 의 `snr_imp > 4 dB`

**검증**: `pytest tests/test_wavelet.py -q && python scripts/demo_wavelet.py`

---

### STEP 12. B02 Wiener bound `[NO-DATA]`

**산출 파일**: `ecgdn/methods/oracle.py` (`OracleWienerDenoiser`)

**구현 사양**: clean/noise의 실제 PSD를 알 때의 주파수영역 Wiener 이득
`H(f) = Sx(f) / (Sx(f) + Sn(f))`. **선형 시불변 필터의 성능 상한**을 준다.

**DoD**: 정상 잡음(AWGN, PLI)에서 `B02 ≥ M01`.
**해석 포인트**: `M04/M06` 이 `B02`를 넘으면 → "비선형/시변 처리가 실제로 필요하다"는 정량적 근거.

**검증**: `python scripts/demo_bounds.py`

---

## PHASE 4 — 모델 기반 (Sameni)

### STEP 13. Sameni EKF/EKS `[NO-DATA]`

**산출 파일**: `ecgdn/methods/kalman_sameni.py`

**구현 사양**
```
상태 x_k = [θ_k, z_k]ᵀ
θ_{k+1} = (θ_k + ω δ + π) mod 2π - π
z_{k+1} = z_k - Σ_i (δ α_i ω / b_i²) Δθ_i exp(-Δθ_i²/(2b_i²)) + η_k
관측    s_k = z_k + v_k      (선택적으로 위상 관측 추가)
```
파이프라인:
```
1) front-end (HPF 필수 — A-2 #2)
2) R-peak 검출 → 선형 위상 할당 θ(t): R에서 0, beat 사이 [-π,π] 선형보간
3) phase-wrapped 평균으로 ECG template 생성
4) template 에 Gaussian 5~7개 최소자승 fitting → (α_i, b_i, θ_i)      [A-2 #3]
5) R 추정: TP segment(등전위) 분산                                    [A-2 #4]
   Q 추정: template 잔차 분산
6) EKF forward  →  **EKS backward (RTS smoother)**                    [A-2 #5]
7) 역정규화                                                            [A-2 #6]
```
- 야코비안은 **해석적으로** 유도해 구현 (수치미분 금지 — 느리고 불안정)

**자가진단 스크립트** `scripts/diagnose_sameni.py` — A-2 표의 6개 항목을 자동 점검:
| 체크 | 통과 기준 |
|---|---|
| C1 위상 | `θ(t)` 톱니가 beat당 정확히 1회이고, **`θ` 의 0-crossing 이 각 R-peak 와 ±10 ms** 이내. (±π 불연속은 R 이 아니라 beat 중간에 나타나는 것이 정상 — `[-π,π)` 규약) |
| C2 baseline | HPF 전/후 EKS `snr_imp` 차이를 표로 출력 |
| C3 fitting | template vs fitted Gaussian 합의 `R² > 0.98` |
| C4 Q/R | 추정된 `R` 이 실제 잡음 분산의 ±3 dB 이내 (합성 데이터에서 검증 가능) |
| C5 EKS | `snr_imp(EKS) > snr_imp(EKF)` |
| C6 정규화 | `gain_bias` 가 `0.9 ~ 1.1` |

**DoD (이 STEP의 핵심)**
> **합성 ECG(D0)에서, 생성에 사용한 참 파라미터 `(a_i,b_i,θ_i)`를 EKF에 그대로 주었을 때
> SNR 5 dB 입력에서 `snr_imp ≥ 8 dB` 를 달성해야 한다.**
> 달성하지 못하면 구현 버그이며, **이 상태로 비교 실험에 넣으면 안 된다.**

그 다음 단계로, 파라미터를 **추정**해서 넣었을 때의 성능 저하폭을 측정한다.
`Δ = snr_imp(true params) - snr_imp(estimated params)` → 이 값이 Sameni 계열의 **실용상 한계**를 정량화한다.

**검증**: `python scripts/diagnose_sameni.py --data synthetic`

---

## PHASE 5 — 실데이터 적재 `[NEEDS-MITDB]`

> **주의**: 원격 세션에서는 `physionet.org` 가 차단될 수 있다. 이 PHASE는 **로컬에서 실행**한다.

### STEP 14. 데이터 다운로드

**산출 파일**: `scripts/download_data.py`

```bash
python scripts/download_data.py --db mitdb --out data/raw
python scripts/download_data.py --db nstdb --out data/raw
```
- `wfdb.dl_database('mitdb', 'data/raw/mitdb')` 사용
- 다운로드 후 `RECORDS` 개수 및 각 record의 샘플 수를 검증하고 `data/raw/manifest.json` 기록

**DoD**: `mitdb` 48 records, `nstdb` 의 `bw/ma/em` 3 records 존재 + 체크섬 기록

**주의 — 이 STEP 은 원격 세션에서 수행할 수 없다.** PhysioNet 접근이 조직 egress
정책으로 차단되어 있다(403). 따라서 다운로드는 **사용자 로컬에서** 실행하고, 결과물을
저장소로 공유한다 (`docs/09_data_upload.md`).

그동안 실데이터 경로가 한 번도 실행되지 않은 채로 남으면, 데이터가 도착한 **그날**
처음 깨진다. 그래서 데이터 없이도 이 경로를 강제로 통과시키는 검증을 따로 둔다:

```bash
python scripts/check_realdata_path.py            # WFDB 형식 fixture 로 전 경로 점검
python scripts/check_realdata_path.py --real     # 실제 data/raw 가 도착한 뒤 같은 점검
```
fixture 는 **신호만 합성**이고 파일 포맷(`.hea`/`.dat` fmt 212, `.atr`), 리드 이름,
표본율 360 Hz, 채널 구성은 MIT-BIH/NSTDB 와 동일하다. 즉 `load_record` → `resample_to`
→ `NoiseBank` → `ECGDenoiseDataset` → 평가 엔진까지 **실데이터와 같은 코드 경로**를 탄다.
102/104 처럼 MLII 가 없는 기록의 lead fallback 도 재현한다.

---

### STEP 15. 적재 · 리샘플 · split

**산출 파일**: `ecgdn/data/mitdb.py`, `ecgdn/data/nstdb.py`, `ecgdn/data/splits.py`

**구현 사양**
- `load_record(rec, lead="MLII")` → `(x_mV, fs, ann)`; 물리단위 변환(`p_signal`) 사용
- `resample_to(x, fs_in, fs_out)` = `resample_poly(x, 25, 36)` (360→250)
- **annotation 인덱스도 함께 리샘플** (`round(idx * 25/36)`) — 잊으면 beat-type 층화가 전부 어긋난다
- `splits.py`: `MITDB_SPLIT = {"train": [...], "val": [...], "test": [...], "paced": [102,104,107,217]}`
- `nstdb.py`: `NoiseBank(kind, split)` — **시간축 disjoint** (A-7). `sample(n, rng)` 로 랜덤 crop

**DoD**
1. TRAIN/VAL/TEST record 집합의 교집합이 공집합
2. 리샘플 전후 R-peak 위치가 ±1 sample 이내로 대응
3. `NoiseBank("ma","train")` 과 `NoiseBank("ma","test")` 의 인덱스 구간이 겹치지 않음

**검증**: `pytest tests/test_data_real.py -q`

---

### STEP 16. 학습 데이터셋 (on-the-fly)

**산출 파일**: `ecgdn/data/dataset.py`

**구현 사양**
```python
class ECGDenoiseDataset(torch.utils.data.Dataset):
    # __getitem__(i):
    #   rng = rng("ds", epoch_salt, record_id, window_idx)
    #   x = clean window (1024)
    #   noise = 랜덤 조합: bw/ma/em/pli/awgn/impulse 중 1~3종을 랜덤 가중 합
    #   snr = U(-5, 20)
    #   y = mix_at_snr(x, noise, snr)
    #   scale = robust_scale(y)                         # A-6: noisy 기준
    #   return y/scale, x/scale, {"scale": scale, ...}
```
- `robust_scale(y) = np.percentile(np.abs(y - median), 99) + eps` (이상치에 강건)
- **평가용 고정 데이터셋**은 별도 함수 `build_eval_set(seed)` 로 생성해 **디스크에 저장** → 모든 방법이 동일 `y` 사용

**DoD**
- 같은 seed로 두 번 만든 배치가 **비트 단위 동일**
- 역정규화(`xhat * scale`) 후 진폭이 원래 mV 스케일로 복원됨

**검증**: `pytest tests/test_dataset.py -q`

---

## PHASE 6 — 딥러닝 주력 모델

### STEP 17. 모델 블록 + M06

**산출 파일**: `ecgdn/models/blocks.py`, `resunet1d.py`, `ecgdn/models/README.md`

**구현 사양**: `01_design.md` 4.1 참조.
`README.md` 에 **수용영역(receptive field) 계산 결과**를 반드시 기록:
```
RF = 1 + Σ_layers (k-1) * dilation * Π(이전 stride)
목표: RF ≥ 0.5 s (125 samples) 이상, 이상적으로 1~2 s
```

**DoD**
1. `forward` 형상: `(2,1,1024) -> (2,1,1024)`
2. 파라미터 수 < 1 M
3. **overfit test**: 배치 1개(32 windows)를 300 step 학습해 `train loss < 1e-4` 로 수렴
   → 통과 못 하면 모델·학습 루프 버그. 대규모 학습 전에 반드시 확인.

**검증**: `python scripts/overfit_test.py --model resunet1d`

---

### STEP 18. 학습 루프

**산출 파일**: `ecgdn/train.py`, `scripts/train.py`, `configs/m06_l1.yaml`

**구현 사양**
- AdamW + cosine(warmup 5 %), AMP, grad clip 1.0
- 매 epoch val에서 `snr_imp_scaled` 를 계산해 **loss가 아니라 이 값으로 best 체크포인트 선택**
  (loss 최소 ≠ 우리가 원하는 성능)
- 로깅: `results/{exp_id}/log.csv`, 체크포인트 `best.pt`, `last.pt`, `manifest.json`
- seed 3개 반복 옵션

**DoD**: `configs/m06_l1.yaml` 로 학습이 끝까지 돌고 `best.pt` 와 학습곡선이 생성됨

**검증**: `python scripts/train.py -c configs/m06_l1.yaml`

---

### STEP 19. Loss ablation (L1→L2→L3→L4)

**산출 파일**: `ecgdn/models/losses.py`, `configs/m06_l{1,2,3}.yaml`,
`configs/m08_l{1,3,4}.yaml`, `configs/abl_loss.yaml`, `scripts/make_ablation_table.py`

```bash
bash scripts/run_all_training.sh              # 전 loss 를 같은 조건으로 학습
python scripts/run_exp.py -c configs/abl_loss.yaml
python scripts/make_ablation_table.py         # -> results/ablation_loss.csv
```

**DoD**: 전 설정의 학습 완료 + `results/ablation_loss.csv`
표에 `snr_imp_scaled`, `qrs_dur_err_ms`, `beat_cc` 를 함께 실어 **"loss가 morphology를 실제로 바꾸는지"** 를 본다.

**필수 조건 — 학습 조건이 손실 말고는 전부 같아야 한다.** 특히 `frontend` 가 섞이면
loss 효과와 front-end 효과가 분리되지 않는다 (F-9 에서 실제로 그렇게 됐다).
`make_ablation_table.py` 는 체크포인트 manifest 의 `frontend`/git hash 를 표에 함께 싣고,
설정이 섞여 있으면 **경고를 내며 종료코드 2** 로 끝난다.

**평가는 TEST split 에서 한다.** 학습 로그의 VAL `snr_imp` 는 early stopping 이 best 를
고른 대상이므로 그 위에서의 비교는 낙관적으로 편향된다.

---

### STEP 20. DL 래퍼 (긴 신호 추론)

**산출 파일**: `ecgdn/methods/dl_wrapper.py`

**구현 사양**: 체크포인트를 `Denoiser` 계약으로 감싼다.
```
y (임의 길이) → frame(win,hop) → per-window robust_scale → model → ×scale → overlap_add → x̂
```
**DoD**: 임의 길이 입력에 대해 같은 길이 반환 + 윈도우 경계에 불연속 없음
(검증: `x̂` 의 1차 차분에서 hop 배수 위치의 이상치가 배경 대비 3σ 이내)

**검증**: `pytest tests/test_dl_wrapper.py -q`

---

## PHASE 7 — Wavelet-aware 딥러닝

### STEP 21. TorchSWT / TorchISWT

**산출 파일**: `ecgdn/models/swt_torch.py`

**구현 사양** (B-2)
- à trous 알고리즘: level `j` 의 필터를 `dilation = 2^(j-1)` 로 `conv1d`, downsampling 없음
- 경계 처리는 `pywt` 의 `periodization` 과 맞추는 것이 가장 쉽다 (`mode='periodization'`)
- ISWT: 각 level에서 짝/홀 위상 재구성 평균

**DoD (착수 조건)**
> **`TorchSWT(x)` 와 `pywt.swt(x, 'sym4', level=5)` 의 최대 절대오차 < 1e-6**
> **`TorchISWT(TorchSWT(x)) == x` 의 최대 절대오차 < 1e-6**
> **`torch.autograd.gradcheck` 통과**
>
> 이 3개를 통과하기 전에는 STEP 22로 넘어가지 않는다.

**검증**: `pytest tests/test_swt_torch.py -q`

---

### STEP 22. M07 (순차 hybrid)

**산출 파일**: `configs/m07_l1.yaml`

**구현 사양**: `M04`(SWT denoiser)를 데이터 전처리로 적용한 뒤 `M06` 구조를 학습.
**중요**: `M04` 는 학습 중에 **고정**(파라미터 학습 없음). 전처리를 `Dataset` 안에서 하면 느리므로
**미리 계산해 캐시**한다.

**DoD**: 학습 완료 + `M06` 대비 성능 비교표

---

### STEP 23. M08 (Wavelet-subband Residual U-Net) ★

**산출 파일**: `ecgdn/models/wavelet_unet.py`, `configs/m08_l{3,4}.yaml`

**구현 사양**: `01_design.md` 4.1 후반부.

**DoD**
1. overfit test 통과
2. 학습 완료
3. **RQ4 답변표 생성**: `M06`(raw) vs `M07`(전처리) vs `M08`(표현공간) 3자 비교

---

## PHASE 8 — 실험 실행 및 결과 생성

### STEP 24. EXP-A (SNR sweep)

**산출 파일**: `scripts/run_exp.py`, `configs/exp_a.yaml`

```bash
python scripts/run_exp.py -c configs/exp_a.yaml
```
- TEST record × SNR 6점 × mixed noise × 전 방법
- 고정 시드로 `y` 를 미리 생성해 **모든 방법이 동일 입력**을 받게 함
- 산출: `results/exp_a/metrics.parquet`

**DoD**: `M00`(identity)의 `snr_imp ≈ 0 ± 0.01 dB` (파이프라인 정합성 확인)

---

### STEP 25. EXP-B (잡음 종류) / EXP-C (distortion floor)

**EXP-C 구현** (C-1):
```
x_clean 을 그대로 각 방법에 통과 → snr_distortion = 10log10(var(x)/var(method(x)-x))
```
**DoD**: 전 방법의 `snr_distortion` 표 + `snr_imp` vs `snr_distortion` **Pareto 산점도** 생성

---

### STEP 26. EXP-E (안전성 프로브)

**산출 파일**: `scripts/run_safety_probe.py`

| Probe | 구현 |
|---|---|
| P1 | clean에서 1 beat 구간(R±200 ms)을 0으로 → 강한 잡음 첨가 → 출력의 그 구간 에너지 측정 |
| P2 | 3 s 구간을 등전위선으로 치환 → 상동 |
| P3 | MIT-BIH annotation의 `V` beat만 골라 `beat_cc`, `qrs_dur_err` 계산 |

**DoD**
- `halluc_energy` 표 생성
- **P3에서 `V` beat의 `beat_cc` 가 `N` beat 대비 유의하게 낮은 방법**을 식별 (Wilcoxon)

---

### STEP 27. 통계 · 표 · 그림 자동 생성

**산출 파일**: `scripts/make_report.py`, `ecgdn/viz/plots.py`

생성 산출물:
| # | 그림 |
|---|---|
| F1 | 대표 구간 파형 스택 (clean / noisy / 전 방법), 동일 y축 |
| F2 | QRS 확대 (F1의 한 beat) |
| F3 | PSD 비교 |
| F4 | `snr_imp` vs 입력 SNR 곡선 (+ **실측 추정 SNR 위치 수직선**) |
| F5 | `snr_imp` vs `snr_distortion` Pareto 산점도 |
| F6 | 잡음 종류 × 방법 히트맵 |
| F7 | beat-type 층화 막대그래프 |
| T1 | 주 결과표 (mean±std, floor 미만 차이는 n.s. 표기) |
| T2 | 통계 검정표 (paired Wilcoxon + Holm + effect size) |
| T3 | 계산 비용표 (params/MACs/latency/RTF) |

**DoD**: `python scripts/make_report.py` 한 번으로 **모든 표·그림이 `results/report/` 에 재생성**된다.

---

## PHASE 9 — 실측 데이터 `[NEEDS-HW]`

### STEP 28. Arduino 데이터 수집

**참조**: `00_review.md` B-4 프로토콜(S1~S6) + CSV 스키마.
**산출 파일**: `data/arduino/*.csv`, `docs/08a_acquisition_log.md`

**DoD**: 전 세션 수집 완료 + 각 파일 헤더에 `fs_hz`, `adc_bits`, `vref_v`, `gain`, `session` 기록

---

### STEP 29. 실측 SNR 검증 — **"15 dB" 재확인**

```bash
python scripts/estimate_real_snr.py --in data/arduino --out results/real_snr.csv
```
- STEP 08의 3가지 추정기를 모두 적용
- 세션별(S1, S6-USB, S6-battery) SNR 분포를 표로 출력
- **이 값이 EXP-A 곡선의 어느 지점인지 F4에 표시**

**DoD**: `docs/08b_real_snr.md` (`estimate_real_snr.py` 가 자동 생성) 에 3가지 추정치와 그 차이에 대한 해석 기록

---

### STEP 30. EXP-F (device transfer)

```
(a) pretrain-only  : D1으로 학습한 M06/M08을 실측에 그대로 적용
(b) fine-tuned     : D2(MIT-BIH clean + Arduino noise-only)로 fine-tune 후 적용
```
평가: ground truth가 없으므로 **`hsc`, R-peak Se/PPV, HR 안정성, PSD, 파형 육안 비교**

**DoD**: (a) vs (b) 비교표 + 대표 파형 그림. domain gap의 크기를 정량화.

---

## 부록 A. 매 STEP 공통 규칙

1. **새 코드를 쓰기 전에 그 STEP의 DoD를 먼저 테스트로 작성한다.**
2. 커밋 단위 = STEP 단위. 커밋 메시지 접두사 `STEP xx:`
3. 매직넘버는 `config.py` 로 올린다.
4. 랜덤이 개입하는 모든 함수는 `rng` 를 인자로 받는다 (전역 seed 사용 금지).
5. 결과를 만드는 스크립트는 **반드시 `manifest.json`**(git hash, config, 패키지 버전)을 남긴다.

## 부록 B. 최소 성공 조건 (시간이 부족할 때의 절단선)

| 우선순위 | 범위 | 결과물 |
|---|---|---|
| **필수** | STEP 00~13, 24, 25, 27 | M00~M05 + bound의 완전한 비교 + 그림/표. **이것만으로 과제 성립** |
| **핵심** | + STEP 14~20 | 딥러닝(M06) 추가 → DSP vs DL 비교 |
| **차별점** | + STEP 21~23, 26 | M08 + 안전성 프로브 → RQ4·RQ5 답변 |
| **완성** | + STEP 28~30 | 실측 검증 → RQ6 답변 |

**절단선 위쪽만 해도 완결된 과제가 되도록 순서를 배치했다.**
