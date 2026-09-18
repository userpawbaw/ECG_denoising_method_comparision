# ECG Denoising Method Comparison

아날로그로 취득한 ECG의 디지털 잡음 제거 기법 비교 연구.
**고전 DSP · 모델 기반(Kalman) · 딥러닝 · 하이브리드**를 동일 조건에서 정량 비교한다.

> **범위**: ECG morphology를 보존하면서 acquisition noise를 줄이는 **연구용 신호처리 시스템**이다.
> 임상 진단용 의료기기가 아니며, 진단 목적으로 사용해서는 안 된다.

---

## 여기부터 읽는다

| 알고 싶은 것 | 문서 |
|---|---|
| **무엇을 했고 결과가 무엇인가** | [`docs/91_report.md`](docs/91_report.md) — 종합 보고서. **연구 파트의 유일한 진입점**이다 |
| **지금 어디까지 왔나 · 다음은 무엇인가** | [`docs/99_status.md`](docs/99_status.md) |
| **이 저장소를 읽는 법 — 기록 규약** | [`docs/19_record_keeping.md`](docs/19_record_keeping.md) |
| **시연 화면의 설계·검증 (UI/UX 파트)** | [`docs/ui/00_index.md`](docs/ui/00_index.md) — 이 파트는 판단 기준이 달라 분리돼 있다 |
| **작업 규약 (사람·AI 공통)** | [`CLAUDE.md`](CLAUDE.md) |
| 설계 근거 · 실험 절차서 | [`docs/01_design.md`](docs/01_design.md) · [`docs/02_procedure.md`](docs/02_procedure.md) |
| 용어·지표 사전 | [`docs/18_glossary.md`](docs/18_glossary.md) |

`docs/` 의 나머지 문서를 여기에 나열하지 않는다. 진입점이 자기 아래를 안내하고,
목록을 README 에 베껴 두면 **낡는다** — 실제로 그렇게 낡았다(D-35).

## 기록 체계

갈림길과 실패를 지우지 않고 남기는 것이 이 과제의 산출물 중 하나다.

| 기호 | 무엇 | 어디 |
|---|---|---|
| **F** | 발견 — 이상한 것을 본 순간, 원인을 알기 전에 적는다 | [`docs/20_findings.md`](docs/20_findings.md) |
| **D** | 결정 — **실행 전에** 적는다. 무엇을 왜 버렸는지가 값이다 | [`docs/21_decisions.md`](docs/21_decisions.md) |
| **O** | 운영 사고 — 반복되면 규칙으로 승급시킨다 | [`docs/22_incidents.md`](docs/22_incidents.md) |
| **R** | AI 협업 교훈 — **재사용 규칙**으로 끝난다 | [`docs/23_ai_review.md`](docs/23_ai_review.md) |

UI/UX 파트는 같은 규격에 번호만 나눈 **UF · UD · UO · UR** 을 쓴다(`docs/ui/`).
공통 규칙 넷:

- **기각한 가설과 틀린 예측을 지우지 않는다.** 정정은 덧붙이고, 원문은 남긴다.
- 문장마다 근거 태그를 단다 — `[측정] [로그] [커밋] [문헌] [추론] …`. 없으면 **「기록 없음」**이라고 적는다.
- 수치는 로그 마지막 줄이 아니라 **파일에서 읽어** 옮긴다.
- 위 셋을 사람이 아니라 검사가 강제한다 — `scripts/check_records.py` · `tests/test_repo_integrity.py`.

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
| `pytest tests/` | 단위·회귀 검사 + **기록 무결성 검사** (빠진 F/D 기록과 낡은 수치를 잡는다) |
| `scripts/check_synthetic.py` | 합성 ECG: ODE vs 위상영역 일치, R-peak 100 % |
| `scripts/check_noise.py` | 잡음 6종의 PSD 대역 |
| `scripts/check_snr_estimator.py` | SNR 추정기 교정표 생성 |
| `scripts/diagnose_sameni.py` | Sameni 구현 6항목 자가진단 |

`push` 하면 GitHub Actions 가 같은 검사를 돌리고 **시연 화면 스크린샷을 올린다**
(`.github/workflows/checks.yml` · 근거와 한계는 `docs/99_status.md`).

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
| `M_FE` | front-end | 취득단 전처리 단독 — front-end 만으로 어디까지 가나 |
| `M01`/`M01d` | classical | Bandpass 0.5–40 / 0.5–100 Hz + 자동 notch |
| `M02` | classical | Savitzky-Golay |
| `M03` | time-freq | DWT soft threshold (대조군) |
| `M04` | time-freq | **SWT adaptive threshold** (level별 k + QRS 보호 + garrote) |
| `M04s` | time-freq | SWT soft threshold — **교과서 기본 설정**. M04 의 구성요소를 가르는 대조군 |
| `M04np` | time-freq | SWT, QRS 보호 없음 — 보호 항의 기여만 떼어 본다 |
| `M05`/`M05f` | model-based | **Sameni EKS / EKF** |
| `M06` | deep | **Residual 1D U-Net** |
| `M07` | hybrid | SWT → Residual U-Net (순차) |
| `M08` | hybrid | **Wavelet-subband Residual U-Net** (표현공간) |
| `B01` | bound | Oracle wavelet threshold — wavelet 계열의 **상한** |
| `B02` | bound | Oracle Wiener — 선형 시불변 필터의 **상한** |

`M06`~`M08` 은 체크포인트가 있어야 레지스트리에 올라간다(`ecgdn/methods/dl_wrapper.py`
의 `register_dl`). 나머지는 import 시점에 등록된다 — **등록된 방법이 이 표에 없으면
검사가 잡는다**(`tests/test_repo_integrity.py`).

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
├─ registry.py          방법 등록 — oracle 계열의 ctx 접근을 여기서 강제한다
├─ data/                synthetic · noise · mixer · windows · mitdb · nstdb · arduino · dataset
├─ methods/             base(계약) · frontend · bandpass · savgol · wavelet · kalman_sameni ·
│                       oracle(bound) · dl_wrapper
├─ models/              blocks · resunet1d · swt_torch(미분가능) · wavelet_unet · losses
├─ eval/                signal_metrics · rpeak · morphology · spectral · snr_estimation ·
│                       engine · stats
└─ viz/plots.py

scripts/                실행 진입점 (모든 산출물은 여기서만 생성)
configs/                실험 1개 = yaml 1개
ui/palette.json         **색의 단일 원본** — 시연 화면과 보고서 그림이 여기서 색을 받는다
demo/                   박람회 시연 화면 (무빌드 · file:// 로 열린다) · gallery.html
results/                실험 산출물 · 보고서 그림 · 화면 스크린샷
hardware/               Arduino 스케치
tests/                  pytest
```

`data/arduino/` 는 **git 에 올리지 않는다** — 개인 생체정보다.

**모든 denoiser는 하나의 계약을 따른다**:

```python
x_hat = denoiser(y, fs, ctx=None)      # 길이·스케일 보존
```

덕분에 기법을 추가해도 실험 스크립트를 고칠 필요가 없다.
`ctx['x_clean']`은 oracle 계열만 받을 수 있고, 그 경우 이름이 `oracle_`로 시작해야 한다
(`BaseDenoiser`가 강제 검사한다).
