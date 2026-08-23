# 99. 현재 상태 · 다음 절차

> **이 문서의 목적**: 세션이 끊기거나 문맥이 압축돼도 "지금 어디까지 됐고 다음에
> 무엇을 하는가"를 저장소만 보고 복원할 수 있게 한다.
> 모든 항목은 **작성 시점에 실제로 검증한 사실**이다. 검증하지 못한 것은
> `[불확실]` 로 표시했다. 추측은 싣지 않는다.
>
> 최종 갱신: 2026-08-23 / P-1~P-5 완료 (D0 완결)

---

## 1. 프로젝트 목표 (변경 없음)

> ECG morphology를 보존하면서 acquisition noise를 감소시키는 디지털 후처리 기법들을
> **동일한 조건에서 정량 비교**하고, 실제 자체 제작 취득 시스템의 잡음 조건에서
> 어떤 기법이 적합한지 규명한다.

의도적으로 하지 않는 주장: "임상 진단 가능한 병원급 ECG 복원", "의료기기 개발".
본 과제는 **연구용 신호처리 시스템**이다. (`01_design.md` 1.2)

연구 질문 RQ1~RQ6 과 대응 실험 EXP-A~F 는 `01_design.md` 1.3 / 4.4 에 있고 변경 없다.

---

## 2. 데이터 축 — 지금 가장 중요한 구분

| 축 | 내용 | 상태 |
|---|---|---|
| **D0** | 합성 ECG (McSharry ODE). 파이프라인 검증용 | 데이터 항상 사용 가능 |
| **D1** | MIT-BIH Arrhythmia + NSTDB 실잡음 | **2026-08-23 도착.** `data/raw/` 에 mitdb 48기록(90 MB), nstdb bw/ma/em(31 MB) |
| D2 | MIT-BIH clean + Arduino 잡음 (device adaptation) | 하드웨어 대기 |
| D3 | Arduino 실측 | 하드웨어 대기 |

### 2.1 `source: auto` 가 만든 함정 — 반드시 인지할 것

모든 학습·실험 config 가 `source: auto` 이고, `get_source("auto")` 는
**`data/raw/mitdb` 에 `.hea` 가 있으면 무조건 MIT-BIH 를 고른다.**

즉 **D1 데이터가 도착한 순간, 같은 config 를 같은 명령으로 돌려도 D0 가 아니라
D1 이 학습된다.** 실제로 확인했다 — `get_source('auto').kind == "mitdb"`,
train 기록이 `101, 106, 108, …` (MIT-BIH DS1).

이것은 **재현성 결함**이다. 같은 커밋 + 같은 config 가 작업 디렉터리 상태에 따라
다른 결과를 낸다. 산출물의 `manifest.json` 에 `source` 가 기록되므로 사후 판별은
가능하지만, 실행 전에는 알 수 없다.

→ 미해결 항목 M-1 (아래 5절).

---

## 3. 현재 구현 상태 (검증 기준)

### 3.1 코드

전부 존재하고 **테스트 182개 통과**. 패키지 44 모듈, 스크립트 20개, config 14개.

`ecgdn/data/` 의 10개 파일이 **git 에 커밋된 적 없었다** (`.gitignore` 의 `data/`
패턴이 모든 경로의 data 디렉터리를 무시). `a893f2b` 에서 복구하고 회귀 테스트로
고정했다. 그 전까지 저장소를 클론하면 `import ecgdn.data` 가 실패했다.

### 3.2 학습 체크포인트 — **D0 완결**

`results/d0/` 에 7종, 전부 `frontend: true` · `source: synthetic` · 같은 커밋:

| run | loss | best VAL | best epoch |
|---|---|---|---|
| m06_l1 | L1 | +6.214 dB | 13 |
| m06_l2 | L2 | +7.027 dB | 9 |
| m06_l3 | L3 | +7.081 dB | 9 |
| m07_l1 | L1 | +2.628 dB | 8 |
| m08_l1 | L1 | +6.737 dB | 15 |
| m08_l3 | L3 | +6.769 dB | 11 |
| m08_l4 | L4 | +6.843 dB | 14 |

**VAL 값은 모델 선택에 쓴 값이라 성능 주장의 근거가 될 수 없다.** 판정은
TEST split(`results/d0/`)이 한다.

`m07_l1` 의 VAL 이 낮은 것은 성능 저하가 아니다. 이 모델은 front-end 에 더해
SWT 전처리(`pre_denoise: M04`)까지 받은 뒤 학습하므로 신경망에 남은 개선 여지가
가장 적다 — `snr_imp` 는 모델 입력 대비 값이다.

D1(MIT-BIH) 체크포인트는 아직 없다.

### 3.3 실험 결과 — **D0 완결**

`results/d0/` 에 exp_a(173,712행) · exp_b(202,664) · exp_c(28,952) ·
abl_loss(37,884) · exp_e · report. 문서는 `90_results_d0.md`,
`07_safety_probe_d0.md`, `10_loss_ablation_d0.md`.

주요 결과와 해석은 `91_report.md` 5장(전면 재작성)에 있다. 요약:

- **RQ1 의 답이 바뀌었다.** 15 dB 최고가 `M04`(7.79) → **`M08`(8.71)**.
  `M06`/`M08` 이 `M01` 대비 유의(이전 p=0.101 → 현재 9e−6, effect r≈1.0).
- **F-11**: oracle `B01`/`B02` 와 `M_FE` 가 34.13 dB 로 동일 — 그건 front-end 의
  천장이다. wavelet 은 bw·em·impulse 에서 oracle 조차 기여 0.
- **RQ3 에 조건이 붙었다**: 딥러닝의 morphology 영향은 입력 SNR 에 따라 부호가
  뒤집힌다(0 dB 개선 → 20 dB 악화).
- STEP 19 DoD 충족: `M06` 은 L2/L3 가 L1 을 지배(trade-off 없음),
  `M08` 의 L4 는 SNR 을 얻고 `beat_cc` 를 잃는다.

### 3.4 보조 산출물 — 전부 **합성 고정**

| 문서 | 스크립트 | 데이터 | D1 재실행 필요 |
|---|---|---|---|
| `03_metric_floor.md` | `measure_metric_floor.py` | `synth_ecg` 직접 호출 | 필요 |
| `04_snr_estimator_calibration.md` | `check_snr_estimator.py` | `synth_ecg` 직접 호출 | 필요 |
| `05_swt_tuning.md` | `tune_swt.py` | `synth_ecg` 직접 호출 | **필수** — `00_review` B-1 이 "MIT-BIH 확보 후 TRAIN split 으로 재탐색" 을 명시 |
| `06_sameni_diagnosis.md` | `diagnose_sameni.py` | `synth_ecg` 직접 호출 | 필요 |
| `07_safety_probe_d0.md` | `run_safety_probe.py` | `get_source("auto")` | 자동 전환됨 |

앞의 4개는 `get_source` 를 쓰지 않으므로 **D1 데이터가 있어도 자동으로 바뀌지 않는다.**
→ 미해결 항목 M-2.

### 3.5 승인된 파이프라인 수정 6개 항목 — 코드 대조 완료

`e6d92ff`~`939b485` 에서 전부 반영. 각 항목의 확인 근거:

| # | 항목 | 확인 |
|---|---|---|
| 1 | `fe` 스위치 + DL 학습·추론 FE 경로 | `dataset.py:61`, `train.py:93`(체크포인트 기록), `dl_wrapper.py:64`(그것을 읽음), 학습 config 7개 전부 명시 |
| 2 | `M_FE` 를 전 실험 필수 행으로 | exp_a/nofe/b/c/e/abl_loss 전부 포함 |
| 3 | `Se`/`PPV` 주 지표 승격 | `make_report.py:26` `MAIN_METRICS` |
| 4 | SNR 구간별 통계 검정 | `table_stats_per_snr.csv` 생성 경로 |
| 5 | D2 split 규약 / `protect_qrs` 단독 ablation | `D2_ADAPT_SPLITS`, `assert_adaptation_records`, `M04np` |
| 6 | "체계적 편향" → 산포 정정 (A-2) | `91_report.md:954` |
| A-3 | `oracle_gap` 적용 범위 한정 | `01_design.md:292` |
| C-4 | `family="bound"` → `"oracle"` | `oracle.py:125,132` |

**코드 반영은 끝났다. 그 수정이 결과를 어떻게 바꾸는지는 아직 측정되지 않았다** —
그러려면 3.2/3.3 의 재학습·재실행이 선행돼야 한다.

---

## 4. 확정된 설계 결정 (변경하려면 이 목록을 먼저 고칠 것)

1. **평가를 기법보다 먼저 만든다.** 지표의 분해능(`03_metric_floor.md`)을 먼저 재고,
   그 미만의 차이는 차이로 보고하지 않는다.
2. **상한선을 함께 그린다.** `B01`(wavelet 계열), `B02`(LTI 계열). 단
   **각자 자기 계열의 상한**이며 모든 방법의 상한이 아니다 (`M07` 은 저 SNR 에서 `B01` 을 넘는다).
3. **공정성 규약 5개** (`91_report.md` 4.2): 동일 입력 / SNR sweep 에서 잡음 실현 고정 /
   공통 front-end / guard band 5 s / 동일 R-peak 검출기.
4. **주 조건은 `frontend: true`**(실사용 시나리오), 부록이 `frontend: false`(알고리즘 본질).
5. **SNR 을 셋으로 분리**: `strict` / `scaled` / `gain_bias`.
6. **record 단위 inter-patient split** (DS1/DS2), 잡음도 시간축 disjoint.
7. **모든 결과표에 `M_FE`(front-end 단독)를 필수 행으로.** 규약은 선언이 아니라
   측정으로 확인한다 (F-10).

발견 목록 F-1 ~ F-10 은 `02_procedure.md` 에 있다.

---

## 5. 미해결 문제

| ID | 문제 | 영향 |
|---|---|---|
| ~~M-1~~ | ~~`source: auto` 가 D0/D1 을 조용히 바꾼다~~ | **해결(P-1)**: `--source` 명시 + `results/{tag}/` 분리. auto 는 해석 결과를 출력한다 |
| **M-2** | 보조 스크립트 4개가 `synth_ecg` 하드코딩 | D1 로 재실행할 수단이 없다. 특히 SWT 튜닝은 TRAIN split 재탐색이 설계상 필수 |
| ~~M-3~~ | ~~D0 학습 미완~~ | **해결(P-3)**: 7종 전부 완료, STEP 19 DoD 표 생성 |
| ~~M-4~~ | ~~장시간 작업이 죽으면 재개 수단이 없다~~ | **해결(P-2)**: `--resume` 으로 optimizer·LR 위치·best 추적까지 복원. 러너는 기본 재개 켬 |
| ~~M-5~~ | ~~D0 실험 결과 무효~~ | **해결(P-4/P-5)**: 전 실험 재실행, 보고서 5장 재작성, 경고 배너 해제 |

### M-4 상세 — 실제로 두 번 잃었다

1. `nohup … &` 로 띄운 학습이 turn 종료 시 프로세스 그룹째 정리됨.
   `results/.train.lock` 이 남아 있던 것이 증거 (정상 종료면 `trap EXIT` 이 지운다).
2. harness 추적 백그라운드로 다시 띄운 학습이 **컨테이너 재시작**으로 종료.

**해결(P-2)**: `Trainer.try_resume()` 과 `train.py --resume` 을 넣었다.
`last.pt` 에 optimizer·scaler 상태, LR 스케줄 위치(`step`), `best_epoch`,
history 를 함께 담는다. 가중치만 복원하는 것은 재개가 아니라 다른 학습이다.
러너는 `RESUME=0` 을 주지 않는 한 재개를 켠 채 돌고, 이미 목표 epoch 까지
끝난 학습은 건너뛴다.

**남는 한계**: 프로세스가 죽는 것 자체는 막지 못한다. 죽어도 잃는 것이
마지막 epoch 하나로 줄었을 뿐이다. LR 스케줄은 `epochs x len(loader)` 로
정규화되므로 재개할 때 `epochs` 를 바꾸면 남은 구간의 LR 곡선이 달라진다.

---

## 6. 다음 절차 (이 순서를 지킨다)

**원칙: D0 를 완결시켜 파이프라인과 보고서를 일관되게 만든 뒤에 D1 로 넘어간다.**
D0 는 파이프라인 검증 역할이고, D0/D1 비교 자체가 "합성 벤치마크가 실제를
얼마나 예측하는가"라는 별도의 결과가 된다 (F-8 의 연장).

| 순서 | 작업 | 완료 조건 |
|---|---|---|
| ~~P-1~~ | ✅ `--source` 명시 + `results/{tag}/` 분리 (`824adc3`) | 양축 1 epoch 실행으로 검증 |
| ~~P-2~~ | ✅ `--resume` (optimizer·LR·best 복원) | 2→4 epoch 재개 시 log 연속·best 유지 확인, 회귀 테스트 5개 |
| ~~P-3~~ | ✅ D0 학습 완결 | 7종 전부 `frontend: true` · `source: synthetic` · 동일 커밋 |
| ~~P-4~~ | ✅ D0 실험 재실행 | exp_a/b/c/e + abl_loss + 리포트, STEP 19 DoD 충족 |
| ~~P-5~~ | ✅ 결과 해석 + 문서 갱신 | 5장 재작성(5.7 에 수정 전후 대조), 7·8장 갱신, F-11 기록, 상태표 정정 |
| **P-6** | M-2 해결: 보조 스크립트 4개에 소스 선택 추가 | D1 로 재실행 가능 |
| **P-7** | D1 사전 작업: metric floor, **SWT 튜닝(TRAIN split)**, Sameni 진단 재측정 | D1 기준 파라미터 확정 |
| **P-8** | D1 학습 + 실험 + 보고 | D0/D1 비교표 |
| P-9 | STEP 28~30 (Arduino 수집 → 실측 SNR → EXP-F) | 하드웨어 대기 |

`P-5` 를 건너뛰고 `P-7` 로 가지 않는다. 이전에 그렇게 해서 무효 수치가 보고서에
경고 없이 남았다.
