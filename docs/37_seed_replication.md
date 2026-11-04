# 37. 그림 7 판정의 재실험 — 시드 10 개 · 한 환경 · Holm 보정 (D-44 실행 지시)

> **이 문서는 다른 세션이 그대로 따라 하도록 쓴 실행 지시서다.** 왜 하는지와 사전 예측은 `docs/21_decisions.md`
> D-44 에 있다. 이 문서는 **무엇을 어떤 순서로 하고, 끝났을 때 무엇이 남아야 하는가**만 적는다.
>
> **옵션 전부는 `docs/24_cli_reference.md`.** 아래 명령은 이 실험에 필요한 것만 보인다.

---

## 0. 한 줄 요약

판넬 그림 7(딥러닝 개선 과정)의 25 개 비교는 시드 3~6 개, 보정 없는 대응 t-검정에 기대고 있다. 이것을
**arm 17 개 × 시드 0~9 = 학습 170 회**를 **한 보관소(`results/ext/rep10`) · 한 환경**에서 다시 돌리고, **사전에 정한
검정(대응 t-검정 + 그림 전체 Holm 보정, Wilcoxon 확인, TOST 동등성)**으로 판정한다.

## 1. 시작 전에 반드시 지킬 것 (`CLAUDE.md` 의 명판)

- `data/arduino/` 는 git 에 넣지 않는다. PhysioNet 403 을 우회하지 않는다 — 데이터는 `data/raw/` 에 이미 있다.
- 무인 실행을 띄우기 **전에** 감시자를 확인하고(`list_triggers`), **락을 잡는 러너**(`scripts/run_sweep_locked.sh`)로만
  띄운다(O-23 · O-24 · O-25). 프로세스를 끝낼 때는 띄울 때 적어 둔 `$!` 로 끝낸다 — 대상 이름과 `kill` 을 같은 줄에
  두지 않는다(O-15 외 5 회).
- 수치는 로그 마지막 줄이 아니라 **파일에서 읽는다**(`results/ext/rep10/*/summary.json`).
- 결과가 사전 예측과 달라도 예측을 지우지 않는다(D-29). 판정이 바뀐 판넬 문장은 철회 표시와 함께 바꾼다.
- 커밋 전 `pytest tests/` 를 돌린다.

## 2. 실행할 arm (17 개)

| 단계(그림 7) | arm | 비교 기준 |
|---|---|---|
| ① 기준 모델 | `m06_l1` | — |
| ② 구조 | `m07_l1` · `m08_l1` · `m10_l1` · `m09_l1` | `m06_l1` |
| ③ 규모 | `m06_l1_half` · `m06_l1_quarter` · `m06_l1_deep2` · `m06_l1_deep4` · `m06_l1_nb4` | `m06_l1` |
| ④ 학습 기법 | `m06_l1_nonorm` · `m06_l1_ema` | `m06_l1` |
| ⑤ 원인 분석 | `m06_l1_band_hi` | `m06_l1` |
| ⑥ 학습 목표 | `m06_l3` · `m06_l6` · `m06_cond` | `m06_l1` |
| | `m06_cond_l6` | **`m06_l6`** |

설정 파일은 모두 `configs/<arm>.yaml` 에 있다. 데이터는 MIT-BIH(`--source mitdb`), 시드는 **0~9 를 모든 arm 에 같게**.

## 3. 순서

### 3.1 분석 코드를 먼저 넣는다 — 결과를 보기 전에

`scripts/analyze_seed_sweep.py` 는 지금 대응 t-검정만 한다. 학습을 띄우기 전에 다음을 넣고 시험과 함께 커밋한다.

- 비교 목록(위 표의 16 개 비교 × 두 지표 `fixed_snr_imp_scaled` · `band_15_20`)을 코드에 고정한다.
- **Holm 보정** — 그 목록 전체를 한 계열로. 계열을 결과를 본 뒤 좁히지 않는다.
- **Wilcoxon 부호 순위 검정(정확)** — 확인용. t-검정과 판정이 다르면 둘 다 보고한다.
- **TOST(±0.5 dB)** — p_Holm ≥ 0.05 인 비교에만. 「동등」 · 「판정 불가」를 구분해 적는다.
- 출력: 비교마다 n · 평균 Δ · 표준편차 · p · p_Holm · Wilcoxon p · TOST p · 판정(향상 / 저하 / 동등 / 판정 불가)을
  `results/ext/rep10/analysis.csv` 와 마크다운 표로.
- 시험(`tests/`): 알려진 p 값 목록에 대한 Holm 결과, n = 4 · 10 의 Wilcoxon 최소 p, TOST 의 경계 사례.

### 3.2 보정 시험 — 이 환경이 저장소 값을 재현하는가 (약 30 분)

```bash
python3 scripts/run_seed_sweep.py --stage calib --out results/ext/rep10_calib --source mitdb
```

`m06_l1` 시드 0 의 `best_metric` 이 3.437 근처(±0.3 dB)면 기존 sweep 값과 같은 표에 실을 수 있다. 벗어나면
`rep10` 은 **그 안에서만** 짝짓는다(D-44 「되돌려야 하는 조건」). 어느 쪽이든 `summary.json` 의 `env` 를 기록에 옮긴다.

### 3.3 본 실험

**이 저장소의 CPU 컨테이너에서** (학습 1 회 약 20~33 분, 전체 약 70 시간):

```bash
bash scripts/run_sweep_locked.sh --arms m06_l1,m07_l1,m08_l1,m09_l1,m10_l1,m06_l1_half,m06_l1_quarter,m06_l1_deep2,m06_l1_deep4,m06_l1_nb4,m06_l1_nonorm,m06_l1_ema,m06_l1_band_hi,m06_l3,m06_l6,m06_cond,m06_cond_l6 0-9 --out results/ext/rep10
```

**외부 GPU 에서** (Colab · Kaggle 등, 붙여 넣는 절차는 `docs/35_external_compute.md` §1):

```bash
python3 scripts/run_seed_sweep.py --arms m06_l1,m07_l1,m08_l1,m09_l1,m10_l1,m06_l1_half,m06_l1_quarter,m06_l1_deep2,m06_l1_deep4,m06_l1_nb4,m06_l1_nonorm,m06_l1_ema,m06_l1_band_hi,m06_l3,m06_l6,m06_cond,m06_cond_l6 --seeds 0-9 --out results/ext/rep10 --source mitdb --device cuda --amp off
```

- `--amp off` — 혼합 정밀도는 CPU 결과와 수치가 달라진다.
- 끊겨도 같은 명령을 다시 실행하면 `summary.json` 이 있는 학습은 건너뛴다.
- **여러 기계에 나누어 돌리지 않는다.** 나누어야 하면 같은 GPU 종류 · 같은 torch 판인지 `env` 로 확인하고, 다르면
  한 보관소에 합치지 않는다. 시드 단위가 아니라 **arm 전체를 같은 환경에** 두는 것이 짝의 전제다.

### 3.4 분석

```bash
python3 scripts/analyze_seed_sweep.py results/ext/rep10
```

(3.1 에서 넣은 옵션 이름을 쓴다.) 결과를 D-44 의 예측 P1~P8 과 하나씩 대조한다.

### 3.5 기록

- `docs/20_findings.md` 에 F 기록을 새로 단다: 판정표, 예측과의 대조(맞음 · 틀림), 지금 판넬 판정과 달라진 항목.
- D-44 의 상태를 「완결 → F-xx」로 바꾼다.
- 보고서(`docs/91_report.md`) 5.10 절과 6 장에 요약을 넣는다(`tests/test_repo_integrity.py` 가 F 의 보고서 반영을 검사한다).

### 3.6 판넬 · 브로셔 반영

- `scripts/make_panel_figs.py` 의 `FLOW` 에서 모든 비교의 sweep 을 `rep10` 으로 바꾼다(기준 sweep 인자는 `None`).
- 채운 표식의 조건을 **p_Holm < 0.05** 로 바꾼다(지금은 보정 전 p). 범례 · 캡션에 「대응 t-검정, Holm 보정, 시드
  n = 10」을 적는다.
- 판정 문장(개조식 두 줄)을 분석 결과대로 고친다. 표현 규칙은 `docs/95_academic_style.md`, 고친 뒤 §3.1 의 용어 대조
  절차를 다시 거친다.
- 그림 저장 전 `pytest tests/test_figure_text.py`, 저장 뒤 **눈으로 본다**. 판넬 재생성:

```bash
python3 scripts/build_panel.py --template <양식 pptx> --final --render
```

  양식 pptx 는 학과 자료라 저장소에 없다 — 사용자에게 받는다.

## 4. 완료 조건

- [ ] `results/ext/rep10/` 에 arm 17 개 × 시드 10 개 = `summary.json` 170 개
- [ ] 분석 코드(Holm · Wilcoxon · TOST)가 **결과보다 먼저** 커밋되어 있다
- [ ] `analysis.csv` 와 판정표, 예측 P1~P8 대조가 F 기록에 있다
- [ ] 판넬 그림 7 의 채운 표식이 p_Holm 기준이고, 캡션에 검정과 n 이 적혀 있다
- [ ] `pytest tests/` 통과, 커밋 · 푸시
