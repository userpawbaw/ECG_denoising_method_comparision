# 24. 실행 스크립트 옵션 — **전부**

> **이 문서는 자동 생성이다.** 고칠 것이 있으면 문서가 아니라
> **스크립트의 `help=`** 를 고치고 다시 만든다:
>
> ```bash
> python3 scripts/make_cli_reference.py
> ```
>
> `tests/test_repo_integrity.py` 가 이 파일과 스크립트를 대조한다 —
> 옵션을 늘리고 재생성을 안 하면 거기서 걸린다 (O-31).

**왜 이 문서가 있나.** 매뉴얼은 「이 절차를 따라 하면 된다」까지만
데려다준다. 스스로 여러 판을 돌려보려면 **무엇을 고를 수 있는지**가
한자리에 있어야 한다 — 실제로 `--record`·`--noise` 를 못 찾아
예시 한 조합만 돌려본 일이 있었다(O-31).

각 절차서는 자주 쓰는 몇 개만 설명하고 **전부는 여기를 가리킨다.**

| | |
|---|---|
| 스크립트 | **33 개** |
| 옵션 | **173 개** |
| 설명이 빈 옵션 | **0 개** (`(설명 없음)` 으로 표시된다) |

---

## 시연 — 이 둘만 직접 띄운다

실시간 시연(모드 A)의 실행 파일. 절차는 `docs/30_realtime_demo.md` 6.2~6.3.

### `scripts/fake_arduino.py`

R-7: **가상 아두이노.** 진짜 시리얼 포트 뒤에 펌웨어를 놓는다.

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--fs` | `250` · `500` · `1000` | `500` | 켜질 때의 fs. 펌웨어 기본은 500 이고 PC 가 명령으로 바꾼다 |
| `--baud` | 정수 | `115200` | 스케치의 SERIAL_BAUD. **송신 버퍼 드롭이 여기서 정해진다** |
| `--mode` | `ascii` · `bin` | `ascii` | 켜질 때의 형식. 펌웨어 기본은 ascii 다 |
| `--drift-ppm` | 실수 | `3000.0` | 보드 클럭 오차. Uno 의 세라믹 레조네이터는 약 ±5000 ppm |
| `--leadoff-every` | 실수 | `0.0` | N 초마다 전극이 떨어진다 [s] |
| `--leadoff-len` | 실수 | `0.7` | 전극이 떨어져 있는 시간 [s] |
| `--dur` | 실수 | `0.0` | 0 이면 무한 |
| `--signal-s` | 실수 | `60.0` | 신호 길이 [s]. 끝나면 처음으로 돌아간다. d1 에서 0 이면 기록 전체 |
| `--snr-db` | 실수 | `8.0` | 입력 SNR [dB]. 이 값이 되도록 잡음을 섞는다 |
| `--seed` | 정수 | `7` | 잡음 성분과 crop 위치를 고정한다. mixed 를 재현하려면 이것 |
| `--source` | `synth` · `d1` | `synth` | synth = 합성 심전도, d1 = MIT-BIH 기록 + NSTDB 잡음 |
| `--record` | 문자열 | `100` | --source d1 일 때의 기록 번호 (예: 100 · 105 · 119) |
| `--noise` | 문자열 | `mixed` | d1 잡음: mixed · bw · em · ma · pli · none 등 |
| `--split` | `train` · `val` · `test` | `test` | NSTDB 잡음 구간. **보고서와 같은 자리를 보려면 test** |
| `--offset-s` | 실수 | `0.0` | 기록에서 몇 초 지점부터 실을 것인가 |
| `--lead` | 문자열 | `MLII` | MIT-BIH 유도. 기록에 없으면 첫 채널을 쓴다 |
| `--gain` | 실수 | `1100.0` | AFE 총 이득. **브리지의 --gain 과 같아야** mV 축이 맞는다 |
| `--vref` | 실수 | `5.0` | ADC 기준 전압 [V]. 브리지의 --vref 와 같아야 한다 |
| `--firmware` | `current` · `old` | `current` | old 면 fs·형식 명령을 무시한다 — 구 스케치가 꽂힌 판 |
| `--boot-s` | 실수 | `1.6` | 리셋 뒤 부트로더가 **조용한** 시간 [s]. 0 이면 즉시 시작 |
| `--hold-open` | **켬/끔** | 끔 | slave 를 붙들어 열림 감지를 끈다 (리셋 흉내도 꺼진다) |
| `--usb-poll-ms` | 실수 | `1.0` | 호스트 폴링 간격 [ms]. full-speed USB 는 1 ms 프레임이다 |
| `--usb-packet` | 정수 | `64` | bulk 최대 패킷 [B] |
| `--hiccup-every` | 실수 | `0.0` | N 초마다 호스트가 잠깐 안 가져간다 [s] — 뭉침의 극단 |
| `--hiccup-ms` | 실수 | `40.0` | 그 «안 가져가는» 시간 [ms] |
| `--attach` | 문자열 | — | PTY 대신 이 포트에 붙는다 (com0com·socat) |
| `--list` | **켬/끔** | 끔 | 고를 수 있는 기록·잡음·방법을 보여주고 끝낸다 |
| `--port-file` | 문자열 | — | 포트 이름을 이 파일에 적는다 |
| `--quiet` | **켬/끔** | 끔 | 진행 줄을 안 찍는다 |

### `scripts/serial_bridge.py`

R-5/R-6: 아두이노 -> 실시간 처리 -> 화면. **하드웨어 없이도 끝까지 돈다.**

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--port` | 문자열 | — | 시리얼 포트. 예: /dev/ttyACM0, COM3 |
| `--replay` | `synth` · `csv` | — | 하드웨어 없이 같은 선 규격으로 흉내낸다 |
| `--csv` | 문자열 | — | --replay csv 일 때 읽을 파일 |
| `--baud` | 정수 | `115200` | 시리얼 속도. 스케치의 SERIAL_BAUD 와 맞춘다 |
| `--board-fs` | `250` · `500` · `1000` | `250` | 보드의 샘플링률. **250 이면 리샘플이 없다** |
| `--ascii` | **켬/끔** | 끔 | ASCII 모드로 읽는다 (손실을 셀 수 없다 — 디버깅용) |
| `--methods` | 문자열 | `M_FE,M01,M04` | 쉼표로. 딥러닝은 체크포인트가 있어야 한다 |
| `--axis` | 문자열 | `d1` | 딥러닝 체크포인트 축 |
| `--fe` | `causal` · `median` · `zerophase` | `zerophase` | front-end 모드. 화면에서 실행 중에도 바꿀 수 있다 |
| `--d` | 정수 | `12` | 미래 문맥 [샘플] |
| `--hop` | 정수 | `12` | 추론 간격 [샘플] |
| `--fe-hop` | 정수 | `6` | front-end 블록 [샘플]. 작을수록 지연이 줄고 CPU 만 조금 든다 |
| `--fps` | 실수 | `25.0` | 화면 갱신률 |
| `--serve` | **켬/끔** | 끔 | 브라우저용 SSE 서버를 연다 |
| `--http-port` | 정수 | `8765` | --serve 가 여는 웹서버 포트 |
| `--dur` | 실수 | `0.0` | 0 이면 무한 |
| `--drift-ppm` | 실수 | `3000.0` | --replay 전용 |
| `--drop` | 실수 | `0.0` | --replay 전용 드롭률 |
| `--leadoff-every` | 실수 | `0.0` | --replay 전용 [s] |
| `--adc-bits` | 정수 | `10` | ADC 분해능 [bit]. 카운트를 mV 로 바꿀 때 쓴다 |
| `--vref` | 실수 | `5.0` | ADC 기준 전압 [V]. 가상 보드의 --vref 와 같아야 한다 |
| `--gain` | 실수 | `1100.0` | 아날로그 프런트엔드 총 이득 (AD8232 기본 약 1100) |
| `--quiet` | **켬/끔** | 끔 | 진행 줄을 안 찍는다 (파이프로 넘기면 자동으로 켜진다) |
| `--diag` | 실수 | `0.0` | N 초마다 버퍼 크기와 RSS 를 적는다 (성능 저하 추적용) |

## 데이터 준비

외부 데이터를 받고, 아두이노에서 기록한다.

### `scripts/download_data.py`

STEP 14: PhysioNet 데이터 다운로드 + 검증.

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--db` | `mitdb` · `nstdb` | `['mitdb', 'nstdb']` | 받을 데이터베이스 (mitdb · nstdb). 여러 개를 나열할 수 있다 |
| `--out` | 문자열 | `data/raw` | 받은 것을 둘 곳 |
| `--verify-only` | **켬/끔** | 끔 | 받지 않고 **이미 있는 것만** 검사한다 |

### `scripts/log_arduino.py`

STEP 28: Arduino 시리얼 로거 — 규격에 맞는 CSV 를 만든다.

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--port` | 문자열 | **필수** | 예: /dev/ttyACM0, COM3 |
| `--baud` | 정수 | `115200` | 시리얼 속도. 스케치의 SERIAL_BAUD 와 맞춘다 |
| `--session` | `S1` · `S2` · `S3` · `S4` · `S5` · `S6` | **필수** | 기록 세션 이름. CSV 헤더와 파일명에 들어간다 |
| `--dur` | 실수 | `300.0` | 기록 길이 [s] |
| `--fs` | 실수 | **필수** | 스케치의 FS_HZ 와 같은 값 |
| `--adc-bits` | 정수 | `10` | ADC 분해능 [bit]. CSV 헤더에 적힌다 |
| `--vref` | 실수 | `5.0` | ADC 기준 전압 [V]. CSV 헤더에 적힌다 |
| `--gain` | 실수 | `1100.0` | 아날로그 프런트엔드 총 이득 |
| `--note` | 문자열 | `` | CSV 헤더에 남길 메모 (자세·상태 등) |
| `--out` | 문자열 | `data/arduino` | CSV 를 둘 곳. **data/arduino/ 는 git 에 넣지 않는다** |

## 학습 · 실험

체크포인트와 실험 결과를 만든다. **무인 실행 규칙**이 붙는다 — `docs/17_checklists.md` §2.

### `scripts/train.py`

STEP 18: 학습 진입점. 실험 1개 = yaml 1개.

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `-c` · `--config` | 문자열 | **필수** | 학습 설정 YAML (configs/). **어느 축인지가 여기서 정해진다** |
| `--epochs` | 정수 | — | 설정의 epoch 수를 덮어쓴다 |
| `--out` | 문자열 | — | 체크포인트와 학습 이력을 둘 곳 |
| `--device` | 문자열 | — | cpu · cuda. 없으면 있는 것을 고른다 |
| `--threads` | 정수 | — | torch 스레드 수. 0 이면 torch 기본값 |
| `--workers` | 정수 | `0` | DataLoader 워커 수 |
| `--source` | `auto` · `synthetic` · `mitdb` | — | config 의 data.source 를 덮어쓴다. 재현성을 위해 명시를 권한다 |
| `--resume` | **켬/끔** | 끔 | 출력 디렉터리의 last.pt 에서 이어서 학습한다 |

### `scripts/run_exp.py`

STEP 24-25: 실험 실행기 (EXP-A / EXP-B / EXP-C).

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `-c` · `--config` | 문자열 | **필수** | 실험 설정 YAML (configs/) |
| `--limit` | 정수 | — | 평가 항목 수 상한 (연습용) |
| `--out` | 문자열 | — | 산출물을 둘 곳 |
| `--source` | `auto` · `synthetic` · `mitdb` | — | config 의 data.source 를 덮어쓴다. 재현성을 위해 명시를 권한다 |

### `scripts/run_seed_sweep.py`

외부 컴퓨팅 자원(Colab · Kaggle · Lightning · 로컬 GPU)에서 **학습 런 간 산포**를 재는 판.

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--stage` | `calib` · `capacity` · `loss` · `spread` · `structure` | — | 미리 정의된 arm 묶음. --arms 와 함께 쓰면 --arms 가 이긴다 |
| `--arms` | 문자열 | — | 쉼표로 구분한 config stem |
| `--seeds` | 문자열 | — | 예: 0-9  또는  0,1,2,5 |
| `--out` | 문자열 | — | 기본 results/ext/<stage 또는 custom> |
| `--source` | `mitdb` · `synthetic` | `mitdb` | clean 신호의 출처. auto 는 파일이 생기면 조용히 바뀐다 — 99_status 2.1 |
| `--device` | 문자열 | — | cuda \| cpu (기본: 있으면 cuda) |
| `--amp` | `on` · `off` | `off` | 기본 off — CPU(fp32) 결과와 같은 계보로 두기 위해서다 |
| `--epochs` | 정수 | — | 빠른 확인용 축소 |
| `--workers` | 정수 | `2` | 동시에 돌릴 학습 수. 올리면 메모리와 코어를 나눠 쓴다 |
| `--threads` | 정수 | — | torch 스레드 수. 이 저장소의 학습은 4 로 돌았다 — 맞춰 두면 부동소수점 축약 순서까지 같아진다 |
| `--keep-ckpt` | **켬/끔** | 끔 | best.pt/last.pt 를 남긴다 (업로드가 커진다) |
| `--dry-run` | **켬/끔** | 끔 | 무엇을 돌릴지만 출력 |

### `scripts/overfit_test.py`

STEP 17 DoD: 배치 1개를 과적합시켜 모델/학습 루프의 버그를 잡는다.

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--model` | 문자열 | `resunet1d` | 과적합시킬 모델 이름 |
| `--loss` | 문자열 | `L1` | 쓸 손실 이름 |
| `--steps` | 정수 | `300` | 몇 step 을 돌릴 것인가 |
| `--batch` | 정수 | `16` | 배치 크기 |
| `--lr` | 실수 | `0.001` | 학습률 |
| `--target` | 실수 | `0.0001` | 이 손실보다 작아지면 통과. **못 내려가면 배관이 틀린 것** |

### `scripts/tune_swt.py`

STEP 11 보조: SWT thresholding 파라미터 튜닝 (docs/00_review.md B-1).

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--dur` | 실수 | `90.0` | 한 구간의 길이 [s] |
| `--tune-seeds` | 정수 | `[0, 1]` | 탐색에 쓸 seed 수 |
| `--holdout-seeds` | 정수 | `[7, 8]` | **탐색에 안 쓴** seed 수. 고른 값을 여기서 확인한다 |
| `--snrs` | 정수 | `[5, 10, 15]` | 쓸 입력 SNR 목록 [dB] |
| `--source` | `synthetic` · `mitdb` | `synthetic` | synthetic 이 기본 — D0 결과(보고서 인용값)와의 연속성을 지킨다 |

### `scripts/run_safety_probe.py`

STEP 26 / EXP-E — 안전성 프로브 (docs/00_review.md C-3, C-4).

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `-c` · `--config` | 문자열 | `configs/exp_e.yaml` | 탐침 설정 YAML (configs/) |
| `--limit` | 정수 | — | 앞에서 N 개만 돌린다 (빠른 확인용) |
| `--source` | `auto` · `synthetic` · `mitdb` | — | config 의 data.source 를 덮어쓴다 |

## 산출물 생성

보고서·그림·슬라이드·시연 카드. 자동 생성 문서 규칙이 붙는다 — §4.

### `scripts/make_report.py`

STEP 27: 모든 표·그림을 한 번에 재생성한다.

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--real-snr` | 실수 | — | 실측 장비 SNR 추정치 [dB]. F4 에 수직선으로 표시. |
| `--fig-snr` | 실수 | `5.0` | 대표 파형 그림(F1-F3)의 입력 SNR [dB]. |
| `--no-waveforms` | **켬/끔** | 끔 | F1 파형 그림을 건너뛴다 (느린 단계다) |
| `--source` | `auto` · `synthetic` · `mitdb` | `auto` | 어느 데이터축의 결과를 읽어 보고서를 만들지 |

### `scripts/make_slides.py`

발표용 시각화 세트.

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--only` | 문자열 | — | S1 S2 ... 일부만 생성 |
| `--snr` | 실수 | `-5.0` | 파형 그림의 입력 SNR (기본 -5 dB — 차이가 보이는 구간) |

### `scripts/make_ablation_table.py`

STEP 19 DoD — loss ablation 표 생성.

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--source` | `auto` · `synthetic` · `mitdb` | `auto` | clean 신호의 출처. auto 는 파일이 생기면 조용히 바뀐다 — 99_status 2.1 |

### `scripts/make_cli_reference.py`

`docs/24_cli_reference.md` 를 만든다 — **실행 스크립트의 옵션 전부.**

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--check` | **켬/끔** | 끔 | 다시 만들지 않고 **낡았는지만** 본다. 낡았으면 1 을 낸다 |
| `--out` | 문자열 | `/home/user/ECG_denoising_method_comparision/docs/24_cli_reference.md` | 쓸 파일. 기본은 docs/24_cli_reference.md |

### `scripts/build_demo_bank.py`

R-1: 모드 B(기존 데이터) 시연용 파형 은행을 **미리** 만든다.

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--axis` | `d0` · `d1` | `['d0', 'd1']` | 데이터 축 (d0 = 합성 · d1 = MIT-BIH). 산출물 경로가 갈린다 |
| `--conds` | 문자열 | — | 잡음 종류 부분집합 (확인용) |
| `--snrs` | 실수 | — | 입력 SNR 부분집합 (확인용) |
| `--out` | 문자열 | `demo/demo_bank.js` | 산출물을 둘 곳 |

### `scripts/build_card_bank.py`

C3 카드용 데이터 은행 — **잡음을 고르면 시간축과 스펙트럼이 함께 바뀐다.**

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--source` | `mitdb` · `synthetic` | `mitdb` | clean 신호의 출처. auto 는 파일이 생기면 조용히 바뀐다 — 99_status 2.1 |
| `--snr` | 실수 | `5.0` | 카드에 쓸 입력 SNR [dB] |
| `--out` | 문자열 | `demo/card_bank.js` | 산출물을 둘 곳 |

### `scripts/build_metric_cards.py`

시연용 **지표 설명 카드** — "SNR 하나로는 왜 안 되는가" 를 여섯 장으로.

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--only` | `C1` · `C2` · `C3` · `C4` · `C5` | — | 이 이름의 카드만 다시 만든다 |

## 검증 · 분석

숫자를 새로 만들지 않고 **이미 있는 것을 확인하거나 재는** 것들.

### `scripts/analyze_loss_by_noise.py`

EXP-G 분석 — **`L6` 의 이득이 잡음 종류별로도 유지되는가.**

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--axis` | 문자열 | `['d0', 'd1']` | 데이터 축 (d0 = 합성 · d1 = MIT-BIH). 산출물 경로가 갈린다 |
| `--out` | 문자열 | `docs/11_loss_by_noise.md` | 산출물을 둘 곳 |

### `scripts/analyze_seed_sweep.py`

`run_seed_sweep.py` 가 만든 표를 읽어 **판정**한다.

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--base` | 문자열 | — | 짝지어 비교할 기준 arm |
| `--targets` | 문자열 | `1.4,0.5,0.25` | 필요 판 수를 계산할 차이 [dB] |

### `scripts/check_ckpts.py`

실험 실행 전 체크포인트 게이트.

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--source` | `auto` · `synthetic` · `mitdb` | `auto` | clean 신호의 출처. auto 는 파일이 생기면 조용히 바뀐다 — 99_status 2.1 |
| `-c` · `--configs` | 문자열 | `['exp_c', 'exp_a', 'exp_b', 'abl_loss', 'abl_window', 'exp_e']` | config 이름 또는 경로 (기본: 실험 러너가 도는 전부) |
| `--ignore-lock` | **켬/끔** | 끔 | 학습이 돌고 있어도 통과시킨다 (의도적 부분 실행용) |
| `--ignore-stale` | **켬/끔** | 끔 | 학습 코드가 바뀐 체크포인트도 통과시킨다 |

### `scripts/check_freshness.py`

산출물이 낡았는지 검사한다 — 생성 코드가 그 뒤로 바뀌었는가.

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--root` | 문자열 | `results` | 검사할 산출물 루트 |
| `--strict` | **켬/끔** | 끔 | 낡은 산출물이 있으면 종료코드 2 |

### `scripts/check_realdata_path.py`

실데이터 경로(STEP 15~17) 예행 검증 — PhysioNet 없이.

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--real` | **켬/끔** | 끔 | fixture 대신 실제 data/raw 를 검사한다 |
| `--root` | 문자열 | `data/raw` | --real 일 때의 경로 |
| `--keep` | **켬/끔** | 끔 | fixture 를 지우지 않는다 |

### `scripts/check_snr_estimator.py`

STEP 08: ground-truth 없는 SNR 추정기의 교정/검증 (docs/00_review.md B-3).

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--dur` | 실수 | `120.0` | 한 구간의 길이 [s] |
| `--n-rep` | 정수 | `3` | 같은 조건을 몇 번 반복할 것인가 |
| `--snrs` | 정수 | `[0, 5, 10, 15, 20]` | 쓸 입력 SNR 목록 [dB] |
| `--source` | `synthetic` · `mitdb` | `synthetic` | synthetic 이 기본 — D0 결과(보고서 인용값)와의 연속성을 지킨다 |

### `scripts/compare_causal_fe.py`

인과 front-end 설계 비교 — **두 대가를 같은 단위로 잰다** (F-27 · D-19).

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--axis` | 문자열 | `['d0', 'd1']` | 데이터 축 (d0 = 합성 · d1 = MIT-BIH). 산출물 경로가 갈린다 |

### `scripts/compare_four_fe.py`

네 방식만 나란히 — **재중심화 없이** 본다.

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--axis` | 문자열 | `['d1', 'd0']` | 데이터 축 (d0 = 합성 · d1 = MIT-BIH). 산출물 경로가 갈린다 |
| `--recenter` | `none` · `dc` · `both` | `both` | none=손대지 않음, dc=직류 1 개만 뺌, both=둘 다 만듦 |

### `scripts/diagnose_sameni.py`

STEP 13 자가진단 — docs/00_review.md A-2 의 6개 실패 원인 점검.

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--data` · `--source` | `synthetic` · `mitdb` | `synthetic` | config 없이 도는 보조 스크립트다. synthetic 이 기본 — D0 결과(이미 보고서에 인용된 값)와의 연속성을 지킨다 |
| `--dur` | 실수 | `90.0` | 한 구간의 길이 [s] |
| `--snr` | 실수 | `5.0` | 입력 SNR [dB] |

### `scripts/estimate_real_snr.py`

STEP 29: 실측 Arduino 신호의 SNR 추정 — **"정말 15 dB 인가?" 에 답한다.**

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--in` | 문자열 | `data/arduino` | 읽을 CSV 가 있는 폴더 |
| `--out` | 문자열 | `results/real_snr` | 산출물을 둘 곳 |
| `--seg-s` | 실수 | `60.0` | 구간 길이. 여러 구간으로 나눠 분포를 본다. |

### `scripts/explore_lookahead_fe.py`

위상 왜곡을 **필터 차수 말고 다른 방법으로** 없앨 수 있는가 (F-27 · D-19 후속).

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--axis` | 문자열 | `['d0', 'd1']` | 데이터 축 (d0 = 합성 · d1 = MIT-BIH). 산출물 경로가 갈린다 |

### `scripts/measure_metric_floor.py`

STEP 07: metric noise floor 측정 (docs/00_review.md A-8).

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--probe-snr` | 실수 | `40.0` | 교란 강도 [dB]. 클수록 약한 교란. |
| `--n-seed` | 정수 | `10` | seed 를 몇 개 쓸 것인가 |
| `--dur` | 실수 | `120.0` | 한 구간의 길이 [s] |
| `--n-record` | 정수 | `5` | 기록을 몇 개 쓸 것인가 |
| `--source` | `synthetic` · `mitdb` | `synthetic` | config 없이 도는 보조 스크립트다. synthetic 이 기본 — D0 결과(이미 보고서에 인용된 값)와의 연속성을 지킨다 |

### `scripts/measure_stream_latency.py`

스트리밍 출력 지연 대 품질 — **재학습 없이** 기존 체크포인트로 잰다.

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--axis` | `d0` · `d1` · `both` | `both` | 데이터 축 (d0 = 합성 · d1 = MIT-BIH). 산출물 경로가 갈린다 |
| `--seeds` | 정수 | `3` | seed 를 몇 개 쓸 것인가 |
| `--out` | 문자열 | `results/stream_latency.json` | 산출물을 둘 곳 |

### `scripts/measure_stream_seam.py`

스트리밍 재생의 이음매와 품질 — 오프라인 경로와 직접 대조한다.

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--axis` | 문자열 | `d1` | 데이터 축 (d0 = 합성 · d1 = MIT-BIH). 산출물 경로가 갈린다 |
| `--seconds` | 실수 | `20.0` | 재는 구간의 길이 [s] |
| `--seeds` | 정수 | `3` | seed 를 몇 개 쓸 것인가 |
| `--out` | 문자열 | `results/stream_seam.json` | 산출물을 둘 곳 |

### `scripts/verify_stream_processor.py`

R-4 검증 — `StreamProcessor` 가 오프라인 경로와 얼마나 다른가.

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--axis` | 문자열 | `['d0', 'd1']` | 데이터 축 (d0 = 합성 · d1 = MIT-BIH). 산출물 경로가 갈린다 |
| `--methods` | 문자열 | `['M06', 'M06L6', 'M04']` | 대조할 방법. 쉼표로 나열한다 |
| `--seconds` | 실수 | `40.0` | 재는 구간의 길이 [s] |
| `--segments` | 정수 | `3` | 구간을 몇 개 볼 것인가 |
| `--grid` | 문자열 | `['12:12', '12:25', '12:64', '50:128']` | d:hop 쌍 |
| `--hp-hz` | 실수 | — | 인과 FE 의 고역통과 차단 [Hz]. 기본은 공통 FE 와 같다(0.5) |
| `--fe-order` | 정수 | — | 인과 FE 의 차수. 기본은 공통 FE 와 같다(4) |
| `--fe-mode` | `causal` · `zerophase` · `median` | `causal` | 실시간 front-end (docs/13 · docs/14) |
| `--force` | **켬/끔** | 끔 | 행이 줄어드는 덮어쓰기를 허용한다 |
| `--out` | 문자열 | — | 기본은 모드별로 다른 파일이다 — 서로 덮어쓰지 않게 (O-20) |

### `scripts/watchdog.py`

학습 러너가 살아 있는지 판정하고, 죽었으면 **증거를 남기고** 재개한다.

| 옵션 | 값 | 기본 | 설명 |
|---|---|---|---|
| `--restart` | **켬/끔** | 끔 | stalled 로 판정되면 재개까지 한다 |
| `--report` | **켬/끔** | 끔 | 그동안의 사건 요약 |
| `--cmd` | 문자열 | — | 재개에 쓸 명령 (기본: run_all_training.sh 전체) |

