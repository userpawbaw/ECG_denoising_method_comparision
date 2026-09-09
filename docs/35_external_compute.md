# 35. 외부 컴퓨팅에서 산포를 재는 판 — 실행 안내

> **왜 필요한가.** F-40 이 학습 런 간 산포를 **sd 0.741 dB(폭 1.43)** 로 쟀는데,
> 그것이 5.9·5.10 의 구조 효과(0.1~0.4 dB)와 D-23 B 의 용량 효과(0.5 dB)보다
> 크다. 그래서 네 번의 「구조 기각」이 **「차이 없음」이 아니라 「못 잼」**으로
> 내려앉았다. 이 저장소의 CPU 로는 팔당 138 판(≈230 시간)이 필요해 낼 수 없다.
> **GPU 가 있으면 이 판정을 되찾을 수 있다.**

이 문서는 Colab · Kaggle · Lightning AI · 개인 GPU 어디서든 그대로 붙여 넣어
돌릴 수 있게 쓴 것이다. 결과는 수 MB 짜리 폴더 하나로 나온다.

---

## 0. 먼저 알아야 할 것

**PhysioNet 을 받을 필요가 없다.** MIT-BIH 와 NSTDB 원본(121 MB)이 이미
`data/raw/` 에 커밋돼 있다(`.gitignore` 의 예외 규칙). `git clone` 한 번이면
데이터까지 전부 온다. **조직 egress 정책을 우회하는 절차가 아니다** — 공개
데이터를 이미 받아 둔 저장소를 그대로 복제하는 것뿐이다.

**개인 생체정보는 애초에 없다.** `data/arduino/` 는 git 에 들어가지 않으므로
외부 환경으로 나가지 않는다.

---

## 1. 붙여 넣을 것 (Colab · Kaggle 공통)

```bash
!git clone --depth 1 -b claude/ecg-denoising-dsp-dl-comparison-b5wjvj \
    https://github.com/userpawbaw/ECG_denoising_method_comparision.git ecgdn
%cd ecgdn
!pip -q install PyWavelets wfdb           # 나머지는 Colab/Kaggle 에 이미 있다
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

> `--depth 1` 이면 85 MB 대신 필요한 것만 받는다. 브랜치 이름을 그대로 써야
> F-40 이후의 코드가 온다.

---

## 2. 단계 — **반드시 이 순서로**

### 2-1. 캘리브레이션 (약 5~15 분) — 이것부터

```bash
!python scripts/run_seed_sweep.py --arms m06_l1,m09_l1 --seeds 0-2 --out results/ext/calib
!python scripts/analyze_seed_sweep.py results/ext/calib
```

이 환경의 `m06_l1` seed 0/1/2 가 이 저장소의 CPU 값
(**3.437 / 4.864 / 4.493 dB**)을 재현하는지 본다. `m09_l1` 도 세 판이 있으므로
(**3.440 / 4.288 / 4.169**) `--arms m06_l1,m09_l1 --seeds 0-2` 로 돌리면
**두 구조에서** 재현성을 볼 수 있다 — 그쪽이 낫다.

| 결과 | 뜻 | 다음 |
|---|---|---|
| 최대 차이 **< 0.05 dB** | 같은 계보다 | 외부·CPU 결과를 **한 표에 합쳐도 된다** |
| 그보다 크다 | 커널·수치정밀도가 다르다 | 합치지 않는다. **외부 집합 안에서만** 비교한다 — 그래도 판정은 유효하다 |

**어느 쪽이든 실험은 계속한다.** 재현이 안 되는 것은 실패가 아니라 **기록할
사실**이다(F-9 와 같은 종류다). 이 표의 두 줄을 그대로 알려 주면 된다.

여기서 **epoch 당 몇 초**가 나오는지도 함께 출력된다. 이 저장소 CPU 는 약 66 초다.
그 배율이 아래 판 수를 정한다.

### 2-2. 산포 (판 수 × 한 판)

```bash
!python scripts/run_seed_sweep.py --stage spread --seeds 0-9
```

`m06_l1` 을 seed 0~9 로 돌린다. sd 의 자유도를 **2 에서 9** 로 올린다 —
신뢰구간이 0.37~1.77 dB(df=4)에서 약 0.51~1.35 dB 로 좁아진다.
캘리브레이션에서 이미 seed 0~2 를 돌렸다면 **3~9 만 추가로 돌면 된다.**

### 2-3. 구조 재판정 — **이 판의 본론**

```bash
!python scripts/run_seed_sweep.py --stage structure --seeds 0-9
!python scripts/analyze_seed_sweep.py results/ext/structure
```

`M06` · `M07` · `M08` · `M09` · `M10` 다섯 구조를 **같은 seed 집합**에서 돌린다.
5 arm × 10 seed = **50 판**. 이것이 5.9·5.10 의 네 번의 기각을
「효과 없음」인지 「못 잼」인지로 확정한다.

### 2-4. 용량 · 손실 (여유가 있으면)

```bash
!python scripts/run_seed_sweep.py --stage capacity --seeds 0-9   # 30 판
!python scripts/run_seed_sweep.py --stage loss     --seeds 0-9   # 30 판
```

---

## 3. 왜 이렇게 설계했나

### 3-1. 짝지은 설계 — 같은 seed 집합을 모든 arm 에 쓴다

독립 표본으로 0.25 dB 를 가르려면 팔당 약 100 판이 필요하다. 그러나 두 arm 이
**같은 seed 에서 함께 높거나 함께 낮다면**, 그 공통 성분은 차이를 빼면 사라진다.

**이건 이제 가설이 아니라 측정이다** — D-23 A 가 `M06`·`M09` 를 같은 세 seed 로
돌려 **ρ = +0.99** 를 냈다(seed 0 에서 3.437 vs 3.440). 짝 이득 **0.33**,
판 수로는 **1/9** 다.

| 재려는 Δ | 독립 표본 | **짝지은 설계** |
|---:|---:|---:|
| 0.50 dB | 24 | **3** |
| **0.25 dB** | **96** | **11** |
| 0.10 dB | 596 | 67 |

**그래서 seed 10 판이면 0.25 dB 가 잡힌다.** 다만 이 ρ 는 **한 쌍**에서 나온
값이라, **다른 arm 쌍에서도 이만큼인지가 이 판의 첫 확인 대상**이다. 분석기가
쌍마다 ρ 와 「짝 이득」을 출력한다.

**추가 비용이 0 이다.** 어차피 각 arm 을 K 판 돌릴 것이고, seed 를 맞춰
돌리기만 하면 된다.

### 3-2. 고정 평가 — 산포를 두 몫으로 가른다

`best_metric` 은 **seed 마다 다른 val 잡음 뽑기**(`salt=("val", seed)`) 위에서
고른 값이다. 그래서 1.43 dB 안에는 두 가지가 섞여 있다.

| 섞인 것 | 무엇인가 |
|---|---|
| **모델이 달라서** | 초기화 · 배치 순서가 다른 모델이 나온다 |
| **val 뽑기가 운이 좋아서** | 같은 모델이라도 쉬운 잡음을 뽑으면 높게 나온다 |

그래서 학습이 끝나면 best 체크포인트를 **seed 와 무관한 고정 잡음 뽑기**
(`salt=("fixed_eval", 20260909)`)로 한 번 더 잰다. 두 산포를 비교하면 몫이 갈린다.
**「모델이 얼마나 흔들리나」는 고정 평가 쪽 숫자로 말해야 한다.**

### 3-3. AMP 는 기본 꺼져 있다

GPU 에서 `torch.amp` 를 켜면 fp16 으로 학습한다. 그 값은 CPU 의 fp32 값과
**같은 표에 올릴 수 없다.** 기본을 `--amp off` 로 두어 계보를 하나로 유지했다.
속도가 문제면 `--amp on` 을 쓰되, **그때는 CPU 결과와 합치지 않는다**
(manifest 에 `amp` 가 남으므로 나중에 갈라낼 수 있다).

---

## 4. 중단됐을 때

**같은 명령을 다시 실행하면 된다.** `summary.json` 이 있는 판은 건너뛴다.
Colab 이 끊겨도, 세션이 죽어도 마찬가지다.

Colab 이라면 결과를 Drive 에 두는 쪽이 안전하다:

```python
from google.colab import drive; drive.mount('/content/drive')
!python scripts/run_seed_sweep.py --stage structure --seeds 0-9 \
    --out /content/drive/MyDrive/ecgdn_sweep/structure
```

---

## 5. 돌려줄 것

```bash
!zip -qr sweep_structure.zip results/ext/structure
!ls -lh sweep_structure.zip
```

체크포인트는 기본적으로 지우므로 **수 MB** 로 끝난다
(`--keep-ckpt` 를 주면 커진다 — 줄 필요 없다).

폴더 안에 있는 것:

| 파일 | 무엇 |
|---|---|
| `sweep.csv` | **이 한 장이면 분석이 된다** — 판마다 한 줄 |
| `<arm>__s<seed>/summary.json` | best · 고정 평가 · 벽시계 시간 · 환경 지문 |
| `<arm>__s<seed>/history.json` · `log.csv` | epoch 별 곡선 |
| `<arm>__s<seed>/manifest.json` | 학습 조건과 **코드 해시** (F-9) |

`sweep.csv` 만 붙여 넣어도 판정은 나온다. 폴더째면 더 좋다.

---

## 6. 판 수를 고를 때

한 판이 `t` 분이면 `arm 수 × seed 수 × t` 분이다. 캘리브레이션이 `t` 를 알려 준다.

| 한 판 | structure 5 arm × 10 seed | × 20 seed |
|---:|---:|---:|
| 3 분 | 2.5 시간 | 5 시간 |
| 5 분 | 4 시간 | 8 시간 |
| 10 분 | 8 시간 | 17 시간 |

**seed 를 나눠 여러 번 돌려도 된다** — `--seeds 0-4` 로 한 번, 나중에
`--seeds 5-9` 로 한 번. 같은 `--out` 을 쓰면 표가 합쳐진다.

---

## 7. 이 판이 못 답하는 것

- **테스트축 결과가 아니다.** 여기서 재는 것은 **val** 의 `snr_imp_scaled` 다.
  주 결과표(`90_results_d1.md`)의 테스트 지표가 아니므로 그 표를 대체하지 않는다.
  이 판의 용도는 **「차이를 가릴 수 있나」** 하나다.
- **CPU 산포와 GPU 산포가 같다는 보장이 없다.** 2-1 이 그것을 잰다.
- **`set_epoch()` 은 그대로다.** train 쪽 잡음 실현은 seed 를 안 따르므로
  (`self.salt = ("epoch", epoch)`), 여기서 재는 산포도 **하한**이다. 이것을
  고치면 F-40 의 세 판과 비교할 수 없게 되므로 이 판에서는 건드리지 않았다.
