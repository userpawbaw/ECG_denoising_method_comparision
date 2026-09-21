# 35. 외부 컴퓨팅에서 산포를 재는 판 — 실행 안내

> **왜 필요한가.** F-40 이 학습 런 간 산포를 **sd 0.741 dB(폭 1.43)** 로 쟀는데,
> 그것이 5.9·5.10 의 구조 효과(0.1~0.4 dB)와 D-23 B 의 용량 효과(0.5 dB)보다
> 크다. 그래서 네 번의 「구조 기각」이 **「차이 없음」이 아니라 「못 잼」**으로
> 내려앉았다. 이 저장소의 CPU 로는 팔당 138 판(≈230 시간)이 필요해 낼 수 없다.
> **GPU 가 있으면 이 판정을 되찾을 수 있다.**

> **옵션 전부는 `docs/24_cli_reference.md`.** 아래 명령은 흔한 쓰임만 보이고, 각 스크립트가 받는 옵션·기본값·선택지는 그 문서에 자동 생성으로 모여 있다 (`--help` 도 같은 내용을 찍는다).

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

## 2. 지금 돌릴 것 — **조건화 · 블록 두 판** (2026-09-13 기준)

> **D-27 깊이 판은 끝났다** — seed 0~3 을 받아 K=6 을 채웠다. 결론은 **F-44**
> (깊이는 값을 하지 않고, 저 SNR 에서는 유의하게 손해다). 아래 §2.0 이 다음에
> 돌릴 두 판이고, §2.1 부터는 끝난 D-27 의 기록이다.

### 2.0 다음 두 판 — 순서대로

> **받은 것**: 깊이(D-27 · F-44) · 조건화(D-28 2 · **F-45**) · 블록(D-28 5).
> 셋 다 K=6 으로 닫혔다. 아래가 남은 둘이다.

```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")     # 🔑 Secrets 에 넣어 둔 것
```
```bash
!pip -q install -U huggingface_hub
!hf auth whoami || huggingface-cli whoami
```

**분담은 그대로다** — Colab 이 seed 0~3, 이 컨테이너가 4·5.

**① D-28 3 번 2×2** (4 arm × seed 0~3 = 12 판) — **이쪽이 우선이다**

```bash
!python scripts/run_seed_sweep.py --stage l6cond --seeds 0-3 \
    --out /content/ecgdn_sweep/l6cond \
    --hf-repo userpaw/ecg_data --hf-every 10
!python scripts/analyze_seed_sweep.py /content/ecgdn_sweep/l6cond
```

네 칸은 `m06_l1`(둘 다 없음) · `m06_l6`(손실만) · `m06_cond`(조건만) ·
`m06_cond_l6`(둘 다). 묻는 것은 **「조건화가 있으면 `L6` 의 이득이 줄어드는가」**
이고, 그것이 **동등성 주장**이라 마진을 사전에 선언해 뒀다 — **|Δ| < 0.5 dB**.
동등성은 차이보다 비싸서 **K=8 이 필요하다.** 이 컨테이너 몫(seed 4·5)에서는
상호작용 −0.883 이 나왔지만 그 seed 들에서 과장되는 값이라 판정 전이다.

> **끝났다 (K=6, 24 판) → F-47.** 로컬 기계가 멈춰 seed 0~3 도 이 컨테이너에서
> 돌렸다. 답은 **「같다」가 아니라 「겹친다」** 다 — 15~20 에서 상호작용
> **−1.688** (95 % CI [−2.407, −0.969]) 로 마진을 **배제**하고, `L6` 위에
> 조건화를 더해 얻는 것은 총합 −0.033 (p=0.75) 이다. 총합 상호작용(−0.320)의
> 동등성은 **선언하지 못한다** — TOST(±0.5) p=0.192, n≈20 이 필요하다.

**② D-28 4 번 조건 교란** (1 arm × seed 0~3 = 4 판)

```bash
!python scripts/run_seed_sweep.py --arms m06_cond --seeds 0-3 \
    --out /content/ecgdn_sweep/condpert8 \
    --cond-perturb --cond-offsets=-8,-5,-2,2,5,8 \
    --hf-repo userpaw/ecg_data --hf-every 10
```

> `--cond-offsets` 앞의 **`=` 를 빼면 안 된다.** 값이 `-` 로 시작해서
> argparse 가 옵션으로 읽는다.

**학습을 다시 하는 게 아니라** 같은 가중치를 **틀린 조건값**으로도 평가한다.
묻는 것은 「조건 오차 X dB → 손실 Y dB」이고, 거기에 F-13 의 추정기 편향
(**+7.59 dB, 과대추정**)을 대입하면 추정 SNR 의 실무 비용이 나온다.
격자를 ±8 까지 넓힌 이유가 그 지점이다.

**끊기면 같은 명령을 그대로 다시 실행한다.** `--hf-repo` 를 주면 `/content` 가
비어 있어도 된다(**§4.2**).

### 2.1 끝난 판 — D-27 깊이 (기록)

### 분담

| | seed | 판 |
|---|---|---:|
| **이 컨테이너 (CPU)** | **4 · 5** | 8 |
| **Colab (T4)** | **0 · 1 · 2 · 3** | 16 |
| 합계 | 4 arm × 6 seed | **24** |

**한 seed 의 모든 arm 은 반드시 같은 환경에서** 돈다. 갈라지면 짝지은 차분이
하드웨어 흩어짐(sd 0.184)을 먹는다. 그래서 seed 로 나눴지 arm 으로 나누지 않았다.

### 붙여 넣을 것

```python
from google.colab import drive; drive.mount('/content/drive')
```

```bash
!python scripts/run_seed_sweep.py \
    --arms m06_l1,m06_l1_deep2,m06_l1_deep3,m06_l1_deep4 \
    --seeds 0-3 --out /content/drive/MyDrive/ecgdn_sweep/depth
!python scripts/analyze_seed_sweep.py /content/drive/MyDrive/ecgdn_sweep/depth
```

**끊기면 같은 명령을 그대로 다시 실행한다** (§4 — 이제 판 안에서도 이어진다).

### 무엇을 묻는 판인가

**파라미터를 976 K 에 맞춰 두고 폭을 깎아 깊이를 산다.**

| arm | 레벨당 ResBlock | conv 층 | params | 수용영역 |
|---|---:|---:|---:|---:|
| `m06_l1` (기준) | 1 | 34 | 976,489 | 887 (3.55 s) |
| `m06_l1_deep2` | 2 | 50 | 970,474 | 1127 |
| `m06_l1_deep3` | 3 | 66 | 979,574 | 1367 |
| `m06_l1_deep4` | 4 | 82 | 966,846 | 1607 |

용량 실험(F-42)은 **폭만** 바꿨다. 이 판이 그 범위 단서를 닫는다. 근거와
사전 예측은 `docs/21` **D-27**, 선행 연구 위치는 `docs/36_related_work.md`.

### 그다음 — **D-28 2 번 조건화 판** (구현 끝, 바로 돌려도 된다)

깊이 판을 마치면 이어서 이것을 돌린다. **분담은 같다** — Colab 이 seed 0~3,
이 컨테이너가 4·5.

```bash
!python scripts/run_seed_sweep.py \
    --arms m06_l1,m06_cond \
    --seeds 0-3 --out /content/drive/MyDrive/ecgdn_sweep/cond
!python scripts/analyze_seed_sweep.py /content/drive/MyDrive/ecgdn_sweep/cond
```

`m06_l1` 을 다시 도는 게 낭비로 보이지만 **짝을 같은 판에서 만들어야** 한다.
덤도 있다 — 깊이 판의 같은 seed 값과 맞아떨어지는지가 공짜 재현 확인이 된다.

| arm | 무엇 | params |
|---|---|---:|
| `m06_l1` (기준) | 조건 없음 | 976,489 |
| `m06_cond` | 참 SNR → FiLM (γ, β) | 974,693 (−0.18 %) |

사전 예측: **15~20 dB 대역에서 +3 ~ +5 dB.** D-26 의 전담 모델(+4.41)이
상한이고 조건화는 그 아래일 것이다. **상한을 넘으면 평가 누수부터 의심한다.**
근거·구현 선택은 `docs/21` **D-28**.

그 뒤로는 2×2(`L6` × 조건화) → 추정 SNR + 교란. 둘 다 구현이 더 필요하다.

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

### 3-2. 고정 평가 — 산포를 두 몫으로 가른다  **← 이것이 판을 살렸다 (F-41)**

`best_metric` 은 **seed 마다 다른 val 잡음 뽑기**(`salt=("val", seed)`) 위에서
고른 값이다. 그래서 1.43 dB 안에는 두 가지가 섞여 있다.

| 섞인 것 | 무엇인가 |
|---|---|
| **모델이 달라서** | 초기화 · 배치 순서가 다른 모델이 나온다 |
| **val 뽑기가 운이 좋아서** | 같은 모델이라도 쉬운 잡음을 뽑으면 높게 나온다 |

그래서 학습이 끝나면 best 체크포인트를 **seed 와 무관한 고정 잡음 뽑기**
(`salt=("fixed_eval", 20260909)`)로 한 번 더 잰다.

**2-1 이 그 크기를 쟀다: sd 0.557 → 0.198, 2.82 배.** 1.43 dB 산포의 대부분은
모델이 아니라 **각자 다른 시험지로 채점한 결과**였다. 고정 평가의 0.198 은
하드웨어 재현 바닥(0.184)에 거의 붙어 있다 — **공통 자로 재면 세 seed 의 모델은
사실상 구별되지 않는다.**

**「모델이 얼마나 흔들리나」는 고정 평가 쪽 숫자로 말해야 한다.**

### 3-3. AMP 는 기본 꺼져 있다

GPU 에서 `torch.amp` 를 켜면 fp16 으로 학습한다. 그 값은 CPU 의 fp32 값과
**같은 표에 올릴 수 없다.** 기본을 `--amp off` 로 두어 계보를 하나로 유지했다.
속도가 문제면 `--amp on` 을 쓰되, **그때는 CPU 결과와 합치지 않는다**
(manifest 에 `amp` 가 남으므로 나중에 갈라낼 수 있다).

---

## 4. 중단됐을 때 — **이제 판 안에서도 이어진다**

두 층으로 재개한다.

| 무엇이 끊겼나 | 어떻게 이어지나 |
|---|---|
| **판 사이** (한 판 끝나고 다음 판 전에) | `summary.json` 이 있는 판은 건너뛴다 |
| **판 안** (학습 도중 30 epoch 쯤에서) | **`last.pt` 에서 그 epoch 부터 이어 간다** |

**`last.pt` 는 epoch 마다 저장되고** optimizer 모멘텀 · LR 스케줄 위치 ·
`best_epoch` 까지 담는다. 그래서 「이어서」가 「다른 학습」이 되지 않는다.
지난번엔 이게 연결돼 있지 않아 **한 판을 통째로 다시 돌렸다.**

**할 일은 없다. 같은 명령을 다시 실행하면 된다.**

```bash
!python scripts/run_seed_sweep.py --arms ... --seeds ... --out ... --dry-run
```

`--dry-run` 이 상태를 세 가지로 보여 준다:

```
  ✓ m06_l1__s0                                        끝났다
  ↻ m07_l1__s0   epoch 23 까지 갔다가 끊겼다 — 거기서 재개한다
    m08_l1__s0                                        아직 안 돌았다
```

> **`epochs` 를 바꾸지 마라.** LR 스케줄이 `epochs × step/epoch` 로 정규화돼
> 있어서, 재개할 때 값을 바꾸면 남은 구간의 LR 곡선이 원래와 달라진다.
> 처음부터 다시 돌리고 싶으면 `--no-resume` 을 준다.

**재개는 `last.pt` 가 살아 있어야 된다.** Colab 로컬 디스크(`/content`)는 세션이
죽으면 같이 사라지므로, **재개할 것을 세션 밖에 둬야** 한다. 두 길이 있다 —
Drive 마운트(§4.1)와 Hugging Face 동기화(**§4.2**).

### 4.1 Drive 마운트

```python
from google.colab import drive; drive.mount('/content/drive')
!python scripts/run_seed_sweep.py --arms m06_l1,m06_cond \
    --seeds 0-3 --out /content/drive/MyDrive/ecgdn_sweep/cond
```

간단하지만 **마운트 자체가 끊긴다.** 끊긴 뒤의 쓰기는 조용히 로컬 디스크로
떨어지고, 세션이 죽으면 그것도 사라진다.

### 4.2 Hugging Face 동기화 — `--hf-repo`

**세션과 무관한 곳에 둔다.** 시작할 때 받아 오고, N epoch 마다 `last.pt` 를
올리고, 판이 끝나면 그 판을, sweep 이 끝나면 **전체 출력**을 올린다.

**준비** (한 번만). HF 에서 **dataset** 저장소를 만들고 터미널에 토큰을 넣은
뒤라면, Colab 셀에서는 이것만 하면 된다:

```bash
!pip -q install -U huggingface_hub
!hf auth whoami || huggingface-cli whoami     # 토큰이 들어 있는지 확인
# !hf auth login          (구버전이면 huggingface-cli login)
```

> 최근 `huggingface_hub` 는 명령을 `hf` 로 바꿨고 `huggingface-cli` 는 경고와
> 함께 남겨 뒀다. 어느 쪽이 되는지 몰라서 위처럼 둘 다 걸어 뒀다.

> Colab 은 세션마다 새 기계라 로그인이 남지 않는다. 매번 치기 싫으면
> **Colab 비밀(🔑 Secrets)** 에 `HF_TOKEN` 을 넣고 「노트북 접근」을 켠 뒤:
> ```python
> import os
> from google.colab import userdata
> os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
> ```
> `huggingface_hub` 가 `HF_TOKEN` 을 알아서 읽는다. **토큰은 쓰기 권한**이어야
> 올라간다.

**돌리기** — 기존 명령에 두 줄만 붙인다:

```bash
!python scripts/run_seed_sweep.py --arms m06_l1,m06_cond --seeds 0-3 \
    --out /content/ecgdn_sweep/cond \
    --hf-repo userpaw/ecg_data --hf-every 10
```

**끊기면 똑같은 명령을 다시 실행한다.** 이번엔 `/content` 가 비어 있어도 된다 —
시작할 때 HF 에서 받아 오므로, 끝난 판은 건너뛰고 끊긴 판은 **올라가 있던
epoch 에서** 이어 간다.

| 언제 | 무엇이 올라가나 | 어디에 |
|---|---|---|
| 시작 전 | *(받아 온다)* | → `--out` 폴더, **있는 파일은 안 덮는다** |
| `--hf-every` 마다 | `last.pt` · `log.csv` | `<저장소>/<out 폴더이름>/<판>/` |
| 판 하나 끝 | 그 판 폴더 전부 | 〃 (죽은 `last.pt` 는 지운다) |
| sweep 끝 | **전체 출력** (`sweep.csv` 포함) | `<저장소>/<out 폴더이름>/` |

저장소 안의 경로는 `--out` 폴더 이름을 접두어로 쓴다 — `--out .../cond` 면
`cond/m06_cond__s0/...` 다. **판이 다른 sweep 끼리 안 섞인다.**

**왜 이렇게 짰나 — 몇 가지 결정**

- **올리기 실패는 학습을 죽이지 않는다.** 모든 통신은 예외를 삼키고 경고만
  찍는다. 30 분짜리 판을 네트워크 딸꾹질로 잃는 것이 원래 막으려던 일이다.
- **받아올 때 있는 파일은 덮지 않는다.** 돌고 있는 기계의 것이 최신이고 HF
  것은 지난 세션의 흔적이다. 반대로 덮으면 이번 세션의 진행을 지운다.
- **`--hf-every 1` 은 권하지 않는다.** HF 는 git 이라 이력이 쌓인다.
  `last.pt` 는 optimizer 상태까지 담아 **모델 크기의 약 3 배**(976 K 모델이면
  10 MB 대)이고, 매 epoch 올리면 판 하나에 50 번이 쌓인다.
- **저장소는 기본 비공개**로 만든다(`--hf-public` 으로 뒤집을 수 있다).
  **`data/arduino/` 같은 실측 생체신호는 어느 쪽으로도 올리지 않는다** —
  이 sweep 이 올리는 것은 MIT-BIH 로 학습한 산출물뿐이다.
- **`--hf-repo` 를 안 주면 아무 일도 안 일어난다.** `huggingface_hub` 가 없어도
  된다 — 늦게 부른다.

**받은 것을 여기로 가져올 때**는 zip 없이 바로 된다:

```bash
python3 -c "
from huggingface_hub import snapshot_download
import shutil, pathlib
p = snapshot_download('userpaw/ecg_data', repo_type='dataset',
                      allow_patterns=['cond/**'])
shutil.copytree(pathlib.Path(p)/'cond', 'results/ext/cond', dirs_exist_ok=True)
"
```

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

F-41 이후 필요한 판 수가 이렇다 — **`fixed_snr_imp_scaled` 기준**(sd 0.198):

| 재려는 Δ | 팔당 판 수 (독립 표본) |
|---:|---:|
| 0.50 dB | 3 |
| **0.25 dB** | **10** |
| 0.10 dB | 63 |

`best` 로 재면 같은 0.25 dB 에 **78 판**이 필요하다. 자를 바꾼 것만으로 8 배다.

**한 판의 시간**: T4 에서 36 분이었는데, 파이프라인 캐시(2.31 배)를 넣었으니
**약 18 분**으로 내려갈 것이다. 캘리브레이션 없이도 첫 판의 `sec_per_epoch`
출력이 실측을 알려 준다.

| 한 판 | structure 5 arm × 4 seed | × 8 seed |
|---:|---:|---:|
| 18 분 | 6 시간 | 12 시간 |
| 25 분 | 8 시간 | 17 시간 |

**seed 를 나눠 여러 번 돌리는 것이 정석이다** — `--seeds 0-3` 한 번, 나중에
`--seeds 4-7` 한 번. 같은 `--out` 을 쓰면 표가 합쳐지고, **중간에 멈춰도 그때까지
나온 판으로 분석이 된다.**

> **폴더를 꼭 챙길 것.** 지난번엔 Colab 이 끊겨 `sweep.csv` 에 한 줄만 남았다.
> Drive 에 두면(§4) 세션이 죽어도 남는다. 최소한 **`sweep.csv` 한 장**은 챙긴다.

## 7. 이 판이 못 답하는 것

- **테스트축 결과가 아니다.** 여기서 재는 것은 **val** 의 `snr_imp_scaled` 다.
  주 결과표(`90_results_d1.md`)의 테스트 지표가 아니므로 그 표를 대체하지 않는다.
  이 판의 용도는 **「차이를 가릴 수 있나」** 하나다.
- **CPU 산포와 GPU 산포가 같다는 보장이 없다.** 2-1 이 그것을 잰다.
- **`set_epoch()` 은 그대로다.** train 쪽 잡음 실현은 seed 를 안 따르므로
  (`self.salt = ("epoch", epoch)`), 여기서 재는 산포도 **하한**이다. 이것을
  고치면 F-40 의 세 판과 비교할 수 없게 되므로 이 판에서는 건드리지 않았다.
