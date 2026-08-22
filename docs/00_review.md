# 00. 기존 제안 검토서

> 대상: 프로젝트 착수 전 AI와 나눈 4차례 대화(① DSP 방식 탐색, ② 딥러닝 도입, ③ 모델 선택 및 hybrid, ④ 평가지표)
> 목적: 확정 설계로 넘어가기 전에 **그대로 가져갈 것 / 고칠 것 / 빠진 것**을 분리한다.
> 원칙: 파트를 억지로 채우지 않는다. 실제로 지적할 내용이 있는 항목만 쓴다.

---

## 총평

대화의 **큰 흐름(classical → time-frequency → model-based → DL → hybrid, 그리고 평가를 SNR 단독으로 하지 않기)** 은
그대로 채택할 만하다. 특히 아래 세 가지는 판단이 정확했다.

- 실측 신호는 clean reference가 없으므로 **synthetic(통제) 평가와 real(비통제) 평가를 분리**해야 한다.
- **record 단위 split**이 필요하다 (window 단위 split은 leakage).
- `SNR`만으로 순위를 매기면 **과도한 smoothing이 이기는** 잘못된 결론이 나온다.

반면 **"따라 하면 결과가 나오는" 수준으로 내려오면 아직 결정되지 않은 항목이 많고, 몇 군데는 그대로 구현하면 결과가 틀어진다.**
아래에 세 파트로 나눈다.

---

# A. 지적할 부분 (그대로 두면 결과가 틀어지는 것)

## A-1. `SNR improvement` 계산에 **scale bias 보정 규약**이 없다 — 이 프로젝트에서 가장 위험한 항목

대화에서 준 식은 옳다.

```
SNR_in  = 10*log10( P(x)   / P(y - x) )
SNR_out = 10*log10( P(x)   / P(x̂ - x) )
```

문제는 **`x̂`에 체계적인 이득(gain) 오차가 붙는 방법이 섞여 있다**는 점이다.

- **soft threshold는 구조적으로 진폭을 줄인다.** `sign(D)·max(|D|-λ, 0)` 는 살아남은 계수에서도 항상 `λ` 만큼을 뺀다(shrinkage bias).
  → SWT 출력은 clean 대비 평균적으로 조금 작다.
- Kalman/EKS는 process noise `Q` 설정에 따라 진폭이 눌린다.
- DL(MSE 학습)도 불확실한 구간에서 평균 쪽으로 수축한다.

이 상태로 위 식을 그대로 쓰면 **"파형은 정확한데 크기만 3% 작은 출력"이 "파형이 3% 틀어진 출력"과 동일하게 처벌된다.**
방법 간 순위가 실제로 뒤집힐 수 있다.

**확정 규약(본 프로젝트에서 강제):** 두 값을 **모두** 리포트한다.

| 이름 | 정의 | 의미 |
|---|---|---|
| `SNR_out_strict` | `x̂` 그대로 사용 | "그대로 써도 되는가" (실사용 기준) |
| `SNR_out_scaled` | `α* = <x, x̂>/<x̂, x̂>` 로 보정 후 | "파형 구조 자체는 맞는가" (알고리즘 기준) |

그리고 **`α*` 자체를 `gain_bias` 지표로 표에 넣는다.** `α* = 1.05` 는 "출력이 5% 작다"는 뜻이고,
이건 감점 요인이 아니라 **보고해야 할 특성**이다. 대화에는 이 구분이 아예 없었다.

## A-2. Sameni EKF/EKS가 "안 되는" 원인 진단이 생략된 채 방법 교체로 넘어갔다 — **공정 baseline 문제**

대화는 "실측 노이즈 종류가 논문과 달라서"로 정리했다. 그럴 수 있지만, **비교연구에서 이건 치명적이다.**
Sameni를 약한 상태로 두고 "SWT가 이겼다"고 결론내면 그 결론 자체가 무효다.

Sameni 계열 구현이 실패하는 원인은 실제로는 거의 항상 아래 6개 중 하나다. **방법을 바꾸기 전에 체크리스트로 확인해야 한다.**

| # | 실패 원인 | 증상 | 확인 방법 |
|---|---|---|---|
| 1 | R-peak 검출 오류 → **위상 `θ` 가 어긋남** | 출력이 원신호와 무관하게 흔들림 | `θ(t)` 를 시간축에 그려서 `-π→π` 톱니가 매 beat 정확히 1회인지 확인 |
| 2 | **baseline wander를 EKF 앞단에서 제거하지 않음** | state model이 baseline을 설명 못 해 발산 | HPF 적용 전/후 비교 |
| 3 | Gaussian kernel 파라미터 `(α_i, b_i, θ_i)` 를 **phase-averaged template에 fitting하지 않고 논문 기본값 사용** | 출력이 "다른 사람의 ECG" 모양 | 추정 template과 fitting 결과를 겹쳐 그림 |
| 4 | `Q`, `R` 을 **실측 noise로 추정하지 않고 임의값** | 개선 폭이 0 근처 | `R` 을 등전위 구간(TP segment) 분산으로 추정 |
| 5 | **EKS(backward smoothing)를 구현하지 않고 EKF만 사용** | 논문 대비 3~5 dB 부족 | EKF와 EKS를 둘 다 리포트 |
| 6 | 진폭 정규화 누락 (모델은 `z`의 스케일을 가정) | 상동 | 입력 정규화 후 역정규화 |

**본 프로젝트에서는 `STEP 06`에서 이 6개를 자가진단 스크립트로 전부 확인하고, 그 결과 자체를 문서 산출물로 남긴다.**
Sameni가 그래도 안 되면, 그때는 "왜 안 되는가"가 결과가 된다. 지금처럼 "잘 안 되더라"는 결과가 아니다.

## A-3. `rbio3.9 / level 5 / threshold scale 0.5` 를 그대로 쓰라는 조언은 위험하다

이 값은 **특정 논문의 특정 샘플링 주파수**에 묶인 값이다. wavelet level은 물리적으로 주파수 대역에 대응한다.

`fs = 250 Hz` 기준 SWT level ↔ 대략적 통과대역:

| Band | 대역 (Hz) | 주 내용 |
|---|---|---|
| D1 | 62.5 – 125 | 거의 전부 noise (EMG 고주파, ADC/front-end) |
| D2 | 31.2 – 62.5 | **60 Hz PLI + EMG**, QRS 고주파 꼬리 일부 |
| D3 | 15.6 – 31.2 | **QRS 주 에너지** |
| D4 | 7.8 – 15.6 | **QRS 주 에너지 + T wave 상단** |
| D5 | 3.9 – 7.8 | P / T wave |
| A5 | 0 – 3.9 | baseline, ST level |

여기서 바로 두 가지가 따라 나온다.

1. **`level`은 `fs`에 종속이다.** 360 Hz에서 level 5는 250 Hz에서 level 5와 다른 대역을 자른다.
   본 프로젝트는 `fs=250`으로 통일하므로, 위 표를 기준으로 재설계한다.
2. **모든 level에 같은 threshold를 쓰면 안 된다.** D1/D2는 공격적으로, **D3/D4는 보수적으로** 가야 한다.
   D3/D4를 세게 자르면 그게 바로 "QRS가 뭉개지는" 현상이다.

→ 본 프로젝트의 SWT는 **level-dependent threshold + QRS 구간 보호**를 기본형으로 한다 (B-1 참조).
`rbio3.9`는 후보 중 하나로 두고 wavelet 선택 자체를 ablation에 넣는다.

## A-4. `PRD` 정의가 하나로 확정되지 않았다

대화는 `PRD = 100·sqrt(Σ(x-x̂)²/Σx²)` 를 쓰고 "DC에 민감하다"고 지적했다. 지적은 맞는데 **해결책이 안 나왔다.**

문헌에는 두 정의가 공존한다.

```
PRD1 (raw)          = 100 * sqrt( Σ(x-x̂)² / Σ x²        )
PRDN (mean-removed) = 100 * sqrt( Σ(x-x̂)² / Σ (x-x̄)²   )
```

MIT-BIH의 원 신호는 **ADC offset(대략 1024 LSB)이 얹혀 있어** `PRD1`은 실제 오차보다 항상 작게 나온다.
**본 프로젝트는 `PRDN`만 사용하고, 표기도 `PRDN`으로 고정한다.** `PRD1`은 계산하지 않는다.

## A-5. 윈도우 경계 처리(overlap-add)가 빠져 있다

"1024 샘플 window로 자른다"까지만 있고, **긴 신호를 window로 잘라 처리한 뒤 다시 붙이는 규약이 없다.**
그대로 구현하면:

- window 경계마다 불연속(클릭) 발생 → PSD 고주파가 인위적으로 올라감
- DL은 경계에서 receptive field가 잘려 성능이 나쁨 → 그 구간이 metric을 오염

**확정 규약:** 모든 방법(DSP·DL 공통)에 **hop = window/2 (50% overlap) + Hann 제곱근 cross-fade** 로 overlap-add.
Hann 창의 50% overlap COLA 성질을 이용하면 정확히 복원된다. 경계 window는 edge-pad 후 잘라낸다.
**동일 규약을 모든 방법에 적용해야 비교가 공정하다.**

## A-6. 진폭 정규화 전략이 없다 — 그대로 두면 morphology 지표를 계산할 수 없다

DL 입력에 per-window z-score를 쓰면 편하지만, **그 순간 R-peak 진폭 정보가 사라진다.**
그러면 `R-peak amplitude error`, `ST level` 같은 지표를 계산할 수 없다.

**확정 규약:**
```
scale = robust_scale(y_window)      # 예: MAD 기반 또는 정해진 percentile
x_in  = y_window / scale            # 모델 입력
x̂     = model(x_in) * scale         # 반드시 역정규화하여 원 스케일 복원
```
`scale`은 **noisy 입력에서만** 계산한다(clean을 보면 정보 누설). window별로 저장·복원한다.

## A-7. NSTDB noise의 train/test 분리가 언급되지 않았다 — **noise leakage**

대화는 ECG record split만 다뤘다. 그런데 `bw/ma/em`은 각 30분짜리 **단 3개의 기록**이다.
여기서 랜덤 crop을 하면 train과 test가 **같은 noise 파형 구간**을 볼 수 있다.

**확정 규약:** noise 기록도 시간축으로 **disjoint 분할** 한다.
```
noise[  0% : 60%]  -> train
noise[ 60% : 75%]  -> val
noise[ 75% :100%]  -> test
```

## A-8. `QRS duration error`를 주 지표로 올리기 전에 **지표 자체의 오차(noise floor)** 를 재야 한다

QRS onset/offset 자동 검출(NeuroKit2 DWT 등)은 clean 신호에서도 자체 오차가 있다.
**방법 간 차이가 delineator 자체 오차보다 작으면 그 표는 아무 의미가 없다.**

**확정 규약:** clean reference에 delineator를 돌려 얻은 값과, clean에 아주 약한 교란(예: 40 dB AWGN)을 준 뒤 얻은 값의 차이를
`metric noise floor`로 먼저 측정해 문서에 박아 둔다. 그 값보다 작은 차이는 **"구분 불가"로 표기**한다.

## A-9. Beat-template 정렬 기준이 정해지지 않으면 timing error가 숨는다

"beat template correlation"을 제안했는데, **어느 R-peak로 정렬하느냐**에 따라 값이 완전히 달라진다.

- denoised 신호의 **자체 R-peak**로 정렬 → timing error가 상쇄되어 사라짐 (부정확)
- **clean reference의 R-peak 위치**로 양쪽 모두 정렬 → timing error가 correlation에 반영됨 (정확)

**확정 규약: 후자.** synthetic 실험에서는 항상 reference R-peak로 양쪽을 자른다.

## A-10. 각 방법에 공통 front-end를 줄지 말지가 정해지지 않았다

`SWT` 앞에만 notch를 붙이고 `U-Net`에는 안 붙이면 불공정하다. 반대도 마찬가지다.

**확정 규약:** `common front-end`(zero-phase HPF + 선택적 notch)를 **정의하고 모든 방법에 동일 적용**한다.
추가로 **front-end 유무 2조건**을 모두 돌려서, "front-end가 각 방법에 얼마나 기여했는가"를 별도 표로 낸다.

## A-11. 용어 정정 (작지만 문서에 남으면 곤란한 것)

- SWT는 **shift-invariant가 아니라 shift-equivariant(translation-equivariant)** 다. 신호를 밀면 계수도 같이 밀린다.
  "계수가 안 변한다"가 아니라 "같이 밀린다"가 정확하다.
- `PRD`는 morphology 지표가 아니라 **정규화된 재구성 오차**다 (대화 후반에 스스로 정정한 것이 맞다).
- PyWavelets의 `swt`는 신호 길이가 **`2^level`의 배수**여야 한다. `1024 = 2^10` 이므로 level 5까지 안전하다.
  실측 신호를 통째로 넣을 때는 **반드시 padding**이 필요하다 (구현 시 자주 터지는 지점).

---

# B. 발전시킬 부분 (방향은 맞는데 구현 사양이 비어 있는 것)

## B-1. SWT thresholding: "adaptive threshold" 를 실제 알고리즘으로 확정

대화는 `λ ∝ σ√(2 ln N)` 과 "SURE/BayesShrink" 를 언급하고 끝났다. 실제로 필요한 사양은 4개다.

**(1) noise 표준편차 추정 — 최고주파 대역 하나에서만**

> ⚠️ **초판 정정**: 이 항목은 처음에 "level별 σ 추정"을 권했으나, **구현 후 실측으로 틀린 것이 확인되어 정정한다.**
> (정정 근거는 `docs/05_swt_tuning.md`, 측정값은 아래 표)

`fs=250 Hz`, 입력 SNR 10 dB(AWGN)에서 실제로 측정한 **대역별 SNR**:

| Band | 대역 (Hz) | band SNR | MAD(D_j) / 참 σ |
|---|---|---|---|
| D1 | 62.5–125 | **−13.2 dB** | 1.00 |
| D2 | 31–62.5 | +0.5 dB | 1.05 |
| D3 | 15.6–31 | +12.3 dB | 1.23 |
| D4 | 7.8–15.6 | +17.5 dB | **2.01** |
| D5 | 3.9–7.8 | +18.5 dB | **4.98** |
| A5 | 0–3.9 | +19.5 dB | 7.25 |

**D3 이상은 ECG가 계수를 지배**한다. 그 대역의 MAD는 잡음이 아니라 ECG를 재고,
σ를 2~5배 과대추정해서 **threshold가 QRS를 잘라낸다.**

SWT(`norm=False`)는 백색잡음의 계수 분산을 level에 무관하게 보존하므로,
**잡음이 지배적인 대역(D1 또는 D2) 하나에서 추정한 σ를 전 level에 적용**하는 것이 맞다.

```
σ = MAD(D2) / 0.6745        # D1 은 front-end LPF(100 Hz)에 일부 잘려 과소추정된다
```

실측 차이(합성 ECG, 잡음 7종 × SNR 3점 평균 `snr_imp_scaled`):

| σ 출처 | mean [dB] |
|---|---|
| `d2` | **+6.39** |
| `d1` | +6.38 |
| `level` (초판 권고) | **−1.75 ~ +2.44** |

**(2) level별 threshold 스케일** — σ는 전역, `k`만 level별

```
λ_j = k_j * σ * sqrt(2 * ln(N))       # σ 는 (1) 의 전역 추정치
```
`k`는 `scripts/tune_swt.py`로 탐색한다(탐색 seed와 검증 seed 분리). 수렴값:
```
k = (D1 1.6, D2 1.6, D3 0.4, D4 0.3, D5 0.2)
A5 는 thresholding 하지 않는다 (baseline은 별도 HPF로 처리)
```
D1/D2가 크고 D3~D5가 작게 수렴한 것은 위 대역 SNR 표와 **정확히 같은 방향**이다.
이 `k` 벡터는 계속 **ablation 대상**이며, MIT-BIH 확보 후 TRAIN split으로 재탐색한다.

**(3) QRS 구간 보호 (본 프로젝트의 핵심 변형)**
R-peak 위치 `t_R` 주변 `±60 ms` 에서는 threshold를 낮춘다.
```
λ_j(t) = λ_j * ( 1 - (1-ρ) * g(t) )     # g(t): t_R 중심 raised-cosine 창, ρ = 0.3 초기값
```
"고주파 = noise"라는 가정이 깨지는 유일한 구간이 QRS라서, 여기만 예외 처리하는 것이 가장 효율이 좋다.
**대화에서 개념으로만 있던 부분을 실행 가능한 식으로 내린 것.**

**(4) threshold 함수**
soft를 기본으로 하되, **garrote**를 함께 구현한다.
```
soft(d)    = sign(d) * max(|d| - λ, 0)
garrote(d) = d * max(0, 1 - λ²/d²)
```
garrote는 큰 계수에서 shrinkage bias가 작아 **A-1의 gain bias 문제를 완화**한다. ECG에는 이쪽이 유리할 가능성이 높다.

## B-2. Wavelet-subband CNN: 텐서 형상과 iSWT 미분가능성 — 대화에서 가장 애매하게 남은 지점

"subband를 CNN에 넣는다"까지만 있고, **역변환을 어떻게 학습 그래프에 넣을지**가 없다. 여기가 실제 구현이 막히는 곳이다.

**확정 사양:**

```
입력  y : (B, 1, 1024)
SWT(level=5, 'sym4')  ->  (B, 6, 1024)      # [A5, D5, D4, D3, D2, D1]
        |
   band-wise Conv1d (groups=6)               # 각 대역을 독립 처리 (대역별 특성 보존)
        |
   1x1 Conv 로 fusion  -> (B, C, 1024)
        |
   Residual U-Net (depth 4)                  # 시간축 multi-scale
        |
   출력 head -> (B, 6, 1024)                 # 예측한 "subband residual(=noise)"
        |
   ŝ = swt(y) - residual                     # subband 도메인 정제
        |
   ISWT (torch 구현, 미분 가능)  -> (B, 1, 1024)
```

**핵심 포인트: SWT/ISWT는 전부 선형 FIR 연산이므로 `torch.nn.functional.conv1d` 로 직접 구현하면 자동미분이 통한다.**
- SWT (à trous): level `j`에서 필터를 `2^(j-1)` 만큼 dilation한 conv, **downsampling 없음**
- ISWT: 각 level에서 짝수/홀수 위상 두 개의 재구성을 평균 (`swt`의 표준 역변환)
- 구현 후 **`pywt.swt/iswt` 와 수치 일치(≤1e-6)를 단위테스트로 강제**한다. 이게 통과해야 그 다음이 의미가 있다.

이렇게 하면 **time-domain loss와 subband-domain loss를 동시에** 걸 수 있다:
```
L = L_time(x̂, x) + λ_w * Σ_j w_j * L_band(ŝ_j, s_j)
```
`w_j`를 QRS 대역(D3/D4)에 크게 주면 morphology 보존을 loss 수준에서 강제할 수 있다.

## B-3. 실측 신호의 SNR을 **추정하는 방법**을 정한다 — "15 dB" 질문에 직접 답하는 부분

대화는 "15 dB인지 알 수 없다"에서 멈췄다. 하지만 **추정할 수 있다.** 세 가지 방법을 구현하고 셋 다 리포트한다.

**(a) Beat-averaged SNR (주력)** — 실측 ECG SNR 추정의 표준적 접근
```
1) R-peak 검출 → 각 beat 를 [-250 ms, +400 ms] 로 잘라 정렬
2) 형태가 유사한 beat 만 남김 (template correlation > 0.9)
3) 평균 beat  m(t) = mean_i( b_i(t) )        <- signal 추정
4) 잔차       r_i(t) = b_i(t) - m(t)          <- noise 추정
5) SNR = 10*log10( P(m) / mean_i P(r_i) )
   ※ N개 beat 평균은 noise를 1/N 로 줄이므로 bias 보정: P(m) := P(m) - P(r)/N
```
가정: "beat 간 형태는 반복되고, noise는 beat 간 무상관". ECG에서 상당히 합리적이다.

**(b) 등전위 구간(TP segment) 기반**
각 beat의 T파 종료 ~ 다음 P파 시작 구간은 생리학적으로 평탄해야 한다.
이 구간의 분산 = noise power 추정. `SNR = 10log10( P(전체 - 이 추정치) / 이 추정치 )`

**(c) noise-only 별도 측정 기반**
전극을 붙인 채 심박이 없는 조건은 만들 수 없으므로, **전극을 몸에서 뗀 채 회로만 동작**시켜 측정한 구간을 쓴다.
이건 front-end/ADC noise floor만 잡고 근전도·움직임은 못 잡으므로 **하한 추정**으로만 쓴다.

세 값이 크게 다르면 그 자체가 정보다. **(a)를 대표값으로 쓰고, "15 dB" 를 이 정의 위에서 재확인한다.**

## B-4. Arduino 데이터 수집 프로토콜을 실행 체크리스트로 내린다

"noise도 따로 모으자"는 아이디어는 좋은데, 실제로 뭘 몇 분 찍을지가 없다. 확정안:

| 세션 | 조건 | 길이 | 용도 |
|---|---|---|---|
| S1 | 정상 안정 측정 (앉은 자세, 움직임 없음) | 5 분 × 3회 | real-data 평가용 ECG |
| S2 | 전극 부착, **팔 근육에 힘주기** 반복 | 2 분 × 3회 | EMG artifact 수집 |
| S3 | 전극 부착, **케이블 흔들기 / 전극 누르기** | 2 분 × 3회 | electrode-motion artifact |
| S4 | 전극 부착, **천천히 큰 호흡 + 몸통 이동** | 2 분 × 2회 | baseline wander |
| S5 | 전극을 몸에서 떼고 회로만 동작 (더미 저항 부착) | 2 분 × 2회 | front-end/ADC noise floor |
| S6 | 전원 조건 변경 (USB 전원 vs 배터리) 각각 S1 반복 | 2 분 × 2회 | PLI 유무 비교 |

**S2~S5 = noise-only 데이터** → MIT-BIH clean ECG와 합성해 device-adapted 학습셋을 만든다 (대화의 아이디어를 실행 가능하게).

CSV 스키마도 고정한다:
```
# 파일명: arduino_{session}_{yyyymmdd}_{idx}.csv
# 헤더 주석 3줄 필수:
# fs_hz=500
# adc_bits=10, vref_v=5.0, gain=...
# session=S2, note=forearm muscle contraction
t_ms,adc_raw
0,512
2,514
...
```
`fs`를 파일에 안 적어두면 나중에 100% 문제가 된다.

## B-5. 실험 인터페이스(계약)를 먼저 고정한다

대화의 폴더 구조 제안은 좋지만, **함수 시그니처가 없다.** 이게 없으면 방법을 추가할 때마다 실험 스크립트를 고치게 된다.

**확정 계약 — 모든 denoiser는 이 하나의 형태를 따른다:**
```python
class Denoiser(Protocol):
    name: str
    def __call__(self, y: np.ndarray, fs: float, ctx: dict | None = None) -> np.ndarray:
        """y: (N,) noisy, 반환: (N,) denoised, 같은 길이 · 같은 스케일."""
```
`ctx`에는 `r_peaks`(있으면) 같은 선택적 정보만 넣는다. **`ctx`에 clean을 절대 넣지 않는다** (oracle 제외, oracle은 이름에 `oracle_` 접두사 강제).

평가도 하나의 계약:
```python
evaluate(x_clean, y_noisy, x_hat, fs, ...) -> dict[str, float]
```

## B-6. 재현성 사양

- 모든 실험은 `configs/*.yaml` 1개 = 실험 1개.
- 난수는 `(experiment_seed, record_id, window_idx)` 로부터 **결정론적으로 유도** (전역 seed 하나에 의존하지 않음).
- 결과는 `results/{exp_id}/metrics.parquet` + `manifest.json`(코드 git hash, 패키지 버전, config 사본).
- 표·그림은 `results/`만 읽어서 재생성 (원본 데이터 재접근 불필요).

---

# C. 추가로 제시할 부분 (대화에 없었고, 넣으면 프로젝트 급이 올라가는 것)

## C-1. Distortion floor: "이 필터가 깨끗한 신호에 해를 끼치는가?" — **가장 저비용·고효율의 추가 실험**

**clean ECG를 그대로 각 방법에 통과시킨다(noise 없음).**
```
SNR_distortion = 10*log10( P(x) / P(method(x) - x) )
```
이 값이 낮은 방법은 **"노이즈가 없어도 신호를 망가뜨리는 방법"** 이다.

- 이것은 morphology 보존을 재는 가장 깔끔한 단일 숫자다. delineator에 의존하지 않는다.
- 그리고 **denoiser 성능의 이론적 천장을 준다**: 출력 SNR은 절대 `SNR_distortion`을 넘을 수 없다.
- 예상되는 결과: `Bandpass` 는 낮고(대역을 잘라 버리므로), `SWT-garrote` 는 높고, `DL`은 학습에 따라 갈린다.
  **DL이 여기서 낮게 나오면 "hallucination 경향"의 직접적 증거**가 된다.

발표 그림으로도 강력하다: 가로축 `SNR improvement`, 세로축 `SNR_distortion` 의 산점도 = Pareto front.

## C-2. Oracle wavelet threshold — wavelet 계열의 성능 상한선

clean을 알고 있다고 가정하고, 각 wavelet 계수를 "살릴지 죽일지"를 최적으로 결정한다.
```
keep_j[n] = 1  if  |s_j[n]|² > σ_j²   else 0        # oracle diagonal estimator
```
이건 실제로 쓸 수 없는 방법이지만, **"SWT가 상한 대비 어디까지 왔는가"** 를 보여준다.

- SWT가 oracle에 가까우면 → **threshold 튜닝은 끝났다. 더 얻으려면 wavelet 표현 자체를 바꿔야 한다.** → hybrid/DL의 정당화
- SWT가 oracle과 크게 벌어지면 → threshold 설계를 더 밀어야 한다.

**"왜 딥러닝이 필요한가"에 대한 정량적 근거**를 제공하는 실험이다. 대화에는 이런 상한 개념이 없었다.

## C-3. Hallucination probe — 의료 신호 DL의 안전성을 실제로 검증

대화는 "함정 ①: AI가 ECG를 새로 만들어버림"을 지적만 하고 검증 방법을 주지 않았다.
**검증 가능하다.** 3개의 프로브를 설계한다.

| Probe | 조작 | 통과 조건 |
|---|---|---|
| **P1 (dropout)** | clean ECG에서 **1개 beat 구간(±200 ms)을 0으로 지운** 뒤 강한 noise를 덮음 | 모델이 그 자리에 **QRS를 만들어내면 실패**. 정상 모델은 flat 또는 저진폭 출력 |
| **P2 (asystole)** | 3초 구간을 등전위선 + noise로 대체 | 상동. 심정지 구간에 beat를 생성하면 임상적으로 치명적 |
| **P3 (ectopic)** | MIT-BIH annotation의 **PVC(V) beat** 구간만 골라 평가 | PVC의 넓은 QRS가 정상 QRS 모양으로 "교정"되면 실패 |

P3는 특히 좋다. **MIT-BIH가 beat annotation을 제공하므로 추가 라벨링 비용이 0**이다.
"모델이 정상 beat에 과적합되어 부정맥을 지운다"는 것은 이 분야의 실제 알려진 위험이고, 이걸 실측한 졸업과제는 드물다.

## C-4. Beat type 층화 평가 (normal vs abnormal)

C-3의 P3를 확장. MIT-BIH annotation symbol로 층화한다.
```
N (normal), V (PVC), A (APB), L/R (bundle branch block), ...
```
`SNR_imp`, `beat template CC` 를 **beat type별로** 낸다.
평균 성능이 같아도 "V beat에서만 무너지는 방법"이 드러난다. 추가 구현 비용이 매우 작다.

## C-5. Half-sample consistency — ground truth 없는 실측 데이터의 정량 지표

실측 Arduino 데이터에는 clean이 없어 RMSE를 못 쓴다. 대화는 여기서 "SQI"로 넘어갔는데, SQI는 정의가 모호하다.
더 명확한 지표를 제안한다.

```
1) R-peak 검출 → beat 를 홀수 index 집합 O, 짝수 index 집합 E 로 나눔
2) template_O = mean(beats in O),  template_E = mean(beats in E)
3) HSC = corr( template_O , template_E )
```
- noise는 O와 E에 **무상관**으로 들어가므로, noise가 줄수록 `HSC → 1`.
- signal은 O와 E에 **동일**하게 들어가므로 morphology 왜곡은 `HSC`를 올리지 못한다(양쪽에 같은 왜곡이 걸리므로 상쇄됨 → 그래서 단독으로는 부족).
- 따라서 `HSC`(noise 감소) + `beat 진폭·폭 변화`(왜곡) 를 **함께** 본다.

계산이 간단하고 가정이 명시적이라 발표에서 방어하기 쉽다.

## C-6. 방법 간 비교의 통계 처리

대화에는 "통계처리"가 한 줄만 있었다. 확정안:

- 집계 단위는 **window가 아니라 record**. (window는 서로 독립이 아니다 → p값이 과대해진다)
- 방법 A vs B 비교는 **paired Wilcoxon signed-rank** (같은 record에 두 방법을 다 적용했으므로 paired가 맞다).
- 다중비교는 **Holm-Bonferroni** 보정.
- 효과크기는 **rank-biserial correlation** 을 함께 리포트. p값만 쓰면 "통계적으로 유의하지만 0.2 dB 차이" 같은 무의미한 주장이 된다.
- 표에는 `mean ± std (median [IQR])` 를 같이 적는다.

## C-7. 계산 비용 실측 프로토콜

"inference time"만 적으면 비교가 안 된다. 확정안:

| 항목 | 측정 방법 |
|---|---|
| `params` | 학습 파라미터 수 (DSP는 0) |
| `MACs` | 4.096 s window 1개당 곱셈-누산 수 |
| `latency_cpu_ms` | 단일 스레드, 100회 median (warm-up 10회 제외) |
| `RTF` | `latency / 4.096 s` — **1보다 작아야 실시간 가능** |
| `peak_mem_mb` | 추론 시 최대 메모리 |

`RTF`를 쓰면 "Arduino+PC 구성에서 실시간이 되는가"에 직접 답할 수 있다.

## C-8. 합성 ECG 생성기(McSharry ODE) 내장 — **본 세션에서 발견된 제약에 대한 대응이자, 그 이상의 가치**

> **환경 제약 보고:** 현재 원격 세션에서 `physionet.org` 는 조직 egress 정책으로 **차단(403)** 되어 있다.
> 따라서 MIT-BIH/NSTDB 실제 다운로드는 **사용자 로컬 환경에서 실행**해야 한다. 다운로더 스크립트는 제공한다.

이 제약과 무관하게, **합성 ECG 생성기를 넣는 것 자체가 프로젝트에 이득**이다.

McSharry 등의 3-state ODE는 ECG를 극좌표 상의 극한주기(limit cycle) + Gaussian 합으로 생성한다.
```
θ̇ = ω
ż = -Σ_i a_i * Δθ_i * exp(-Δθ_i² / (2 b_i²)) - (z - z0)
Δθ_i = (θ - θ_i) mod 2π
```
**이 모델이 정확히 Sameni EKF의 상태방정식과 같은 파라미터화**를 쓴다. 따라서:

1. **PhysioNet 없이도 전체 파이프라인을 개발·검증**할 수 있다 (본 세션에서 즉시 필요).
2. **Sameni EKF 구현의 정답이 있는 검증 케이스**가 된다. 생성에 쓴 `(a_i, b_i, θ_i)` 를 EKF에 그대로 주면
   EKF는 이론적 최적 성능을 내야 한다. **안 나오면 구현 버그다.** (A-2의 진단을 자동화)
3. HR variability, beat 형태를 원하는 대로 통제할 수 있어 **층화 실험**이 쉽다.

---

# D. 검토 결과 요약 — 확정 사항 12개

| # | 항목 | 확정 |
|---|---|---|
| 1 | 샘플링 주파수 | **250 Hz 통일** (MIT-BIH 360 → 250 리샘플, Arduino 실측 → 250) |
| 2 | window / hop | **1024 / 512**, Hann√ overlap-add |
| 3 | SNR 리포트 | `strict` + `scaled` + `gain_bias` **3종 동시** |
| 4 | PRD | **`PRDN`만** 사용 |
| 5 | SWT | level 5, **level-dependent threshold + QRS 보호**, soft/garrote 비교 |
| 6 | Sameni | **EKF·EKS 둘 다**, 6항목 자가진단 통과 필수, 합성 ECG로 정답 검증 |
| 7 | DL 주력 | **Residual 1D U-Net** → **Wavelet-subband Residual U-Net** |
| 8 | 정규화 | noisy 기준 robust scale, **반드시 역정규화** |
| 9 | split | record 단위 + **noise 시간축 disjoint** |
| 10 | 주 지표 | `SNR_imp`, `RMSE`, `CC`, `R-peak MAE`, `QRS duration err` |
| 11 | 추가 실험 | `distortion floor`, `oracle bound`, `hallucination probe`, `beat-type 층화` |
| 12 | 통계 | record 단위 · paired Wilcoxon · Holm · effect size |

다음 문서: `01_design.md` (확정 설계안), `02_procedure.md` (구현 단계별 절차).
