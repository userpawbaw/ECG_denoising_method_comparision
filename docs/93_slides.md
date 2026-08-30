# 93. 발표용 그림 색인

> 자동 생성: `python scripts/make_slides.py` (수정하지 말 것 — 스크립트를 고칠 것)

보고서 그림(`results/{d0,d1}/report/`)은 **모든 방법을 빠짐없이 싣는 기록용**
이고, 이것은 **화면에 띄워 설명하는 용도**다. 방법을 6 개로 줄이고,
차이가 보이는 -5 dB 를 주력으로 쓰고, 잔차를 나란히 그린다.

한글 폰트가 없는 환경에서는 라벨이 영문으로 자동 대체된다
(`apt-get install -y fonts-nanum` 후 matplotlib 폰트 캐시 삭제).

| 그림 | 무엇을 보여주는가 |
|---|---|
| [`S1_input.png`](../results/slides/S1_input.png) | **도입.** 무엇을 되살려야 하는가. D1 의 큰 아래쪽 치우침이 기저선 변동이다. |
| [`S2_methods_d0.png`](../results/slides/S2_methods_d0.png) | D0 방법별 출력 + 잔차. **잔차 열이 핵심** — 출력만 보면 다 비슷하다. |
| [`S2_methods_d1.png`](../results/slides/S2_methods_d1.png) | D1 같은 그림. M08 +21.8 dB vs 나머지 +17.3~17.6 dB, 잔차가 눈에 띄게 평평하다. |
| [`S3_qrs_d0.png`](../results/slides/S3_qrs_d0.png) | D0 beat 평균 템플릿 — 잡음을 평균으로 지우고 **체계적 왜곡만** 남긴다. |
| [`S3_qrs_d1.png`](../results/slides/S3_qrs_d1.png) | D1 같은 그림. **M01(bandpass)이 QRS 를 가장 크게 왜곡**(±0.09 mV)하고 M04(SWT)는 거의 평평하다 — 40 Hz 절단의 대가다. |
| [`S4_noise_d0.png`](../results/slides/S4_noise_d0.png) | D0 잡음 종류별. 어느 계열이 어디에 강한가. |
| [`S4_noise_d1.png`](../results/slides/S4_noise_d1.png) | **가장 설득력 있는 그림.** 기저선 변동은 front-end 가 다 해결하고(M_FE ≈ M04 ≈ +20.0 dB), 임펄스는 딥러닝만 해결한다(+6.4 → +20.1 dB). 임펄스 행에서 스파이크가 M_FE·M04 에 그대로 남아 있는 것이 눈으로 보인다. |
| [`S5_crossover.png`](../results/slides/S5_crossover.png) | **핵심 슬라이드.** 딥러닝이 front-end 위에 더하는 값이 입력 SNR 에 따라 줄고, **D1 에서는 13 dB 부근에서 0 을 지난다.** 저 SNR(-5 dB)에서는 두 축이 거의 같다(+7.9 vs +7.1) — 합성이 딥러닝을 과대평가한 것이 아니라 **도움이 되는 SNR 범위**가 좁아진 것이다. |
| [`S6_safety.png`](../results/slides/S6_safety.png) | 없는 파형을 지어내는가. **딥러닝이 두 축·두 프로브 모두에서 가장 낮다** — residual 구조(출력 = 입력 - 예측잡음)의 직접적 결과이고, "딥러닝이 파형을 지어낸다" 는 통념과 반대다. |
| [`S7_loss_gap.png`](../results/slides/S7_loss_gap.png) | **S5 의 후속.** S5 가 보인 고 SNR 열세를 **손실만 바꿔서** 되돌린다. 20 dB 에서 D0 는 부호가 뒤집히고(-1.6 → +3.0) D1 은 격차의 74 %가 사라진다(-4.8 → -1.3). 구조를 세 번 바꿔도(M07·M08·M10) 안 되던 것이 손실 한 항으로 움직였다는 것이 이 슬라이드의 요지다. |
| [`S9_structure_vs_loss.png`](../results/slides/S9_structure_vs_loss.png) | **결론 한 장.** 구조 4연속 기각 vs 손실 성공. 한 번의 null 로는 '그 구조가 나빴다' 와 구분되지 않으므로 넷을 함께 싣는다. |
| [`S8_clean.png`](../results/slides/S8_clean.png) | **S7 의 '왜'.** L6 는 "입력이 이미 깨끗하면 건드리지 마라" 를 손실에 넣은 것인데, **바로 그 지표(EXP-C)가 올라갔다.** D1 에서 M06 22.2 → 37.4 dB, M08 23.5 → 38.4 dB 로 SWT(40.1)와의 17 dB 격차가 1.8 dB 가 된다. 이득이 우연이 아니라 **의도한 기제를 통해** 왔다는 근거다. |
| [`S10_loss_by_noise.png`](../results/slides/S10_loss_by_noise.png) | **S7·S8 의 범위.** L6 의 이득이 어디까지 가는가 — 잡음 7 종 × 입력 SNR 3 단계(EXP-G). **결론의 축은 잡음 종류가 아니라 입력 SNR 이다**: 20 dB 에서는 손해가 한 칸도 없고, 손해 다섯 칸은 전부 0 dB 다. 구조가 뚜렷한 잡음(전원선 +6.3, 기저선 변동 +4.6 평균)에서 크게 벌고 광대역 백색잡음(+0.3)에서 가장 적게 번다 — 잡음이 신호와 분리 가능해야 '건드리지 않는다' 가 선택지가 되기 때문이다. |

## 색 배정 (그림마다 바뀌지 않는다)

| 방법 | 색 |
|---|---|
| `M01` | `#2a78d6` |
| `M04` | `#eb6834` |
| `M08` | `#1baf7a` |
| `M05` | `#eda100` |
| `M_FE` | `#e87ba4` |

참조는 회색 굵은 선으로 뒤에 깔고, 입력(잡음)은 진한 중성색이다.
색은 dataviz 팔레트에서 가져와 **색각이상 분리도를 검증**했다.
`M_FE`(magenta)와 `M04`(orange)는 정상시야 분리도가 기준 미달이라
**겹쳐 그리지 않는다** — 패싯으로만 쓴다.

