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
| [`S10_loss_by_noise.png`](../results/slides/S10_loss_by_noise.png) | **S7·S8 의 범위.** L6 의 이득이 어디까지 가는가 — 잡음 7 종 × 입력 SNR 7 단계(EXP-G), 두 축. **결론의 축은 잡음 종류가 아니라 입력 SNR 이다** — 왼쪽(저 SNR)이 붉고 오른쪽으로 갈수록 파래진다. 칸 수와 판정은 **그림의 부제가 격자에서 직접 계산**하고, 해석은 보고서 5.8.10 에 있다 (같은 산문을 두 곳에 두면 한쪽이 낡는다 — O-27). |
| [`S11_nofe_grid.png`](../results/slides/S11_nofe_grid.png) | **front-end 를 아예 빼면.** 2x2 격자(모델 x 손실)로 재면 격차를 만드는 것은 **모델이 아니라 손실**이다(L1 5.05 dB vs L6 1.75 dB). 그리고 FE 를 빼면 여덟 판이 0.97 dB 안에 몰려 **무엇을 골라도 비슷해진다** — FE 판의 3.61 dB 와 견주면 띠 폭 자체가 결론이다. |
| [`S12_fe_tradeoff.png`](../results/slides/S12_fe_tradeoff.png) | **실시간 front-end 는 무엇을 지킬 것인가의 문제다.** 평평함(가로)과 ST 충실도(세로)가 **같은 방향이 아니고**, 좌하단의 '둘 다 좋음' 자리가 비어 있다. 점 크기가 추가 지연이다 — 인과 방식만 어느 축으로도 3 위다. |
| [`S13_frontend_flip.png`](../results/slides/S13_frontend_flip.png) | **F-10 의 결과 한 장.** 바꾼 것은 '딥러닝에도 같은 front-end 를 준다' 하나인데 결론이 둘 뒤집혔다 — 15 dB 승자가 M04(7.79)에서 M08(8.71)로, M06 대 M01 이 n.s.(p=0.101)에서 유의(p=9e-6)로. 왜곡 하한 두 행은 **참조 정의가 달라 방향만** 읽는다. |
| [`S14_measurement_traps.png`](../results/slides/S14_measurement_traps.png) | **6 장 머리말 한 장.** 측정 틀이 틀렸던 아홉 건과 각각을 잡아낸 것. 아홉 중 다섯이 '참조를 무엇으로 둘 것인가' 이고, 아홉 다 p 값이 유의했다 — 통계는 전제가 틀렸다는 것을 알려주지 않는다. |
| [`S15_kalman_phantom.png`](../results/slides/S15_kalman_phantom.png) | **판넬 결과 R3.** 칼만은 참값에 없는 자리에 **깨끗한 QRS** 를 그린다. 왼쪽은 가장 뚜렷한 예시(그렇게 골랐다고 그림에 적었다), 오른쪽은 **비율** — 오검출 개수는 front-end 와 같아도 심박 모양인 것이 많다. 근거 D-42 · F-58. |
| [`S16_wavelet_placement.png`](../results/slides/S16_wavelet_placement.png) | **판넬 결과 R4.** 같은 backbone 에 wavelet 의 자리만 바꿨다. 직렬(M07)은 seed 짝으로 유의하게 손해, 특성공간(M08)은 이 예산으로 못 가린다. 깨끗한 신호 보존(EXP-C)도 같은 순서다. |
| [`S17_dl_journey.png`](../results/slides/S17_dl_journey.png) | **판넬 결과 R5.** 딥러닝 개선 시도 16 개를 같은 자(seed 짝 · 고정 평가)로. 좋은 쪽으로 유의하게 움직인 것은 전부 「무엇을 맞추게 하느냐」(손실 · 동작점) 묶음이고, 구조 · 깊이 · 용량 · 학습 세부는 못 가리거나 손해다. |
| [`S10p_loss_by_snr.png`](../results/slides/S10p_loss_by_snr.png) | S10 의 판넬판. 잡음 7 종을 합쳐 입력 SNR 별 유의한 개선 · 저하 수만 보인다. 잡음을 임의로 묶지 않고 7 종을 그대로 센다. |
| [`S17p_dl_journey_panel.png`](../results/slides/S17p_dl_journey_panel.png) | S17 의 판넬판. 16 행 이름표 대신 범주 4 개를 두고, 유의한 항목에만 이름을 단다. |
| [`S18_pipeline.png`](../results/slides/S18_pipeline.png) | **판넬 작품 내용 A2.** 취득 → 공통 front-end → 기법 네 계열 + 상한 → 평가 4 층 · 안전성 프로브 · 시연. 비교의 판 전체를 한 장에. |
| [`S19_swt.png`](../results/slides/S19_swt.png) | **판넬 이론 (1).** SWT 세부계수 다섯 단과 단별 임계 띠, QRS 자리에서 좁아지는 보호 창. 실험과 같은 M04 인스턴스(D0 튜닝값 — garrote · QRS 보호)로 그렸고, D1 튜닝은 hard · 보호 끔을 골랐다는 것을 부제가 설정에서 읽어 적는다. |
| [`S20_kalman_model.png`](../results/slides/S20_kalman_model.png) | **판넬 이론 (2).** Sameni 모형 — 위상 원 위의 가우시안 커널 합. 위상이 검출 R-peak 로 매겨진다는 것이 S15 의 기제다. |
| [`S21_resunet.png`](../results/slides/S21_resunet.png) | **판넬 이론 (3).** Residual U-Net 과 출력단 잔차(x̂ = y - n̂), 손실 L6. 채널과 파라미터 수는 체크포인트에서 읽는다. |

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

