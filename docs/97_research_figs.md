# 97. 연구 과정 시각화 색인

> 자동 생성: `python scripts/make_research_figs.py` (수정하지 말 것 — 스크립트를 고칠 것)

시각화 제안서 `docs/96_visual_proposal.md` 의 후보 P1~P11 을 그린 것이다. 결과 그림 색인은
`docs/93_slides.md` 이다. 수치는 모두 산출물 파일에서 읽으며, 수정 전 상태처럼 산출물이 남지 않은
값만 연구 기록의 기록값을 쓰고 그림 안에 그렇게 적었다.

| 그림 | 제안서 | 설명 |
|---|---|---|
| [`S22_resolution.png`](../results/slides/research/S22_resolution.png) | P1 | 측정 도구의 눈금. 위 네 띠는 같은 설정의 학습 반복 간 변동이 평가 잡음 고정으로 줄어드는 과정(F-40 · F-41)이고, 아래 점은 구조 · 용량 · 손실 변경의 효과이다. 고정 평가 집합의 띠를 넘은 개선은 손실 함수 변경 두 건뿐이다. |
| [`S23_band_specialist.png`](../results/slides/research/S23_band_specialist.png) | P2 | 입력 SNR 대역 전용 모델. 전 대역 모델(검정)이 저 SNR 전용 모델(주황)과 겹쳐 내려가 고 SNR 구간에서 음수가 된다. 오른쪽 화살표는 회수 가능한 상한(고 SNR 전용 모델)과 SNR 조건화로 실제 회수한 양이다(F-43 · F-45). |
| [`S24_axis_dumbbell.png`](../results/slides/research/S24_axis_dumbbell.png) | P3 | 합성 ECG(회색)에서 MIT-BIH(파랑)로 옮겼을 때 대역통과 필터 대비 차이의 변화. SNR 개선량은 크게 줄고 유의성을 잃지만 상관계수 · PRDN · PSD 거리의 우위는 유지된다. 데이터는 자동 생성 문서 `docs/92_axis_gap.md`. |
| [`S25_flip_before_after.png`](../results/slides/research/S25_flip_before_after.png) | P4 | 측정 오류 네 건의 수정 전 · 후. 그림 S14 의 보조 그림이다. 수정 전 값 중 산출물이 남지 않은 것은 연구 기록의 기록값이다. |
| [`S26_l6_conditions.png`](../results/slides/research/S26_l6_conditions.png) | P5 | 무잡음 입력 보존 손실이 작동하는 세 조건(F-56). 잔차 연결이 있고 MAE 를 포함한 경우에만 무잡음 입력의 출력 SNR 이 크게 오른다. |
| [`S27_cond_l6_interaction.png`](../results/slides/research/S27_cond_l6_interaction.png) | P6 | SNR 조건화와 무잡음 입력 보존 손실의 2x2 상호작용(F-47). 보존 손실 선이 거의 수평이므로 조건화를 더해도 추가 개선이 없다. |
| [`S28_cond_error_cost.png`](../results/slides/research/S28_cond_error_cost.png) | P7 | 조건 SNR 오차의 비대칭 비용(F-46). 과소 추정 쪽은 급하게 떨어지고 과대 추정 쪽은 완만하다. SNR 추정기가 과대 추정 쪽으로 편향되어 있으므로(F-13) 실제 비용은 작은 쪽에 해당한다. |
| [`S29_metric_floor.png`](../results/slides/research/S29_metric_floor.png) | P8 | 지표 분해능 대비 효과 크기. QRS 폭 오차는 MIT-BIH 에서 한 기법을 빼고 모든 기법의 차이가 판별 하한 아래에 있어 판정에 쓸 수 없다(F-16). 합성 ECG 에서는 모두 판별 하한 위에 있다. |
| [`S30_phantom_threshold.png`](../results/slides/research/S30_phantom_threshold.png) | P9 | 그림 S15 의 강건성 보조. 가성 QRS군 판정 임계값을 바꾸어도 칼만 필터가 가장 많다(F-58, D-42). |
| [`S31_findings_map.png`](../results/slides/research/S31_findings_map.png) | P10 | 발견 기록 전체에서 보고서 8 장의 방법론적 요약이 유형으로 묶은 기록의 위치. 측정 전제의 오류 아홉 건이 연구 초기와 실시간 처리 연구에 모여 있다. 분류는 보고서 본문에서 읽으므로 보고서가 바뀌면 그림도 바뀐다. |
| [`S32_step_optimum.png`](../results/slides/research/S32_step_optimum.png) | P11 | 학습 곡선을 step 축으로 다시 그린 것. 데이터량과 학습량을 4 배로 늘려도 최적점의 step 수가 비슷하다(5.10.8 절의 부수 관측). |
