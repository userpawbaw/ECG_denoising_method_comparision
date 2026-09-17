---
name: ecg-ui-validator
description: Converge creative/data/motion candidates for the ECG demo into KEEP / TUNE / REJECT using this project's contracts (waveform, time, units, reference, data scope), design quality, motion rules, accessibility and the evidence-level matrix. Use after expo-ui-art-director diverges, and before any UI implementation is approved.
---

역할은 **수렴**이다. 후보를 걸러 UD 에 들어갈 판정과 「되돌릴 조건」을 만든다.

## 판정

- **KEEP** — 그대로 시안화할 가치가 있다.
- **TUNE** — 아이디어는 남기되 강도·구조·표현을 조정한다. 무엇을 어떻게, 적는다.
- **REJECT** — 데이터·UX·접근성·성능·프로젝트 정체성과 충돌한다. **「취향」은 이유가
  아니다.** 어느 계약의 어느 줄에 걸리는지 적는다.

기각도 산출물이다 — UD 「버린 것과 이유」에 그대로 남긴다.

## 계약 — 이 순서로 대본다

1. **데이터 무결성** (`docs/21_decisions.md` D-18 · `docs/ui/01_system.md` · 보고서 5 장)
   - 입력/출력/Reference 는 **같은 표본·같은 시간 매핑**으로 비교한다. 방법별 autoscale 금지.
   - Reference 는 FE(x) 다 — raw MIT-BIH 를 「무잡음 참값」이라 부르지 않는다 (F-12).
   - 축 D0(합성)·D1(MIT-BIH) 를 섞어 인용하지 않는다 (F-28). 대표 장면 하나를 전체
     성능처럼 보이게 하지 않는다.
   - 단위·스케일을 보존한다. 높이·면적·3D 로 서로 다른 단위를 비교하지 않는다.
   - 합성/재생/데모를 실측처럼, 연구용 시스템을 임상 기기처럼 표현하지 않는다.
2. **읽힘** (`docs/ui/02_benchmark.md`) — 격자 25 mm/s · 10 mm/mV 규약, 신호 속도가
   화면 폭에 딸려 가지 않는가(UF-3), 위계·간격·타이포 난립(관측 8~10 단계).
3. **모션** — `motion-review` 로 넘긴다. 현행 스윕 계약은 UF-4.
4. **접근성** — 색만으로 신원을 알리지 않는다 · 히트 타겟 ≥ 44 px(관람객 기준, UD-1) ·
   키보드/포커스 · reduced-motion · 대비. 새 색은 **`scripts/validate_palette.py` 를 먼저
   통과**해야 한다. 눈으로 고르지 않는다.
5. **구역 적합** — HIGH 에 절제된 것, LOW 에 대담한 것 둘 다 잘못이다.

## 증거 레벨 — 판정에 붙인다

| 변경 | 최소 증거 |
|---|---|
| 간격·타이포 | L1 소스 + L2 `shoot_screens.py` 정적 비교 |
| 공유 컨트롤·상태 | L1 + L3 `pytest tests/ -m screens` |
| 전환·모션 | L3 런타임 + reduced-motion |
| 파형 스타일 | L3 + 작은 굴곡 가독성(픽셀 측정) |
| 축·단위·Difference | 수치 fixture + L3 |
| 데이터 시각화 | 원본 수치 대조(`check_records.py`) + L2/L3 |
| attract·내러티브 | 3 초/15 초 메시지 시험 + L4 실기(1920×1080) |

낮은 레벨의 PASS 를 높은 레벨로 확대 해석하지 않는다. 「자동 검사 PASS」는 「실기 PASS」가 아니다.

## 도구가 켜져 있으면

`design` 플러그인의 `design-critique` · `accessibility-review` 에 `results/screens/` 의
스크린샷을 넣어 **두 번째 눈**으로 쓴다. 그 결과도 근거 태그 `[플러그인]` 으로 인용한다 —
판정을 대신하지 않는다.

## 출력

후보별 한 줄: `판정 · 걸린 계약 · 조정(TUNE 이면) · 필요한 증거 레벨 · 되돌릴 조건`.
그리고 **왜 이번엔 이 순서로 대봤는지** 한 줄. 이것이 UD 「고른 것과 근거」의 재료다.
