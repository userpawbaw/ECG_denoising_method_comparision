---
name: motion-review
description: Review or propose motion for the ECG demo screens — sweep, fade, glow, reveal, transitions, hover — while protecting waveform legibility, data integrity, reduced-motion behaviour and Canvas performance. Use before changing any animation and when validating motion candidates from the art director.
---

## 현행 계약 — 먼저 안다

**스윕(`demo/ui/layout_b.html`)은 이미 「지우기 경계 페이드 + 새로 그려진 선단 톤업」이다.**
사용자 제안이고, **구현·실측됐다** (UF-4 · `tests/test_demo_screens.py` 의
`test_sweep_fades_into_the_cursor_instead_of_cutting` · `test_sweep_glows_behind_the_cursor`).
외부 문서(`ecg-gui-design-review` `03_MOTION` · `docs/21` §7.9)의 「선단 glow 는 미승인」은
**이 저장소보다 뒤**의 상태다 — 그것을 근거로 되돌리지 않는다.

이 계약을 바꾸려면 **UD 먼저**, 그리고 L3(`pytest tests/ -m screens`).

## 측정 함정 — UF-4

`destination-out` 은 **RGB 를 남기고 알파만 줄인다.** 캔버스 픽셀을 RGB 로만 읽으면
지워진 픽셀이 흰색(255)으로 읽힌다. **알파로 가중해서** 잰다(`_INK_JS` 참조). 이 함정
때문에 있지도 않은 「흰 줄」을 쫓은 적이 있다.

## 후보마다 답한다

1. **사용자 이득** — attention / 상태 변화 / 연속성 중 무엇인가. 장식뿐이면 그렇게 쓴다.
2. **구역** — HIGH(attract · 전환) / MEDIUM(요약 · reveal) / LOW(파형 판독).
3. **겹침·교차** — 옛 주기와 새 표본이 섞여 보이나. 작은 P/T/Q/S 굴곡이 가려지나.
4. **상태 일관성** — play / pause / wrap / 확대 / 선택 / 다크. pause 에서 장식이 정리되나.
5. **reduced-motion** — 장식은 줄이고 **사용자가 켠 재생 자체는 멈추지 않는다**
   (`test_animated_screens_respect_reduced_motion` 이 그 동작을 잰다).
6. **비용** — UI 는 transform/opacity 중심. `transition: all` 금지. 연속 blur/filter 금지.
   Canvas 는 draw cost 를 **재서** 말한다 — 「부드럽다」는 측정 뒤에만.
7. **되돌리기** — 끄는 스위치가 있나.

## Scorecard (외부 `03_MOTION_AND_POLISH` 의 10 항, 1~5)

Legibility · Attention Guidance · Signal Integrity · Subtlety · Cognitive Load · Context Fit ·
Reduced Motion · Performance Risk · State Consistency · Reversibility.
**비교 도구일 뿐이다.** 데이터 무결성 항목이 하나라도 critical FAIL 이면 총점과 무관하게
재설계다.

## 절대선

- 존재하지 않는 중간 파형을 만들지 않는다 — 방법 사이 모핑 · 크로스페이드 금지.
- 잔광·halo 가 데이터처럼 읽히게 하지 않는다.
- 시선을 계속 빼앗는 배경 모션을 두지 않는다.
