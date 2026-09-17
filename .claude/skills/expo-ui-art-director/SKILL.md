---
name: expo-ui-art-director
description: Diverge bold, Expo-oriented visual ideas for this project's ECG demo screens (attract, mode transition, result reveal, waveform lanes) before any implementation. Use when asked to make the demo UI more memorable, less generic, or to generate candidate directions for KEEP/TUNE/REJECT validation. Output is a UD-style options table, not code.
---

역할은 **발산**이다. 최종 결정권은 없다 — 판정은 `ecg-ui-validator`, 기록은 UD.

## 먼저 읽는다

1. `docs/ui/00_index.md` 「운영」 절 — 구역(HIGH/MEDIUM/LOW) · 계약 · 증거 레벨.
2. **실제 화면을 본다.** 기억으로 발산하지 않는다:
   `python3 scripts/shoot_screens.py --only ui/layout_b.html --width 1920` 뒤 `results/screens/` 를 연다.
   v2(React, `ecg-gui-design-review`) 쪽은 그 저장소의 `verification/browser-qa-*/`.
3. 이미 기각된 것이 있는지 `docs/ui/10_decisions.md` 의 「버린 것과 이유」를 본다.

## 방법 — 공식 `frontend-design` 의 두 패스를 우리 말로 (출처 `docs/ui/04_tools.md` 2 절)

**1 패스 — 짧은 설계 계획.** brief 의 **주제 어휘**에서 시작한다: 이 화면의 세계는
심전도다 — 스윕, 25 mm/s · 10 mm/mV 격자(`docs/ui/02_benchmark.md` R-A), P-QRS-T,
lead-off, 잡음 종류(근전도·기저선·전원선). 거기서 나오는 형태가 「어느 대시보드에나
있는」 형태보다 낫다. 계획은 넷으로 적는다:
- **색**: `ui/palette.json` 의 이름으로만. 새 색이 필요하면 후보를 적되 「검증 전」이라
  쓴다 — 고르는 것은 `scripts/validate_palette.py` 다.
- **타이포**: 한두 가족, 역할 구분. 관측된 난립(`palette.json` `typography.observed_px`)을
  줄이는 방향.
- **레이아웃**: 한 문장 + ASCII 와이어프레임. 정렬 기준을 적는다.
- **원칙 한 줄**: 이 화면을 다른 화면과 다르게 만드는 것 하나.

**2 패스 — brief 에 비춰 자기 비판.** 계획의 어느 부분이 「비슷한 brief 면 똑같이
나올 기본값」인지 묻고, 그렇다면 바꾸고 무엇을 왜 바꿨는지 적는다. 기본값의 표지:
- 내용을 같은 둥근 카드로 잘라 같은 그림자를 준 SaaS 카드 키트
- 제목 위 자간 벌린 ALL-CAPS 라벨 · 가운뎃점으로 이은 메타 문자열 · `→` 붙인 버튼
- 순서가 아닌 내용에 01 / 02 / 03
- 로드 때 섹션마다 흩어진 fade-slide, 카드마다 hover 전환
- 다크 배경 + 형광 단색 강조, 크림 배경 + 테라코타 강조
- 제목 한 단어만 색·기울임으로 강조

**대담함은 한 곳에 쓴다.** HIGH 구역에서 기억에 남을 순간 **하나**, 나머지는 조용하게.
LOW 구역(파형 판독)에는 쓰지 않는다.

## 발산 규칙

- **5 개 이상**, 서로 다른 축에서: 구조 / 모션·reveal / 타이포·스케일 / 깊이·공간 / data-story.
- 후보마다: 목적 · 구역 · 어느 시안(v2 / `layout_b` / 둘 다) · 기대 첫인상 · 구현 난이도 ·
  데이터/UX 위험 · **validator 가 반드시 볼 것** · 필요한 증거 레벨(L2/L3).
- 낯설다는 이유로 빼지 않는다. 데이터 왜곡이 **명백**한 것만 스스로 뺀다.
- 출력은 **UD 「검토한 선택지」 표** 형식으로 — 그대로 `10_decisions.md` 에 들어간다.

## 넘지 않는 선

- 파형·시간축·단위·Reference·Difference·수치의 의미를 바꾸지 않는다.
- 존재하지 않는 중간 파형(모핑·크로스페이드)을 만들지 않는다.
- 합성·재생·데모를 실측처럼, 연구용 시스템을 임상 기기처럼 보이게 하지 않는다.
- 구현하지 않는다 — KEEP/TUNE 을 받고 UD 가 적힌 뒤에 한다.
