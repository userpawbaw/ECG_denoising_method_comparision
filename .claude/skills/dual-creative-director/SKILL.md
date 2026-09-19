---
name: dual-creative-director
description: Run two independent creative divergences for the ECG demo screens — a Native Director (codebase + tokens + frontend-design) first, then a Reference Director (external scenes as Reference Cards) — with model split and no cross-contamination before cross review. Use for "새 디자인 라운드", "이중 디렉터", "네이티브 디렉터만", "레퍼런스 디렉터만", "시안 가지치기", or any new HIGH-zone visual direction.
---

역할은 **발산의 독립을 지키는 것**이다. 후보를 만드는 건 `expo-ui-art-director` 와
`frontend-design` 이 하고, 이 스킬은 **누가 무엇을 언제 보는지**를 정한다.
운영 원문: `docs/ui/00_index.md` 「이중 디렉터」 · `docs/ui/02_benchmark.md` §8 · UD-10.

## 먼저 정한다 — 구역

`docs/ui/00_index.md` 구역표로 대상 화면의 구역을 본다.
HIGH 새 방향 → 이중. MEDIUM → B 만 돌리고 A 는 한 줄로 제안. LOW → 발산하지 않는다.

## 순서 — B 가 먼저다

1. **B 네이티브 디렉터.** 읽는 것: brief · 기준선 캡처(`results/screens/`) ·
   `ui/palette.json` · `demo/ui/tokens.css` · `scripts/_screens.py` · 지난 UD 의 「버린 것」.
   **읽지 않는 것: 외부 레퍼런스, A 의 어떤 것도.** `frontend-design` 2 패스로 토큰 계획을
   만들고 자기 계획을 클리셰 목록에 대 본 뒤, 후보 4~6 을 UD 「검토한 선택지」표 형식으로.
2. **모델을 바꾼다** — 사용자에게 `/model` 전환을 청한다. 어느 모델이 B 였는지 적어 둔다.
3. **A 레퍼런스 디렉터.** `02_benchmark` §8 카드 3~8 (어디를 볼지 · `chrome --screenshot`
   L1 캡처 · Imitation Distance 3~4). 후보 5~8, **후보마다 REF 번호와 빌린 원리**.
   **A 는 자기 표를 다 쓴 뒤에야 B 의 절을 연다.**
4. **사용자 정렬 — 판정보다 먼저다.** 두 디렉터의 후보를 **전부** 사용자에게 보인다:
   후보마다 한 줄 설명 · 어느 REF 에서 왔나 · **빌린 경험 원리** · 위험. 레퍼런스는
   **URL 과 「어디를 볼지」를 함께** 준다 — 그것이 레퍼런스 마이닝의 존재 이유다
   (사용자가 직접 보고 판단하라고 만든 단계다). 사용자가 고르거나 버린 뒤에 다음으로 간다.
   **검증기를 먼저 돌려 후보를 줄이지 않는다** — 그러면 사용자는 남은 것만 보게 되고,
   고르는 일이 판정으로 바뀐다 (UO-5).
5. **교차 검토.** 두 표를 나란히. 겹침은 합치고, 상보적 강점이 **실제로** 보일 때만
   Hybrid 하나. 어느 쪽 후보가 어느 축(구조·모션·타이포·깊이·데이터)에 몰렸는지 센다 —
   분포가 거의 같으면 격리가 실패한 것이고 그것을 UD 에 적는다.
6. **검증기 넷**(`ecg-ui-validator` · `motion-review` · `design-critique` ·
   `accessibility-review`) → KEEP/TUNE/REJECT → **UD 를 먼저 적는다.** 구현은 그 뒤.

## 시안 가지치기 (승인된 후보만)

- 2~4 개만. `demo/ui/layout_b.html` 을 복제해 `demo/ui/branches/UD-nn-{a,b,c}.html`.
  **같은 baseline**, 바꾸는 건 방향 하나씩. 픽셀·색을 미리 못 박지 않는다.
- `python3 scripts/shoot_screens.py` 로 L2 캡처 → 갤러리 → 필요하면 Artifact 발행.
- 브랜치는 탐색물이다. `SCREENS` 에 안 올린다. 파일명의 UD 가 본문에서 그 파일을 불러야
  한다(`check_records.py` 가 본다). 생성기(=이 세션)는 자기 결과를 **승인하지 않는다** —
  검증기 넷이 한다.
- production 으로 덮어쓰지 않는다. 채택안은 UD 의 변경 계약을 거쳐 `layout_b` 로 옮긴다.

## 하지 않는 것

- 파형 기하 · 시간축 · 단위 · Reference/Difference 의미 · 지표 의미를 표현을 위해 바꾸지 않는다.
- 낯설다는 이유로 발산 중에 버리지 않는다. 데이터 무결성을 **직접** 깨는 것만 뺀다.
- 원본 레퍼런스의 팔레트·서체·오브젝트를 그대로 가져오지 않는다(Distance 0~2 는 이유 필수).
- Superdesign 을 실행한 척하지 않는다 — UD-10 으로 보류됐다.

이 스킬은 자문이다. `docs/ui/00_index.md` 와 최신 사용자 지시가 우선한다.
