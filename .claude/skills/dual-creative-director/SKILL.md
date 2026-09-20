---
name: dual-creative-director
description: Run two independent creative divergences for the ECG demo screens — a Native Director (codebase + tokens, driven through Superdesign) first, then a Reference Director (external scenes as Reference Cards) — kept apart by environment split, then ranked for visual surprise and shown to the user BEFORE any validator runs. Use for "새 디자인 라운드", "이중 디렉터", "네이티브 디렉터만", "레퍼런스 디렉터만", "줄세우기", "시안 가지치기", or any new HIGH-zone visual direction.
---

역할은 **발산의 독립을 지키는 것**이다. 후보를 만드는 건 `expo-ui-art-director` 와
`frontend-design` 이 하고, 이 스킬은 **누가 무엇을 언제 보는지**를 정한다.
운영 원문: `docs/ui/00_index.md` 「이중 디렉터」 · `docs/ui/02_benchmark.md` §8·§11 ·
UD-10 → **UD-12 가 B 의 도구를 바꿨다**(`frontend-design` → Superdesign, 로컬 전용).

## 먼저 정한다 — 구역

`docs/ui/00_index.md` 구역표로 대상 화면의 구역을 본다.
HIGH 새 방향 → 이중. MEDIUM → B 만 돌리고 A 는 한 줄로 제안. LOW → 발산하지 않는다.

## 순서 — B 가 먼저다

1. **B 네이티브 디렉터.** 읽는 것: brief · 기준선 캡처(`results/screens/`) ·
   `ui/palette.json` · `demo/ui/tokens.css` · `scripts/_screens.py` · 지난 UD 의 「버린 것」.
   **읽지 않는 것: A 의 어떤 것도.**
   **도구는 Superdesign 이다** (UD-12). 로컬 세션에서:
   `npx -y @superdesign/cli@latest` · `DO_NOT_TRACK=1` · `search-prompts --tags style` 과
   `extract-website --url … --design-md` 로 **B 자신의 레퍼런스 경로**를 잡고,
   `create-project --template demo/ui/layout_b.html` 로 우리 화면을 넘기고,
   `iterate-design-draft --mode branch -p … -p …` 로 방향 여럿.
   **클라우드 세션에서는 못 돈다** — `api.superdesign.dev` 가 `403 to CONNECT`(UF-8).
   그때는 `frontend-design` 2 패스로 내려가되 **그것이 축소판임을 UD 에 적는다**
   (라운드 ① 이 그 축소판이었고 「과감하지 않다」는 평가를 받았다).
   후보 4~6 을 `dir_B.md` 형식으로.
2. **환경을 바꾼다** — 클라우드 ↔ 로컬. 어느 쪽이 B 였는지 적어 둔다.
   모델 분리는 선택지가 하나로 줄면 격리가 아니다(UD-12 3 행).
3. **A 레퍼런스 디렉터.** `02_benchmark` §8 카드 3~8 (어디를 볼지 · L1 캡처 ·
   Imitation Distance 3~4). 후보 5~8, **후보마다 REF 번호와 빌린 원리**.
   **A 는 자기 표를 다 쓴 뒤에야 B 의 절을 연다.**
4. **VS 줄세우기 — 판정이 아니라 순서다.** `02_benchmark` §11.
   A·B 후보 **전부**를 한 줄로 세운다. 동점 금지 · 이웃마다 왜 위인지 한 줄 ·
   줄 세울 때는 **후보 표만** 보고 설계안 절은 다시 읽지 않는다.
   네 물음: 주제 교체 시험 · 클리셰 대조 · 하루 뒤 한 문장 · 대담함 개수(1 이 최고).
   **아무것도 떨어뜨리지 않는다.** `vs_rank.md` 로 적는다.
5. **사용자 정렬 — 판정보다 먼저다.** 두 디렉터의 후보를 **전부** 사용자에게 보인다:
   후보마다 한 줄 설명 · 어느 REF 에서 왔나 · **빌린 경험 원리** · 위험 · **VS 순위**.
   레퍼런스는 **URL 과 「어디를 볼지」를 함께** 준다 — 그것이 레퍼런스 마이닝의 존재 이유다
   (사용자가 직접 보고 판단하라고 만든 단계다). 사용자가 고르거나 버린 뒤에 다음으로 간다.
   **검증기를 먼저 돌려 후보를 줄이지 않는다** — 그러면 사용자는 남은 것만 보게 되고,
   고르는 일이 판정으로 바뀐다 (UO-5).
6. **교차 검토.** 두 표를 나란히. 겹침은 합치고, 상보적 강점이 **실제로** 보일 때만
   Hybrid 하나. 어느 쪽 후보가 어느 축(구조·모션·타이포·깊이·데이터)에 몰렸는지 센다 —
   분포가 거의 같으면 격리가 실패한 것이고 그것을 UD 에 적는다.
7. **검증기 넷**(`ecg-ui-validator` · `motion-review` · `design-critique` ·
   `accessibility-review`) → KEEP/TUNE/REJECT → **UD 를 먼저 적는다.** 구현은 그 뒤.

## 산출물은 라운드 폴더 하나에 (UD-12 · `02_benchmark` §8.5)

```
results/screens/refs/round<N>/  dir_A.md · dir_B.md · vs_rank.md · REF-nn.png
```

**발산 표 원문을 UD 안에 두지 않는다.** UD 는 요약과 판정만 담는다 — 라운드 ① 에서 B 의
발산이 UD-11 안에만 있어서 사용자가 A 와 나란히 읽을 수 없었다.
`check_records.py` 가 이 폴더의 파일명과 「어느 UD 가 부르는가」를 본다.

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
- **Superdesign 을 실행한 척하지 않는다.** 이 세션에선 `403` 이라 한 줄도 안 돈다(UF-8 · UD-12).
  돌린 척한 출력은 기록이 아니라 위조다. 못 돌면 못 돌았다고 적는다.
- 생성기가 낸 시안을 **커밋하지 않고** 판정하지 않는다 — 내려와서 `demo/ui/branches/` 에 앉아야 기록이다.
- `ui/palette.json` 을 Superdesign 의 `design-system.md` 로 대체하지 않는다(UD-3 · UD-12).
- `confirm-generation` 을 부르지 않는다 — 크레딧을 쓰는 유일한 명령이고 이 UI 에 생성 이미지 자리가 없다.
- `data/arduino/` 를 **어떤 형태로도** 넘기지 않는다 `[사용자 지시]`. 넘기는 것은 `layout_b.html` 과 집계 수치뿐.

이 스킬은 자문이다. `docs/ui/00_index.md` 와 최신 사용자 지시가 우선한다.
