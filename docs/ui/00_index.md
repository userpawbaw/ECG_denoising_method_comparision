# UI/UX 파트 — 진입점

> **이 파트는 연구·구현 파트와 분리돼 있다.** 시연 화면의 설계·디자인 시스템·
> 검증만 다룬다. 신호처리 방법, 실험 설계, 학습은 `docs/91_report.md` 쪽이다.
>
> **분리한 이유**: UI 작업은 판단 기준이 다르다 — 신호처리는 「재서 가릴 수 있나」로
> 정하고, 화면은 「보는 사람에게 전달되나」로 정한다. 한 문서에 섞으면 둘 중
> 하나가 상대의 기준으로 평가받는다. 다만 **기록 규격은 그대로 따른다**
> (`docs/19_record_keeping.md`) — 고른 순간에 적고, 기각한 것을 지우지 않고,
> 수치는 파일에서 읽는다.

## 기록 체계 — 번호만 다르고 규격은 같다

| 여기 | 연구 파트 | 무엇 |
|---|---|---|
| **UD-n** | `D-n` (`docs/21_decisions.md`) | 결정 — **실행 전에** 적는다. 무엇을 왜 버렸는지가 값이다 |
| **UF-n** | `F-n` (`docs/20_findings.md`) | 발견 — 이상한 것을 본 순간, 원인을 알기 전에 적는다 |
| **UO-n** | `O-n` (`docs/22_incidents.md`) | 사고 — 같은 실수가 반복되면 규칙으로 승급시킨다 |
| **UR-n** | `R-n` (`docs/23_ai_review.md`) | AI 협업 교훈 — **재사용 규칙**으로 끝난다. 없으면 UR 이 아니다 (UD-5) |

번호를 나눈 것은 **섞이지 않게 하려는 것**이지 격을 낮추려는 것이 아니다.
UI 결정이 연구 결론에 영향을 주면(예: 그림 색이 바뀌면 보고서 그림이 바뀐다)
양쪽에 상호 참조를 남긴다.

## 문서

| 파일 | 무엇 |
|---|---|
| `docs/ui/01_system.md` | **설계 계획서** — 진단 · 외부안 판정 · 단계별 계획. 이 파트의 기준선 |
| `docs/ui/10_decisions.md` | UD 기록 |
| `docs/ui/11_findings.md` | UF 기록 |
| `docs/ui/12_incidents.md` | UO 기록 |
| `demo/ui/layout_b.html` | **3 단계 레이아웃 시안 — 경로 B**(직접 구현). 갤러리에서 열린다 |
| `docs/ui/02_benchmark.md` | **레퍼런스 분석** — 표시 규약 · 비교 UI · 레이아웃 · 타이포 · 반응형, 그리고 **빌리지 않기로 한 것** |
| `docs/ui/03_external_system_review.md` | **외부 UI/UX 시스템 이식 검토** — `userpawbaw/ecg-gui-design-review` 의 오케스트레이션 계층을 판정하고, 이식 시 충돌 지점과 도구 조합을 적었다. 결정은 **UD-5** |
| `docs/ui/04_tools.md` | **도구·스킬 대장** — 역할 · 상태(켜짐/켜야 함/로컬/기각) · 출처 SHA · **확인한 환경·날짜**. 로컬 스킬 출처 등록도 여기 |
| `docs/ui/13_ai_collaboration.md` | UR 기록 |

기록 파일 다섯(UD·UF·UO·UR·도구 대장)이 모두 있다. **외부 시스템의 원문**(`ecg-gui-design-review/docs/uiux_system/`)은 복제하지 않고 인용한다(UD-5 ③·④).

## 운영 — 외부 시스템에서 **체계만** 들인 것 (UD-5)

### 역할 넷 — 한 역할에 「창의적이되 절대 안전하게」를 요구하지 않는다

| 역할 | 하는 일 | 어디 |
|---|---|---|
| **발산** (Creative Art Director) | 후보 **5 개 이상**, 구역을 붙여, UD 「검토한 선택지」 표 형식으로. 결정권 없음 | `.claude/skills/expo-ui-art-director` |
| **데이터 스토리** | 한 문장 insight → 관계(rank · slope · trade-off · before/after) → 표현 후보. **순서와 전환**이 우리에게 빈 능력이다(UR-1) | `dataviz` · Flourish(초안) — `04_tools.md` |
| **검증** (Validator) | KEEP / TUNE / REJECT + 되돌릴 조건. 「취향」은 이유가 아니다 | `.claude/skills/ecg-ui-validator` · `motion-review` |
| **구현** | 승인 범위만. 무빌드 시연판을 유지한다(UD-1) | 이 세션 · `pytest tests/ -m screens` · CI |

### 구역 — 대담함은 HIGH 에, 절제는 LOW 에

| 구역 | 이 프로젝트에서 | 시안 (UD-5 ①) | 자유도 |
|---|---|---|---|
| **HIGH** | attract/idle · Replay↔Live 전환 · 결과 reveal | **v2**(Attract 있음) — `layout_b` 에는 아직 없다 | 높음. 단 데이터로 오해될 표현은 금지 |
| **MEDIUM** | 결과 요약 · 지표 · 방법 설명 · 내비게이션 | 둘 다 — **첫 실험에서 가른다** | 중간 |
| **LOW** | 파형 판독 레인 · 시간축 · 단위 · Reference/Difference · 수치 | **`layout_b`** (UF-4 실측이 앞선다) | 낮음. 정확성·비교 가능성 우선 |

### 증거 레벨 — 근거 태그(출처)와 다른 축(깊이)

L0 IDEA → L1 SOURCE(코드·명세) → **L2 STATIC**(`scripts/shoot_screens.py`) → **L3 INTERACTIVE**(`pytest tests/ -m screens`) → **L4 TARGET**(노트북 1920×1080 실기). 낮은 레벨의 PASS 를 높은 레벨로 읽지 않는다 — 「자동 검사 PASS」는 「실기 PASS」가 아니다.

### 한 번의 개선 — 순서

**BASELINE**(화면을 실제로 연다) → 분류(CREATIVE / DATA / MOTION / UX / IMPLEMENTATION / RESEARCH) → **DIVERGE**(≥5) → **UD 먼저**(후보와 기각 이유) → **CONVERGE**(KEEP/TUNE/REJECT) → 변경 계약(대상 · 금지 · 유지할 계약 · 수용 기준 · 증거 레벨) → 구현 → 검증 → 기록(UF/UD/UO/UR).

| 분류 | 먼저 읽을 것 | 도구 |
|---|---|---|
| CREATIVE | 이 절 · `docs/ui/02_benchmark.md` 6 절(빌리지 않기로 한 것) · `10_decisions` 의 「버린 것」 | art-director (+ 공식 `frontend-design` 의 방법) |
| DATA | `docs/34_visual_plan.md` 4 절(주장 ↔ 그림) · 보고서 5 장 | `dataviz` · Flourish(초안) |
| MOTION | UF-4 · `motion-review` | MDN |
| UX | `02_benchmark` · `01_system` 7 절(화면 규격) | validator · `design` 플러그인 |
| IMPLEMENTATION | 해당 UD 의 변경 계약 | 이 세션 · CI |
| RESEARCH | `02_benchmark` 0 절(무엇을 봤나) | WebSearch · `chrome --screenshot` |

도구의 상태·출처·없을 때의 대안은 `docs/ui/04_tools.md`.

### 사용자 최소 입력

**화면 위치(번호·요소) + 불편한 점 + 원하는 결과.** 움직임 문제면 재현 순서나 짧은 영상. 버전 · 영향 범위 · 검증 항목은 AI 가 채운다 — 긴 양식이 요청의 진입 조건이 되지 않게 한다.

## 연구 파트와의 접점

분리했지만 끊긴 것은 아니다. 세 자리에서 만난다.

| 접점 | 어디 |
|---|---|
| **시연 설계의 전사(前史)** | `docs/30_realtime_demo.md` · `docs/31_demo_design_review.md` · `docs/32_metric_cards.md` · `docs/33_card_design_samples.md` · `docs/34_visual_plan.md` — 이 파트가 시작되기 전에 쌓인 것들이고, 그대로 유효하다 |
| **그림 색** | 보고서 그림과 시연 화면이 같은 팔레트를 쓴다. 바꾸면 양쪽이 함께 바뀐다 (UD-2) |
| **기록 규약** | `docs/19_record_keeping.md` · `docs/17_checklists.md` |

## 현재 상태

| 단계 | 무엇 | 상태 |
|---|---|---|
| 0 | 레퍼런스 분석 | **✅ 끝** — `02_benchmark.md`. 3 단계로 일곱 항목을 넘겼다 (UF-3 포함) |
| 1 | 값을 한 곳으로 — 검증기 · 팔레트 원본 · 토큰 생성 | **✅ 끝** — 색 부분. 타이포·간격은 0 단계 뒤 (UD-3) |
| 2 | 화면을 여는 장치 — 갤러리 · 스크린샷 · 화면 검사 | **✅ 끝** (UD-4). 상태별 갤러리는 3 단계 뒤 |
| 3 | 레이아웃 재설계 (경로 둘로 짝지어 비교) | **경로 B ✅** `demo/ui/layout_b.html` (미감 개선 ①~⑤ · 스윕 잔광/페이드 포함) · 경로 A(Figma) 착수 |
| 4 | 실패 상태 화면 · 색각 이상 검증 | 대기 |
| 체계 | 외부 시스템 이식(UD-5) — 운영 절 · `04_tools` · UR · 검사 · 로컬 스킬 셋 | **✅ 끝.** 첫 시운전(발산 → 검증 → 후보) 이 다음 |

**경로 B 를 눌러 볼 수 있다** — `https://claude.ai/artifact/8k1bdcp4EEGfRjj5Cvvnuh`
(비공개. 휴대폰에서도 열린다. 뷰포트를 1280 으로 고정해 **설계된 폭 그대로**
보이게 했다 — 폰 폭으로 접으면 검사하려던 레이아웃이 아니게 된다).
고치면 같은 URL 로 다시 올린다. 이 링크는 **저장소 밖에만 있다** —
`docs/99_status.md` 10.1 에도 적어 뒀다.

**push 하면 Actions 가 검사하고 스크린샷 42 장을 올린다**
(`.github/workflows/checks.yml` · 근거와 한계는 `docs/99_status.md`).
**「미감적으로 별로」는 CI 가 못 잡는다** — 그래서 찍어서 올린다. 보는 일을
없애지는 못하고 **보는 비용을 0 으로** 만드는 장치다.

확정된 화면 규격: **노트북 1920×1080** (최소 방어폭 1280×720) · **라이트·다크 둘 다** ·
**조작은 관람객과 발표자 둘 다**(기본을 관람객 기준으로).

## 도구

```
python3 scripts/validate_palette.py "<hex 목록>" --mode light [--scope lane|legend]
python3 scripts/build_tokens.py            # ui/palette.json → CSS · Python
python3 scripts/build_tokens.py --check    # 생성물이 최신인지
python3 -m pytest tests/test_palette.py
```

**색을 고칠 곳은 `ui/palette.json` 하나다.** 고치면 `build_tokens.py` 를 돌린다 —
잊으면 검사가 잡는다.

## 작업 직전 — 무엇을 하면 무엇을 도나

| 하려는 것 | 할 것 |
|---|---|
| **화면(HTML/CSS/JS)을 고쳤다** | `pytest tests/ -m screens` — 기본 실행에서 빠져 있다 |
| **색·토큰을 고쳤다** | `python3 scripts/build_tokens.py` → `pytest tests/test_palette.py` |
| **화면이 어떻게 보이는지 보고 싶다** | `demo/ui/gallery.html` 을 열거나 `python3 scripts/shoot_screens.py --all` |
| **새 색을 고른다** | `scripts/validate_palette.py` 로 **먼저 재고** 고른다. 눈으로 고르지 않는다 |
| **그림 스크립트를 돌린다** | 그 명령은 **저장한다**(UO-1). 한글 폰트가 있는 환경인지 먼저 보고, 결과를 열어 본 뒤 커밋한다 |
| **화면을 더 인상적으로 만들고 싶다** | `expo-ui-art-director` 로 발산 → **UD 먼저** → `ecg-ui-validator` 로 판정. 위 「운영」 절 |
| **AI/도구의 제안이 판단을 바꿨다** | UR 후보 — `docs/ui/13_ai_collaboration.md` |
| **새 도구·스킬을 들이고 싶다** | `docs/ui/04_tools.md` 3 절. 카탈로그를 **실제로** 검색한다(UR-1) |
| **검사와 커밋을 한 줄로 잇는다** | `\| tail` 뒤에 `&&` 를 두지 않는다 — 종료 코드를 직접 받는다(UO-2) |
