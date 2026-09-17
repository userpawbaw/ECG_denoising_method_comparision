# 외부 UI/UX 시스템 이식 검토 — `ecg-gui-design-review` `[검토 중]`

> **UI/UX 파트 문서다.** 진입점과 기록 체계는 `docs/ui/00_index.md`.
>
> **출처**: 사용자가 다른 AI 와 구축한 UI/UX 오케스트레이션 시스템 —
> `userpawbaw/ecg-gui-design-review` @ `62b65b1` (`docs/uiux_system/` 13 문서 ·
> `records/` 4 · `cases/` 2 · `.claude/skills/` 4 · `scripts/check-uiux-records.cjs`).
> 이 문서는 `docs/ui/01_system.md` 가 이전 외부 대화(Figma·Mobbin·shadcn·Storybook 안)를
> 판정한 것과 **같은 방식**으로 그 시스템을 판정한다.
>
> **이 문서의 성격**: 판정과 추천이다. 결정이 아니다 — 8 절의 셋을 사용자가 고른 뒤
> **UD-5 로 적고** 그 다음에 이식한다(`docs/19_record_keeping.md` §5 「실행 전에 적는다」).

---

## 0. 한 줄 결론

**뼈대는 가져올 가치가 있고, 살은 대부분 다시 붙여야 한다.**

| | 판정 |
|---|---|
| **가져온다** | 발산/검증 역할 분리 · Creative Freedom Zone(HIGH/MEDIUM/LOW) · 증거 레벨 L0~L4 · **R(AI 협업 교훈) 기록 종류** · CASE 해설 계층 · 작업 6 분류 라우팅 · handoff/return packet 형식 |
| **그대로 옮기면 충돌하는 곳 넷** | ① 기술 기준선(React vs **무빌드 — UD-1 과 정면 충돌**) · ② 기록 ID(외부 `F-001`~ 이 연구 파트 `F-1..49` 와 **충돌**) · ③ 모션 계약(외부 문서는 이 저장소가 **이미 구현·실측한 UF-4 보다 뒤**에 있다) · ④ 도구 목록(외부가 「설치 확인」한 8 개 중 이 세션에 켜진 것은 **Figma 하나**) |
| **시스템 자체의 상태** | **설계됐고 시운전은 안 됐다.** 기록 15 건의 근거 태그가 전부 `[대화]`·`[추론]`·`[커밋]`·`[코드]`·`[플러그인]` 이고 **`[캡처]`·`[런타임]`·`[테스트]` 가 0** 이다 `[측정]`. 검증 계층(04·08)이 한 번도 작동한 적이 없다 — CASE 「한계」가 이를 인정한다 |
| **도구 조합** | 외부안이 창의 발산에 지목한 「Creative Production · Product Design」은 이 조직 카탈로그에 **없다**. 그 역할에는 외부안이 놓친 **Anthropic 공식 `frontend-design`** 이 맞다. 검증 역할에는 공식 **`design`** 플러그인(critique · accessibility). 최종 조합은 7 절 |
| **이식 전 사용자 결정 셋** | 어느 시안이 박람회 화면인가 · 브랜치 둘을 어떻게 합치나 · 외부 기록을 옮기나 인용하나 — 8 절 |

---

## 1. 무엇을 읽었고 무엇을 안 했나 `[코드]`

**읽은 것** (전문): `README.md`, `docs/uiux_system/README.md`, `00`~`12` 열세 문서,
`records/{F,D,O,R}`, `cases/CASE-001` 둘, `.claude/skills/*` 넷, `AGENTS.md`,
`WORK_RESUME_POLICY.md`, `WORK_STATE.json`, `docs/21`·`22`(현행 refinement 규칙과
최신 polish 결정), `scripts/check-uiux-records.cjs`, `docs/templates/ui_change_request.md`.

**구조만 본 것**: `prototype/v2/` — React 19 + TS + Vite 7 + Tailwind 4 + Radix/CVA.
`src/main.tsx` 29 KB 한 파일에 상태·라우팅이 모여 있고, `Plot.tsx` 5.8 KB 캔버스,
`engine.ts` 3 KB 트랜스포트, `style.css` 14.6 KB. 단위 테스트 4 · Playwright spec 1(69 줄).

**여기서 실제로 돌린 것**: `node scripts/check-uiux-records.cjs` → **PASS, 15 records**
(Node 22 — `engines` 는 ≥24 라고 적혀 있다) `[테스트]`.

**안 한 것**: 그쪽 `npm test`(jsdom 필요) · `prototype/v2` 실행(98 장면×600 s 재생
자료 328 MB ZIP 이 git 밖) · 그쪽 Playwright QA.

**도구 조사** (2026-09-17, 이 세션의 카탈로그 도구로): 플러그인 검색 8 회 ·
MCP 레지스트리 검색 7 회 · 스킬 검색 4 회(결과 0) · 이 세션의 플러그인/스킬/커넥터
목록 · 상류 저장소 넷 fetch · 웹 검색 3 회. 근거는 10 절.

---

## 2. 시스템이 무엇인가 — 한 표

| 층 | 무엇 | 파일 |
|---|---|---|
| L1 | 항상 읽는 진입점 · 우선순위 · 6 분류 라우팅 | `AGENTS.md` · `00_UIUX_MASTER.md` |
| 설계 계층 | 창의 발산(01) · 데이터 스토리(02) · 모션(03) · 검증 KEEP/TUNE/REJECT(04) · 도구 라우팅(05) · Chat/Work 분리(06) · 스킬 출처(07) · 실험 단위(08) · 도구 도입 감사(09) | `01`~`09` |
| L2 | 트리거별 체크리스트 + 승급 대장 | `11_CHECKLISTS.md` |
| L3 | **F / D / O / R** 네 종류 + 근거 태그 12 개 | `10_RECORD_KEEPING.md` · `records/` |
| CASE | 여러 F/D/O/R 을 묶은 방법론 서사 + 영문 one-page | `cases/` |
| 검사 | 필수 절 · ID 중복 · 근거 태그 · CASE→기록 연결 · 스킬 provenance · 진입점 | `check-uiux-records.cjs` (`npm test` 포함) |

역할 넷(Creative Art Director / Data Storyteller / Validator / Implementation),
자유도 구역 셋(HIGH·MEDIUM·LOW), 증거 레벨 다섯(L0 IDEA ~ L4 TARGET).

**이 저장소와의 관계**: 그쪽 `12_RECORD_SYSTEM_LINEAGE` 가 밝히듯 **원형이 여기다** —
`docs/19_record_keeping.md` · `17_checklists.md` · `check_records.py` · CLAUDE.md 60 줄.
그러니 「이식」은 왕복이다: 우리 규약이 UI 도메인으로 나갔다가 R·CASE·구역·증거 레벨을
얻어 돌아온다.

---

## 3. 판정 — 채택 · 변형 · 기각

| # | 외부안 항목 | 판정 | 이유 · 이 저장소의 대응물 |
|---|---|---|---|
| 1 | 발산 ↔ 검증 역할 분리 (D-001) | **✓ 채택** | 여기 없던 것. 「창의적이되 절대 위험하지 않게」를 한 역할에 요구하지 않는다는 논리는 맞다. 스킬 둘로 구현(7 절) |
| 2 | Creative Freedom Zone (D-002) | **✓ 채택 · 변형** | 이 저장소 화면에 매핑하면 **HIGH 구역이 비어 있다** — `_screens.py` 의 `SCREENS` 에 attract/intro 가 없다. 외부 v2 에는 Attract 가 있다. 구역표는 어느 시안이 canonical 인지에 딸린다(8 절 ①) |
| 3 | Data Storyteller · Flourish (D-003) | **△ 변형** | 여기서는 **`dataviz` 스킬이 이미 DATA 계층**이다 — `validate_palette.py` 의 원형이 거기서 나왔다(UD-1). 스토리 후보는 보고서 그림 S1~S6(`docs/34`) 에 있다. Flourish 는 OPTIONAL |
| 4 | Chat / GitHub / Work 3 분할 (D-004) | **△ 변형** | 이 세션(Claude Code web)은 브레인과 실행이 **한 곳**이다 — Chromium·pytest·CI·아티팩트 발행이 다 있다. 둘로 접는다: **세션 ↔ GitHub**. handoff/return packet 형식은 세션 압축·인수인계(`99_status` §10)에 그대로 쓴다 — 그 부분은 채택 |
| 5 | 작업 6 분류 라우팅 (D-005) | **✓ 채택 · 재접지** | 분류는 좋다. 도구 열은 **전부 다시 쓴다**(6·7 절) |
| 6 | F/D/O/R + CASE (D-006) | **✓ 채택 · 변형** | **R 은 새 종류**다 → `UR-n` (가칭 `13_ai_collaboration.md`). CASE 는 링크. **ID 충돌**: 외부 `F-001`~`F-003`, `D-001`~`D-006`, `O-001`, `R-001`~`R-005` 는 연구 파트 번호와 겹친다 → 들여온다면 전부 `UF/UD/UO/UR`. 검사기는 **하나만** — `check_records.py` 확장. Python 저장소에 Node 검사기를 두 번째로 들이지 않는다 |
| 7 | 증거 레벨 L0~L4 | **✓ 채택** | 「태그는 출처 축, L 은 깊이 축」이라는 구분이 정확하다. 매핑: **L2 = `shoot_screens.py`**, **L3 = `test_demo_screens.py`**(Playwright), **L4 = 노트북 1920×1080 실기**(UD-1 규격) |
| 8 | Motion Scorecard 10 항 | **✓ 채택** | 비교 도구로만. 「data-integrity critical FAIL 은 총점 무관」 조항 포함 |
| 9 | 03 §2 스윕 계약 · docs/21 §7.9 | **✗ 낡았다** | 외부: 「선단 glow 는 후보, blanket 승인 아님」「페이드는 지우기 경계에만」. 여기: **UF-4 로 둘 다 구현·실측** — 페이드 + 선단 톤업, 알파 가중 프로브로 측정. 그대로 옮기면 **출하된 것을 미승인으로 되돌린다.** 계약을 UF-4 기준으로 다시 쓴다 |
| 10 | 00 §7 기술 기준선 — React 19 + Vite + Tailwind, 「새 HTML 목업으로 재작성하지 않는다」 | **✗ UD-1 과 정면 충돌** | 사용자가 **「무빌드 시연판 + 개발용 검증판」** 을 골랐고 React 전환을 기각했다(D-18 · 「시연 중에 실패할 계산이 없다」). 지금 시안이 **둘** 산다 — v2.2.1(React · 98 장면×600 s · Attract·Pin·Difference) 과 `layout_b.html`(무빌드 · 스윕 페이드/글로우 · 토큰). **어느 쪽이 박람회 화면인가**는 시스템 문제가 아니라 제품 결정이다 → 8 절 ① |
| 11 | `AGENTS.md` (영문 ~50 줄) | **✗ 형식 기각 · 내용 채택** | 여기 조종석은 CLAUDE.md **60 줄 상한(54/60)**. 라우팅은 `docs/ui/00_index.md` 「작업 직전」표를 넓히고 CLAUDE.md 는 한 줄 |
| 12 | 로컬 스킬 넷 | **△ 변형** | 각 10 줄. 경로가 `docs/uiux_system/…` 로 박혀 있어 그대로는 못 쓴다. `.claude/skills/` 는 이 저장소에 아직 없다. 재작성해서 들인다(7 절). `ecg-ui-design` 과 `project-capability-audit` 는 스킬보다 `00_index` 의 절이 맞다 |
| 13 | 08 실험 ID `UX-YYYYMMDD-NN` | **✗ 기각** | 세 번째 번호 체계가 된다. 실험 카드는 **UD 안의 「실험 설계」절**로 — UD-1 이 이미 그렇게 한다(축 셋과 자) |
| 14 | `WORK_RESUME_POLICY` · `WORK_STATE.json` · 6 시간 automation | **△ 참고만** | 대응물은 watchdog Routine + **락을 잡는 러너**(O-23~25). 외부 O-001(stale global lock)은 우리가 반대 방향에서 얻은 것과 **같은 교훈**이다 — 자원 단위 락. 이미 일치한다 |
| 15 | 04 §5 과장 금지 | **✓ 채택** | CLAUDE.md 「임상·진단 능력을 주장하지 않는다」와 같은 계열. 「automated DOM PASS 를 target-PC PASS 로 확대하지 않는다」를 UI 파트 규칙으로 |
| 16 | 07 스킬 provenance 절차 · 09 도입 감사 | **✓ 채택** | INSTALL/OPTIONAL/REDUNDANT/REJECT 4 분류와 「project-local 먼저」. 검사 항목 「`.claude/skills/*` 가 provenance 에 있나」는 `check_records.py` 에 |
| 17 | `ui_change_request.md` 템플릿 | **△ 참고** | 「사용자 최소 입력」 절(화면 위치 + 불편 + 원하는 결과)은 좋다. AI 변경 계약 절은 UD 양식과 겹친다 — 최소 입력만 `00_index` 에 |

---

## 4. 시스템 자체의 결함 — 이식과 무관하게 올릴 것

### A. 근거가 전부 대화이고, 그 대화가 없다 `[측정]`

`records/*.md` 태그를 셌다:

| `[대화]` | `[커밋]` | `[추론]` | `[코드]` | `[플러그인]` | `[캡처]` | `[런타임]` | `[테스트]` | `[재구성]` | `기록 없음` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 28 | 11 | 9 | 3 | 3 | **0** | **0** | **0** | **0** | **0** |

자기 규약 §6 에서 `[대화]` 는 「당시 대화에 명시됨」이다. 그런데 CASE-001 머리말은
「이 저장소에는 현재 Chat 의 machine-export transcript 가 없다」고 쓴다. **대조할 수
없는 `[대화]` 28 개**다. 우리 규약(`19` §8)대로면 이것은 `[재구성]` 이거나 원문
부록(`docs/41` 방식)이 있어야 한다. 그럴듯해서 더 위험한 종류 — `19` §8 첫 문단이
바로 이 실수를 기록한 것이고, 그 저장소는 그 문단을 읽고도 같은 자리에 섰다.

그리고 **L2 이상 근거가 0** 이다. 검증 계층(04 매트릭스 · 08 비교 실험 · L0~L4)이
설계만 됐고 한 번도 돌지 않았다.

### B. D 가 구현 뒤에 적혔다 `[커밋]`

D-001~D-005 의 시점이 전부 2026-09-16 이고 시스템 v1 커밋(`c499516`)이 그 결과다.
규약 §3 「기록 시점은 구현 전」 위반이며, 정직하게는 `[재구성]` 표시가 맞다.
(우리도 F-1~9 에서 같은 일을 겪었다.)

### C. 검사기의 구멍 셋 `[코드]`

| 어디 | 무엇 | 결과 |
|---|---|---|
| `checkCases` 의 `\b([FDOR]-\d+)\b` | CASE 가 **연구 저장소** ID(예: D-28)를 인용하면 「존재하지 않는 기록」으로 실패 | 교차 저장소 참조 규약이 없다. 이식하면 UD/UF 접두가 이 문제를 푼다 |
| `checkSkillProvenance` | `provenance.includes(name)` — 문자열 포함 | 이름이 어디든 적혀 있으면 통과 |
| 근거 검사 | 태그 **하나라도** 있으면 통과 | `[추론]` 만으로 F 가 성립한다. 우리 `check_evidence_tags` 도 같은 수준이라 함께 올릴 것 |

### D. 「없는 기록」 탐지를 포기한 이유가 여기서는 성립하지 않는다

외부는 「component/token/route registry 가 canonical 하지 않아 가짜 ledger 를 만들지
않는다」고 포기했다(옳은 판단). 그런데 **여기는 레지스트리가 있다** —
`scripts/_screens.py` 의 `SCREENS` (화면) 와 `ui/palette.json` (토큰). 그러므로
「`SCREENS` 에 화면이 늘었는데 UD 가 없다」「`methods` 색이 바뀌었는데 UD 가 없다」를
**기계로 유도**할 수 있다. 연구 파트가 `MODELS` 에서 D 의무를 유도한 것과 같은 장치다.
외부보다 한 단계 올라간다.

### E. 문서가 많고 겹친다

13 문서 + 기록 4 + CASE 2 = 19 파일, 약 1,700 줄. 우리 UI 파트는 6 파일이다.
겹치는 곳: `00` §5 = `01` §3 = `ecg-ui-design` 스킬(구역표 세 벌) · `04` §2 = `08` §2 =
`11` §2(KEEP/TUNE/REJECT 세 벌) · `05` §5 = `07` §4 = `09`(도입 절차 세 벌).
이식하며 **합친다**(9 절 초안): 01+02+03 → 한 문서, 04+08 → 한 문서, 05+06+07+09 → 한 문서.

### F. 「설치/연결 확인」은 그 환경의 스냅샷이다 `[플러그인]`

`05` §1 이 확인했다는 GitHub · Product Design · Figma · Creative Production · Flourish ·
Context7 · TinyFish · Vercel 중 이 세션에 켜진 것은 **Figma 하나**다
(ListConnectors: Canva · Figma · Google Drive · HyperFrames / ListPlugins: 없음).
도구 표에는 「어느 환경에서 확인했나」열이 있어야 하고, 환경이 바뀌면 다시 접지한다.

### G. 브랜치 분기 `[커밋]`

외부 `12_RECORD_SYSTEM_LINEAGE` 는 이 저장소의 `docs/40~42` 를 원형으로 인용한다.
그 셋은 **`origin/claude/ecg-denoising-dsp-dl-comparison-b5wjvj` 에만 있다** —
merge-base `be3c7d8` 에서 그 브랜치 +11, 이 브랜치 +32. 그 브랜치의 마지막 커밋이
「소급 D 셋 (D-29·30·31)」이고, **이 브랜치의 D-29 는 UD-1 로 이관됐고 D-30 은 FE 캠페인**이다.
**같은 ID 가 두 브랜치에서 다른 것을 가리킨다.** `main` 은 없다. 이식 전에 합치거나
기준을 정해야 한다 — `test_ui_records_use_their_own_numbering` 이 막으려는 것과 같은
종류의 단절이다.

---

## 5. 이 세션이 외부 시스템에 더 줄 수 있는 것

| 무엇 | 외부 상태 | 여기 |
|---|---|---|
| **L3 브라우저 검증** | `WORK_STATE`: 「browser pixels NOT VERIFIED」 — Chromium 다운로드가 반복 timeout | Chromium 내장 + Playwright. v2 의 QA 를 **여기서** 돌릴 수 있다(재생 ZIP 328 MB 가 있으면) |
| CI 스크린샷 | 없음 | `checks.yml` 이 push 마다 42 장을 올린다 |
| 폰에서 열리는 링크 | Vercel 을 후보로 둠 | 아티팩트 발행 — 이미 쓰고 있다(`00_index`) |
| **정량 검증기** | 없음 — 대비 5.02:1 같은 숫자는 손계산 | `validate_palette.py`(ΔE·명도대·색각) · `test_demo_screens.py`(콘솔·넘침·색만으로 신원·reduced-motion·다크 명도) |

---

## 6. 도구 조사 — 역할별 후보와 판정

판정 기준은 외부 `09` 의 넷 — 관련성 · 기존 기능과 중복 · 출처/유지보수 ·
권한/네트워크 — 에 **「이 세션에서 지금 되나」** 를 더했다. 분류는 외부의
INSTALL / OPTIONAL / REDUNDANT / REJECT 를 그대로 쓴다.

### 6.1 CREATIVE — 발산

| 후보 | 실측 | 판정 |
|---|---|---|
| Creative Production · Product Design (외부안) | 그 이름의 플러그인이 카탈로그에 **없다** — 세 번 검색. 가장 가까운 것은 Anthropic `design` 인데 그건 **검증** 쪽이다 | — (존재 확인 불가) |
| **`frontend-design`** (Anthropic 공식, `anthropics/claude-code/plugins/frontend-design`) `[문헌]` | 「distinctive, intentional visual design … choices that don't read as templated defaults」. 2 패스 — 디자인 계획(팔레트·타이포 전략·레이아웃 개념) → **brief 에 비춰 자기 비판** → 코드. 금지 클리셰를 **이름으로** 적는다: 크림 배경+테라코타, 다크+형광 단색, SaaS 카드 키트, 자간 벌린 ALL-CAPS 라벨, 내용 없는 01/02/03, 로드 시 흩뿌린 fade-slide. 「boldness 는 한 곳에 쓴다」 | **INSTALL** — 「발산하되 평범으로 수렴하지 않는다」역할 그 자체. 단 **기존 디자인 시스템 유지에는 안 맞는다고 스스로 밝힌다** → HIGH/MEDIUM 구역 전용, LOW(파형·축·수치)엔 쓰지 않는다. project-local, SHA 고정 |
| `algorithmic-art` (세션 내장, p5.js seeded) | attract 화면의 ECG 모티프(스윕·격자·펄스) 생성 실험 | OPTIONAL — HIGH 구역이 생기면 |
| `canvas-design` (세션 내장) | 정적 포스터·PDF | OPTIONAL — **UI 밖**(박람회 포스터·부스 인쇄물) |
| 로컬 `expo-ui-art-director` | 10 줄. 「5 개 이상, 서로 다른 축, 각 후보에 목적·구역·기대 반응·난이도·위험·검증 항목」은 좋다 | 채택 · **재작성** — 출력이 UD 「검토한 선택지」표로 바로 들어가게 |

### 6.2 DATA — 스토리

| 후보 | 실측 | 판정 |
|---|---|---|
| Flourish (외부안) | MCP 존재 · 미설치 · `list_templates` / `create_visualisation` / `update_visualisation_data_bindings` | OPTIONAL — 인터랙티브 후보 탐색이 실제로 필요할 때. 값 왜곡 검사(외부 `02` §6 REJECT 기준) 전제 |
| **`dataviz`** (세션 내장) | **이미 이 저장소의 DATA 계층** — 색 검증기의 원형(UD-1), 형태 휴리스틱·마크 규격·팔레트 검증 | **KEEP** (이미 쓴다) |
| Anthropic `data` 플러그인 | create-viz · build-dashboard · SQL/웨어하우스 지향 | REDUNDANT |
| 보고서 그림 S1~S6 | `docs/34_visual_plan.md` 가 이미 「가장 설득력 있는 그림」을 골랐다(S5 crossover) | 스토리 후보의 **출발점** |

### 6.3 MOTION

| 후보 | 실측 | 판정 |
|---|---|---|
| Context7 (외부안) | MCP 존재 · 미설치 · `resolve-library-id` / `query-docs` — **라이브러리** 문서 지향 | **REJECT** — 우리 스택은 바닐라 Canvas/CSS. 볼 라이브러리가 없다 |
| Motion AI Kit (외부안 「가능 환경에서」) `[문헌]` | motion.dev 의 키트(`npx motion-ai`). MotionScore·spring 생성·transition editor 는 **Motion+ 유료**. React `motion` 라이브러리 중심 | **REJECT** — 유료·라이브러리 지향. 우리 스윕은 캔버스 직접 그리기(`destination-out`) |
| **MDN MCP** | 레지스트리에 있음 · **authless** · `get-doc` / `get-compat` / `search` | **INSTALL** — CSS·Canvas API·호환성 사실 확인. 가볍고 인증 없음 |
| **`modern-web-guidance`** (Google Chrome 플러그인) `[문헌]` | 108 features — scroll-driven animations · view transitions · container queries · anchor positioning · popover · dialog. 오프라인 로컬 검색, 네트워크 없음 | OPTIONAL — HIGH 구역 전환(Replay↔Live 같은 context switch)을 설계할 때 |
| Figma 플러그인의 `figma-use-motion` · `figma-implement-motion` | 카탈로그에 있음 | OPTIONAL — 경로 A(Figma 시안)에서 전환 시안을 만들 때 |
| 로컬 `motion-review` | 10 항 체크 + Scorecard | 채택 · **재작성** — UF-4 계약(페이드 + 선단 톤업 구현됨) 반영. 「존재하지 않는 중간 파형을 만들지 않는다」는 그대로 |
| design-taste 의 `motion-taste` · `motion-performance` refs | 마크다운 2 개 | 한 번 읽어 우리 `03`(모션) 문서에 흡수 |

### 6.4 UX · VALIDATION — 검증

| 후보 | 실측 | 판정 |
|---|---|---|
| **`design`** (Anthropic, knowledge-work-plugins) `[플러그인]` `[문헌]` | `design-critique`(usability · hierarchy · consistency) · `accessibility-review`(WCAG 2.1 AA, severity + remediation) · `design-system` · `design-handoff`. 입력: 스크린샷 / Figma 링크 / URL | **INSTALL** — Validator 역할에 정확히 맞는 공식 도구. 스크린샷을 넣으면 우리 CI 산출물 42 장이 그대로 입력이다 |
| design-taste (`arez-xd/ux-ui-design-taste`, 외부안) `[커밋]` | 상류 **1 star · 12 commits** · 마크다운 10 개 · 스크립트 없음. 고정 SHA `a5c03fb` = 현재 HEAD(일치) | OPTIONAL → **흡수 후 REJECT** — 유지보수 신호가 약하다. 내용은 짧고 읽을 만하니 한 번 읽어 `02_benchmark` 에 흡수하고 의존하지 않는다 |
| ui-ux-pro-max (`nextlevelbuilder/…`, 외부안) `[커밋]` | 상류 **128k stars** · Python BM25 검색 + CSV(79 styles · 192 palettes · 74 font pairings · 192 industry rules) · CLI 설치. 고정 SHA `15de38f` = 현재 HEAD(일치) | OPTIONAL — **이미 이 저장소가 측정했다**: UD-1 「축 1 의 첫 데이터 포인트」(dataviz 와 함께 다섯 항목 추가 발견). docs/22 §6 도 「Charts & Data LOW 기본값은 부적합」. 검색(패턴/안티패턴)만 쓰고 **디자인 시스템 생성기는 안 쓴다**. 스크립트 실행은 세션 밖 |
| `validate_palette.py` · `test_demo_screens.py` | 여기 것 | **KEEP** — 외부 시스템에 없는 정량 검증 |
| Canva 플러그인 `canva-design-feedback` | 커넥터 연결됨 | REJECT (UI) — 부스 인쇄물엔 가능 |

### 6.5 IMPLEMENTATION · QA

| 후보 | 실측 | 판정 |
|---|---|---|
| Work/Codex (외부안) | 이 세션이 그 역할을 이미 한다 — Chromium · pytest · CI · 아티팩트 · GitHub MCP | — (접힘, 3 절 #4) |
| TinyFish (외부안) | 플러그인·MCP 존재 · 미설치 · 검색/fetch/브라우저 에이전트 | **REJECT** — 세션 내장 WebSearch/WebFetch + Playwright 와 중복 |
| `browser-use` | 플러그인 존재 | REJECT — 같은 이유 |
| Vercel (외부안) | MCP 존재 · 미설치 | **REJECT** — 무빌드 `file://` 시연에 배포 미리보기는 대상이 없다. 폰 링크는 아티팩트가 맡는다 |
| 공식 Playwright 플러그인 | `claude.com` 은 egress 차단이라 본문 확인 못 함. 검색 결과로는 MCP — 접근성 트리 스냅샷·스크린샷·스크립트 `[문헌]` | OPTIONAL — `_screens.py` 가 같은 일을 pytest 안에서 한다. **대화형** 탐색이 필요할 때만 |

### 6.6 RESEARCH — 레퍼런스

| 후보 | 실측 | 판정 |
|---|---|---|
| Mobbin (외부안 「조건부」) | **MCP 존재** · 미설치 · `search_flows` / `search_screens` / `search_sections` | **REJECT 유지** — UD-1 이 구독을 기각(모바일 앱 플로우 DB vs 데스크톱 계측 화면). 되돌릴 조건: 데스크톱 계측/모니터링 화면이 DB 에 있음을 확인했을 때 |
| Magic Patterns MCP | 디자인 반복 도구 | REJECT — 경로 B 는 코드 직접, 경로 A 는 Figma |
| WebSearch · WebFetch · `chrome --screenshot` | 세션 내장 · UD-1 이 정한 경로 | KEEP |

### 6.7 DESIGN SYSTEM

| 후보 | 실측 | 판정 |
|---|---|---|
| **Figma MCP** | **연결됨** — `get_design_context` · `get_screenshot` · `get_variable_defs` · `create_design_system_rules` … | **KEEP** — UD-1 §5.2 「시안 보드이지 거울이 아니다」 유지. `ui/palette.json` ↔ Variables 양방향 대조 |
| `figma` 플러그인(스킬 14) | 카탈로그에 있음 | OPTIONAL — 경로 A 착수 시 |
| HyperFrames (연결됨) | HTML 모션 그래픽 → **영상** | REJECT — attract 를 영상으로 틀 계획이 없다. 생기면 재검토 |
| Google Drive (연결됨) | — | 무관 |

---

## 7. 최종 추천 조합

| 역할 | 1 순위 | 보조 (필요할 때) | 쓰지 않는다 |
|---|---|---|---|
| **CREATIVE** | **`frontend-design`**(공식 · 로컬 · SHA 고정) + `expo-ui-art-director`(재작성) | `algorithmic-art`(attract 모티프) | Creative Production · Product Design(카탈로그에 없음) |
| **DATA** | **`dataviz`**(내장) + 보고서 그림 | Flourish MCP | `data` 플러그인 |
| **MOTION** | `motion-review`(재작성 · UF-4 반영) + **MDN MCP** | `modern-web-guidance` · `figma-use-motion`(경로 A) | Context7 · Motion AI Kit |
| **VALIDATION** | **`design`** 플러그인(critique · accessibility) + `validate_palette.py` + `test_demo_screens.py` | ui-ux-pro-max **검색만**(세션 밖) | design-taste(흡수 뒤) · Canva |
| **IMPLEMENTATION / QA** | 이 세션 + `pytest -m screens` + CI 스크린샷 + 아티팩트 | Playwright 플러그인(대화형) | TinyFish · browser-use · Vercel |
| **RESEARCH** | WebSearch/WebFetch + `chrome --screenshot` | — | Mobbin(UD-1) · Magic Patterns |
| **DESIGN SYSTEM** | **Figma MCP**(연결됨) + `ui/palette.json` | `figma` 플러그인 | HyperFrames |

**실제로 새로 들이는 것은 셋뿐이다**: `frontend-design`(project-local) ·
`design` 플러그인(카탈로그에서 켠다) · MDN MCP(authless). 나머지는 이미 있거나 안 쓴다.
**설치는 UD-5 뒤에**, 외부 `07` §4 절차(inventory → gap → 후보 → 중복 → scripts/권한 →
분류 → local 먼저) 그대로. 첫 사용 결과를 **UR-1** 로 남긴다.

외부안이 **통째로 놓친 역할 둘**: ① **정량 검증기**(색·화면) — 여기 이미 있다.
② **공식 창의 발산 스킬** — 외부안은 1 star 저장소와 128k star 저장소를 고르면서
Anthropic 자체 것을 못 봤다.

---

## 8. 사용자 결정이 필요한 것 — 이식 전

| # | 갈림길 | 선택지 | 내 권고 |
|---|---|---|---|
| ① | **어느 시안이 박람회 화면인가** | v2.2.1(React, 외부) / `layout_b.html`(무빌드, 여기) / 둘 다 — 역할 분담 | **권고 없음** — 제품 결정이다. 시스템은 어느 쪽이든 붙는다. 다만 「둘 다」면 HIGH 구역(attract)은 v2, LOW 구역(파형 판독)은 layout_b 의 스윕 실측이 앞서 있다는 것만 적어 둔다 |
| ② | **브랜치** — `b5wjvj`(`docs/40~42` · 소급 D-29~31) 와 이 브랜치(`docs/ui` · D-30 캠페인) | 합친다 / 이 브랜치를 기준으로 40~42 만 가져온다 / 둔다 | **합친다** — 연구 파트 D 번호가 갈라진 채 두면 커밋 역추적이 끊긴다. 충돌하는 D-29~31 은 한쪽을 다시 번호 매기고 이관 표시 |
| ③ | **외부 기록을 옮기나 인용하나** | UR 새로 시작 + CASE 링크 + 외부 기록은 인용만 / 외부 F/D/O/R 15 건을 UF/UD/UO/UR 로 번호 바꿔 들여온다 | **전자** — 외부 기록의 근거가 그 저장소의 대화이고 여기서 재검증할 수 없다. 들여온다면 전부 `[재구성]` 이 붙어야 하고, 그러면 값이 떨어진다. 그 저장소가 그 대화의 canonical 기록으로 남는 것이 맞다 |

---

## 9. 결정 뒤 이식 순서 — 초안

앞 단계 산출물이 있을 때만 다음으로 (UD-1 의 순서 규칙과 같다).

| 순 | 무엇 | 산출물 |
|---:|---|---|
| 1 | **UD-5** — 이 문서를 근거로 ①②③ 의 답과 이식 범위를 적는다 | `docs/ui/10_decisions.md` |
| 2 | 브랜치 정리(②) | 합쳐진 브랜치 · D 번호 이관 표시 |
| 3 | `docs/ui/00_index.md` 에 **6 분류 라우팅 · L0~L4 · 구역표** · CLAUDE.md 한 줄 | 진입점 |
| 4 | **UR** 문서 + `check_records.py` 확장 — UR 필수 절(AI 가 내놓은 것 / 사람이 문제 삼은 것 / 검증 / **재사용 규칙**) · 스킬 provenance · **`SCREENS`→UD 유도**(4-D) | 가칭 `13_ai_collaboration.md` · 검사 |
| 5 | 외부 01+02+03 → 가칭 `04_creative_data_motion.md`(스윕 계약은 UF-4 기준) · 04+08 → `05_validation.md` · 05+06+07+09 → `06_tools.md`(6·7 절 표) | 문서 셋 |
| 6 | `.claude/skills/` — `frontend-design`(vendored · SHA) · `expo-ui-art-director` · `motion-review` · `ecg-ui-validator` | 스킬 넷 + provenance |
| 7 | `design` 플러그인 · MDN MCP 켜기 → 첫 사용을 **UR-1** 로 | 도구 셋 |
| 8 | 첫 실전: **4 단계(실패 상태 화면 · 색각 검증)** 를 새 라우팅으로 돌려 본다 — 시스템의 첫 시운전 | UF/UD + L2/L3 근거가 붙은 첫 기록 |

---

## 10. 근거 — 이 문서가 틀렸다면 무엇을 확인하나

| 주장 | 확인할 곳 |
|---|---|
| 외부 시스템의 내용 | `userpawbaw/ecg-gui-design-review` @ `62b65b18f07a4efb4c2f97dc37d801a66b4a2364` `[커밋]` |
| 검사기가 여기서 PASS | `node scripts/check-uiux-records.cjs` → `PASS — 15 F/D/O/R records` (Node v22.22.2) `[테스트]` |
| 태그 통계 | `grep -o '\[대화\]' docs/uiux_system/records/*.md \| wc -l` 등 `[측정]` |
| 브랜치 분기 | `git merge-base origin/claude/ecg-denoising-dsp-dl-comparison-b5wjvj HEAD` = `be3c7d8`; `git rev-list --count` +11 / +32; `git ls-tree` 로 `docs/40~42` 와 `docs/ui/` 가 서로 다른 쪽에만 `[커밋]` |
| 카탈로그·레지스트리 검색 | 이 세션의 `SearchPlugins` · `SearchMcpRegistry` · `ListPlugins` · `ListConnectors` 출력, 2026-09-17 `[플러그인]` — 환경이 바뀌면 다시 돈다 |
| 상류 SHA 일치 | `git ls-remote https://github.com/arez-xd/ux-ui-design-taste HEAD` = `a5c03fb…`; `…/nextlevelbuilder/ui-ux-pro-max-skill HEAD` = `15de38f…` `[커밋]` |
| `frontend-design` 본문 | [SKILL.md](https://github.com/anthropics/claude-code/blob/main/plugins/frontend-design/skills/frontend-design/SKILL.md) · [블로그](https://claude.com/blog/improving-frontend-design-through-skills) `[문헌]` |
| `design` 플러그인 | [anthropics/knowledge-work-plugins/design](https://github.com/anthropics/knowledge-work-plugins/tree/main/design) `[문헌]` — README 수준. 스킬 본문은 fetch 에 안 나와 **입력 형식은 README 의 예시로 추정** |
| `modern-web-guidance` | [GoogleChrome/modern-web-guidance](https://github.com/googlechrome/modern-web-guidance) `[문헌]` |
| Motion AI Kit | [motion.dev/ai-kit](https://motion.dev/ai-kit) · [docs](https://motion.dev/docs/ai-kit) · [motiondivision/ai-kit](https://github.com/motiondivision/ai-kit) `[문헌]` |
| Playwright 플러그인 | `claude.com/plugins/playwright` 는 **egress 차단**으로 못 봤다 → 세부는 `[근거 없음]`. 검색 결과: [QASkills](https://qaskills.sh/blog/playwright-mcp-claude-code-setup-2026) · [ap7i](https://ap7i.com/posts/giving-claude-code-eyes-with-playwright-mcp/) `[문헌]` |
| 외부 스킬 상류 상태(1 star · 128k stars 등) | 각 GitHub 페이지 2026-09-17 fetch `[문헌]` — 숫자는 그날 것 |
