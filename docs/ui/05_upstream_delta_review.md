# 외부 UI/UX 시스템 **2 차** 검토 — `62b65b1` → `08ffec3` 의 변경분 `[판정 완료 → UD-10]`

> **무엇을 검토하나.** 사용자가 「다른 프로젝트에서 검증된 AI UI/UX 작업 시스템을
> 이식하라」는 프롬프트를 가져왔다(2026-09-18) `[대화]`. 그런데 이 세션은 **같은
> 시스템을 이미 한 번 이식했다** — `03_external_system_review.md` · UD-5, 원본
> `userpawbaw/ecg-gui-design-review @ 62b65b1`. 그래서 이 문서는 **새 이식이 아니라
> 재검토**다: 그 뒤 원본이 움직인 만큼(**5 커밋 · 26 파일 · +2,838 줄**)만 본다.
>
> 원본 시점 고정: **`08ffec369b07c75d52f0728e1ee28869f241efa1`** (2026-09-18) `[커밋]`.
> 채택 결정: **UD-10**(2026-09-18) — 「선택」으로 적은 `modern-web-guidance` 는 Superdesign 대체안이 **아니다**(UR-3). 원본 = 방법론 계보(upstream). 이 저장소 = 이 프로젝트의 운영 원본. 원본이 이후
> 바뀌어도 여기 행동은 **자동으로 안 바뀐다** — 다시 들이려면 이런 검토를 한 번 더 한다.
>
> **UI 를 바꾸지 않는다.** 이 문서는 분석과 제안까지다. 채택은 사용자 확인 뒤 **UD 로
> 먼저** 적고 그 다음 만든다.

## 0. 한 줄 결론

원본이 새로 만든 것은 셋 — **레퍼런스 마이닝**(13 · CASE-002), **Superdesign 생성기
계층**(14 · 15 · CASE-003), **Dual Creative Director**(핸드오프 674 줄, **아직 원본에도
구현 안 됨**). 셋 중 **방법론은 전부 가져올 만하고, 도구는 하나도 그대로 못 가져온다.**
그리고 프롬프트 자체에 **이 표적에서는 틀리는 조항**이 하나 있다(§2).

## 1. 원본 변경분 — 무엇이 늘었나 `[커밋]`

| 커밋 | 무엇 | 파일 |
|---|---|---|
| `7f9598f` | 레퍼런스 마이닝 — Reference Card · **viewing instruction 필수** · Imitation Distance 0~4 · 경험 원리 추출 | `13_…MINING.md` · `reference-mining` 스킬 · CASE-002 + 전사 발췌 · F-004 · D-007 · R-006 |
| `39a442c` | 기록 CI — Node 검사기 + GitHub Actions | `uiux-records-check.yml` · `check-uiux-records.cjs` |
| `ee678f7` `88a8c03` | Superdesign 을 「조건부 핵심 시안 생성기」로 채택 | `14_…LAYER.md` · `15_…EXAMPLES.md` · `superdesign-routing` 스킬 · CASE-003 · F-006 · D-009 · R-008 |
| `08ffec3` | **Dual Creative Director 핸드오프** — 사용자 반론(「Superdesign 이 스스로 발산하길 기대했다」)으로 Superdesign 을 `NATIVE_DIRECTOR + CONCRETIZER` 두 모드로 다시 놓고, 두 디렉터를 **1 차 패스에서 서로 안 보게** 하는 설계. **`16_DUAL…` 문서 · `dual-creative-director` 스킬 · CASE-004 는 아직 없다** | `handoffs/DUAL_…_2026-09-18.md` |
| 여러 | CASE 를 **사람이 다시 읽는 산출물**로 격상 — 인용 ≥ 2 또는 `[재구성]`/`기록 없음` 명시, 사용자/AI 기여 분리를 검사기가 본다 | `10` · `11` · 검사기 · F-005 · D-008 · R-007 |

## 2. 프롬프트 자체의 문제 — 표적이 ECG 프로젝트일 때 2-C 가 뒤집힌다

프롬프트는 **다른 프로젝트로** 옮길 때를 위해 쓰였다. 2-C 「SOURCE-PROJECT-SPECIFIC —
가져오면 안 되는 것」에 *ECG waveform semantics · Reference/Difference 계약 · 단위·시간축
계약 · ECG 지표* 가 있고, 3 절은 「원본 ECG 의 invariant 를 현재 프로젝트 invariant 대신
쓰지 마라」고 한다.

**여기서는 그것이 우리 invariant 다.** `docs/21` §7 계열 · UF-3(50 mm/s) · UF-4(스윕
계약) · `motion-review` 의 절대선 · LOW 구역 전부가 그 목록이다. 2-C 를 글자대로 따르면
**LOW 구역이 지워진다.** 그래서 C 를 둘로 가른다:

| | 무엇 | 처리 |
|---|---|---|
| **C-1 원본의 산출물** | `prototype/v2` 경로 · 2.2.1 코드베이스 · Expo 화면명(Lab/Evidence/Method Explorer) 을 **파일 경로로** 쓰는 것 · Chat/Work/Codex 3 분할과 핸드오프 패킷 · Node 검사기 · `records/` 3 자리 번호 · `AGENTS.md` · `.superdesign/` 상태 · ChatGPT 플러그인 디렉터리 조사 | **안 가져온다** (03 검토 판정 4·6·10·11·14 그대로) |
| **C-2 도메인 계약** | 파형 기하 · 시간축 · 단위 · Reference/Difference 의미 · 지표 의미 · 존재하지 않는 중간 파형 금지 | **가져올 것이 아니라 이미 우리 것이다.** 게다가 **우리 쪽이 더 새롭다** — 03 판정 9: 원본 스윕 계약은 UF-4 이전 판이다 |

이 자체가 UR 감이다 — 「이식 프롬프트가 다른 표적을 위해 쓰였으면, 버리라는 목록에
**자기 invariant 가 들어 있는지** 먼저 본다」(§11 UR-3 후보).

## 3. 세 분류 — 이번 변경분만

### A. DOMAIN-INDEPENDENT — 그대로 (또는 이미 있다)

| 원칙 | 상태 | 어디 |
|---|---|---|
| 발산 ↔ 검증 분리 · 생성기 ≠ 검증기 · KEEP/TUNE/REJECT | **있다** | UD-5 · `00_index` 운영 절 · 검증기 넷 |
| F/D/O/R · 기각 안 지움 · 구현 전 D · `[재구성]`/`기록 없음` | **있다** | `19_record_keeping` · UF/UD/UO/UR |
| progressive loading · capability 라우팅 | **있다** | CLAUDE.md 60 줄 · `00_index` 「작업 직전」 · `04_tools` |
| **독립 이중 디렉터** — 다른 inspiration diet · 1 차 패스 격리 · 교차 검토 · 필요할 때만 Hybrid | **새것. 채택** | §5 |
| **레퍼런스 = 시안 이전의 저비용 시각 정렬 대리물** — Reference Card · viewing instruction · Imitation Distance 3~4 기본 | **새것. 채택** | §4 |
| **생성기는 같은 baseline 에서 2~4 가지만, 방향 프롬프트로, 자기 결과 승인 금지** | **새것. 절차만 채택** — 도구는 아니다 | §6 |
| CASE = 사람이 다시 읽는 산출물 · 인용 ≥ 2 / 부재 명시 · 기여 분리 | **UR 이 이미 그 형식이다** — UR-1·UR-2 가 사용자 발화를 `[대화]` 로 인용한다. **검사기만 없다** → `check_records.py` 에 한 줄 | §10 |
| 「새 디자인 AI 에는 capability 한 자리만 준다」(R-008) | 채택 — UD-8 이 이미 그렇게 했다(`design` 플러그인의 스킬 둘만) | — |

### B. DOMAIN-ADAPTIVE — 원리는 두고 우리 맥락으로

| 원본 | 우리 맥락에서 바꿀 것 |
|---|---|
| 레퍼런스 소스 표(Awwwards · Godly · SiteInspire · Lapa · Mobbin · 21st.dev) | **소스가 다르다.** 우리는 부스 노트북의 **연구용 계측 시연**이다 — `02_benchmark` 의 범주(임상 기록지 · 환자 모니터 · 오실로스코프 · 과학관 인터랙티브 · NASA Eyes 류 · 데이터 저널리즘)가 맞다. Awwwards 류는 HIGH 구역(attract) 에만. **Mobbin 은 UD-1 에서 기각** — 되살리지 않는다 |
| Reference Card 의 「ECG translation」 | 이름만 「프로젝트 번역」이 아니라 **「구역 + 계약 무손실」** — 우리 UD 「검토한 선택지」표의 「구역 · 데이터/UX 위험 · validator 가 볼 것」열과 합친다 |
| 레퍼런스 링크는 사라진다 → 장면 설명을 남겨라 | **우리는 캡처할 수 있다** — `chrome --screenshot` 이 있으니 Reference Card 마다 **L1 캡처**를 `results/screens/refs/round<N>/` 에 담는다(원본이 못 하는 것) |
| Superdesign 의 baseline = `prototype/v2`(React) | **우리 canonical 은 둘**(UD-5 ①): `layout_b`(무빌드, 여기) + v2(React, 외부 저장소). Superdesign 은 React 코드베이스용이라 **v2 에는 맞고 `layout_b` 에는 안 맞는다** → §6 |
| 검증 차원 8 개(13 §13) | 우리 검증기 넷에 **「imitation risk / reference anchoring」한 차원만 더한다** — 나머지는 있다 |
| 구역별 디렉터 예산(HIGH=듀얼 자동 · MEDIUM=A 먼저 · LOW=없음) | 그대로 — 우리 구역표(`00_index`)에 열 하나 |
| 환경별 동작(Chat/Work/Codex/Claude Code) | **이 세션 하나**다 — 브레인·셸·브라우저·CI 가 한 곳. 핸드오프 패킷은 `99_status` §10 인수인계로 이미 접었다(03 판정 4). 단 **격리**는 남는다 → §11 ④ |
| 기록 CI(Node) | **이미 있다** — `checks.yml` 의 `pytest tests/` 안에 `check_records` 가 든다. 추가 없음 |

### C. 안 가져오는 것 — §2 의 C-1 그대로. 덧붙여:

- 원본 F-004~006 · D-007~009 · R-006~008 · CASE-002/003 — **인용만**(UD-5 ③). 번호도 안 옮긴다.
- `15_SUPERDESIGN_USAGE_EXAMPLES.md` 의 화면별 예시 — Attract/Evidence/Method Explorer 는 v2 의 화면이다.
- Superdesign 「web app」 경로 · `.superdesign/design-system.md` — 우리 디자인 시스템 원본은 `ui/palette.json` 하나다(UD-3).

## 4. 레퍼런스 마이닝 — 가져올 것과 우리 자리

원본 13 절의 파이프라인은 우리 운영 절과 **한 단계만 다르다**: DIVERGE **앞에**
「REFERENCE CARDS → 경험 원리 → 번역」이 들어간다. 채택안:

- **Reference Card 규격**(`REF-nn` · 출처 · URL · **어디를 볼지** · 잊히지 않는 순간 · 기제 ·
  기대 감각 · 프로젝트 번역 · **복사 금지 목록** · 구역 · Imitation Distance · 위험 · 최소
  시안)을 `02_benchmark.md` 에 **절 하나로** 얹는다 — 새 파일이 아니다. 02 가 이미
  레퍼런스 분석 문서다.
- **L1 캡처 의무**: 카드마다 그 장면을 `chrome --screenshot` 으로 찍어 둔다. 링크가
  죽어도 카드가 산다. 이건 원본이 못 하고 우리가 하는 것.
- 자동/제안/생략 트리거는 원본 그대로 쓰되 화면명만 우리 것 — HIGH 구역(attract ·
  Replay↔Live · reveal) 새 방향이면 자동, MEDIUM 이면 제안, LOW·polish 면 생략.
- **UD-6 · UD-9 는 레퍼런스 없이 돌았다.** 둘 다 `[캡처]` 기준선은 봤지만 외부 장면은
  안 봤다. 다음 라운드부터 HIGH 구역은 카드 먼저.

## 5. Dual Creative Director — 우리에게 맞는 모양

원본 정의를 그대로 옮기면 **B 가 Superdesign 이어야** 하는데(§6 에서 못 쓴다), 원칙은
도구가 아니라 **「서로 다른 inspiration diet 를 가진 두 독립 발산, 교차 검토는 뒤에」**다.
우리 자리:

| | Director A — 레퍼런스 디렉터 | Director B — 네이티브 디렉터 |
|---|---|---|
| diet | 외부 실제 장면(§4 카드) | **코드베이스 + 디자인 토큰 + `frontend-design` 2 패스** — UD-9 가 정확히 이 모양이었다 |
| 스킬 | `expo-ui-art-director` + (신설) 레퍼런스 카드 절 | `frontend-design` (복제판) + `ui/palette.json` · `_screens.py` |
| 공유 | brief · 기준선 캡처 · invariant · 구역 · **기각 대장**(UD-6 「버린 것」) | 같음 |
| 1 차 패스 금지 | B 의 후보·토큰 계획 | **A 의 카드·URL·후보** |
| 출력 | 후보 5~8, UD 「검토한 선택지」표 형식 | 색·서체·레이아웃 토큰 계획 + 후보 4~6 |
| 그 다음 | **교차 검토** → 필요할 때만 Hybrid 하나 → 검증기 넷 → KEEP/TUNE/REJECT | |

**격리를 어떻게 지키나** — 원본은 Chat/Work 로 자연히 갈렸지만 여기는 한 세션이다.
같은 문맥에서 A 다음 B 를 돌리면 B 가 A 를 본다(UD-9 가 UD-6 을 본 것처럼 — **이미
한 번 오염된 판이다**). 선택지 셋: ① B 를 **깨끗한 문맥의 서브에이전트**로(사용자 지시가
있어야 띄운다) ② **모델을 가른다** — UD-6 이 이미 Fable/Opus A/B 를 예측으로 걸어 뒀다,
그 실험이 곧 이것이다 ③ 순서를 바꿔 **B 먼저**(코드베이스만 보고), A 는 그 뒤 —
A 가 B 를 보는 오염은 방향이 반대라 덜 해롭다(레퍼런스는 어차피 외부에서 온다).
**권고: ② + ③.** 비용 없이 격리가 된다.

## 6. 생성기 계층 — Superdesign 은 지금 안 들인다, 절차만 들인다

| 잰 것 `[측정]` 2026-09-18 | 결과 |
|---|---|
| claude.ai 플러그인 카탈로그 | **없다.** `design` 플러그인만 뜬다(이미 UD-8 로 복제) |
| npm 레지스트리 `@superdesign/cli` | **닿는다 — 0.14.0.** node 22 · npx 있음 |
| 원본이 검토한 판 | ~~매니페스트 **0.6.0** (`f9f05cd`, 2026-08-21) — **버전이 두 배 넘게 뛰었다.**~~ **정정(UD-10)**: 다른 둘을 비교했다. 스킬 패키지는 지금도 0.6.0 이고 **CLI** 가 0.14.0 이며 스킬이 그것을 **`@latest` 로** 부른다 — 위험은 「뛰었다」가 아니라 **핀이 없다**는 것. 원본 CASE-003 의 경고(「버전을 적어라」)는 그대로 맞다 |
| 계정·인증 | 필요. 이 세션엔 없다 → **NEEDS USER ACTION** |
| 컨텍스트 전송 | 외부 서비스로 UI 소스를 보낸다. `04_tools` §5(`data/arduino/` 절대 금지 · 파형은 D0 합성/MIT-BIH 만) 안에서는 **가능** |
| 대상 코드베이스 | **React 기존 코드베이스**가 강점. `layout_b` 는 단일 HTML `file://` — Superdesign 의 값(코드베이스 분석 · 브랜치)이 여기선 거의 안 나온다. **v2 에는 맞는데 v2 는 외부 저장소에 있다** |

그래서 **도구는 보류, 절차는 채택**:

- 「같은 baseline · 2~4 브랜치 · 방향 프롬프트 · 생성기는 승인 안 함 · canvas 에서 멈춤」을
  **우리 도구로** 한다 — `layout_b.html` 을 복제해 `demo/ui/branches/<UD>-<a|b|c>.html`
  로 2~4 개, `shoot_screens.py --only` 로 L2 캡처, 갤러리에서 나란히. **비용이 Superdesign
  보다 싸고 오프라인이다.**
- Superdesign 은 **v2 쪽 작업이 이 저장소로 오거나**(UD-5 ① 의 「둘 다」가 「하나」로
  좁혀지면) 사용자가 계정을 열면 다시 본다. 그때는 **CLI 버전·모델을 UD 에 적는다**.
- 브랜치 HTML 은 exploratory 다 — `SCREENS` 레지스트리에 안 올린다. 대신 **§10 의
  「없는 기록」유도**의 원천이 된다.

## 7. 이 저장소의 invariant — 프롬프트 5 번

프롬프트가 「원본 invariant 를 쓰지 마라」고 한 자리에 **우리 것**을 적는다. 전부 문서에
있고 대부분 검사가 문다.

| invariant | 근거 | 무는 검사 |
|---|---|---|
| 무빌드 · `file://` · 시연 중 계산 없음 | UD-1 §3.1 · D-18 | `test_demo_screens` |
| 50 mm/s · 10 mm/mV · 격자가 규약 | UF-3 | 스윕 검사 |
| 존재하지 않는 중간 파형 금지 · 크로스페이드 금지 | `motion-review` 절대선 · UF-4 | UF-7 판 잔광 검사 |
| Reference/Difference 의미 · 색은 방법에 고정 · 원본은 `ui/palette.json` 하나 | UD-2 · UD-3 · D-33 | `test_palette` · `build_tokens --check` |
| `data/arduino/` 는 저장소 밖 · Flourish/Figma/Canva 에 안 올린다 | CLAUDE.md · `04_tools` §5 | — (사람) |
| 연구용 신호처리 시스템 — 진단 아님, 화면 문구 상시 | `01_design` 1.2 | `ecg-ui-validator` |
| 결정은 구현 전 · 기각 안 지움 · 수치는 파일에서 | `19` · D-32 | `check_records` · `test_repo_integrity` |
| CLAUDE.md ≤ 60 줄 · 테스트 개수 대조 · 루트 문서도 검사 | D-35 | `test_claude_md_test_count…` |
| 화면 레지스트리 = `scripts/_screens.py` · 화면마다 UD | UD-5 · 03 4-D | `test_every_registered_screen_has_a_decision` |
| 표시 1920×1080 · 라이트/다크 둘 다 · 관람객 기준 | UD-1 | `-m screens` · CI 스크린샷 |

## 8. capability 지도 — 프롬프트 9 번 `[측정]` 2026-09-18 이 세션

| capability | 판정 | 무엇으로 |
|---|---|---|
| web/reference research | **AVAILABLE** | WebFetch · WebSearch · **`chrome --screenshot`(L1 캡처)** |
| creative direction | **AVAILABLE** | `expo-ui-art-director` · `frontend-design` |
| visual generation | **PARTIAL** | 세션 내 HTML 브랜치 + Chromium L2 (§6). Superdesign CLI 는 닿지만 계정·용도 불일치 → **NEEDS USER ACTION** |
| data visualization | **AVAILABLE** | `dataviz` · Flourish MCP |
| UX validation | **AVAILABLE** | `ecg-ui-validator` · `design-critique` |
| visual/taste validation | **PARTIAL** | `design-critique` + 클리셰 목록. **취향의 최종은 L4 = 사람** — 도구로 안 메워진다 |
| motion review | **AVAILABLE** | `motion-review` · MDN |
| design system / Figma | **AVAILABLE** | Figma MCP · `ui/palette.json` |
| frontend implementation | **AVAILABLE** | 이 세션 |
| library documentation | **AVAILABLE** | MDN MCP (Context7 기각) |
| browser/runtime verification | **AVAILABLE** | Playwright · `pytest -m screens` · `serial_bridge --replay` · CI 스크린샷 |
| GitHub | **AVAILABLE** | GitHub MCP |
| automated tests | **AVAILABLE** | `pytest` 699 · `check_records` · CI |

**막는 공백은 없다.** 부족한 것(프롬프트 7 번): Superdesign 계정(v2 쪽에서만 값) ·
`modern-web-guidance`(선택, HIGH 전환용). Mobbin 은 기각 유지.

## 9. 만들 것 — 최소 구조 (프롬프트 8 · 12 번)

원본식 `docs/uiux_system/` 열다섯 파일을 **만들지 않는다.** 우리 것은 `docs/ui/` 이고 이미
아홉이다. 이번 변경분으로 느는 것:

| 파일 | 무엇 | 언제 |
|---|---|---|
| `docs/ui/05_upstream_delta_review.md` | 이 문서 | **지금** (분석) |
| `docs/ui/02_benchmark.md` §「레퍼런스 카드」 | 카드 규격 · 소스 표(우리 범주) · L1 캡처 의무 · Imitation Distance | 확인 뒤 |
| `docs/ui/00_index.md` 운영 절 | 「이중 디렉터」한 절 — A/B 정의 · 공유/금지 · 구역별 예산 · 교차 검토 · 브랜치 절차 · 짧은 명령 | 확인 뒤 |
| `.claude/skills/dual-creative-director/SKILL.md` | 격리 강제 · 순서(B 먼저 권고) · 모델 분리 · 교차 검토 전 상대 출력 비공개 · 브랜치 2~4 | 확인 뒤 · `04_tools` §2 등록 |
| `scripts/check_records.py` | UR 대화 근거 검사(인용 ≥ 2 또는 `[재구성]`/`기록 없음`) · 브랜치 HTML ↔ UD 대조(§10) | 확인 뒤 |
| `docs/ui/10_decisions.md` **UD-10** | 채택 결정 — 위 전부, 되돌릴 조건 포함 | 확인 뒤, **구현 전** |
| `docs/ui/13_ai_collaboration.md` **UR-3** | §2 의 교훈 | 확인 뒤 |
| `docs/ui/04_tools.md` | Superdesign 행(보류 · 0.14.0 · 계정) · 이중 디렉터 스킬 행 | 확인 뒤 |

새 문서는 **하나**(이 검토서). 나머지는 있는 문서에 절을 더한다.

## 10. 기록·검사 설계 — 프롬프트 10 번

원칙 그대로: **수동 장부를 하나 더 만들지 않는다.** 「잊을 수 없는 원천」에서 빠진
기록을 유도한다.

| 원천 | 유도하는 의무 | 상태 |
|---|---|---|
| `scripts/_screens.py` `SCREENS` | 화면마다 UD | **있다** |
| `ui/palette.json` | 색 바꾸면 `build_tokens` + 검증 | **있다** |
| `.claude/skills/*` 디렉터리 | `04_tools` §2 출처 | **있다** |
| **`demo/ui/branches/*.html`**(생성기 산출물) | 파일명의 `UD-nn` 이 실재하고 그 UD 가 그 브랜치를 부른다 | **신설** — 생성기 계층의 「없는 기록」탐지 |
| `docs/ui/13_ai_collaboration.md` UR | 「사람이 문제 삼은 것」에 인용 ≥ 2 또는 부재 명시 | **신설** — 원본 F-005 의 교훈, 우리 형식으로 |

CI 는 손대지 않는다 — `pytest tests/` 에 다 든다.

## 11. 충돌·불확실 — 프롬프트 11 번

1. **프롬프트 2-C 가 우리 invariant 를 버리라 한다** — §2 로 가른다. 재해석 없이 따르면 LOW 구역이 지워진다.
2. **Dual Director 는 원본에도 없다.** 핸드오프뿐이고 `16_` · 스킬 · CASE-004 가 미작성이다. 우리가 **첫 구현자**가 된다 — 원본이 나중에 다르게 만들면 갈린다. 그래서 SHA 고정 + 명시적 재검토 절차(이 문서의 방식).
3. **Superdesign 은 React 용이고 우리 canonical 은 무빌드**다. 그리고 검토판 0.6.0 ↔ 현재 0.14.0. 절차만 들이고 도구는 보류(§6).
4. **한 세션에서의 격리.** A/B 가 같은 문맥이면 독립이 아니다. UD-9 가 UD-6 을 보고 돈 것이 그 증거. 권고는 모델 분리 + B 먼저(§5) — **서브에이전트는 사용자 지시가 있어야** 띄운다.
5. **UO-4 의 전제가 틀렸다** — `checks.yml:110` 이 `-m screens` 를 **이미 돈다.** 빨강은 Actions 에 떠 있었고 **아무도 안 읽었다.** 승급 후보가 「도는 주체를 적어라」에서 「**CI 결과를 읽는 규칙**」으로 바뀐다. 지금은 멈춘 개선이라 UO-4 추기만 남길 것.
6. **레퍼런스 소스 접근** — Awwwards/Godly 는 JS 렌더라 WebFetch 가 본문을 못 읽을 수 있다. `chrome --screenshot` 이 대안이나 **라운드마다 한 번 실측**해야 한다. 아직 안 쟀다 `[추론]`.

## 12. activation — 프롬프트 9 번

| 명령 | 하는 일 |
|---|---|
| 「**새 디자인 라운드**」 | 구역을 보고 단일/이중 결정. HIGH → 이중 자동, MEDIUM → A 만 + B 제안, LOW → 발산 없음 |
| 「**레퍼런스 디렉터만**」 | A — 카드 3~8 + L1 캡처 + 후보 5~8 |
| 「**네이티브 디렉터만**」 | B — `frontend-design` 2 패스, 코드베이스만 보고 |
| 「**이중 디렉터**」 | B 먼저(권고) → A → 교차 검토 → 검증기 넷 |
| 「**시안 가지치기** UD-nn a/b/c」 | 승인된 2~4 후보를 `branches/` HTML 로, L2 캡처, 갤러리 |

전부 **UD 먼저**(구현 전) — 기존 문턱 그대로다.

## 13. 근거

| 주장 | 근거 |
|---|---|
| 원본 변경분 5 커밋 · 26 파일 | `git log 62b65b1..08ffec3` · `git diff --stat` `[커밋]` |
| Dual Director 미구현 | 원본 트리에 `16_*` · `dual-creative-director` · `CASE-004*` 없음 `[코드]` |
| Superdesign 카탈로그 없음 · npm 0.14.0 | `SearchPlugins` · `npm view @superdesign/cli version` `[측정]` |
| CI 가 screens 를 돈다 | `.github/workflows/checks.yml:110` `[코드]` |
| UD-9 가 UD-6 을 봤다 | UD-9 본문이 UD-6 #1·#3 을 인용한다 `[코드]` |
