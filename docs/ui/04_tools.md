# 도구·스킬 대장 — 역할 · 상태 · 출처 · **확인한 환경과 날짜**

> **UI/UX 파트 문서다.** 진입점은 `docs/ui/00_index.md`. 이 대장이 답하는 것은
> 「이 역할에 무엇을 쓰나, 지금 이 환경에서 되나, 어디서 왔나」 셋뿐이다.
> 도구를 고른 **이유**는 `docs/ui/03_external_system_review.md` 6·7 절과 UD-5 에 있다.
>
> **확인한 환경·날짜 열이 있는 이유** (UR-1): 다른 환경에서 「설치 확인」한 표는
> 여기서 사실이 아니다. 외부 저장소가 확인한 8 개 중 이 세션에 켜진 것은 하나였다.

## 1. 역할별 대장

상태: **켜짐** = 이 세션에서 지금 쓸 수 있다 · **켜야 함** = 사용자가 claude.ai 에서
연결/활성화해야 한다 · **로컬** = `.claude/skills/` · **기각** = 안 쓴다(이유는 03 6 절).

| 역할 | 도구 | 상태 | 확인 | 무엇에 |
|---|---|---|---|---|
| **CREATIVE** 발산 | `expo-ui-art-director` | **로컬** | 2026-09-17 · 이 세션 | 5 개 이상 후보를 UD 「검토한 선택지」 형식으로 |
| | 공식 **`frontend-design`** | **로컬 (복제)** — `.claude/skills/frontend-design/`. **Apache 2.0** 이라 그대로 가져왔다(UD-7 추기). 플러그인 판은 안 쓴다 | 2026-09-17 · `diff -q` 동일 확인 `[테스트]` | 2 패스(계획 → brief 대조 자기 비판) · 클리셰 목록 · 「대담함은 한 곳에」 |
| | `algorithmic-art` (세션 내장 스킬) | 켜짐 | 2026-09-17 | HIGH 구역 attract 의 ECG 모티프 실험 |
| | **`dual-creative-director`** | **로컬** | 2026-09-18 · 이 세션 | 발산 둘의 **독립** — B(코드베이스+`frontend-design`) 먼저, 모델 바꿔 A(레퍼런스 카드), 교차 검토 뒤 검증기. `00_index` 「이중 디렉터」 |
| **생성** 시안 브랜치 | 이 세션 + `shoot_screens.py` + 갤러리 + Artifact 발행 | 켜짐 | 2026-09-18 | `demo/ui/branches/UD-nn-{a,b,c}.html` — 같은 baseline 2~4 개, L2 캡처. **생성기는 승인 안 함** |
| | **Superdesign** (`superdesigndev/superdesign-skill` `f9f05cd` 스킬 0.6.0 · CLI `@superdesign/cli` **0.14.0**, `@latest` 로 부름) | **보류** — NEEDS USER ACTION(계정) | 2026-09-18 · 카탈로그 없음 `[측정]` · npm 닿음 `[측정]` | 9 기능 중 5 는 있고 2 는 의도된 공백, 2 는 범위 밖(UD-10 표). React 코드베이스용이라 무빌드 canonical 과 안 맞는다. **v2 작업이 여기로 오면 다시 본다** |
| **DATA** 스토리 | `dataviz` (세션 내장 스킬) | 켜짐 | 2026-09-17 | 그리기 규격 · 팔레트 검증(`validate_palette.py` 의 원형) |
| | **Flourish** (MCP 커넥터) | **켜짐** — 사용자가 연결했다 | 2026-09-17 · 이 세션의 도구 목록 | **스토리 초안** — 한 문장 insight 마다 표현 후보를 빠르게 만든다. MCP 는 단일 시각화 생성·데이터 바인딩·설정 편집까지이고 **Stories(순서·전환)는 Flourish 편집기에서** 만든다 `[문헌]`. 결과는 무빌드 시연에 그대로 못 들어간다(외부 임베드) — 채택안은 Canvas/SVG 로 재구현하거나 내보낸다 |
| **MOTION** | `motion-review` | **로컬** | 2026-09-17 | Scorecard 10 항 · 현행 스윕 계약(UF-4) |
| | **MDN** (MCP, 무인증) | **켜짐** — 사용자가 연결했다 | 2026-09-17 · 이 세션의 도구 목록 | CSS·Canvas API·호환성 사실 확인 |
| | `modern-web-guidance` (Google Chrome 플러그인) | 켜야 함 (선택) | 2026-09-17 · 카탈로그 | scroll-driven · view transitions · anchor 등 HIGH 구역 전환 |
| **VALIDATION** | `ecg-ui-validator` | **로컬** | 2026-09-17 | KEEP/TUNE/REJECT · 증거 레벨 |
| | `design-critique` · `accessibility-review` | **로컬 (복제)** — `design` 플러그인의 스킬 둘만. **MCP 9 개는 안 켰다**(UD-8). Apache 2.0 | 2026-09-17 · `diff -q` 동일 확인 `[테스트]` | 다섯 축 비평(첫 축이 **2 초 첫인상**) · **WCAG 2.1 AA 조항 대조표** |
| | `scripts/validate_palette.py` · `tests/test_demo_screens.py` | 켜짐 | 항상 | 정량 검증 — ΔE·명도대·색각 / 콘솔·넘침·색만으로 신원·reduced-motion·다크 |
| | `ui-ux-pro-max` (`nextlevelbuilder/…` `15de38f`) | 참조 (선택) | 2026-09-17 · upstream HEAD 일치 | 패턴/안티패턴 **검색만**. 디자인 시스템 생성기는 안 쓴다(docs/22 §6 · UD-1). 스크립트는 세션 밖에서 |
| | `design-taste` (`arez-xd/…` `a5c03fb`) | 참조 → 흡수 뒤 정리 | 2026-09-17 · upstream HEAD 일치 · **1 star** | motion·polish 참고 문서 두 개를 `02_benchmark` 에 흡수하면 의존을 끊는다 |
| **IMPLEMENTATION · QA** | 이 세션(Claude Code web) + Chromium + `pytest -m screens` + CI(`checks.yml`) + 아티팩트 발행 | 켜짐 | 항상 | 구현 · L2/L3 검증 · 폰에서 보기 |
| **RESEARCH** | WebSearch · WebFetch · `chrome --screenshot` | 켜짐 | 항상 | 레퍼런스 (UD-1 · `02_benchmark`) |
| **DESIGN SYSTEM** | **Figma** (MCP 커넥터) | **켜짐** | 2026-09-17 · `ListConnectors` | 경로 A 시안 보드 (UD-1 §5.2) · `ui/palette.json` ↔ Variables 대조 |

**기각** (이유는 `03` 6 절): Context7 · Motion AI Kit · TinyFish · browser-use · Vercel ·
Mobbin(UD-1) · Magic Patterns · Canva(UI) · HyperFrames · Anthropic `data` 플러그인.
「Creative Production」·「Product Design」은 이 조직 카탈로그에 **없다**.

## 2. 로컬 스킬 출처 등록 — `.claude/skills/`

`scripts/check_records.py` 가 이 표에 없는 스킬 디렉터리를 잡는다.

| 디렉터리 | 무엇 | 출처 | 우리가 바꾼 것 |
|---|---|---|---|
| `expo-ui-art-director` | 발산 — 후보 5 개 이상, UD 표 형식 | 외부 `ecg-gui-design-review` 의 동명 스킬(`62b65b1`) + 공식 `frontend-design` 의 방법(`68ac8bb`) | 경로를 이 저장소로, 팔레트는 `ui/palette.json` 을 통해서만, 시안 둘(v2 · `layout_b`) 표시 |
| `motion-review` | 모션 후보 검토 | 외부 동명 스킬(`62b65b1`) + `03_MOTION_AND_POLISH` Scorecard | 스윕 계약을 **UF-4 현행**(지우기 경계 페이드 + 선단 톤업)으로. 알파 가중 측정 함정 명시 |
| `ecg-ui-validator` | KEEP/TUNE/REJECT 판정 | 외부 `04_VALIDATION_AND_GUARDRAILS` + `ecg-ui-design` | 계약을 이 저장소 문서(`docs/21` §7 계열 · `02_benchmark`)로, 검증 명령을 우리 것으로 |
| `frontend-design` | 발산의 방법 — 2 패스 · 클리셰 목록 · 「대담함은 한 곳에」 | **Anthropic 공식**, `anthropics/claude-plugins-official` `plugins/frontend-design/skills/frontend-design/` · 판 `1.1.0` / `1aa8f02ec832` · **Apache 2.0** | **아무것도 안 바꿨다** — `SKILL.md`·`LICENSE.txt` 바이트 단위 동일. 출처·판·「고친 곳 없음」은 같은 폴더 `NOTICE` 에 |
| `design-critique` | 다섯 축 비평틀 — 2 초 첫인상 · 사용성 · 위계 · 일관성 · 접근성 | **Anthropic**, `anthropics/knowledge-work-plugins` `design/skills/design-critique/` · 플러그인 판 `1.2.0` · **Apache 2.0** | **안 바꿨다.** 본문의 `../../CONNECTORS.md` 링크는 여기 없다(플러그인 문서를 안 가져왔다) — `NOTICE` 에 적었다 |
| `accessibility-review` | WCAG 2.1 AA 조항 대조표 + 흔한 실패 8 · 검사 순서 | 같은 저장소 `design/skills/accessibility-review/` · 같은 판 · **Apache 2.0** | **안 바꿨다.** 위와 같다 |
| `dual-creative-director` | 발산 둘의 독립 — 순서·격리·교차 검토·가지치기 | 외부 `ecg-gui-design-review` **핸드오프** `handoffs/DUAL_CREATIVE_DIRECTOR_IMPLEMENTATION_2026-09-18.md`(`08ffec3`) — **원본에도 스킬은 아직 없다**, 여기가 첫 구현 | B 를 Superdesign 이 아니라 `frontend-design` 으로 · Chat/Work 분리 대신 **모델 분리 + B 먼저** · 브랜치 파일명 ↔ UD 검사 |

외부 `project-capability-audit` 는 스킬이 아니라 아래 3 절이 됐다.

## 3. 도구를 들이는 절차 (외부 `07` §4 · `09` 를 줄인 것)

1. 지금 있는 것부터 센다 — 이 표 · `ListSkills` · `ListConnectors` · `ListPlugins`.
2. **빈 능력**만 적는다. 「있으면 좋다」는 빈 능력이 아니다.
3. 후보를 **그 환경의 카탈로그에서** 검색한다. 이름만 듣고 적지 않는다(UR-1).
4. 겹침 · 출처/유지보수 · 스크립트/권한/네트워크 · 컨텍스트 비용을 본다.
5. INSTALL / OPTIONAL / REDUNDANT / REJECT — 이유를 `03` 방식으로 적는다.
6. 처음은 **project-local** 이고 SHA 를 고정한다. 여러 프로젝트에서 값이 확인되면 global.
7. 첫 사용 결과를 **UR** 로 남긴다 — 도구가 판단을 바꿨으면 그것이 R 의 재료다.

## 4. 없을 때 — 역할은 남기고 도구만 바꾼다

| 없으면 | 대신 |
|---|---|
| `design` 플러그인 | `ecg-ui-validator` + `02_benchmark` + `validate_palette.py` |
| Flourish | 같은 insight 를 먼저 한 문장으로 적고, `dataviz` 규격으로 정적 후보를 그린다 |
| MDN MCP | WebFetch 로 MDN 원문 |
| Figma | 경로 B(직접 HTML)만으로 간다 — UD-1 이 이미 그 경로를 열어 뒀다 |

## 5. 외부 서비스에 보내면 안 되는 것

- **`data/arduino/`** — 개인 생체정보. Flourish · Figma · Canva 어디에도 올리지 않는다 `[사용자 지시]`.
- 스토리 재료는 **집계 지표**(`results/` 의 요약 · 보고서 표)로 한정한다. 원 파형이 필요하면 D0 합성 또는 MIT-BIH 공개 기록만.
- 프로젝트 공개는 사용자가 허용했다(「공개로 해도 상관없다」 `[사용자 지시]`) — 그래도 위 둘은 예외다.

## 6. 플러그인을 어떻게 들이나 — 이 환경에는 `/plugin` UI 가 없다

**결론부터: 우리는 플러그인을 안 쓴다.** `frontend-design` 은 **Apache 2.0** 이라
`.claude/skills/frontend-design/` 에 그대로 복제했고, 그게 첫 세션부터 실리는 유일한
방법이다 (UD-7 추기). 아래는 **다른 플러그인이 필요해질 때**를 위한 기록이다.

근거와 기각한 안은 **UD-7**, AI 협업 교훈은 **UR-2**, 사고는 **UO-3**.

### 무엇이 되고 무엇이 안 되나 `[테스트]`

| 방법 | 마켓플레이스 등록 | 플러그인 설치 | **그 세션에 스킬이 실리나** |
|---|---|---|---|
| `.claude/settings.json` `extraKnownMarketplaces` | **✓** | — | — |
| 같은 파일 `enabledPlugins` | — | **✗** | ✗ |
| `SessionStart` 훅에서 `claude plugin install` | ✓ | ✓ (3.3 s) | **✗ — 훅은 스킬 목록 작성 뒤에 돈다** |
| **클라우드 환경 설정 스크립트** | ✓ | ✓ | **✓ — Claude Code 가 뜨기 전에 돈다** |
| **`.claude/skills/` 에 복제** (라이선스가 허락할 때) | — | — | **✓ · 설치도 네트워크도 필요 없다** |

### 설정 스크립트를 쓴다면 — **사용자만 할 수 있다**

claude.ai → 환경 설정 → **Setup script**. 끝나면 파일시스템이 스냅샷돼 이후 세션은 건너뛴다.

```bash
claude plugin marketplace add anthropics/claude-plugins-official
claude plugin install <plugin>@claude-plugins-official --scope user
```

네트워크 접근 수준은 **Trusted** 여야 한다 — 기본값이고 `github.com` ·
`raw.githubusercontent.com` 이 기본 허용 목록에 있다. **None** 이면 실패한다.
로컬에서는 `/plugin install <plugin>@claude-plugins-official` 한 번이면 끝이다.

### 확인

`claude plugin list` 에 `… enabled` 가 뜨는 것은 **설치됐다**는 뜻이지
**이번 세션이 그 스킬을 갖는다**는 뜻이 아니다 — 세션에게 직접
「그 스킬이 지금 있나」를 물어야 한다 (UR-2 재사용 규칙 3).

### 동봉이냐 외부냐 — 선언으로 되는 것의 경계

마켓플레이스 항목의 `source` 가 `"./plugins/foo"` 같은 **상대 경로면 동봉**,
`{"source":"url"|"github"|"npm", …}` 이면 **외부**다. 예: `frontend-design` 은 동봉,
`modern-web-guidance` · `figma` 는 외부다 `[코드]`.

### `design` 플러그인 — **스킬 둘만 가져왔고 MCP 아홉은 안 켰다** (UD-8)

플러그인을 통째로 켜면 `.mcp.json` 의 **MCP 서버 9 개**가 따라온다 `[코드]`. 세어 보면:

| MCP | 이 프로젝트에서 |
|---|---|
| `figma` | **이미 커넥터로 켜져 있다** — 중복 |
| `slack` · `asana` · `atlassian` · `linear` · `notion` · `intercom` | 쓸 자리가 없다 — 팀 이슈추적기·워크스페이스가 없다 |
| `google calendar` · `gmail` | **URL 이 빈 문자열**이라 안 붙는다 |

**플러그인이 들고 오는 MCP 만 끄는 설정 키는 없다** — `disabledMcpjsonServers` 는
**프로젝트 `.mcp.json`** 대상이다 `[문헌]`. 그래서 스킬 둘만 복제했다(2 절).

필요한 MCP 가 생기면 **플러그인 묶음이 아니라 커넥터 하나**로 켠다 — Figma · Flourish ·
MDN 이 그렇게 켜져 있다(1 절).

### claude.ai 에서 켠 플러그인은 여기 오나 — **이 세션에는 안 왔다** `[테스트]`

`~/.claude/plugins/synced/` 에 버킷이 있는데 **비어 있고 날짜가 컨테이너가 뜬 날**이다.
동기화는 **세션이 시작될 때** 한 번 돌므로, 그 뒤에 claude.ai 에서 켠 것은 **그 세션에
안 내려온다.** 새 세션이면 내려올 수 있고, v2.1.273 이상은 `claude plugin list` 에
`synced` 로 뜬다 `[문헌]` (여기는 2.1.274).

**복제한 스킬은 이 문제가 아예 없다** — 저장소에 있으니 클론하는 순간 있다.

### 「MCP 로그인 대기 중 유휴」가 무슨 뜻인가 — 재로그인 요청이 아니다

공식 문서 그대로 `[문헌]`:

> A session counts as inactive while it waits for you to approve an MCP connector
> tool call or to sign in to an MCP server, and it can expire during that wait.
> Reopen the session from claude.ai/code to provision a fresh VM with your
> conversation history restored.

즉 **승인·로그인을 기다리는 시간이 「활동」으로 안 세어져** 비활동 타이머가 돈다는 뜻이다.
오래 방치하면 **VM 이 회수**되고, 다시 열면 **대화 기록은 살아 있는 채 새 VM** 이 뜬다.
사람이 옆에 있으면 몇 초 만에 승인하니 실제 위험은 낮다 — **자리를 비운 사이**가 문제다.
회수되면 그때 돌던 백그라운드 작업(서브에이전트·셸 명령)은 복구되지 않는다.
