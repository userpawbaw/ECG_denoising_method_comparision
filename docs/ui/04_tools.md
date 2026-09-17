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
| | 공식 `frontend-design` (`anthropics/claude-code` `plugins/frontend-design`) | **참조** — 원문은 「All rights reserved」(`LICENSE.md`)라 복제하지 않고 방법만 우리 말로 로컬 스킬에 담았다. 고정 SHA `68ac8bb` | 2026-09-17 · raw fetch | 2 패스(계획 → brief 대조 자기 비판) · 클리셰 목록 · 「대담함은 한 곳에」 |
| | `algorithmic-art` (세션 내장 스킬) | 켜짐 | 2026-09-17 | HIGH 구역 attract 의 ECG 모티프 실험 |
| **DATA** 스토리 | `dataviz` (세션 내장 스킬) | 켜짐 | 2026-09-17 | 그리기 규격 · 팔레트 검증(`validate_palette.py` 의 원형) |
| | **Flourish** (MCP 커넥터) | **켜야 함** — 레지스트리에 있음, 미연결 | 2026-09-17 · `SearchMcpRegistry` | **스토리 초안** — 한 문장 insight 마다 표현 후보를 빠르게 만든다. MCP 는 단일 시각화 생성·데이터 바인딩·설정 편집까지이고 **Stories(순서·전환)는 Flourish 편집기에서** 만든다 `[문헌]`. 결과는 무빌드 시연에 그대로 못 들어간다(외부 임베드) — 채택안은 Canvas/SVG 로 재구현하거나 내보낸다 |
| **MOTION** | `motion-review` | **로컬** | 2026-09-17 | Scorecard 10 항 · 현행 스윕 계약(UF-4) |
| | **MDN** (MCP, 무인증) | **켜야 함** | 2026-09-17 · 레지스트리 | CSS·Canvas API·호환성 사실 확인 |
| | `modern-web-guidance` (Google Chrome 플러그인) | 켜야 함 (선택) | 2026-09-17 · 카탈로그 | scroll-driven · view transitions · anchor 등 HIGH 구역 전환 |
| **VALIDATION** | `ecg-ui-validator` | **로컬** | 2026-09-17 | KEEP/TUNE/REJECT · 증거 레벨 |
| | **`design`** (Anthropic 플러그인: `design-critique` · `accessibility-review`) | **켜야 함** — 카탈로그에 있음, 미활성 | 2026-09-17 · `SearchPlugins` | 스크린샷 입력 → 위계·일관성 비평 · WCAG 2.1 AA |
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
