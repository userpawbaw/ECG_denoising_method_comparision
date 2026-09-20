# Superdesign **자료만** 읽고 무엇을 들일 것인가 — 도구 없이 (UD-13)

> **이 문서가 답하는 것**: Superdesign 의 CLI 를 붙이지 않고 **문서만** 읽었을 때,
> 우리 Director B 가 가져다 쓸 방법이 무엇인가.
> 사용자 지시 「superdesign CLI를 붙이지 말고 superdesign의 자료 자체만 참고해보라는 거였어.
> 가능하면 그것부터 해보자」 `[대화]`.
>
> 채택 결정은 **UD-13**. 도구 쪽 결정은 **UD-12**(CLI 채택 · 로컬 전용 · 계정 대기)에 있고
> 이 문서는 그것과 **독립**이다 — 아래 방법들은 계정이 없어도 오늘 쓴다.

## 0. 무엇을 읽었나 `[문헌]`

`github.com` 은 이 환경에서 열린다(UF-8). 그래서 **자료는 전부 읽을 수 있었다** —
막힌 것은 `api.superdesign.dev` 뿐이다.

| 파일 | 크기 | 무엇 |
|---|---|---|
| `skills/superdesign/SKILL.md` | 16.5 KB | Core scenarios 9 · 라우팅 · init 완료 판정 |
| `references/SUPERDESIGN.md` | **48.4 KB** | 설계 SOP · 프롬프트 규칙 · 반복 모드 · 컨텍스트 예산 |
| `references/INIT.md` | 8.4 KB | 코드베이스 → 컨텍스트 6 파일 규격 |
| `references/WEBSITE.md` | 4.6 KB | URL → 디자인 DNA 레시피 |
| `references/GRAPHIC.md` | 16.5 KB | 그래픽 — **골격 메뉴**가 여기 있다 |
| `references/RESUME.md` · `PRESENTATION.md` · `ASSET_GENERATION.md` · `design-with-your-model.md` | 44.5 KB | 범위 밖 · CLI 결속 |
| 패키지 `README.md` (`npm pack`) | 13.9 KB | 명령 표면 · 인증 · 텔레메트리 |

전부 `raw.githubusercontent.com` 에서 `http=200` `[측정]`.

## 1. 들이는 것 여덟 — 왜, 그리고 어디에 앉히나

### S1. 스타일 원천은 **하나**다 — 둘을 섞지 않는다

> *"Pick ONE primary style source — do NOT blend two competing styles… do NOT layer a
> library style prompt on top of the extracted DNA (two competing styles dilute the result)."*
> — `SUPERDESIGN.md` SOP: BRAND NEW PROJECT Step 2

**우리에게 무슨 뜻인가**: 우리 변경 계약은 A1+A2+A5+B5+A8 **다섯**을 합쳤다. 언뜻 이 규칙에
걸린다. 걸리지 않는 이유를 적어 두는 것이 이 항의 값이다 — **다섯 중 시각 스타일 원천은
하나(UD-9 C1 「기록지 레인 · 흑연 데크」)뿐**이고 나머지는 구조(A1·A2·A8)와 색 정책(B5)이다.
**구조는 합쳐도 되고 스타일은 안 된다.** 다음 라운드에서 팔레트를 둘 섞으려는 순간 이 줄이 막는다.

**앉는 곳**: `dual-creative-director` 「하지 않는 것」.

### S2. **스타일은 못 박고, 발산은 구조·내용·서사로만** — 이게 라운드 ① 의 진짜 구멍이다

> *"The design system is a hard constraint, not a suggestion: iteration prompts explore
> layout/structure/content direction, **never visual style**."*
> *"Without explicit constraints, the design agent will invent random fonts, random colors…
> This happens because vague prompts like 'bold design' or 'modern feel' give the design
> agent creative freedom to deviate."* — `SUPERDESIGN.md` DESIGN SYSTEM FIDELITY

**우리에게 무슨 뜻인가**: 라운드 ① B 의 다섯 칸 중 **B5 는 색 정책**이었고, 1 패스의 상당 부분이
색 대비 표였다. 색은 **UD-9 에서 이미 정해진 것**이다 — 즉 **발산 슬롯 하나를 변수가 아닌 것에
썼다.** 다섯 중 넷만 구조·서사에 갔다. 도구가 0 이었던 것(UD-12)과 별개의, **절차 쪽 원인**이다.

**앉는 곳**: 발산 브리프 — 「이번 라운드의 고정값」을 표로 먼저 적고, 후보는 고정값을 건드리지 않는다.

### S3. 각 방향은 **「바꾸는 것」과 「그대로 두는 것」을 같이** 적는다

> *"Each -p must describe ONE distinct direction … and should specify what to change/explore
> **and what to keep the same**."* — `SUPERDESIGN.md` PROMPT RULE

**우리에게 무슨 뜻인가**: 우리 후보 표에는 「무엇 · 기대 첫인상 · 난이도 · 위험」이 있고
**「그대로 두는 것」이 없다.** 그래서 후보끼리 무엇이 겹치는지 표에서 안 보이고, 합칠 때
비로소 드러난다. 칸을 하나 더 만든다.

**앉는 곳**: 후보 표 규격(`02_benchmark` §8.4 · `dir_?.md` 4 절).

### S4. 방향 이름은 **목적**으로 짓는다 — 기제나 형용사로 짓지 않는다

> 예시 방향: *"conversion-focused hero", "editorial storytelling", "dense power-user layout"*
> 그리고: *"Do NOT fill a reproduction prompt with design-system adjectives ('premium',
> 'elegant layered shadows', 'Playfair display') — with any context gap the model will render
> those adjectives as a generic marketing page instead of your real page."*

**우리에게 무슨 뜻인가**: 라운드 ① 후보 이름은 전부 **기제**였다 — 스윕 · 펄스 · 사다리 · 순회.
기제로 이름 지으면 **현행 기제의 변주**가 나온다. 목적으로 지으면(「멀리서 판정을 읽히는 화면」,
「한 사람만 붙잡는 화면」, 「기계임을 증명하는 화면」) 기제가 **결과로** 나온다.
형용사 금지는 같은 규칙의 반대쪽이다 — 「깔끔한」·「모던한」은 VS 의 V3 에서 이미 0 점이다(§11).

**앉는 곳**: 후보 이름 규칙 — `02_benchmark` §8.4.

### S5. **골격 메뉴에서 정확히 하나** — 그리고 라우팅 힌트

`GRAPHIC.md` 는 포스터 골격 다섯(type-hero · full-bleed · split · list · centered-badge)을
**닫힌 메뉴**로 주고 「정확히 하나 고르라」고 한다. 그리고 **라우팅 힌트**를 붙인다 —
「슬로건/인용 → type-hero, 공연/분위기 → full-bleed…」.

**우리에게 무슨 뜻인가**: 이게 「전부 현행 화면의 변주」를 구조적으로 막는 장치다. 후보마다
**골격을 하나 고르게 하고, 한 라운드에 같은 골격을 둘 쓰지 못하게** 하면 변주가 불가능해진다.
우리 골격 메뉴를 **`02_benchmark` §13 에 만들었다** — 우리 어휘(기록계·계측기)로, 아직 안 써 본
골격(접촉 인화 격자 · 세로 드럼 · 중첩 한 축)을 일부러 섞어서.

### S6. `branch` 와 `replace` 를 가른다 — **고른 뒤의 피드백은 새 가지가 아니다**

> *"branch — explore alternatives … replace — creatively refine the selected direction.
> **Selection stops exploratory branching.** … **Never spend a branch on correcting a defect.**"*

**우리에게 무슨 뜻인가**: 우리 `demo/ui/branches/UD-nn-{a,b,c}.html` 규칙에는 이 구분이 없다.
고른 뒤에 들어오는 피드백으로 `UD-nn-d.html` 을 또 만들면 **탐색물이 쌓이고 무엇이 살아 있는지
안 보인다.** 고른 뒤에는 **같은 파일의 다음 커밋**이 판이다 — git 이 버전 이력이다(UD-10 9 행).

**앉는 곳**: `dual-creative-director` 「시안 가지치기」.

### S7. 저쪽(혹은 다음 디렉터)이 볼 수 있는 **가벼운 복제본**을 미리 만든다

> *"The purpose of replica html template is creating a lightweight version of existing UI so
> design agent can iterate on top of it (Since superdesign doesn't have access to your
> codebase directly, this is important context)"* — 패키지 `README.md`

**우리에게 무슨 뜻인가**: `layout_b.html` 은 44.6 KB 자기완결이지만 **`demo_bank.js` 8.65 MB 를
읽고 파형을 `<canvas>` 로 그린다.** 그래서 파일만 넘기면 **파형 자리가 빈 상자**로 보인다.
필요한 것은 **정적 복제본** 하나 — 데이터·애니메이션을 떼고, 파형 자리에 L2 캡처를 넣고,
「이 자리는 실제 파형, 기하 불변」을 주석으로 박은 단일 HTML.

**아직 안 만들었다.** 규격만 적는다(§2). 이것이 CLI 경로와 자료 경로 **양쪽을 싸게** 만든다.

### S8. 디자인 시스템 문서는 **코드베이스 없이도 구현 가능**해야 한다 — 우리는 아직 아니다

> *"Write standalone design system … **Must be implementable without the codebase**"* — `README.md`

**우리에게 무슨 뜻인가**: 시험해 보면 `ui/palette.json` + 브리프만으로는 다크 attract 를 못
그린다 — **방법 색의 `dark` 가 전부 `null`** 이다(UD-11 선행 조건, 아직 안 풀림). 새 사실은
아니고, **이 시험이 그 구멍을 재현 가능하게 만든다**는 것이 값이다.

## 2. 규격 — 정적 복제본 (아직 안 만듦)

```
demo/ui/replica/attract_static.html     단일 HTML · 인라인 CSS · JS 없음
```

- `layout_b.html` 에서 **크롬·레인 상자·띠·푸터의 DOM 과 CSS 만** 남긴다.
- 파형 자리 ↔ `results/screens/` 의 L2 캡처 **한 장**을 `<img>` 로. 주석에
  「이 자리는 실제 `<canvas>` 파형. 기하 · 50 mm/s · 10 mm/mV 불변」.
- `demo_bank.js` 를 **불러오지 않는다.** 수치는 한 장면 것만 하드코딩.
- 토큰은 `demo/ui/tokens.css` 를 인라인으로 복사(빌드 없음).
- **production 이 아니다** — `SCREENS` 에 안 올리고, `layout_b` 를 이것으로 덮지 않는다.

## 3. 안 들이는 것

| | 왜 |
|---|---|
| `.superdesign/init/` 6 파일 규격 | React 라우트·컴포넌트 트리 전제. 우리는 단일 HTML — §2 의 복제본 하나가 그 자리를 다 채운다 |
| `design-system.md` 를 원본으로 | `ui/palette.json` 이 유일 원본(UD-3 · UD-12) |
| `resume.json` 웜 상태 | git 이 이력이다. 파생 상태를 또 만들지 않는다(UD-10 1 행의 그 이유는 **여기선 여전히 맞다**) |
| PRESENTATION · ASSET_GENERATION | 범위 밖(UD-12) |
| 컨텍스트 ~900 줄 트리밍 규칙 | 우리 파일이 그 아래다. 필요해지면 그때 |

## 4. CLI 를 붙이려면 — 경로만 (실행은 로컬)

계정은 사용자가 만들었고 무료 토큰도 받았다 `[대화]`. **이 세션에서는 못 붙인다** —
`api.superdesign.dev` 가 `403 to CONNECT`(UF-8 추기). 아래는 **로컬 세션용 경로**다.

**측정한 것** `[코드]` — `npm pack` 으로 연 0.14.0 의 `dist/` 에서 확인한 환경변수:

```
SUPERDESIGN_TOKEN          API 토큰 직접 주입. 디스크 설정을 이긴다.
                           저장 안 되고 텔레메트리로도 안 나간다 → 에이전트/CI 용
SUPERDESIGN_CONFIG_DIR     설정 디렉터리 이동 (기본 $XDG_CONFIG_HOME/superdesign 또는 ~/.superdesign)
SUPERDESIGN_API_URL        API 주소 바꾸기
SUPERDESIGN_TELEMETRY_DISABLED / DO_NOT_TRACK    텔레메트리 끄기
SUPERDESIGN_DEBUG          디버그
```

`extract-website` 와 `extract-brand-guide` **둘 다 0.14.0 에 있다** `[측정]`
(`dist/` 에서 각각 5 · 2 파일에서 발견). 그런데 **`extract-brand-guide` 는 쓰지 않는다** —
`SUPERDESIGN.md` COMMAND CONTRACT 가 명시한다 `[문헌]`:

> *"`extract-website` … **supersedes `extract-brand-guide`, which the command list still shows —
> do not use that one.**"*

패키지 README 가 후자를 「Inspiration & Style Tools」에 올려 둔 것은 **낡은 줄**이다.

**`extract-website` 의 함정** `[문헌]` — 같은 절에서:

| | |
|---|---|
| **시간** | **서버 쪽 크롤 60~120 s / 한 사이트.** URL 열 개면 10~20 분이다 |
| 기본값 | 선택자 없이 부르면 **`--design-md` 로 동작** |
| `--all` 의 함정 | 모든 payload 를 가져오지만 **clone HTML 을 안 쓰고 brand 바이너리를 안 받는다** → `--clone` · `--brand-assets` 를 **따로** 줘야 한다 |
| 포함 관계 | `--brand-assets` 는 `--brand` 를 포함한다 |
| **이미지·GIF 직링크** | `extract-website` 가 아니다. 안전하게 내려받아 **`upload-asset --purpose reference`** 로 올린다(`WEBSITE.md` URL ROUTING) |
| **PDF** | **안 된다.** 지원 보류 — 필요한 쪽을 이미지로 달라고 한다 |
| 로그인 뒤 페이지 | 안 된다. 크롤이 **공개 HTTP(S)** 안에서만 리디렉트를 따른다 |

**붙이는 순서** (로컬, 저장소 루트에서):

```bash
export DO_NOT_TRACK=1
npx -y @superdesign/cli@latest auth status --json     # 항상 exit 0 — 게이트가 아니라 질의
npx -y @superdesign/cli@latest login --no-browser --json   # URL + 코드를 찍고 폴링
npx -y @superdesign/cli@latest auth status --json     # authenticated · team · configPath 확인
npx -y @superdesign/cli@latest list-models            # 모델 카탈로그 + 스타일 메모
```

**「레퍼런스 발산만」 하고 멈추는 최소 경로** — 크레딧을 안 쓴다:

```bash
npx -y @superdesign/cli@latest search-prompts --tags "style" --json
npx -y @superdesign/cli@latest get-prompts --slugs "<고른 slug>" --full
npx -y @superdesign/cli@latest extract-website --url "<사이트>" --design-md --brand-assets
#   → .superdesign/website/<domain>/design.md · 스크린샷
```

여기까지가 **발산의 먹이**이고, 이것만으로 `dir_B.md` 를 쓸 수 있다.
그 다음(시안 생성)은 `create-project --template` → `iterate-design-draft --mode branch` 인데,
**여기서 플랜 소모가 시작될 수 있다** — 우리가 확정한 것은 **`confirm-generation`(이미지·영상)만
크레딧을 쓴다**는 것뿐이고, 드래프트 생성의 과금은 **못 쟀다**(`superdesign.dev` 403).

**그래서 순서 제안**: 로컬에서 위 네 줄(`auth status` → `login` → `auth status` → `list-models`)만
먼저 돌리고 **그 출력을 그대로 붙여 달라.** `auth status --json` 의 `team`, `list-models` 의
플랜/한도 표기로 「우리 프로젝트에 쓸 만한 양인가」를 그때 판정한다.

**안 하는 것**: `confirm-generation` 호출 · `data/arduino/` 업로드 · `demo_bank.js`(8.65 MB) 업로드.

## 5. 덤 — `playground` 플러그인이 우리 구조와 정확히 맞는다 `[측정]`

공식 마켓플레이스에 있고(아래 UF-11), 스킬 원문이 요구하는 것이 이렇다:

> *"Single HTML file. Inline all CSS and JS. No external dependencies. … Live preview.
> Updates instantly on every control change. … Keep a single state object. Every control
> writes to it, every render reads from it."*

**이게 `layout_b` 의 구조 그대로다** — 무빌드 단일 HTML · `state` 객체 · `render()`.
그리고 **네트워크가 필요 없다** — 403 과 무관하게 오늘 쓴다.

쓸 자리: 3 m 가독(§10)과 레인 기하를 **사용자가 직접 밀어 보는** 탐색기.
레인 높이 · 게인 · 스윕 속도 · 글자 크기를 슬라이더로 놓고 live preview 를 두면,
「15.6″ 에서 69 px」 같은 표가 **손으로 확인되는 값**이 된다. 지금은 표로만 있다.

**아직 결정 아님.** UD-13 「열린 결정」에 올린다.
