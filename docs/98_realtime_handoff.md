# 98. 실시간 시연(모드 A) 인수인계 — 실보드 투입 이후 (2026-10-02 ~ 10-05)

> **이 문서 하나로 새 세션이 바로 이어 간다.** 원격(클라우드)이든 사용자 PC 의 로컬 세션이든 같다.
> 결론과 근거의 **원문은 F · D · O 기록**에 있고, 여기는 지도와 «다음에 무엇을» 이다.
> 이 절의 작업이 다 끝나면 이 문서는 지우지 말고 **«끝남»** 으로 표시한다 (D-29 — 경과도 산출물이다).

---

## 1. 지금 어디에 있나

| | |
|---|---|
| 작업 브랜치 | `claude/adoring-cray-nv6pvq` — 원격에 푸시돼 있다 |
| 기본 브랜치 | `claude/ecg-denoising-dsp-dl-comparison-b5wjvj` — 작업 브랜치는 이것보다 **앞서기만** 한다(뒤처진 커밋 0). **아직 합치지 않았고 PR 도 없다** |
| 검사 | `pytest tests/` 전부 통과 — 709 passed · 11 skipped (2026-10-05) |
| 사용자 PC | Windows · VS Code · Python 3.13(Microsoft Store 판) · Arduino Uno + AD8232 · `COM13` · `--ascii --board-fs 250` |

**새 세션의 첫 동작**

```sh
git fetch origin claude/adoring-cray-nv6pvq
git checkout claude/adoring-cray-nv6pvq
pip install -r requirements.txt          # 화면 확인까지 하려면 + pip install playwright
python3 -m pytest tests/ -q              # 약 1.5 분
python3 scripts/screenshot_live.py --methods M_FE,M04,M05S   # 화면을 띄워 찍는다 (§8)
```

---

## 2. 무엇을 했나 — 기록 지도 (시간순)

| 번호 | 한 줄 | 상태 |
|---|---|---|
| **F-55** | 실보드 화면이 1.5 s 마다만 바뀌었다 — Windows 에서 `read(4096)` 가 4096 B 를 다 채울 때까지 막혔다 (사용자 PC `--read-timing` 1640 ms · 4096 B 로 확정) | 해결 |
| **D-38** | 읽기는 `in_waiting` 만큼 · 큐 상한 2000 · fs 점검은 도착 시각 이동 창 · 「도착 간격」 표시 · `--trace-io` | 유효 |
| **F-56** | 덩어리 크기가 파형 값을 바꿨다 — 블록 영위상 FE 버퍼 자르기 결함(2000 샘플에서 죽음) 수정, `StreamProcessor` 는 브리지가 hop 씩 나눠 넣는다 | 해결 |
| **O-37** | 09-13 이후 M06 · M08 체크포인트가 전부 안 올라갔다 — 키 이름 변경, 적재 전 훅으로 복구 | 해결 |
| **F-57** | 실시간 RTF 가 hop 이 크면 부풀었다(분모 버그) · M05 가 화면을 멈춘 것은 계산량(창당 0.17~0.34 s) | 해결 |
| **D-39** | `--hop-for` 방법별 hop · 홀짝 병렬 칼만 기각(근거 포함) | 유효 |
| **D-40** | **M05S** — 스트리밍 Sameni EKF. 미래 문맥 L 36 · RTS d 12 · hop 12(지연 240 ms). 오프라인 14.46 dB 대 M05 16.84 dB. M05 · M05f 는 브리지가 **자동으로 hop 128** | 유효 |
| **O-38** | M05S 적합(wfdb import · XQRS · least_squares)이 본체 스레드를 막아 시작 직후 멎었다 — 작업 스레드로 | 해결 · **실보드 재확인 대기** |
| **D-41** | 모드 A 화면을 ECG Signal Studio **v2.2.1** 표시 규약으로 — 어두운 판 · 방법별 색 · 지우기 경계 fade · **선단 밝기(잔광)** · reduced-motion | 유효 · **실보드 확인 대기** |
| **D-42** | 표시 길이(2.5/5/10 s) · 표시 범위 ±mV(0 = 자동) · **표시 보간 두 배**(블록을 반씩 두 번) · 지우기 영역 화면의 10 %(디버그 5~20 %) · 스크롤 양 끝 경계 | 유효 · **실보드 확인 대기** |

같이 생긴 도구: `scripts/probe_serial.py --read-timing` (F-55) · `scripts/serial_bridge.py --trace-io FILE` (D-38) ·
`--hop-for M=N` (D-39) · `scripts/screenshot_live.py` (D-41, 화면 확인 자동화). 옵션 전부는 `24_cli_reference.md`.

---

## 3. 실보드에서 돌리는 명령 (사용자 PC)

```bat
:: 기본 — 고전 + 스트리밍 칼만
python scripts\serial_bridge.py --port COM13 --ascii --board-fs 250 --methods M_FE,M04,M05S --serve
::   -> 브라우저(Chrome/Edge 권장)에서 http://127.0.0.1:8765/live.html

:: 창 방식 칼만 — hop 128 이 자동이다(지연 560 ms)
python scripts\serial_bridge.py --port COM13 --ascii --board-fs 250 --methods M_FE,M04,M05 --serve

:: 멈춤 · 끊김을 되짚을 기록을 남길 때
python scripts\serial_bridge.py ... --trace-io trace.txt

:: 포트 점검 (브리지를 닫고)
python scripts\probe_serial.py --port COM13 --read-timing
```

읽을 숫자 — 진행 줄과 화면 상태 칸의 **도착 간격**(정상 수십 ms), **RTF**(0.2 안팎), **처리 밀림**(0).

---

## 4. 남은 일 — 우선순위 순

| # | 무엇 | 왜 · 어디 | 막힌 것 |
|---|---|---|---|
| 1 | **M05S 실보드 1~2 분 재확인** — 시작 20 s 안 멎음이 사라졌나 | O-38. 안 사라지면 `--trace-io` 파일을 받는다 | 사용자 장비 문제로 보류 중 |
| 2 | **새 화면(D-41) 실보드 확인** — 선단 밝기가 P · T 파를 가리는가, 프레임이 버벅이는가 | D-41 「되돌려야 하는 조건」. 가리면 `GLOW_MAX` 를 0.25 로 | 1 과 같이 |
| 3 | **Windows torch DLL 오류**(WinError 1114, `c10.dll`) — 딥러닝(M06 · M08 · M09)을 실보드에 못 올린다 | 코드 문제 아님. VC++ 재배포 패키지 → CPU 판 torch 재설치 → python.org 판 Python 순으로 | 사용자 PC |
| 4 | **작업 브랜치를 기본 브랜치에 합친다** — 아래 5 절의 충돌 위험부터 본다 | 6 커밋 앞섬 | 사용자 결정 |
| 5 | 넷째 겹치기 색 `#d7a6ff` 의 색각 대비 검사 | D-41 표 — 다시 안 돌렸다 | 없음 |
| 6 | M07 · M10 을 브리지에 잇기 | 체크포인트는 있다. 사용자가 «없어도 된다» 고 했다 | 보류 |
| 7 | 화면 지터 버퍼(R3) | D-38 에서 보류. 지금 20 FPS 로 충분하다는 판단 | 보류 |

---

## 5. 합칠 때 부딪히는 것 — 다른 브랜치의 `demo/live.html` (2026-10-05 분석)

| 원격 브랜치 | 이 작업 브랜치에 없는 커밋 | `live.html` 에서 바꾼 것 | 처리 |
|---|---|---|---|
| `claude/wonderful-gates-0enr3f` | 74 (전체 177 파일) | `frame()` 의 **6 줄** — reduced-motion 이면 스윕을 매 프레임 다시 그리지 않고 데이터가 올 때만 그린다(커서가 쉬지 않고 도는 것을 멈춘다). 커밋 `69311e8` | **가져왔다** — D-41 의 선단 밝기 초기값과 같은 `REDUCED` 상수 하나로 묶었다 |
| `claude/demo_UIUX_refactoring` | 22 | 위와 **같은 6 줄** | `wonderful-gates` 의 부분집합이다(고유 커밋 0) — 따로 볼 것이 없다 |
| `waveform-generate-only` | — | **`live.html` 이 없다** | 이 저장소와 **공통 조상이 없는** 8 월의 옛 이력이다. ~~«610 줄을 새로 가진다»~~ 는 공통 조상을 못 찾아 빈 값으로 비교한 **잘못된 측정**이었다 (지우지 않고 남긴다, D-29) |

**`wonderful-gates` 를 통째로 합치면 안 되는 이유** — 별도 작업 트리에서 시험 병합해 봤다 `[측정]`:

- 충돌은 넷뿐이다(`.gitignore` · `CLAUDE.md` · `24_cli_reference.md` · `91_report.md`). 그런데
- **학습 결과를 덮는다** — `results/d0/m06_l1/best.pt` 등 기존 체크포인트가 그 브랜치의 재학습판으로 바뀐다(수정 ·
  추가 89 개). 보고서 수치의 근거가 바뀌므로 CLAUDE.md 「산출물 지우기」 와 §5 를 거쳐야 한다.
- **기록 번호가 겹친다** — 합치면 `F-47` 이 둘이 된다. 그 브랜치는 기록 규약 D 번호도 옮겨 적었다(`D-29` → `D-32`).
- 그 브랜치에는 «UI/UX 와 연구는 세션을 나눠 쓴다» 는 자체 규칙이 있다.

그래서 통합은 **사용자 결정 + 전용 세션**의 일이다. 할 때는 체크포인트를 어느 쪽으로 둘지부터 정한다.

## 6. 외부 참조 — `userpawbaw/ecg-gui-design-review` (전시용 GUI, 별도 저장소)

이 세션에서는 **읽기만** 된다(공개 저장소 · 익명 clone). 쓰려면 push 권한으로 다시 붙여야 한다.

| 무엇 | 경로 |
|---|---|
| v2.2.1 그리기 | `prototype/v2/src/Plot.tsx` (판 · 격자 · 색 · 눈금) · `src/engine.ts` (`visiblePoints` — 지우기 경계 gap 0.12 s · fade 0.08 s) |
| 화면 스타일 | `prototype/v2/src/style.css` (스위치 · 버튼 · 패널 색) |
| 선단 밝기 피드백 | 그 저장소 `docs/22` «UI 마감·사용성 개선 논의» §4 (UI-02) · `docs/uiux_system/03_MOTION_AND_POLISH.md` §2 |
| fade 계약 | 그 저장소 `docs/21` «UI 수정 워크플로우(최종)» 9 번 — «Sweep 페이드는 지우기 경계에만» (선단 밝기는 fade 가 아니라 밝기다) |

### v2.2.1 에 실시간 화면을 합치려면 — 무엇이 필요한가

v2.2.1 에는 **실시간 데이터를 받는 탭이 없다.** 실험실 화면은 저장된 배열을 `Transport`(벽시계로 시간이 가고 끝이
있는 재생기)로 도는 구조이고(`data.ts` 의 `Loaded` 는 길이 `n` 이 정해진 배열과 Reference `clean` 을 요구한다),
「계측」 은 옛 화면을 iframe 으로 띄운 **「실제 장치 미연결 · UI 상태 미리보기」** 다. 길은 둘이다.

| | A. 탭 하나 + iframe (빠른 길) | B. React 로 직접 (제대로 된 길) |
|---|---|---|
| 하는 일 | `main.tsx` 의 경로에 `live` 탭을 더하고 `http://127.0.0.1:8765/live.html` 을 iframe 으로 띄운다 — 「상세 분석·계측」 이 이미 쓰는 방식 | SSE `/stream` 을 받는 `LiveSource` 와 링버퍼, 그 위에서 그리는 `Plot` 변형을 만든다 |
| 필요한 것 | 브리지를 따로 띄워 둔다 · 화면 모양은 D-41 로 이미 맞췄다 · 출처 표시를 «실측 · 장치 연결됨» 으로 바꾼다 | 아래 표 전부 |
| 품 | 작다 | 크다 |

B 에 필요한 것:

| 무엇 | 왜 |
|---|---|
| **시간축** — `time = head / fs` 인 실시간 전송기 | `Transport.tick` 은 벽시계로 간다. 실시간은 «도착한 만큼» 간다 (30 문서 6.2 (1)). 끝(duration)도 seek 도 없다 |
| **Reference · 차이 · 지표를 끈다** | 실측에는 참값이 없다 — SNR 을 띄우면 지어낸 숫자다 (30 문서 난관 6). v2.2.1 의 `metric` · 차이 보기 · Reference 겹침은 모두 `clean` 에 기댄다 |
| **방법 목록이 고정이다** | 실시간 출력은 브리지를 띄울 때 `--methods` 로 정해진다. 실험실의 hover 미리보기 · Pin 처럼 아무 방법이나 고를 수 없다 |
| **전환 · 끊김 메시지** | front-end 전환(`POST /fe` · `{"reset":true}`), 선 끊김(`link_lost`), 상태(손실 · lead-off · 처리 밀림 · RTF · 도착 간격) |
| **같은 출처(origin)** | 브리지가 8765 에서 `/stream` 을 낸다. Vite 개발 서버의 proxy 로 묶거나, 브리지가 v2 빌드(`dist`)를 서빙하거나, 브리지에 CORS 헤더를 단다 |
| **그리기** | `visiblePoints` 는 저장 배열의 시각 → 칸 매핑이다. 실시간은 절대 인덱스의 나머지로 쓰는 링버퍼다(F-52) — `live_axis.js` 의 `absAt` · `have` · 이득 규칙(D-36)을 옮긴다 |
| **그 저장소의 규약** | `AGENTS.md` · `WORK_RESUME_POLICY.md` · `docs/uiux_system`(CASE 연결 · 기록 무결성 CI). «장치 미연결 · deviceSession null» 이 그 앱의 표시 계약이라 그것을 바꾸는 결정 기록이 먼저다 |
| **쓰기 권한** | 이 세션에서는 그 저장소가 읽기만 된다 — push 권한으로 다시 붙여야 한다 |
| **릴리스** | release ZIP 은 98 조건 × 600 s 자료와 함께 묶이고 Git 에 없다. 그 자료와 Node 24 · 브라우저 검수 환경이 있는 **사용자 PC 의 로컬 세션**이 유리하다 |

**권하는 순서** — A 로 먼저 한 앱 안에서 보이게 하고, 전시 동선이 정해진 뒤 B 로 옮긴다.

**A 는 2026-10-05 에 했다** — 그 저장소 브랜치 `claude/live-tab-iframe` (D-016, 미병합). 네 번째 탭 «실시간 측정» 이
`<브리지>/live.html?embed=1` 을 iframe 으로 띄우고, 이 저장소의 `live.html` 은 `?embed` 이면 자기 머리글을 숨긴다.
브리지가 없으면 실행 명령을 보이고 5 s 마다 다시 찾는다. 무인 운영(Attract 자동 전환)은 실험실에서만 돈다.
쓰는 법: 브리지를 띄우고(`--serve`, 기본 `127.0.0.1:8765`), v2 앱(`prototype/v2` 에서 `npm run dev`)의 «실시간 측정» 탭.
다른 주소면 탭의 입력칸이나 앱 주소의 `?bridge=호스트:포트`. 실보드 · 전시 PC 는 아직 안 봤다.

---

## 7. 새 세션에 줄 첫 메시지 (복사해서 쓴다)

> `docs/98_realtime_handoff.md` 를 읽고 4 절의 1 번부터 이어서 한다. 작업 브랜치는
> `claude/adoring-cray-nv6pvq`. 실보드는 사용자 Windows PC(`COM13`, `--ascii --board-fs 250`)다.
> 화면을 고치면 CLAUDE.md 「화면 고치기」 규칙대로 `scripts/screenshot_live.py` 로 띄워 보고 PNG 를 눈으로 본다.
