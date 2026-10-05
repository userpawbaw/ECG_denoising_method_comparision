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

## 5. 합칠 때 부딪히는 것 — `demo/live.html` 을 건드리는 다른 브랜치

| 원격 브랜치 | 이 작업 브랜치에 없는 커밋 | `live.html` | 합칠 때 |
|---|---|---|---|
| `claude/wonderful-gates-0enr3f` | 74 | `frame()` 에 `prefers-reduced-motion` **6 줄** | D-41 도 reduced-motion 을 다룬다(선단 밝기 초기값). 두 처리가 **같은 일을 두 번** 하지 않게 하나로 합친다 |
| `claude/demo_UIUX_refactoring` | 22 | 같은 6 줄 | 위와 같다 |
| `waveform-generate-only` | 60 | **610 줄을 새로 가진다** | 이력이 다른 판일 수 있다 — 합치기 전에 그 파일을 먼저 읽는다 |

`99_status.md` §7.7 이 앞 둘을 이미 적어 두었다. 그때의 판단(«`frame()` 을 안 건드렸으니 그대로 들어간다»)은
**D-41 뒤로 더는 맞지 않는다** — D-41 은 `frame()` 이 아니라 스위치 초기값에서 reduced-motion 을 읽지만,
그리기 함수(`strokeBuf` · `drawSplit` · `drawOverlay`)를 크게 바꿨다.

---

## 6. 외부 참조 — `userpawbaw/ecg-gui-design-review` (전시용 GUI, 별도 저장소)

이 세션에서는 **읽기만** 된다(공개 저장소 · 익명 clone). 쓰려면 push 권한으로 다시 붙여야 한다.

| 무엇 | 경로 |
|---|---|
| v2.2.1 그리기 | `prototype/v2/src/Plot.tsx` (판 · 격자 · 색 · 눈금) · `src/engine.ts` (`visiblePoints` — 지우기 경계 gap 0.12 s · fade 0.08 s) |
| 화면 스타일 | `prototype/v2/src/style.css` (스위치 · 버튼 · 패널 색) |
| 선단 밝기 피드백 | 그 저장소 `docs/22` «UI 마감·사용성 개선 논의» §4 (UI-02) · `docs/uiux_system/03_MOTION_AND_POLISH.md` §2 |
| fade 계약 | 그 저장소 `docs/21` «UI 수정 워크플로우(최종)» 9 번 — «Sweep 페이드는 지우기 경계에만» (선단 밝기는 fade 가 아니라 밝기다) |

**v2.2.1 앱 자체에 이 실시간 화면을 합치는 것은 하지 않았다.** 그 저장소의 release ZIP 은 대용량 자료(98 조건 × 600 s)
와 함께 묶이고 Git 에 없다. 그 통합을 하려면:

- 그 ZIP 과 Node 24 가 있는 **사용자 PC 의 로컬 세션**이 유리하다(원격에는 자료가 없다).
- 이 저장소의 SSE 형식(`/stream` 의 `{"i","n","fs","raw","ok","out","stat"}` — F-52)을 그 앱의 `Transport` 에 물리는
  어댑터가 필요하다. 그 앱은 저장 출력 재생기라 «도착한 만큼 진행» 하는 시간축(6.2 (1))이 없다.
- 그 저장소의 규약(`AGENTS.md` · `WORK_RESUME_POLICY.md` · `docs/uiux_system`)을 먼저 따른다.

---

## 7. 새 세션에 줄 첫 메시지 (복사해서 쓴다)

> `docs/98_realtime_handoff.md` 를 읽고 4 절의 1 번부터 이어서 한다. 작업 브랜치는
> `claude/adoring-cray-nv6pvq`. 실보드는 사용자 Windows PC(`COM13`, `--ascii --board-fs 250`)다.
> 화면을 고치면 CLAUDE.md 「화면 고치기」 규칙대로 `scripts/screenshot_live.py` 로 띄워 보고 PNG 를 눈으로 본다.
