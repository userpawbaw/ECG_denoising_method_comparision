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
| `docs/ui/02_benchmark.md` | **레퍼런스 분석** — 표시 규약 · 비교 UI · 레이아웃 · 타이포 · 반응형, 그리고 **빌리지 않기로 한 것** |

네 기록 파일이 모두 있다.

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
| 3 | 레이아웃 재설계 (경로 둘로 짝지어 비교) | 대기 |
| 4 | 실패 상태 화면 · 색각 이상 검증 | 대기 |

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
