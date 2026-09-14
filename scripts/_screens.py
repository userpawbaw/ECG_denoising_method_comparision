"""시연 화면을 **브라우저로 열어 보는** 공통 모듈.

`docs/ui/01_system.md` 6.2 — 이 저장소의 결함 대부분이 「화면을 열면 5 초에
보이는 것」이었는데 여는 장치가 없었다. `docs/33` 이 겪은 「문서가 코드보다
앞섰다」도, UO-1 도 같은 뿌리다.

`scripts/shoot_screens.py`(찍기)와 `tests/test_demo_screens.py`(검사)가 함께 쓴다.

## 브라우저 찾기

Playwright 가 번들과 다른 빌드 번호를 기대할 수 있으므로 **직접 지정**한다.
`PLAYWRIGHT_BROWSERS_PATH` 아래를 먼저 보고, 없으면 시스템에 깔린 것을 쓴다.
아무것도 없으면 `None` 이고, 검사는 **실패가 아니라 skip** 이다 — 브라우저가
없는 환경에서 커밋을 막을 이유가 없다.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"

# 검사 폭 — `ui/palette.json` 의 viewport.test_widths 와 같아야 한다.
# 확정: 노트북 1920×1080, 최소 방어폭 1280×720 (docs/ui/01_system.md 7 절).
WIDTHS = (1280, 1440, 1920)
HEIGHT = 900


class Screen:
    """검사 대상 화면 하나.

    `needs` 는 **그 화면이 실제로 그릴 수 있으려면 있어야 하는 파일**이다.
    은행(`demo_bank.js` 8.6 MB)은 생성물이라 없을 수 있고, 그때는 화면이
    비어 보이는 것이 정상이다 — 검사를 실패시키지 않고 건너뛴다.
    """

    def __init__(self, path: str, label: str, *, needs: tuple[str, ...] = (),
                 expect_canvas: bool = True, animated: bool = False,
                 allow_console: tuple[str, ...] = (), note: str = ""):
        self.path = DEMO / path
        self.label = label
        self.needs = tuple(DEMO / n for n in needs)
        self.expect_canvas = expect_canvas
        self.animated = animated
        # **예상된** 콘솔 오류만 좁혀서 허용한다. 통째로 무시하면 진짜 오류를 놓친다.
        self.allow_console = allow_console
        self.note = note

    @property
    def name(self) -> str:
        return self.path.relative_to(DEMO).as_posix()

    @property
    def url(self) -> str:
        return self.path.as_uri()

    def ready(self) -> bool:
        return self.path.exists() and all(n.exists() for n in self.needs)

    def __repr__(self) -> str:
        return f"<Screen {self.name}>"


SCREENS = [
    Screen("index.html", "모드 B — 방법 비교 (본 화면)",
           needs=("demo_bank.js",), animated=True),
    Screen("live.html", "모드 A — 실측 (연결 끊김 상태)",
           expect_canvas=False, animated=True,
           # `file://` 로 열면 SSE 엔드포인트(`/stream`)가 파일로 해석돼 못 찾는다.
           # **그것이 이 화면의 정상 상태**다 — 서버 없이 열면 끊김이다.
           allow_console=("net::ERR_FILE_NOT_FOUND",),
           note="SSE 서버 없이 열면 «끊김» 이 기본 상태다. D-18 이 «실제로 일어난다» 고 "
                "적은 그 화면이라, 이 상태로 검사하는 것이 맞다"),
    Screen("mockup_expo.html", "박람회 시안",
           needs=("demo_bank.js",), animated=True),
    Screen("cards/A_monitor.html", "지표 카드 A — 임상 모니터",
           needs=("card_bank.js",), animated=True),
    Screen("cards/B_notebook.html", "지표 카드 B — 실험 노트",
           needs=("card_bank.js",)),
    Screen("cards/C_cardnews.html", "지표 카드 C — 카드뉴스",
           needs=("card_bank.js",)),
    Screen("ui/layout_b.html", "3 단계 레이아웃 시안 — 경로 B (직접 구현)",
           needs=("demo_bank.js",),
           note="0 단계가 넘긴 일곱을 한 화면에 넣은 시안. 경로 A(Figma)와 짝지어 "
                "비교한다 — docs/ui/01_system.md 6.3"),
]


def find_chromium() -> str | None:
    """쓸 수 있는 Chromium 실행 파일. 없으면 `None`."""
    base = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if base:
        # chromium-<build>/chrome-linux/chrome · headless_shell 도 본다
        for pat in ("chromium-*/chrome-linux/chrome",
                    "chromium-*/chrome-linux/headless_shell",
                    "chromium_headless_shell-*/chrome-linux/headless_shell"):
            found = sorted(Path(base).glob(pat))
            if found:
                return str(found[-1])
    for exe in ("chromium", "chromium-browser", "google-chrome", "chrome"):
        p = shutil.which(exe)
        if p:
            return p
    return None


def launch(pw, *, exe: str | None = None):
    """`sync_playwright()` 컨텍스트에서 브라우저를 띄운다."""
    return pw.chromium.launch(executable_path=exe or find_chromium(),
                              args=["--no-sandbox", "--disable-gpu",
                                    "--allow-file-access-from-files"])


def new_page(browser, width: int = WIDTHS[-1], height: int = HEIGHT,
             *, theme: str = "light", reduced_motion: str = "no-preference"):
    """**테마와 모션 선호를 실제로 켜서** 연다.

    다크를 「자동 반전이니 괜찮겠지」로 넘기면 안 되고(`docs/33`), 모션도
    `prefers-reduced-motion` 을 켠 상태가 따로 있다(U-14).
    """
    return browser.new_page(viewport={"width": width, "height": height},
                            color_scheme=theme, reduced_motion=reduced_motion,
                            device_scale_factor=1)


def settle(page, *, animated: bool = False, timeout: int = 45_000) -> list[str]:
    """페이지가 자리를 잡을 때까지 기다리고 **콘솔 오류를 모아 돌려준다**."""
    errors: list[str] = []
    page.on("console", lambda m: errors.append(f"{m.type}: {m.text}")
            if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    page.wait_for_load_state("load", timeout=timeout)
    # 은행이 8.6 MB 라 파싱과 첫 렌더에 시간이 걸린다. 애니메이션 화면은
    # 한 프레임 이상 돌아야 캔버스에 무엇이 그려진다.
    page.wait_for_timeout(2500 if animated else 1200)
    return errors
