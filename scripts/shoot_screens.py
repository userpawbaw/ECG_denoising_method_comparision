#!/usr/bin/env python3
"""시연 화면을 **찍어서 눈으로 볼 수 있게** 한다.

`docs/ui/01_system.md` 6.2 — 검사(`tests/test_demo_screens.py`)는 잴 수 있는 것만
잰다. 여백 리듬 · 타이포 위계 · 「이게 예쁜가」는 **사람이 봐야 한다.** 그런데
화면 여섯을 세 폭과 두 테마로 열어 보는 일은 손으로 하면 안 하게 된다.

UO-1 이 그 대가를 보여줬다 — 「눈으로 본다」는 규칙이 CLAUDE.md 에 있었는데도
열어 보지 않고 저장해 tracked 산출물을 깨진 것으로 덮었다.

## 쓰기

    python3 scripts/shoot_screens.py                  # 기본 폭·라이트
    python3 scripts/shoot_screens.py --all            # 세 폭 × 두 테마
    python3 scripts/shoot_screens.py --only index.html --width 1280
    python3 scripts/shoot_screens.py --full-page      # 스크롤 전체

찍은 것은 `results/screens/` 에 쌓이고 **git 에 넣지 않는다** — 화면이 바뀔 때마다
갱신되는 이미지라 커밋 잡음만 쌓인다(`results/logs/` 와 같은 이유). 보고 나서
버린다. 남길 것이 있으면 문서에 근거로 인용한다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _screens import (  # noqa: E402
    ROOT, SCREENS, WIDTHS, HEIGHT, find_chromium, launch, new_page, settle,
)

OUT = ROOT / "results" / "screens"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="시연 화면 스크린샷 (docs/ui 6.2)")
    ap.add_argument("--only", nargs="*", help="화면 이름 일부 (기본: 전부)")
    ap.add_argument("--width", type=int, action="append",
                    help="폭 (여러 번 줄 수 있다). 기본 1920")
    ap.add_argument("--theme", choices=("light", "dark", "both"), default="light",
                    help="어느 테마로 찍을지. both 는 화면당 두 장 — 다크는 자동 반전이 아니라 따로 고른 값이라 둘 다 봐야 한다 (UF-5)")
    ap.add_argument("--all", action="store_true", help="세 폭 × 두 테마")
    ap.add_argument("--full-page", action="store_true", help="스크롤 전체를 찍는다")
    a = ap.parse_args(argv)

    exe = find_chromium()
    if not exe:
        print("Chromium 을 못 찾았다. PLAYWRIGHT_BROWSERS_PATH 를 보거나 "
              "chromium 을 설치할 것", file=sys.stderr)
        return 2
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright 가 없다: pip install playwright", file=sys.stderr)
        return 2

    widths = WIDTHS if a.all else (a.width or [WIDTHS[-1]])
    themes = ("light", "dark") if (a.all or a.theme == "both") else (a.theme,)
    targets = [s for s in SCREENS
               if not a.only or any(k in s.name for k in a.only)]

    OUT.mkdir(parents=True, exist_ok=True)
    made = 0
    with sync_playwright() as pw:
        browser = launch(pw, exe=exe)
        for s in targets:
            if not s.ready():
                missing = [n.name for n in s.needs if not n.exists()]
                print(f"  건너뜀 {s.name} — 없는 파일: {', '.join(missing) or s.path.name}")
                continue
            for w in widths:
                for theme in themes:
                    page = new_page(browser, width=w, height=HEIGHT, theme=theme)
                    page.goto(s.url)
                    errors = settle(page, animated=s.animated)
                    unexpected = [e for e in errors
                                  if not any(p in e for p in s.allow_console)]
                    f = OUT / f"{s.stem}__{w}__{theme}.png"
                    page.screenshot(path=str(f), full_page=a.full_page)
                    page.close()
                    made += 1
                    flag = f"  ⚠ 콘솔 오류 {len(unexpected)}" if unexpected else ""
                    print(f"  {f.relative_to(ROOT)}{flag}")
        browser.close()

    print(f"\n{made} 장 — {OUT.relative_to(ROOT)}")
    print("**열어서 볼 것.** 검사는 잴 수 있는 것만 잰다 (UO-1).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
