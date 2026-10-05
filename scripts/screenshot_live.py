#!/usr/bin/env python3
"""실시간 화면(모드 A)을 **하드웨어 없이 띄워 찍는다** — 체크리스트 §8 의 「띄워 봤나」 를 한 번에.

    python3 scripts/screenshot_live.py                       # replay synth · M_FE,M01,M04
    python3 scripts/screenshot_live.py --methods M_FE,M04,M05S --out /tmp/shots

하는 일 (D-41 화면 작업에서 손으로 하던 것을 그대로 옮겼다):

1. 브리지를 `--replay synth --serve` 로 띄운다 (Popen 객체로 끝낸다 — 이름으로 찾지 않는다, O-15).
2. 헤드리스 Chromium 으로 `live.html` 을 열어 **분리·스크롤 → 분리·스윕 → 커서 근처 확대 →
   겹치기·스윕 → 범례 칩 끄고 켜기 → front-end 전환** 을 차례로 찍는다.
3. **스트림 중간에 새 창**을 reduced-motion 으로 열어 찍는다 (F-52 · D-41).
4. 페이지 오류 · 콘솔 오류 · 404 를 모아 출력한다. 하나라도 있으면 종료 코드 1.

찍은 PNG 는 **반드시 눈으로 본다** — 이 스크립트는 화면이 «떴다» 까지만 확인한다.
Playwright(파이썬)와 Chromium 이 있어야 한다. 원격 세션에는 `/opt/pw-browsers` 에 있다.
"""
import _bootstrap  # noqa: F401

import argparse
import glob
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def find_chromium() -> str | None:
    """원격 세션의 미리 깔린 Chromium. 없으면 None — Playwright 기본값을 쓴다."""
    hits = sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome"))
    return hits[-1] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser(description="모드 A 화면을 replay 로 띄워 찍는다 (§8)")
    ap.add_argument("--methods", default="M_FE,M01,M04",
                    help="브리지에 줄 방법 목록. 고를 수 있는 것은 docs/30_realtime_demo.md 의 "
                         "--methods 절 (예: M_FE,M04,M05S)")
    ap.add_argument("--out", default=str(ROOT / "results" / "screens"),
                    help="PNG 를 둘 폴더. 없으면 만든다 (git 에는 안 들어간다 — .gitignore)")
    ap.add_argument("--http-port", type=int, default=8899,
                    help="브리지 SSE 서버 포트. 다른 브리지가 쓰고 있으면 바꾼다")
    ap.add_argument("--warm", type=float, default=9.0,
                    help="첫 장을 찍기 전 기다리는 시간 [s]. warm-up(4.1 s) + 정렬 여유")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright 가 없다:  pip install playwright   (브라우저는 따로 — 원격 세션에는 이미 있다)")
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "bridge.log", "w")
    br = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "serial_bridge.py"),
                           "--replay", "synth", "--methods", args.methods, "--serve",
                           "--http-port", str(args.http_port), "--dur", "180"],
                          stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{args.http_port}"
    errs: list[str] = []
    try:
        for _ in range(300):
            if br.poll() is not None:
                print(f"브리지가 먼저 끝났다 — {out / 'bridge.log'} 를 볼 것")
                return 1
            try:
                urllib.request.urlopen(url + "/fe", timeout=5)
                break
            except Exception:
                time.sleep(0.2)
        with sync_playwright() as p:
            exe = find_chromium()
            b = p.chromium.launch(**({"executable_path": exe} if exe else {}))

            def page(ctx=None):
                # 문맥(ctx)은 만들 때 viewport 를 받았다 — new_page 는 그것을 안 받는다
                pg = ctx.new_page() if ctx else b.new_page(viewport={"width": 1440, "height": 1100})
                pg.on("pageerror", lambda e: errs.append(f"PAGEERROR {e}"))
                pg.on("console", lambda m: m.type == "error" and errs.append(f"console {m.text}"))
                pg.on("response", lambda r: r.status >= 400 and errs.append(f"{r.status} {r.url}"))
                return pg

            pg = page()
            pg.goto(url + "/live.html")
            time.sleep(args.warm)
            shots = []

            def shot(name, **kw):
                f = out / f"{name}.png"
                pg.screenshot(path=str(f), **kw)
                shots.append(f)

            shot("1_split_scroll", full_page=True)
            pg.click("#p-sweep")
            time.sleep(3)
            shot("2_split_sweep", full_page=True)
            box = pg.locator("#panels canvas").nth(1).bounding_box()
            shot("3_sweep_canvas", clip=box)
            pg.click("#m-over")
            time.sleep(3)
            shot("4_overlay_sweep", full_page=True)
            chips = pg.locator("#legend button")
            if chips.count() > 1:
                chips.nth(1).click()
                time.sleep(0.4)
                chips.nth(1).click()
            fe = pg.locator("#feseg button")
            if fe.count():
                fe.nth(0).click()
                time.sleep(6)
                shot("5_after_fe_switch", full_page=True)
            ctx = b.new_context(viewport={"width": 1440, "height": 1100}, reduced_motion="reduce")
            pg = page(ctx)
            pg.goto(url + "/live.html")
            time.sleep(4)
            glow = pg.evaluate("document.getElementById('sw-glow')?.checked")
            shot("6_new_window_reduced_motion", full_page=True)
            b.close()
    finally:
        br.terminate()
        br.wait(timeout=10)
    print("찍었다 —", ", ".join(s.name for s in shots), f"(폴더 {out})")
    print(f"reduced-motion 새 창의 선단 밝기: {'켜짐' if glow else '꺼짐 (정상)'}")
    if errs:
        print("오류:", *errs, sep="\n  ")
        return 1
    print("페이지 오류 · 콘솔 오류 · 404 없음. **이제 PNG 를 눈으로 본다** (§8).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
