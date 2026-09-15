"""시연 화면을 **실제 브라우저로 열어서** 본다.

## 왜 이 파일이 있나

이 저장소의 UI 결함은 거의 전부 「화면을 열면 5 초에 보이는 것」이었는데, 여는
장치가 없었다 — 그림은 `tests/test_figure_text.py` 가 보지만 HTML 은 아무도 안
봤다. 그래서:

* `docs/33` — 「세 안이 레인 렌더러를 공유한다」고 적힌 채 두 안이 **첫 판 그대로**
  였다. 「문서가 코드보다 앞서 있었고, **화면을 열지 않으면 드러나지 않는다**」
* U-1 ~ U-14 — 방법 색 없음 · 가로 넘침 · 호버 없음 · 범례 없음 · `@media` 0 개
* UO-1 — 그림을 **열어 보지 않고** 저장해 tracked 산출물을 깨진 것으로 덮었다

## 왜 기본 실행에서 빠져 있나 (`screens` 마커)

브라우저를 띄우고 8.6 MB 짜리 은행을 파싱하므로 화면당 몇 초가 든다.
`pytest tests/` 는 커밋 직전에 매번 도는 검사라 1 분 안에 끝나야 한다(`pytest.ini`).

    pytest tests/ -m screens        # 화면 검사만
    pytest tests/ -m ""             # 전부

**화면을 고쳤으면 이것을 돌린다** — `docs/17_checklists.md` 에 트리거로 올려 뒀다.

브라우저가 없는 환경에서는 **실패가 아니라 skip** 이다. 커밋을 막을 이유가 없다.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from _screens import (  # noqa: E402
    SCREENS, WIDTHS, find_chromium, launch, new_page, settle,
)

pytestmark = pytest.mark.screens

# **CI 에서는 건너뛰기를 금지한다.**
#
# 브라우저가 없으면 이 파일은 통째로 skip 되고 pytest 는 **0 으로 끝난다** —
# 「1 skipped」에 초록불이다. 개발 머신에서는 그게 맞다(브라우저 없다고 커밋을
# 막을 이유가 없다). **CI 에서는 정반대다**: 검사하라고 만든 것이 아무것도
# 검사하지 않은 채 통과하면, 없느니만 못한 「검사가 있다」는 착각만 남는다.
#
# 이 저장소는 그 형태를 이미 여러 번 겪었다 — UF-2(아무 일도 안 하는 줄이
# 검사를 통과했다), O-30(가드가 `NameError` 를 삼켜 중간 체크포인트를 올려
# 둬도 통과했다). 그래서 `CI` 가 켜져 있으면 **skip 이 아니라 실패**다.
_CI = os.environ.get("CI", "").lower() in ("1", "true", "yes")


def _no_browser(msg: str):
    if _CI:
        raise RuntimeError(
            f"CI 인데 화면 검사를 돌릴 수 없다: {msg}. "
            "CI 에서 이 파일이 skip 되면 «검사했다» 가 거짓이 된다 — "
            "워크플로가 playwright 와 Chromium 을 설치하는지 본다.")
    pytest.skip(msg, allow_module_level=True)


try:
    import playwright.sync_api as playwright        # noqa: F401
except ImportError:
    _no_browser("playwright 가 없다 (개발용 검증판)")
CHROMIUM = find_chromium()
if not CHROMIUM:
    _no_browser("Chromium 을 못 찾았다")

READY = [s for s in SCREENS if s.ready()]
IDS = [s.name for s in READY]
if _CI and not IDS:
    raise RuntimeError(
        "CI 인데 검사할 화면이 하나도 없다 — `demo/demo_bank.js` 같은 "
        "`needs` 파일이 체크아웃에 없는 것이다. 그대로 두면 0 개를 검사하고 "
        "초록으로 끝난다.")


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as pw:
        b = launch(pw, exe=CHROMIUM)
        yield b
        b.close()


@pytest.fixture(scope="module")
def loaded(browser):
    """화면마다 **한 번만** 열어 두고 재사용한다 — 은행 파싱이 비싸다."""
    pages = {}
    for s in READY:
        page = new_page(browser)
        errors = settle_page(page, s)
        pages[s.name] = (page, errors)
    yield pages
    for page, _ in pages.values():
        page.close()


def settle_page(page, screen):
    page.goto(screen.url)
    return settle(page, animated=screen.animated)


def _screen(name):
    return next(s for s in READY if s.name == name)


# ------------------------------------------------------------------ 기본
@pytest.mark.parametrize("name", IDS)
def test_screen_loads_without_console_errors(loaded, name):
    """콘솔에 오류가 없어야 한다. **화면이 조용히 반쯤 죽는 것**이 가장 흔한 실패다."""
    s = _screen(name)
    _, errors = loaded[name]
    unexpected = [e for e in errors
                  if not any(pat in e for pat in s.allow_console)]
    assert not unexpected, f"{name}:\n  " + "\n  ".join(unexpected)


@pytest.mark.parametrize("name", IDS)
def test_screen_has_no_horizontal_overflow(browser, name):
    """**확정된 세 폭에서** 가로 스크롤이 생기면 안 된다 (U-9 — `@media` 가 0 개였다).

    1920 은 표시 장치, 1280 은 최소 방어폭이다 (`docs/ui/01_system.md` 7 절).
    """
    s = _screen(name)
    bad = []
    for w in WIDTHS:
        page = new_page(browser, width=w)
        settle_page(page, s)
        sw = page.evaluate("document.documentElement.scrollWidth")
        cw = page.evaluate("document.documentElement.clientWidth")
        if sw > cw + 1:
            bad.append(f"{w}px 에서 {sw} > {cw} ({sw - cw}px 넘침)")
        page.close()
    assert not bad, f"{name}: " + " · ".join(bad)


@pytest.mark.parametrize("name", [s.name for s in READY if s.expect_canvas])
def test_canvas_actually_drew_something(loaded, name):
    """캔버스가 **실제로 그려졌는지** 본다 — 크기만 있고 빈 경우가 조용한 실패다."""
    page, _ = loaded[name]
    painted = page.evaluate("""() => {
      const out = [];
      for (const cv of document.querySelectorAll('canvas')) {
        if (!cv.width || !cv.height) { out.push([cv.id || '(익명)', 0, 'w/h 가 0']); continue; }
        const g = cv.getContext('2d');
        const d = g.getImageData(0, 0, cv.width, cv.height).data;
        let ink = 0;
        for (let i = 3; i < d.length; i += 4 * 37) if (d[i] > 8) ink++;
        out.push([cv.id || '(익명)', ink, '']);
      }
      return out;
    }""")
    assert painted, f"{name}: 캔버스가 하나도 없다"
    empty = [f"{cid} ({note or '픽셀 0'})" for cid, ink, note in painted if ink == 0]
    assert len(empty) < len(painted), (
        f"{name}: 캔버스 {len(painted)} 개가 전부 비었다 — " + ", ".join(empty))


@pytest.mark.parametrize("name", IDS)
def test_text_does_not_spill_out_of_its_box(loaded, name):
    """글자가 **자기 상자 밖으로 흘렀는지** 본다.

    `docs/33` 이 겪은 결함 셋 중 둘이 이 종류였다 — 축 라벨이 화면 밖으로 나가고,
    곡선이 그림 상자를 넘었다. 그때는 사람이 열어서 찾았다.
    """
    page, _ = loaded[name]
    spills = page.evaluate("""() => {
      const bad = [];
      for (const el of document.querySelectorAll('body *')) {
        const st = getComputedStyle(el);
        if (st.overflow !== 'visible' || st.display === 'none') continue;
        if (!el.firstElementChild && el.scrollWidth > el.clientWidth + 2
            && el.clientWidth > 0) {
          bad.push((el.tagName + '.' + (el.className || '')).slice(0, 48)
                   + ` (${el.scrollWidth}>${el.clientWidth})`);
        }
      }
      return bad.slice(0, 8);
    }""")
    assert not spills, f"{name}: 글이 상자 밖으로 흐른다 — " + ", ".join(spills)


# ------------------------------------------------------------------ 규약
@pytest.mark.parametrize("name", IDS)
def test_animated_screens_respect_reduced_motion(browser, name):
    """`prefers-reduced-motion: reduce` 를 켠 채 열고 **화면이 실제로 멈춰 있는지** 잰다.

    `ui-ux-pro-max` 의 사전 인도 체크리스트: 「자동 회전 콘텐츠는 정지 수단이 있고
    reduced-motion 에서 멈춘다」.

    **소스에 문자열이 있는지로 재지 않는다.** `mockup_expo.html` 에는
    `@media (prefers-reduced-motion:reduce){canvas{animation:none}}` 가 있지만
    캔버스는 CSS 애니메이션이 아니라 `requestAnimationFrame` 으로 그리므로
    **그 한 줄은 아무것도 하지 않는다.** 문자열 검사는 그런 것을 통과시킨다.
    """
    s = _screen(name)
    if not s.animated:
        pytest.skip("애니메이션이 없는 화면")

    page = new_page(browser, reduced_motion="reduce")
    settle_page(page, s)
    snap = """() => {
      const out = [];
      for (const cv of document.querySelectorAll('canvas')) {
        if (!cv.width || !cv.height) { out.push(''); continue; }
        const d = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
        let h = 0;
        for (let i = 0; i < d.length; i += 4 * 101) h = (h * 31 + d[i] + d[i+3]) | 0;
        out.push(String(h));
      }
      return out.join('|');
    }"""
    before = page.evaluate(snap)
    page.wait_for_timeout(1500)
    after = page.evaluate(snap)
    page.close()

    assert before == after, (
        f"{name}: reduced-motion 을 켰는데 캔버스가 계속 바뀐다 — "
        "자동으로 도는 애니메이션을 멈춰야 한다")


@pytest.mark.parametrize("name", [s.name for s in READY if s.expect_canvas])
def test_identity_is_not_color_alone(loaded, name):
    """**색만으로 신원을 알리면 안 된다** — 범례든 직접 라벨이든 글자가 있어야 한다.

    `docs/33` 이 카드에 건 규칙(「aqua ↔ orange 의 tritan 분리도가 경고 대역이라
    색만으로 신원을 알리면 안 된다」)을 **본 화면까지 넓힌 것**이다. 우리 팔레트에는
    대비 3:1 을 못 넘는 색이 있고, 그 WARN 은 면제가 아니라 보조 부호 의무다.
    """
    page, _ = loaded[name]
    labelled = page.evaluate("""() => {
      const t = document.body.innerText || '';
      const hits = (t.match(/\\bM(_FE|0\\d|\\d\\d)\\b|\\bB01\\b/g) || []);
      return new Set(hits).size;
    }""")
    assert labelled >= 2, (
        f"{name}: 화면에 방법 이름이 {labelled} 개뿐이다 — 색만으로 구분하게 된다")


# ------------------------------------------------------------------ 테마
@pytest.mark.parametrize("name", IDS)
def test_dark_theme_does_not_leave_text_on_its_own_ground(browser, name):
    """다크에서 **글자와 배경이 같은 쪽으로 붙지 않아야** 한다.

    다크를 선언한 화면만 본다. 토큰의 유일한 정의가 다크 블록 안에 있으면
    테마를 안 고른 사용자에게 한쪽 테마의 글자가 다른 쪽 배경 위에 얹힌다 —
    `mockup_expo` 만 다크를 갖고 있고 나머지는 아직 라이트뿐이다.
    """
    s = _screen(name)
    if "prefers-color-scheme" not in s.path.read_text(encoding="utf-8"):
        pytest.skip("다크를 선언하지 않은 화면")
    page = new_page(browser, theme="dark")
    settle_page(page, s)
    lum = page.evaluate("""() => {
      const rgb = s => (s.match(/\\d+/g) || [0,0,0]).slice(0,3).map(Number);
      const L = c => { const [r,g,b] = c.map(v => v/255);
        return 0.2126*r + 0.7152*g + 0.0722*b; };
      const st = getComputedStyle(document.body);
      return [L(rgb(st.backgroundColor)), L(rgb(st.color))];
    }""")
    page.close()
    bg, fg = lum
    assert abs(fg - bg) > 0.2, (
        f"{name}: 다크에서 글자({fg:.2f})와 배경({bg:.2f})의 밝기가 너무 가깝다")


# ------------------------------------------------------------------ 테마 두 벌
@pytest.mark.parametrize("name", IDS)
def test_explicit_dark_choice_darkens_the_page(browser, name):
    """**OS 는 라이트인데 화면에서 다크를 고른** 경우를 따로 본다.

    다크를 매체 질의에만 적어 두면 이 조합에서 다크 토큰 일부(다른 파일에서 온
    방법 색)만 바뀌고 배경은 라이트로 남는다 — `layout_b.html` 이 실제로 그랬다.
    `tokens.css` 는 두 벌을 갖고 있었으므로 **한 화면 안에서 두 테마가 섞였다**.
    """
    s = _screen(name)
    src = s.path.read_text(encoding="utf-8")
    if 'data-theme="dark"' not in src and "prefers-color-scheme" not in src:
        pytest.skip("테마를 선언하지 않은 화면")
    page = new_page(browser, theme="light")          # OS 는 라이트
    settle_page(page, s)
    page.evaluate("() => document.documentElement.dataset.theme = 'dark'")
    page.wait_for_timeout(300)
    lum = page.evaluate("""() => {
      const rgb = s => (s.match(/\\d+/g) || [255,255,255]).slice(0,3).map(Number);
      const L = c => { const [r,g,b] = c.map(v => v/255);
        return 0.2126*r + 0.7152*g + 0.0722*b; };
      const st = getComputedStyle(document.body);
      return [L(rgb(st.backgroundColor)), L(rgb(st.color))];
    }""")
    page.close()
    bg, fg = lum
    assert bg < 0.25, (
        f"{name}: data-theme=\"dark\" 인데 배경이 아직 밝다 (L={bg:.2f}) — "
        "다크가 매체 질의에만 적혀 있다")
    assert fg - bg > 0.3, f"{name}: 다크에서 글자({fg:.2f})가 배경({bg:.2f}) 에 묻힌다"


# ------------------------------------------------------------------ 스윕
SWEEP = "ui/layout_b.html"

# 커서 앞 잉크량을 열 단위로 재는 탐침. **알파로 가중한다** — `destination-out` 은
# RGB 를 그대로 두고 알파만 깎으므로, RGB 만 보면 완전히 지워진 픽셀도 «순백» 으로
# 읽힌다. 처음 이 검사를 쓸 때 그 함정에 그대로 걸렸다.
_INK_JS = """() => {
  const p = panels[1];
  const cv = document.querySelectorAll('.lane .plot canvas')[1];
  const d = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
  const col = [];
  for (let x = 0; x < cv.width; x++){
    let s = 0;
    for (let y = 0; y < cv.height; y++){
      const o = (y * cv.width + x) * 4;
      const L = 0.2126*d[o] + 0.7152*d[o+1] + 0.0722*d[o+2];
      if (L > 80) s += d[o+3] / 255;
    }
    col.push(s);
  }
  const x1 = Math.round((((animIdx % p.n) + p.n) % p.n) * (p.pxPerSec / B.fs));
  return {x1: x1, w: cv.width, col: col};
}"""


def _sweep_ink(browser):
    """스윕을 **두 바퀴째까지 돌려** 커서 기준 잉크 단면을 얻는다.

    첫 바퀴에는 커서 앞이 원래 비어 있어서 페이드가 지울 것이 없다. 이 효과는
    지난 바퀴의 파형이 남아 있어야 보이므로 한 바퀴를 넘겨야 한다.
    """
    s = _screen(SWEEP)
    page = new_page(browser, theme="dark")
    settle_page(page, s)
    page.get_by_role("button", name="스윕", exact=True).click()
    page.wait_for_timeout(13_000)
    r = page.evaluate(_INK_JS)
    page.close()
    return r


@pytest.mark.skipif(SWEEP not in IDS, reason="스윕 화면이 아직 없다")
def test_sweep_fades_into_the_cursor_instead_of_cutting(browser):
    """커서 앞의 지난 파형은 **다가올수록 옅어져야** 한다 — 딱 잘리면 안 된다.

    두 번 같은 자리에서 틀렸다. 페이드 자체는 처음부터 있었는데, 커서 앞을 통째로
    비우는 «지움 막대» 를 페이드 **바깥쪽**에 두는 바람에 어느 픽셀이든 커서가
    닿기 한참 전에 이미 지워져 있었다. 페이드는 칠할 것이 없었고 화면에는 딱
    잘린 경계만 남았다. 폭을 50 % → 22 % 로 줄여 봐도 순서가 그대로라 그대로였다.

    그래서 **폭이 아니라 모양을 잰다**: 커서 바로 앞은 비어 있고, 멀어질수록
    잉크가 늘어나며, 그 사이가 계단이 아니라 경사여야 한다.
    """
    r = _sweep_ink(browser)
    col, w, x1 = r["col"], r["w"], r["x1"]
    band = lambda a, b: sum(col[(x1 + k) % w] for k in range(a, b))

    near, mid, far = band(10, 90), band(150, 230), band(300, 380)
    assert near < far * 0.15, (
        f"커서 바로 앞이 안 비었다 — 가까이 {near:.0f} vs 멀리 {far:.0f}")
    assert far > 20, f"커서 앞 멀리에 지난 바퀴가 안 남았다 ({far:.0f})"
    # 계단이 아니라 경사: 중간 띠가 양 끝 **사이**에 있어야 한다. 딱 잘라 지우면
    # 중간은 near 나 far 중 한쪽에 붙는다.
    assert near < mid < far, (
        f"페이드가 경사가 아니라 계단이다 — 가까이 {near:.0f} · 중간 {mid:.0f} · "
        f"멀리 {far:.0f}")


@pytest.mark.skipif(SWEEP not in IDS, reason="스윕 화면이 아직 없다")
def test_sweep_glows_behind_the_cursor(browser):
    """커서 **뒤**는 밝게 톤업됐다가 멀어지며 원래 색으로 돌아와야 한다.

    감시장치 형광체를 흉내 낸 것이다. 밝기가 오르면 밝은 픽셀이 늘어나므로
    같은 탐침으로 잡힌다 — 커서 바로 뒤 띠가 그보다 앞선 띠보다 진해야 한다.
    """
    r = _sweep_ink(browser)
    col, w, x1 = r["col"], r["w"], r["x1"]
    band = lambda a, b: sum(col[(x1 + k) % w] for k in range(a, b))

    hot, cool = band(-60, -10), band(-180, -130)
    assert hot > cool, (
        f"커서 뒤 잔광이 없다 — 바로 뒤 {hot:.0f} 가 먼 쪽 {cool:.0f} 보다 진하지 않다")


# ------------------------------------------------------------------ 갤러리
GALLERY = ROOT / "demo" / "ui" / "gallery.html"


def test_gallery_lists_every_screen():
    """갤러리와 검사가 **같은 화면 목록**을 봐야 한다.

    어긋나면 「갤러리에는 있는데 검사는 안 하는 화면」이나 그 반대가 생긴다.
    이 저장소는 그 종류를 이미 겪었다 — `docs/33` 이 세 안이 공유한다고 적은 채
    두 안이 첫 판 그대로였다.
    """
    if not GALLERY.exists():
        pytest.skip("갤러리가 아직 없다")
    src = GALLERY.read_text(encoding="utf-8")
    # 갤러리는 자기 위치(demo/ui/) 기준 상대경로를 쓰므로 파일명으로 대조한다
    missing = [s.name for s in SCREENS
               if s.name not in src and Path(s.name).name not in src]
    assert not missing, f"갤러리가 안 싣는 화면: {missing}"
