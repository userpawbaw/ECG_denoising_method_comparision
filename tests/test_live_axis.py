"""실시간 화면의 축 (`demo/live_axis.js`) — **두 번 재발한 자리**를 고정한다.

여기 있는 것은 셋이다.

1. **이득이 왕복하지 않는다** — 사다리의 최대 단차가 히스테리시스 밴드보다
   넓으면 한 칸 올라가자마자 내려올 조건이 성립한다. F-32 의 수정(1-2-5
   사다리 + `+10 %/−45 %` 밴드)이 정확히 그 경우였고, 그래서 파형이 계속
   들썩였다 (F-43). 옛 규칙으로도 같은 검사를 돌려 **검사가 진짜로 그것을
   잡는지**까지 본다.
2. **화면 칸 ↔ 절대 샘플 대응이 맞는다** — 스트림 중간에 창을 열면
   «받은 개수» 로 유효성을 판정하던 옛 코드가 화면 한가운데에 평평한 선을
   그렸다 (F-44).
3. **로버스트 피크가 R 피크를 담는다** — 평균 절대 편차로 잡으면 겹치기에서
   R 피크가 잘렸다 (F-45).

`node` 가 없으면 건너뛴다 — 파이썬으로 같은 규칙을 다시 구현하면 그것은
**구현이 둘**이 되는 것이고, 둘이 어긋나는 순간 테스트가 거짓말을 한다.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
AXIS = ROOT / "demo" / "live_axis.js"
NODE = shutil.which("node") or shutil.which("nodejs") or "/opt/node22/bin/node"


def run_js(body: str) -> dict:
    """`live_axis.js` 를 **그대로** 불러 돌리고 마지막 JSON 한 줄을 읽는다."""
    if not Path(NODE).exists():
        pytest.skip("node 가 없다 — 축 계산은 JS 구현 하나뿐이라 여기서 멈춘다")
    src = f'const AX = require({json.dumps(str(AXIS))});\n{body}\n'
    out = subprocess.run([NODE, "-e", src], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------- 이득
def test_gain_band_is_wider_than_the_ladder_step():
    """다시 잡은 **직후**의 `need/s` 가 밴드 안쪽에 여유를 두고 들어가는가.

    이것이 성립하면 «한 번 바꾼 뒤 곧바로 되돌아가기» 가 **불가능**하다.
    F-43 의 근본 원인은 이 부등식이 깨진 것이었다.
    """
    r = run_js("""
      let lo = Infinity, hi = -Infinity;
      for (let v = 1e-3; v < 1e3; v *= 1.0007){
        const s = AX.ladder(v * AX.AIM), q = v / s;
        if (q < lo) lo = q; if (q > hi) hi = q;
      }
      console.log(JSON.stringify({lo, hi, LO: AX.LO, HI: AX.HI, step: AX.STEP_MAX}));
    """)
    assert r["lo"] > r["LO"], "다시 잡자마자 밴드 아래로 떨어진다 = 왕복한다"
    assert r["hi"] <= r["HI"], "다시 잡자마자 밴드 위로 벗어난다 = 왕복한다"
    # 사다리 단차보다 밴드가 넓어야 한다는 것과 같은 말이다.
    assert r["HI"] / r["LO"] > r["step"]


# R 피크가 창에 들고 나며 진폭이 숨쉬는 것을 흉내낸 자극. **사다리 경계를
# 가로지르게** 만든 것이 요점이다 — 경계를 안 넘으면 어느 규칙이든 안 흔들린다.
BREATHE = """
      const wave = (c, k) => c * (1 + 0.12 * Math.sin(k / 18));   // 약 3 s 주기
      const CENTERS = [];
      for (let b = 1.0; b < 10; b += 0.05) CENTERS.push(b * 0.1);
"""


def test_gain_does_not_oscillate_on_a_breathing_amplitude():
    """진폭이 ±12 % 로 숨쉬는 동안 이득이 몇 번 바뀌나.

    중심을 한 자릿수 전체에 걸쳐 훑는다 — **어디에 걸리느냐**가 문제였기
    때문이다. 옛 규칙은 사다리의 2 → 5 경계 근처에서만 흔들렸고, 그래서
    «가끔 멀쩡해 보였다».
    """
    r = run_js(BREATHE + """
      let worst = 0, at = 0;
      for (const c of CENTERS){
        const g = AX.makeGain();
        let t = 0, flips = 0, last = 0;
        for (let k = 0; k < 4000; k++){
          t += 40;                                  // 25 fps -> 160 s
          const s = g.fit(wave(c, k), t);
          if (last && s !== last) flips++;
          last = s;
        }
        if (flips > worst){ worst = flips; at = c; }
      }
      console.log(JSON.stringify({worst, at}));
    """)
    assert r["worst"] <= 1, (
        f"진폭 중심 {r['at']:.3f} 에서 160 s 동안 이득이 {r['worst']} 번 바뀌었다 — 또 들썩인다")


def test_the_old_rule_really_did_oscillate():
    """**검사가 진짜로 그것을 잡는지** 본다.

    옛 규칙(1-2-5 사다리 · `+10 %/−45 %` 밴드)을 **같은 자극**에 물리면
    왕복이 나와야 한다. 안 나오면 위 테스트는 아무것도 안 지키고 있는 것이다
    (`test_record_checker_still_detects_missing_records` 와 같은 취지).

    왜 옛 규칙이 무너지는지도 여기 숫자로 남는다 — 사다리의 2 → 5 단차가
    2.5 배인데 밴드 폭은 1.10/0.55 = 2.0 배라, **올라간 자리가 곧바로
    내려올 조건**(`need < 0.55 s`)을 만족한다.
    """
    r = run_js(BREATHE + """
      const old = v => { const e = Math.floor(Math.log10(v)), b = v / 10 ** e;
                         return (b <= 1 ? 1 : b <= 2 ? 2 : b <= 5 ? 5 : 10) * 10 ** e; };
      let worst = 0, at = 0;
      for (const c of CENTERS){
        let s = 0, flips = 0, last = 0;
        for (let k = 0; k < 4000; k++){
          const need = wave(c, k);
          if (!s || need > s * 1.10 || need < s * 0.55) s = old(need);
          if (last && s !== last) flips++;
          last = s;
        }
        if (flips > worst){ worst = flips; at = c; }
      }
      console.log(JSON.stringify({worst, at}));
    """)
    assert r["worst"] > 50, "옛 규칙이 안 흔들린다면 이 자극으로는 아무것도 못 잡는다"


# ------------------------------------------------------- 화면 칸 ↔ 절대 샘플
@pytest.mark.parametrize("play", ["scroll", "sweep"])
def test_screen_column_maps_to_the_sample_it_shows(play):
    """칸 `i` ↔ 절대 샘플이 **일대일**이고 창 안에 있는가.

    스윕은 «자리 고정» 이므로 버퍼 자리가 곧 칸 번호여야 하고, 스크롤은
    오른쪽 끝이 최신이어야 한다. 둘 다 한 버퍼로 나온다.
    """
    r = run_js(f"""
      const cap = 2500, head = 31499 + 777, play = {play!r};
      const slot = new Set();
      let bad = 0, newest = null, fixed = 0;
      for (let i = 0; i < cap; i++){{
        const a = AX.absAt(i, head, cap, play);
        slot.add(((a % cap) + cap) % cap);
        if (((a % cap) + cap) % cap === i) fixed++;
        if (a === head) newest = i;
        if (a > head || a <= head - cap) bad++;            // 창 밖을 가리킨다
      }}
      console.log(JSON.stringify({{bad, newest, fixed, slots: slot.size, cap}}));
    """)
    assert r["bad"] == 0
    assert r["slots"] == r["cap"], "두 칸이 같은 버퍼 자리를 가리킨다"
    if play == "scroll":
        assert r["newest"] == r["cap"] - 1, "스크롤에서 최신 샘플은 **오른쪽 끝**이다"
    else:
        assert r["fixed"] == r["cap"], "스윕에서 칸의 자리는 **고정**이어야 한다"


def test_a_window_opened_mid_stream_fills_from_the_right_edge():
    """스트림 **중간**에 붙었을 때 아직 안 받은 자리를 «있다» 고 하지 않는가.

    F-44 의 증상은 여기서 나왔다 — 옛 코드는 «받은 개수» 와 버퍼 자리를
    비교해서, 한가운데 자리가 유효로 판정되고 거기서 평평한 선이 돋아났다.
    """
    r = run_js("""
      const cap = 2500, first = 31500;          // 126 s 째에 창을 열었다
      const out = [];
      for (const got of [12, 300, 1200, 2600]){  // 받은 샘플 수
        const head = first + got - 1;
        let n = 0, lo = cap, hi = -1;
        for (let i = 0; i < cap; i++)
          if (AX.have(AX.absAt(i, head, cap, 'scroll'), first, head, cap)){
            n++; if (i < lo) lo = i; if (i > hi) hi = i;
          }
        out.push({got, n, lo, hi});
      }
      console.log(JSON.stringify({out, cap}));
    """)
    cap = r["cap"]
    for row in r["out"]:
        want = min(row["got"], cap)
        assert row["n"] == want, f"{row['got']} 개 받았는데 {row['n']} 칸이 유효하다"
        assert row["hi"] == cap - 1, "채워지는 쪽은 **오른쪽 끝**이어야 한다"
        assert row["lo"] == cap - want, "가운데가 아니라 끝에서부터 이어져야 한다"


def test_an_index_restart_does_not_resurrect_the_previous_screen():
    """front-end 전환처럼 절대 인덱스가 **되돌아가도** 옛 구간이 안 살아난다."""
    r = run_js("""
      const cap = 2500, first = 0, head = 1443;   // 전환 뒤 5.8 s 째
      let n = 0, hi = -1;
      for (let i = 0; i < cap; i++)
        if (AX.have(AX.absAt(i, head, cap, 'scroll'), first, head, cap)){ n++; hi = i; }
      console.log(JSON.stringify({n, hi, cap}));
    """)
    assert r["n"] == 1444
    assert r["hi"] == r["cap"] - 1


# ----------------------------------------------------------------- 진폭 재기
def test_robust_peak_holds_the_r_peak_but_not_a_single_spike():
    """R 피크는 **끝까지** 담고, 한 점짜리 스파이크에는 안 끌려간다.

    99.5 백분위만 쓰면 실측에서 진짜 최대의 82~93 % 라 밴드 위쪽까지 흘러갔을
    때 R 피크 끝이 잘린다 — 그래서 **최대값을 쓰되 백분위의 1.4 배로 묶는다.**
    """
    r = run_js("""
      const cap = 2500, a = new Float32Array(cap);
      // 0.1 mV 잡음 위에 250 표본마다 1.0 mV R 피크(폭 5 표본)
      for (let i = 0; i < cap; i++) a[i] = 0.02 * Math.sin(i / 3);
      for (let k = 0; k < cap; k += 250) for (let j = 0; j < 5; j++) a[k + j] = 1.0;
      const base = AX.scan(a, 0, cap - 1, cap);
      a[1234] = 12.0;                                  // 스파이크 한 점
      const hit = AX.scan(a, 0, cap - 1, cap);
      console.log(JSON.stringify({base: base.peak, hit: hit.peak}));
    """)
    # `scan` 이 주는 것은 **중심에서의 편차**다 — 중심이 0.02 이므로 0.98 이 정답.
    assert r["base"] >= 0.96, f"R 피크가 안 들어간다: {r['base']:.3f}"
    assert r["base"] <= 2.0, f"너무 크게 잡아 파형이 납작해진다: {r['base']:.3f}"
    assert r["hit"] <= r["base"] * 1.6, "스파이크 한 점에 끌려간다"


def test_the_drawn_half_height_never_clips_a_real_beat():
    """밴드 안을 **끝까지 흘러가도** R 피크가 안 잘리는가.

    `need` 가 `HI × s` 까지 갔을 때 진짜 최대가 `s` 를 넘으면 잘린다. 백분위만
    쓰던 때가 그랬다 — 겹치기에서 실제로 잘린 화면이 나왔다.
    """
    r = run_js("""
      const cap = 2500, a = new Float32Array(cap);
      for (let i = 0; i < cap; i++) a[i] = 0.02 * Math.sin(i / 3);
      for (let k = 0; k < cap; k += 250){                 // R 피크 (폭 5, 높이 1.0)
        for (let j = 0; j < 5; j++) a[k + j] = 1.0;
        a[k + 2] = 1.12;                                  // 뾰족한 끝
      }
      const r = AX.scan(a, 0, cap - 1, cap);
      let mx = 0;
      for (let i = 0; i < cap; i++) mx = Math.max(mx, Math.abs(a[i] - r.m));
      console.log(JSON.stringify({need: r.peak, mx, HI: AX.HI}));
    """)
    # need 가 밴드 위 끝(HI·s)까지 흘러간 최악의 경우: s = need / HI
    s = r["need"] / r["HI"]
    assert r["mx"] <= s, f"밴드 끝에서 R 피크가 잘린다 (max {r['mx']:.3f} > 반높이 {s:.3f})"


def test_the_page_uses_the_module_rather_than_its_own_copy():
    """`live.html` 이 축 계산을 **다시 구현**하고 있지 않은가.

    구현이 둘이 되면 테스트는 안 쓰이는 쪽을 지키게 된다.
    """
    page = (ROOT / "demo" / "live.html").read_text(encoding="utf-8")
    assert 'src="live_axis.js"' in page
    assert "LiveAxis" in page
    for gone in ["function ladder(", "function rawScale(", "function vscale("]:
        assert gone not in page, f"{gone} 가 화면 쪽에 남아 있다 — 구현이 둘이다"


def test_gain_waits_for_something_to_measure():
    """화면이 빈 동안(전환 직후 warm-up)에는 이득을 **잡지 않는다.**

    0 으로 한 번 잡아 버리면 터무니없는 값이 걸리고, 최소 유지 시간 때문에
    첫 표본이 들어와도 2 초를 그대로 간다 — 그 2 초가 «파형이 폭발한» 화면이다.
    """
    r = run_js("""
      const g = AX.makeGain();
      const empty = g.fit(0, 0);                 // 아직 아무것도 없다
      const firstReal = g.fit(0.4, 40);          // 첫 표본이 왔다
      console.log(JSON.stringify({empty, firstReal, changes: g.changes()}));
    """)
    assert r["empty"] == 0, "잴 것이 없는데 이득을 잡았다"
    assert r["firstReal"] > 0.4, "첫 표본이 오면 **기다리지 않고** 바로 잡아야 한다"
    assert r["changes"] == 1
