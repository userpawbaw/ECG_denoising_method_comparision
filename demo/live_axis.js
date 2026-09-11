/* 실시간 화면의 **축** — 이득·중심·«어느 자리에 정말 값이 있는가».
 *
 * 그리기(live.html)에서 떼어 둔 이유는 하나다. **여기가 두 번 재발한
 * 자리**라 테스트가 붙어야 한다 — 이득이 심박 주기로 들썩인 F-32, 그
 * 수정이 사다리 단차보다 좁은 밴드를 써서 다시 들썩인 F-43, 그리고 절대
 * 인덱스와 «받은 개수» 를 섞어 화면 한가운데서 신호가 돋아난 F-44.
 * `tests/test_live_axis.py` 가 node 로 이 파일을 그대로 돌린다.
 *
 * 여기에는 DOM 도 캔버스도 없다. 들어오는 것은 숫자뿐이다.
 */
(function (root) {
  "use strict";

  // ------------------------------------------------------------------ 사다리
  /* **사다리의 최대 단차가 히스테리시스 밴드보다 좁아야 한다.**
   * 이것이 이 파일의 핵심 불변식이다. 넓으면 한 칸 올라가자마자 내려올
   * 조건이 성립해 이득이 두 값 사이를 왕복한다 — 1-2-5 사다리(단차 2.5 배)에
   * +10 %/−45 % 밴드(폭 2.0 배)를 물린 것이 정확히 그 경우였다 (F-43).
   *
   * 그래서 사다리를 촘촘하게 바꿨다. 단차가 최대 5/3 = 1.67 배라 화면도
   * 덜 빈다 — 1-2-5 는 필요 폭의 2.5 배를 잡을 수 있어 파형이 화면의 40 %
   * 밖에 못 썼다. */
  var STEPS = [1, 1.5, 2, 3, 5, 7.5];
  var STEP_MAX = 5 / 3;                    // 이웃한 두 단의 최대 비

  /** `v` 이상인 가장 작은 사다리 값. */
  function ladder(v) {
    if (!(v > 0) || !isFinite(v)) return STEPS[0];
    var e = Math.floor(Math.log10(v)), p = Math.pow(10, e), b = v / p;
    for (var i = 0; i < STEPS.length; i++) if (b <= STEPS[i] * (1 + 1e-12)) return STEPS[i] * p;
    return STEPS[0] * p * 10;
  }

  // -------------------------------------------------------------------- 이득
  /* 밴드 [LO, HI] 는 **필요 반높이 / 현재 단** 으로 잰다.
   *
   * 다시 잡을 때 `AIM` 배의 여유를 두므로 잡은 직후의 비는
   *     (1/(AIM·STEP_MAX), 1/AIM] = (0.48, 0.80]
   * 이고, 밴드 [0.42, 1.00] 안쪽에 **양쪽 다 여유를 두고** 들어간다.
   * 그러므로 한 번 바꾼 뒤 곧바로 되돌아올 일이 없다 — 왕복이 불가능하다.
   * `HOLD_MS` 는 그 위에 얹는 둘째 잠금이다: 진짜로 진폭이 변해도 이득은
   * 2 초에 한 번보다 자주 안 바뀐다. */
  var LO = 0.42, HI = 1.00, AIM = 1.25, HOLD_MS = 2000;

  /** 이득 하나. **화면 전체가 이 하나를 같이 쓴다** — 계열마다 따로 잡으면
   *  겹쳐 놓아도 크기를 비교할 수 없고, 계열마다 다른 순간에 바뀌어 화면이
   *  제각각 들썩인다. */
  function makeGain(opt) {
    var o = opt || {};
    var lo = o.lo == null ? LO : o.lo, hi = o.hi == null ? HI : o.hi;
    var aim = o.aim == null ? AIM : o.aim, hold = o.holdMs == null ? HOLD_MS : o.holdMs;
    var s = 0, t = 0, n = 0;
    return {
      /** 화면 반높이가 담아야 할 값 `need` 를 주면 **쓸 이득**을 준다. */
      fit: function (need, now) {
        // **잴 것이 없으면 잡지 않는다.** 화면이 빈 동안(전환 직후의 warm-up)
        // 0 으로 한 번 잡아 버리면 터무니없는 이득이 걸리고, 최소 유지 시간
        // 때문에 첫 표본이 와도 2 초를 그대로 간다.
        if (!(need > 0) || !isFinite(need)) return s;
        if (!s) { s = ladder(need * aim); t = now; n = 1; return s; }
        if ((need > s * hi || need < s * lo) && now - t >= hold) {
          s = ladder(Math.max(need, 1e-9) * aim); t = now; n++;
        }
        return s;
      },
      value: function () { return s; },
      changes: function () { return n; },
      reset: function () { s = 0; t = 0; n = 0; },
    };
  }

  /** 중심은 **계열마다 따로** 둔다. 입력은 front-end 이전이라 전극 DC 가
   *  실려 있고(AD8232 는 전원 중간에 띄운다) 출력들은 0 둘레다. 그 차이는
   *  신호가 아니라 회로의 기준점이므로 중심만 빼고 **이득은 같이 쓴다.** */
  function makeCenter(opt) {
    var o = opt || {};
    var dead = o.dead == null ? 0.20 : o.dead;
    var hold = o.holdMs == null ? HOLD_MS : o.holdMs;
    var st = {};
    return {
      at: function (key, want, s, now) {
        var c = st[key];
        if (!c) { st[key] = { m: want, t: now }; return want; }
        if (Math.abs(want - c.m) > dead * s && now - c.t >= hold) { c.m = want; c.t = now; }
        return c.m;
      },
      clear: function () { st = {}; },
    };
  }

  // ------------------------------------------------------------ 보이는 구간
  /** 화면에 남아 있는 **가장 오래된 절대 인덱스**. */
  function windowStart(first, head, cap) { return Math.max(first, head - cap + 1); }

  /** 화면 칸 `i`(0..cap-1) 는 **어느 절대 샘플**인가.
   *
   *  스크롤이면 오른쪽 끝이 `head` 다. 스윕이면 자리가 고정이라, 칸 `i` 는
   *  `i ≡ a (mod cap)` 인 샘플 중 `head` 를 안 넘는 가장 최근 것이다.
   *
   *  **여기를 «받은 개수» 로 대신하면 안 된다.** 버퍼는 절대 인덱스의
   *  나머지로 쓰는데 개수는 0 부터 세므로, 스트림 중간에 창을 열거나
   *  front-end 를 바꿔 인덱스가 다시 시작하면 둘이 어긋난다. 그러면 아직
   *  안 쓴 0 자리가 «유효» 로 판정돼 **화면 한가운데에서 평평한 선이
   *  돋아나고**, 정작 새 표본은 오른쪽 끝에서 따로 나타난다 (F-44). */
  function absAt(i, head, cap, play) {
    if (play === "sweep") {
      var back = ((((head % cap) - i) % cap) + cap) % cap;
      return head - back;
    }
    return head - (cap - 1 - i);
  }

  /** 그 절대 자리에 **정말 받은 샘플이 있나.** */
  function have(a, first, head, cap) { return a >= windowStart(first, head, cap) && a <= head; }

  // -------------------------------------------------------------- 진폭 재기
  /** 보이는 구간의 **중심과 로버스트 피크**.
   *
   *  피크를 평균 절대 편차(MAD)의 몇 배로 잡던 것이 겹치기에서 R 피크를
   *  화면 밖으로 밀어냈다 — 잘린 파형은 어느 방법이 어디서 다른지를 가린다.
   *  그렇다고 진짜 최대값으로 잡으면 스파이크 한 점에 파형 전체가 납작해진다.
   *  그래서 **MAD 단위 히스토그램의 99.5 백분위**를 쓴다: R 피크(창 하나에
   *  수십 표본)는 들어가고 한두 점짜리 잡음은 안 들어간다. */
  function scan(a, lo, hi, cap, q) {
    var n = hi - lo + 1, k, m = 0;
    if (n <= 0) return { m: 0, peak: 0, n: 0 };
    for (k = lo; k <= hi; k++) m += a[((k % cap) + cap) % cap];
    m /= n;
    var mad = 0;
    for (k = lo; k <= hi; k++) mad += Math.abs(a[((k % cap) + cap) % cap] - m);
    mad /= n;
    if (!(mad > 0)) return { m: m, peak: 0, n: n };
    var NB = 96, W = 0.5, h = new Int32Array(NB + 1), b;
    for (k = lo; k <= hi; k++) {
      b = Math.abs(a[((k % cap) + cap) % cap] - m) / mad / W;
      h[b >= NB ? NB : b | 0]++;
    }
    var want = Math.ceil(n * (q == null ? 0.995 : q)), c = 0, bin = NB;
    for (k = 0; k <= NB; k++) { c += h[k]; if (c >= want) { bin = k; break; } }
    return { m: m, peak: (bin + 1) * W * mad, n: n };
  }

  var api = {
    STEPS: STEPS, STEP_MAX: STEP_MAX, LO: LO, HI: HI, AIM: AIM, HOLD_MS: HOLD_MS,
    ladder: ladder, makeGain: makeGain, makeCenter: makeCenter,
    windowStart: windowStart, absAt: absAt, have: have, scan: scan,
  };
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.LiveAxis = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
