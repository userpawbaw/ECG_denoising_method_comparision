/* 실시간 화면의 **축** — 이득·중심·«어느 자리에 정말 값이 있는가».
 *
 * 그리기(live.html)에서 떼어 둔 이유는 하나다. **여기가 두 번 재발한
 * 자리**라 테스트가 붙어야 한다 — 이득이 심박 주기로 들썩인 F-32, 그
 * 수정이 사다리 단차보다 좁은 밴드를 써서 다시 들썩인 F-51, 그리고 절대
 * 인덱스와 «받은 개수» 를 섞어 화면 한가운데서 신호가 돋아난 F-52.
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
   * +10 %/−45 % 밴드(폭 2.0 배)를 물린 것이 정확히 그 경우였다 (F-51).
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
   *  돋아나고**, 정작 새 표본은 오른쪽 끝에서 따로 나타난다 (F-52). */
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
  /** 보이는 구간의 **중심과 «담아야 할 반높이»**.
   *
   *  셋 다 안 된다는 것을 하나씩 봤다.
   *   - **MAD 의 몇 배**: 겹치기에서 R 피크가 화면 밖으로 밀렸다 — 잘린 파형은
   *     어느 방법이 어디서 다른지를 가린다.
   *   - **진짜 최대값**: 스파이크 한 점에 파형 전체가 납작해진다.
   *   - **99.5 백분위만**: 실측에서 진짜 최대의 **82~93 %** 다 `[측정]` —
   *     밴드 위쪽(`HI`)까지 흘러가면 R 피크 끝이 잘린다.
   *
   *  그래서 **최대값을 쓰되 백분위의 `SPIKE` 배로 묶는다.** 정상 파형은
   *  최대/백분위가 1.1~1.2 라 최대가 그대로 들어가고(안 잘린다), 스파이크는
   *  묶여서 파형을 못 눌린다. 백분위는 MAD 단위 히스토그램으로 센다 —
   *  정렬 없이 한 번 훑으면 된다. */
  var SPIKE = 1.4;

  function scan(a, lo, hi, cap, q) {
    var n = hi - lo + 1, k, m = 0, v;
    if (n <= 0) return { m: 0, peak: 0, n: 0 };
    for (k = lo; k <= hi; k++) m += a[((k % cap) + cap) % cap];
    m /= n;
    var mad = 0, mx = 0;
    for (k = lo; k <= hi; k++) {
      v = Math.abs(a[((k % cap) + cap) % cap] - m);
      mad += v; if (v > mx) mx = v;
    }
    mad /= n;
    if (!(mad > 0)) return { m: m, peak: mx, n: n };
    var NB = 96, W = 0.5, h = new Int32Array(NB + 1), b;
    for (k = lo; k <= hi; k++) {
      b = Math.abs(a[((k % cap) + cap) % cap] - m) / mad / W;
      h[b >= NB ? NB : b | 0]++;
    }
    var want = Math.ceil(n * (q == null ? 0.995 : q)), c = 0, bin = NB;
    for (k = 0; k <= NB; k++) { c += h[k]; if (c >= want) { bin = k; break; } }
    return { m: m, peak: Math.min(mx, (bin + 1) * W * mad * SPIKE), n: n };
  }

  // ------------------------------------------------- 스윕 표현 (D-41, v2.2.1 규약)
  /* **지우기 경계** — 처음에는 ECG Signal Studio v2.2.1 `engine.ts visiblePoints` 의 값
   * 그대로(커서 앞 0.12 s 비움 + 0.08 s 페이드 = 0.20 s)였다. 초로 두면 표시 길이가 바뀔 때
   * 화면에서 지워지는 비율이 달라진다 — 10 s 화면에서는 2 % 라 거의 안 보였다. 사용자가
   * **화면 폭의 고정 비율, 기본 10 %** 로 정했다 (D-42). 나누는 비는 v2.2.1 의 것(비움 3 :
   * 페이드 2)을 그대로 둔다. 5·10·15·20 % 는 화면의 디버그 버튼으로 고른다. */
  var ERASE_FRAC = 0.10, ERASE_CHOICES = [0.05, 0.10, 0.15, 0.20];
  var ERASE_GAP_PART = 0.6;

  /** 지우기 영역 [샘플] — 비움 `gap` 과 그 앞 페이드 `fade`. */
  function eraseSpans(cap, frac) {
    var f = frac == null ? ERASE_FRAC : frac, tot = Math.max(0, f) * cap;
    return { gap: tot * ERASE_GAP_PART, fade: tot * (1 - ERASE_GAP_PART) };
  }

  /** 나이 `age`(= head - 절대 인덱스, 0 이 가장 새 것)인 샘플의 투명도. 스윕 전용. */
  function sweepAlpha(age, cap, frac, soft) {
    var d = cap - age;                       // 커서 앞으로 몇 칸 떨어졌나 (1..cap)
    var e = eraseSpans(cap, frac), gap = e.gap, fade = e.fade;
    if (d > 0 && d < gap) return 0;
    if (d >= gap && d < gap + fade) return soft ? (d - gap) / fade : 0;
    return 1;
  }

  /** 스크롤의 **왼쪽 끝**(사라지는 쪽) 투명도. 칸 `i` 는 0 이 가장 왼쪽이다.
   *  스크롤에는 지울 자리가 따로 없으므로 비움 없이 **페이드만** 건다 — 폭은 스윕의
   *  지우기 영역 전체(같은 비율)라, 비율 버튼 하나로 두 방식이 같이 움직인다 (D-42). */
  function scrollAlpha(i, cap, frac, soft) {
    if (!soft) return 1;
    var w = eraseSpans(cap, frac);
    var n = w.gap + w.fade;
    return n > 0 ? Math.max(0, Math.min(1, i / n)) : 1;
  }

  /* **표시 보간 — 받은 블록을 두 번에 나눠 보인다** (D-42).
   * 브리지는 처리기가 hop 을 낼 때마다 보낸다 — 초당 약 20 번이라 스크롤이 계단으로 걷는다.
   * 처리량을 늘리면 RTF 가 무너지므로 **화면 쪽에서** 블록 하나를 반씩 두 번 보인다:
   * 도착하면 앞 절반까지, 도착 간격의 절반이 지나면 끝까지. 파형을 만들어 내는 것이 아니라
   * **이미 받은 표본을 드러내는 시점만** 나눈다. 대가는 지연 = 도착 간격의 절반(약 25 ms).
   * 사용자 요구가 «딱 두 배» 라 단계는 둘로 고정한다. */
  function makeReveal() {
    var mid = -1, target = -1, tArr = 0, dt = 50;
    return {
      push: function (h, now) {
        if (target >= 0 && h > target && h - target < 4096) {
          var gapMs = now - tArr;
          if (gapMs > 0 && gapMs < 500) dt = 0.8 * dt + 0.2 * gapMs;
          mid = target + Math.floor((h - target) / 2);
        } else mid = h;       // 첫 블록 · 시간축이 끊긴 자리는 그대로
        target = h; tArr = now;
      },
      at: function (now) {
        if (target < 0) return -1;
        return now - tArr < dt / 2 ? mid : target;
      },
      interval: function () { return dt; },
      reset: function () { mid = target = -1; tArr = 0; },
    };
  }

  /* **선단 밝기 (잔광)** — 처음에는 그 저장소 docs/22 UI-02 의 시작값(폭 2~3 % · 24~48 px,
   * 흰색 25~40 %)으로 넣었다. 사용자가 «선 바로 왼쪽과 먼 쪽에 차이가 없다» 고 했고, 재 보니
   * 파형 최대 밝기 차이가 **201 -> 219, 약 9 %** 였다 `[측정]` — 출력 색(#67e7c3)이 이미 밝아
   * 흰색을 섞어도 오를 여지가 없었다. 그래서 폭을 6 %(48~120 px), 흰색을 60 % 로 올리고,
   * 선단 쪽 선 굵기 · 같은 색의 짧은 빛 번짐 · 빛점은 화면(`live.html`)이 `glowLevel` 로 건다 (D-41 개정). */
  var GLOW_FRAC = 0.06, GLOW_MIN_PX = 48, GLOW_MAX_PX = 120, GLOW_MAX = 0.6;

  /** 선단 밝기가 걸리는 폭 [샘플]. 그림 너비 `pw`[px] 에 `cap` 샘플이 펼쳐져 있다. */
  function glowWidth(pw, cap) {
    if (!(pw > 0)) return 0;
    var px = Math.min(GLOW_MAX_PX, Math.max(GLOW_MIN_PX, GLOW_FRAC * pw));
    return Math.max(1, Math.round(px / pw * cap));
  }

  /** 나이 `age` 인 샘플에 섞을 흰색 비율 (0..GLOW_MAX).
   *
   *  **위치가 아니라 나이로 정한다.** 위치로 정하면 wrap 때 커서 근처의 옛 주기가
   *  새 표본처럼 빛난다. 그리고 **현재 주기 안만** 빛난다 — `pos`(커서가 이번 주기에서
   *  간 칸 수)보다 오래된 것은 오른쪽 끝으로 넘어간 꼬리라, 그것까지 빛나면 화면
   *  양 끝에 빛이 갈라져 보인다. */
  function glowLevel(age, width, pos) {
    if (pos == null) pos = Infinity;         // 스크롤에는 주기가 없다
    if (!(width > 0) || age < 0 || age >= width || age > pos) return 0;
    return 1 - age / width;                  // 선단 1 -> 폭 끝 0
  }

  function glowMix(age, width, pos) {
    var u = glowLevel(age, width, pos);
    return GLOW_MAX * Math.pow(u, 1.5);
  }

  /** `#rrggbb` 를 투명도 `a` 의 `rgba(...)` 로 (빛 번짐 색). */
  function rgba(hex, a) {
    var n = parseInt(hex.slice(1), 16);
    return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," + (n & 255) + ","
      + Math.max(0, Math.min(1, a)) + ")";
  }

  /** 스트림이 멈추면 장식을 정리한다 — 0.3 s 까지 그대로, 그 뒤 0.3 s 동안 0 으로. */
  function glowLive(sinceMs) {
    if (!(sinceMs > 300)) return 1;
    return Math.max(0, 1 - (sinceMs - 300) / 300);
  }

  /** `#rrggbb` 에 흰색을 `f` 만큼 섞은 `rgb(...)`. */
  function mixWhite(hex, f) {
    var n = parseInt(hex.slice(1), 16), r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
    f = Math.max(0, Math.min(1, f));
    return "rgb(" + Math.round(r + (255 - r) * f) + "," + Math.round(g + (255 - g) * f) + ","
      + Math.round(b + (255 - b) * f) + ")";
  }

  var api = {
    STEPS: STEPS, STEP_MAX: STEP_MAX, LO: LO, HI: HI, AIM: AIM, HOLD_MS: HOLD_MS,
    ladder: ladder, makeGain: makeGain, makeCenter: makeCenter,
    windowStart: windowStart, absAt: absAt, have: have, scan: scan, SPIKE: SPIKE,
    ERASE_FRAC: ERASE_FRAC, ERASE_CHOICES: ERASE_CHOICES, eraseSpans: eraseSpans,
    sweepAlpha: sweepAlpha, scrollAlpha: scrollAlpha, makeReveal: makeReveal,
    GLOW_MAX: GLOW_MAX, glowWidth: glowWidth, glowLevel: glowLevel, glowMix: glowMix,
    glowLive: glowLive, mixWhite: mixWhite, rgba: rgba,
  };
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.LiveAxis = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
