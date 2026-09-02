/* C3 카드 세 디자인이 **공유하는 것만** 담는다 — 은행 해독, 좌표 계산, 상태.
 *
 * 칠하는 방식은 테마마다 다르므로 공유하지 않는다. 공유하면 세 안이 결국
 * 같은 그림이 되고, 그러면 "여러 디자인 샘플" 이 아니게 된다.
 */
export const B = window.CARD_BANK;

export const SCENE = Object.fromEntries(B.scenes.map(s => [s.cond, s]));

/** int16 + base64 -> Float64Array. `build_card_bank.py` 의 규약과 짝이다. */
export function dec(b64, scale) {
  const bin = atob(b64), n = bin.length / 2, out = new Float64Array(n);
  const dv = new DataView(new ArrayBuffer(2));
  for (let i = 0; i < n; i++) {
    dv.setUint8(0, bin.charCodeAt(2 * i)); dv.setUint8(1, bin.charCodeAt(2 * i + 1));
    out[i] = dv.getInt16(0, true) * scale;
  }
  return out;
}

export function trace(cond, key) { const s = SCENE[cond]; return dec(s.traces[key], s.scale); }
export function psd(cond, key)   { return dec(SCENE[cond].psd[key], B.psd_scale); }

/** **화면이 그리는 방법은 셋뿐이다.** dataviz 팔레트의 앞 세 슬롯만
 *  all-pairs 검증을 통과한다(blue·orange·aqua). 네 번째를 얹으면 정상시야
 *  분리도가 무너진다 — 색을 늘리는 대신 방법을 줄인다. */
export const SHOW = ["M01", "M04", "M08"];
export const LABEL = {
  clean: "참값", input: "입력 (잡음 섞임)",
  M01: "M01 · 대역통과 + notch", M04: "M04 · SWT + QRS 보호",
  M08: "M08 · wavelet U-Net (딥러닝)",
};
/** 왜 M08 인가: 슬롯 3(aqua)이 이 저장소에서 **M08 에 고정된 색**이고
 *  (`make_slides.py`), 그리고 이 카드의 질문이 "어느 대역을 건드리나" 라
 *  wavelet subband 모델이 그 자리에 가장 맞다. */

export const HZ = B.psd_f;

/** 캔버스를 화면 배율에 맞춘다. 안 하면 레티나에서 뭉갠다. */
export function fit(c) {
  const r = window.devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
  if (c.width !== w * r || c.height !== h * r) { c.width = w * r; c.height = h * r; }
  const g = c.getContext("2d"); g.setTransform(r, 0, 0, r, 0, 0);
  g.clearRect(0, 0, w, h); return { g, w, h };
}

/** 값 -> 화면. 시간축은 조합이 스케일 하나를 공유하므로 y 범위도 하나다. */
export function mapper(w, h, n, lo, hi, pad) {
  const p = pad || { l: 0, r: 0, t: 0, b: 0 };
  const iw = w - p.l - p.r, ih = h - p.t - p.b;
  return {
    x: i => p.l + (i / (n - 1)) * iw,
    y: v => p.t + ih - ((v - lo) / (hi - lo)) * ih,
    iw, ih, p,
  };
}

export function line(g, xs, ys, arr, color, lw, alpha) {
  g.save(); g.globalAlpha = alpha == null ? 1 : alpha;
  g.beginPath(); g.strokeStyle = color; g.lineWidth = lw;
  g.lineJoin = "round"; g.lineCap = "round";
  for (let i = 0; i < arr.length; i++) {
    const px = xs(i), py = ys(arr[i]);
    i ? g.lineTo(px, py) : g.moveTo(px, py);
  }
  g.stroke(); g.restore();
}

/** 잡음이 사는 대역. **말로 하지 않고 보이려고** PSD 위에 띠로 깐다. */
export function bandOf(cond) { return SCENE[cond].band; }

/** 부드러운 전환 — 잡음을 바꿀 때 값이 튀지 않게. */
export function ease(t) { return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; }

export function animate(ms, step, done) {
  const t0 = performance.now();
  (function frame(now) {
    const t = Math.min(1, (now - t0) / ms);
    step(ease(t));
    if (t < 1) requestAnimationFrame(frame); else if (done) done();
  })(t0);
}

/** 화면이 쓰는 `psd_logdist` — **보고서와 같은 정의**여야 한다.
 *  참값 PSD 와 출력 PSD 의 로그 거리(dB 차의 평균 절댓값). 카드가 이 숫자로
 *  "SNR 은 좋은데 이 지표로는 진다" 를 말하므로, 손으로 다른 식을 쓰면 안 된다. */
export function logdist(a, b) {
  let d = 0; for (let i = 0; i < a.length; i++) d += Math.abs(a[i] - b[i]);
  return d / a.length;
}

/** 대역 평균 dB. 잡음이 사는 대역을 얼마나 눌렀는지 잴 때 쓴다. */
export function bandDb(p, f0, f1) {
  let s = 0, n = 0;
  for (let i = 0; i < HZ.length; i++) if (HZ[i] >= f0 && HZ[i] <= f1) { s += p[i]; n++; }
  return n ? s / n : NaN;
}
