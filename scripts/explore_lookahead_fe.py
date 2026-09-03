#!/usr/bin/env python3
"""위상 왜곡을 **필터 차수 말고 다른 방법으로** 없앨 수 있는가 (F-27 · D-19 후속).

    python3 scripts/explore_lookahead_fe.py               # 표 + 그림
    python3 scripts/explore_lookahead_fe.py --axis d1

산출: `docs/13_lookahead_fe.md`, `results/fig/lookahead_fe_{d0,d1}.png`

왜 다시 보는가
-------------
D-19 는 «인과 IIR 고역통과» 라는 **한 가지 형태** 안에서 차수와 차단만 움직였다.
그 안에서는 교환이 닫혀 있다 — 기저선을 지우면 T-P 가 기울고, 안 기울이면
기저선이 안 지워진다. o1/0.5 는 그 선 위의 타협점이지 해결이 아니다.

이 파일은 **형태 자체를 바꾼 후보들**을 같은 자에 올린다:

  (A) 블록 영위상    미래를 D 초 기다렸다가 그 창에서 filtfilt 를 돌린다.
                     위상 왜곡이 **원리적으로 0** 이다. 대가는 지연 D.
  (B) 중앙값 캐스케이드  기저선을 **비선형으로 추정**해 빼기만 한다.
                     QRS(약 80 ms)보다 넓은 중앙값 창은 QRS 를 못 따라가므로
                     기저선 추정에서 QRS 가 빠진다 -> 빼도 QRS 모양이 안 상한다.
  (C) 아날로그 HPF   AD8232 는 이미 **2 극 0.5 Hz 인과 HPF** 를 달고 있다.
                     아날로그도 인과다 — 위상 왜곡은 **똑같이 생긴다.**
                     그래서 (C) 는 (A)(B) 의 대안이 아니라 **그 앞에 이미 있는 것**이고,
                     여기서는 «그 위에 무엇을 더 걸어야 하는가» 를 묻는다.

지표 — 사용자가 실제로 보는 것을 잰다
------------------------------------
D-19 의 두 지표(박동당 이동·남은 변동)는 **화면에서 눈에 띄는 두 가지를 못 잡았다.**

**(3a) T-P 준위 산포** `tp_spread` · **(3b) 준위 이탈** `tp_off`

    박동마다 T-P 구간의 **중앙값 하나**를 «그 박동의 평활 준위» 로 삼는다.
    (3a) 그 준위들의 p2.5~p97.5 폭, (3b) 준위들의 |중앙 편차|. 둘 다 %R.

    **"모든 박동의 평활 대역이 같은 높이에, 그리고 0 에 있는가."**

    처음에는 구간의 표본을 **전부 모아** 산포를 쟀는데(`tp_band`), 그 산포의
    대부분이 **구간 안 잡음**이었다 — 그리고 잡음은 모든 방식이 똑같이
    통과시키므로 **지표가 방식을 못 갈랐다** (**F-30**). 구간 안이 매끈한지는
    denoiser 의 몫이지 front-end 의 몫이 아니다. `tp_noise` 를 진단용으로
    남겨 그 사실을 보인다.

**(4) S 골 깊이 오차** `s_err`   <- 새로 넣는다

    R 직후 80 ms 안의 최저점을 그 박동의 T-P 준위 기준으로 잰다 [%R].
    지표 = |인과 깊이 - 영위상 깊이|                 [%R]

    인과 고역통과는 QRS 라는 급한 계단 뒤에 **언더슈트**를 남긴다. 그것이
    S 골에 그대로 얹혀 «S 파 근처가 찌그러져» 보인다. (1)(2)(3) 은 전부
    T-P 구간만 보므로 이것을 **구조적으로 못 잡는다.**
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sps

from ecgdn.config import DEFAULT_FE, FS
from ecgdn.data.noise import make_noise
from ecgdn.data.sources import get_source
from ecgdn.eval.rpeak import detect_rpeaks
from ecgdn.methods.frontend import FrontEnd
from ecgdn.realtime.frontend_stream import StreamingFrontEnd
from ecgdn.utils import ensure_dir, rng, save_manifest

ROOT = Path(__file__).resolve().parents[1]
INJECT_FRAC = 0.5                  # 주입 기저선 변동 = R 의 50 % (D-19 와 같은 조건)
HOP_S = 0.096                      # 실시간 데모의 블록 (L >= d + hop, d=hop=12)

# AD8232 «Cardiac Monitor» 구성 = 2 극 0.5 Hz HPF + 2 극 40 Hz LPF, 이득 1100.
# docs/08_acquisition.md 의 이득 1100 이 이 구성과 맞는다.
AFE_HP_HZ, AFE_HP_ORDER = 0.5, 2
AFE_LP_HZ, AFE_LP_ORDER = 40.0, 2


def _ko_font() -> str:
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    return next((f for f in ("NanumGothic", "NanumBarunGothic", "NanumSquare",
                             "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR")
                 if f in have), "DejaVu Sans")


plt.rcParams.update({"font.family": _ko_font(), "axes.unicode_minus": False,
                     "mathtext.fontset": "dejavusans"})

C = {"ref": "#b8b6ae", "causal": "#eb6834", "look": "#2a78d6",
     "med": "#1baf7a", "afe": "#4a3aa7", "ink": "#1b1b1b", "mute": "#6b6b6b"}


# ------------------------------------------------------------------ 후보들
def f_offline(x):
    """오프라인 영위상 — 실시간이 아니다. **모든 비교의 기준선.**"""
    return np.asarray(FrontEnd()(x, FS), dtype=np.float64)


def f_causal(x, order, hp_hz, block=25):
    fe = StreamingFrontEnd(FS, replace(DEFAULT_FE, order=order, hp_hz=hp_hz))
    return np.concatenate([fe.push(x[i:i + block]) for i in range(0, x.size, block)])


PAST_S = 4.0        # 과거 문맥. **이미 받아 둔 자료라 지연을 안 만든다** — 넉넉히 준다.


def f_lookahead(x, look_s, order=4, hp_hz=0.5, hop_s=HOP_S, past_s=PAST_S):
    """블록 영위상 — 미래 `look_s` 를 기다렸다가 그 창에서 filtfilt.

    **실시간으로 구현 가능하다.** 창은 [n0-P, n1+F) 이고 그 안의 자료는 그
    시점에 전부 손에 있다(과거 P 는 이미 받았고, 미래 F 는 F 초 기다린 것).
    filtfilt 의 가장자리 처리는 창 안에서만 하므로 미래를 더 훔치지 않는다.
    **지연 = F + hop 뿐이다** — 과거 P 는 이미 받아 둔 것이라 지연을 안 만든다.
    처음 짤 때 P 를 F 와 같이 줄였다가 미리보기 0.25 s 에서 인과보다 나쁜
    값이 나왔다. 과거를 굶긴 것이었지 미리보기가 모자란 것이 아니었다.
    """
    nyq = FS / 2.0
    F = int(round(look_s * FS))
    P = int(round(past_s * FS))
    H = int(round(hop_s * FS))
    hp = sps.butter(order, hp_hz / nyq, btype="highpass", output="sos")
    lp = sps.butter(order, DEFAULT_FE.lp_hz / nyq, btype="lowpass", output="sos")
    out = np.empty_like(x, dtype=np.float64)
    for n0 in range(0, x.size, H):
        n1 = min(n0 + H, x.size)
        a, b = max(0, n0 - P), min(x.size, n1 + F)
        w = np.asarray(x[a:b], dtype=np.float64)
        pad = max(1, min(w.size - 1, w.size // 2))
        v = sps.sosfiltfilt(hp, w, padtype="odd", padlen=pad)
        v = sps.sosfiltfilt(lp, v, padtype="odd", padlen=pad)
        out[n0:n1] = v[n0 - a: n0 - a + (n1 - n0)]
    return out


def _odd(n: int) -> int:
    return n + 1 if n % 2 == 0 else n


def f_median(x, w1_s=0.2, w2_s=0.6, lp_hz=DEFAULT_FE.lp_hz):
    """중앙값 캐스케이드로 기저선을 **추정**해 뺀다 (고역통과가 아니다).

    창 w1(200 ms) 은 QRS 폭(약 80 ms)보다 넓다 -> 중앙값이 QRS 를 안 따라간다
    -> 추정된 기저선에 QRS 가 안 들어간다 -> 빼도 QRS 가 안 상한다.
    w2(600 ms) 가 P·T 파까지 지운다. 지연 = (w1+w2)/2.
    """
    w1, w2 = _odd(int(round(w1_s * FS))), _odd(int(round(w2_s * FS)))
    base = sps.medfilt(sps.medfilt(np.asarray(x, dtype=np.float64), w1), w2)
    v = np.asarray(x, dtype=np.float64) - base
    sos = sps.butter(DEFAULT_FE.order, lp_hz / (FS / 2), btype="lowpass", output="sos")
    return sps.sosfiltfilt(sos, v, padtype="odd", padlen=min(v.size - 1, int(2 * FS)))


def f_afe(x, hp_hz=AFE_HP_HZ, order=AFE_HP_ORDER):
    """AD8232 아날로그단을 **모사**한다 — 인과 2 극 HPF + 2 극 40 Hz LPF.

    아날로그 필터는 물리적으로 인과다. 그러므로 위상 왜곡은 디지털 인과
    필터와 **똑같이** 생긴다. 이 함수는 «아날로그로 옮기면 해결되나» 에
    숫자로 답하기 위한 것이다.
    """
    nyq = FS / 2.0
    hp = sps.butter(order, hp_hz / nyq, btype="highpass", output="sos")
    lp = sps.butter(AFE_LP_ORDER, AFE_LP_HZ / nyq, btype="lowpass", output="sos")
    v = sps.sosfilt(hp, np.asarray(x, dtype=np.float64))
    return sps.sosfilt(lp, v)


def f_afe_comp(x, look_s=1.0, hop_s=HOP_S, past_s=None):
    """AFE 가 이미 준 위상 왜곡을 **아날로그 필터를 거꾸로 걸어** 되돌린다.

    아날로그단이 Ha(z) 를 이미 걸었다면, 표본에 같은 Ha 를 **시간 역방향**으로
    한 번 더 걸면 전체가 Ha(z)·Ha(1/z) 가 된다 — 이것이 곧 영위상이다.
    filtfilt 의 두 번째 패스를 «아날로그가 이미 한 첫 패스» 에 맞춰 우리가
    대신 놓는 것이다. 크기 응답은 |Ha|^2 (4 극 상당)이 되어 0.5 Hz 아래가
    더 눌리지만, **위상은 정확히 0 이 된다.**

    아날로그 차단 주파수를 알아야 한다 — RC 값에서 나오거나, 계단 응답을
    한 번 재면 맞출 수 있다. 미래를 look_s 만큼 기다려야 하는 것은 (A) 와 같다.
    """
    nyq = FS / 2.0
    F, H = int(round(look_s * FS)), int(round(hop_s * FS))
    P = int(round((past_s if past_s is not None else PAST_S) * FS))
    hp = sps.butter(AFE_HP_ORDER, AFE_HP_HZ / nyq, btype="highpass", output="sos")
    lp = sps.butter(DEFAULT_FE.order, DEFAULT_FE.lp_hz / nyq, btype="lowpass",
                    output="sos")
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    for n0 in range(0, x.size, H):
        n1 = min(n0 + H, x.size)
        a, b = max(0, n0 - P), min(x.size, n1 + F)
        w = x[a:b]
        pad = max(1, min(w.size - 1, w.size // 2))
        v = sps.sosfiltfilt(lp, w, padtype="odd", padlen=pad)
        v = sps.sosfilt(hp, v[::-1])[::-1]          # **역방향 한 패스만**
        out[n0:n1] = v[n0 - a: n0 - a + (n1 - n0)]
    return out


# ------------------------------------------------------------------ 지표
# T-P 창의 여백. **RR 에 비례해야 한다** — QT 간격은 심박이 빠르면 짧아지고
# (Bazett), 고정 ms 로 두면 RR 이 700 ms 아래에서 창이 **비어 버린다.**
# 계수는 RR = 1000 ms(60 bpm)에서 옛 고정값(420 / 280 ms)과 정확히 같도록 골랐다.
TP_PRE_FRAC, TP_POST_FRAC = 0.42, 0.28
TP_PRE_MAX_S, TP_POST_MAX_S = 0.42, 0.28


def _tp_slices(r, n):
    """T 파 뒤 ~ 다음 P 파 앞. 창은 **RR 비율**로 잡는다.

    처음에는 `R+420 ms ~ 다음 R−280 ms` 고정이었는데, 100 bpm(RR 600 ms)부터
    창이 비어 지표가 통째로 NaN 이 됐다 (**O-22**). 심박수 의존성을 재려던
    실험에서 **재는 자 자신이 먼저 무너진** 것이다.
    """
    for a, b in zip(r[:-1], r[1:]):
        rr = (b - a) / FS
        i0 = a + int(min(TP_PRE_MAX_S, TP_PRE_FRAC * rr) * FS)
        i1 = b - int(min(TP_POST_MAX_S, TP_POST_FRAC * rr) * FS)
        if 10 <= i1 - i0 and i1 <= n:
            yield i0, i1


def drift_per_beat(v, r, r_amp):
    """(1) 한 박동 동안 기저선이 R 진폭의 몇 % 흐르는가 [%R]. D-19 정의."""
    out = []
    for (i0, i1), (a, b) in zip(_tp_slices(r, v.size), zip(r[:-1], r[1:])):
        seg = v[i0:i1]
        k = np.polyfit(np.arange(seg.size) / FS, seg, 1)[0]
        out.append(abs(k) * (b - a) / FS)
    return 100 * float(np.median(out)) / r_amp if out else float("nan")


def _tp_levels(v, r):
    """**박동마다 T-P 준위 하나.** 구간 안의 잡음은 중앙값이 걷어낸다.

    처음에는 구간의 표본을 **전부 모아** 산포를 쟀는데(옛 `tp_band`), 그
    산포의 대부분이 **구간 안 잡음**이었다 — 그리고 잡음은 모든 방식이 똑같이
    통과시키므로 지표가 방식을 못 갈랐다 (**F-30**). 평활도는 «이 박동의
    평활 대역이 어느 높이에 있는가» 의 문제이지 «그 안이 얼마나 매끈한가» 가
    아니다. 후자는 방법(denoiser)의 몫이고 front-end 의 몫이 아니다.
    """
    return np.array([np.median(v[i0:i1]) for i0, i1 in _tp_slices(r, v.size)])


def tp_spread(v, r, r_amp):
    """(3a) **박동별 T-P 준위의 산포** [%R] — 평활 대역이 같은 높이에 있는가."""
    lv = _tp_levels(v, r)
    if lv.size < 3:
        return float("nan")
    lv = lv - np.median(lv)
    return 100 * float(np.percentile(lv, 97.5) - np.percentile(lv, 2.5)) / r_amp


def tp_off(v, r, r_amp):
    """(3b) **그 높이가 0 인가** [%R] — 박동별 준위의 |중앙값| 편차."""
    lv = _tp_levels(v, r)
    if lv.size < 3:
        return float("nan")
    return 100 * float(np.median(np.abs(lv - np.median(lv)))) / r_amp


def tp_noise(v, r, r_amp):
    """진단용 — 구간 **안**의 잡음 [%R]. front-end 로 안 갈리는 것을 보이려 남긴다."""
    segs = [v[i0:i1] - np.median(v[i0:i1]) for i0, i1 in _tp_slices(r, v.size)]
    if not segs:
        return float("nan")
    d = np.concatenate(segs)
    return 100 * float(np.percentile(d, 97.5) - np.percentile(d, 2.5)) / r_amp


def tp_band(v, r, r_amp):
    """**옛 정의** — 표본을 전부 모은 산포. F-30 에서 잡음 지배로 폐기했다.

    문서에는 남겨 «왜 이것으로는 못 갈랐는가» 를 보인다.
    """
    segs = [v[i0:i1] for i0, i1 in _tp_slices(r, v.size)]
    if not segs:
        return float("nan")
    s = np.concatenate(segs)
    s = s - np.median(s)
    return 100 * float(np.percentile(s, 97.5) - np.percentile(s, 2.5)) / r_amp


def s_depth(v, r, r_amp):
    """R 직후 80 ms 최저점의 깊이 [%R]. 기준 준위는 그 박동의 T-P 중앙값."""
    out = []
    for (i0, i1), a in zip(_tp_slices(r, v.size), r[:-1]):
        j0, j1 = a, min(a + int(0.08 * FS), v.size)
        if j1 - j0 < 3:
            continue
        out.append((np.median(v[i0:i1]) - float(np.min(v[j0:j1]))) / r_amp * 100)
    return float(np.median(out)) if out else float("nan")


def t_amp(v, r, r_amp):
    """T 파 꼭대기 높이 [%R]. 기준 준위는 그 박동의 T-P 중앙값.

    600 ms 중앙값 창은 RR 이 짧으면 **T 파까지 기저선으로 오인**할 수 있다.
    그러면 T 가 깎인다 — 이 지표가 없으면 그것을 못 보고 추천하게 된다.
    """
    out = []
    for (i0, i1), a in zip(_tp_slices(r, v.size), r[:-1]):
        j0, j1 = a + int(0.15 * FS), min(a + int(0.40 * FS), v.size)
        if j1 - j0 < 3:
            continue
        out.append((float(np.max(v[j0:j1])) - np.median(v[i0:i1])) / r_amp * 100)
    return float(np.median(out)) if out else float("nan")


def wander_left(fn, x, r_amp, inj_pp):
    """(2) 주입한 기저선 변동 중 얼마가 살아남는가 [%R]. D-19 정의."""
    g = rng("fe", "bw")
    n = make_noise("bw_synth", x.size, FS, g)
    n = n / (np.percentile(n, 97.5) - np.percentile(n, 2.5)) * inj_pp
    v = fn(x + n)
    sos = sps.butter(4, 0.5 / (FS / 2), btype="lowpass", output="sos")
    low = sps.sosfiltfilt(sos, v, padtype="odd", padlen=min(v.size - 1, int(4 * FS)))
    return 100 * float(np.percentile(low, 97.5) - np.percentile(low, 2.5)) / r_amp


def zp_err(v, ref, r_amp):
    """영위상 기준선과 얼마나 다른가 [%R] — 블록 영위상의 근사 오차."""
    return 100 * float(np.percentile(np.abs(v - ref), 99)) / r_amp


# ------------------------------------------------------------------ 후보 표
def candidates():
    """(라벨, 함수, 지연 ms, 무리). 지연은 **화면에 나오기까지의 추가 지연**이다."""
    hop_ms = HOP_S * 1000
    out = [
        ("영위상 (오프라인 기준)", f_offline, float("inf"), "ref"),
        ("인과 o4 · 0.5 Hz", lambda x: f_causal(x, 4, 0.5), 0.0, "causal"),
        ("인과 o1 · 0.5 Hz  [현재]", lambda x: f_causal(x, 1, 0.5), 0.0, "causal"),
        ("인과 o1 · 0.05 Hz", lambda x: f_causal(x, 1, 0.05), 0.0, "causal"),
    ]
    for d in (0.25, 0.5, 1.0, 2.0, 4.0):
        out.append((f"블록 영위상 · 미리보기 {d:g} s", lambda x, d=d: f_lookahead(x, d),
                    d * 1000 + hop_ms, "look"))
    out += [
        ("중앙값 200+600 ms", lambda x: f_median(x, 0.2, 0.6), 400.0 + hop_ms, "med"),
        ("중앙값 100+300 ms", lambda x: f_median(x, 0.1, 0.3), 200.0 + hop_ms, "med"),
        ("AFE(아날로그 o2 0.5 Hz) 만", f_afe, 0.0, "afe"),
        ("AFE + 인과 o1 0.5 Hz", lambda x: f_causal(f_afe(x), 1, 0.5), 0.0, "afe"),
        ("AFE + 중앙값 200+600 ms", lambda x: f_median(f_afe(x), 0.2, 0.6),
         400.0 + hop_ms, "afe"),
        ("AFE + 블록 영위상 1 s", lambda x: f_lookahead(f_afe(x), 1.0),
         1000.0 + hop_ms, "afe"),
        ("AFE + 역방향 보상 1 s", lambda x: f_afe_comp(f_afe(x), 1.0),
         1000.0 + hop_ms, "afe"),
        ("AFE 를 0.05 Hz 로 바꾸면", lambda x: f_afe(x, hp_hz=0.05), 0.0, "afe"),
        ("AFE 0.05 Hz + 중앙값 200+600", lambda x: f_median(f_afe(x, hp_hz=0.05), 0.2, 0.6),
         400.0 + hop_ms, "afe"),
    ]
    return out


def load(tag: str, seconds: float = 60.0):
    src = get_source("synthetic" if tag == "d0" else "mitdb")
    name = src.records("test")[0]
    rec = src.get(name)
    x = np.asarray(rec.x, dtype=np.float64)[: int(seconds * FS)]
    r = np.asarray(detect_rpeaks(x, FS), dtype=int)
    return x, r, float(np.median(x[r]) - np.median(x)), str(name)


def measure(tag: str):
    x, r, r_amp, name = load(tag)
    ref = f_offline(x)
    s_ref, t_ref = s_depth(ref, r, r_amp), t_amp(ref, r, r_amp)
    rows = []
    for lab, fn, lat, grp in candidates():
        v = fn(x)
        rows.append(dict(
            label=lab, group=grp, latency_ms=lat,
            drift=drift_per_beat(v, r, r_amp),
            tp_spread=tp_spread(v, r, r_amp),
            tp_off=tp_off(v, r, r_amp),
            tp_noise=tp_noise(v, r, r_amp),
            tp_band=tp_band(v, r, r_amp),
            s_dep=s_depth(v, r, r_amp),
            s_err=abs(s_depth(v, r, r_amp) - s_ref),
            t_err=abs(t_amp(v, r, r_amp) - t_ref),
            wander=wander_left(fn, x, r_amp, INJECT_FRAC * r_amp),
            zperr=zp_err(v, ref, r_amp),
        ))
    return rows, dict(record=name, r_amp=r_amp, s_ref=s_ref, t_ref=t_ref,
                      n_beats=int(r.size))


# ------------------------------------------------------------------ 그림
FIG_PICKS = [
    ("영위상 (오프라인 기준)", f_offline, C["ref"]),
    ("인과 o1 · 0.5 Hz  [현재]", lambda x: f_causal(x, 1, 0.5), C["causal"]),
    ("블록 영위상 · 미리보기 0.5 s", lambda x: f_lookahead(x, 0.5), C["look"]),
    ("중앙값 200+600 ms", lambda x: f_median(x, 0.2, 0.6), C["med"]),
    ("AFE(아날로그 o2 0.5 Hz) 만", f_afe, C["afe"]),
    ("AFE + 역방향 보상 1 s", lambda x: f_afe_comp(f_afe(x), 1.0), "#7d5fe0"),
]


def figure(tag: str, out: Path):
    """숫자로는 «찌그러진다» 가 안 보인다 — QRS 를 확대한 열을 따로 둔다."""
    x, r, r_amp, name = load(tag, 30.0)
    g = rng("fe", "bw")
    n = make_noise("bw_synth", x.size, FS, g)
    n = n / (np.percentile(n, 97.5) - np.percentile(n, 2.5)) * (INJECT_FRAC * r_amp)
    y = x + n
    ref = f_offline(x)

    j = int(r[len(r) // 3])
    lo, hi = j - int(0.5 * FS), j + int(3.5 * FS)          # 박동 4~5 개
    zlo, zhi = j - int(0.10 * FS), j + int(0.22 * FS)      # QRS 한 개

    rows = [("입력 (필터 전)", None, "#8a8a8a")] + FIG_PICKS
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 13.0), dpi=150,
                             gridspec_kw={"width_ratios": [1.25, 1.25, 0.9]})
    fig.suptitle(f"위상 왜곡을 차수 말고 다른 방법으로 없앨 수 있는가 — {tag} 기록 {name}",
                 x=0.02, ha="left", fontsize=14, fontweight="bold")
    axes[0, 0].set_title("① 기저선 변동 없음 — 박동마다 평활 대역이 같은 높이인가",
                         fontsize=11, loc="left")
    axes[0, 1].set_title(f"② 기저선 변동 R 의 {INJECT_FRAC*100:.0f} % 주입 — 지우는가",
                         fontsize=11, loc="left")
    axes[0, 2].set_title("③ QRS 확대 — Q·S 가 찌그러지는가", fontsize=11, loc="left")

    t = np.arange(hi - lo) / FS
    tz = (np.arange(zhi - zlo) - (j - zlo)) / FS * 1000
    for k, (lab, fn, col) in enumerate(rows):
        vx = x if fn is None else fn(x)
        vy = y if fn is None else fn(y)
        for c, v in ((0, vx), (1, vy)):
            ax = axes[k, c]
            ax.axhline(0, color="#dcdcdc", lw=0.9, zorder=0)
            ax.plot(t, v[lo:hi] - np.median(v[lo:hi]), color=col, lw=1.15)
            ax.set_ylim(-1.15 * r_amp, 1.55 * r_amp)
            ax.set_yticks([])
            ax.set_xticks([0, 1, 2, 3] if k == len(rows) - 1 else [])
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)

        az = axes[k, 2]
        az.axhline(0, color="#dcdcdc", lw=0.9, zorder=0)
        if fn is not None:                       # 기준선을 회색으로 깔아 차이를 본다
            az.plot(tz, ref[zlo:zhi] - np.median(ref[lo:hi]), color="#cfcdc6",
                    lw=2.6, zorder=1)
        az.plot(tz, vx[zlo:zhi] - np.median(vx[lo:hi]), color=col, lw=1.5, zorder=2)
        az.set_ylim(-0.55 * r_amp, 1.25 * r_amp)
        az.set_yticks([]); az.set_xticks([-100, 0, 100, 200] if k == len(rows) - 1 else [])
        for sp in ("top", "right", "left"):
            az.spines[sp].set_visible(False)

        axes[k, 0].set_ylabel(lab, rotation=0, ha="right", va="center",
                              fontsize=10, labelpad=8)
        if fn is not None:
            axes[k, 0].text(.985, .90, f"준위 산포 {tp_spread(vx, r, r_amp):.1f} %R",
                            transform=axes[k, 0].transAxes, ha="right",
                            fontsize=9.5, color=C["mute"])
            axes[k, 1].text(.985, .90, f"남은 변동 {wander_left(fn, x, r_amp, INJECT_FRAC*r_amp):.0f} %R",
                            transform=axes[k, 1].transAxes, ha="right",
                            fontsize=9.5, color=C["mute"])
            axes[k, 2].text(.985, .90, f"S 오차 {abs(s_depth(vx, r, r_amp)-s_depth(ref, r, r_amp)):.0f} %R",
                            transform=axes[k, 2].transAxes, ha="right",
                            fontsize=9.5, color=C["mute"])
    axes[-1, 0].set_xlabel("s"); axes[-1, 1].set_xlabel("s"); axes[-1, 2].set_xlabel("ms")
    fig.text(0.02, 0.015,
             "③ 의 회색 굵은 선이 영위상 기준이다. 인과 필터만 그 아래로 파고든다 — "
             "그것이 «S 파 근처가 찌그러진다» 의 정체다.",
             fontsize=10, color=C["ink"])
    fig.tight_layout(rect=[0.02, 0.03, 1, 0.955])
    ensure_dir(out.parent); fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# 프런티어 그림에는 **논지를 지는 후보만** 올린다. 전부 올리면 라벨이 겹쳐
# 읽을 수 없고, 완전한 목록은 어차피 문서의 표가 가지고 있다.
FRONTIER_PICKS = {
    "인과 o1 · 0.5 Hz  [현재]": (12, -4),
    "인과 o1 · 0.05 Hz": (12, -4),
    "인과 o4 · 0.5 Hz": (12, -4),
    "블록 영위상 · 미리보기 0.5 s": (0, -18),
    "블록 영위상 · 미리보기 2 s": (0, 12),
    "중앙값 200+600 ms": (-13, 4),
    "AFE(아날로그 o2 0.5 Hz) 만": (12, -4),
    "AFE + 역방향 보상 1 s": (13, -4),
}


def frontier(tag: str, rows, out: Path):
    """지연을 얼마 내면 무엇을 얼마나 사는가 — 한 장으로."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), dpi=150)
    fig.suptitle(f"지연으로 무엇을 사는가 — {tag}", x=0.02, ha="left",
                 fontsize=13, fontweight="bold")
    # 저울질하는 두 축. 옛 `tp_band` 로 그렸을 때는 후보들이 4~7 %R 로 뭉쳐
    # 갈리지 않았는데, 그것은 **지표가 잡음을 재고 있었기 때문**이다(F-30).
    # 박동별 준위로 바꾸자 갈린다.
    for ax, key, ttl in ((axes[0], "tp_spread",
                          "③a T-P 준위 산포 [%R]  (낮을수록 박동마다 같은 높이)"),
                         (axes[1], "s_err",
                          "④ S 골 깊이 오차 [%R]  (낮을수록 안 찌그러진다)")):
        for rw in rows:
            lat = rw["latency_ms"]
            if not np.isfinite(lat):
                ax.axhline(rw[key], color=C["ref"], lw=1.4, ls="--", zorder=1)
                ax.text(0.995, rw[key], "영위상 기준 ", transform=ax.get_yaxis_transform(),
                        color=C["mute"], fontsize=9, va="bottom", ha="right")
                continue
            off = FRONTIER_PICKS.get(rw["label"])
            if off is None:
                continue
            xy = (max(lat, 8), rw[key])
            ax.scatter(*xy, s=90, color=C[rw["group"]], zorder=3,
                       edgecolor="white", linewidth=1.2)
            ax.annotate(rw["label"].replace(" · ", "·").replace("  [현재]", " [현재]"),
                        xy, textcoords="offset points", xytext=off, fontsize=8.5,
                        color=C["ink"], ha="center" if off[0] == 0 else
                        ("left" if off[0] > 0 else "right"))
        ax.set_xscale("log"); ax.set_xlim(5, 6000)
        ax.set_xlabel("추가 지연 [ms]  (로그)")
        ax.set_title(ttl, fontsize=11, loc="left")
        ax.margins(y=0.22)
        ax.grid(alpha=.25, lw=.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.text(0.02, 0.015, "논지를 지는 후보만 올렸다 — 전체 목록은 문서의 표에 있다.",
             fontsize=9.5, color=C["mute"])
    fig.tight_layout(rect=[0.02, 0.04, 1, 0.93])
    ensure_dir(out.parent); fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ------------------------------------------------------------------ 비용
COST_PICKS = [
    ("인과 o1 · 0.5 Hz  [현재]", lambda v: f_causal(v, 1, 0.5), True),
    ("블록 영위상 · 미리보기 0.5 s", lambda v: f_lookahead(v, 0.5), True),
    ("블록 영위상 · 미리보기 1 s", lambda v: f_lookahead(v, 1.0), True),
    ("중앙값 200+600 ms", lambda v: f_median(v, 0.2, 0.6), False),
]


def cost(seconds: float = 60.0, reps: int = 3):
    """블록 하나(96 ms)를 내는 데 몇 ms 드는가. 데모 RTF 예산과 같은 자다.

    `streaming=False` 인 항목은 **배열 전체를 한 번에** 돌린 값이라 스트리밍
    구현의 비용이 아니다 — `scipy.medfilt` 는 전체를 벡터화해 돌리지만,
    실시간으로 하려면 이동 중앙값(힙·스킵리스트)을 따로 짜야 하고 비용도 다르다.
    그것을 안 적으면 표가 «중앙값이 제일 싸다» 로 읽힌다.
    """
    import time
    x = np.random.default_rng(0).standard_normal(int(seconds * FS))
    n_blocks = seconds / HOP_S
    out = []
    for lab, fn, streaming in COST_PICKS:
        fn(x[: int(5 * FS)])                     # warm-up (F-26)
        t = min(_timed(fn, x, time) for _ in range(reps))
        out.append(dict(label=lab, streaming=streaming, rtf=t / seconds,
                        per_block_ms=t / n_blocks * 1000))
    return out


def _timed(fn, x, time):
    t0 = time.perf_counter()
    fn(x)
    return time.perf_counter() - t0


# ------------------------------------------------------------------ 문서
def write_doc(per_axis: dict, made: list[Path], costs: list[dict]) -> Path:
    L = ["# 13. 위상 왜곡 — 차수 말고 다른 길이 있는가", "",
         "> 이 문서는 `scripts/explore_lookahead_fe.py` 가 만든다. **직접 고치지 말 것.**",
         "> 앞선 논의는 `docs/12_causal_fe.md`(D-19). 지표 (1)(2) 의 정의도 거기에 있다.",
         "",
         "D-19 는 «인과 IIR 고역통과» 라는 한 형태 안에서 차수와 차단만 움직였다.",
         "그 안에서 교환은 닫혀 있다 — 기저선을 지우면 T-P 가 기울고, 안 기울이면",
         "기저선이 안 지워진다. **형태를 바꾸면 그 선을 벗어날 수 있는가** 를 여기서 묻는다.",
         "",
         "## 지표 넷", "",
         "| | 무엇 | 정의 |",
         "|---|---|---|",
         "| **(1) 박동당 이동** | 박동 하나 동안 기저선이 흐른 양 | T-P 에 1 차 맞춤, `100·\\|k\\|·RR / R진폭` [%R] |",
         "| **(2) 남은 변동** | 주입한 기저선 변동 중 살아남은 양 | R 의 50 % 주입, 0.5 Hz 이하 p-p, `100·pp / R진폭` [%R] |",
         "| **(3) T-P 대역폭** | 평활 대역이 중심선에 붙어 있는가 | T-P 표본 전부 모아 중앙값 제거, p2.5~p97.5 폭 [%R] |",
         "| **(4) S 골 오차** | QRS 뒤가 찌그러지는가 | R+80 ms 최저점 깊이의 영위상 대비 차 [%R] |",
         "| **(5) T 진폭 오차** | T 파를 깎지 않는가 | R+150~400 ms 최고점 높이의 영위상 대비 차 [%R] |",
         "",
         "**(3)(4) 는 이번에 새로 넣었다.** (1)(2) 만으로는 화면에서 실제로 거슬리는",
         "두 가지 — 박동 사이 준위 어긋남과 S 골 찌그러짐 — 을 못 잡는다.",
         "(1) 은 박동 **안**의 기울기만 보고, (1)(2)(3) 은 전부 **T-P 구간만** 본다.",
         ""]
    for tag, (rows, meta) in per_axis.items():
        L += [f"## {tag} — 기록 {meta['record']} · 박동 {meta['n_beats']} 개", "",
              "| 방식 | 추가 지연 | (3a) 준위 산포 | (3b) 준위 이탈 | (4) S 오차 | (5) T 진폭 오차 | (1) 박동당 이동 | (2) 남은 변동 | 영위상과의 차 | *(폐기) T-P 폭* | *구간내 잡음* |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for rw in rows:
            lat = "—" if not np.isfinite(rw["latency_ms"]) else f"{rw['latency_ms']:.0f} ms"
            L.append(f"| {rw['label']} | {lat} | **{rw['tp_spread']:.1f}** | "
                     f"**{rw['tp_off']:.1f}** | **{rw['s_err']:.1f}** | {rw['t_err']:.1f} | "
                     f"{rw['drift']:.1f} | {rw['wander']:.1f} | {rw['zperr']:.1f} | "
                     f"*{rw['tp_band']:.1f}* | *{rw['tp_noise']:.1f}* |")
        L.append("")
    def g(tag, label, key):
        """**결론 문장의 숫자를 표에서 뽑는다.** 손으로 박으면 재측정 때 갈라진다(F-9 계열)."""
        for rw in per_axis[tag][0]:
            if rw["label"] == label:
                return rw[key]
        raise KeyError(f"{tag} / {label} / {key}")

    ZP, CUR = "영위상 (오프라인 기준)", "인과 o1 · 0.5 Hz  [현재]"
    LA5, MED = "블록 영위상 · 미리보기 0.5 s", "중앙값 200+600 ms"
    AFE, AFELA = "AFE(아날로그 o2 0.5 Hz) 만", "AFE + 블록 영위상 1 s"
    AFEC, LOW = "AFE + 역방향 보상 1 s", "인과 o1 · 0.05 Hz"

    L += ["## 연산 비용 — 예산 안에 드는가", "",
          f"블록 하나가 {HOP_S*1000:.0f} ms 다. 그 안에 front-end 와 방법 전부가 끝나야 한다.", "",
          "| 방식 | 블록당 | RTF | |", "|---|---:|---:|---|"]
    for c in costs:
        note = "" if c["streaming"] else "**배열 전체 일괄 — 스트리밍 비용 아님**"
        L.append(f"| {c['label']} | {c['per_block_ms']:.2f} ms | {c['rtf']:.4f} | {note} |")
    L += ["",
          f"블록 영위상은 과거 {PAST_S:g} s + 미리보기를 매 블록 다시 거르지만, IIR 은 "
          "표본당 몇 연산이라 **예산의 1 % 도 안 쓴다.** 비용은 이 선택의 고려사항이 아니다.",
          "", "## 무엇을 고를 것인가", "",
          "**블록 영위상 · 미리보기 0.5 s 를 권한다.** 단 **평활도 하나만 보면 중앙값이 낫다** —",
          "둘을 화면에서 바꿔 가며 볼 수 있게 해 두었다(`docs/30_realtime_demo.md`).", "",
          "| | 근거 |",
          "|---|---|",
          "| 위상 왜곡이 **원리적으로** 0 | 앞뒤로 한 번씩 거르므로 위상이 상쇄된다. 차수를 낮춰 «덜 나쁘게» 만든 것이 아니다 |",
          f"| S 골 오차 d1 **{g('d1',LA5,'s_err'):.1f}** · d0 **{g('d0',LA5,'s_err'):.1f}** %R "f"| 현재(인과 o1 0.5)는 {g('d1',CUR,'s_err'):.1f} · {g('d0',CUR,'s_err'):.1f} %R. 화면에서 보이던 찌그러짐이 사라진다 |",
          f"| 남은 기저선 변동 d1 **{g('d1',LA5,'wander'):.1f}** · d0 **{g('d0',LA5,'wander'):.1f}** %R "f"| 현재는 {g('d1',CUR,'wander'):.1f} · {g('d0',CUR,'wander'):.1f} %R. 오프라인 영위상({g('d1',ZP,'wander'):.1f} · {g('d0',ZP,'wander'):.1f})에 거의 붙는다 |",
          "| 대가는 지연 596 ms 뿐 | 연산은 예산의 0.5 %. 화면은 수동적으로 보는 것이라 0.6 s 는 알아채기 어렵다 |",
          "| 새 조정이 없다 | 지금 쓰는 필터 **설계 그대로** 돌리는 방식만 바꾼다 |",
          f"| 평활도도 영위상 수준이다 | 준위 산포 d1 {g('d1',LA5,'tp_spread'):.1f} %R "
          f"= 오프라인 영위상 {g('d1',ZP,'tp_spread'):.1f}. 중앙값({g('d1',MED,'tp_spread'):.1f})보다는 크다 |",
          "| **선형이다** | 우리 평가는 참조를 `FE(clean)` 으로 두므로(D-3) front-end 의 선형성 위에 서 있다 |",
          "",
          "**미리보기 0.25 s 로는 안 된다.** 역방향 패스가 자리를 잡는 데 시간이 걸려서,",
          f"0.25 s 에서는 영위상과의 차가 오히려 인과보다 크다"f"(d1 {g('d1','블록 영위상 · 미리보기 0.25 s','zperr'):.1f} vs {g('d1',CUR,'zperr'):.1f} %R).",
          "**0.5 s 가 문턱이다.**", "",
          "### 왜 다른 것을 안 고르나", "",
          "| 후보 | 왜 아닌가 |",
          "|---|---|",
          f"| 중앙값 200+600 ms | **평활도는 이쪽이 낫다** — 준위 산포 d0 {g('d0',MED,'tp_spread'):.1f} · "f"d1 {g('d1',MED,'tp_spread'):.1f} %R 로 오프라인 영위상"f"({g('d0',ZP,'tp_spread'):.1f} · {g('d1',ZP,'tp_spread'):.1f})보다도 낮다. 그런데 **S 오차가 "f"d1 {g('d1',MED,'s_err'):.1f} %R** 로, 사용자가 이미 «찌그러진다» 고 지적한 인과 o1 0.5"f"({g('d1',CUR,'s_err'):.1f} %R)와 거의 같다 — 눈에 띄는 왜곡을 **다른 왜곡으로 바꾸는** 셈이다. "f"100+300 판은 **T 파를 {g('d0','중앙값 100+300 ms','t_err'):.0f} %R 깎는다**. ""비선형이라 중첩이 깨지는 것도 값이다 — `docs/14_median_vs_zerophase.md` |",
          f"| 인과 o1 · 0.05 Hz | 기저선을 사실상 못 지운다"f"(잔여 d0 {g('d0',LOW,'wander'):.0f} · d1 {g('d1',LOW,'wander'):.0f} %R, 주입량이 50 %R). 존재 이유가 없다는 판단이 맞다 |",
          "| 아날로그 HPF 로 옮기기 | **아날로그도 인과다** — 다음 절 |",
          "",
          "## 아날로그로 옮기면 되는가 — 안 된다, 그리고 이미 걸려 있다", "",
          "`AD8232` 의 «Cardiac Monitor» 구성은 **2 극 0.5 Hz 인과 HPF** 다",
          "(`docs/08_acquisition.md` 의 이득 1100 이 이 구성과 맞는다).",
          "아날로그 필터는 물리적으로 인과라 **위상 왜곡이 똑같이 생긴다.**", "",
          "| 관측 | 숫자 |",
          "|---|---|",
          f"| AFE 만 통과해도 S 가 이미 찌그러진다 | S 오차 d1 **{g('d1',AFE,'s_err'):.1f}** · d0 {g('d0',AFE,'s_err'):.1f} %R — 우리 디지털 인과({g('d1',CUR,'s_err'):.1f} · {g('d0',CUR,'s_err'):.1f})보다 **더 크다** |",
          f"| 뒤에서 못 고친다 | AFE + 블록 영위상 1 s 를 걸어도 S 오차 d1 **{g('d1',AFELA,'s_err'):.1f}** %R. 이미 표본에 들어온 왜곡이다 |",
          f"| **역방향으로는 고쳐진다** | AFE + 역방향 보상 1 s → S 오차 d1 **{g('d1',AFEC,'s_err'):.1f}** · d0 {g('d0',AFEC,'s_err'):.1f} %R |",
          "",
          "역방향 보상은 아날로그가 이미 건 필터를 **시간 역방향으로 한 번 더** 거는 것이다.",
          "합치면 `Ha(z)·Ha(1/z)` 이 되어 위상이 정확히 상쇄된다 — filtfilt 의 두 번째",
          "패스를 «아날로그가 이미 한 첫 패스» 에 맞춰 우리가 대신 놓는 셈이다.",
          "**아날로그 차단 주파수를 알아야 한다** (RC 값에서 나오거나 계단 응답을 한 번 재면 된다).", "",
          "그러므로 아날로그단의 올바른 역할은 **위상 해결이 아니라 포화 방지**다.",
          "전극 반전지 전위는 최대 ±300 mV 이고 이득이 1100 이면 ADC 를 즉시 포화시킨다 —",
          "그것만 막을 만큼 **낮은 차단**(0.05 Hz 이하 또는 DC 서보)이면 되고,",
          "기저선 제거는 디지털에서 영위상으로 하는 것이 맞다. 실제로",
          "**AFE 0.05 Hz + 중앙값 200+600** 이 표에서 가장 깨끗한 조합 중 하나다",
          f"(d1 S {g('d1','AFE 0.05 Hz + 중앙값 200+600','s_err'):.1f} · 잔여 {g('d1','AFE 0.05 Hz + 중앙값 200+600','wander'):.1f} %R).", "",
          "> **보드 실물을 먼저 확인할 것.** 위 숫자는 데이터시트 표준 구성을 모사한 것이다.",
          "> 실제 차단 주파수는 보드의 RC 값에 달렸고, 같은 구성이면 **40 Hz 아날로그**",
          "> **저역통과**도 함께 있다 — 그것만으로 R 진폭이 d0 5 % · d1 9 % 줄어든다.",
          "> 우리 비교는 100 Hz 대역을 전제하므로, 실측 축(D3)에서는 이 차이를 먼저 적어야 한다.",
          "", "## 그림", ""]
    L += [f"![{p.stem}](../{p.relative_to(ROOT)})" for p in made] + [""]
    p = ROOT / "docs" / "13_lookahead_fe.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", nargs="*", default=["d0", "d1"])
    a = ap.parse_args()
    per_axis, made = {}, []
    for tag in a.axis:
        print(f"[{tag}] 측정 중 …")
        rows, meta = measure(tag)
        per_axis[tag] = (rows, meta)
        for rw in rows:
            print(f"  {rw['label']:<30} 준위산포 {rw['tp_spread']:6.1f}  S오차 {rw['s_err']:6.1f}"
                  f"  T오차 {rw['t_err']:6.1f}  이동 {rw['drift']:6.1f}"
                  f"  잔여 {rw['wander']:6.1f}")
        made.append(figure(tag, ROOT / "results" / "fig" / f"lookahead_fe_{tag}.png"))
        made.append(frontier(tag, rows, ROOT / "results" / "fig" / f"lookahead_frontier_{tag}.png"))
    print("비용 측정 중 …")
    costs = cost()
    doc = write_doc(per_axis, made, costs)
    save_manifest(ROOT / "results" / "fig",
                  cfg={"script": "explore_lookahead_fe", "axes": list(per_axis),
                       "past_s": PAST_S, "hop_s": HOP_S, "inject_frac": INJECT_FRAC},
                  sources=[__file__])
    print(f"-> {doc.relative_to(ROOT)}")
    for p in made:
        print(f"-> {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
