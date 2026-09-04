#!/usr/bin/env python3
"""**채점 기준을 감사한다** — 지표가 정말 그 이름의 것을 재고 있는가.

    python3 scripts/audit_fe_metrics.py

산출: `docs/16_fe_metric_audit.md`, `results/fig/fe_audit_d0.png`

왜 필요한가
-----------
`docs/13`·`docs/15` 는 세 실시간 front-end 를 숫자로 갈랐고 블록 영위상이
이겼다. 그런데 사용자가 세 가지를 지적했다.

1. «S 골 오차 4 %R» 이 정말 S 골이 4 %R 얕아졌다는 뜻이면, S 골 깊이가
   10~14 %R 인데 **3 분의 1 이 사라졌다**는 소리다. 그림에는 그만한 차이가
   안 보인다.
2. 영위상 출력의 T-P 구간이 회색선 아래 있는 것이 **느린 기저선 변동이
   남아서**인가, 아니면 **직류 오프셋**인가. 후자라면 화면에서 맞춰 주면
   될 일이다.
3. 참조를 `FE_off(clean)`(= 오프라인 영위상) 으로 두고 블록 영위상과 중앙값을
   비교하면, **영위상 계열이 유리하게 짜인 것 아닌가.**

셋 다 «측정이 답할 수 있는» 질문이라 잰다. **F-30·F-33 이 같은 계열이었다** —
지표가 이름과 다른 것을 재고 있었고, 참조가 후보와 함께 움직였다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sps

from ecgdn.config import DEFAULT_FE, FS
from ecgdn.data.noise import make_noise
from ecgdn.data.sources import get_source
from ecgdn.eval.rpeak import detect_rpeaks
from ecgdn.utils import ensure_dir, rng, save_manifest
from explore_lookahead_fe import (INJECT_FRAC, _ko_font, _tp_levels, _tp_slices,
                                  f_causal, f_lookahead, f_median, f_offline,
                                  s_depth, t_amp, tp_off, tp_spread)

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "16_fe_metric_audit.md"
FIG = ROOT / "results" / "fig"
plt.rcParams.update({"font.family": _ko_font(), "axes.unicode_minus": False,
                     "mathtext.fontset": "dejavusans"})

# 감사 대상. 이름은 문서와 그림이 공유한다.
CANDS = [("오프라인 영위상", f_offline),
         ("인과 o1 0.5 Hz", lambda z: f_causal(z, 1, 0.5)),
         ("블록 영위상 0.5 s", lambda z: f_lookahead(z, 0.5)),
         ("중앙값 200+600 ms", f_median)]
ST_AT_S = 0.100        # ST 를 읽는 지점: R+100 ms (J 점 뒤 약 60 ms)


# ---------------------------------------------------------------- 기준선 없는 지표
def rs_pp(v, r, r_amp):
    """**R 꼭대기 − S 골** [%R]. 기준 준위가 필요 없다 — 두 점의 차뿐이다.

    지금의 `s_depth` 는 «그 박동의 T-P 중앙값 − S 골» 이라 **기준 준위가
    끼어 있다.** 기준 준위가 움직이면 S 골이 그대로여도 숫자가 변한다.
    둘을 나란히 놓아야 어느 쪽이 움직였는지 갈린다.
    """
    out = []
    for a in r[:-1]:
        j1 = min(a + int(0.08 * FS), v.size)
        if j1 - a >= 3:
            out.append((float(v[a]) - float(np.min(v[a:j1]))) / r_amp * 100)
    return float(np.median(out)) if out else float("nan")


def tp_curve(v, r, r_amp):
    """평활 구간 **안**의 굴곡 [%R] — 직선을 뺀 잔차 RMS (F-37).

    준위 산포(구간 **사이**)와도, 이음매(경계의 불연속)와도 다른 양이다.
    셋이 다른 원인을 갖는다는 것이 이 절의 요지다.
    """
    out = []
    for i0, i1 in _tp_slices(r, v.size):
        seg = v[i0:i1]
        if seg.size < 12:
            continue
        t = np.arange(seg.size)
        out.append(np.sqrt(np.mean((seg - np.polyval(np.polyfit(t, seg, 1), t)) ** 2)))
    return 100 * float(np.median(out)) / r_amp if out else float("nan")


def st_level(v, r, r_amp, at=ST_AT_S):
    """R+100 ms 의 준위 [%R], 그 박동의 T-P 중앙값 기준.

    임상에서 ST 편위를 읽는 방식과 같은 꼴이다 — **그 박동 자신의 기준선**에
    대해 읽는다. 절대 준위가 아니라 **기준선과의 거리**가 뜻을 갖는다.
    """
    out = []
    for (i0, i1), a in zip(_tp_slices(r, v.size), r[:-1]):
        j = a + int(at * FS)
        if j < v.size:
            out.append((float(v[j]) - float(np.median(v[i0:i1]))) / r_amp * 100)
    return float(np.median(out)) if out else float("nan")


def inject_st(x, r, amp, t0=0.040, t1=0.240, ramp=0.020):
    """ST 구간에 **알고 있는 크기**의 상승을 심는다.

    가장자리를 cosine 으로 눕혀 고주파를 안 넣는다 — 넣으면 40 Hz 저역통과가
    그것을 깎고, 그 손실이 방식의 차로 잘못 읽힌다.
    """
    y = x.copy()
    n0, n1, nr = int(t0 * FS), int(t1 * FS), int(ramp * FS)
    w = np.ones(n1 - n0)
    w[:nr] = 0.5 * (1 - np.cos(np.pi * np.arange(nr) / nr))
    w[-nr:] = w[:nr][::-1]
    for a in r:
        if a + n1 <= y.size:
            y[a + n0:a + n1] += amp * w
    return y


def resid_db(ref, v, r_amp):
    """잔차 파워 [dB, R² 대비]. **양쪽의 중앙값을 뺀다** — 직류는 안 센다.

    참조 자신을 넣으면 잔차가 0 이라 −inf 가 된다. 그것은 «무한히 좋다» 가
    아니라 **잴 것이 없다** 는 뜻이므로 NaN 으로 돌려 표에서 빠지게 한다.
    """
    a, b = ref - np.median(ref), v - np.median(v)
    n = min(a.size, b.size)
    e = float(np.mean((a[:n] - b[:n]) ** 2))
    return float("nan") if e <= 0 else 10 * np.log10(e / r_amp ** 2)


def lp_only(x):
    """세 방식이 **공유하는** 40 Hz 저역통과만 건 참값.

    이것을 참값으로 쓰면 공통 부분은 아무에게도 안 물린다. 남는 차이는
    **기저선을 어떻게 없앴는가** 뿐이다.
    """
    sos = sps.butter(DEFAULT_FE.order, DEFAULT_FE.lp_hz / (FS / 2),
                     btype="lowpass", output="sos")
    return sps.sosfiltfilt(sos, np.asarray(x, float),
                           padtype="odd", padlen=min(x.size - 1, int(2 * FS)))


# ---------------------------------------------------------------- 자료
def take(tag, name=None, seconds=60.0):
    src = get_source("synthetic" if tag == "d0" else "mitdb")
    name = name or src.records("test")[0]
    rec = src.get(name)
    x = np.asarray(rec.x, float)[: int(seconds * FS)]
    r = np.asarray(detect_rpeaks(x, FS), int)
    return x, r, float(np.median(x[r]) - np.median(x)), str(name)


def with_wander(x, r_amp, seed=("fe", "bw")):
    n = make_noise("bw_synth", x.size, FS, rng(*seed))
    n = n / (np.percentile(n, 97.5) - np.percentile(n, 2.5)) * (INJECT_FRAC * r_amp)
    return x + n


# ============================================================ 감사 1: S 골
def audit_s(tag):
    """«S 골 오차» 가 정말 S 골을 재는가 — **기준선 없는 지표와 나란히 놓는다.**"""
    x, r, r_amp, name = take(tag)
    rows = [dict(who="입력 (아무것도 안 함)", rs=rs_pp(x, r, r_amp),
                 tps=s_depth(x, r, r_amp), st=st_level(x, r, r_amp))]
    for lab, fn in CANDS:
        v = fn(x)
        rows.append(dict(who=lab, rs=rs_pp(v, r, r_amp), tps=s_depth(v, r, r_amp),
                         st=st_level(v, r, r_amp)))
    return rows, dict(record=name, r_amp=r_amp)


# ============================================================ 감사 2: T-P 준위
def audit_tp(tag):
    """T-P 준위가 회색선 아래인 것이 **직류인가 변동인가.**"""
    x, r, r_amp, name = take(tag)
    out = []
    for cond, src in (("원본", x), (f"+{INJECT_FRAC*100:.0f} %R 변동", with_wander(x, r_amp))):
        for lab, fn in [("입력", None)] + CANDS:
            v = src if fn is None else fn(src)
            lv = _tp_levels(v, r) / r_amp * 100
            out.append(dict(cond=cond, who=lab, dc=float(np.median(lv)),
                            spread=tp_spread(v, r, r_amp), off=tp_off(v, r, r_amp)))
    return out, dict(record=name)


# ============================================================ 감사 3: 순환성
def audit_reference(n_rec=8):
    """**참조를 바꾸면 순위가 바뀌는가.**

    D0 만 답할 수 있다. D1 의 «clean» 은 MIT-BIH 기록 자체라 **자기 기저선
    변동을 갖고 있고**, 그것을 참값으로 쓰면 «변동을 잘 없앤 방식» 이 벌을
    받는다. 참값이 진짜로 있는 축은 합성축뿐이다.

    한 기록이면 우연일 수 있으므로 test split 을 훑는다.
    """
    src = get_source("synthetic")
    names = list(src.records("test"))[:n_rec]
    rows = []
    for nm in names:
        x, r, r_amp, _ = take("d0", nm)
        y = with_wander(x, r_amp, seed=("fe", "bw", nm))
        truth, off = lp_only(x), f_offline(y)
        for lab, fn in CANDS:
            v = fn(y)
            rows.append(dict(record=nm, who=lab,
                             vs_truth=resid_db(truth, v, r_amp),
                             vs_off=resid_db(off, v, r_amp),
                             st_err=st_level(v, r, r_amp) - st_level(truth, r, r_amp),
                             spread=tp_spread(v, r, r_amp)))
    return rows, names


# ============================================================ 감사 4: ST
def audit_st(amp_frac=0.10, n_rec=8):
    """**알고 있는 ST 상승**을 심고 얼마가 살아 나오는지 본다.

    감사 3 의 «ST 준위 오차» 는 **정적 편향**이고, 이것은 «변화를 전달하는가»
    다. 둘은 다른 질문이다 — 편향이 있어도 변화는 그대로 전달될 수 있다.
    """
    src = get_source("synthetic")
    rows = []
    for nm in list(src.records("test"))[:n_rec]:
        x, r, r_amp, _ = take("d0", nm)
        xi = inject_st(x, r, amp_frac * r_amp)
        for lab, fn in [("아무것도 안 함", lambda z: z)] + CANDS:
            got = st_level(fn(xi), r, r_amp) - st_level(fn(x), r, r_amp)
            rows.append(dict(record=nm, who=lab, got=got,
                             keep=got / (amp_frac * 100) * 100))
    return rows


# ============================================================ 감사 5: 굴곡
def audit_curvature():
    """**평활 구간 안의 굴곡** — 창 확장 방식이 원인이다 (F-37).

    이음매를 고친 뒤에도 남던 결함이다. 이음매(경계 불연속)와 굴곡(구간 내
    저주파 요동)은 **다른 원인**이라 대책도 다르다.
    """
    from ecgdn.realtime.frontend_modes import (BlockZeroPhaseFE, CausalFE,
                                               MedianBaselineFE)
    rows = []
    for tag in ("d0", "d1"):
        x, r, r_amp, name = take(tag)
        y = with_wander(x, r_amp)
        H = 6
        outs = [("오프라인 영위상 (바닥)", f_offline(y)),
                ("블록 영위상 — `odd` 확장 (전)",
                 _stream(_OddPad(FS, hop_s=H / FS), y, H)),
                ("블록 영위상 — `constant` 확장 (후)",
                 _stream(BlockZeroPhaseFE(FS, hop_s=H / FS), y, H)),
                ("중앙값 200+600 ms", _stream(MedianBaselineFE(FS, hop_s=H / FS), y, H)),
                ("인과 o1 0.5 Hz", _stream(CausalFE(FS), y, H))]
        for who, v in outs:
            rows.append(dict(axis=tag, record=name, who=who,
                             curve=tp_curve(v, r, r_amp),
                             spread=tp_spread(v, r, r_amp)))
    return rows


def _stream(fe, y, H):
    return np.concatenate([fe.push(y[i:i + H]) for i in range(0, y.size, H)])


class _OddPad:
    """**고치기 전** 판 — 창 확장이 `odd` 다. 전후를 나란히 놓으려고 남긴다."""

    def __new__(cls, fs=FS, **kw):
        from ecgdn.realtime.frontend_modes import BlockZeroPhaseFE

        class _Impl(BlockZeroPhaseFE):
            def _process(self, w):
                pad = max(1, min(w.size - 1, w.size // 2))
                v = sps.sosfiltfilt(self._hp, w, padtype="odd", padlen=pad)
                return sps.sosfiltfilt(self._lp, v, padtype="odd", padlen=pad)

        return _Impl(fs, **kw)


# ============================================================ 그림
def seam_figure(out: Path, tag: str = "d0", show_s: float = 3.2):
    """**이음매를 눈으로 본다** — 교차 페이드 전후를 같은 구간에서 (F-36).

    숫자(«경계 차분이 내부의 23 배»)는 맞지만, 화면에서 무엇이 달라지는지는
    보여야 안다. 블록 경계에 눈금을 찍어 **계단이 거기에 맞는지**까지 보인다.
    """
    from ecgdn.realtime.frontend_modes import BlockZeroPhaseFE

    x, r, r_amp, name = take(tag)
    y = with_wander(x, r_amp)
    # T-P 구간이 넓게 들어오도록 R 하나 뒤에서 시작한다
    j = int(r[len(r) // 3])
    lo = max(0, j - int(0.1 * FS))
    hi = min(x.size, lo + int(show_s * FS))
    t = np.arange(hi - lo) / FS

    rows = [("① 처음  ·  odd 확장, 교차 페이드 없음", 0.048, 0.0, True, "#c05010"),
            ("② 이음매만 고침  ·  odd 확장, 교차 페이드", 0.024, None, True, "#b08020"),
            ("③ 지금  ·  constant 확장, 교차 페이드", 0.024, None, False, "#2a78d6")]
    # **세로 배율을 공유한다.** 안 그러면 칸마다 다른 배율로 그려져 «위 칸이
    # 더 크게 흔들린다» 가 배율 탓인지 결함 탓인지 못 가른다.
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 8.4), dpi=150,
                             sharex=True, sharey=True)
    fig.suptitle("블록 영위상 — 두 수정이 각각 무엇을 고쳤나 (같은 지연 548 ms)   ·   "
                 f"합성 (D0) 기록 {name}", x=0.02, ha="left", fontsize=13.5,
                 fontweight="bold")
    for ax, (lab, hop_s, xf, odd, col) in zip(axes, rows):
        fe = (_OddPad(FS, look_s=0.5, hop_s=hop_s, xfade_s=xf) if odd
              else BlockZeroPhaseFE(FS, look_s=0.5, hop_s=hop_s, xfade_s=xf))
        H = int(round(hop_s * FS))
        v = np.concatenate([fe.push(y[i:i + H]) for i in range(0, y.size, H)])
        seg = v[lo:hi]
        for b in range(H - (lo % H), hi - lo, H):      # 블록 경계
            ax.axvline(b / FS, color="#e4e4e4", lw=0.8, zorder=1)
        ax.axhline(0, color="#c8c8c8", lw=1.0, zorder=1)
        ax.plot(t, seg, color=col, lw=1.5, zorder=3)
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=10.5,
                      labelpad=10)
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.text(.995, .06,
                f"지연 {fe.latency_samples / FS * 1000:.0f} ms · "
                f"굴곡 {tp_curve(v, r, r_amp):.2f} %R",
                transform=ax.transAxes, ha="right", fontsize=9.5, color="#6b6b6b")
    axes[0].set_ylim(-0.30 * r_amp, 0.42 * r_amp)      # T-P 를 크게 — QRS 는 잘린다
    axes[-1].set_xlabel("s")
    fig.text(0.02, 0.015,
             "세로 눈금이 블록 경계다. ①의 계단이 그 눈금에 맞는 것이 «이음매», "
             "②에 남은 완만한 휨이 «굴곡» 이다 — 원인이 달라 대책도 달랐다. "
             "QRS 는 세로로 잘려 있다 (T-P 를 크게 보려고).",
             fontsize=10, color="#1b1b1b")
    fig.tight_layout(rect=[0.02, 0.035, 1, 0.955])
    ensure_dir(out.parent); fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def figure(ref_rows, out: Path):
    """**참조를 바꾸면 순위가 뒤집힌다** 를 한 장으로."""
    import pandas as pd
    df = pd.DataFrame(ref_rows)
    p = df.pivot_table(index="record", columns="who", values=["vs_truth", "vs_off"])
    blk, med = "블록 영위상 0.5 s", "중앙값 200+600 ms"
    d_true = (p[("vs_truth", blk)] - p[("vs_truth", med)]).values
    d_off = (p[("vs_off", blk)] - p[("vs_off", med)]).values
    recs = list(p.index)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 4.6), dpi=150)
    y = np.arange(len(recs))
    a1.axvline(0, color="#8a8a8a", lw=1.0)
    a1.barh(y + .19, d_off, .36, color="#2a78d6", label="참조 = 오프라인 영위상")
    a1.barh(y - .19, d_true, .36, color="#c05010", label="참조 = 참값 (합성 원본)")
    a1.set_yticks(y); a1.set_yticklabels(recs, fontsize=9)
    # U+2212 는 나눔 글꼴에 없다 — 그림 문자열에는 ASCII 하이픈만 쓴다
    a1.set_xlabel("블록 영위상 - 중앙값  [dB]   (왼쪽 = 블록이 좋다)", fontsize=10)
    a1.set_title("참조를 바꾸면 순위가 바뀐다", fontsize=12, loc="left",
                 fontweight="bold")
    a1.legend(fontsize=9, loc="lower right", frameon=False)
    for sp in ("top", "right"):
        a1.spines[sp].set_visible(False)

    g = df.groupby("who")[["spread", "st_err"]].mean()
    cols = {"오프라인 영위상": "#b8b6ae", "인과 o1 0.5 Hz": "#eb6834",
            blk: "#2a78d6", med: "#1baf7a"}
    for who, row in g.iterrows():
        a2.scatter(row["spread"], abs(row["st_err"]), s=150,
                   color=cols.get(who, "#666"), zorder=3)
        a2.annotate(who, (row["spread"], abs(row["st_err"])),
                    textcoords="offset points", xytext=(9, 5), fontsize=9.5)
    a2.set_xlabel("박동별 T-P 준위 산포 [%R]   (작을수록 평평)", fontsize=10)
    a2.set_ylabel("ST 준위 오차 [%R]   (작을수록 충실)", fontsize=10)
    a2.set_title("둘은 같은 방향이 아니다 — 고르는 문제다", fontsize=12,
                 loc="left", fontweight="bold")
    a2.set_xlim(0, None); a2.set_ylim(-0.15, None)
    for sp in ("top", "right"):
        a2.spines[sp].set_visible(False)
    fig.tight_layout()
    ensure_dir(out.parent); fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ============================================================ 문서
def tbl(head, rows, fmt):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        out.append("| " + " | ".join(f.format(v) if isinstance(f, str) else f(v)
                                     for f, v in zip(fmt, r)) + " |")
    return "\n".join(out)


def main() -> int:
    import pandas as pd
    s0, m0 = audit_s("d0")
    s1, m1 = audit_s("d1")
    t0, _ = audit_tp("d0")
    t1, _ = audit_tp("d1")
    curve = audit_curvature()
    ref, names = audit_reference()
    st = pd.DataFrame(audit_st())

    rf = pd.DataFrame(ref)
    g = (rf.groupby("who")[["vs_truth", "vs_off", "st_err", "spread"]].mean()
         .reindex([c[0] for c in CANDS]))
    p = rf.pivot_table(index="record", columns="who", values=["vs_truth", "vs_off"])
    blk, med = "블록 영위상 0.5 s", "중앙값 200+600 ms"
    d_true = p[("vs_truth", blk)] - p[("vs_truth", med)]
    d_off = p[("vs_off", blk)] - p[("vs_off", med)]
    sg = st.groupby("who")[["got", "keep"]].agg(["mean", "std"])

    fig_p = figure(ref, FIG / "fe_audit_d0.png")
    seam_p = seam_figure(FIG / "fe_seam_d0.png")
    print(f"-> {seam_p.relative_to(ROOT)}")
    write_doc(s0, m0, s1, m1, t0, t1, g, d_true, d_off, sg, names, fig_p, curve)
    save_manifest(FIG, cfg={"script": "audit_fe_metrics"},
                  extra={"n_records": len(names)}, sources=[__file__])
    print(f"-> {DOC.relative_to(ROOT)}\n-> {fig_p.relative_to(ROOT)}")
    return 0


def write_doc(s0, m0, s1, m1, t0, t1, g, d_true, d_off, sg, names, fig_p,
              curve) -> None:
    """**숫자에서** 쓴다. 손으로 옮기지 않는다 (그러다 한 번 틀렸다)."""
    import pandas as pd

    def srow(rows):
        return [(r["who"], r["rs"], r["tps"], r["st"]) for r in rows]

    def trow(rows, cond):
        return [(r["who"], r["dc"], r["spread"], r["off"])
                for r in rows if r["cond"] == cond]

    f4 = ("{}", "{:.1f}", "{:.2f}", "{:+.2f}")
    ft = ("{}", "{:+.2f}", "{:.2f}", "{:.2f}")
    n_flip = int((d_true > 0).sum())
    cond_w = f"+{INJECT_FRAC*100:.0f} %R 변동"
    cv = pd.DataFrame(curve).pivot_table(index="who", columns="axis", values="curve")
    _order = ["오프라인 영위상 (바닥)", "블록 영위상 — `odd` 확장 (전)",
              "블록 영위상 — `constant` 확장 (후)", "중앙값 200+600 ms",
              "인과 o1 0.5 Hz"]
    cv = cv.reindex([o for o in _order if o in cv.index])
    curve_md = "\n".join(
        ["| | 굴곡 D0 [%R] | 굴곡 D1 [%R] |", "|---|---|---|"]
        + [f"| {w} | {r['d0']:.3f} | {r['d1']:.3f} |" for w, r in cv.iterrows()])
    _base = cv.loc["오프라인 영위상 (바닥)", "d0"]
    curve_ratio = cv.loc["블록 영위상 — `odd` 확장 (전)", "d0"] / _base
    curve_after = cv.loc["블록 영위상 — `constant` 확장 (후)", "d0"] / _base
    # T 파 높이는 «왜 D0 만 T-P 가 내려가나» 의 근거다 — 직접 잰다
    t_d0, t_d1 = (t_amp(f_offline(take(t)[0]), *take(t)[1:3]) for t in ("d0", "d1"))

    body = f"""# 16. 채점 기준 감사 — **지표가 그 이름의 것을 재고 있는가** `[측정]`

생성: `scripts/audit_fe_metrics.py` · 그림 `{fig_p.relative_to(ROOT)}`,
`results/fig/fe_seam_d0.png`

`docs/13`·`docs/15` 는 세 실시간 front-end 를 숫자로 갈랐고 블록 영위상이
이겼다. 그 숫자들에 대해 세 가지 의심이 제기됐고, **셋 다 측정이 답할 수
있는 질문**이라 쟀다. **하나는 지적이 옳았고, 하나는 지표 이름이 틀렸고,
하나는 절반만 맞았다.**

---

## 1. 「S 골 오차」는 **S 골을 재고 있지 않았다**

의심: *S 골 깊이가 10~14 %R 인데 오차가 4 %R 이면 3 분의 1 이 사라졌다는
뜻이다. 그림에는 그만한 차이가 안 보인다.*

**단위 해석은 맞다.** `s_depth` 는 «그 박동의 T-P 중앙값 − R+80 ms 최저점»
이므로, 4 %R 은 R 진폭의 4 % 다. 문제는 이 정의에 **기준 준위가 끼어 있다**는
것이다. 기준 준위가 움직이면 S 골이 그대로여도 숫자가 변한다.

그래서 **기준 준위가 필요 없는 지표**를 나란히 놓았다 — R 꼭대기와 S 골의
차(`R−S 봉우리차`). 두 점의 차뿐이라 기준선이 어디 있든 상관없다.

**D0 (합성, 기록 {m0['record']})**

{tbl(["", "R−S 봉우리차 [%R]", "T-P 기준 S 깊이 [%R]", "ST 준위 [%R]"], srow(s0), f4)}

**D1 (MIT-BIH, 기록 {m1['record']})**

{tbl(["", "R−S 봉우리차 [%R]", "T-P 기준 S 깊이 [%R]", "ST 준위 [%R]"], srow(s1), f4)}

**R−S 봉우리차는 네 방식이 사실상 같다** (D1 에서 112.2~113.0, 폭 0.8 %R).
**S 골 자체는 아무도 안 깎는다.** 그런데 T-P 기준 S 깊이는 D1 에서
9.3~19.6 으로 **10 %R 이나 벌어진다.**

움직인 것은 **S 골이 아니라 QRS 옆의 기준 준위**다. 오른쪽 열이 그것을
확인한다 — ST 준위(R+{ST_AT_S*1000:.0f} ms, 그 박동의 T-P 기준)가 D1 에서
오프라인 {s1[1]['st']:+.2f}, 중앙값 {s1[4]['st']:+.2f} 로 {abs(s1[4]['st']-s1[1]['st']):.1f} %R 차이나고,
이것이 S 깊이 차 {abs(s1[4]['tps']-s1[1]['tps']):.1f} %R 과 거의 같다.

> **그림에 안 보이는 것이 당연하다.** 그림에서 눈은 «S 골이 얼마나 깊나» 를
> 보는데, 실제로 달라진 것은 «그 박동의 기준선이 어디 있나» 다. 지표 이름이
> 「S 골 오차」인 것이 잘못이었다. **「QRS 직후 기준 준위 이동」이 맞다.**
> 이것은 F-30 과 같은 계열이다 — 지표가 이름과 다른 것을 재고 있었다.

이 이름이 왜 중요한가: **QRS 직후의 준위가 곧 ST 구간**이다. 「S 골이 조금
얕다」는 파형 미용 문제지만, 「ST 준위가 {abs(s1[4]['st']-s1[1]['st']):.1f} %R 이동한다」는
성질이 다르다. 4 절에서 따로 잰다.

---

## 2. T-P 구간이 회색선 아래인 것 — **직류이고, 축마다 다르다**

의심: *영위상 출력의 T-P 가 회색선 아래인 것은 느린(0.1 Hz 급) 기저선 변동이
남아서 아닌가? 그렇다면 화면에서 박동별로 맞춰 주면 될 일이다.*

박동별 T-P 준위를 **직류(전체 중앙값)** 와 **그 주변 산포**로 갈랐다.

**D0 (합성) — 원본 입력**

{tbl(["", "T-P 직류 [%R]", "산포 [%R]", "이탈 [%R]"], trow(t0, "원본"), ft)}

**D1 (MIT-BIH) — 원본 입력**

{tbl(["", "T-P 직류 [%R]", "산포 [%R]", "이탈 [%R]"], trow(t1, "원본"), ft)}

두 축이 **다른 이야기를 한다.**

* **D1 에서는 영위상이 회색선 아래에 있지 않다.** 직류가 오프라인 +{[r for r in t1 if r['cond']=='원본' and r['who']=='오프라인 영위상'][0]['dc']:.2f},
  블록 +{[r for r in t1 if r['cond']=='원본' and r['who']=='블록 영위상 0.5 s'][0]['dc']:.2f} 로 **거의 0 이다.** 아래 있는 것은
  **입력**({[r for r in t1 if r['cond']=='원본' and r['who']=='입력'][0]['dc']:.2f} %R) 이고, 그것은 기록 자신의 오프셋이다 — 지적하신 그대로다.
* **D0 에서는 영위상이 −8.6 %R 에 있다.** 중앙값은 −0.07. 차이의 원인은
  T 파 크기다 — D0 의 T 는 {t_d0:.0f} %R 로 크고(D1 은 {t_d1:.0f} %R), 선형 고역통과의 «0» 은
  **기록 전체 평균**이라 R·T 가 큰 만큼 T-P 가 아래로 밀린다. D1(기록 {m1['record']})은
  T 가 작아 그 효과가 거의 없다.

**그래서 이것은 직류이지 남은 변동이 아니다.** 산포(직류를 뺀 뒤의 폭)는
D0 원본에서 오프라인 {[r for r in t0 if r['cond']=='원본' and r['who']=='오프라인 영위상'][0]['spread']:.2f}, 중앙값 {[r for r in t0 if r['cond']=='원본' and r['who']=='중앙값 200+600 ms'][0]['spread']:.2f} %R 이다.

### 화면에서 어떻게 그릴 것인가 — **직류는 맞춰도 되고, 박동별은 안 된다**

제안하신 대로 **직류 하나를 빼는 것은 옳다.** 그것은 정보가 없는 상수이고,
빼면 네 방식을 같은 자리에서 볼 수 있다.

**그런데 「박동별로 자기 기준선에 맞춘다」는 다르다.** 그 연산이 바로
**중앙값 계열 기저선 보정 그 자체**다. 박동마다 T-P 준위를 0 으로 옮기면
산포가 **정의상 0** 이 되므로, 그 화면은 네 방식을 구별하지 못한다 —
비교하려던 바로 그 양을 지운다.

> 정리하면: **「박동별로 맞춰 그린다」와 「중앙값 front-end 를 쓴다」는 같은
> 연산이다.** 사슬의 어디에 두느냐만 다르다. 그러므로 그것은 그리기 방식의
> 문제가 아니라 **front-end 선택 그 자체**다.

`compare_four_fe.py` 는 **직류만 빼는 열을 추가**했다(`--recenter dc`).
기본은 예전대로 손대지 않은 그림이다.

**기저선 변동 {INJECT_FRAC*100:.0f} %R 을 주입하면** 산포는 이렇게 된다.

**D0**

{tbl(["", "T-P 직류 [%R]", "산포 [%R]", "이탈 [%R]"], trow(t0, cond_w), ft)}

**D1**

{tbl(["", "T-P 직류 [%R]", "산포 [%R]", "이탈 [%R]"], trow(t1, cond_w), ft)}

D0 에서 «블록 영위상의 T-P 가 찌그러진다» 고 보신 것은 사실이다. 다만
**오프라인 영위상도 똑같다**({[r for r in t0 if r['cond']==cond_w and r['who']=='블록 영위상 0.5 s'][0]['spread']:.2f} 대 {[r for r in t0 if r['cond']==cond_w and r['who']=='오프라인 영위상'][0]['spread']:.2f} %R) —
블록화의 문제가 아니라 **4 차 0.5 Hz 고역통과라는 형태의 공통 한계**다.
중앙값은 {[r for r in t0 if r['cond']==cond_w and r['who']=='중앙값 200+600 ms'][0]['spread']:.2f} 로 확실히 낫다.

---

## 3. **참조가 영위상 계열에 유리했다 — 지적이 맞다** `[측정]`

의심: *참조를 `FE_off(clean)`(= 오프라인 영위상)로 두면, 블록 영위상은 같은
계열이니 당연히 가깝다. 「더 낫다」가 아니라 「더 닮았다」를 재는 것 아닌가.*

**맞다.** 그리고 이것은 잴 수 있다 — **합성축에는 참값이 있다.**

D1 로는 답할 수 없다. D1 의 «clean» 은 MIT-BIH 기록 자체라 **자기 기저선
변동을 갖고 있고**, 그것을 참값으로 쓰면 변동을 잘 없앤 방식이 벌을 받는다.
참값이 진짜로 있는 축은 합성축뿐이다.

세 방식이 **공유하는** 40 Hz 저역통과만 건 원본을 참값으로 삼았다(공통 부분은
아무에게도 안 물린다). 합성축 test {len(names)} 기록, {cond_w} 주입:

| | 참값 기준 [dB] | `FE_off` 기준 [dB] | ST 준위 오차 [%R] | 준위 산포 [%R] |
|---|---|---|---|---|
| 오프라인 영위상 | {g.loc['오프라인 영위상','vs_truth']:.2f} | (자기 자신) | {g.loc['오프라인 영위상','st_err']:+.2f} | {g.loc['오프라인 영위상','spread']:.2f} |
| 인과 o1 0.5 Hz | {g.loc['인과 o1 0.5 Hz','vs_truth']:.2f} | {g.loc['인과 o1 0.5 Hz','vs_off']:.2f} | {g.loc['인과 o1 0.5 Hz','st_err']:+.2f} | {g.loc['인과 o1 0.5 Hz','spread']:.2f} |
| **블록 영위상 0.5 s** | {g.loc['블록 영위상 0.5 s','vs_truth']:.2f} | **{g.loc['블록 영위상 0.5 s','vs_off']:.2f}** | {g.loc['블록 영위상 0.5 s','st_err']:+.2f} | {g.loc['블록 영위상 0.5 s','spread']:.2f} |
| **중앙값 200+600 ms** | **{g.loc['중앙값 200+600 ms','vs_truth']:.2f}** | {g.loc['중앙값 200+600 ms','vs_off']:.2f} | {g.loc['중앙값 200+600 ms','st_err']:+.2f} | {g.loc['중앙값 200+600 ms','spread']:.2f} |

**순위가 뒤집힌다.**

* `FE_off` 기준: 블록 영위상이 **{abs(d_off.mean()):.2f} dB 우세**, {len(names)} 기록 **전부** 블록 승.
* 참값 기준: 중앙값이 **{abs(d_true.mean()):.2f} dB 우세**, {len(names)} 기록 중 {n_flip} 기록에서 중앙값 승.

즉 `total_db` 가 잰 것의 상당 부분은 «어느 쪽이 더 나은가» 가 아니라
**«어느 쪽이 오프라인 filtfilt 를 더 닮았는가»** 였다. 블록 영위상은 미리보기를
늘리면 오프라인 영위상으로 **수렴하도록 정의된 방식**이므로, 그 참조에 대해
유리한 것이 구조적으로 당연하다.

> **F-33 과 같은 계열이지만 한 겹 더 깊다.** F-33 은 참조가 후보와 **함께
> 움직이는** 문제였고, 그것을 고정해서 `total_db` 를 만들었다. 그런데
> 고정한 그 참조가 **후보 중 한 계열의 극한**이었다. 참조를 고정하는 것만으로는
> 부족하다 — **후보 밖에 있어야** 한다.

**다만 절반만 맞다.** 참값 기준으로도 두 방식의 차는
{abs(d_true.mean()):.2f} dB 로 작고 기록마다 부호가 갈린다({len(names)} 중 {n_flip}). 그리고
**인과 방식은 어느 기준으로도 확실히 진다**({g.loc['인과 o1 0.5 Hz','vs_truth']:.2f} 대 {g.loc['블록 영위상 0.5 s','vs_truth']:.2f}/{g.loc['중앙값 200+600 ms','vs_truth']:.2f}).
바뀌는 것은 **1 위와 2 위**이지 «인과를 버린다» 는 결론이 아니다.

---

## 4. 그럼 무엇으로 고를 것인가 — **ST**

3 절의 표에서 두 열이 반대 방향을 가리킨다.

* **준위 산포**: 중앙값 {g.loc['중앙값 200+600 ms','spread']:.2f} 대 블록 {g.loc['블록 영위상 0.5 s','spread']:.2f} %R — 중앙값 압승
* **ST 준위 오차**: 중앙값 {g.loc['중앙값 200+600 ms','st_err']:+.2f} 대 블록 {g.loc['블록 영위상 0.5 s','st_err']:+.2f} %R — 블록 압승

「평평한 것이 좋다」는 것만으로는 못 고른다. **평평하게 만드는 연산이
ST 를 건드리기 때문**이다. 2 절의 D0 원본 표가 그것을 극단적으로 보인다 —
**인과 o1 0.5 Hz 의 산포가 오프라인 영위상보다 작다**(0.90 대 2.60). 인과
필터가 저주파를 더 깎아 T-P 가 평평해진 것인데, 같은 이유로 ST 를
+{s0[2]['st']-s0[1]['st']:.1f} %R 왜곡한다. **평평함만 보면 가장 왜곡이 큰 방식이 1 위가 된다.**

그래서 임상에서 실제로 읽는 양으로 갈랐다 — **ST 편위는 그 박동 자신의
기준선에 대해 읽는다.** 절대 준위는 뜻이 없다.

합성축 {len(names)} 기록에 **알고 있는 크기의 ST 상승 10 %R** 을 심고, 각
방식을 통과시킨 뒤 얼마가 살아 나오는지 쟀다.

| | 회복된 ST [%R] | 보존율 [%] |
|---|---|---|
| 아무것도 안 함 | {sg.loc['아무것도 안 함',('got','mean')]:.2f} | {sg.loc['아무것도 안 함',('keep','mean')]:.1f} |
| 오프라인 영위상 | {sg.loc['오프라인 영위상',('got','mean')]:.2f} | {sg.loc['오프라인 영위상',('keep','mean')]:.1f} ± {sg.loc['오프라인 영위상',('keep','std')]:.1f} |
| 인과 o1 0.5 Hz | {sg.loc['인과 o1 0.5 Hz',('got','mean')]:.2f} | {sg.loc['인과 o1 0.5 Hz',('keep','mean')]:.1f} ± {sg.loc['인과 o1 0.5 Hz',('keep','std')]:.1f} |
| **블록 영위상 0.5 s** | {sg.loc['블록 영위상 0.5 s',('got','mean')]:.2f} | **{sg.loc['블록 영위상 0.5 s',('keep','mean')]:.1f} ± {sg.loc['블록 영위상 0.5 s',('keep','std')]:.1f}** |
| **중앙값 200+600 ms** | {sg.loc['중앙값 200+600 ms',('got','mean')]:.2f} | **{sg.loc['중앙값 200+600 ms',('keep','mean')]:.1f} ± {sg.loc['중앙값 200+600 ms',('keep','std')]:.1f}** |

중앙값은 심은 것의 **{sg.loc['중앙값 200+600 ms',('keep','mean')]:.0f} %** 만 전달하고, 무엇보다 **기록마다
±{sg.loc['중앙값 200+600 ms',('keep','std')]:.0f} % 로 들쭉날쭉하다.** 블록 영위상은 {sg.loc['블록 영위상 0.5 s',('keep','mean')]:.0f} ± {sg.loc['블록 영위상 0.5 s',('keep','std')]:.0f} % 다.

기제는 2 절에서 이미 보인 것과 같다. 200 ms 중앙값 창이 QRS 직후에 놓이면
창 안이 S·ST 로 눌려 있어 **기저선 추정이 그쪽으로 끌려간다.** 그 추정을 빼면
ST 가 기준선 쪽으로 당겨진다 — 평평해 보이는 이유이자, ST 를 못 지키는 이유가
**같은 하나**다.

> **이 과제는 진단 기기가 아니다.** ST 를 판독하지 않고, 판독한다고 주장하지도
> 않는다. 그러나 «무엇이 더 나은 파형인가» 를 물을 때, «평평함» 하나만 보면
> **평평하게 만드는 연산이 무엇을 지우는지** 를 못 본다. 그래서 지표를 하나
> 늘렸다.

---

## 5. 결론 — **채점표를 다시 쓴다**

| 고칠 것 | 어떻게 |
|---|---|
| 「S 골 오차」라는 이름 | **「QRS 직후 기준 준위 이동」** 으로 고친다. S 골 자체는 `R−S 봉우리차` 로 따로 잰다 (네 방식 모두 온전) |
| `total_db` 를 front-end 선택 근거로 쓰는 것 | **그만둔다.** 참조가 후보 한 계열의 극한이라 편향돼 있다. 합성축 참값 기준을 함께 본다 |
| 「평평할수록 좋다」 | **조건부다.** 평평하게 만드는 연산이 ST 를 건드린다. `ST 보존율` 을 함께 본다 |

**세 방식의 자리는 이렇게 정리된다** (합성축 참값 기준):

| | 평평함 | ST 충실도 | 지연 | 비고 |
|---|---|---|---|---|
| 인과 o1 0.5 Hz | 나쁨 ({g.loc['인과 o1 0.5 Hz','spread']:.1f}) | 나쁨 ({g.loc['인과 o1 0.5 Hz','st_err']:+.1f}) | **0 ms** | 어느 기준으로도 3 위 |
| 블록 영위상 0.5 s | 보통 ({g.loc['블록 영위상 0.5 s','spread']:.1f}) | **가장 좋음 ({g.loc['블록 영위상 0.5 s','st_err']:+.1f})** | 596 ms | ST 보존 {sg.loc['블록 영위상 0.5 s',('keep','mean')]:.0f} % |
| 중앙값 200+600 ms | **가장 좋음 ({g.loc['중앙값 200+600 ms','spread']:.1f})** | 나쁨 ({g.loc['중앙값 200+600 ms','st_err']:+.1f}) | 400 ms | ST 보존 {sg.loc['중앙값 200+600 ms',('keep','mean')]:.0f} ± {sg.loc['중앙값 200+600 ms',('keep','std')]:.0f} % |

**D-20 은 「숫자가 이겼으니 블록 영위상」이 아니라 「무엇을 지킬 것인가」의
문제로 다시 놓인다.** 화면의 평평함을 사면 ST 충실도를 판다. 판단은 사용자
몫이고, 이 표가 그 판단의 재료다.

---

## 6. 블록 영위상이 T-P 에서 **튀던 것** — 이음매였다, 그리고 고쳤다 `[측정]`

3~5 절과 별개로, 블록 영위상 출력이 T-P 구간에서 **주기적으로 계단처럼
튀는** 것이 화면에서 보였다. 원인과 대책을 갈랐다.

### 무엇이 튀는가

블록 경계(hop 96 ms)의 1 차 차분을 그 사이 구간과 비교했다 — D0, 변동 50 %R,
T-P 구간 안에서만:

| | 경계 차분 [%R] | 내부 차분 [%R] | 비 |
|---|---|---|---|
| 오프라인 영위상 | 0.019 | 0.021 | **0.93** |
| 블록 영위상 0.5 s | 2.220 | 0.097 | **23.0** |

**경계에서 내부의 23 배**다. 오프라인에는 없다. 즉 «남은 기저선 변동» 이
아니라 **블록화 그 자체**가 원인이다. T-P 에서 유독 크게 보이는 이유는
그 구간이 원래 평평해서(내부 차분 0.1 %R) 2.2 %R 계단이 상대적으로
거대하기 때문이다 — QRS 근처에서는 신호 자체가 급변해 묻힌다.

### 안 통하는 대책 둘 — **먼저 이것을 배제했다**

| 시도 | 이음매 비 | 판정 |
|---|---|---|
| 과거 문맥 4 s (지금) | 23.0 | — |
| 과거 문맥 8 / 16 / 32 s | 23.9 / 24.2 / 24.2 | **전혀 안 낫는다** |
| 미리보기 0.75 s | 25.4 | 안 낫는다 |
| 미리보기 1.0 s | 44.6 | **오히려 나쁘다** |
| padding 을 짧게 | 131.1 | 훨씬 나쁘다 (지금 설정이 이미 완화책이었다) |

«필터가 덜 정착해서» 라는 첫 가설은 **틀렸다.** 과거를 8 배로 늘려도 그대로다.
원인은 정착이 아니라 **창마다 다른 유한 구간**이다 — 창이 hop 만큼 미끄러지면
같은 표본의 filtfilt 결과가 미세하게 달라지고, 그 차이가 경계에서 계단이 된다.
0.5 Hz 고역통과의 임펄스 응답은 수 초에 걸쳐 있어 **유한 창으로는 원리적으로
정확히 못 낸다.**

### 통하는 대책 — **겹쳐 취해 평균한다**

블록마다 hop 보다 `X` 만큼 넓게 취해 raised-cosine 으로 섞는다. 같은 표본을
서로 다른 창으로 계산한 값의 가중 평균이 되고, 창 위치에 따른 요동은 부호가
오가므로 **상쇄된다.**

| hop | 교차 페이드 | 총 지연 | 이음매 비 (D0) | 오프라인차 p99 (D0) |
|---|---|---|---|---|
| 96 ms | 0 (전) | 596 ms | 22.8 | 7.43 %R |
| 96 ms | 24 ms | 620 ms | **0.43** | 6.81 |
| 96 ms | 96 ms | 692 ms | 0.94 | 5.26 |
| 48 ms | 0 (지금 브리지) | **548 ms** | 15.0 | 8.55 |
| **24 ms** | **24 ms (채택)** | **548 ms** | **0.65** | **6.19** |

**이음매는 24 ms 면 이미 사라지고**, 그 위로는 오프라인 근사도만 산다.

마지막 두 줄이 핵심이다. 교차 페이드를 **hop 과 같게** 두면, **hop 을 절반으로
줄여 그 지연을 그대로 되찾을 수 있다.** 브리지의 FE hop 을 48 -> 24 ms 로 바꾸면
총 지연이 **548 ms 로 예전과 똑같은데** 이음매가 15.0 -> 0.65 로 사라지고
오프라인 근사도까지 8.55 -> 6.19 %R (28 %) 좋아진다.

**공짜인 이유**는 FE 가 싸기 때문이다 — 블록당 0.57 ms 라 hop 24 ms 예산의
2 % 다. 그래서 **FE 의 hop 을 추론 hop 과 분리**했다(`--fe-hop`, 기본 6 표본).
묶어 두면 딥러닝 추론까지 2 배로 자주 돌아 CPU 가 2 배가 된다.

### 중앙값 모드에는 필요 없다 — 그리고 그것이 기제를 확인해 준다

같은 조건에서 중앙값 모드의 이음매 비는 교차 페이드 **없이도 0.95** 다.
중앙값은 창 `w1`·`w2` 안의 자료만 보는 **국소 연산**이라 창이 hop 만큼
미끄러져도 결과가 거의 안 바뀐다. `filtfilt` 는 **창 전체의 함수**라 바뀐다.
그래서 중앙값 모드는 교차 페이드를 끈다 — 켜면 지연만 hop 만큼 는다.

이것은 `dl_wrapper` 가 window 를 50 % 겹쳐 `Hann²` overlap-add 하는 것과
**같은 원리**다. `measure_stream_seam.py` 의 머리말이 이미 «오프라인은 겹쳐서
이음매를 지우는데 스트리밍에서는 그럴 수 없다» 고 적어 두었는데, **front-end
에서는 할 수 있다** — 미리보기 창이 이미 미래를 갖고 있기 때문이다.

### 왜 실기록에서 덜 보였나 — **denoiser 를 거치면 다시 드러난다**

D1(기록 100)에서는 교차 페이드 없이도 이음매 비가 1.50 뿐이다. T-P 안의
잡음이 분모를 키워 묻히기 때문이다. **그런데 그 출력을 `M06` 에 통과시키면
2.33 으로 커진다** — denoiser 가 잡음을 지우면 이음매만 남는다.

| 기록 100 | FE 출력 | `M06` 통과 후 |
|---|---|---|
| 교차 페이드 없음 | 1.50 | **2.33** |
| 교차 페이드 48 ms | 0.97 | **1.03** |

**화면에 보이는 것은 denoiser 출력**이므로, FE 단계에서 잡음에 묻혀 안 보이던
것이 화면에서는 보인다. 지표만 보고 «실기록에서는 문제가 아니다» 라고 했다면
틀렸을 것이다.

> **회귀 시험을 붙였다** (`tests/test_fe_seam.py`). 이 결함은 **눈으로만
> 보였다** — 스트리밍이 오프라인과 일치하는지도, 지연이 맞는지도, 정렬이
> 되는지도 전부 통과했다. 시험은 교차 페이드를 껐을 때 이음매가 **나타나는지**도
> 함께 확인한다. 그러지 않으면 «결함을 못 잡는 시험» 이 통과할 수 있다.
>
> 시험을 짜면서 한 번 틀렸다. 평평한 구간을 «차분이 작은 하위 60 %» 로
> 골랐는데, 그러면 **이음매가 큰 표본이 스스로 걸러진다** — 재려는 것을 빼고
> 재는 꼴이라 결함이 있는데도 비가 1.0 근처로 나왔다. 위치로 골라야 한다.

---

## 7. 이음매를 고쳤더니 **굴곡**이 보였다 — 창 확장이 원인 `[측정]`

계단이 없어지자 다음 것이 보였다: 블록 영위상만 평활 구간이 **완만하게
휜다.** 이음매(구간 **사이**의 불연속)와 다른 양이라 지표도 따로 필요했다 —
**평활 구간 안에서 직선을 뺀 잔차 RMS**.

{curve_md}

**블록 영위상만 바닥의 {curve_ratio:.1f} 배였다**(D0). 중앙값과 인과는 바닥과 같다 —
블록화 방식 고유의 결함이다. **교차 페이드는 여기에 거의 영향이 없었다**
(1.32 -> 1.23) — 이음매와 굴곡은 원인이 다르다. 아래에서 고쳐 **{curve_after:.1f} 배**로
내렸고, 표의 «후» 행이 그 결과다.

### 원인 — 미리보기가 짧아 **역방향 패스가 정착 못 한다**

미리보기 `F` 를 늘리며 재면 단조롭게 준다 (D0, 교차 페이드 켠 채):

| 미리보기 | 0.5 s | 1 s | 2 s | **4 s** | 8 s |
|---|---|---|---|---|---|
| 굴곡 [%R] | 1.229 | 0.676 | 0.515 | **0.206** | 0.208 |

**4 s 면 바닥(0.209)과 구별이 안 된다.** 4 차 0.5 Hz 필터의 역방향 패스가
정착하는 데 그만큼 걸린다는 뜻이다. 과거 문맥은 반대로 **늘릴수록 나쁘다**
(2 s 0.960 -> 16 s 1.325) — 정방향은 이미 정착했고 문제는 오른쪽 끝이다.

그런데 **미리보기 = 지연**이다. 4 s 는 화면용으로 못 쓴다.

### 안 통하는 대책 — 차수와 차단

| | 굴곡 | 남은 변동 | ST 차 |
|---|---|---|---|
| 1 차 0.5 Hz | 2.216 | 10.6 | +1.69 |
| 2 차 0.5 Hz | 1.546 | 8.4 | +1.66 |
| **4 차 0.5 Hz (지금)** | **1.229** | **6.7** | **+1.61** |
| 4 차 0.7 Hz | 0.888 | 1.4 | **-3.47** |
| 4 차 1.0 Hz | 1.799 | 0.5 | **-6.82** |

**차수를 낮추면 오히려 나빠진다** — 통과대역 감쇠가 완만해 저주파를 덜 지우고
그 잔여가 굴곡으로 보인다. 차단을 올리면 굴곡은 줄지만 **ST 를 판다**(0.7 Hz
에서 -3.5 %R). 둘 다 답이 아니다.

### 통하는 대책 — **창 확장을 `odd` 에서 `constant` 로**

`scipy` 의 `filtfilt` 는 창 밖을 확장해 가장자리 효과를 줄인다. 기본이자
지금까지 쓰던 `odd` 는 **끝점을 중심으로 점대칭 반사**라 **끝점의 기울기를
그대로 연장**한다. ECG 는 창 끝이 어디에 떨어지든 — T 파 사면이든 — 기울기가
있으므로 **늘 가짜 램프가 붙고**, 램프는 저주파가 커서 0.5 Hz 고역통과가
그것을 과도현상으로 바꾼다. `constant`(끝값 유지)는 기울기가 0 이라 DC 만
더하고, 고역통과는 DC 를 깨끗이 지운다.

| 창 확장 | 굴곡 | 남은 변동 | ST 차 | 오프라인차 p99 |
|---|---|---|---|---|
| `odd`, 창의 절반 (전) | 1.229 | 6.7 | +1.61 | 6.19 |
| `constant`, 0.25 s | 0.656 | 3.9 | -1.46 | 5.04 |
| **`constant`, 1 s (채택)** | **0.366** | **5.6** | **+0.80** | **4.88** |
| `constant`, 2 s | 0.551 | 5.9 | +1.14 | 4.98 |
| Gustafsson 초기조건 | 0.199 | 8.3 | -1.36 | 9.63 |

**네 지표가 모두 좋아진다 — 무엇도 팔지 않는다.** 지연도 연산도 그대로다.
Gustafsson(초기조건 최적 추정)은 굴곡이 가장 작지만 남은 변동과 오프라인차가
크게 나빠져 채택하지 않았다.

### 기제 확인 — 그리고 **첫 설명은 틀렸다**

처음에는 «기저선 변동이 있을 때 램프가 생긴다» 고 봤다. 주입 변동을 0~100 %R
로 훑어 보니 **비가 3.3~3.4 로 일정했다** — 변동이 0 이어도 `odd` 가 3.4 배
나쁘다. 원인은 기저선이 아니라 **ECG 자신의 기울기**다.

같은 것을 회귀 시험에서도 확인했다. 시험 신호에 QRS 만 넣었을 때는 `odd` 가
**오히려 굴곡이 작게** 나왔다(0.0016 대 0.0029). **T 파를 넣자 뒤집혔다**
(0.0143 대 0.0063). 창 끝이 **완만한 사면에 떨어질 수 있어야** 재현된다 —
QRS 만 있으면 창 끝은 거의 항상 평평한 곳이다.

> **재사용할 것.** «가장자리 처리» 는 보통 신경 안 쓰는 기본값인데, **창을
> 짧게 쓰는 실시간 경로에서는 가장자리가 결과의 상당 부분**이다. 오프라인은
> 창이 기록 전체라 가장자리가 전체의 0.1 % 지만, 여기서는 미리보기 0.5 s
> 자체가 가장자리다. **같은 필터라도 창이 짧아지면 다른 기본값이 필요하다.**
"""
    DOC.write_text(body)


if __name__ == "__main__":
    raise SystemExit(main())
