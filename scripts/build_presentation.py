#!/usr/bin/env python3
"""졸업 작품 **발표 자료**(16:9, 7 쪽)를 판넬 · 브로셔와 같은 그림 · 수치로 만든다 (docs/38 §13).

    python scripts/build_presentation.py                      # pptx
    python scripts/build_presentation.py --render             # + PDF · 쪽별 PNG 미리보기
    python scripts/build_presentation.py --video-dir <릴리스 video 폴더> --render   # 영상 클립부터 다시 자른다

산출: `results/presentation/presentation.pptx` (+ `.pdf`, `slide-N.png`)

| 쪽 | 내용 | 자료 |
|---|---|---|
| 1 | 연구 개요 — 목표 · 분석 항목 · 비교 실험 구성 · 연구 단계 | 구성도(PowerPoint 도형) |
| 2 | 이론적 배경 — SWT · 확장 칼만 필터 · 잔차 학습 U-Net | T1~T3, 수식 |
| 3 | 결과 ① 잡음 종류와 입력 SNR 에 따른 우세 기법 | F4, F5a, F5b |
| 4 | 결과 ② 가성 QRS군 생성과 실시간 처리의 파형 왜곡 | B3, B4 |
| 5 | 결과 ③ 딥러닝 개선 과정 | F7 |
| 6 | 시연 ① 전시 화면 | 도입부 영상 클립(전원 간섭 장면), 캡처 |
| 7 | 시연 ② 실시간 측정 | 실시간 측정 영상 클립, 상태 수치, 개발 단계 |

지키는 것 (판넬과 같다):
1. **수치는 산출물에서 읽는다** — `build_panel.numbers()` 를 그대로 쓰고, 실시간 왜곡은 `audit_fe_metrics` 로 다시 잰다.
2. **학술 서술 규칙**(`docs/95_academic_style.md`) — 이 파일의 한글 문자열은 `check_academic_style.py` 의 검사 대상이다.
   의문문을 쓰지 않는다(D5).
3. **그림 속 글자** — 본문 대비 0.6 배 이상(95 F 절). 그림마다 놓인 크기에서의 글자 크기를 계산해 출력하고, 미달이면 멈춘다.
4. **그림은 PNG** — 다른 PC 에서 SVG 가 표시되지 않았다(판넬과 같은 이유).
5. **장치 사진은 넣지 않는다** — AFE 담당 조원이 만든다(사용자 지시 2026-11-07).

실시간 측정 영상 · 캡처는 사용자 본인의 심전도라 git 제외 폴더(`data/arduino/captures/`)에서 읽는다. 생성물에 싣는 것은
사용자가 허락하였다(2026-11-05).
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
from PIL import Image

import build_panel as BP
import make_brochure_figs as MB
import make_panel_figs as PF
from panel_eq import render_eq

OUT = Path("results/presentation")
CAP = Path("results/panel/captures")
LIVE = Path("data/arduino/captures")
W, H = 33.867, 19.05                # 16:9 (13.333 × 7.5 in)
X0, X1 = 1.2, W - 1.2
TITLE_PT, BODY_PT, CAP_PT = 26, 15, 12
FIG_MIN_PT = BODY_PT * 0.6          # 그림 속 글자 하한 — 본문의 0.6 배(95 F 절, 브로셔 10:6 비율)
INK, INK2, ACC = BP.INK, BP.INK2, BP.ACC

# 영상 클립 — 릴리스 `video` 의 원본에서 자른다(docs/38 §9). (출력, 원본 파일, 시작 s, 길이 s, ffmpeg 필터, crf)
CLIPS = [
    (CAP / "clip_intro_power.mp4", "intro0.mp4", 38, 14, "fps=30,scale=1280:-2", 27),
    # 실시간 측정 화면은 그래프 영역만 자른다 — 범례 칩 · 해설 글(내부 코드 · 구어, docs/39 §3)이 영상에 남지 않게.
    (LIVE / "clip_live_overlay.mp4", "live2.mp4", 15, 10, "fps=30,crop=816:410:275:150", 24),
]
POSTER_AT = 1.0     # 클립의 첫 화면(포스터)으로 쓸 시각 [s]
FIG_PT = {   # 그림을 그린 크기(cm 폭)와 그림 속 최소 글자(pt) — 놓인 폭에 비례해 커지거나 작아진다
    "F4": (PF.SIZES["F4"][0], 13.0), "F5a": (PF.SIZES["F5a"][0], 13.0), "F5b": (PF.SIZES["F5b"][0], 13.0),
    "F7": (PF.SIZES["F7"][0], 13.0), "T1": (PF.SIZES["T1"][0], 13.0), "T2": (PF.SIZES["T2"][0], 13.0),
    "T3": (PF.SIZES["T3"][0], 13.0), "B3": (MB.SIZES["B3"][0], 6.6), "B4": (MB.SIZES["B4"][0], 6.6),
}


def make_clips(video_dir: Path) -> None:
    """원본 영상에서 짧은 클립을 자른다 — 음성 제거, 30 fps, H.264(crf), 빠른 시작."""
    for out, src, ss, dur, vf, crf in CLIPS:
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(ss), "-t", str(dur), "-i", str(video_dir / src),
                        "-an", "-vf", vf, "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True, timeout=600)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(POSTER_AT), "-i", str(out), "-frames:v", "1",
                        str(poster(out))], check=True, timeout=120)
        print(f"  클립 {out} ({out.stat().st_size / 1e6:.1f} MB)")


def poster(clip: Path) -> Path:
    return clip.with_name(clip.stem + "_poster.png")


def causal_numbers() -> dict:
    """실시간 공통 전처리의 QRS 직후 왜곡 — 브로셔 그림 B4 와 같은 계산(MIT-BIH 기록 100, 평균 박동).
    오프라인 영위상 대비 편차의 최댓값을 R파 진폭 대비 %로, QRS 이후 구간(R파 +40~600 ms)에서 잰다."""
    import audit_fe_metrics as A
    x, r, ramp, _ = A.take("d1")
    outs = {lab: MB._beat_avg(fn(x), r, A.FS) for lab, fn in A.CANDS}
    t, ref = outs["오프라인 영위상"]
    m = (t > 0.04) & (t < 0.6)
    dev = {lab: float(np.abs((outs[lab][1] - ref)[m]).max() / ramp * 100) for lab in outs}
    return {"causal": dev["인과 o1 0.5 Hz"], "block": dev["블록 영위상 0.5 s"]}


def fig_pt(key: str, placed_w: float) -> float:
    w0, pt0 = FIG_PT[key]
    return pt0 * placed_w / w0


# ------------------------------------------------------------------ 쪽 공통
class Deck:
    def __init__(self):
        from pptx import Presentation
        from pptx.util import Cm
        self.prs = Presentation()                    # 기본 양식(Office 테마, 흰 바탕)
        self.prs.slide_width, self.prs.slide_height = Cm(W), Cm(H)
        self.blank = self.prs.slide_layouts[6]
        self.n = 0
        self.fig_check: list[tuple[str, float]] = []

    def page(self, label: str, title: str, notes: str) -> BP.Sheet:
        s = self.prs.slides.add_slide(self.blank)
        self.n += 1
        sh = BP.Sheet(self.prs, s, body=(2.8, H - 1.0))
        head = [(label + "  ", {"bold": True, "color": ACC})] if label else []
        sh.text(X0, 0.75, X1 - X0, 1.3, [head + [(title, {"bold": True, "color": INK if label else ACC})]],
                size=TITLE_PT)
        sh.text(X1 - 3.0, H - 0.95, 3.0, 0.5, f"{self.n} / 7", size=10, color=INK2, align="r")
        s.notes_slide.notes_text_frame.text = notes
        return sh

    def figure(self, sh: BP.Sheet, key: str, path: Path, x, y, w, h):
        xx, yy, ww, hh = sh.image(path, x, y, w, h, valign="top")
        self.fig_check.append((f"{key} ({path.name})", fig_pt(key, ww)))
        return xx, yy, ww, hh


def box(sh: BP.Sheet, x, y, w, h, fill="f4f3ef", line="d9d5cc", radius=0.06):
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.oxml.ns import qn
    from pptx.util import Cm, Pt
    b = sh.s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Cm(x), Cm(y), Cm(w), Cm(h))
    b.adjustments[0] = radius
    b.fill.solid(); b.fill.fore_color.rgb = RGBColor.from_string(fill)
    b.line.color.rgb = RGBColor.from_string(line); b.line.width = Pt(0.75)
    st = b._element.find(qn("p:style"))
    if st is not None:
        b._element.remove(st)
    return b


def key_points(sh: BP.Sheet, x, y, w, h, pts: list[str], size=16):
    """요지 상자 — 옅은 바탕 위에 문단 간격을 둔 요지 문장. 결과 쪽마다 같은 모양으로 둔다."""
    from pptx.dml.color import RGBColor
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.util import Cm, Pt
    b = box(sh, x, y, w, h, radius=0.04)
    tf = b.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for m in ("margin_left", "margin_right"):
        setattr(tf, m, Cm(0.5))
    for m in ("margin_top", "margin_bottom"):
        setattr(tf, m, Cm(0.3))
    for i, t in enumerate(pts):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.space_after = Pt(8)
        para.alignment = PP_ALIGN.LEFT              # 도형 글은 기본이 가운데 정렬이다
        r = para.add_run(); r.text = t
        r.font.name, r.font.size, r.font.color.rgb = BP.FONT, Pt(size), RGBColor(*INK)


def arrow(sh: BP.Sheet, xa, ya, xb, yb):
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_CONNECTOR
    from pptx.oxml.ns import qn
    from pptx.util import Cm, Pt
    c = sh.s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Cm(xa), Cm(ya), Cm(xb), Cm(yb))
    c.line.color.rgb = RGBColor(0x52, 0x51, 0x4e); c.line.width = Pt(1.4)
    st = c._element.find(qn("p:style"))
    if st is not None:
        c._element.remove(st)
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))


def movie(sh: BP.Sheet, clip: Path, poster: Path, x, y, w):
    """영상 클립을 넣는다(클릭하면 재생). 첫 화면은 같은 장면의 캡처."""
    from pptx.util import Cm
    iw, ih = Image.open(poster).size
    h = w * ih / iw
    sh.s.shapes.add_movie(str(clip), Cm(x), Cm(y), Cm(w), Cm(h), poster_frame_image=str(poster),
                          mime_type="video/mp4")
    sh.img_area += w * h
    return h


# ------------------------------------------------------------------ 1 쪽 연구 개요
PSTEPS = [("① 비교 체계", "동일 입력 · 지표", "1쪽"),
          ("② 조건별 비교", "입력 SNR · 잡음 종류", "3쪽"),
          ("③ 안전성 검증", "가성 파형 · 왜곡", "4쪽"),
          ("④ 딥러닝 개선", "학습 목표 재설정", "5쪽"),
          ("⑤ 시연", "전시 화면 · 실시간 측정", "6~7쪽")]


def p1_overview(d: Deck, n: dict):
    sh = d.page("", "연구 개요", (
        "본 연구는 자체 제작한 심전도 취득 장치의 잔류 잡음을 줄이는 후처리 기법을 정량적으로 비교하였다. 네 계열의 기법을 "
        "동일한 입력, 공통 전처리, 동일한 평가 지표로 비교하였으며, 평가 체계는 기법 구현보다 먼저 확정하였다. 분석 항목은 "
        "조건별 적합 기법, 가성 QRS군 생성, 실시간 처리의 파형 왜곡의 세 가지이다."))
    B = {"bold": True}
    left = [
        [("연구 목표", {"bold": True, "color": ACC})],
        "자체 제작한 심전도(ECG) 취득 장치(AD8232, Arduino Uno)의 잔류 잡음을 파형을 보존하면서 줄이는 디지털 "
        "후처리 기법을 규명한다.",
        "",
        [("분석 항목", {"bold": True, "color": ACC})],
        [("① ", B), ("잡음 종류와 입력 SNR에 따른 적합 기법의 변화", {})],
        [("② ", B), ("입력에 없는 파형(가성 QRS군)의 생성", {})],
        [("③ ", B), ("실시간 처리에서의 파형 왜곡", {})],
        "",
        [("비교 대상", {"bold": True, "color": ACC})],
        "고전 필터 · 웨이블릿 임계값 처리 · 칼만 필터 · 딥러닝",
        "",
        [("데이터", {"bold": True, "color": ACC})],
        "합성 ECG(McSharry 모델), MIT-BIH 부정맥 데이터베이스와 NSTDB 실측 잡음",
    ]
    lw = 12.3
    sh.text(X0, 2.9, lw, 11.6, left, size=BODY_PT)
    s = 0.74                                           # 구성도 25.7 × 13.2 cm 설계 → 19.0 × 9.8 cm
    xd = X1 - BP.D1_W * s
    BP.draw_pipeline(sh, xd, 2.95, s=s, fpt=11)
    d.fig_check.append(("구성도(PowerPoint 도형)", 11.0))
    sh.text(xd, 2.95 + BP.D1_H * s + 0.1, BP.D1_W * s, 0.9,
            "그림 1. 비교 실험 구성. 평가 지표 4범주: 재구성 오차 · 파형 보존 · R파 검출 · 주파수 특성.",
            size=CAP_PT, color=INK2)
    old = BP.STEPS
    BP.STEPS = PSTEPS
    try:
        BP.draw_steps(sh, X0, 14.75, X1 - X0, fpt=13, h=2.4, gap=0.6)
    finally:
        BP.STEPS = old
    sh.text(X0, H - 0.95, 20.0, 0.5, "본 시스템은 연구용 신호처리 시스템이며 임상 진단 용도가 아니다.", size=10, color=INK2)


# ------------------------------------------------------------------ 2 쪽 이론적 배경
def p2_theory(d: Deck, n: dict):
    sh = d.page("", "이론적 배경", (
        "비교한 기법 가운데 세 계열의 원리를 정리하였다. 정상 웨이블릿 변환은 대역별 임계값 처리, 확장 칼만 필터는 심전도의 "
        "동역학 모델에 기반한 상태 추정, 잔차 학습 U-Net은 잡음 추정 후 감산 방식이다. 각 기법의 장점과 한계는 이후 결과의 "
        "해석 근거가 된다."))
    B = {"bold": True}
    eqd = OUT / "eq"
    cards = [
        ("T1", "T1_swt.png", "① 정상 웨이블릿 변환(SWT) 적응 임계값 처리",
         "입력을 다섯 주파수 대역의 계수로 분해하고, 대역별 임계값을 적용한 뒤 재합성한다.",
         [BP.EQ("eq_swt", r"\lambda_j = k_j\,\sigma\,\sqrt{2\ln N}\;\left(1-(1-\rho)\,g(t)\right)")],
         [("장점 ", B), (f"무잡음 입력의 왜곡이 가장 작다(출력 SNR {n['clean_swt']:.1f} dB).", {})],
         [("한계 ", B), ("기저선 변동 · 임펄스 잡음의 제거 성능이 낮다.", {})]),
        ("T2", "T2_kalman.png", "② 확장 칼만 필터(McSharry 모델)",
         "심전도 한 주기를 심장 위상 θ 위 가우시안 함수의 합으로 모델링하고 상태 [θ, z]를 추정한다.",
         [BP.EQ("eq_eks", r"\dot{\theta} = \omega,\quad \dot{z} = -\sum_i \frac{\alpha_i\,\omega}{b_i^2}\,"
                          r"\Delta\theta_i \exp\!\left(-\frac{\Delta\theta_i^2}{2\,b_i^2}\right)")],
         [("장점 ", B), ("모델과 일치하는 합성 ECG에서 SNR 개선량이 크다(+16.2 dB).", {})],
         [("한계 ", B), ("R파 오검출 위치에 가성 QRS군을 생성한다.", {})]),
        ("T3", "T3_unet.png", "③ 잔차 학습 1D U-Net",
         "인코더 · 디코더와 스킵 연결로 잡음을 추정하고, 입력에서 추정 잡음을 감산한다.",
         [BP.EQ("eq_res", r"\hat{x} = y - f_\theta(y)")],
         [("장점 ", B), (f"입력 SNR {n['cross_db']:.0f} dB 미만 · 임펄스 잡음에서 우수하다.", {})],
         [("한계 ", B), (f"무잡음 입력도 변형한다(출력 SNR {n['clean_m06']:.1f} dB, 보존 손실 적용 시 "
                       f"{n['clean_l6']:.1f} dB).", {})]),
    ]
    gap = 0.6
    cw = (X1 - X0 - 2 * gap) / 3
    y0, y1 = 2.9, H - 1.15
    for i, (key, fig, head, princ, eqs, pro, con) in enumerate(cards):
        x = X0 + i * (cw + gap)
        box(sh, x, y0, cw, y1 - y0)
        xi, wi = x + 0.35, cw - 0.7
        y = y0 + 0.3
        h = BP._text_h([head], wi, 15)
        sh.text(xi, y, wi, h, [[(head, {"bold": True, "color": ACC})]], size=15)
        y += h + 0.15
        h = BP._text_h([princ], wi, 14)
        sh.text(xi, y, wi, h, princ, size=14)
        y += h + 0.2
        for eq in eqs:
            y += sh.equation(render_eq(eq, 20, eqd), xi, y, wi) + 0.2
        _, _, _, fh = d.figure(sh, key, BP.PFIG / fig, xi, y, wi, 6.6)
        y += fh + 0.25
        sh.text(xi, y, wi, y1 - y - 0.2, [pro, con], size=14)


# ------------------------------------------------------------------ 3 쪽 결과 ①
def p3_noise(d: Deck, n: dict):
    m = n["noise_means"]
    sh = d.page("결과 ①", "잡음 종류와 입력 SNR에 따른 우세 기법", (
        f"MIT-BIH 실측 기록에서 딥러닝의 우위는 입력 SNR 약 {n['cross_db']:.0f} dB 미만에 한정되었다. 잡음 종류별로는 "
        "근전도와 임펄스 잡음에서 딥러닝이, 기저선 변동에서는 공통 전처리 단독이, 전원선 간섭에서는 SWT가 가장 높았다. "
        "따라서 기법의 우열은 잡음 종류와 입력 SNR을 지정해야 정의된다."))
    sc = 1.08
    cells = [("F4", "F4_crossover.png", "(가)"), ("F5a", "F5a_dl_wins.png", "(나)"), ("F5b", "F5b_classic_wins.png", "(다)")]
    ws = [PF.SIZES[k][0] * sc for k, _, _ in cells]
    g = (X1 - X0 - sum(ws)) / 2
    x, xs, y = X0, [], 3.15
    for (k, fig, lab), w in zip(cells, ws):
        d.figure(sh, k, BP.PFIG / fig, x, y, w, PF.SIZES[k][1] * sc)
        sh.text(x, y - 0.6, 1.5, 0.6, [[(lab, {"bold": True})]], size=CAP_PT)
        xs.append(x); x += w + g
    yc = y + PF.SIZES["F4"][1] * sc + 0.1
    sh.text(xs[0], yc, ws[0], 1.4, "(가) 공통 전처리 대비 딥러닝의 SNR 개선량(입력 SNR별).", size=CAP_PT, color=INK2)
    sh.text(xs[1], yc, xs[2] + ws[2] - xs[1], 1.4,
            f"(나) 딥러닝 우세 잡음, (다) 고전 처리 우세 잡음. MIT-BIH, 입력 SNR +10 dB, {n['noise_n']}개 구간 평균 SNR "
            "개선량[dB], 음영 = 최고, 아래 행 = 잔차.", size=CAP_PT, color=INK2)
    pts = [
        f"• MIT-BIH에서 딥러닝의 우위는 입력 SNR 약 {n['cross_db']:.0f} dB 미만에 한정되었고, 그 이상에서는 "
        "공통 전처리 단독이 우세하였다(가).",
        f"• 근전도 · 임펄스 잡음에서는 딥러닝이 SWT보다 각각 {m['ma_synth']['M08'] - m['ma_synth']['M04']:.1f} dB, "
        f"{m['impulse']['M08'] - m['impulse']['M04']:.1f} dB 높았다(나).",
        "• 기저선 변동은 공통 전처리 단독, 전원선 간섭은 SWT가 가장 높았고, 딥러닝은 두 잡음에서 잔차를 남겼다(다).",
    ]
    yt = yc + 1.6
    key_points(sh, X0, yt, X1 - X0, H - 1.25 - yt, pts)


# ------------------------------------------------------------------ 4 쪽 결과 ②
def p4_safety(d: Deck, n: dict, c: dict):
    sh = d.page("결과 ②", "가성 QRS군 생성과 실시간 처리의 파형 왜곡", (
        "칼만 필터는 R파 오검출 위치에 박동 형태의 파형을 출력하였다. 이 오류는 오차 에너지 지표로는 드러나지 않아, 평균 "
        "박동과의 유사도로 별도 판정하였다. 실시간 처리에서는 미래 표본을 사용할 수 없으므로 인과 필터가 QRS 직후 파형을 "
        "왜곡하며, 0.5초 블록 영위상 필터는 지연을 대가로 이 왜곡을 줄였다."))
    gap = 1.0
    cw = (X1 - X0 - gap) / 2
    cols = [
        ("B3", "B3_phantom.png", "칼만 필터의 가성 QRS군(MIT-BIH)", [
            f"• 칼만 필터는 R파 오검출 위치에 박동 형태의 파형을 출력하였다. 가성 QRS군은 {n['ph_M05']}건으로 SWT 대비 약 "
            f"{n['ph_x_swt']:.0f}배, 딥러닝 대비 약 {n['ph_x_dl']:.0f}배였다(입력 SNR −5 · 0 dB).",
            "• 오차 에너지 지표에서는 드러나지 않았으며, 위양성 검출 가운데 평균 박동과의 상관계수가 0.8을 넘는 경우로 "
            "판정하였다.",
        ], "그림 2. 칼만 필터 출력(음영: 박동이 없는 위치)과 기법별 가성 QRS군 수."),
        ("B4", "B4_causal.png", "실시간 공통 전처리의 QRS 직후 왜곡(MIT-BIH)", [
            f"• 인과 고역통과 필터는 QRS 직후 구간에서 오프라인 영위상 필터 대비 최대 {c['causal']:.1f} %R의 편차를 보였다.",
            f"• 0.5초 블록 영위상 필터의 편차는 최대 {c['block']:.1f} %R로 오프라인 처리와 거의 같았다. 대신 약 0.5초의 "
            "처리 지연이 추가된다.",
        ], "그림 3. 공통 전처리 방식별 평균 박동의 편차(기록 100, R파 진폭 대비 %). 편차는 R파 이후 40~600 ms 구간에서 측정."),
    ]
    fh = max(cw * MB.SIZES[k][1] / MB.SIZES[k][0] for k, *_ in cols)    # 두 그림 중 큰 높이 — 캡션 줄을 맞춘다
    for i, (k, fig, head, pts, capt) in enumerate(cols):
        x = X0 + i * (cw + gap)
        sh.text(x, 2.9, cw, 0.8, [[(head, {"bold": True, "color": ACC})]], size=16)
        d.figure(sh, k, BP.PFIG / fig, x, 3.75, cw, fh)
        y = 3.75 + fh + 0.1
        hc = BP._text_h([capt], cw, CAP_PT)
        sh.text(x, y, cw, hc, capt, size=CAP_PT, color=INK2)
        y += hc + 0.35
        key_points(sh, x, y, cw, H - 1.25 - y, pts)


# ------------------------------------------------------------------ 5 쪽 결과 ③
def p5_flow(d: Deck, n: dict):
    st = PF.flow_stats()
    m07 = next(v for (a, b, k), v in st.items() if a == "m07_l1" and k == "all")
    sh = d.page("결과 ③", "딥러닝 개선 과정", (
        "기준 모델에서 출발하여 구조, 규모, 학습 기법, 학습 목표의 순서로 개선을 시도하였다. 모든 비교는 동일 평가 집합에서 "
        "동일 시드 대응 t-검정으로 판정하고 Holm 보정을 적용하였다. 유의한 성능 향상은 학습 목표를 바꾼 두 시도에서만 "
        "확인되었으며, 고 SNR 구간의 열세는 단일 모델의 절충으로 해석된다."))
    w7 = X1 - X0 - 1.2
    h7 = w7 * PF.SIZES["F7"][1] / PF.SIZES["F7"][0]
    d.figure(sh, "F7", BP.PFIG / "F7_flow.png", X0 + 0.6, 2.85, w7, h7)
    y = 2.85 + h7 + 0.1
    capt = ("그림 4. 딥러닝 개선 과정. 기준 모델 대비 SNR 개선량 변화(동일 평가 집합, 동일 시드 대응 t-검정, "
            f"시드 n = {n['seed_lo']}~{n['seed_hi']}, 비교 {n['n_tests']}개 Holm 보정).")
    sh.text(X0, y, X1 - X0, 0.6, capt, size=CAP_PT, color=INK2)
    pts = [
        "• 유의한 성능 향상은 학습 목표를 바꾼 두 시도(무잡음 입력 보존 손실, SNR 조건화)에서만 확인되었다.",
        f"• 구조 변경 가운데 SWT 직렬 결합은 성능 저하를 보였다(p_Holm = {m07[3]:.2f}).",
        "• 고 SNR 구간의 열세는 단일 모델이 전 입력 SNR 대역을 학습하며 절충한 결과로 해석된다.",
        f"• 15~20 dB 전용 모델은 기준 모델보다 {n['band_hi']:+.1f} dB 높았으나 유의하지 않았다"
        f"(p_Holm = {n['band_hi_ph']:.2f}, 시드 {len(next(v for (a, b, k), v in st.items() if a == 'm06_l1_band_hi')[1])}개).",
    ]
    yt = y + 0.75
    sh.text(X0, yt, X1 - X0, H - 1.1 - yt, pts, size=BODY_PT)


# ------------------------------------------------------------------ 6 쪽 시연 ①
def p6_exhibit(d: Deck):
    sh = d.page("시연 ①", "전시 화면", (
        "전시 화면의 도입부는 잡음의 발생 원인을 세 장면으로 제시한다. 전원 간섭, 호흡에 따른 기저선 변동, 근육 활동에 따른 "
        "근전도 잡음이다. 이후 동일 심장 박동의 입력과 처리 결과를 겹쳐 표시하고, 실험실 화면에서 기법별 출력을 같은 축으로 "
        "비교한다. 영상은 전원 간섭 장면이다."))
    vw = 17.6
    vh = movie(sh, CAP / "clip_intro_power.mp4", CAP / "intro_05_power_wave.png", X0, 2.9, vw)   # 클립 13 s 의 장면
    y = 2.9 + vh + 0.1
    capt = "영상 1. 도입부의 전원 간섭 장면(14 s). 전원선 주변의 전기장이 신체를 거쳐 심전도 잡음으로 나타나는 과정."
    hc = BP._text_h([capt], vw, CAP_PT)
    sh.text(X0, y, vw, hc, capt, size=CAP_PT, color=INK2)
    y += hc + 0.3
    iw, ih = Image.open(CAP / "intro_scenes_strip.png").size
    hs = H - 1.75 - y
    _, _, _, hs = sh.image(CAP / "intro_scenes_strip.png", X0, y, min(vw, hs * iw / ih), hs, valign="top")
    sh.text(X0, y + hs + 0.08, vw, 0.6, "그림 5. 도입부의 잡음 원인 세 장면: 전원 간섭 · 기저선 변동 · 근전도.",
            size=CAP_PT, color=INK2)
    xr = X0 + vw + 0.8
    wr = X1 - xr
    _, _, _, h1 = sh.image(CAP / "intro_08_heart_overlay_crop.png", xr, 2.9, wr, 6.4, valign="top")
    yy = 2.9 + h1 + 0.08
    sh.text(xr, yy, wr, 0.6, "그림 6. 동일 박동의 입력(잡음)과 처리 결과의 중첩 표시.", size=CAP_PT, color=INK2)
    yy += 0.95
    _, _, _, h2 = sh.image(CAP / "lab_01_compare.png", xr, yy, wr, H - 1.75 - yy, valign="top")
    sh.text(xr, yy + h2 + 0.08, wr, 0.6, "그림 7. 실험실 화면: 입력과 기법별 출력의 동일 축 비교.", size=CAP_PT,
            color=INK2)


# ------------------------------------------------------------------ 7 쪽 시연 ②
def p7_live(d: Deck):
    sh = d.page("시연 ②", "실시간 측정", (
        "자체 장치에서 취득한 심전도를 실시간으로 처리한 화면이다. 입력 위에 SWT와 잔차 학습 U-Net의 출력을 겹쳐 표시하였다. "
        "처리 지연 644 ms는 공통 전처리의 블록 영위상 필터에 필요한 미리보기 구간 548 ms와 기법 처리 지연 96 ms의 합이며, "
        "측정 중 샘플 손실은 없었다. 실측 신호에는 기준 신호가 없으므로 이 화면에는 SNR을 표시하지 않는다."))
    vw = 16.6
    clip = LIVE / "clip_live_overlay.mp4"
    vh = movie(sh, clip, poster(clip), X0, 2.9, vw)
    capt = ("영상 2. 자체 장치 신호의 실시간 처리(10 s, 그래프 영역). 입력(회색) 위에 SWT(파랑)와 잔차 학습 U-Net(노랑) "
            "출력을 중첩 표시.")
    hc = BP._text_h([capt], vw, CAP_PT)
    sh.text(X0, 2.9 + vh + 0.1, vw, hc, capt, size=CAP_PT, color=INK2)
    xr = X0 + vw + 0.8
    wr = X1 - xr
    # 측정 경로 — 상자 넷
    sh.text(xr, 2.9, wr, 0.7, [[("측정 경로", {"bold": True, "color": ACC})]], size=15)
    steps = ["전극 ·\nAD8232", "Arduino Uno\n500 Hz 표본화", "PC 처리\n250 Hz · 기법", "화면\n중첩 표시"]
    bw, gp = (wr - 0.45 * 3) / 4, 0.45
    for i, t in enumerate(steps):
        x = xr + i * (bw + gp)
        b = box(sh, x, 3.7, bw, 1.55, fill="e4ecf7" if i == 2 else "f4f3ef")
        _fill(b, t, 11)
        if i:
            arrow(sh, x - gp + 0.05, 4.47, x - 0.05, 4.47)
    # 상태 수치 — 큰 수 셋
    sh.text(xr, 5.65, wr, 0.7, [[("측정 상태(경과 231.8 s)", {"bold": True, "color": ACC})]], size=15)
    stats = [("644 ms", "처리 지연", "미리보기 548 ms\n+ 기법 96 ms"), ("0", "샘플 손실", "전극 분리 0\n재동기화 0"),
             ("0.53", "실시간 계수", "처리 시간 ÷ 신호 시간\n(1 미만: 실시간 처리)")]
    sw = (wr - 0.4 * 2) / 3
    for i, (big, lab, sub) in enumerate(stats):
        x = xr + i * (sw + 0.4)
        box(sh, x, 6.45, sw, 3.6, fill="ffffff", line="c9c5bb")
        sh.text(x, 6.6, sw, 1.2, [[(big, {"bold": True, "color": ACC})]], size=26, align="c")
        sh.text(x, 7.85, sw, 0.6, [[(lab, {"bold": True})]], size=13, align="c")
        sh.text(x + 0.15, 8.5, sw - 0.3, 1.4, sub.split("\n"), size=11, color=INK2, align="c")
    sh.text(xr, 10.25, wr, 1.5, "실측 신호에는 기준 신호가 없으므로 이 화면은 파형과 처리 상태만 표시하며, 기법 간 정량 "
            "비교는 3~5쪽의 결과로 제시한다.", size=12, color=INK2)
    # 개발 단계 — 아래 전폭
    yb = 2.9 + vh + 0.1 + hc + 0.45
    sh.text(X0, yb, 10.0, 0.7, [[("시연 화면의 개발 단계", {"bold": True, "color": ACC})]], size=15)
    yb += 0.75
    hb = H - 1.15 - yb
    stages = [("① 벤치마크 화면", "기록 신호(MIT-BIH)에 기준 신호를 겹쳐 기법별 출력을 비교", Path("results/panel/demo_mode_b.png")),
              ("② 실시간 측정 화면", "자체 장치를 연결하여 실시간 처리 경로를 검증(2026-10-16)",
               LIVE / "live_02_split_t32.png"),
              ("③ 최종 GUI", "벤치마크 비교와 실시간 측정을 한 프로그램의 탭으로 통합, 실시간 측정 탭은 ②와 동일", None)]
    gp = 0.7
    bw = (X1 - X0 - 2 * gp) / 3
    for i, (head, desc, img) in enumerate(stages):
        x = X0 + i * (bw + gp)
        box(sh, x, yb, bw, hb, fill="e4ecf7" if i == 2 else "f4f3ef")
        tw = bw - 0.5
        if img is not None:
            iw = min(3.6, hb - 0.4) * 1.45
            sh.image(img, x + bw - iw - 0.2, yb + 0.2, iw, hb - 0.4)
            tw = bw - iw - 0.6
        sh.text(x + 0.25, yb + 0.2, tw, hb - 0.4, [[(head, {"bold": True})], desc], size=12)
        if i:
            arrow(sh, x - gp + 0.08, yb + hb / 2, x - 0.08, yb + hb / 2)


def _fill(shape, text: str, size: float):
    from pptx.dml.color import RGBColor
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.util import Cm, Pt
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, Cm(0.05))
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = line
        r.font.name, r.font.size, r.font.color.rgb = BP.FONT, Pt(size), RGBColor(*INK)


# ------------------------------------------------------------------
def build(n: dict, c: dict) -> Path:
    d = Deck()
    p1_overview(d, n)
    p2_theory(d, n)
    p3_noise(d, n)
    p4_safety(d, n, c)
    p5_flow(d, n)
    p6_exhibit(d)
    p7_live(d)
    print("  그림 속 최소 글자(놓인 크기 기준):")
    bad = []
    for name, pt in d.fig_check:
        print(f"    {name}: {pt:.1f} pt")
        if pt < FIG_MIN_PT - 1e-6:
            bad.append(name)
    assert not bad, f"그림 속 글자가 본문({BODY_PT} pt)의 0.6 배({FIG_MIN_PT:.1f} pt) 미만이다: {bad}"
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "presentation.pptx"
    d.prs.save(str(p))
    return p


def render(pptx: Path, dpi: int = 110) -> None:
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(pptx.parent), str(pptx)],
                   check=True, capture_output=True, timeout=300)
    for old in pptx.parent.glob("slide-*.png"):
        old.unlink()
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), str(pptx.with_suffix(".pdf")), str(pptx.parent / "slide")],
                   check=True, timeout=300)
    print(f"  {pptx.with_suffix('.pdf')} · slide-N.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video-dir", type=Path, default=None,
                    help="릴리스 video 의 원본 영상 폴더(intro0.mp4 · live2.mp4). 주면 영상 클립부터 다시 자른다. "
                         "없으면 이미 잘라 둔 클립(results/panel/captures · data/arduino/captures)을 쓴다")
    ap.add_argument("--render", action="store_true", help="LibreOffice 로 PDF 와 쪽별 PNG 미리보기도 만든다")
    a = ap.parse_args()
    if a.video_dir:
        make_clips(a.video_dir)
    missing = [str(o) for o, *_ in CLIPS if not o.exists()]
    if missing:
        raise SystemExit(f"영상 클립이 없다: {missing} — --video-dir <릴리스 video 폴더> 로 먼저 자른다")
    n = BP.numbers()
    c = causal_numbers()
    print("[causal]", {k: round(v, 2) for k, v in c.items()})
    p = build(n, c)
    print(f"  {p} ({p.stat().st_size / 1e6:.1f} MB)")
    if a.render:
        render(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
