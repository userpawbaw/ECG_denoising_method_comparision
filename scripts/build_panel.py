#!/usr/bin/env python3
"""전시 판넬(A1) · 브로셔(A4) **시안**을 학과 양식 위에 만든다 (docs/94_panel_plan.md).

    python scripts/build_panel.py --template <A00_panel 양식.pptx>            # pptx 만
    python scripts/build_panel.py --template <양식.pptx> --render             # + PDF · PNG 미리보기

산출: `results/panel/`
    panel_draft.pptx / .png      A1 세로 59.4 x 84.1 cm, 양식의 네 구역에 그림과 문안
    brochure_draft.pptx / .png   A4 세로, 브로셔 양식(①목표 ②개요 ③내용 및 이미지)의 순서

**시안이다.** 문안과 배치를 팀이 보고 고치기 위한 판이고, 최종 판은 양식 파일에서
손으로 다듬는다. 그래서 두 가지만 지킨다.

1. **수치는 산출물에서 읽는다** — 판넬 문안의 숫자를 손으로 적으면 그림·보고서와
   갈라진다(O-27). 아래 `numbers()` 가 전부 읽어 온다.
2. **그림 면적 비율을 잰다** — 「이미지가 절반 이상」이 요구라서, 본문 구역(머리·꼬리
   제외) 대비 그림 면적을 계산해 출력한다.

장비 사진 · 실시간 화면 사진은 저장소에 없다 — 그 자리는 **빈 틀**로 남기고 틀 안에
무엇을 넣을지 적는다.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
from PIL import Image

import make_panel_figs as PF
from panel_eq import render_eq

OUT = Path("results/panel")
SL = Path("results/slides")
DG = SL / "diagrams"     # mermaid 블록선도 (scripts/make_diagrams.py)
PFIG = OUT / "figs"      # 판넬 전용 그림 (scripts/make_panel_figs.py, 칸 크기 그대로)
FONT = "맑은 고딕"
INK = (0x1a, 0x16, 0x13)
INK2 = (0x5a, 0x55, 0x50)
ACC = (0x18, 0x4f, 0x95)


# ------------------------------------------------------------------ 수치
def numbers() -> dict:
    """판넬 문안에 들어가는 수치 — 전부 산출물에서 읽는다."""
    import pandas as pd

    from make_slides import _paired
    n: dict = {}
    s = json.loads(Path("results/d1/phantom/summary.json").read_text())
    thr = str(s["threshold"])
    for m in ("M05", "M04", "M06", "M_FE"):
        n[f"ph_{m}"] = s["methods"][m]["beatlike"][thr]
    n["ph_x_swt"] = n["ph_M05"] / max(n["ph_M04"], 1)
    n["ph_x_dl"] = n["ph_M05"] / max(n["ph_M06"], 1)
    for k, (sw, arm, base, key) in {
        "m07": ("structure", "m07_l1", "m06_l1", "fixed_snr_imp_scaled"),
        "m08": ("structure", "m08_l1", "m06_l1", "fixed_snr_imp_scaled"),
        "l6_hi": ("l6cond", "m06_l6", "m06_l1", "band_15_20"),
        "cond_hi": ("cond", "m06_cond", "m06_l1", "band_15_20"),
    }.items():
        m, d, p = _paired(sw, arm, base, key)
        n[k], n[k + "_p"] = m, p
    # 딥러닝(M08)이 front-end 위에 더하는 값 — D1, 입력 SNR 별 기록 평균 (5.8.2)
    d = pd.read_parquet("results/d1/exp_a/metrics.parquet")
    d = d[(d.metric == "snr_imp_scaled") & d.method.isin(["M08", "M_FE"])]
    t = d.groupby(["snr_in_target", "record", "method"]).value.mean().unstack()
    g = (t["M08"] - t["M_FE"]).groupby(level=0).mean()
    n["dl_gain_lo"] = float(g.loc[-5.0])
    xs, ys = g.index.to_numpy(float), g.to_numpy(float)
    k = int(np.argmax(ys < 0))                       # 처음 음수가 되는 칸
    n["cross_db"] = float(xs[k - 1] + (xs[k] - xs[k - 1]) * ys[k - 1] / (ys[k - 1] - ys[k]))
    c = pd.read_parquet("results/d1/exp_c/metrics.parquet")
    c = c[c.metric == "snr_out_strict"].groupby("method").value.mean()
    n["clean_m06"], n["clean_l6"], n["clean_swt"] = (float(c[m]) for m in ("M06", "M06-L6", "M04"))
    # 가성 파형 생성 검증 — 박동 삭제 조건(p1)의 생성 에너지, D1 (보고서 5.8.4)
    e = pd.read_csv("results/d1/exp_e/probe.csv")
    e = e[e.probe == "p1"].groupby("method").halluc_energy.mean()
    n["he"] = {m: float(v) for m, v in e.items()}
    n["he_x_bp"] = n["he"]["M01"] / n["he"]["M06"]
    # 판정 해상도(그림 S22) — 같은 설정의 시드 3 회 학습: 시드별 검증 잡음 대 고정 평가 집합 (F-40 · F-41)
    import torch
    seed_best = [float(torch.load(f"results/d1/{r}/best.pt", map_location="cpu", weights_only=False)["best_metric"])
                 for r in ("m06_l1", "m06_l1_s1", "m06_l1_s2")]
    cal = pd.read_csv("results/ext/calib/sweep.csv")
    n["sd_seed"] = float(np.std(seed_best, ddof=1))
    n["rng_seed"] = float(np.ptp(seed_best))
    n["sd_fixed"] = float(cal[cal.arm == "m06_l1"].fixed_snr_imp_scaled.std(ddof=1))
    n["l6_all"] = _paired("structure", "m06_l6", "m06_l1", "fixed_snr_imp_scaled")[0]
    return n


# ------------------------------------------------------------------ 수식
BODY_PT = 18            # 본문 글자 크기 (A1 판넬, 1~2 m 거리에서 읽는 기준 18~24 pt)
CAP_PT = 15             # 그림 캡션
OVERVIEW_PT = 22        # 작품 개요 — 양식 안내문의 지정값
EQ_PT = BODY_PT * 1.5   # 수식 글자 크기 — 본문의 1.5 배


class EQ(tuple):
    """카드 안의 수식 한 줄: (이름, LaTeX 원문)."""
    def __new__(cls, name: str, tex: str):
        return super().__new__(cls, (name, tex))


def _text_h(paras, width_cm: float, size_pt: float) -> float:
    """글상자 높이 추정(cm). 한글 1 em, 영문 · 숫자 0.55 em, 공백 0.3 em, 줄 간격 1.2 em."""
    import math
    em = size_pt * 0.03528
    lines = 0
    for para in paras:
        runs = para if isinstance(para, list) else [(para, {})]
        w = 0.0
        for txt, st in runs:
            k = st.get("size", size_pt) / size_pt
            for ch in txt:
                w += k * (1.0 if ord(ch) > 0x2E80 else 0.3 if ch == " " else 0.58)
        lines += max(1, math.ceil(w * em * 1.06 / width_cm))   # 1.06: 줄바꿈이 낱말 단위라 생기는 빈칸
    return lines * 1.25 * em + 0.1


# ------------------------------------------------------------------ pptx 도우미
class Sheet:
    """cm 단위로 놓고, 그림 면적을 센다."""

    def __init__(self, prs, slide, body: tuple[float, float]):
        from pptx.util import Cm, Pt
        self.prs, self.s, self.Cm, self.Pt = prs, slide, Cm, Pt
        self.body = body                              # (y0, y1) — 면적 비율의 분모
        self.img_area = 0.0

    def text(self, x, y, w, h, paras, size=18, color=INK, bold=False, align="l"):
        from pptx.dml.color import RGBColor
        from pptx.enum.text import PP_ALIGN
        tb = self.s.shapes.add_textbox(self.Cm(x), self.Cm(y), self.Cm(w), self.Cm(h))
        tf = tb.text_frame
        tf.word_wrap = True
        for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
            setattr(tf, m, 0)
        if isinstance(paras, str):
            paras = [paras]
        for i, para in enumerate(paras):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER}[align]
            runs = para if isinstance(para, list) else [(para, {})]
            for txt, st in runs:
                r = p.add_run()
                r.text = txt
                f = r.font
                f.name, f.size = FONT, self.Pt(st.get("size", size))
                f.bold = st.get("bold", bold)
                f.color.rgb = RGBColor(*st.get("color", color))
        return tb

    def image(self, path: Path, x, y, w, h, valign="center"):
        """상자 (x, y, w, h) 안에 비율을 지켜 맞춘다. 실제로 놓인 크기를 돌려준다."""
        iw, ih = Image.open(path).size
        sc = min(w / iw, h / ih)
        ww, hh = iw * sc, ih * sc
        xx, yy = x + (w - ww) / 2, (y if valign == "top" else y + (h - hh) / 2)
        self.s.shapes.add_picture(str(path), self.Cm(xx), self.Cm(yy), self.Cm(ww), self.Cm(hh))
        self.img_area += ww * hh
        return xx, yy, ww, hh

    def equation(self, path: Path, x, y, max_w) -> float:
        """수식 그림을 원래 크기(600 dpi)로 놓는다. 칸보다 넓으면 줄인다. 놓인 높이(cm)를 돌려준다.
        수식은 글이므로 그림 면적 비율에 넣지 않는다."""
        iw, ih = Image.open(path).size
        w, h = iw / 600 * 2.54, ih / 600 * 2.54
        if w > max_w:
            print(f"  경고: 수식 {path.name} 이 칸보다 넓어 {max_w / w:.0%} 로 줄였다")
            w, h = max_w, h * max_w / w
        self.s.shapes.add_picture(str(path), self.Cm(x), self.Cm(y), self.Cm(w), self.Cm(h))
        return h

    def frame(self, x, y, w, h, label):
        """사진이 들어갈 빈 틀 — 저장소에 없는 자료. 면적은 그림으로 센다(채울 자리다)."""
        from pptx.dml.color import RGBColor
        from pptx.enum.shapes import MSO_SHAPE
        sh = self.s.shapes.add_shape(MSO_SHAPE.RECTANGLE, self.Cm(x), self.Cm(y), self.Cm(w), self.Cm(h))
        sh.fill.solid(); sh.fill.fore_color.rgb = RGBColor(0xee, 0xec, 0xe6)
        sh.line.color.rgb = RGBColor(0xb8, 0xb4, 0xab)
        self.text(x + 0.6, y + h / 2 - 1.2, w - 1.2, 2.4, label, size=15, color=INK2, align="c")
        self.img_area += w * h

    def ratio(self, x0=1.5, x1=57.9) -> float:
        y0, y1 = self.body
        return self.img_area / ((x1 - x0) * (y1 - y0))


def _set(shape, paras, size, bold=False, color=INK):
    """양식에 있던 글상자의 글을 바꾼다 (위치·크기는 양식 그대로)."""
    from pptx.dml.color import RGBColor
    from pptx.util import Pt
    tf = shape.text_frame
    for p in list(tf.paragraphs)[1:]:
        p._p.getparent().remove(p._p)
    tf.paragraphs[0].text = ""
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        runs = para if isinstance(para, list) else [(para, {})]
        for txt, st in runs:
            r = p.add_run()
            r.text = txt
            r.font.name, r.font.size = FONT, Pt(st.get("size", size))
            r.font.bold = st.get("bold", bold)
            r.font.italic = False
            r.font.color.rgb = RGBColor(*st.get("color", color))


# ------------------------------------------------------------------ 판넬
def build_panel(template: Path, n: dict, final: bool = False) -> Path:
    from pptx import Presentation
    prs = Presentation(str(template))
    slide = prs.slides[0]
    box = {sh.name: sh for sh in slide.shapes}
    H = lambda t: [(t, {"bold": True, "color": ACC})]  # noqa: E731

    _set(box["TextBox 16"], [[("심전도 잡음 제거 기법의 정량적 비교 연구", {"bold": True, "size": 50})]], 50)
    _set(box["TextBox 17"], [[("조원 이름 : (입력)", {"bold": True})]], 24)
    _set(box["TextBox 18"], [[("담당교수 : (입력)", {"bold": True})]], 24)
    _set(box["TextBox 19"], [
        [("작품 개요", {"bold": True, "size": 24})],
        [("자체 제작한 ECG 취득 장치(AD8232, Arduino Uno)의 잔류 잡음을 줄이는 후처리 기법을 정량적으로 "
          "비교하였다. 고전 필터 · 웨이블릿 임계값 처리 · 칼만 평활기 · 딥러닝을 동일한 입력 · 공통 전처리 · "
          "평가 지표로, 합성 ECG와 MIT-BIH 실측 기록에서 평가하였다. 본 시스템은 연구용 신호처리 시스템이며 "
          "임상 진단 용도가 아니다.", {"size": OVERVIEW_PT})]], OVERVIEW_PT)
    _set(box["TextBox 20"], [H("작품 내용")], 26)
    _set(box["TextBox 22"], [H("이론적 배경")], 26)
    _set(box["TextBox 21"], [H("결과 및 고찰")], 26)

    sh = Sheet(prs, slide, body=(17.5, 77.8))
    # 글자 크기(A1, 1~2 m 거리): 본문 BODY_PT, 캡션 CAP_PT, 그림 속 글자 14 pt 이상(make_panel_figs.py).
    # ---- 작품 내용 (좌) x 1.9 ~ 27.6
    intro = "잡음 종류와 입력 SNR에 따라 적합한 기법이 달라지는지를 분석하였다. 평가 체계는 기법 구현보다 먼저 확정하였다."
    hi = _text_h([intro], 25.7, BODY_PT)
    sh.text(1.9, 20.4, 25.7, hi, intro, size=BODY_PT)
    y = 20.4 + hi + 0.2
    # 블록선도는 mermaid 판(scripts/make_diagrams.py). 상자에는 핵심어만 두고 세부는 캡션으로 옮겼다.
    _, _, _, ih = sh.image(DG / "D1_pipeline.png", 1.9, y, 25.7, 13.4, valign="top")
    y += ih + 0.15
    c1 = "그림 1. 비교 실험 구성. 지표 4범주: 재구성 오차 · 파형 보존 · R파 검출 · 주파수 특성."
    hc = _text_h([c1], 25.7, CAP_PT)
    sh.text(1.9, y, 25.7, hc, c1, size=CAP_PT, color=INK2)
    y += hc + 0.3
    ph = 50.0 - CAP_PT * 0.0353 * 1.3 - y                # 사진 칸 높이: 캡션 한 줄을 남기고 구분선(약 51 cm)까지
    sh.frame(1.9, y, 12.5, ph, "사진: 신호 취득 장치\n(전극, AD8232, Arduino Uno)\n[사진 삽입 예정]")
    sh.image(OUT / "demo_mode_b.png", 15.1, y, 12.5, ph)
    sh.text(1.9, y + ph + 0.15, 12.5, 0.8, "그림 2. 신호 취득 장치.", size=CAP_PT, color=INK2)
    sh.text(15.1, y + ph + 0.15, 12.5, 0.8, "그림 3. 시연 화면.", size=CAP_PT, color=INK2)

    # ---- 이론적 배경 (우) x 29.5 ~ 57.5
    B = {"bold": True}
    cards = [
        ("T1_swt.png", [
            H("① 정상 웨이블릿 변환(SWT) 적응 임계값 처리"),
            [("가정 ", B), ("잡음은 크기가 작은 웨이블릿 계수에 분포한다.", {})],
            EQ("eq_swt", r"\lambda_j = k_j\,\sigma\,\sqrt{2\ln N}\;\left(1-(1-\rho)\,g(t)\right)"),
            "임계값 이하 계수를 제거한다. g(t): QRS 구간 완화.",
            [("장점 ", B), (f"무잡음 입력의 왜곡이 가장 작다(출력 SNR {n['clean_swt']:.1f} dB).", {})],
            [("한계 ", B), ("기저선 변동 · 임펄스 잡음은 제거하지 못한다.", {})],
        ]),
        ("T2_kalman.png", [
            H("② 확장 칼만 평활기(Sameni 모델)"),
            [("가정 ", B), ("ECG 한 주기는 위상 θ 위의 가우시안 합이다.", {})],
            EQ("eq_eks", r"\dot{\theta} = \omega,\quad \dot{z} = -\sum_i \frac{\alpha_i\,\omega}{b_i^2}\,"
                         r"\Delta\theta_i \exp\!\left(-\frac{\Delta\theta_i^2}{2\,b_i^2}\right)"),
            [("장점 ", B), ("모델과 맞는 합성 ECG에서 개선량이 크다(+16.2 dB).", {})],
            [("한계 ", B), ("R파 오검출 위치에 가성 QRS군을 만든다.", {})],
        ]),
        ("D2_resunet.png", [
            H("③ 잔차 학습 1D U-Net"),
            [("가정 ", B), ("신경망이 잡음을 추정하여 입력에서 뺀다.", {})],
            EQ("eq_res", r"\hat{x} = y - f_\theta(y)"),
            EQ("eq_loss", r"\mathcal{L} = \mathcal{L}_{\mathrm{MSE}} + 0.5\,\mathcal{L}_{\mathrm{MAE}}"
                          r" + 0.3\,\mathcal{L}_{\mathrm{diff}} + \lambda\,\mathcal{L}_{\mathrm{clean}}"),
            [("장점 ", B), (f"{n['cross_db']:.0f} dB 미만 · 임펄스 잡음에서 우수하고, 가성 파형이 가장 적다.", {})],
            [("한계 ", B), (f"무잡음 입력도 변형한다(출력 SNR {n['clean_m06']:.1f} dB, 보존항 적용 시 "
                          f"{n['clean_l6']:.1f} dB).", {})],
        ]),
    ]
    y = 20.6
    for fig, paras in cards:
        below = fig.startswith("D2")                 # 가로로 긴 그림(D2)은 글 아래에 칸 전체 폭으로 둔다
        fw, fh = (0.0, 0.0) if below else PF.SIZES[fig.split("_")[0]]
        tw = 28.0 if below else 28.0 - fw - 0.4
        # 수식은 글상자 안에 두지 않고 한 줄을 통째로 쓰는 그림으로 놓는다(본문의 1.5 배).
        yy, block = y, []
        for para in paras + [None]:
            if para is None or isinstance(para, EQ):
                if block:
                    h = _text_h(block, tw, BODY_PT)
                    sh.text(29.5, yy, tw, h, block, size=BODY_PT)
                    yy += h
                    block = []
                if para is not None:
                    yy += sh.equation(render_eq(para, EQ_PT), 29.5, yy + 0.1, tw) + 0.25
            else:
                block.append(para)
        if below:
            _, _, _, ih = sh.image(DG / fig, 29.5, yy + 0.1, 28.0, 6.0, valign="top")
            y = yy + 0.1 + ih + 0.4
        else:
            sh.image(PFIG / fig, 29.5 + tw + 0.4, y, fw, fh, valign="top")
            print(f"  이론 카드 {fig}: 글 {yy - y:.1f} cm, 그림 {fh:.1f} cm")
            y += max(yy - y, fh) + 0.4      # 다음 카드는 글과 그림 중 긴 쪽 아래에서 시작한다
    print(f"  이론적 배경 구역의 끝: {y:.1f} cm (구분선 51.0 cm)")
    if y > 51.0:
        print(f"  경고: 이론적 배경 구역의 글이 구분선(51.0 cm)을 넘을 수 있다 (끝 {y:.1f} cm)")

    # ---- 결과 및 고찰 (하단 전폭) x 1.8 ~ 57.6, y 53.9 ~ 77.6
    # 판넬 그림은 make_panel_figs.py 가 칸 크기 그대로 그린 것이다(축소 없음 — 그림 속 글자 14 pt 이상).
    r1 = 53.9
    row1 = [("F4_crossover.png", 1.8, f"그림 4. 공통 전처리 대비 딥러닝의 SNR 개선량. MIT-BIH는 약 {n['cross_db']:.0f} dB 이상에서 0 이하."),
            ("F5_phantom.png", 20.4, f"그림 5. 칼만 평활기의 가성 QRS군(MIT-BIH). {n['ph_M05']}건, SWT의 약 {n['ph_x_swt']:.0f}배."),
            ("F6_noise.png", 39.0, "그림 6. 잡음 종류별 출력(MIT-BIH, 입력 SNR −5 dB). 임펄스 잡음은 딥러닝만 제거.")]
    yb = r1
    for fig, x, capt in row1:
        w, h = PF.SIZES[fig.split("_")[0]]
        sh.image(PFIG / fig, x, r1, w, h)
        hc = _text_h([capt], w, CAP_PT)
        sh.text(x, r1 + h + 0.1, w, hc, capt, size=CAP_PT, color=INK2)
        yb = max(yb, r1 + h + 0.1 + hc)
    r2 = yb + 0.35
    w7, h7 = PF.SIZES["F7"]
    sh.image(PFIG / "F7_journey.png", 1.8, r2, w7, h7)
    c7 = "그림 7. 잔차 U-Net 변경 16종의 효과(채운 점: p < 0.05)."
    sh.text(1.8, r2 + h7 + 0.1, w7, _text_h([c7], w7, CAP_PT), c7, size=CAP_PT, color=INK2)
    xg = 1.8 + w7 + 0.8
    disc = [
        H("고찰"),
        "• 모든 조건에서 최적인 단일 기법은 없었다. 잡음 종류와 입력 SNR에 따라 기법을 선택해야 한다.",
        "• 칼만 평활기는 모델 가정이 맞을 때 우수하나, 실측 기록에서는 실제 박동과 비슷한 가성 QRS군을 만들었다.",
        f"• 잔차 학습 딥러닝은 가성 파형 생성이 가장 적었고, 무잡음 입력의 변형은 보존항으로 줄었다"
        f"({n['clean_m06']:.1f} → {n['clean_l6']:.1f} dB).",
        "• 구조 · 규모의 변경은 유의한 개선이 없었고, 유의한 개선은 손실 함수와 SNR 조건화에서만 관찰되었다(그림 7).",
        [("한계 ", {"bold": True}), ("공개 데이터 두 종으로 평가하였고, 자체 장치 데이터 평가는 진행 중이다.", {})],
    ]
    hd = _text_h(disc, 57.6 - xg, BODY_PT)
    sh.text(xg, r2, 57.6 - xg, hd, disc, size=BODY_PT)
    print(f"  결과 구역의 끝: 그림 7 {r2 + h7 + 0.8:.1f} cm, 고찰 {r2 + hd:.1f} cm (하단 77.6 cm)")
    if not final:
        sh.text(40.0, 0.4, 18.0, 0.6, "시안: build_panel.py 자동 생성, 수치는 실험 산출물에서 읽음",
                size=9, color=INK2)
    print(f"  panel 그림 면적 비율 (본문 구역 대비): {sh.ratio():.1%}")
    p = OUT / ("panel.pptx" if final else "panel_draft.pptx")
    prs.save(str(p))
    return p


# ------------------------------------------------------------------ 브로셔
def build_brochure(n: dict, final: bool = False) -> Path:
    from pptx import Presentation
    from pptx.util import Cm
    prs = Presentation()
    prs.slide_width, prs.slide_height = Cm(21.0), Cm(29.7)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    sh = Sheet(prs, slide, body=(4.5, 28.5))
    H = lambda t: [(t, {"bold": True, "size": 12, "color": ACC})]  # noqa: E731
    sh.text(1.6, 1.2, 17.8, 1.0, [[("작품명 | ", {"bold": True}),
                                   ("심전도 잡음 제거 기법의 정량적 비교 연구", {"bold": True})]], size=15)
    sh.text(1.6, 2.4, 17.8, 0.8, [[("학 과 | ", {"bold": True}), ("전기정보공학과   ", {}),
                                   ("조 원 | ", {"bold": True}), ("(입력)   ", {}),
                                   ("지도교수 | ", {"bold": True}), ("(입력)", {})]], size=10.5)
    sh.text(1.6, 4.0, 17.8, 0.7, [H("1. 작품 목표")])
    sh.text(1.6, 4.8, 17.8, 2.0, "자체 제작한 심전도(ECG) 취득 장치의 잔류 잡음을 파형을 보존하면서 줄이는 디지털 "
            "후처리 기법을 규명하는 것을 목표로 한다. 고전 필터, 웨이블릿 임계값 처리, 칼만 필터, 딥러닝의 네 계열을 "
            "동일한 입력 · 전처리 · 평가 지표로 비교하여, 잡음 종류와 입력 SNR에 따른 적합 기법을 제시한다.", size=10)
    sh.text(1.6, 7.2, 17.8, 0.7, [H("2. 작품 개요")])
    sh.image(DG / "D1_pipeline.png", 1.6, 8.0, 17.8, 8.0)
    sh.text(1.6, 16.2, 17.8, 3.8, [
        "- 공통 전처리(0.5~100 Hz bandpass, 전원선 잡음 검출 시 notch) 후 네 계열의 기법과 이상적 상한(기준 "
        "신호를 아는 최적 필터)을 잡음 7종, 입력 SNR 7단계에서 비교",
        "- 4개 범주의 평가 지표(재구성 오차, 파형 보존, R파 검출, 주파수 특성)와 가성 파형 생성 검증(박동을 지운 "
        "구간에 출력이 만드는 파형의 측정)",
        "- 합성 ECG와 MIT-BIH 실측 기록 두 데이터에서 결과의 재현성 확인",
        "- 녹음 신호 비교 화면과 실시간 처리 화면으로 구성된 시연 시스템",
        [("- 응용 분야: ", {"bold": True}), ("저가형 · 착용형 심전도 취득 장치의 신호 후처리 연구, 생체신호 "
                                             "잡음 제거 기법의 평가 방법론(임상 진단 용도 아님)", {})],
    ], size=9.5)
    sh.text(1.6, 20.3, 17.8, 0.7, [H("3. 작품 내용 및 이미지")])
    w = 5.75
    for i, (img, capt) in enumerate([
            (OUT / "demo_mode_b.png", "[그림 1] 시연 화면(녹음 신호 비교 모드)"),
            (SL / "S5_crossover.png", f"[그림 2] 딥러닝이 공통 전처리보다 높은 SNR 개선량을 보인 입력 SNR 범위(MIT-BIH, 약 {n['cross_db']:.0f} dB 이하)"),
            (SL / "S15_kalman_phantom.png", f"[그림 3] 칼만 평활기의 가성 QRS군 생성({n['ph_M05']}건)")]):
        x = 1.6 + i * (w + 0.275)
        sh.image(img, x, 21.1, w, 4.6)
        sh.text(x, 25.8, w, 0.9, capt, size=8.5, align="c")
    sh.text(1.6, 27.0, 17.8, 1.6, f"칼만 평활기는 위양성 검출 위치(실제 R파가 없는 위치)에서 박동 형태의 가성 QRS군을 생성하였으며(SWT 대비 약 "
            f"{n['ph_x_swt']:.0f}배), 잔차 학습 구조의 딥러닝 기법들의 가성 파형 생성 에너지가 비교 기법 중 가장 낮았다. 딥러닝에서 "
            "유의한 개선은 손실 함수 변경과 SNR 조건화(입력 SNR을 신경망에 함께 입력)에서만 관찰되었고, 구조 · "
            "규모의 변경에서는 관찰되지 않았다.", size=9.5)
    print(f"  brochure 그림 면적 비율: {sh.ratio(1.6, 19.4):.0%}")
    p = OUT / ("brochure.pptx" if final else "brochure_draft.pptx")
    prs.save(str(p))
    return p


def render(pptx: Path, dpi: int = 60) -> None:
    """LibreOffice 로 PDF → PNG. 글꼴은 이 환경의 대체 글꼴로 그려진다(맑은 고딕 없음)."""
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(pptx.parent),
                    str(pptx)], check=True, capture_output=True, timeout=300)
    pdf = pptx.with_suffix(".pdf")
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(pdf), str(pptx.with_suffix(""))],
                   check=True, timeout=300)
    if dpi <= 60:
        pdf.unlink()                                  # 시안은 미리보기만 남긴다. 최종판은 인쇄용 PDF 를 남긴다
    print(f"  {pptx.with_suffix('.png')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--template", required=True,
                    help="학과 판넬 양식 pptx (A00_panel). 저장소에 넣지 않는다 — 받은 파일 경로를 준다")
    ap.add_argument("--render", action="store_true",
                    help="LibreOffice 로 PNG 미리보기도 만든다 (soffice · pdftoppm 필요)")
    ap.add_argument("--final", action="store_true",
                    help="최종판: 시안 표지를 빼고 panel.pptx · brochure.pptx 로 저장, --render 시 PDF 와 100 dpi PNG 를 남긴다")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    n = numbers()
    print("[numbers]", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in n.items()})
    p1, p2 = build_panel(Path(a.template), n, a.final), build_brochure(n, a.final)
    print(f"  {p1}\n  {p2}")
    if a.render:
        dpi = 100 if a.final else 60
        render(p1, dpi)
        render(p2, 200 if a.final else 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
