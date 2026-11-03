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

from panel_eq import render_eq

OUT = Path("results/panel")
SL = Path("results/slides")
DG = SL / "diagrams"     # mermaid 블록선도 (scripts/make_diagrams.py)
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
BODY_PT = 14            # 이론 카드 본문 글자 크기
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
        [("자체 제작한 심전도(ECG) 취득 장치(AD8232, Arduino Uno)의 잔류 잡음을 줄이기 위한 디지털 후처리 "
          "기법을 정량적으로 비교하였다. 고전 필터, 웨이블릿 임계값 처리, 칼만 필터, 딥러닝의 네 계열을 동일한 "
          "입력과 공통 전처리(0.5~100 Hz bandpass, 전원선 잡음 검출 시 notch), 동일한 평가 지표로 평가하였으며, 합성 ECG와 MIT-BIH 실측 기록에서 잡음 7종과 "
          "입력 SNR 7단계(−5~25 dB)를 대상으로 하였다. 본 시스템은 연구용 신호처리 시스템이며 임상 진단 "
          "용도가 아니다.", {"size": 18})]], 18)
    _set(box["TextBox 20"], [H("작품 내용")], 24)
    _set(box["TextBox 22"], [H("이론적 배경")], 24)
    _set(box["TextBox 21"], [H("결과 및 고찰")], 24)

    sh = Sheet(prs, slide, body=(17.5, 77.8))
    # ---- 작품 내용 (좌) x 1.9 ~ 27.6
    sh.text(1.9, 20.4, 25.7, 2.2, "단일 조건에서 기법의 우열을 판정하는 대신, 잡음 종류와 입력 SNR에 따라 적합한 "
            "기법이 달라지는지를 분석하였다. 비교의 공정성을 위해 평가 체계를 기법 구현보다 먼저 확정하였다.",
            size=16)
    # 블록선도는 mermaid 판(scripts/make_diagrams.py). 상자에는 핵심어만 두고 세부는 캡션으로 옮겼다.
    sh.image(DG / "D1_pipeline.png", 1.9, 22.9, 25.7, 13.4)
    sh.text(1.9, 36.4, 25.7, 1.4, "그림 1. 비교 실험 구성. 지표 4범주: 재구성 오차 · 파형 보존 · R파 검출 · 주파수 특성. "
            "가성 파형 생성 검증: 박동 삭제 · 무수축 · 심실조기수축. 이상적 상한: 기준 신호를 아는 최적 필터.",
            size=13, color=INK2)
    sh.frame(1.9, 38.1, 12.5, 11.6, "사진: 신호 취득 장치\n(전극, AD8232, Arduino Uno)\n[사진 삽입 예정]")
    sh.image(OUT / "demo_mode_b.png", 15.1, 38.1, 12.5, 11.6)
    sh.text(1.9, 49.9, 12.5, 1.0, "그림 2. 신호 취득 장치.", size=13, color=INK2)
    sh.text(15.1, 49.9, 12.5, 1.6, "그림 3. 시연 화면(녹음 신호 비교 모드, 임펄스 잡음, 입력 SNR 0 dB).",
            size=13, color=INK2)

    # ---- 이론적 배경 (우) x 29.5 ~ 57.5
    sh.text(29.5, 20.4, 28.0, 1.4, "대표 기법 세 가지의 기본 가정과 그에 따른 장점 및 한계", size=16)
    B = {"bold": True}
    cards = [
        ("S19_swt.png", (9.4, 7.8), [
            H("① 정상 웨이블릿 변환(SWT) 적응 임계값 처리"),
            [("가정 ", B), ("잡음은 크기가 작은 웨이블릿 계수에 분포한다.", {})],
            EQ("eq_swt", r"\lambda_j = k_j\,\sigma\,\sqrt{2\ln N}\;\left(1-(1-\rho)\,g(t)\right)"),
            "임계값 이하 계수를 제거하고 초과 계수는 축소한다(garrote thresholding). g(t)는 QRS 구간의 임계값 완화 함수이다.",
            [("장점 ", B), (f"실제 적용 가능한 기법 중 무잡음 입력의 왜곡이 가장 작다(MIT-BIH, 출력 SNR "
                          f"{n['clean_swt']:.1f} dB).", {})],
            [("한계 ", B), ("기저선 변동과 임펄스 잡음은 계수 크기로 구분되지 않아 제거되지 않는다.", {})],
            [("비고 ", B), ("MIT-BIH 실측 기록에서는 hard thresholding(초과 계수 유지)이 최적이었다.", {})],
        ]),
        ("S20_kalman_model.png", (13.5, 7.8), [
            H("② 확장 칼만 평활기(EKS, Sameni 모델)"),
            [("가정 ", B), ("ECG 한 주기는 심장 위상 θ 위의 가우시안 함수 합이다.", {})],
            EQ("eq_eks", r"\dot{\theta} = \omega,\qquad \dot{z} = -\sum_i \frac{\alpha_i\,\omega}{b_i^2}\,"
                         r"\Delta\theta_i \exp\!\left(-\frac{\Delta\theta_i^2}{2\,b_i^2}\right)"),
            "상태 [θ, z]를 확장 칼만 필터로 추정한 뒤 후방 평활(RTS)한다.",
            [("장점 ", B), ("박동 파형이 모델 가정과 일치하는 합성 ECG에서 SNR 개선량이 크다(입력 5 dB, +16.2 dB).", {})],
            [("한계 ", B), ("위상 기준을 입력의 R파 검출로 정하므로, 위양성 검출(R파가 없는 위치를 R파로 판정) "
                          "위치에 QRS 파형을 생성한다.", {})],
        ]),
        ("D2_resunet.png", "below", [
            H("③ 잔차 학습 1D U-Net(Residual U-Net)"),
            [("가정 ", B), ("신경망이 잡음을 추정하여 입력에서 뺀다(잔차 학습).", {})],
            EQ("eq_res", r"\hat{x} = y - f_\theta(y)"),
            EQ("eq_loss", r"\mathcal{L} = \mathcal{L}_{\mathrm{MSE}} + 0.5\,\mathcal{L}_{\mathrm{MAE}}"
                          r" + 0.3\,\mathcal{L}_{\mathrm{diff}} + \lambda\,\mathcal{L}_{\mathrm{clean}}"),
            "각 항: 평균제곱오차, 평균절대오차, 미분 오차, 무잡음 입력 보존항.",
            [("장점 ", B), (f"입력 SNR {n['cross_db']:.0f} dB 미만 구간과 임펄스 잡음에서 SNR 개선량이 공통 전처리보다 높고, "
                          f"가성 파형 생성 에너지가 대역통과 필터의 약 1/{n['he_x_bp']:.1f}이다(MIT-BIH).", {})],
            [("한계 ", B), (f"무잡음 입력도 변형하였으나(출력 SNR {n['clean_m06']:.1f} dB), 무잡음 입력 보존항을 "
                          f"추가하여 {n['clean_l6']:.1f} dB로 높였다(MIT-BIH).", {})],
        ]),
    ]
    y = 22.2
    for fig, box, paras in cards:
        below = box == "below"              # 가로로 긴 그림(D2)은 글 아래에 칸 전체 폭으로 둔다
        fw, fh = (0.0, 0.0) if below else box
        tw = 28.0 if below else 28.0 - fw - 0.4
        # 수식은 글상자 안에 두지 않고 한 줄을 통째로 쓰는 그림으로 놓는다(본문 14 pt 의 1.5 배).
        # 글상자를 수식 앞 · 뒤로 나누고, 앞 글상자의 높이는 글자 폭으로 추정한다.
        yy, block = y, []
        for para in paras + [None]:
            if para is None or isinstance(para, EQ):
                if block:
                    h = _text_h(block, tw, 14)
                    sh.text(29.5, yy, tw, h, block, size=14)
                    yy += h
                    block = []
                if para is not None:
                    yy += sh.equation(render_eq(para, EQ_PT), 29.5, yy + 0.12, tw) + 0.3
            else:
                block.append(para)
        if below:
            _, _, _, ih = sh.image(DG / fig, 29.5, yy + 0.1, 28.0, 6.0, valign="top")
            y = yy + 0.1 + ih + 0.45
        else:
            _, _, _, ih = sh.image(SL / fig, 29.5 + tw + 0.4, y, fw, fh, valign="top")
            print(f"  이론 카드 {fig}: 글 {yy - y:.1f} cm, 그림 {ih:.1f} cm")
            y += max(yy - y, ih) + 0.45    # 다음 카드는 글과 그림 중 긴 쪽 아래에서 시작한다
    print(f"  이론적 배경 구역의 끝: {y:.1f} cm (구분선 51.0 cm)")
    if y > 51.0:
        print(f"  경고: 이론적 배경 구역의 글이 구분선(51.0 cm)을 넘을 수 있다 (끝 {y:.1f} cm)")

    # ---- 결과 및 고찰 (하단 전폭) x 1.8 ~ 57.6, y 53.6 ~ 77.6
    r1, r2 = 54.0, 66.0
    cap = 13
    sh.image(SL / "S5_crossover.png", 1.8, r1, 16.9, 9.6)
    sh.text(1.8, r1 + 9.8, 16.9, 1.8,
            f"그림 4. 공통 전처리 대비 딥러닝(웨이블릿 부대역 U-Net)의 SNR 개선량. MIT-BIH 실측 기록에서 입력 SNR −5 dB일 때 "
            f"+{n['dl_gain_lo']:.1f} dB이며, 약 {n['cross_db']:.0f} dB 이상에서는 0 이하로 감소한다.", size=cap)
    sh.image(SL / "S15_kalman_phantom.png", 19.1, r1, 17.6, 9.6)
    sh.text(19.1, r1 + 9.8, 17.6, 1.8,
            f"그림 5. 칼만 평활기의 가성 QRS군 생성 사례와 기법별 발생 수(MIT-BIH, 입력 SNR −5·0 dB). "
            f"{n['ph_M05']}건으로 SWT 대비 약 {n['ph_x_swt']:.0f}배, 잔차 U-Net 대비 약 {n['ph_x_dl']:.0f}배이다.",
            size=cap)
    sh.image(SL / "S4_noise_d1.png", 37.4, r1, 20.2, 9.6)
    sh.text(37.4, r1 + 9.8, 20.2, 1.8, "그림 6. 잡음 종류별 기법 출력(MIT-BIH, 입력 SNR −5 dB). "
            "기저선 변동은 공통 전처리만으로 제거되었고, 임펄스 잡음은 실제 적용 가능한 기법 중 딥러닝에서만 제거되었다.", size=cap)
    # 판넬판 그림(시각화 제안서 6 절): S17 의 16 행 이름표 대신 범주 4 개(S17p), 판정 해상도(S22)
    sh.image(SL / "S17p_dl_journey_panel.png", 1.8, r2, 17.2, 9.9)
    sh.text(1.8, r2 + 9.0, 17.2, 1.7, f"그림 7. 잔차 U-Net의 구조 · 학습 조건 변경 16종에 따른 SNR 개선량 변화. 유의한 증가는 손실 함수 "
            f"변경과 SNR 조건화(입력 SNR을 신경망에 함께 입력)에서만 관찰되었다(무잡음 입력 보존항, 15~20 dB 구간 "
            f"{n['l6_hi']:+.1f} dB).", size=cap)
    sh.image(OUT.parent / "slides" / "research" / "S22_resolution.png", 19.4, r2, 17.4, 9.9)
    sh.text(19.4, r2 + 10.0, 17.4, 1.7, f"그림 8. 판정 전에 측정한 학습 반복 간 변동. 평가 잡음을 고정하자 표준편차가 "
            f"{n['sd_seed']:.2f} dB에서 {n['sd_fixed']:.2f} dB로 줄었고, 이 변동을 넘은 개선은 손실 함수 변경뿐이었다"
            f"(무잡음 입력 보존항 {n['l6_all']:+.2f} dB).", size=cap)
    sh.text(37.4, r2, 20.2, 11.6, [
        H("고찰"),
        "• 모든 조건에서 최적인 단일 기법은 관찰되지 않았다. 잡음 종류와 입력 SNR에 따라 기법을 "
        "선택하는 것이 타당하다.",
        "• 모델 기반 필터는 박동 파형이 모델의 가정과 일치할 때(합성 ECG) SNR 개선량이 크나, 불일치 시 실제 "
        "박동과 구분하기 어려운 가성 파형을 생성하였다. 비교한 기법 중 잔차 학습 구조의 딥러닝 기법들의 가성 파형 생성 에너지가 가장 "
        "낮았다(MIT-BIH, 대역통과 필터 대비 2.8~3.9배 낮음).",
        f"• 딥러닝에서 구조 · 규모를 바꾼 시도는 SNR 개선량을 유의하게 높이지 못하였고, 유의한 증가는 손실 함수 "
        f"변경과 SNR 조건화에서만 관찰되었다. "
        f"보존항을 추가한 결과 무잡음 입력의 출력 SNR이 {n['clean_m06']:.1f} dB에서 "
        f"{n['clean_l6']:.1f} dB로 증가하였다(SWT {n['clean_swt']:.1f} dB).",
        f"• 개선 시도 16종의 유의성 판정은, 같은 설정을 난수 시드만 바꾸어 다시 학습했을 때의 SNR 개선량 "
        f"변동(범위 {n['rng_seed']:.2f} dB)을 먼저 측정하고, 평가 잡음을 고정한 뒤 수행하였다(그림 8).",
        [("한계 ", {"bold": True}), ("두 데이터 모두 공개 데이터이며, 자체 장치의 실측 데이터 평가는 진행 "
                                      "중이다.", {})],
    ], size=13)
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
