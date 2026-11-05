#!/usr/bin/env python3
"""전시 판넬(A1) · 브로셔(A4) **시안**을 학과 양식 위에 만든다 (docs/94_panel_plan.md).

    python scripts/build_panel.py --template <A00_panel 양식.pptx>            # pptx 만
    python scripts/build_panel.py --template <양식.pptx> --render             # + PDF · PNG 미리보기

산출: `results/panel/`
    panel_draft.pptx / .png      A1 세로 59.4 x 84.1 cm, 양식의 네 구역에 그림과 문안
    brochure_draft.docx / .png   학과 브로셔 양식(docx, 19.0 x 26.8 cm 한 쪽)의 표에 ①목표 ②개요 ③내용 및 이미지

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
PFIG = OUT / "figs"         # 판넬 전용 그림 (scripts/make_panel_figs.py, 칸 크기 그대로)
CAP = OUT / "captures"      # 시연 영상 스틸 (릴리스 video, docs/38 §9)
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
    # 고찰 — 딥러닝 개선 과정(그림 7)의 원인 진단과 한계 (F-43 · F-46 · F-47)
    n["band_hi"] = _paired("band", "m06_l1_band_hi", "m06_l1", "band_15_20")[0]
    n["l6cond_add"], _, n["l6cond_add_p"] = _paired("l6cond", "m06_cond_l6", "m06_l6", "band_15_20")
    from make_slides import _sweep
    pert = [(j["pert_m8"] - j["fixed_snr_imp_scaled"], j["pert_p8"] - j["fixed_snr_imp_scaled"])
            for j in _sweep("condpert8").values()]
    n["pert_m8"], n["pert_p8"] = (float(np.mean(v)) for v in zip(*pert))
    # 브로셔 그림 3 · 4 — 잡음 종류별 실험(EXP-B), MIT-BIH, 입력 SNR +10 dB 의 구간 평균 (make_brochure_figs._expb_means)
    eb = pd.read_parquet("results/d1/exp_b/metrics.parquet")
    eb = eb[(eb.metric == "snr_imp_scaled") & (eb.snr_in_target == 10.0)]
    em = eb.groupby(["cond", "method"]).value.mean().unstack("method")
    n["noise_means"] = {c: {k: float(em.loc[c, k]) for k in ("M_FE", "M04", "M08")} for c in em.index}
    n["noise_n"] = int(eb[eb.method == "M_FE"].groupby("cond").size().min())
    # 그림 7 과 같은 Holm 보정(비교 전체 한 계열, D-44). 판넬 · 브로셔 문장이 이 판정에 기대므로 판정이 바뀌면 멈춘다.
    st = PF.flow_stats()
    n["n_tests"] = len(st)
    ns = [len(v[1]) for v in st.values()]
    n["seed_lo"], n["seed_hi"] = min(ns), max(ns)
    n["band_hi_ph"] = st[("m06_l1_band_hi", "m06_l1", "hi")][3]
    up = sorted({a for (a, _, _), v in st.items() if v[4] and v[0] > 0})
    down = sorted({a for (a, _, _), v in st.items() if v[4] and v[0] < 0})
    assert up == ["m06_cond", "m06_l6"] and down == ["m07_l1"], (
        f"그림 7 의 Holm 판정이 바뀌었다(향상 {up}, 저하 {down}) — 판넬 고찰 · 브로셔 문장과 그림 7 판정 문구를 고칠 것")
    return n


# ------------------------------------------------------------------ 편집 가능한 그림(SVG)
SVG_EXT_URI = "{96DAC541-7B7A-43D3-8B79-37D633B846F1}"


def add_picture(slide, png: Path, x, y, w, h):
    """그림을 넣는다. 같은 이름의 SVG 가 있으면 SVG 를 본 그림으로, PNG 를 대체 그림으로 함께 넣는다.

    PowerPoint(2016 이후)는 SVG 를 표시하고 「도형으로 변환」으로 편집할 수 있다. SVG 를 지원하지 않는
    뷰어(LibreOffice 미리보기 등)는 PNG 를 표시한다. 저장 방식은 PowerPoint 가 SVG 를 넣을 때와 같다
    (a:blip 의 확장 asvg:svgBlip).
    """
    from lxml import etree
    from pptx.opc.constants import RELATIONSHIP_TYPE as RT
    from pptx.opc.package import Part
    from pptx.opc.packuri import PackURI
    from pptx.util import Cm
    pic = slide.shapes.add_picture(str(png), Cm(x), Cm(y), Cm(w), Cm(h))
    svg = Path(png).with_suffix(".svg")
    if not svg.exists():
        return pic
    pkg = slide.part.package
    idx = 1 + sum(1 for p in pkg.iter_parts() if str(p.partname).startswith("/ppt/media/svg"))
    part = Part(PackURI(f"/ppt/media/svg{idx}.svg"), "image/svg+xml", pkg, svg.read_bytes())
    rid = slide.part.relate_to(part, RT.IMAGE)
    blip = pic._element.blipFill.find("{http://schemas.openxmlformats.org/drawingml/2006/main}blip")
    A = "http://schemas.openxmlformats.org/drawingml/2006/main"
    ext_lst = etree.SubElement(blip, f"{{{A}}}extLst")
    ext = etree.SubElement(ext_lst, f"{{{A}}}ext", uri=SVG_EXT_URI)
    sb = etree.SubElement(ext, "{http://schemas.microsoft.com/office/drawing/2016/SVG/main}svgBlip",
                          nsmap={"asvg": "http://schemas.microsoft.com/office/drawing/2016/SVG/main"})
    sb.set("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed", rid)
    return pic


# ------------------------------------------------------------------ 실험 구성도(PowerPoint 도형)
# make_diagrams.py 의 D1 과 같은 구성을 PowerPoint 도형으로 그린다 — 상자 · 글 · 화살표를 판넬에서 바로 고칠 수 있다.
# 좌표는 25.7 × 13.2 cm 설계 공간 기준이고 s 배로 늘이거나 줄인다.
_D1_FILL = {"data": ("ece9e1", "8f8a80"), "noise": ("f3e6cf", "b39257"), "fe": ("f6d6e3", "b5718d"),
            "cls": ("d5e4f7", "5b86bd"), "tf": ("fbdccc", "c77c55"), "mb": ("fbecc2", "b8962e"),
            "dl": ("cdeee1", "3f9a77"), "ub": ("e4e2dc", "8f8a80"), "ev": ("e6e5f4", "6f6aa8"),
            "grp": ("faf9f6", "c9c5bb")}
D1_W, D1_H = 25.7, 13.2


def draw_pipeline(sheet, x0, y0, s=1.0, fpt=16):
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.oxml.ns import qn
    from pptx.util import Cm, Pt
    sh = sheet.s.shapes

    def box(x, y, w, h, text, kind, size=fpt, dash=False, top=False):
        b = sh.add_shape(MSO_SHAPE.RECTANGLE, Cm(x0 + x * s), Cm(y0 + y * s), Cm(w * s), Cm(h * s))
        fc, lc = _D1_FILL[kind]
        b.fill.solid(); b.fill.fore_color.rgb = RGBColor.from_string(fc)
        b.line.color.rgb = RGBColor.from_string(lc); b.line.width = Pt(1.0)
        if dash:
            b.line.dash_style = 4                    # 점선 — 이상적 상한(실제로 적용할 수 없는 기준)
        b.shadow.inherit = False
        st = b._element.find(qn("p:style"))          # 양식 테마의 그림자 · 효과를 끊는다
        if st is not None:
            b._element.remove(st)
        tf = b.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.TOP if top else MSO_ANCHOR.MIDDLE
        for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
            setattr(tf, m, Cm(0.08))
        for i, line in enumerate(text.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run(); r.text = line
            r.font.name, r.font.size, r.font.color.rgb = FONT, Pt(size), RGBColor(*INK)
        return b

    def arrow(xa, ya, xb, yb):
        c = sh.add_connector(MSO_CONNECTOR.STRAIGHT, Cm(x0 + xa * s), Cm(y0 + ya * s), Cm(x0 + xb * s),
                             Cm(y0 + yb * s))
        c.line.color.rgb = RGBColor(0x52, 0x51, 0x4e); c.line.width = Pt(1.4)
        st = c._element.find(qn("p:style"))
        if st is not None:
            c._element.remove(st)
        ln = c.line._get_or_add_ln()
        ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))

    g = fpt - 1
    # 1 줄: 입력 신호 → 잡음 → 공통 전처리
    box(0.0, 0.0, 12.0, 3.3, "입력 신호", "grp", size=g, top=True)
    for i, t in enumerate(["합성 ECG", "MIT-BIH\n실측 기록", "자체 장치\nAD8232"]):
        box(0.35 + i * 3.85, 1.05, 3.55, 1.8, t, "data")
    box(12.9, 0.3, 6.1, 2.7, "잡음 7종\n입력 SNR 7단계\n(-5~25 dB)", "noise")
    box(19.9, 0.3, 5.8, 2.7, "공통 전처리\n0.5~100 Hz 대역 제한\n전원선 노치 필터", "fe")
    arrow(12.0, 1.65, 12.9, 1.65); arrow(19.0, 1.65, 19.9, 1.65)
    # 2 줄: 잡음 제거 기법 다섯
    box(0.9, 4.3, 24.0, 4.0, "잡음 제거 기법(동일 입력)", "grp", size=g, top=True)
    for i, (t, k) in enumerate([("고전 필터", "cls"), ("웨이블릿\n임계값 처리", "tf"), ("칼만 필터", "mb"),
                                ("딥러닝", "dl"), ("이상적 상한", "ub")]):
        box(1.4 + i * 4.7, 5.35, 4.3, 2.4, t, k, dash=(k == "ub"))
    arrow(22.8, 3.0, 22.8, 4.3)
    # 3 줄: 평가
    box(5.6, 9.3, 14.6, 3.9, "평가(동일 지표)", "grp", size=g, top=True)
    box(6.2, 10.4, 6.3, 2.2, "지표 4범주", "ev")
    box(13.3, 10.4, 6.3, 2.2, "가성 파형\n생성 검증", "ev")
    arrow(12.9, 8.3, 12.9, 9.3)
    sheet.img_area += D1_W * s * D1_H * s
    return D1_H * s


# 연구 단계(작품 내용의 목차). 각 단계가 결과 및 고찰의 어느 그림에서 다뤄지는지를 적어 두 구역을 잇는다.
STEPS = [("① 비교 체계", "동일 입력 · 지표", "그림 1"),
         ("② 조건별 비교", "입력 SNR · 잡음", "그림 4 · 5"),
         ("③ 안전성 검증", "가성 QRS군 생성", "그림 6"),
         ("④ 딥러닝 개선", "학습 목표 재설정", "그림 7"),
         ("⑤ 실시간 시연", "장치 · 시연 화면", "그림 2 · 3")]


def draw_steps(sheet, x0, y0, w, fpt=15, h=2.5, gap=0.5):
    """다섯 단계를 PowerPoint 도형(상자 + 화살표)으로 한 줄에 놓는다. 높이를 돌려준다."""
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.oxml.ns import qn
    from pptx.util import Cm, Pt
    sh = sheet.s.shapes
    bw = (w - gap * (len(STEPS) - 1)) / len(STEPS)
    for i, (title, desc, ref) in enumerate(STEPS):
        x = x0 + i * (bw + gap)
        hl = title.startswith("④")                   # 이 과제의 중심 단계
        b = sh.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Cm(x), Cm(y0), Cm(bw), Cm(h))
        b.adjustments[0] = 0.12
        b.fill.solid(); b.fill.fore_color.rgb = RGBColor.from_string("e4ecf7" if hl else "f1f0ec")
        b.line.color.rgb = RGBColor.from_string("184f95" if hl else "c9c5bb"); b.line.width = Pt(1.0)
        st = b._element.find(qn("p:style"))
        if st is not None:
            b._element.remove(st)
        tf = b.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
            setattr(tf, m, Cm(0.1))
        for j, (txt, bold, col) in enumerate([(title, True, ACC), (desc, False, INK), (ref, False, INK2)]):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run(); r.text = txt
            r.font.name, r.font.size, r.font.bold = FONT, Pt(fpt), bold
            r.font.color.rgb = RGBColor(*col)
        if i:
            c = sh.add_connector(MSO_CONNECTOR.STRAIGHT, Cm(x - gap + 0.05), Cm(y0 + h / 2), Cm(x - 0.05),
                                 Cm(y0 + h / 2))
            c.line.color.rgb = RGBColor(0x52, 0x51, 0x4e); c.line.width = Pt(1.4)
            st = c._element.find(qn("p:style"))
            if st is not None:
                c._element.remove(st)
            ln = c.line._get_or_add_ln()
            ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    sheet.img_area += w * h
    return h


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
            for txt0, st in runs:
                # 「p_Holm」은 p 와 아래 첨자 Holm 으로 나누어 쓴다(그림의 표기와 같게)
                for k, txt in enumerate(txt0.split("p_Holm")):
                    if k:
                        for t2, sub in (("p", False), ("Holm", True)):
                            r = p.add_run(); r.text = t2
                            f = r.font
                            f.name, f.size, f.italic = FONT, self.Pt(st.get("size", size)), not sub
                            f.bold = st.get("bold", bold)
                            f.color.rgb = RGBColor(*st.get("color", color))
                            if sub:
                                r._r.get_or_add_rPr().set("baseline", "-25000")
                    if not txt:
                        continue
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
        add_picture(self.s, path, xx, yy, ww, hh)
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
        add_picture(self.s, path, x, y, w, h)
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
          "비교하였다. 고전 필터 · 웨이블릿 임계값 처리 · 칼만 필터 · 딥러닝을 동일한 입력 · 공통 전처리 · "
          "평가 지표로, 합성 ECG와 MIT-BIH 실측 기록에서 평가하였다. 본 시스템은 연구용 신호처리 시스템이며 "
          "임상 진단 용도가 아니다.", {"size": OVERVIEW_PT})]], OVERVIEW_PT)
    _set(box["TextBox 20"], [H("작품 내용")], 26)
    _set(box["TextBox 22"], [H("이론적 배경")], 26)
    _set(box["TextBox 21"], [H("결과 및 고찰")], 26)

    sh = Sheet(prs, slide, body=(17.5, 77.8))
    # 글자 크기(A1, 1~2 m 거리): 본문 BODY_PT, 캡션 CAP_PT, 그림 속 글자 14 pt 이상(make_panel_figs.py).
    # ---- 작품 내용 (좌) x 1.9 ~ 27.6
    intro = ("잡음 종류와 입력 SNR에 따라 적합한 기법이 달라지는지를 다섯 단계로 분석하였다. 평가 체계는 기법 "
             "구현보다 먼저 확정하였다.")
    hi = _text_h([intro], 25.7, BODY_PT)
    sh.text(1.9, 20.4, 25.7, hi, intro, size=BODY_PT)
    y = 20.4 + hi + 0.2
    y += draw_steps(sh, 1.9, y, 25.7, fpt=15) + 0.45     # 연구 단계(목차) — PowerPoint 도형
    # 블록선도는 mermaid 판(scripts/make_diagrams.py). 상자에는 핵심어만 두고 세부는 캡션으로 옮겼다.
    y += draw_pipeline(sh, 1.9, y, s=1.0, fpt=15) + 0.2       # PowerPoint 도형 — 판넬에서 바로 고칠 수 있다
    c1 = "그림 1. 비교 실험 구성. 지표 4범주: 재구성 오차 · 파형 보존 · R파 검출 · 주파수 특성."
    hc = _text_h([c1], 25.7, CAP_PT)
    sh.text(1.9, y, 25.7, hc, c1, size=CAP_PT, color=INK2)
    y += hc + 0.3
    ph = 50.0 - CAP_PT * 0.0353 * 1.3 - y                # 사진 칸 높이: 캡션 한 줄을 남기고 구분선(약 51 cm)까지
    sh.frame(1.9, y, 12.5, ph, "사진: 신호 취득 장치\n(전극, AD8232, Arduino Uno)\n[사진 삽입 예정]")
    # 시연 사진은 실시간 측정 화면을 우선한다(사용자 지시 2026-11-05). 본인 심전도라 git 제외 폴더에서 읽는다.
    sh.image(Path("data/arduino/captures/live_01_overlay_t16.png"), 15.1, y, 12.5, ph)
    sh.text(1.9, y + ph + 0.15, 12.5, 0.8, "그림 2. 신호 취득 장치.", size=CAP_PT, color=INK2)
    sh.text(15.1, y + ph + 0.15, 12.5, 0.8, "그림 3. 실시간 측정 화면(자체 장치 신호, SWT · 딥러닝 출력).", size=CAP_PT,
            color=INK2)

    # ---- 이론적 배경 (우) x 29.5 ~ 57.5
    B = {"bold": True}
    cards = [
        ("T1_swt.png", [
            H("① 정상 웨이블릿 변환(SWT) 적응 임계값 처리"),
            "입력을 다섯 개 주파수 대역의 세부 계수로 분해하고, 대역마다 다른 임계값을 적용한 뒤 다시 합성한다.",
            EQ("eq_swt", r"\lambda_j = k_j\,\sigma\,\sqrt{2\ln N}\;\left(1-(1-\rho)\,g(t)\right)"),
            "k: 대역별 임계값 배율, σ: 잡음 표준편차, g(t): QRS 구간의 임계값 완화 함수.",
            [("장점 ", B), (f"무잡음 입력의 왜곡이 가장 작다(출력 SNR {n['clean_swt']:.1f} dB).", {})],
            [("한계 ", B), ("기저선 변동 · 임펄스 잡음은 제거하지 못한다.", {})],
        ]),
        ("T2_kalman.png", [
            H("② 확장 칼만 필터(McSharry 모델)"),
            "ECG 한 주기를 심장 위상 θ 위 가우시안 함수(P · Q · R · S · T파)의 합으로 모델링한다.",
            EQ("eq_eks", r"\dot{\theta} = \omega,\quad \dot{z} = -\sum_i \frac{\alpha_i\,\omega}{b_i^2}\,"
                         r"\Delta\theta_i \exp\!\left(-\frac{\Delta\theta_i^2}{2\,b_i^2}\right)"),
            "위상은 검출된 R파마다 0으로 재설정되고, 칼만 필터(순방향 추정 뒤 역방향 평활)로 상태 [θ, z]를 추정한다.",
            [("장점 ", B), ("모델과 맞는 합성 ECG에서 SNR 개선량이 크다(+16.2 dB).", {})],
            [("한계 ", B), ("R파 오검출 위치에 가성 QRS군을 만든다.", {})],
        ]),
        ("T3_unet.png", [
            H("③ 잔차 학습 1D U-Net"),
            "인코더가 신호 길이를 반씩 줄이며 채널을 늘리고, 디코더가 스킵 연결로 복원한 추정 잡음을 입력에서 뺀다.",
            EQ("eq_res", r"\hat{x} = y - f_\theta(y)"),
            EQ("eq_loss", r"\mathcal{L} = \mathcal{L}_{\mathrm{MSE}} + 0.5\,\mathcal{L}_{\mathrm{MAE}}"
                          r" + 0.3\,\mathcal{L}_{\mathrm{diff}} + \lambda\,\mathcal{L}_{\mathrm{clean}}"),
            [("장점 ", B), (f"{n['cross_db']:.0f} dB 미만 · 임펄스 잡음에서 우수하고, 가성 파형 생성 에너지가 고전 기법보다 낮다.", {})],
            [("한계 ", B), (f"무잡음 입력도 변형한다(출력 SNR {n['clean_m06']:.1f} dB, 무잡음 입력 보존 손실 적용 시 "
                          f"{n['clean_l6']:.1f} dB).", {})],
        ]),
    ]
    y = 20.6
    for fig, paras in cards:
        below = False                                # 세 카드 모두 그림을 글 오른쪽에 둔다(폭 통일)
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
    # 왼쪽 37 cm: 위 줄 = 그림 4 · 5(가)(나) · 6(조건별 최적 기법 · 안전성), 아래 줄 = 그림 7(딥러닝 개선 과정, 전폭).
    # 오른쪽 = 고찰(의의 → 한계와 원인 → 발전 방향). 그림 번호는 작품 내용의 연구 단계 순서를 따른다.
    # 캡션은 두 줄까지 — 제목과 그림이 보여 주는 결과 한 문장.
    B = {"bold": True}
    r0, xl, wl = 53.9, 1.8, 37.0
    # 위 줄 네 칸(2026-11-05 사용자 결정): 그림 4 교차점 · 그림 5 (가)(나) 잡음 종류별 우세 기법 두 쌍 · 그림 6 가성 QRS군.
    # 그림 5 의 캡션은 두 칸에 걸쳐 하나로 쓴다.
    cells = [("F4_crossover.png", PF.SIZES["F4"],
              "그림 4. 공통 전처리 대비 딥러닝의 SNR 개선량."),
             ("F5a_dl_wins.png", PF.SIZES["F5a"], None),
             ("F5b_classic_wins.png", PF.SIZES["F5b"], None),
             ("F5_phantom.png", PF.SIZES["F5"],
              "그림 6. 기법별 가성 QRS군(MIT-BIH).")]
    c5 = ("그림 5. 잡음 종류별 우세 기법(MIT-BIH, 입력 SNR +10 dB). 숫자 = 평균 SNR 개선량 [dB], 음영 = 최고, 아래 행 = "
          "잔차.")
    h1 = PF.SIZES["F4"][1]
    g1 = (wl - sum(sz[0] for _, sz, _ in cells)) / 3
    xs, x = [], xl
    for _, sz, _ in cells:
        xs.append(x); x += sz[0] + g1
    w5 = PF.SIZES["F5a"][0] + g1 + PF.SIZES["F5b"][0]
    hc1 = max(_text_h([cells[0][2]], cells[0][1][0], CAP_PT), _text_h([cells[3][2]], cells[3][1][0], CAP_PT),
              _text_h([c5], w5, CAP_PT))
    for (fig, sz, capt), x in zip(cells, xs):
        sh.image(PFIG / fig, x, r0, sz[0], sz[1])
        if capt:
            sh.text(x, r0 + h1 + 0.1, sz[0], hc1, capt, size=CAP_PT, color=INK2)
    sh.text(xs[1], r0 + h1 + 0.1, w5, hc1, c5, size=CAP_PT, color=INK2)
    for lab, x in (("(가)", xs[1]), ("(나)", xs[2])):
        sh.text(x, r0 - 0.05, 1.5, 0.7, [[(lab, {"bold": True})]], size=CAP_PT)
    y7 = r0 + h1 + 0.1 + hc1 + 0.35
    w7, h7 = PF.SIZES["F7"]
    sh.image(PFIG / "F7_flow.png", xl, y7, w7, h7)
    c7 = ("그림 7. 딥러닝 개선 과정. 기준 모델 대비 SNR 개선량 변화(동일 평가 집합, 동일 시드 대응 t-검정, "
          f"시드 n = {n['seed_lo']}~{n['seed_hi']}, 비교 {n['n_tests']}개 Holm 보정).")
    hc7 = _text_h([c7], w7, CAP_PT)
    sh.text(xl, y7 + h7 + 0.1, w7, hc7, c7, size=CAP_PT, color=INK2)
    yb = y7 + h7 + 0.1 + hc7
    xg = xl + wl + 0.9
    disc = [
        H("고찰"),
        [("연구의 의의", B)],
        f"• 기법의 우열은 잡음 종류 · 입력 SNR · 평가 데이터를 지정해야 정의된다. 합성 ECG에서 전 구간에 걸쳐 나타난 딥러닝의 "
        f"우위는 MIT-BIH에서 재현되지 않았고, 약 {n['cross_db']:.0f} dB 이상에서는 공통 전처리 단독이 더 높았다(그림 4).",
        f"• 칼만 필터의 가성 QRS군은 오차 에너지로는 드러나지 않았고, 박동 유사도로 측정하자 SWT의 약 "
        f"{n['ph_x_swt']:.0f}배였다(그림 6). 지표를 여러 층으로 둔 근거이다.",
        "• 시드 대응 비교와 Holm 보정으로 검정한 결과, 성능 향상은 학습 목표를 바꾼 시도(무잡음 입력 보존 손실, "
        "SNR 조건화)에서만 확인되었다(그림 7).",
        [("한계와 원인", B)],
        f"• 고 SNR 구간의 열세는 단일 모델이 전 입력 SNR 대역을 학습하며 절충한 결과로 해석된다. 15~20 dB 전용 모델은 "
        f"{n['band_hi']:+.1f} dB 높았으나 시드 3개의 결과이다(p_Holm = {n['band_hi_ph']:.2f}). 무잡음 입력 보존 손실과 SNR "
        f"조건화의 효과는 가산되지 않았다({n['l6cond_add']:+.1f} dB, p = {n['l6cond_add_p']:.2f}).",
        f"• SNR 조건화에는 실제 입력 SNR을 사용하였다. 실측 신호에는 추정 SNR이 필요하며, 조건 SNR을 8 dB 낮게 "
        f"입력하면 {n['pert_m8']:+.1f} dB, 높게 입력하면 {n['pert_p8']:+.1f} dB 변하였다.",
        "• 평가는 공개 데이터 두 종의 22개 기록에 한정되며, MIT-BIH 기준 신호가 0.5~100 Hz 대역 제한 신호이므로 "
        "결론도 이 대역에 한정된다.",
        [("발전 방향", B)],
        "• 자체 장치의 신호 · 잡음을 수집하여 입력 SNR 구간을 측정하고, 추정 SNR 조건화와 장치 간 도메인 적응을 "
        "검증한다.",
        "• 무잡음 입력 보존 손실을 사전 지정하여 주 실험을 재수행하고, 딥러닝 비교는 시드 10개로 재검정한다.",
    ]
    disc_pt = BODY_PT
    hd = _text_h(disc, 57.6 - xg, disc_pt)
    sh.text(xg, r0, 57.6 - xg, hd, disc, size=disc_pt)
    print(f"  결과 구역의 끝: 그림 {yb:.1f} cm, 고찰 {r0 + hd:.1f} cm (하단 77.6 cm)")
    if not final:
        sh.text(40.0, 0.4, 18.0, 0.6, "시안: build_panel.py 자동 생성, 수치는 실험 산출물에서 읽음",
                size=9, color=INK2)
    print(f"  panel 그림 면적 비율 (본문 구역 대비): {sh.ratio():.1%}")
    p = OUT / ("panel.pptx" if final else "panel_draft.pptx")
    prs.save(str(p))
    return p


# ------------------------------------------------------------------ 브로셔
def pipeline_png(path: Path) -> Path:
    """판넬의 실험 구성도(PowerPoint 도형)를 한 장짜리 pptx 로 그려 PNG 로 렌더한다 — 브로셔(docx)에 넣는 그림."""
    import tempfile

    from pptx import Presentation
    from pptx.util import Cm
    prs = Presentation()
    prs.slide_width, prs.slide_height = Cm(D1_W + 0.4), Cm(D1_H + 0.4)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    draw_pipeline(Sheet(prs, slide, body=(0.0, D1_H + 0.4)), 0.2, 0.2, s=1.0, fpt=16)   # 7.8 cm 로 줄면 약 5 pt — 상자에 들어가는 최대 크기
    with tempfile.TemporaryDirectory() as td:
        pp = Path(td, "pipeline.pptx")
        prs.save(str(pp))
        subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", td, str(pp)], check=True,
                       capture_output=True, timeout=300)
        subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", str(pp.with_suffix(".pdf")),
                        str(path.with_suffix(""))], check=True, timeout=300)
    return path


# 브로셔 양식(학과 HWP 양식, docx 로 변환한 파일)의 글자 규격 — 원본 PDF 에서 잰 값(2026-11-04).
# 본문 굴림 10pt · 줄 간격 16pt(HWP 160 %), 구역 제목 12pt 보통 굵기, 작품명 10pt 굵게. 표 다섯 행의 높이는 양식 그대로
# 두고 내용을 그 안에 맞춘다(「1 page 이내」).
BR_FONT = "굴림"


def _br_run(par, text, size=10.0, bold=False, color=(0, 0, 0)):
    from docx.oxml.ns import qn
    from docx.shared import Pt, RGBColor
    r = par.add_run(text)
    r.font.size, r.font.bold = Pt(size), bold
    r.font.color.rgb = RGBColor(*color)
    r.font.name = BR_FONT
    rpr = r._r.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = rpr.makeelement(qn("w:rFonts"), {}); rpr.insert(0, fonts)
    for k in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        fonts.set(qn(k), BR_FONT)
    return r


def _br_par(container, runs, size=10.0, line=16.0, align="l", before=0.0, after=0.0, indent=0.0, pad=0.15):
    """문단 하나. runs = 문자열 또는 [(글, {"bold": …})]. 줄 간격은 고정값(pt)으로 둔다 — HWP 의 160 % 와 같게."""
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
    from docx.shared import Cm, Pt
    par = container.add_paragraph()
    pf = par.paragraph_format
    if line:                                         # 그림 문단은 line=None — 고정 줄 간격이면 그림이 잘린다
        pf.line_spacing_rule, pf.line_spacing = WD_LINE_SPACING.EXACTLY, Pt(line)
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.left_indent, pf.right_indent = Cm(pad + indent), Cm(pad)   # 양식의 본문은 칸 테두리에서 약 2 mm 안쪽
    if indent:
        pf.first_line_indent = Cm(-indent)
    par.alignment = {"l": WD_ALIGN_PARAGRAPH.LEFT, "c": WD_ALIGN_PARAGRAPH.CENTER,
                     "j": WD_ALIGN_PARAGRAPH.JUSTIFY}[align]
    for txt, st in ([(runs, {})] if isinstance(runs, str) else runs):
        for k, part in enumerate(txt.split("p_Holm")):     # 판넬과 같게 p_Holm 의 Holm 을 아래 첨자로
            if k:
                _br_run(par, "p", size)
                _br_run(par, "Holm", size).font.subscript = True
            if part:
                _br_run(par, part, size, bold=st.get("bold", False))
    return par


def _br_clear(cell, keep: int):
    """셀의 앞 keep 개 문단만 남기고 지운다(양식의 안내문 · 빈 줄)."""
    for p in cell.paragraphs[keep:]:
        p._p.getparent().remove(p._p)


def _br_set(par, text, size=10.0, bold=False):
    for r in list(par.runs):
        r._r.getparent().remove(r._r)
    _br_run(par, text, size, bold=bold)


def _br_inner(cell, widths_cm):
    """셀 안의 테두리 없는 표(그림 나란히 놓기)."""
    from docx.oxml.ns import qn
    from docx.shared import Cm
    t = cell.add_table(rows=1, cols=len(widths_cm))
    t.autofit = False
    tblPr = t._tbl.tblPr
    borders = tblPr.makeelement(qn("w:tblBorders"), {})
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        borders.append(borders.makeelement(qn(f"w:{side}"), {qn("w:val"): "nil"}))
    tblPr.append(borders)
    grid = t._tbl.tblGrid                            # 격자 폭도 맞춘다 — 칸 폭만 바꾸면 LibreOffice · HWP 가 격자를 따른다
    for gc, w in zip(grid.findall(qn("w:gridCol")), widths_cm):
        gc.set(qn("w:w"), str(int(w / 2.54 * 1440)))
    tblW = tblPr.find(qn("w:tblW"))
    if tblW is None:
        tblW = tblPr.makeelement(qn("w:tblW"), {}); tblPr.append(tblW)
    tblW.set(qn("w:w"), str(int(sum(widths_cm) / 2.54 * 1440))); tblW.set(qn("w:type"), "dxa")
    for c, w in zip(t.rows[0].cells, widths_cm):
        c.width = Cm(w)
        for p in c.paragraphs:
            p._p.getparent().remove(p._p)
    return t


def build_brochure(n: dict, template: Path, final: bool = False) -> Path:
    """학과 브로셔 양식(docx)의 표에 채운다. 양식의 행 · 칸 · 글꼴을 바꾸지 않는다."""
    import docx
    from docx.shared import Cm
    d = docx.Document(str(template))
    t = d.tables[0]
    row = lambda i: list({id(c._tc): c for c in t.rows[i].cells}.values())  # noqa: E731
    # 머리 두 줄
    _br_set(row(0)[1].paragraphs[0], "심전도 잡음 제거 기법의 정량적 비교 연구", bold=True)
    _br_set(row(1)[3].paragraphs[0], "(입력)")
    _br_set(row(1)[5].paragraphs[0], "(입력)")
    # 구성(사용자 결정 2026-11-05, docs/38 §11): 작품 내용의 중심은 잡음 종류별 우세 기법(그림 두 쌍) — 가성 QRS군 —
    # 실시간 처리의 QRS 직후 왜곡. 문체는 학술 서술(질문형 금지). 그림 속 글자는 7 pt 까지 허용.
    m = n["noise_means"]
    # 1. 작품 목표
    c = row(2)[0]
    _br_clear(c, 1)
    _br_set(c.paragraphs[0], " 1. 작품 목표", size=12)
    _br_par(c, "자체 제작한 심전도(ECG) 취득 장치의 잔류 잡음을 파형을 보존하면서 줄이는 디지털 후처리 기법을 규명하는 "
               "것을 목표로 한다. 고전 필터, 웨이블릿 임계값 처리, 칼만 필터, 딥러닝의 네 계열을 동일한 입력 · 전처리 · 평가 "
               "지표로 비교하여 ① 잡음 종류와 입력 SNR에 따른 최적 기법, ② 입력에 없는 파형(가성 QRS군)의 생성, ③ 실시간 "
               "처리에서의 파형 왜곡을 분석하고, 그 결과를 실시간 측정과 전시 화면으로 구현한다.", align="j", before=2)
    # 2. 작품 개요 — 왼쪽 실험 구성도, 오른쪽 시연 화면(도입부의 입력 · 출력 겹치기 장면), 아래 응용 분야
    c = row(3)[0]
    _br_clear(c, 1)
    _br_set(c.paragraphs[0], " 2. 작품 개요", size=12)
    it = _br_inner(c, (8.3, 7.8))
    _br_par(it.rows[0].cells[0], "", line=None, align="c", pad=0).add_run().add_picture(
        str(PFIG / "B5_pipeline.png"), width=Cm(7.0))                  # 칸 안쪽 폭(그림 속 약 7 pt)
    _br_par(it.rows[0].cells[0], "[그림 1] 비교 실험 구성", size=8.5, line=11.5, align="c")
    # 시연 사진은 하나만 — 실시간 측정 화면을 우선한다(사용자 지시 2026-11-05). 본인 심전도라 git 제외 폴더에서 읽는다.
    _br_par(it.rows[0].cells[1], "", line=None, align="c", pad=0).add_run().add_picture(
        str(Path("data/arduino/captures/live_01_overlay_t16.png")), width=Cm(6.2))
    _br_par(it.rows[0].cells[1], "[그림 2] 자체 장치 신호의 실시간 처리 화면", size=8.5, line=11.5, align="c")
    _br_par(c, [("  - 응용 분야", {})], before=1)
    _br_par(c, "저가형 · 착용형 심전도 취득 장치의 신호 후처리 연구, 생체신호 잡음 제거 기법의 평가 방법론. 본 시스템은 "
               "연구용 신호처리 시스템이며 임상 진단 용도가 아니다.", align="j", indent=0)
    # 3. 작품 내용 및 이미지
    c = row(4)[0]
    _br_clear(c, 1)
    _br_set(c.paragraphs[0], " 3. 작품 내용 및 이미지", size=12)
    _br_par(c, [("잡음 종류별 우세 기법. ", {"bold": True}),
                (f"근전도 · 임펄스 잡음에서는 딥러닝이 SWT보다 {m['ma_synth']['M08'] - m['ma_synth']['M04']:.1f} · "
                 f"{m['impulse']['M08'] - m['impulse']['M04']:.1f} dB 높았고, 기저선 변동은 공통 전처리 단독이, 전원선 간섭은 "
                 "SWT가 가장 높았다. 이 두 잡음에서 딥러닝은 심전도 대역에 잔차를 남겼다(MIT-BIH, 입력 SNR +10 dB, "
                 f"숫자 = {n['noise_n']}개 구간 평균 SNR 개선량[dB], 음영 = 최고, 아래 행 = 잔차).", {})],
            size=9, line=13, align="j", before=1)
    it = _br_inner(c, (7.95, 7.95))
    for cell, (fig, capt) in zip(it.rows[0].cells, (
            ("N1_dl_wins.png", "[그림 3] 딥러닝이 우세한 잡음(근전도 · 임펄스)"),
            ("N2_classic_wins.png", "[그림 4] 고전 처리가 우세한 잡음(기저선 변동 · 전원선 간섭)"))):
        _br_par(cell, "", line=None, align="c", pad=0).add_run().add_picture(str(PFIG / fig), width=Cm(6.9))
        _br_par(cell, capt, size=8.5, line=11.5, align="c")
    it = _br_inner(c, (7.95, 7.95))
    for cell, (fig, w, capt, body) in zip(it.rows[0].cells, (
            ("B3_phantom.png", 6.9, "[그림 5] 칼만 필터의 가성 QRS군(MIT-BIH)",
             f"R파 오검출 위치에 박동 형태를 출력하였다({n['ph_M05']}건, SWT의 약 {n['ph_x_swt']:.0f}배). 오차 에너지로는 "
             "드러나지 않는다."),
            ("B4_causal.png", 6.9, "[그림 6] 실시간 공통 전처리의 QRS 직후 왜곡(MIT-BIH)",
             "인과 필터는 QRS 직후를 R파 높이의 약 5 % 왜곡하였고, 0.5 s 블록 영위상 필터는 오프라인 처리와 거의 "
             "같았다."))):
        _br_par(cell, "", line=None, align="c", pad=0).add_run().add_picture(str(PFIG / fig), width=Cm(w))
        _br_par(cell, [(capt + " ", {"bold": True}), (body, {})], size=8.5, line=11.5, align="j", pad=0.05)
    # 본문 세 행의 최소 높이를 내용에 맞춰 다시 나눈다 — 합계(양식 13,039 twip)는 그대로라 한 쪽이 유지된다.
    # 양식의 높이는 「최소」라 3 행에 그림 넷을 넣으면 행이 늘어 2 쪽이 되었다(2026-11-05).
    from docx.oxml.ns import qn as _qn
    for i, h in ((2, 1500), (3, 3000), (4, 6800)):
        t.rows[i]._tr.find(_qn("w:trPr")).find(_qn("w:trHeight")).set(_qn("w:val"), str(h))
    # 표 뒤의 빈 문단이 둘이면 2 쪽이 생긴다 — 하나만 남기고 1 pt 로
    from docx.shared import Pt
    body_ps = [p for p in d.paragraphs]
    for p in body_ps[1:]:
        p._p.getparent().remove(p._p)
    if body_ps:
        pf = body_ps[0].paragraph_format
        pf.space_before = pf.space_after = Pt(0); pf.line_spacing = Pt(1)
    p = OUT / ("brochure.docx" if final else "brochure_draft.docx")
    d.save(str(p))
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
    ap.add_argument("--brochure-template", type=Path, default=None,
                    help="학과 브로셔 양식(HWP 양식을 docx 로 변환한 파일). 저장소에 넣지 않는다 — 없으면 브로셔를 만들지 않는다")
    ap.add_argument("--render", action="store_true",
                    help="LibreOffice 로 PNG 미리보기도 만든다 (soffice · pdftoppm 필요)")
    ap.add_argument("--final", action="store_true",
                    help="최종판: 시안 표지를 빼고 panel.pptx · brochure.docx 로 저장, --render 시 PDF 와 PNG 를 남긴다")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    n = numbers()
    print("[numbers]", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in n.items()})
    p1 = build_panel(Path(a.template), n, a.final)
    p2 = build_brochure(n, a.brochure_template, a.final) if a.brochure_template else None
    print(f"  {p1}\n  {p2 or '브로셔: --brochure-template 없음, 건너뜀'}")
    if a.render:
        render(p1, 100 if a.final else 60)
        if p2:
            render(p2, 200 if a.final else 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
