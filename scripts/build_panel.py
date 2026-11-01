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

OUT = Path("results/panel")
SL = Path("results/slides")
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
    return n


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

    def image(self, path: Path, x, y, w, h):
        """상자 (x, y, w, h) 안에 비율을 지켜 맞춘다. 실제로 놓인 크기를 돌려준다."""
        iw, ih = Image.open(path).size
        sc = min(w / iw, h / ih)
        ww, hh = iw * sc, ih * sc
        xx, yy = x + (w - ww) / 2, y + (h - hh) / 2
        self.s.shapes.add_picture(str(path), self.Cm(xx), self.Cm(yy), self.Cm(ww), self.Cm(hh))
        self.img_area += ww * hh
        return xx, yy, ww, hh

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
def build_panel(template: Path, n: dict) -> Path:
    from pptx import Presentation
    prs = Presentation(str(template))
    slide = prs.slides[0]
    box = {sh.name: sh for sh in slide.shapes}
    H = lambda t: [(t, {"bold": True, "color": ACC})]  # noqa: E731

    _set(box["TextBox 16"], [[("ECG 잡음 제거 기법 비교 연구", {"bold": True, "size": 54})]], 54)
    _set(box["TextBox 17"], [[("조원 이름 : (입력)", {"bold": True})]], 24)
    _set(box["TextBox 18"], [[("담당교수 : (입력)", {"bold": True})]], 24)
    _set(box["TextBox 19"], [
        [("작품 개요", {"bold": True, "size": 24})],
        [("자체 제작 ECG 취득 장치(AD8232 + Arduino)의 잔여 잡음을 줄이는 디지털 후처리 기법 — 고전 필터 · "
          "웨이블릿 · 칼만 필터 · 딥러닝 — 을 같은 입력 · 같은 전처리 · 같은 지표로 비교했다. 합성 ECG 와 "
          "MIT-BIH 실기록에서 잡음 7 종 × 입력 SNR 7 단계를 재고, 실시간 시연 화면을 만들었다. "
          "연구용 신호처리 시스템이며 진단 용도가 아니다.", {"size": 19})]], 19)
    _set(box["TextBox 20"], [H("작품 내용")], 24)
    _set(box["TextBox 22"], [H("이론적 배경")], 24)
    _set(box["TextBox 21"], [H("결과 및 고찰")], 24)

    sh = Sheet(prs, slide, body=(17.5, 77.8))
    # ---- 작품 내용 (좌) x 1.9 ~ 27.6
    sh.text(1.9, 20.4, 25.7, 2.2, "「어떤 기법이 최고인가」 대신 「어떤 잡음 · 어떤 SNR 에서 무엇이 맞는가」를 묻는다. "
            "비교가 공정하도록 평가 장치를 기법보다 먼저 만들었다.", size=16)
    sh.image(SL / "S18_pipeline.png", 1.9, 22.9, 25.7, 13.5)
    sh.text(1.9, 36.5, 25.7, 1.0, "그림 1. 비교의 판 — 취득 · 공통 전처리 · 기법 네 계열과 성능 상한 · 평가 4 층",
            size=13, color=INK2)
    sh.frame(1.9, 37.9, 12.5, 9.2, "사진 — 취득 장치\n(전극 · AD8232 · Arduino Uno)\n팀 보유 사진을 넣는다")
    sh.image(OUT / "demo_mode_b.png", 15.1, 37.9, 12.5, 9.2)
    sh.text(1.9, 47.3, 12.5, 1.0, "그림 2. 취득 장치", size=13, color=INK2)
    sh.text(15.1, 47.3, 12.5, 1.6, "그림 3. 시연 화면 — 녹음본 비교 모드 (임펄스 잡음 0 dB)", size=13, color=INK2)
    sh.text(1.9, 48.9, 25.7, 1.8, "실시간 모드(실보드)의 화면 · 시연 영상 스틸은 그림 3 옆 또는 결과 칸 하단에 둔다.",
            size=12, color=INK2)

    # ---- 이론적 배경 (우) x 29.5 ~ 57.5
    sh.text(29.5, 20.4, 28.0, 1.4, "대표 기법 셋 — 무엇을 가정하나 → 그래서 어디서 강하고 어디서 약하나", size=16)
    cards = [
        ("S19_swt.png", 9.4, [
            H("① SWT 적응 임계 (M04) — 「잡음은 작은 계수에 있다」"),
            "λ_j = k_j · σ · √(2 ln N) · (1 − (1−ρ)·g(t))",
            "d̂ = d (1 − λ²/d²)  if |d| > λ  else 0   (garrote)",
            "σ = MAD(고주파 세부계수)/0.6745,  g = QRS 보호 창",
            [("강점 ", {"bold": True}), ("깨끗한 신호를 거의 안 건드린다", {})],
            [("약점 ", {"bold": True}), ("기저선 변동 · 임펄스는 계수 임계로 못 지운다", {})],
        ]),
        ("S20_kalman_model.png", 14.6, [
            H("② Sameni EKF/EKS (M05) — 「ECG 는 위상 위의 봉우리 합」"),
            "θ̇ = ω,   ż = −Σ (α_i ω / b_i²) Δθ_i exp(−Δθ_i² / 2b_i²)",
            "상태 [θ, z] 를 칼만 필터로 추정 · RTS 후방 평활",
            [("강점 ", {"bold": True}), ("모형이 맞으면 강하다", {})],
            [("약점 ", {"bold": True}), ("θ 를 검출 R-peak 로 맞춘다 → 틀린 자리에 심박을 그린다", {})],
        ]),
        ("S21_resunet.png", 14.6, [
            H("③ Residual U-Net (M06) + 손실 L6 — 「잡음을 예측해 뺀다」"),
            "x̂ = y − f_θ(y)",
            "L6 = MSE + 0.5·MAE + 0.3·차분 + λ·‖f(clean) − clean‖",
            [("강점 ", {"bold": True}), ("저 SNR · 임펄스, 없는 파형을 만들지 않는다", {})],
            [("약점 ", {"bold": True}), ("깨끗한 입력을 건드린다 → L6 로 고쳤다", {})],
        ]),
    ]
    y = 22.2
    for fig, fw, paras in cards:
        tw = 28.0 - fw - 0.4
        sh.text(29.5, y, tw, 9.0, paras, size=15)
        sh.image(SL / fig, 29.5 + tw + 0.4, y, fw, 9.0)
        y += 9.5

    # ---- 결과 및 고찰 (하단 전폭) x 1.8 ~ 57.6, y 53.6 ~ 77.6
    r1, r2 = 54.0, 66.0
    cap = 13
    sh.image(SL / "S5_crossover.png", 1.8, r1, 16.9, 9.6)
    sh.text(1.8, r1 + 9.8, 16.9, 1.8,
            f"① 딥러닝이 front-end 위에 더하는 값은 저 SNR 에서 크다(D1 −5 dB +{n['dl_gain_lo']:.1f} dB) — "
            f"실데이터에서는 약 {n['cross_db']:.0f} dB 위로 가면 0 아래가 된다", size=cap)
    sh.image(SL / "S15_kalman_phantom.png", 19.1, r1, 17.6, 9.6)
    sh.text(19.1, r1 + 9.8, 17.6, 1.8,
            f"② 칼만은 없는 자리에 깨끗한 심박을 그린다 — 심박 모양 가짜 {n['ph_M05']} 개, "
            f"SWT 의 {n['ph_x_swt']:.0f} 배 · 딥러닝의 {n['ph_x_dl']:.0f} 배 (D1, −5·0 dB)", size=cap)
    sh.image(SL / "S4_noise_d1.png", 37.4, r1, 20.2, 9.6)
    sh.text(37.4, r1 + 9.8, 20.2, 1.8, "③ 잡음 종류가 답을 바꾼다 — 기저선은 전처리로 끝, "
            "임펄스는 딥러닝만 지운다", size=cap)
    sh.image(SL / "S16_wavelet_placement.png", 1.8, r2, 15.6, 10.0)
    sh.text(1.8, r2 + 10.1, 15.6, 1.6, f"④ wavelet 을 직렬로 붙이면 손해({n['m07']:+.2f} dB), "
            f"특성공간은 {n['m08']:+.2f} dB 로 못 가림", size=cap)
    sh.image(SL / "S17_dl_journey.png", 17.9, r2, 15.0, 10.0)
    sh.text(17.9, r2 + 10.1, 15.0, 1.6, f"⑤ 개선 시도 16 개 — 움직인 것은 학습 목표뿐 "
            f"(L6 고 SNR {n['l6_hi']:+.1f} dB)", size=cap)
    sh.text(33.6, r2, 24.0, 11.6, [
        H("고찰"),
        "• 단일 최선은 없다. 잡음 종류와 입력 SNR 에 따라 처리를 고르는 편이 낫다.",
        "• 모형 기반 필터는 모형이 맞을 때만 강하고, 틀리면 그럴듯하게 틀린다. "
        "잔차 구조의 딥러닝은 지어내지 않았다.",
        f"• 딥러닝을 끌어올린 것은 모델 크기가 아니라 학습 목표였다. L6 로 깨끗한 신호 보존이 "
        f"{n['clean_m06']:.1f} → {n['clean_l6']:.1f} dB (SWT {n['clean_swt']:.1f}).",
        "• 그 판정은 학습 런 간 산포(seed 만 바꿔 1.43 dB)를 먼저 잰 뒤에 내렸다.",
        [("한계 ", {"bold": True}), ("두 데이터축 모두 공개 데이터 — 자체 장비 실측 축은 진행 중이다.", {})],
    ], size=14)
    sh.text(40.0, 0.4, 18.0, 0.6, "시안 — scripts/build_panel.py 자동 생성 · 수치는 results/ 에서 읽음",
            size=9, color=INK2)
    print(f"  panel 그림 면적 비율 (본문 구역 대비): {sh.ratio():.0%}")
    p = OUT / "panel_draft.pptx"
    prs.save(str(p))
    return p


# ------------------------------------------------------------------ 브로셔
def build_brochure(n: dict) -> Path:
    from pptx import Presentation
    from pptx.util import Cm
    prs = Presentation()
    prs.slide_width, prs.slide_height = Cm(21.0), Cm(29.7)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    sh = Sheet(prs, slide, body=(4.5, 28.5))
    H = lambda t: [(t, {"bold": True, "size": 12, "color": ACC})]  # noqa: E731
    sh.text(1.6, 1.2, 17.8, 1.0, [[("작품명 | ", {"bold": True}), ("ECG 잡음 제거 기법 비교 연구", {"bold": True})]], size=15)
    sh.text(1.6, 2.4, 17.8, 0.8, [[("학 과 | ", {"bold": True}), ("전기정보공학과   ", {}),
                                   ("조 원 | ", {"bold": True}), ("(입력)   ", {}),
                                   ("지도교수 | ", {"bold": True}), ("(입력)", {})]], size=10.5)
    sh.text(1.6, 4.0, 17.8, 0.7, [H("1. 작품 목표")])
    sh.text(1.6, 4.8, 17.8, 2.0, "자체 제작 ECG 취득 장치의 잔여 잡음을, 심전도 형태를 보존하면서 줄이는 디지털 후처리 "
            "기법을 찾는다. 고전 필터 · 웨이블릿 · 칼만 필터 · 딥러닝을 같은 조건과 같은 지표로 비교해 어떤 잡음 "
            "조건에서 무엇이 맞는지 규명한다.", size=10)
    sh.text(1.6, 7.2, 17.8, 0.7, [H("2. 작품 개요")])
    sh.image(SL / "S18_pipeline.png", 1.6, 8.0, 17.8, 8.0)
    sh.text(1.6, 16.2, 17.8, 3.8, [
        "- 공통 전처리 위에서 네 계열의 기법과 성능 상한을 잡음 7 종 × 입력 SNR 7 단계로 비교",
        "- 지표 4 층(재구성 · 형태 · R-peak 검출 · 주파수)과 「없는 파형을 만드는가」 안전성 프로브",
        "- 합성 ECG 와 MIT-BIH 실기록 두 데이터축에서 결론이 재현되는지 확인",
        "- 녹음본 비교 화면과 실보드 실시간 화면을 갖춘 시연 시스템",
        [("- 응용 분야: ", {"bold": True}), ("저가 · 웨어러블 심전도 취득 장치의 신호 후처리 연구, "
                                             "생체신호 잡음 제거의 평가 방법론 (진단 용도 아님)", {})],
    ], size=9.5)
    sh.text(1.6, 20.3, 17.8, 0.7, [H("3. 작품 내용 및 이미지")])
    w = 5.75
    for i, (img, capt) in enumerate([
            (OUT / "demo_mode_b.png", "[그림 1] 시연 화면 (녹음본 비교)"),
            (SL / "S5_crossover.png", f"[그림 2] 딥러닝이 유리한 SNR 구간 (~{n['cross_db']:.0f} dB 아래)"),
            (SL / "S15_kalman_phantom.png", f"[그림 3] 칼만이 그린 가짜 심박 ({n['ph_M05']} 개)")]):
        x = 1.6 + i * (w + 0.275)
        sh.image(img, x, 21.1, w, 4.6)
        sh.text(x, 25.8, w, 0.9, capt, size=8.5, align="c")
    sh.text(1.6, 27.0, 17.8, 1.6, f"칼만 필터는 검출이 틀린 자리에 깨끗한 심박을 그렸고(SWT 의 {n['ph_x_swt']:.0f} 배), "
            "딥러닝은 지어내지 않았다. 딥러닝 성능을 끌어올린 것은 모델 크기가 아니라 학습 목표(손실 L6)였다.",
            size=9.5)
    print(f"  brochure 그림 면적 비율: {sh.ratio(1.6, 19.4):.0%}")
    p = OUT / "brochure_draft.pptx"
    prs.save(str(p))
    return p


def render(pptx: Path) -> None:
    """LibreOffice 로 PDF → PNG. 글꼴은 이 환경의 대체 글꼴로 그려진다(맑은 고딕 없음)."""
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(pptx.parent),
                    str(pptx)], check=True, capture_output=True, timeout=300)
    pdf = pptx.with_suffix(".pdf")
    subprocess.run(["pdftoppm", "-png", "-r", "60", "-singlefile", str(pdf), str(pptx.with_suffix(""))],
                   check=True, timeout=300)
    pdf.unlink()
    print(f"  {pptx.with_suffix('.png')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--template", required=True,
                    help="학과 판넬 양식 pptx (A00_panel). 저장소에 넣지 않는다 — 받은 파일 경로를 준다")
    ap.add_argument("--render", action="store_true",
                    help="LibreOffice 로 PNG 미리보기도 만든다 (soffice · pdftoppm 필요)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    n = numbers()
    print("[numbers]", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in n.items()})
    p1, p2 = build_panel(Path(a.template), n), build_brochure(n)
    print(f"  {p1}\n  {p2}")
    if a.render:
        render(p1)
        render(p2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
