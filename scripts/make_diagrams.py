"""판넬용 블록선도를 mermaid 로 조판한다 (실험 구성 D1, 잔차 학습 U-Net D2).

    MMDC=<mmdc 경로> python scripts/make_diagrams.py          # 둘 다
    MMDC=<mmdc 경로> python scripts/make_diagrams.py --only D2

산출: `results/slides/diagrams/D1_pipeline.{mmd,png}`, `D2_resunet.{mmd,png}`.

matplotlib 으로 그리던 S18 · S21 은 상자 안 글이 많고 정렬이 거칠었다(사용자 피드백 2026-11-03).
이 판은 상자에 핵심어만 두고 설명은 판넬 캡션으로 옮긴다. mermaid 원문(.mmd)은 이 스크립트가
만들고 함께 저장하므로, 원문을 직접 고치지 말고 스크립트를 고친다.

- 수치(채널 수, 신호 길이, 파라미터 수)는 체크포인트에서 읽는다.
- D2 는 상자 높이를 해상도(신호 길이)에 비례시킨다 — 기존 S21 에서 유지할 만했던 부분이다.
- 렌더러: mermaid-cli(`@mermaid-js/mermaid-cli`). 저장소 의존성이 아니므로 `MMDC` 환경 변수로
  실행 파일을 준다. 없으면 `npx -y @mermaid-js/mermaid-cli@11.4.2` 를 쓴다. 브라우저는
  `CHROME_PATH`(기본 /opt/pw-browsers/chromium)를 쓴다.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401

OUT = Path("results/slides/diagrams")
FONT = "NanumGothic, Malgun Gothic, sans-serif"

# 색은 make_slides 의 BOXC 계열(옅은 바탕 + 진한 테두리)과 맞춘다
CLASSDEF = """
classDef data fill:#ece9e1,stroke:#8f8a80,color:#1a1613
classDef noise fill:#f3e6cf,stroke:#b39257,color:#1a1613
classDef fe fill:#f6d6e3,stroke:#b5718d,color:#1a1613
classDef cls fill:#d5e4f7,stroke:#5b86bd,color:#1a1613
classDef tf fill:#fbdccc,stroke:#c77c55,color:#1a1613
classDef mb fill:#fbecc2,stroke:#b8962e,color:#1a1613
classDef dl fill:#cdeee1,stroke:#3f9a77,color:#1a1613
classDef ub fill:#e4e2dc,stroke:#8f8a80,color:#1a1613,stroke-dasharray:4 3
classDef ev fill:#e6e5f4,stroke:#6f6aa8,color:#1a1613
classDef enc fill:#c9d6e8,stroke:#5b86bd,color:#1a1613
classDef dec fill:#cdeee1,stroke:#3f9a77,color:#1a1613
classDef op fill:#ffffff,stroke:#52514e,color:#1a1613
classDef out fill:#fbecc2,stroke:#b8962e,color:#1a1613
classDef est fill:#f6c9c9,stroke:#c25b5b,color:#1a1613
"""


def d1_pipeline() -> str:
    """실험 구성 — 위에서 아래로 입력 · 잡음 · 공통 전처리 · 기법 네 계열과 상한 · 평가. 상자에는 핵심어만.

    전체는 위 → 아래(TB)로 두고, 단계 안의 상자는 같은 줄에 놓는다. mermaid 는 바깥과 연결된 묶음의
    `direction` 을 무시하므로, 바깥과 연결되지 않은 안쪽 묶음(테두리 없음)을 하나 더 둔다. 안쪽 묶음의 TB 에서
    서로 연결되지 않은 상자는 같은 순위가 되어 한 줄로 놓인다.
    판넬의 그림 칸(가로 : 세로 약 1.9)에 맞추기 위한 배치이다.
    """
    return f"""flowchart TB
subgraph PREP[" "]
  subgraph PREP_r[" "]
    direction LR
    subgraph IN["입력 신호"]
      A1["합성 ECG"]:::data
      A2["MIT-BIH 실측 기록"]:::data
      A3["자체 장치 · AD8232"]:::data
    end
    N["잡음 7종 × 입력 SNR 7단계<br/>(-5~25 dB)"]:::noise
    FE["공통 전처리<br/>0.5~100 Hz 대역 제한<br/>전원선 notch"]:::fe
    IN --> N --> FE
  end
end
subgraph M["잡음 제거 기법 (동일 입력)"]
  subgraph M_r[" "]
    direction TB
    M1["고전 필터"]:::cls
    M2["웨이블릿<br/>임계값 처리"]:::tf
    M3["칼만 평활기"]:::mb
    M4["딥러닝"]:::dl
    UB["이상적 상한"]:::ub
  end
end
subgraph EV["평가 (동일 지표)"]
  subgraph EV_r[" "]
    direction TB
    E1["지표 4범주"]:::ev
    E2["가성 파형 생성 검증"]:::ev
  end
end
PREP --> M --> EV
{CLASSDEF}
style PREP fill:none,stroke:none
style PREP_r fill:none,stroke:none
style IN fill:#faf9f6,stroke:#c9c5bb
style M fill:#faf9f6,stroke:#c9c5bb
style EV fill:#faf9f6,stroke:#c9c5bb
style M_r fill:none,stroke:none
style EV_r fill:none,stroke:none
"""


def _model_info() -> dict:
    from ecgdn.methods.dl_wrapper import load_checkpoint
    model, _ = load_checkpoint("results/d1/m06_l1/best.pt")
    return {"chs": [int(c) for c in model.chs], "npar": int(sum(p.numel() for p in model.parameters())),
            "L": 1024}


def _blk(text: str, length: int, L: int) -> str:
    """상자 높이를 신호 길이에 비례시킨다(가장 긴 단계 = 130 px, 가장 짧은 단계도 읽히게 하한 44 px)."""
    h = max(44, int(130 * length / L))
    return (f"<div style='height:{h}px;width:58px;display:flex;align-items:center;justify-content:center;"
            f"line-height:1.2'>{text}</div>")


def d2_resunet(info: dict) -> str:
    """잔차 학습 1D U-Net — 인코더(파랑) · 병목 · 디코더(초록) · 스킵 연결(점선) · 출력단 잔차.

    왼쪽 → 오른쪽으로 놓는다. U 자를 세로로 세우는 배치도 시험했으나 mermaid 에는 역방향 화살표가 없어
    디코더의 순위가 어긋났다. 상자 높이가 해상도를 나타낸다.
    """
    chs, L = info["chs"], info["L"]
    k = len(chs) - 1
    lines = ["flowchart LR", f'Y["{_blk("입력<br/>y", 512, L)}"]:::data']
    for i in range(k):
        lines.append(f'E{i}["{_blk(f"{chs[i]}ch<br/>{L >> i}", L >> i, L)}"]:::enc')
    lines.append(f'B["{_blk(f"{chs[-1]}ch<br/>{L >> k}", L >> k, L)}"]:::enc')
    for i in reversed(range(k)):
        lines.append(f'D{i}["{_blk(f"{chs[i]}ch<br/>{L >> i}", L >> i, L)}"]:::dec')
    lines.append('NH["추정<br/>잡음 n̂"]:::est')
    lines.append("SUB((\"<span style='font-size:26px'>-</span>\")):::op")
    lines.append('X["출력<br/>x̂ = y - n̂"]:::out')
    chain = ["Y"] + [f"E{i}" for i in range(k)] + ["B"] + [f"D{i}" for i in reversed(range(k))] + ["NH", "SUB", "X"]
    lines.append(" --> ".join(chain))
    for i in range(k):
        lines.append(f"E{i} -.- D{i}")
    lines.append("Y -- 출력단 잔차 연결 --> SUB")
    lines.append(CLASSDEF)
    lines.append("linkStyle default stroke:#52514e,stroke-width:1.6px")
    return "\n".join(lines) + "\n"


def render(name: str, src: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    mmd = OUT / f"{name}.mmd"
    mmd.write_text(src, encoding="utf-8")
    png = OUT / f"{name}.png"
    cfg = {"theme": "base", "securityLevel": "loose",
           "themeVariables": {"fontFamily": FONT, "fontSize": "20px", "primaryColor": "#ffffff",
                              "lineColor": "#52514e", "textColor": "#1a1613"},
           "flowchart": {"htmlLabels": True, "curve": "basis", "nodeSpacing": 22, "rankSpacing": 30,
                         "padding": 10, "wrappingWidth": 420,
                         "subGraphTitleMargin": {"top": 6, "bottom": 2}}}
    pup = {"executablePath": os.environ.get("CHROME_PATH", "/opt/pw-browsers/chromium"),
           "args": ["--no-sandbox", "--disable-gpu"]}
    mmdc = os.environ.get("MMDC") or shutil.which("mmdc")
    cmd = [mmdc] if mmdc else ["npx", "-y", "@mermaid-js/mermaid-cli@11.4.2"]
    with tempfile.TemporaryDirectory() as td:
        c, p = Path(td, "cfg.json"), Path(td, "pup.json")
        c.write_text(json.dumps(cfg)); p.write_text(json.dumps(pup))
        subprocess.run(cmd + ["-i", str(mmd), "-o", str(png), "-c", str(c), "-p", str(p),
                              "-s", "4", "-b", "#fcfcfb"], check=True, timeout=300)
    _trim(png)
    print(f"  {png}")
    return png


def _trim(png: Path, margin: int = 24) -> None:
    """바탕색만 있는 가장자리를 잘라 낸다. 보이지 않는 묶음(배치용)이 남기는 위쪽 여백을 없앤다."""
    from PIL import Image, ImageChops
    im = Image.open(png).convert("RGB")
    bg = Image.new("RGB", im.size, (0xfc, 0xfc, 0xfb))
    box = ImageChops.difference(im, bg).convert("L").point(lambda v: 255 if v > 6 else 0).getbbox()
    if box:
        x0, y0, x1, y1 = box
        im.crop((max(0, x0 - margin), max(0, y0 - margin), min(im.width, x1 + margin),
                 min(im.height, y1 + margin))).save(png)


def main() -> int:
    ap = argparse.ArgumentParser(description="판넬 블록선도(mermaid): D1 실험 구성, D2 잔차 학습 U-Net")
    ap.add_argument("--only", nargs="*", default=None, help="D1 D2 중 일부만 생성")
    a = ap.parse_args()
    want = a.only or ["D1", "D2"]
    from ecgdn.utils import save_manifest
    if "D1" in want:
        render("D1_pipeline", d1_pipeline())
    if "D2" in want:
        render("D2_resunet", d2_resunet(_model_info()))
    save_manifest(OUT, cfg={"diagrams": want}, sources=["scripts/make_diagrams.py"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
