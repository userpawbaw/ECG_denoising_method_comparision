"""판넬 이론 카드의 수식을 LaTeX 원문에서 그림으로 조판한다 (`build_panel.py` 가 쓴다).

이 환경에는 TeX 가 없어 matplotlib mathtext(Computer Modern 글꼴)로 조판한다. 그림은 600 dpi 로
저장하고 판넬에는 원래 크기 그대로 놓으므로, 그림 속 글자 크기가 지정한 pt 와 같다.
수식 문자열에는 한글이 없고 빼기는 LaTeX 의 '-' 로 쓴다.
"""
from __future__ import annotations

from pathlib import Path

OUT = Path("results/panel/eq")


def render_eq(eq, pt: float, out_dir: Path = OUT) -> Path:
    """eq = (이름, LaTeX 원문). `out_dir/이름.png` 를 만들어 경로를 돌려준다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    name, tex = eq
    out = Path(out_dir) / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"mathtext.fontset": "cm", "axes.unicode_minus": False}):
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${tex}$", fontsize=pt, color="#1a1613")
        fig.savefig(out, dpi=600, transparent=True, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return out
