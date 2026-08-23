"""학습된 딥러닝 모델을 `Denoiser` 계약으로 감싼다 (docs/02_procedure.md STEP 20).

    y (임의 길이)
      -> frame(win, hop)                     50% overlap
      -> window 별 robust_scale (noisy 기준)  docs/00_review.md A-6
      -> model
      -> x scale (역정규화)
      -> overlap-add (Hann^2 가중)            docs/00_review.md A-5

**모든 방법과 동일한 프레이밍 규약을 쓴다.** DSP 는 전체 신호를 한 번에 처리하지만,
경계 처리 규약(guard band)이 같으므로 비교는 공정하다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import HOP, WIN
from ..registry import register_method
from ..utils import robust_scale
from ..data.windows import frame
from .base import BaseDenoiser

__all__ = ["DLDenoiser", "load_checkpoint"]


def load_checkpoint(path: str | Path, device: str = "cpu"):
    """체크포인트에서 모델을 복원한다."""
    import torch

    from ..models import build_model

    ck = torch.load(Path(path), map_location=device, weights_only=False)
    name = ck.get("model_name", "resunet1d")
    kw = ck.get("model_kwargs", {}) or {}
    model = build_model(name, **kw)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, ck


class DLDenoiser(BaseDenoiser):
    def __init__(self, model=None, ckpt: str | Path | None = None,
                 name: str = "M06", win: int = WIN, hop: int = HOP,
                 device: str = "cpu", batch: int = 64, normalize: bool = True,
                 pre: str | None = None, frontend: bool = True):
        import torch

        self.name = name
        self.win, self.hop = int(win), int(hop)
        self.device = device
        self.batch = int(batch)
        self.normalize = bool(normalize)
        self.meta: dict[str, Any] = {}
        if model is None:
            if ckpt is None:
                raise ValueError("model 또는 ckpt 중 하나는 필요하다")
            model, ck = load_checkpoint(ckpt, device)
            self.meta = {k: v for k, v in ck.items() if k != "model"}
        self.model = model
        self.model.eval()
        self._torch = torch
        # 공통 front-end. 학습 때와 **같은** 전처리를 추론에서도 적용해야 한다.
        # (학습은 창 양쪽에 여유를 붙여 필터링하고, 추론은 전체 신호에 한 번 적용한다 —
        #  둘 다 FE 가 충분한 문맥을 보므로 정합한다)
        self.frontend = bool(frontend)
        self._fe = None
        if self.frontend:
            from .frontend import FrontEnd
            self._fe = FrontEnd()
        # 순차 hybrid(M07): 학습 때와 같은 DSP 전처리
        self.pre = None
        if pre is not None:
            from ..registry import build
            self.pre = build(pre)

    def _run(self, y: np.ndarray, fs: float, ctx: dict[str, Any]) -> np.ndarray:
        torch = self._torch
        if self._fe is not None:
            y = self._fe(y, fs)
        if self.pre is not None:
            y = self.pre(y, fs)
        n = y.size
        frames, pl, pr = frame(y, self.win, self.hop, pad="reflect", apply_window=False)
        nf = frames.shape[0]

        if self.normalize:
            scales = np.array([robust_scale(f) for f in frames])[:, None]
        else:
            scales = np.ones((nf, 1))
        inp = frames / scales

        outs = np.empty_like(inp)
        with torch.no_grad():
            for i in range(0, nf, self.batch):
                blk = torch.from_numpy(inp[i:i + self.batch].astype(np.float32))[:, None, :]
                blk = blk.to(self.device)
                pred = self.model(blk)
                if isinstance(pred, tuple):
                    pred = pred[0]
                outs[i:i + self.batch] = pred[:, 0].cpu().numpy().astype(np.float64)
        outs = outs * scales

        # 합성: Hann^2 가중 overlap-add (windows.process_framed 와 동일 규약)
        from ..data.windows import analysis_window

        w2 = analysis_window(self.win) ** 2
        total = (nf - 1) * self.hop + self.win
        acc = np.zeros(total); wsq = np.zeros(total)
        for k in range(nf):
            s = k * self.hop
            acc[s:s + self.win] += outs[k] * w2
            wsq[s:s + self.win] += w2
        out = np.where(wsq > 1e-12, acc / np.maximum(wsq, 1e-12), 0.0)
        return out[pl:pl + n]


def register_dl(method_id: str, ckpt: str | Path, label: str = "", **kw):
    """학습이 끝난 뒤 체크포인트를 레지스트리에 등록한다."""
    @register_method(method_id, family="deep", label=label or method_id, needs_ckpt=True)
    def _f(**kw2):
        return DLDenoiser(ckpt=ckpt, name=method_id, **{**kw, **kw2})
    return _f
