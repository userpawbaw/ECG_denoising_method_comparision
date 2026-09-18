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
                 name: str = "M06", win: int | None = None,
                 hop: int | None = None,
                 device: str = "cpu", batch: int = 64, normalize: bool = True,
                 pre: str | None = None, frontend: bool | None = None,
                 cond_snr: float | str | None = None,
                 cond_snr_offset: float = 0.0):
        import torch

        self.name = name
        self.device = device
        self.batch = int(batch)
        self.normalize = bool(normalize)
        self.meta: dict[str, Any] = {}
        if model is None:
            if ckpt is None:
                raise ValueError("model 또는 ckpt 중 하나는 필요하다")
            model, ck = load_checkpoint(ckpt, device)
            self.meta = {k: v for k, v in ck.items() if k != "model"}
            # **학습 때의 설정을 따른다.** 추론에서만 front-end 를 켜고 끄면
            # train/inference 불일치가 되어 성능이 무너진다.
            if frontend is None:
                if "frontend" in ck:
                    frontend = bool(ck["frontend"])
                else:
                    frontend = False
                    print(f"[warn] {Path(ckpt).parent.name}: 체크포인트에 frontend 기록이 없다. "
                          "front-end 없이 학습된 구형 체크포인트로 간주한다 (재학습 권장).")
            if pre is None and ck.get("pre_denoise"):
                pre = ck["pre_denoise"]
            # **학습 window 를 따른다.** 다른 길이로 학습한 모델을 기본값 1024 로
            # 돌리면 에러 없이 어긋난 채 결과가 나온다 (frontend 와 같은 계열).
            if win is None:
                if "data_win" in ck:
                    win, hop = int(ck["data_win"]), int(ck.get("data_hop", ck["data_win"] // 2))
                else:
                    print(f"[warn] {Path(ckpt).parent.name}: 체크포인트에 window 기록이 "
                          f"없다. 기본값 {WIN} 로 간주한다 (그 이전 체크포인트는 "
                          f"전부 {WIN} 로 학습됐다).")
        if frontend is None:
            frontend = True
        self.win = int(win) if win is not None else WIN
        self.hop = int(hop) if hop is not None else self.win // 2
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
        # 조건화 모델(D-28 2 번)의 SNR 출처.
        #   float -> 고정값 · "est" -> 신호 전체에서 추정 · None -> 없음
        # **추론에서 참 SNR 을 쓸 길은 없다** — 그건 오라클이다(B01·B02 와 같은
        # 자리). 실험 단계에서 조건화 모델을 돌리려면 추정값을 쓰거나 고정해야
        # 하고, 그 대가를 재는 것이 D-28 4 번이다.
        self.cond_snr = cond_snr
        self.cond_snr_offset = float(cond_snr_offset)
        if getattr(self.model, "needs_cond", False) and cond_snr is None:
            raise ValueError(
                f"{self.name}: 조건화 모델인데 cond_snr 이 없다. "
                '추정값은 cond_snr="est", 고정값은 숫자를 준다.')
        if cond_snr is not None and not getattr(self.model, "needs_cond", False):
            raise ValueError(f"{self.name}: 조건화하지 않는 모델에 cond_snr 을 줬다")

    def _cond_value(self, y: np.ndarray, fs: float) -> float:
        """창마다가 아니라 **신호 하나에 한 번** 정한다.

        4.096 s 창에는 박동이 4~5 개뿐이라 추정기가 요구하는 6 개에 못 미친다
        (`estimate_snr_hsc` 가 NaN 을 낸다). 실제 운용에서도 «구간 단위로 한 번
        재고 그 구간에 적용» 이 자연스럽다.
        """
        if self.cond_snr != "est":
            return float(self.cond_snr) + self.cond_snr_offset
        from ..eval.snr_estimation import estimate_snr_hsc
        v, _ = estimate_snr_hsc(y, fs)
        if not np.isfinite(v):
            # 못 재면 학습 범위의 가운데. **조용히 넘어가지 않고 기록한다.**
            v = 0.5 * (self.model.embed.lo + self.model.embed.hi)
            self.meta["cond_snr_fallback"] = True
        return float(v) + self.cond_snr_offset

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

        cond = self._cond_value(y, fs) if self.cond_snr is not None else None
        if cond is not None:
            ctx["cond_snr"] = cond

        outs = np.empty_like(inp)
        with torch.no_grad():
            for i in range(0, nf, self.batch):
                blk = torch.from_numpy(inp[i:i + self.batch].astype(np.float32))[:, None, :]
                blk = blk.to(self.device)
                if cond is None:
                    pred = self.model(blk)
                else:
                    pred = self.model(blk, torch.full((blk.shape[0],), cond,
                                                      device=self.device))
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
