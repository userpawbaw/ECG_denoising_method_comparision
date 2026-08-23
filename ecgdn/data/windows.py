"""윈도잉 / Overlap-Add (docs/02_procedure.md STEP 04, docs/00_review.md A-5).

모든 방법(DSP·DL)이 **동일한** 프레이밍 규약을 쓴다. 그래야 비교가 공정하다.

규약
----
  분석창 = 합성창 = sqrt(periodic Hann), hop = win/2 (50% overlap).
  합성 시 누적된 창제곱합으로 나누어 정규화하므로, 경계 프레임을 포함해
  **임의 길이 N 에 대해 정확 재구성**된다 (COLA 조건에 의존하지 않음).
"""
from __future__ import annotations

import numpy as np

from ..config import HOP, WIN

__all__ = ["analysis_window", "frame", "overlap_add", "process_framed", "n_frames_for"]

_EPS = 1e-12


def analysis_window(win: int = WIN) -> np.ndarray:
    """sqrt(periodic Hann). 분석·합성 양쪽에 곱한다."""
    return np.sqrt(np.hanning(win + 1)[:win]) if win > 1 else np.ones(win)


def n_frames_for(n: int, win: int = WIN, hop: int = HOP) -> int:
    pad_left = win - hop
    target = pad_left + n + hop
    return max(1, int(np.ceil((target - win) / hop)) + 1)


def frame(x: np.ndarray, win: int = WIN, hop: int = HOP, pad: str = "reflect",
          apply_window: bool = True) -> tuple[np.ndarray, int, int]:
    """(frames, pad_left, pad_right) 반환. frames: (n_frames, win)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        raise ValueError("empty signal")
    pad_left = win - hop
    nf = n_frames_for(n, win, hop)
    total = (nf - 1) * hop + win
    pad_right = total - pad_left - n
    if pad_right < 0:  # 방어
        nf += 1
        total = (nf - 1) * hop + win
        pad_right = total - pad_left - n

    mode = pad if n > 1 else "edge"
    if mode == "reflect" and n <= max(pad_left, pad_right):
        mode = "edge"          # reflect 는 pad 폭 < n 이어야 함
    xp = np.pad(x, (pad_left, pad_right), mode=mode)

    idx = np.arange(win)[None, :] + hop * np.arange(nf)[:, None]
    frames = xp[idx]
    if apply_window:
        frames = frames * analysis_window(win)[None, :]
    return frames, pad_left, pad_right


def overlap_add(frames: np.ndarray, n: int, pad_left: int, hop: int = HOP,
                apply_window: bool = True) -> np.ndarray:
    """frame() 의 역연산. 원래 길이 n 의 신호를 복원한다."""
    frames = np.asarray(frames, dtype=np.float64)
    nf, win = frames.shape
    total = (nf - 1) * hop + win
    w = analysis_window(win)

    acc = np.zeros(total)
    wsq = np.zeros(total)
    contrib = frames * w[None, :] if apply_window else frames
    wcontrib = (w ** 2) if apply_window else np.ones(win)
    for k in range(nf):
        s = k * hop
        acc[s:s + win] += contrib[k]
        wsq[s:s + win] += wcontrib
    out = np.where(wsq > _EPS, acc / np.maximum(wsq, _EPS), 0.0)
    return out[pad_left:pad_left + n]


def process_framed(x: np.ndarray, fn, win: int = WIN, hop: int = HOP,
                   pad: str = "reflect") -> np.ndarray:
    """x 를 프레임 단위로 fn 에 통과시키고 OLA 로 되붙인다.

    fn : (n_frames, win) -> (n_frames, win)   (배치 처리; DL 래퍼가 사용)
    주의: 창은 여기서 적용하지 않고(analysis 미적용) fn 에 raw 프레임을 준 뒤
          합성 단계에서 w^2 로 정규화한다 — 비선형 처리에서 창 왜곡을 피하기 위함.
    """
    frames, pl, _ = frame(x, win, hop, pad, apply_window=False)
    out = np.asarray(fn(frames), dtype=np.float64)
    if out.shape != frames.shape:
        raise ValueError(f"fn must preserve shape, got {out.shape} vs {frames.shape}")
    # 합성 시에만 w^2 가중 평균 -> 경계 불연속 제거
    nf, w_ = out.shape
    total = (nf - 1) * hop + w_
    w2 = analysis_window(w_) ** 2
    acc = np.zeros(total); wsq = np.zeros(total)
    for k in range(nf):
        s = k * hop
        acc[s:s + w_] += out[k] * w2
        wsq[s:s + w_] += w2
    y = np.where(wsq > _EPS, acc / np.maximum(wsq, _EPS), 0.0)
    return y[pl:pl + len(np.asarray(x).ravel())]
