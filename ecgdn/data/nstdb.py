"""MIT-BIH Noise Stress Test Database 잡음 적재 (STEP 15).

**시간축 disjoint split 을 강제한다** (docs/00_review.md A-7).
bw/ma/em 은 각 30분짜리 기록 3개뿐이라, 랜덤 crop 을 그냥 하면 train 과 test 가
같은 잡음 파형 구간을 보게 된다 (noise leakage).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import FS, FS_MITDB
from .mitdb import resample_to
from .noise import unit_var
from .splits import NOISE_SPLIT_FRAC

__all__ = ["NoiseBank", "load_noise", "NSTDB_KINDS", "make_banks"]

NSTDB_KINDS = ("bw", "ma", "em")


def load_noise(kind: str, root: str | Path = "data/raw/nstdb",
               fs_out: float = FS, channel: int = 0) -> np.ndarray:
    import wfdb

    if kind not in NSTDB_KINDS:
        raise KeyError(f"unknown noise kind {kind!r}; choose from {NSTDB_KINDS}")
    rec = wfdb.rdrecord(str(Path(root) / kind))
    x = np.asarray(rec.p_signal[:, min(channel, rec.p_signal.shape[1] - 1)],
                   dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return resample_to(x, float(rec.fs), fs_out)


class NoiseBank:
    """한 종류의 실제 잡음에서, 지정된 split 구간 안에서만 crop 을 뽑는다.

    `noise.make_noise` / `mixed_noise` 와 같은 인터페이스(`sample(n, rng)`)를 갖는다.
    """

    def __init__(self, kind: str, split: str, root: str | Path = "data/raw/nstdb",
                 fs_out: float = FS, x: np.ndarray | None = None):
        if split not in NOISE_SPLIT_FRAC:
            raise KeyError(f"unknown split {split!r}")
        self.kind, self.split = kind, split
        full = load_noise(kind, root, fs_out) if x is None else np.asarray(x, float).ravel()
        a, b = NOISE_SPLIT_FRAC[split]
        i0, i1 = int(a * full.size), int(b * full.size)
        self.x = full[i0:i1]
        self.range = (i0, i1)
        if self.x.size < 64:
            raise ValueError(f"noise split too short: {kind}/{split} -> {self.x.size}")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """길이 n 의 crop. 구간보다 길면 타일링한 뒤 자른다."""
        m = self.x.size
        if n <= m:
            s = int(rng.integers(0, m - n + 1))
            return unit_var(self.x[s:s + n])
        reps = int(np.ceil(n / m)) + 1
        s = int(rng.integers(0, m))
        return unit_var(np.tile(self.x, reps)[s:s + n])

    def __repr__(self) -> str:
        return f"<NoiseBank {self.kind}/{self.split} n={self.x.size} range={self.range}>"


def make_banks(split: str, root: str | Path = "data/raw/nstdb",
               fs_out: float = FS, kinds=NSTDB_KINDS) -> dict[str, NoiseBank]:
    """{'bw': NoiseBank, 'ma': ..., 'em': ...}. 파일이 없으면 빈 dict."""
    out: dict[str, NoiseBank] = {}
    for k in kinds:
        try:
            out[k] = NoiseBank(k, split, root, fs_out)
        except Exception:
            pass
    return out
