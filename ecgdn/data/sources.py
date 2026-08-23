"""clean ECG 공급자 추상화.

MIT-BIH 가 없는 환경(원격 세션 등)에서도 **전체 파이프라인이 동작**하도록,
합성 ECG 소스를 같은 인터페이스로 제공한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from ..config import FS
from .splits import MITDB_SPLIT

__all__ = ["CleanRecord", "CleanSource", "SyntheticSource", "MITDBSource",
           "get_source"]


@dataclass
class CleanRecord:
    name: str
    x: np.ndarray
    fs: float
    r_peaks: np.ndarray
    symbols: np.ndarray


class CleanSource(Protocol):
    kind: str

    def records(self, split: str) -> Sequence[str]: ...
    def get(self, name: str) -> CleanRecord: ...


class SyntheticSource:
    """합성 ECG를 '기록' 처럼 제공한다. seed 가 곧 환자 ID 역할을 한다.

    PVC 를 섞어 beat-type 층화(EXP-E P3) 파이프라인도 데이터 없이 시험할 수 있다.
    """

    kind = "synthetic"

    def __init__(self, n_train: int = 18, n_val: int = 4, n_test: int = 22,
                 dur_s: float = 300.0, fs: float = FS, pvc_prob: float = 0.06,
                 kernel_jitter: bool = True):
        self.dur_s, self.fs, self.pvc_prob = float(dur_s), float(fs), float(pvc_prob)
        # **기록마다 다른 morphology 를 준다.** 끄면 모든 기록이 같은 파형이 되어
        # 신경망이 생성기를 외우고 성능이 10 dB 이상 과대평가된다 (synthetic.jitter_kernel 참조).
        self.kernel_jitter = bool(kernel_jitter)
        n = {"train": n_train, "val": n_val, "test": n_test}
        i = 0
        self._split: dict[str, list[str]] = {}
        for k in ("train", "val", "test"):
            self._split[k] = [f"S{j:03d}" for j in range(i, i + n[k])]
            i += n[k]
        self._split["paced"] = []
        self._cache: dict[str, CleanRecord] = {}

    def records(self, split: str) -> Sequence[str]:
        return tuple(self._split[split])

    def get(self, name: str) -> CleanRecord:
        if name in self._cache:
            return self._cache[name]
        from .synthetic import synth_ecg

        from .synthetic import jitter_kernel
        from ..config import DEFAULT_KERNEL, PVC_KERNEL
        from ..utils import rng as make_rng

        sid = int(name[1:])
        if self.kernel_jitter:
            g = make_rng("subject", sid)
            kern = jitter_kernel(DEFAULT_KERNEL, g)
            pvc = jitter_kernel(PVC_KERNEL, g, theta_sd=0.10, amp_sd=0.20,
                                width_sd=0.20, t_invert_p=0.0)
        else:
            kern, pvc = DEFAULT_KERNEL, PVC_KERNEL
        s = synth_ecg(self.dur_s, fs=self.fs,
                      hr_bpm=55.0 + (sid * 7) % 45,
                      hrv_std=0.02 + 0.03 * ((sid * 13) % 5) / 4.0,
                      kernel=kern, pvc_kernel=pvc,
                      pvc_prob=self.pvc_prob, seed=1000 + sid)
        sym = np.asarray(s.beat_labels[:len(s.r_peaks)])
        rec = CleanRecord(name=name, x=s.x, fs=s.fs, r_peaks=s.r_peaks, symbols=sym)
        self._cache[name] = rec
        return rec


class MITDBSource:
    kind = "mitdb"

    def __init__(self, root: str | Path = "data/raw/mitdb", fs: float = FS,
                 lead: str = "MLII"):
        self.root, self.fs, self.lead = Path(root), float(fs), lead
        self._cache: dict[str, CleanRecord] = {}

    def records(self, split: str) -> Sequence[str]:
        return MITDB_SPLIT[split]

    def get(self, name: str) -> CleanRecord:
        if name in self._cache:
            return self._cache[name]
        from .mitdb import load_record

        r = load_record(name, self.root, lead=self.lead, fs_out=self.fs)
        rec = CleanRecord(name=r.name, x=r.x, fs=r.fs, r_peaks=r.r_peaks,
                          symbols=r.symbols)
        self._cache[name] = rec
        return rec


def get_source(kind: str = "auto", **kw) -> CleanSource:
    """'auto' 면 MIT-BIH 가 있으면 그것을, 없으면 합성을 쓴다."""
    if kind in ("mitdb", "auto"):
        root = Path(kw.pop("root", "data/raw/mitdb"))
        if root.exists() and any(root.glob("*.hea")):
            return MITDBSource(root=root, **{k: v for k, v in kw.items()
                                             if k in ("fs", "lead")})
        if kind == "mitdb":
            raise FileNotFoundError(
                f"MIT-BIH not found at {root}. run: python scripts/download_data.py --db mitdb")
    return SyntheticSource(**{k: v for k, v in kw.items()
                              if k in ("n_train", "n_val", "n_test", "dur_s", "fs",
                                       "pvc_prob")})
