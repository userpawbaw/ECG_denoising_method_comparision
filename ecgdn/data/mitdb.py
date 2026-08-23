"""MIT-BIH Arrhythmia 적재 · 리샘플 (docs/02_procedure.md STEP 15).

주의
----
  * annotation 인덱스도 **함께** 리샘플해야 한다. 잊으면 beat-type 층화 평가가 전부 어긋난다.
  * 물리단위(mV) 로 변환해서 쓴다 (`p_signal`). 그래야 R-peak 진폭 지표가 해석 가능하다.
  * 리드는 MLII 우선. 102/104 처럼 MLII 가 없는 기록은 첫 채널을 쓰고 그 사실을 기록한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import FS, FS_MITDB, RESAMPLE_DOWN, RESAMPLE_UP

__all__ = ["MITRecord", "load_record", "resample_to", "resample_index",
           "BEAT_SYMBOLS", "NON_BEAT_SYMBOLS", "available_records"]

# WFDB annotation symbol -> 사람이 읽는 이름 (층화 평가용, docs/00_review.md C-4)
BEAT_SYMBOLS = {
    "N": "normal", "L": "LBBB", "R": "RBBB", "e": "atrial_escape", "j": "nodal_escape",
    "A": "APB", "a": "aberrated_APB", "J": "nodal_premature", "S": "supraventricular",
    "V": "PVC", "E": "ventricular_escape",
    "F": "fusion", "/": "paced", "f": "fusion_paced", "Q": "unclassifiable",
}
NON_BEAT_SYMBOLS = set("[!]x()pt~^|s T*D=\"@+")


@dataclass
class MITRecord:
    name: str
    x: np.ndarray                 # (N,) mV, fs 로 리샘플됨
    fs: float
    lead: str
    r_peaks: np.ndarray           # (K,) 리샘플된 인덱스
    symbols: np.ndarray           # (K,) annotation symbol
    fs_orig: float = FS_MITDB
    meta: dict = field(default_factory=dict)

    def beat_mask(self, symbol: str) -> np.ndarray:
        return self.symbols == symbol

    def __repr__(self) -> str:
        return (f"<MITRecord {self.name} lead={self.lead} n={self.x.size} "
                f"fs={self.fs:g} beats={self.r_peaks.size}>")


def resample_to(x: np.ndarray, fs_in: float = FS_MITDB, fs_out: float = FS) -> np.ndarray:
    """360 -> 250 Hz polyphase 리샘플 (25/36)."""
    from scipy.signal import resample_poly

    if abs(fs_in - fs_out) < 1e-9:
        return np.asarray(x, dtype=np.float64).ravel()
    if abs(fs_in - FS_MITDB) < 1e-9 and abs(fs_out - FS) < 1e-9:
        up, down = RESAMPLE_UP, RESAMPLE_DOWN
    else:
        from math import gcd
        g = gcd(int(round(fs_out)), int(round(fs_in)))
        up, down = int(round(fs_out)) // g, int(round(fs_in)) // g
    return np.asarray(resample_poly(np.asarray(x, dtype=np.float64).ravel(), up, down),
                      dtype=np.float64)


def resample_index(idx: np.ndarray, fs_in: float, fs_out: float, n_out: int) -> np.ndarray:
    """annotation 인덱스를 리샘플 격자로 옮긴다. 범위를 벗어난 것은 버린다."""
    i = np.asarray(idx, dtype=np.float64).ravel() * (fs_out / fs_in)
    i = np.round(i).astype(np.int64)
    return i[(i >= 0) & (i < n_out)]


def available_records(root: str | Path = "data/raw/mitdb") -> list[str]:
    p = Path(root)
    return sorted(q.stem for q in p.glob("*.hea")) if p.exists() else []


def load_record(name: str, root: str | Path = "data/raw/mitdb",
                lead: str = "MLII", fs_out: float = FS,
                beats_only: bool = True) -> MITRecord:
    """한 기록을 mV 단위 · fs_out 으로 적재한다."""
    import wfdb

    p = str(Path(root) / str(name))
    rec = wfdb.rdrecord(p)
    ann = wfdb.rdann(p, "atr")

    names = list(rec.sig_name)
    if lead in names:
        ch, used = names.index(lead), lead
    else:
        ch, used = 0, names[0]

    x = np.asarray(rec.p_signal[:, ch], dtype=np.float64)   # 물리단위 mV
    fs_in = float(rec.fs)
    xr = resample_to(x, fs_in, fs_out)

    sym = np.asarray(ann.symbol)
    smp = np.asarray(ann.sample, dtype=np.int64)
    if beats_only:
        keep = np.array([s in BEAT_SYMBOLS for s in sym], dtype=bool)
        sym, smp = sym[keep], smp[keep]

    scale = fs_out / fs_in
    smp_r = np.round(smp * scale).astype(np.int64)
    inb = (smp_r >= 0) & (smp_r < xr.size)

    return MITRecord(name=str(name), x=xr, fs=float(fs_out), lead=used,
                     r_peaks=smp_r[inb], symbols=sym[inb], fs_orig=fs_in,
                     meta=dict(sig_names=names, lead_requested=lead,
                               lead_fallback=(used != lead), n_orig=int(x.size)))
