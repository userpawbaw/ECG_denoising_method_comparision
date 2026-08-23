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
           "get_source", "resolve_source_kind", "source_tag", "SOURCE_TAG",
           "CleanSegment", "real_clean_segments"]

# 산출물 경로에 붙일 데이터축 태그. D0 와 D1 의 결과가 같은 경로를 쓰면
# 서로를 덮어쓰고, 표에서 어느 쪽 숫자인지 구분할 수 없게 된다.
SOURCE_TAG = {"synthetic": "d0", "mitdb": "d1"}


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


def resolve_source_kind(kind: str = "auto", root: str | Path = "data/raw/mitdb") -> str:
    """`auto` 가 **실제로 무엇을 고르는지** 를 데이터 적재 전에 확정한다.

    `auto` 는 편리하지만 재현성을 해친다 — `data/raw/mitdb` 에 파일이 생기는
    순간, 같은 커밋에 같은 config 로 같은 명령을 돌려도 D0 대신 D1 이 학습된다.
    (실제로 그렇게 됐다: docs/99_status.md 2.1)

    그래서 실행 전에 이 함수로 확정하고, 그 값을 산출물 경로와 manifest 에
    함께 남긴다. `auto` 자체를 금지하지는 않는다 — 대신 결과가 섞이지 않게 한다.
    """
    if kind == "auto":
        r = Path(root)
        return "mitdb" if (r.exists() and any(r.glob("*.hea"))) else "synthetic"
    if kind not in SOURCE_TAG:
        raise KeyError(f"unknown source {kind!r}; choose from {sorted(SOURCE_TAG)} or 'auto'")
    return kind


def source_tag(kind: str = "auto", root: str | Path = "data/raw/mitdb") -> str:
    """산출물 경로에 쓸 태그. synthetic -> d0, mitdb -> d1."""
    return SOURCE_TAG[resolve_source_kind(kind, root)]


@dataclass
class CleanSegment:
    """보조 스크립트가 쓰는 최소 인터페이스 (`synth_ecg` 반환값과 호환).

    `x` 는 **평가 기준이 되는 대역제한 참조**이고 `x_raw` 는 원본이다.
    잡음은 `x_raw` 에 섞고 평가는 `x` 기준으로 한다 — MIT-BIH 원본을 정답으로
    두면 front-end 를 쓰는 방법이 전부 부당하게 진다 (F-12).
    합성 소스에서는 둘이 사실상 같다(front-end 가 53 dB 무해).
    """
    x: np.ndarray
    fs: float
    r_peaks: np.ndarray
    beat_labels: np.ndarray
    name: str = ""
    x_raw: np.ndarray | None = None

    def __post_init__(self):
        if self.x_raw is None:
            self.x_raw = self.x


def real_clean_segments(n: int, dur_s: float, *, fs: float = FS,
                        split: str = "train", root: str | Path = "data/raw/mitdb",
                        offset_s: float = 0.0,
                        ref_frontend: bool = True) -> list[CleanSegment]:
    """MIT-BIH 에서 clean 구간 `n` 개를 뽑는다 (보조 스크립트용).

    **TRAIN split 이 기본이다.** 이 함수의 소비자는 파라미터를 정하거나 지표의
    분해능을 재는 쪽이라, TEST 를 쓰면 그 선택이 평가에 새어 들어간다
    (docs/01_design.md 3.2 의 leakage 규약).

    MIT-BIH 는 '깨끗한 신호' 가 아니라 실제 기록이다. 그래서 `ref_frontend=True`
    (기본)이면 `x` 에 front-end 를 걸어 **"우리가 복원하려는 0.5~100 Hz 대역"**
    으로 맞추고, 원본은 `x_raw` 에 남긴다. 이렇게 하지 않으면 front-end 가 원본의
    기저선 변동을 지울수록 정답에서 멀어져 SNR 이 떨어진다 (F-12).
    """
    from .mitdb import load_record

    recs = MITDB_SPLIT[split]
    out: list[CleanSegment] = []
    n_want = int(round(dur_s * fs))
    o = int(round(offset_s * fs))
    for name in recs[:n]:
        r = load_record(name, root, fs_out=fs)
        if r.x.size < o + n_want:
            o2 = max(0, r.x.size - n_want)
        else:
            o2 = o
        x_raw = r.x[o2:o2 + n_want]
        if ref_frontend:
            from ..methods.frontend import FrontEnd
            x = FrontEnd()(x_raw, float(r.fs))
        else:
            x = x_raw
        sel = (r.r_peaks >= o2) & (r.r_peaks < o2 + x_raw.size)
        out.append(CleanSegment(x=x, fs=float(r.fs),
                                r_peaks=(r.r_peaks[sel] - o2).astype(np.int64),
                                beat_labels=np.asarray(r.symbols)[sel],
                                name=name, x_raw=x_raw))
    return out
