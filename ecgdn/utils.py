"""공통 유틸: 결정론적 난수, 파워 정의, 매니페스트.

핵심 원칙(docs/02_procedure.md 부록 A):
  - 전역 seed 에 의존하지 않는다. (experiment, record, window) 로부터 seed 를 유도한다.
  - 파워(power)의 정의는 이 파일의 `power()` 하나만 쓴다. SNR 정의의 단일 진실 원천.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np

__all__ = [
    "derive_seed", "rng", "power", "power_db", "robust_scale",
    "git_hash", "env_info", "save_manifest", "ensure_dir",
]


# ------------------------------------------------------------------ 난수
def derive_seed(*parts: Any) -> int:
    """임의 개수의 파트로부터 결정론적 32-bit seed 를 만든다.

    >>> derive_seed("exp_a", 100, 3) == derive_seed("exp_a", 100, 3)
    True
    """
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x00")
    return int.from_bytes(h.digest()[:4], "little")


def rng(*parts: Any) -> np.random.Generator:
    """derive_seed 로 만든 seed 를 쓰는 Generator."""
    return np.random.default_rng(derive_seed(*parts))


# ------------------------------------------------------------------ 신호 파워
def power(x: np.ndarray) -> float:
    """평균 제거 파워. SNR / RMSE 계열 전부가 이 정의를 공유한다.

    DC offset 에 불변이어야 한다. (docs/00_review.md A-4)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    return float(np.mean((x - x.mean()) ** 2))


def power_db(x: np.ndarray) -> float:
    p = power(x)
    return float(10.0 * np.log10(p)) if p > 0 else -np.inf


def robust_scale(y: np.ndarray, q: float = 99.0, eps: float = 1e-9) -> float:
    """정규화 스케일. 이상치(임펄스 artifact)에 강건.

    docs/00_review.md A-6: **noisy 입력에서만** 계산한다. clean 을 보면 정보 누설.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    med = np.median(y)
    s = float(np.percentile(np.abs(y - med), q))
    return s + eps


# ------------------------------------------------------------------ 재현성
def git_hash(short: bool = True) -> str:
    try:
        cmd = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
        if not short:
            cmd = ["git", "rev-parse", "HEAD"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git": git_hash(),
    }
    for mod in ("numpy", "scipy", "pywt", "wfdb", "neurokit2", "torch"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            info[mod] = None
    return info


def ensure_dir(p: str | os.PathLike) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_manifest(out_dir: str | os.PathLike, cfg: dict[str, Any] | None = None,
                  extra: dict[str, Any] | None = None) -> Path:
    """결과 디렉토리에 재현 정보를 남긴다. 결과 생성 스크립트는 반드시 호출."""
    d = ensure_dir(out_dir)
    man: dict[str, Any] = {"env": env_info()}
    if cfg is not None:
        man["config"] = cfg
    if extra:
        man.update(extra)
    p = d / "manifest.json"
    p.write_text(json.dumps(man, indent=2, ensure_ascii=False, default=str))
    return p


def as_float64(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(a, dtype=np.float64).ravel() for a in arrays)


def check_same_length(*arrays: Iterable) -> None:
    lens = {len(np.asarray(a).ravel()) for a in arrays}
    if len(lens) != 1:
        raise ValueError(f"length mismatch: {sorted(lens)}")
