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


REPO_ROOT = Path(__file__).resolve().parent.parent


def source_digest(paths: Iterable[str | os.PathLike]) -> dict[str, str]:
    """생성 코드의 내용 해시. **git hash 로는 부족하다.**

    산출물에 커밋 해시만 적어두면 "그 뒤로 생성 경로가 바뀌었는가" 를 알 수
    없다. 문서 커밋 하나에도 해시는 바뀌고, 반대로 커밋하지 않은 수정은
    해시에 나타나지 않는다. 실제로 O-11 이 그렇게 일어났다 — floor 측정
    코드를 두 번 고쳤는데 D0 산출물만 재생성되지 않았고, 산출물의 커밋
    해시만 봐서는 그 사실이 드러나지 않았다.

    내용 해시를 적어두면 재생성 없이도 낡음을 판정할 수 있다.
    """
    import hashlib
    out: dict[str, str] = {}
    for p in paths:
        f = Path(p)
        if not f.is_absolute():
            f = REPO_ROOT / f
        if not f.exists():
            out[str(Path(p))] = "missing"
            continue
        out[str(Path(p))] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    return out


def save_manifest(out_dir: str | os.PathLike, cfg: dict[str, Any] | None = None,
                  extra: dict[str, Any] | None = None,
                  sources: Iterable[str | os.PathLike] | None = None) -> Path:
    """결과 디렉토리에 재현 정보를 남긴다. 결과 생성 스크립트는 반드시 호출.

    `sources` 에 이 산출물을 만든 코드 파일을 주면 내용 해시를 함께 남긴다.
    `scripts/check_freshness.py` 가 그것으로 낡은 산출물을 찾는다.
    """
    d = ensure_dir(out_dir)
    man: dict[str, Any] = {"env": env_info()}
    if cfg is not None:
        man["config"] = cfg
    if sources is not None:
        man["sources"] = source_digest(sources)
    if extra:
        man.update(extra)
    p = d / "manifest.json"
    p.write_text(json.dumps(man, indent=2, ensure_ascii=False, default=str))
    return p


def stale_sources(manifest: str | os.PathLike) -> list[str]:
    """이 산출물을 만든 코드가 그 뒤로 바뀌었는가. 바뀐 파일 목록을 준다.

    빈 목록 = 최신. `sources` 가 기록돼 있지 않으면 판정할 수 없으므로
    `["(sources 미기록)"]` 를 준다 — 모른다는 것과 최신이라는 것은 다르다.
    """
    m = json.loads(Path(manifest).read_text())
    rec = m.get("sources")
    if not rec:
        return ["(sources 미기록)"]
    now = source_digest(rec.keys())
    return sorted(k for k, v in rec.items() if now.get(k) != v)


def archive_run(out_dir: str | os.PathLike, *, keep: int = 20) -> Path:
    """현재 산출물을 `{out_dir}/runs/{git}-{시각}/` 으로 복사해 보존한다.

    **읽는 경로는 그대로 둔다.** `out_dir` 최상단이 항상 최신이므로 이 함수를
    붙여도 소비자 코드는 바뀌지 않는다. 보존은 덤이다.

    O-10 이 이 함수가 없어서 일어났다 — 재실행이 SWT 탐색 24 조합의 개별
    값을 덮어써서, 나중에 "그때 정말 전부 음수였는가" 를 확인할 수 없었다.
    """
    import shutil
    from datetime import datetime, timezone

    d = Path(out_dir)
    git = env_info().get("git", "nogit")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # 이름이 초 단위라 같은 초에 두 번 부르면 충돌한다. 덮어쓰면 이 함수가
    # 막으려던 O-10 이 그대로 재현되므로, 충돌하면 접미사를 붙인다.
    base = d / "runs" / f"{git}-{ts}"
    dst = base
    n = 2
    while dst.exists():
        dst = base.with_name(f"{base.name}-{n}")
        n += 1
    dst = ensure_dir(dst)
    for f in sorted(d.iterdir()):
        if f.is_file():
            shutil.copy2(f, dst / f.name)

    runs = sorted((d / "runs").iterdir(), key=lambda x: (x.stat().st_mtime, x.name))
    for old_run in runs[:-keep] if len(runs) > keep else []:
        shutil.rmtree(old_run, ignore_errors=True)
    return dst


def as_float64(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(a, dtype=np.float64).ravel() for a in arrays)


def check_same_length(*arrays: Iterable) -> None:
    lens = {len(np.asarray(a).ravel()) for a in arrays}
    if len(lens) != 1:
        raise ValueError(f"length mismatch: {sorted(lens)}")
