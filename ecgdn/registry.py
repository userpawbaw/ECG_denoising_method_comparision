"""방법(denoiser) 레지스트리. 실험 스크립트는 문자열 ID 만 알면 된다."""
from __future__ import annotations

from typing import Callable, Any

_REGISTRY: dict[str, Callable[..., Any]] = {}
_META: dict[str, dict[str, Any]] = {}


def register_method(method_id: str, *, family: str = "", label: str = "",
                    needs_clean: bool = False, needs_ckpt: bool = False):
    """denoiser 팩토리를 등록한다.

    needs_clean=True 인 것(oracle bound)은 ID 가 반드시 'B'로 시작하거나
    name 이 'oracle_' 접두사를 가져야 한다. (docs/00_review.md B-5)
    """
    def deco(fn):
        if method_id in _REGISTRY:
            raise KeyError(f"duplicate method id: {method_id}")
        _REGISTRY[method_id] = fn
        _META[method_id] = dict(family=family, label=label or method_id,
                                needs_clean=needs_clean, needs_ckpt=needs_ckpt)
        return fn
    return deco


def build(method_id: str, **kwargs):
    if method_id not in _REGISTRY:
        raise KeyError(f"unknown method '{method_id}'. available: {sorted(_REGISTRY)}")
    return _REGISTRY[method_id](**kwargs)


def meta(method_id: str) -> dict[str, Any]:
    return dict(_META[method_id])


def available() -> list[str]:
    return sorted(_REGISTRY)
