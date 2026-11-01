import re

from .cnn_transformer import CNNTransformer  # noqa: F401
from .dilated_resnet1d import DilatedResNet1D  # noqa: F401
from .blocks import ConvBlock, Down, ResBlock, Up, receptive_field  # noqa: F401
from .losses import DenoiseLoss, make_loss  # noqa: F401
from .resunet1d import ResUNet1D  # noqa: F401
from .swt_torch import TorchISWT, TorchSWT  # noqa: F401
from .wavelet_unet import WaveletSubbandUNet  # noqa: F401

MODELS = {
    "resunet1d": ResUNet1D,
    "wavelet_unet": WaveletSubbandUNet,
    "cnn_transformer": CNNTransformer,
    "dilated_resnet1d": DilatedResNet1D,
}


# n_blocks(D-27) 이전 체크포인트의 인코더 키 — `enc.{i}.c1` 이 지금은 `enc.{i}.0.c1` 이다.
# `ResUNet1D._stack` 이 레벨마다 `nn.Sequential` 로 감싸면서 바뀌었고, 주 실험의
# 체크포인트(`results/{d0,d1}/*/best.pt`)는 전부 그 전에 학습됐다 (O-38).
_LEGACY_ENC = re.compile(r"(^|\.)enc\.(\d+)\.(?=[cn][12]\.)")


def upgrade_state_dict(model, state: dict) -> dict:
    """옛 키 배치의 state_dict 를 **지금 모델의 키로** 옮긴다.

    옮긴 결과가 모델의 키 집합과 **정확히 같을 때만** 옮긴 것을 돌려준다 —
    부분적으로 맞는 재매핑은 다른 모델을 조용히 여는 것이므로 하지 않고,
    그 경우 원본을 그대로 돌려 `load_state_dict` 가 원래 오류를 내게 한다.
    """
    want = set(model.state_dict())
    if set(state) == want:
        return state
    moved = {_LEGACY_ENC.sub(r"\1enc.\2.0.", k): v for k, v in state.items()}
    return moved if set(moved) == want else state


def build_model(name: str, **kw):
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](**kw)
