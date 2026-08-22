from .blocks import ConvBlock, Down, ResBlock, Up, receptive_field  # noqa: F401
from .losses import DenoiseLoss, make_loss  # noqa: F401
from .resunet1d import ResUNet1D  # noqa: F401
from .swt_torch import TorchISWT, TorchSWT  # noqa: F401
from .wavelet_unet import WaveletSubbandUNet  # noqa: F401

MODELS = {
    "resunet1d": ResUNet1D,
    "wavelet_unet": WaveletSubbandUNet,
}


def build_model(name: str, **kw):
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](**kw)
