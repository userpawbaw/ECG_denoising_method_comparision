"""모델 요약 (params / RF / MACs / CPU latency / RTF) — EXP-G 의 기초."""
import _bootstrap  # noqa: F401

import time

import numpy as np
import torch

from ecgdn.config import FS, WIN


def macs_conv1d(model, n=WIN):
    """conv1d 기준 MAC 수 (근사). forward hook 으로 실제 출력 길이를 사용."""
    total = [0]
    hooks = []

    def hook(mod, inp, out):
        if isinstance(mod, torch.nn.Conv1d):
            cin = mod.in_channels // mod.groups
            total[0] += cin * mod.out_channels * mod.kernel_size[0] * out.shape[-1]

    for m in model.modules():
        hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        model(torch.randn(1, model.in_ch if hasattr(model, "in_ch") else 1, n))
    for h in hooks:
        h.remove()
    return total[0]


def latency(model, n=WIN, reps=50, warmup=10):
    torch.set_num_threads(1)
    x = torch.randn(1, 1, n)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        t = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model(x)
            t.append(time.perf_counter() - t0)
    return float(np.median(t))


def main() -> int:
    from ecgdn.models.cnn_transformer import CNNTransformer
    from ecgdn.models.resunet1d import ResUNet1D
    from ecgdn.models.wavelet_unet import WaveletSubbandUNet

    rows = []
    for name, m in [("M06 ResUNet1D", ResUNet1D()),
                    ("M08 WaveletSubbandUNet", WaveletSubbandUNet()),
                    ("M09 CNNTransformer", CNNTransformer())]:
        m.eval()
        lat = latency(m)
        rows.append((name, m.n_params(), getattr(m, "receptive_field_samples", -1),
                     macs_conv1d(m), lat, lat / (WIN / FS)))
    hdr = f"{'model':26s} {'params':>10s} {'RF[smp]':>8s} {'MACs':>12s} {'lat[ms]':>8s} {'RTF':>7s}"
    print(hdr); print("-" * len(hdr))
    for n, p, rf, mc, lat, rtf in rows:
        print(f"{n:26s} {p:10,d} {rf:8d} {mc:12,d} {lat*1e3:8.2f} {rtf:7.4f}")
    print("\nRTF < 1 이면 단일 CPU 스레드로 실시간 처리 가능.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
