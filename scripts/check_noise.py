"""STEP 03 DoD 검증: 각 잡음의 PSD 가 의도한 대역에 에너지를 갖는지."""
import _bootstrap  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sps

from ecgdn.config import FS
from ecgdn.data.noise import NOISE_FNS
from ecgdn.utils import ensure_dir, rng

# 각 잡음이 에너지를 몰아야 하는 대역 (Hz) — DoD 판정 기준
EXPECT = {
    "awgn":     None,              # 평탄 (판정 없음)
    "pli":      (58.0, 62.0),
    "bw_synth": (0.02, 0.6),
    "ma_synth": (20.0, 110.0),
    "em_synth": (0.0, 20.0),
    "impulse":  None,              # 광대역 sparse
}


def band_frac(f, p, lo, hi):
    m = (f >= lo) & (f <= hi)
    return float(np.trapezoid(p[m], f[m]) / np.trapezoid(p, f))


def main() -> int:
    ok = True
    n, fs = 60 * int(FS), FS
    g = rng("check_noise")
    print("=" * 70)
    print("STEP 03 DoD — noise models")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, (name, fn) in zip(axes.ravel(), NOISE_FNS.items()):
        x = fn(n, fs, g)
        f, p = sps.welch(x, fs=fs, nperseg=2048)
        ax.semilogy(f, p, lw=0.9)
        ax.set_title(name); ax.set_xlabel("Hz"); ax.grid(alpha=0.3)
        exp = EXPECT[name]
        if exp is None:
            print(f"[ .. ] {name:10s} (no band criterion)")
            continue
        frac = band_frac(f, p, *exp)
        thr = 0.25 if name in ("ma_synth",) else 0.40
        good = frac >= thr
        ok &= good
        ax.axvspan(exp[0], exp[1], color="orange", alpha=0.2)
        print(f"[{'PASS' if good else 'FAIL'}] {name:10s} power in {exp} = {frac*100:5.1f}% "
              f"(>= {thr*100:.0f}%)")

    fig.suptitle(f"Noise PSD (fs={fs:.0f} Hz). NOTE: PLI 3rd harmonic 180 Hz aliases to 70 Hz.")
    fig.tight_layout()
    d = ensure_dir("results/fig")
    fig.savefig(d / "noise_psd.png", dpi=130); plt.close(fig)
    print(f"[INFO] figure -> {d/'noise_psd.png'}")
    print("=" * 70)
    print("RESULT:", "ALL PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
