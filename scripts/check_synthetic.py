"""STEP 02 DoD 검증: 합성 ECG 생성기.

1) xqrs_detect 로 찾은 R-peak 가 참 R-peak 와 +-20 ms 이내로 일치 (>= 99%)
2) HR 이 지정값과 +-1 bpm 이내
3) ODE 적분과 phase-domain 평가가 일치
4) 파형 그림 저장
"""
import _bootstrap  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt
from wfdb.processing import xqrs_detect

from ecgdn.config import DEFAULT_KERNEL, FS
from ecgdn.data.synthetic import synth_ecg, synth_ecg_ode, ecg_from_phase, wrap_to_pi
from ecgdn.utils import ensure_dir


def main() -> int:
    ok = True
    print("=" * 70)
    print("STEP 02 DoD — synthetic ECG generator")
    print("=" * 70)

    # ---- (3) ODE vs phase-domain
    hr = 70.0
    z, t = synth_ecg_ode(6.0, fs=FS, hr_bpm=hr)
    th = wrap_to_pi(2 * np.pi * hr / 60.0 * t - np.pi)
    err = float(np.max(np.abs(z - ecg_from_phase(th, DEFAULT_KERNEL))))
    p3 = err < 1e-6
    ok &= p3
    print(f"[{'PASS' if p3 else 'FAIL'}] ODE vs phase-domain  max|err| = {err:.3e}  (< 1e-6)")

    # ---- (1)(2) 여러 HR 에서 R-peak / HR 검증
    for hr_bpm in (50.0, 70.0, 100.0):
        s = synth_ecg(120.0, fs=FS, hr_bpm=hr_bpm, seed=int(hr_bpm))
        det = xqrs_detect(s.x, fs=int(s.fs), verbose=False)

        tol = int(round(0.020 * s.fs))            # +-20 ms
        # 참 R-peak 각각에 대해 tol 이내의 검출이 있는지
        hit = 0
        for r in s.r_peaks:
            if len(det) and np.min(np.abs(det - r)) <= tol:
                hit += 1
        rate = hit / max(len(s.r_peaks), 1)
        p1 = rate >= 0.99
        hr_err = abs(s.hr_bpm - hr_bpm)
        p2 = hr_err <= 1.0
        ok &= p1 and p2
        print(f"[{'PASS' if p1 else 'FAIL'}] HR={hr_bpm:5.1f}  R-peak match {rate*100:6.2f}% "
              f"({hit}/{len(s.r_peaks)}, det={len(det)})   "
              f"[{'PASS' if p2 else 'FAIL'}] HR err {hr_err:.3f} bpm")

    # ---- (4) 그림
    s = synth_ecg(10.0, fs=FS, hr_bpm=70.0, pvc_prob=0.15, seed=7)
    d = ensure_dir("results/fig")
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(s.t, s.x, lw=0.9, color="#1f77b4")
    ax[0].plot(s.t[s.r_peaks], s.x[s.r_peaks], "v", ms=6, color="#d62728", label="true R")
    for i, lab in enumerate(s.beat_labels):
        if lab == "V" and i < len(s.r_peaks):
            ax[0].axvspan(s.t[s.r_peaks[i]] - 0.25, s.t[s.r_peaks[i]] + 0.35,
                          color="#ff7f0e", alpha=0.15)
    ax[0].set_ylabel("mV"); ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title("Synthetic ECG (McSharry/Sameni), orange = PVC beat")
    ax[1].plot(s.t, s.theta, lw=0.8, color="#2ca02c")
    ax[1].set_ylabel("phase [rad]"); ax[1].set_xlabel("time [s]")
    fig.tight_layout(); fig.savefig(d / "synth_check.png", dpi=130); plt.close(fig)
    print(f"[INFO] figure -> {d/'synth_check.png'}")

    print("=" * 70)
    print("RESULT:", "ALL PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
