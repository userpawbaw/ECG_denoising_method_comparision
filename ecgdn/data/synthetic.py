"""합성 ECG 생성기 (McSharry ODE / Sameni 파라미터화).

docs/02_procedure.md STEP 02.

두 가지 경로를 제공한다.
  1) `synth_ecg`      : phase-domain 직접 평가. 빠르고 R-peak 위치가 정확히 알려짐. (기본)
  2) `synth_ecg_ode`  : 실제 ODE 적분. (1) 과 수치적으로 일치함을 검증하는 용도.

두 경로가 일치하는 이유:
    zdot = dz/dtheta * thetadot,  thetadot = w
    dz/dtheta = -sum_i (alpha_i / b_i^2) * dtheta_i * exp(-dtheta_i^2/(2 b_i^2))
  이므로 ODE 의 해는 정확히 z(theta) = sum_i alpha_i exp(-dtheta_i^2/(2 b_i^2)) 이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..config import DEFAULT_KERNEL, ECGKernel, FS, PVC_KERNEL
from ..utils import rng as make_rng

__all__ = ["SynthECG", "synth_ecg", "synth_ecg_ode", "wrap_to_pi", "ecg_from_phase",
           "kernel_phase_mean", "jitter_kernel"]


def wrap_to_pi(x: np.ndarray | float) -> np.ndarray | float:
    """[-pi, pi) 로 wrap. 이 wrap 을 틀리면 파형이 깨진다. (STEP 02 사양)"""
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class SynthECG:
    x: np.ndarray                 # (N,) 신호 [mV]
    t: np.ndarray                 # (N,) 시간 [s]
    fs: float
    r_peaks: np.ndarray           # (K,) 참 R-peak 샘플 인덱스
    theta: np.ndarray             # (N,) 위상 [-pi, pi)
    rr: np.ndarray                # (K,) 각 beat 의 RR 간격 [s]
    beat_labels: list[str]        # (K,) 'N' | 'V'
    kernels: list[ECGKernel]      # (K,) beat 별로 실제 사용된 kernel
    params: dict = field(default_factory=dict)

    @property
    def hr_bpm(self) -> float:
        return float(60.0 / np.median(self.rr)) if len(self.rr) else float("nan")


def kernel_phase_mean(kernel: ECGKernel) -> float:
    """커널의 위상 평균  (1/2pi) * integral_{-pi}^{pi} z(theta) dtheta.

    가우시안이 wrap 되지 않을 만큼 b 가 작다고 보면  sum_i alpha_i * b_i / sqrt(2pi).
    """
    return float(sum(a * b for a, b in zip(kernel.alpha, kernel.b)) / np.sqrt(2.0 * np.pi))


def ecg_from_phase(theta: np.ndarray, kernel: ECGKernel,
                   zero_mean: bool = True) -> np.ndarray:
    """z(theta) = sum_i alpha_i exp(-wrap(theta-theta_i)^2 / (2 b_i^2)).

    zero_mean=True (기본): 커널의 **위상 평균을 제거**한다.

    왜 필요한가
        beat 종류(정상/PVC)마다 위상 평균이 다르면, PVC 가 끼어들 때마다 신호의
        국소 평균이 계단처럼 튄다. 이는 생리학적 사실이 아니라 커널 파라미터화의
        부작용이고, 0.5 Hz HPF 가 그 저주파 성분을 제거하면서 '왜곡' 으로 잡힌다.
        (실측: PVC 6% 만 섞여도 front-end 의 distortion floor 가 40 dB -> 13 dB 로 붕괴)
        모든 beat 종류가 같은 등전위선을 공유하도록 만드는 것이 옳다.
    """
    th = np.asarray(theta, dtype=np.float64)
    z = np.zeros_like(th)
    for a, b, t0 in zip(kernel.alpha, kernel.b, kernel.theta):
        d = wrap_to_pi(th - t0)
        z += a * np.exp(-(d ** 2) / (2.0 * b ** 2))
    if zero_mean:
        z = z - kernel_phase_mean(kernel)
    return z


def jitter_kernel(kernel: ECGKernel, gen: np.random.Generator,
                  theta_sd: float = 0.09, amp_sd: float = 0.22, width_sd: float = 0.22,
                  t_invert_p: float = 0.12) -> ECGKernel:
    """기록(=피험자)마다 다른 ECG morphology 를 만든다.

    **왜 필요한가 (실측으로 확인된 함정)**
        모든 합성 기록이 같은 커널을 쓰면, 1 M 파라미터 신경망은 그 생성기를 통째로
        외워버린다. 그러면 record 단위로 split 을 해도 morphology 는 train/test 가
        완전히 동일하므로 **leakage 와 같은 효과**가 난다.
        실측: 학습 morphology 에서 +16.5 dB 였던 모델이 커널을 교란하자 +4~6 dB 로 무너졌다
        (같은 조건에서 SWT 는 +7~12 dB 로 오히려 더 좋았다).
        즉 합성 벤치마크가 DL 을 12 dB 가량 과대평가하고 있었다.

    사람 간 변이를 모사한다: 파형 위치(theta), 진폭(alpha), 폭(b) 을 흔들고,
    낮은 확률로 T 파를 역전시킨다(실제로 흔한 변이).
    """
    a = np.asarray(kernel.alpha, dtype=np.float64).copy()
    b = np.asarray(kernel.b, dtype=np.float64).copy()
    t = np.asarray(kernel.theta, dtype=np.float64).copy()

    t = t + gen.normal(0.0, theta_sd, t.size)
    a = a * (1.0 + gen.normal(0.0, amp_sd, a.size))
    b = np.clip(b * (1.0 + gen.normal(0.0, width_sd, b.size)), 0.02, 1.0)
    if a.size >= 5 and gen.random() < t_invert_p:
        a[4] = -abs(a[4])                      # T 파 역전
    # R 파는 항상 지배적이어야 ECG 로 보인다
    if a.size >= 3:
        a[2] = float(np.sign(a[2]) or 1.0) * max(abs(a[2]), 0.5)
    return ECGKernel(name=f"{kernel.name}_jit", theta=tuple(t), alpha=tuple(a), b=tuple(b))


def _make_rr(n_beats: int, hr_bpm: float, hrv_std: float, resp_hz: float,
             resp_amp: float, gen: np.random.Generator) -> np.ndarray:
    """RR 간격 시퀀스. 백색 변동 + 호흡성 부정맥(RSA)."""
    rr0 = 60.0 / hr_bpm
    k = np.arange(n_beats)
    rsa = resp_amp * np.sin(2.0 * np.pi * resp_hz * np.cumsum(np.full(n_beats, rr0)))
    jitter = hrv_std * gen.standard_normal(n_beats)
    rr = rr0 * (1.0 + jitter + rsa)
    return np.clip(rr, 0.3, 2.5)


def synth_ecg(
    duration_s: float = 60.0,
    fs: float = FS,
    hr_bpm: float = 70.0,
    hrv_std: float = 0.03,
    resp_hz: float = 0.25,
    resp_amp: float = 0.03,
    kernel: ECGKernel = DEFAULT_KERNEL,
    amp_jitter: float = 0.02,
    pvc_prob: float = 0.0,
    pvc_kernel: ECGKernel = PVC_KERNEL,
    seed: object = 0,
) -> SynthECG:
    """합성 ECG 를 만든다.

    Parameters
    ----------
    pvc_prob : beat 하나가 PVC(심실조기수축) 로 대체될 확률.
        안전성 프로브(EXP-E) 파이프라인을 MIT-BIH 없이 시험하기 위한 것.

    Returns
    -------
    SynthECG : x, 참 R-peak 인덱스, 위상, 사용된 파라미터를 모두 포함.
        참 파라미터를 반환하는 것이 핵심 — Sameni EKF 의 정답 검증에 쓴다.
    """
    gen = make_rng("synth", seed, duration_s, hr_bpm)

    n_beats = int(np.ceil(duration_s * hr_bpm / 60.0)) + 4
    rr = _make_rr(n_beats, hr_bpm, hrv_std, resp_hz, resp_amp, gen)

    # R-peak 시각. 첫 R 은 한 beat 뒤에 두어 앞쪽 P 파가 잘리지 않게 한다.
    r_times = np.concatenate([[rr[0]], rr[0] + np.cumsum(rr[:-1])])
    keep = r_times < duration_s
    r_times, rr = r_times[keep], rr[keep]
    if len(r_times) < 3:
        raise ValueError("duration 이 너무 짧다 (beat < 3)")

    n = int(round(duration_s * fs))
    t = np.arange(n) / fs

    # ---- 위상 theta(t): 각 RR 구간에서 0 -> 2pi 선형, 이후 [-pi,pi) 로 wrap
    theta = np.zeros(n, dtype=np.float64)
    x = np.zeros(n, dtype=np.float64)
    beat_labels: list[str] = []
    kernels: list[ECGKernel] = []

    # 구간 경계: 각 R 에서 다음 R 까지. 앞뒤 여유 구간은 가장 가까운 beat 로 외삽.
    edges = np.concatenate([[r_times[0] - rr[0]], r_times, [r_times[-1] + rr[-1]]])
    for bi in range(len(edges) - 1):
        t0, t1 = edges[bi], edges[bi + 1]
        i0, i1 = int(np.ceil(max(t0, 0.0) * fs)), int(np.ceil(min(t1, duration_s) * fs))
        if i1 <= i0:
            continue
        seg_t = t[i0:i1]
        # 이 구간의 시작 R 이 theta=0 (첫 가상 구간은 시작이 R 이 아니므로 그대로 두어도
        # wrap 후 P 파 위치가 맞는다)
        th = 2.0 * np.pi * (seg_t - t0) / (t1 - t0)
        theta[i0:i1] = wrap_to_pi(th)

    # ---- beat 별 kernel 적용 (PVC 는 kernel 교체)
    for bi, rt in enumerate(r_times):
        is_pvc = gen.random() < pvc_prob and bi > 0
        k = pvc_kernel if is_pvc else kernel
        beat_labels.append("V" if is_pvc else "N")
        kernels.append(k)

    # beat 소속 구간마다 해당 kernel 로 z 를 평가
    # (구간 = [이전 R 의 중간지점, 다음 R 의 중간지점] 이 아니라
    #  [R_k - RR_k, R_k + RR_{k+1}) 중 R_k 가 theta=0 인 구간)
    bounds = np.concatenate([[0.0], r_times[:-1] + np.diff(r_times) * 0.65, [duration_s]])
    for bi, k in enumerate(kernels):
        i0 = int(np.ceil(bounds[bi] * fs))
        i1 = int(np.ceil(bounds[bi + 1] * fs))
        i0, i1 = max(i0, 0), min(i1, n)
        if i1 <= i0:
            continue
        amp = 1.0 + amp_jitter * gen.standard_normal()
        x[i0:i1] = amp * ecg_from_phase(theta[i0:i1], k)

    r_peaks = np.round(r_times * fs).astype(int)
    r_peaks = r_peaks[(r_peaks >= 0) & (r_peaks < n)]

    return SynthECG(
        x=x.astype(np.float64), t=t, fs=float(fs), r_peaks=r_peaks, theta=theta,
        rr=rr, beat_labels=beat_labels, kernels=kernels,
        params=dict(hr_bpm=hr_bpm, hrv_std=hrv_std, kernel=kernel, seed=seed,
                    pvc_prob=pvc_prob),
    )


def synth_ecg_ode(duration_s: float = 10.0, fs: float = FS, hr_bpm: float = 70.0,
                  kernel: ECGKernel = DEFAULT_KERNEL) -> tuple[np.ndarray, np.ndarray]:
    """ODE 를 실제로 적분한다 (검증 전용, HRV 없음).

    d(theta)/dt = w
    dz/dt = -sum_i (alpha_i * w / b_i^2) * dtheta_i * exp(-dtheta_i^2 / (2 b_i^2))
    """
    from scipy.integrate import solve_ivp

    w = 2.0 * np.pi * hr_bpm / 60.0
    a = np.asarray(kernel.alpha, dtype=np.float64)
    b = np.asarray(kernel.b, dtype=np.float64)
    th0 = np.asarray(kernel.theta, dtype=np.float64)

    def f(_t, s):
        th, z = s
        d = wrap_to_pi(th - th0)
        dz = -np.sum((a * w / b ** 2) * d * np.exp(-(d ** 2) / (2 * b ** 2)))
        return [w, dz]

    t_eval = np.arange(int(round(duration_s * fs))) / fs
    # theta(0) = -pi 에서 시작하고, z(0) 은 그 위상에서의 참값
    z0 = float(ecg_from_phase(np.array([-np.pi]), kernel)[0])
    sol = solve_ivp(f, (0.0, duration_s), [-np.pi, z0], t_eval=t_eval,
                    max_step=1.0 / (8 * fs), rtol=1e-8, atol=1e-10)
    return sol.y[1], t_eval
