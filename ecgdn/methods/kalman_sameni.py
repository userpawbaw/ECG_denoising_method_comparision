"""M05 — Sameni 계열 model-based Bayesian filtering (EKF / EKS).

docs/02_procedure.md STEP 13, docs/00_review.md A-2.

상태공간 (McSharry/Sameni 파라미터화)
-----------------------------------
    theta_{k+1} = wrap(theta_k + w_k * dt)
    z_{k+1}     = z_k - dt * sum_i (alpha_i * w_k / b_i^2) * dth_i
                        * exp(-dth_i^2 / (2 b_i^2))  + eta_k
    dth_i       = wrap(theta_k - theta_i)

관측
    phi_k = theta_k + u_k     (R-peak 로부터 할당한 위상)
    s_k   = z_k     + v_k     (측정 ECG)

**A-2 의 6가지 실패 원인을 모두 구현 사양에 반영했다**
    #1 위상   : R-peak 기반 선형 위상 할당 + wrap. `diagnose` 로 0-crossing 검증.
    #2 baseline: front-end HPF 를 EKF 앞단에 필수 적용.
    #3 fitting : phase-averaged template 에 Gaussian 커널을 최소자승 적합
                 (논문 기본값을 그대로 쓰지 않는다).
    #4 Q, R    : R 은 등전위(TP) 구간 분산에서, Q 는 모델 잔차에서 추정.
    #5 EKS     : RTS backward smoother 를 구현 (EKF 단독은 개선폭이 작다).
    #6 정규화  : 입력을 R 진폭 기준으로 정규화하고 출력에서 되돌린다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import DEFAULT_KERNEL, ECGKernel
from ..registry import register_method
from .base import BaseDenoiser
from .frontend import FrontEnd

__all__ = ["assign_phase", "phase_average", "fit_kernels", "SameniKalman",
           "KernelFit"]

TWO_PI = 2.0 * np.pi


def _wrap(x):
    return (np.asarray(x) + np.pi) % TWO_PI - np.pi


# --------------------------------------------------------------------- 위상
def assign_phase(r_peaks: np.ndarray, n: int, fs: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """R-peak 로부터 위상 theta(t) 와 각속도 omega(t) 를 만든다.

    규약: **R-peak 에서 theta = 0**, 다음 R-peak 까지 선형으로 2*pi 증가한 뒤 [-pi,pi) 로 wrap.
    (합성 생성기 `data/synthetic.py` 와 동일한 규약 — 그래야 참 파라미터 검증이 성립한다)
    """
    r = np.asarray(r_peaks, dtype=np.int64).ravel()
    theta = np.zeros(n, dtype=np.float64)
    omega = np.zeros(n, dtype=np.float64)
    if r.size < 2:
        omega[:] = TWO_PI * 70.0 / 60.0
        theta[:] = _wrap(np.arange(n) * omega[0] / fs)
        return theta, omega

    rr = np.diff(r).astype(np.float64)
    edges = np.concatenate([[r[0] - rr[0]], r.astype(np.float64), [r[-1] + rr[-1]]])
    for i in range(edges.size - 1):
        t0, t1 = edges[i], edges[i + 1]
        i0, i1 = int(max(np.ceil(t0), 0)), int(min(np.ceil(t1), n))
        if i1 <= i0:
            continue
        span = max(t1 - t0, 1.0)
        idx = np.arange(i0, i1, dtype=np.float64)
        theta[i0:i1] = _wrap(TWO_PI * (idx - t0) / span)
        omega[i0:i1] = TWO_PI / (span / fs)
    return theta, omega


def phase_average(z: np.ndarray, theta: np.ndarray, n_bins: int = 250
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """위상 격자 위의 평균 ECG(template) 와 그 잔차 분산.

    Returns
    -------
    (grid, mean, var) : grid (n_bins,) in [-pi, pi)
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    th = _wrap(theta)
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.clip(np.digitize(th, edges) - 1, 0, n_bins - 1)
    mean = np.zeros(n_bins)
    var = np.zeros(n_bins)
    cnt = np.bincount(idx, minlength=n_bins).astype(np.float64)
    s1 = np.bincount(idx, weights=z, minlength=n_bins)
    s2 = np.bincount(idx, weights=z ** 2, minlength=n_bins)
    ok = cnt > 0
    mean[ok] = s1[ok] / cnt[ok]
    var[ok] = np.maximum(s2[ok] / cnt[ok] - mean[ok] ** 2, 0.0)
    if np.any(~ok):                        # 빈 bin 은 보간
        g = 0.5 * (edges[:-1] + edges[1:])
        mean[~ok] = np.interp(g[~ok], g[ok], mean[ok])
        var[~ok] = np.interp(g[~ok], g[ok], var[ok])
    grid = 0.5 * (edges[:-1] + edges[1:])
    return grid, mean, var


# --------------------------------------------------------------------- 커널 적합
@dataclass
class KernelFit:
    alpha: np.ndarray
    b: np.ndarray
    theta: np.ndarray
    r2: float
    template_grid: np.ndarray
    template: np.ndarray
    fitted: np.ndarray

    def as_kernel(self) -> ECGKernel:
        return ECGKernel(name="fitted", theta=tuple(self.theta),
                         alpha=tuple(self.alpha), b=tuple(self.b))


def _kernel_sum(grid, alpha, b, th0):
    d = _wrap(grid[:, None] - th0[None, :])
    return (alpha[None, :] * np.exp(-(d ** 2) / (2.0 * b[None, :] ** 2))).sum(axis=1)


def fit_kernels(grid: np.ndarray, template: np.ndarray, n_kernels: int = 7,
                init: ECGKernel = DEFAULT_KERNEL) -> KernelFit:
    """phase-averaged template 에 Gaussian 커널 합을 최소자승 적합.

    **논문 기본값을 그대로 쓰지 않고 반드시 여기서 적합한다** (A-2 #3).
    """
    from scipy.optimize import least_squares

    grid = np.asarray(grid, dtype=np.float64)
    tmpl = np.asarray(template, dtype=np.float64)

    a0 = np.asarray(init.alpha, dtype=np.float64)[:n_kernels]
    b0 = np.asarray(init.b, dtype=np.float64)[:n_kernels]
    t0 = np.asarray(init.theta, dtype=np.float64)[:n_kernels]
    if a0.size < n_kernels:                              # 부족하면 균등 배치로 보충
        extra = n_kernels - a0.size
        a0 = np.concatenate([a0, np.full(extra, 0.05)])
        b0 = np.concatenate([b0, np.full(extra, 0.2)])
        t0 = np.concatenate([t0, np.linspace(-2.5, 2.5, extra)])

    # 진폭 초기값을 template 규모에 맞춘다
    scale = np.max(np.abs(tmpl)) / max(np.max(np.abs(a0)), 1e-9)
    a0 = a0 * scale

    p0 = np.concatenate([a0, b0, t0])
    lo = np.concatenate([np.full(n_kernels, -10.0), np.full(n_kernels, 0.01),
                         t0 - 0.6])
    hi = np.concatenate([np.full(n_kernels, 10.0), np.full(n_kernels, 1.2),
                         t0 + 0.6])

    def resid(p):
        a, b, t = p[:n_kernels], p[n_kernels:2 * n_kernels], p[2 * n_kernels:]
        return _kernel_sum(grid, a, b, t) - tmpl

    sol = least_squares(resid, p0, bounds=(lo, hi), method="trf", max_nfev=4000)
    a, b, t = sol.x[:n_kernels], sol.x[n_kernels:2 * n_kernels], sol.x[2 * n_kernels:]
    fitted = _kernel_sum(grid, a, b, t)
    ss_res = float(np.sum((tmpl - fitted) ** 2))
    ss_tot = float(np.sum((tmpl - tmpl.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    return KernelFit(alpha=a, b=b, theta=t, r2=r2, template_grid=grid,
                     template=tmpl, fitted=fitted)


# --------------------------------------------------------------------- EKF / EKS
class SameniKalman(BaseDenoiser):
    """M05 — EKF(forward) 및 EKS(RTS smoother)."""

    def __init__(self, smooth: bool = True, n_kernels: int = 7,
                 use_frontend: bool = True, kernel: ECGKernel | None = None,
                 q_scale: float = 1.0, r_scale: float = 1.0,
                 phase_sigma_ms: float = 12.0, name: str | None = None):
        self.smooth = bool(smooth)
        self.n_kernels = int(n_kernels)
        self.fe = FrontEnd() if use_frontend else None
        self.kernel = kernel                    # 주어지면 적합을 건너뛴다 (정답 검증용)
        self.q_scale = float(q_scale)
        self.r_scale = float(r_scale)
        self.phase_sigma_ms = float(phase_sigma_ms)
        self.name = name or ("M05" if smooth else "M05f")
        self.last_info: dict[str, Any] = {}

    # ---------------- 잡음 공분산 추정 (A-2 #4)
    @staticmethod
    def _estimate_r(z: np.ndarray, r_peaks: np.ndarray,
                    frac: tuple[float, float] = (0.62, 0.85)) -> float:
        """등전위(TP) 구간의 추세 제거 분산 = 측정잡음 분산."""
        r = np.asarray(r_peaks, dtype=int)
        if r.size < 4:
            return float(np.var(np.diff(z)) / 2.0)
        segs = []
        for a, b in zip(r[:-1], r[1:]):
            rr = b - a
            i0, i1 = a + int(rr * frac[0]), a + int(rr * frac[1])
            if i1 - i0 >= 5 and i1 <= z.size:
                s = z[i0:i1]
                t = np.arange(s.size)
                segs.append(float(np.var(s - np.polyval(np.polyfit(t, s, 1), t))))
        if not segs:
            return float(np.var(np.diff(z)) / 2.0)
        return float(max(np.median(segs), 1e-12))

    def _run(self, y: np.ndarray, fs: float, ctx: dict[str, Any]) -> np.ndarray:
        x = self.fe(y, fs) if self.fe is not None else y      # A-2 #2

        # --- R-peak / 위상
        r_peaks = ctx.get("r_peaks")
        if r_peaks is None:
            from ..eval.rpeak import detect_rpeaks
            r_peaks = detect_rpeaks(x, fs)
        r_peaks = np.asarray(r_peaks, dtype=int)
        if r_peaks.size < 3:
            return x                                    # beat 가 없으면 손대지 않는다
        theta, omega = assign_phase(r_peaks, x.size, fs)

        # --- 정규화 (A-2 #6): R 진폭 기준
        amp = float(np.median(np.abs(x[r_peaks])))
        scale = amp if amp > 1e-9 else float(np.std(x) + 1e-9)
        z = x / scale

        # --- 커널 적합 (A-2 #3)
        # 상태방정식은 dz/dtheta 만 규정하므로 z 의 **상수 오프셋은 동역학과 무관**하다.
        # front-end HPF 가 이미 DC 를 없앴으므로 template 도 평균을 제거해 맞춘다.
        # (이 처리를 빼면 R^2 가 0.82 로 떨어지고 q_z 가 과대추정되어 성능이 크게 나빠진다)
        grid, tmpl_raw, tvar = phase_average(z, theta)
        tmpl_mean = float(tmpl_raw.mean())
        tmpl = tmpl_raw - tmpl_mean
        if self.kernel is not None:
            # 참 파라미터 주입 경로 (STEP 13 DoD 검증용).
            # kernel 의 alpha 는 **정규화 스케일**(R 진폭 = 1) 기준이어야 한다.
            k = self.kernel
            a_, b_, t_ = (np.asarray(k.alpha, float), np.asarray(k.b, float),
                          np.asarray(k.theta, float))
            fitted = _kernel_sum(grid, a_, b_, t_)
            fitted = fitted - fitted.mean()
            ss = float(np.sum((tmpl - fitted) ** 2))
            st = float(np.sum((tmpl - tmpl.mean()) ** 2))
            fit = KernelFit(alpha=a_, b=b_, theta=t_, r2=1.0 - ss / max(st, 1e-30),
                            template_grid=grid, template=tmpl, fitted=fitted)
        else:
            fit = fit_kernels(grid, tmpl, self.n_kernels)

        a_i = np.asarray(fit.alpha, dtype=np.float64)
        b_i = np.asarray(fit.b, dtype=np.float64)
        t_i = np.asarray(fit.theta, dtype=np.float64)

        # --- Q, R (A-2 #4)
        r_meas = self._estimate_r(z, r_peaks) * self.r_scale
        # 모델 잔차에서 상태잡음 분산을 유도한다.
        # **주의**: q_z 는 '매 샘플의' 상태잡음 분산이고, 관측되는 모델 잔차는
        # beat 한 개 길이 L 동안 **누적된** 편차다 (위상이 beat 마다 재고정되므로).
        # random walk 의 누적 분산 = L * q_z 이므로 L 로 나눠야 한다.
        # 이 정규화를 빼면 q_z 가 수백 배 과대추정되어 필터가 거의 평활화를 하지 않는다.
        pred = np.interp(theta, grid, fit.fitted, period=TWO_PI)
        resid_var = float(np.var(z - pred))
        samples_per_beat = max(float(x.size) / max(r_peaks.size, 1), 8.0)
        q_z = max(resid_var - r_meas, 1e-12) / samples_per_beat * self.q_scale
        # 위상 잡음: RR 변동 기반
        dt = 1.0 / fs
        q_th = float(np.var(np.diff(omega)) * dt ** 2) + (0.02 * np.median(omega) * dt) ** 2
        r_phase = (self.phase_sigma_ms * 1e-3 * float(np.median(omega))) ** 2

        Q = np.diag([max(q_th, 1e-12), max(q_z, 1e-12)])
        R = np.diag([max(r_phase, 1e-12), max(r_meas, 1e-12)])
        H = np.eye(2)

        n = z.size
        xs = np.zeros((n, 2))          # a posteriori
        Ps = np.zeros((n, 2, 2))
        xp = np.zeros((n, 2))          # a priori
        Pp = np.zeros((n, 2, 2))
        Fs = np.zeros((n, 2, 2))

        xk = np.array([theta[0], z[0]])
        Pk = np.diag([r_phase, r_meas])

        for k in range(n):
            w = omega[k]
            # ---- predict
            d = _wrap(xk[0] - t_i)
            g = np.exp(-(d ** 2) / (2.0 * b_i ** 2))
            drift = -dt * np.sum((a_i * w / b_i ** 2) * d * g)
            xpk = np.array([_wrap(xk[0] + w * dt), xk[1] + drift])
            dfdth = -dt * np.sum((a_i * w / b_i ** 2) * g * (1.0 - (d ** 2) / b_i ** 2))
            F = np.array([[1.0, 0.0], [dfdth, 1.0]])
            Ppk = F @ Pk @ F.T + Q

            # ---- update (관측: 위상, 진폭)
            innov = np.array([_wrap(theta[k] - xpk[0]), z[k] - xpk[1]])
            S = Ppk + R
            K = Ppk @ np.linalg.inv(S)
            xk = xpk + K @ innov
            xk[0] = _wrap(xk[0])
            I_KH = np.eye(2) - K @ H
            Pk = I_KH @ Ppk @ I_KH.T + K @ R @ K.T

            xp[k], Pp[k], Fs[k] = xpk, Ppk, F
            xs[k], Ps[k] = xk, Pk

        out = xs[:, 1]

        # ---- RTS backward smoother (A-2 #5)
        if self.smooth:
            xsm = xs.copy()
            Psm = Ps.copy()
            for k in range(n - 2, -1, -1):
                try:
                    C = Ps[k] @ Fs[k + 1].T @ np.linalg.inv(Pp[k + 1])
                except np.linalg.LinAlgError:      # pragma: no cover
                    continue
                diff = xsm[k + 1] - xp[k + 1]
                diff[0] = _wrap(diff[0])
                xsm[k] = xs[k] + C @ diff
                xsm[k][0] = _wrap(xsm[k][0])
                Psm[k] = Ps[k] + C @ (Psm[k + 1] - Pp[k + 1]) @ C.T
            out = xsm[:, 1]

        self.last_info = dict(r2=fit.r2, q_z=q_z, q_th=q_th, r_meas=r_meas,
                              resid_var=resid_var, samples_per_beat=samples_per_beat,
                              tmpl_mean=tmpl_mean,
                              r_phase=r_phase, scale=scale, n_beats=int(r_peaks.size),
                              fit=fit, theta=theta, omega=omega)
        return out * scale


@register_method("M05", family="model", label="Sameni EKS (EKF + RTS smoother)")
def _m05(**kw):
    return SameniKalman(smooth=True, **kw)          # use_frontend 를 그대로 받는다


@register_method("M05f", family="model", label="Sameni EKF (forward only)")
def _m05f(**kw):
    return SameniKalman(smooth=False, **kw)
