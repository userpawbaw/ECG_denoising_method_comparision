"""M05S — **스트리밍** Sameni EKF + 고정 지연 RTS 평활 (D-40).

M05(`kalman_sameni.py`)는 창마다 R 피크 검출 · 커널 적합 · EKF · RTS 를 처음부터
다시 한다. 실시간 브리지(hop 12)에서는 샘플 하나를 85 번 다시 처리해 코어 3.5~7 개어치가
들었다 (F-57). 칼만 루프 자체는 샘플당 71 µs 라, **상태를 이어 가면** 2 % 면 된다.

그래서 창 전체를 보던 셋을 실시간에 맞게 바꿨다 — 무엇을 왜 바꿨는지는 D-40.

    커널 · Q · R   처음 `init_s` 로 한 번 적합(M05 와 같은 식), `refit_s` 마다 최근
                   `refit_win_s` 로 다시 — 이전 커널에서 출발한다
    R 피크         init 때 XQRS 로 극성·진폭을 잡고, 그 뒤로는 문턱 + 40 ms 최댓값의 인과 검출기
    전진 지연      전진 EKF 를 `lag` 샘플 늦게 돈다 — 그 사이에 확정된 다음 R 로 위상을 **보간**한다
    위상 관측      다음 R 을 알면 M05 처럼 보간, 모르면 RR 중앙값으로 외삽(관측 분산을 키운다)
    위상 과정잡음  RR 변동만큼 키운다 — ω 를 외삽하는 대가 (끄면 14.5 → 3.6 dB)
    평활           hop 마다 최근 hop + d 샘플만 거꾸로 가는 **고정 지연 RTS**

기본값(lag 36 · d 12 · hop 12, 지연 240 ms)은 스윕으로 골랐다 — D-40 「결과」.

**M05 와 같은 방법이 아니다.** 오프라인 대조 결과는 D-40 「결과」에 있다.

인터페이스는 브리지가 `StreamProcessor` 에 기대는 것과 같다 — `push` · `origin` ·
`latency_s` · `warmup_s` · `runs_per_s` · `hop` · `n_runs` · `reset`.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..config import DEFAULT_KERNEL, ECGKernel
from ..registry import register_method
from .base import BaseDenoiser
from .frontend import FrontEnd
from .kalman_sameni import (TWO_PI, SameniKalman, _wrap, assign_phase,
                            fit_kernels, phase_average)

__all__ = ["StreamingSameni", "SameniStream", "fit_params"]


def _wrap1(x: float) -> float:
    return (x + np.pi) % TWO_PI - np.pi


def fit_params(x: np.ndarray, r_peaks: np.ndarray, fs: float, scale: float | None = None,
               init: ECGKernel = DEFAULT_KERNEL, n_kernels: int = 7,
               q_scale: float = 1.0, r_scale: float = 1.0,
               phase_sigma_ms: float = 12.0, max_nfev: int = 4000) -> dict[str, Any] | None:
    """M05 `_run` 의 적합부와 **같은 식**으로 커널 · Q · R 을 구한다. R 피크가 3 개 미만이면 None.

    `scale` 을 주면 그 값으로 정규화한다 — 재적합 때 스케일을 바꾸지 않기 위해서다
    (바꾸면 진행 중인 상태 z 와 저장된 이력의 단위가 갈라진다).
    """
    r = np.asarray(r_peaks, dtype=int)
    r = r[(r >= 0) & (r < x.size)]
    if r.size < 3:
        return None
    theta, omega = assign_phase(r, x.size, fs)
    if scale is None:
        amp = float(np.median(np.abs(x[r])))
        scale = amp if amp > 1e-9 else float(np.std(x) + 1e-9)
    z = x / scale
    grid, tmpl_raw, _ = phase_average(z, theta)
    tmpl = tmpl_raw - float(tmpl_raw.mean())
    fit = fit_kernels(grid, tmpl, n_kernels, init=init, max_nfev=max_nfev)
    r_meas = SameniKalman._estimate_r(z, r) * r_scale
    pred = np.interp(theta, grid, fit.fitted, period=TWO_PI)
    resid_var = float(np.var(z - pred))
    samples_per_beat = max(float(x.size) / max(r.size, 1), 8.0)
    q_z = max(resid_var - r_meas, 1e-12) / samples_per_beat * q_scale
    dt = 1.0 / fs
    q_th = float(np.var(np.diff(omega)) * dt ** 2) + (0.02 * np.median(omega) * dt) ** 2
    r_phase = (phase_sigma_ms * 1e-3 * float(np.median(omega))) ** 2
    return dict(scale=float(scale), a=np.asarray(fit.alpha, float),
                b=np.asarray(fit.b, float), t=np.asarray(fit.theta, float), r2=fit.r2,
                q_th=max(q_th, 1e-12), q_z=max(q_z, 1e-12),
                r_phase=max(r_phase, 1e-12), r_meas=max(r_meas, 1e-12),
                kernel=fit.as_kernel())


def _refine_peaks(x: np.ndarray, pk: np.ndarray, sgn: float, w: int) -> np.ndarray:
    """XQRS 위치를 극성 쪽 최댓값(±w)으로 옮긴다 — 인과 검출기와 같은 규약."""
    out = []
    for v in pk:
        lo, hi = max(0, v - w), min(x.size, v + w + 1)
        out.append(lo + int(np.argmax(sgn * x[lo:hi])))
    return np.unique(np.asarray(out, dtype=int))


def _init_job(xa: np.ndarray, fs: float, conf: int) -> dict[str, Any] | None:
    """init 적합 — XQRS · 극성 · 커널 · Q · R. **본체 스레드 밖에서 돌 수 있게** 입력만 받는다."""
    from ..eval.rpeak import detect_rpeaks
    pk = detect_rpeaks(xa, fs)
    if pk.size < 3:
        return None
    # **위치 규약을 인과 검출기와 맞춘다** — 커널은 이 위상으로 적합되므로, 실시간
    # 검출기가 다른 자리를 R 로 잡으면 템플릿 전체가 그만큼 밀린다.
    sgn = 1.0 if np.median(xa[pk]) >= 0 else -1.0
    pk = _refine_peaks(xa, pk, sgn, conf)
    p = fit_params(xa, pk, fs)
    if p is None or not np.isfinite(p["r2"]) or p["r2"] < 0.3:
        return None
    return dict(p=p, sgn=sgn, thr=0.5 * float(np.median(sgn * xa[pk])), peaks=pk)


def _refit_job(xa: np.ndarray, pk: np.ndarray, fs: float, scale: float,
               kernel: ECGKernel) -> dict[str, Any] | None:
    """재적합. 이전 커널에서 출발하므로 반복 상한을 400 으로 둔다 (init 은 4000)."""
    p = fit_params(xa, pk, fs, scale=scale, init=kernel, max_nfev=400)
    if p is None or not np.isfinite(p["r2"]) or p["r2"] < 0.3:
        return None
    return p


class StreamingSameni:
    """샘플이 들어오는 대로 EKF 를 한 걸음씩, hop 마다 고정 지연 RTS 로 내보낸다."""

    def __init__(self, fs: float = 250.0, hop: int = 12, d: int = 12,
                 lag: int = 36, q_rr: bool = True, background: bool = False,
                 init_s: float = 1024 / 250, refit_s: float = 10.0,
                 refit_win_s: float = 8.0, keep_s: float = 12.0):
        self.fs = float(fs)
        self.hop = int(hop)
        self.d = int(d)                # RTS 가 앞을 보는 샘플 (고정 지연 평활)
        # **전진 EKF 를 lag 샘플 늦게 돈다.** R 피크는 지나간 뒤 40 ms 에 확정되므로,
        # 전진이 R 에 닿을 때 이미 알고 있어야 위상이 QRS 바로 그 자리에서 0 이 된다.
        # 이것을 안 하면 위상이 외삽 RR 로 밀려 좁은 QRS 커널이 엇나갔다 — 적합값과
        # 위상 관측을 M05 와 똑같이 줘도 22.5 dB 가 3.4 dB 가 됐다 (D-40 결과).
        self.lag = int(lag)
        self.q_rr = bool(q_rr)
        self.conf = int(round(0.04 * fs))   # R 확정 창: 문턱을 넘고 이만큼 안의 최댓값
        self.refr = int(round(0.25 * fs))   # 불응기
        self.init_n = int(round(init_s * fs))
        self.refit_n = int(round(refit_s * fs))
        self.refit_win = int(round(refit_win_s * fs))
        self.keep = int(round(keep_s * fs))
        self.origin = 0
        # **적합(XQRS · least_squares)을 본체 스레드에서 돌리지 않는다** (실시간 전용).
        # 본체에서 돌리면 init 때 0.5~1.9 s, 재적합 때 수십~수백 ms 동안 브리지가 멈춰
        # 큐가 밀리고 RTF 와 도착 간격이 튀었다 — 실보드에서 «시작 20 초 안에 갑자기
        # 멈췄다 돌아온다» 로 보였다 (O-38). 오프라인 대조는 결과가 실행마다 같아야
        # 하므로 동기로 둔다.
        self.background = bool(background)
        self._pool = None
        if self.background:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="m05s-fit")
        # **wfdb 를 미리 불러 둔다.** XQRS 를 처음 부를 때 import 가 약 1 s(윈도우는 더)
        # 걸리는데, 그것이 init 순간에 일어났다. 처리기를 만들 때 치르면 포트를 열기 전이다.
        try:
            import wfdb.processing  # noqa: F401
        except Exception:                                   # pragma: no cover
            pass
        self.reset()

    # ----------------------------------------------------------- 브리지 인터페이스
    @property
    def latency_s(self) -> float:
        return (self.lag + self.hop + self.d) / self.fs

    @property
    def warmup_s(self) -> float:
        return self.init_n / self.fs

    @property
    def runs_per_s(self) -> float:
        return self.fs / self.hop

    def reset(self) -> None:
        self.n_in = 0                  # 받은 샘플 수 (절대 인덱스의 끝)
        self.n_fwd = 0                 # EKF 전진을 마친 샘플 수
        self.emitted = 0               # 내보낸 샘플 수
        self.n_runs = 0                # 평활(내보내기) 횟수
        self.buf = np.zeros(0)         # 원 입력. buf[0] 의 절대 번호가 buf0
        self.buf0 = 0
        self.peaks: list[int] = []     # 확정 R 피크 (절대 번호)
        self.p: dict[str, Any] | None = None
        self.t_refit = 0
        self.t_init_try = 0
        self.n_refit = 0
        self.passthrough_until = 0     # 이 앞은 적합 전이라 입력을 그대로 낸다
        self.n_det = 0                 # 인과 검출기가 훑은 샘플 수
        self._cand = -1                # 문턱을 넘은 첫 샘플 (확정 대기)
        self.sgn = 1.0                 # R 의 극성
        self.thr = 0.0                 # 검출 문턱 [입력 단위]
        self._hist: list[tuple] = []   # (xp, Pp, F, xs, Ps) — 인덱스 n_hist0 부터
        self._hist0 = 0
        self._x = None
        self._P = None
        self._job = None               # (종류, 낸 시점 n_in, 창 시작, Future)
        self._gen = getattr(self, "_gen", 0) + 1   # reset 전에 낸 작업의 결과는 버린다

    # ----------------------------------------------------------------- 입력
    def push(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float64).ravel()
        if block.size:
            self.buf = np.concatenate([self.buf, block])
            self.n_in += block.size
        out: list[np.ndarray] = []
        self._collect()
        if self.p is None:
            self._try_init()
        if self.p is not None:
            self._detect_causal()
            self._maybe_refit()
            self._forward()
            while self.n_fwd - self.emitted >= self.hop + self.d:
                out.append(self._emit(self.hop))
        else:
            out.append(self._passthrough())
        # 오래된 입력은 버린다 — 재적합 창 + 여유만 남긴다 (F-31: 무한히 자라면 느려진다)
        drop = min(self.n_in - self.keep, self.emitted) - self.buf0
        if drop > 0:
            self.buf = self.buf[drop:]
            self.buf0 += drop
        out = [o for o in out if o.size]
        return np.concatenate(out) if out else np.zeros(0)

    def finish(self) -> np.ndarray:
        """끝까지 평활해 남은 것을 모두 낸다 (오프라인 전용)."""
        if self.p is None:
            n = self.n_in - self.emitted
            seg = self.buf[self.emitted - self.buf0:self.n_in - self.buf0]
            self.emitted += n
            return seg.copy()
        self.n_det = self.n_in
        self.lag, lag0 = 0, self.lag
        self._forward()
        self.lag = lag0
        rest = self.n_fwd - self.emitted
        return self._emit(rest) if rest > 0 else np.zeros(0)

    # ----------------------------------------------------------------- 내부
    def _x_abs(self, a: int, b: int) -> np.ndarray:
        return self.buf[a - self.buf0:b - self.buf0]

    def _passthrough(self) -> np.ndarray:
        """적합 전에는 입력을 같은 지연으로 그대로 낸다 — 화면을 세우지 않는다 (D-40).

        단 처음 `init_n` 은 내지 않는다. 그 구간은 적합에 성공하면 EKF 로 다시 돌기
        때문이다 — 다른 방법의 warm-up 과 같은 자리다.
        """
        if self.n_in < self.init_n:
            return np.zeros(0)
        end = self.n_in - self.lag - self.hop - self.d
        if end <= self.emitted:
            return np.zeros(0)
        seg = self._x_abs(self.emitted, end).copy()
        self.emitted = end
        self.passthrough_until = end
        return seg

    def _detect_causal(self) -> None:
        """문턱(R 진폭 중앙값의 절반)을 넘으면 후보, 그 뒤 40 ms 안의 최댓값을 R 로 확정한다.

        XQRS(M05 의 검출기)를 0.5 s 마다 돌리면 R 을 아는 데 최대 0.6 s 가 걸렸다. 그만큼
        전진을 늦출 수는 없다. 그래서 init 때 XQRS 로 극성과 진폭만 잡고, 그 뒤로는 이
        검출기를 쓴다 (D-40 결과).
        """
        k = max(self.n_det, self.buf0)
        end = self.n_in
        last = self.peaks[-1] if self.peaks else -10 ** 9
        while k < end:
            if self._cand < 0:
                if self.sgn * self.buf[k - self.buf0] > self.thr and k - last > self.refr:
                    self._cand = k
                k += 1
                continue
            if end - self._cand <= self.conf:      # 확정 창이 아직 다 안 찼다
                break
            seg = self.sgn * self._x_abs(self._cand, self._cand + self.conf + 1)
            r = self._cand + int(np.argmax(seg))
            self.peaks.append(r)
            last = r
            # 문턱은 R 진폭을 천천히 따라간다 (전극·자세가 바뀌면 진폭이 바뀐다)
            self.thr = 0.9 * self.thr + 0.1 * 0.5 * float(seg.max())
            self._cand = -1
            k = r + 1
        self.n_det = k if self._cand < 0 else self._cand
        if len(self.peaks) > 64:
            del self.peaks[:-64]

    # ------------------------------------------------------- 적합 작업 (init · 재적합)
    def _submit(self, kind: str, fn, *args) -> None:
        a = args[-1]
        args = args[:-1]
        if self._pool is None:
            self._apply(kind, self.n_in, a, fn(*args))
            return
        self._job = (kind, self.n_in, a, self._gen, self._pool.submit(fn, *args))

    def _collect(self) -> None:
        """끝난 작업이 있으면 그 결과를 적용한다 (본체 스레드에서만 상태를 바꾼다)."""
        if self._job is None or not self._job[4].done():
            return
        kind, n_at, a, gen, fut = self._job
        self._job = None
        if gen != self._gen:
            return
        try:
            res = fut.result()
        except Exception:                                   # pragma: no cover
            return
        self._apply(kind, n_at, a, res)

    def _apply(self, kind: str, n_at: int, a: int, res) -> None:
        if res is None:
            return
        if kind == "refit":
            self.p = res
            self.n_refit += 1
            return
        # ---- init: 적합한 창은 [a, n_at). 그 뒤는 인과 검출기가 이어서 훑는다
        self.p = res["p"]
        self.sgn, self.thr = res["sgn"], res["thr"]
        self.peaks = [int(v) + a for v in res["peaks"]]
        self.n_det = n_at
        self._cand = -1
        self.t_refit = n_at
        start = self.emitted                    # 아직 안 낸 곳부터 EKF 를 시작한다
        self.n_fwd = start
        self._hist, self._hist0 = [], start
        x0 = self._x_abs(start, start + 1)[0] / self.p["scale"]
        self._x = np.array([self._phase_obs(start)[0], x0])
        self._P = np.diag([self.p["r_phase"], self.p["r_meas"]])

    def _try_init(self) -> None:
        if self._job is not None:
            return
        if self.n_in < self.init_n or self.n_in - self.t_init_try < self.init_n // 2:
            return
        self.t_init_try = self.n_in
        a = max(self.buf0, self.n_in - self.refit_win)
        self._submit("init", _init_job, self._x_abs(a, self.n_in).copy(), self.fs,
                     self.conf, a)

    def _rr(self) -> tuple[float, float]:
        """최근 RR 의 중앙값 [샘플] 과 변동계수."""
        if len(self.peaks) < 2:
            return self.fs * 60.0 / 70.0, 0.1
        rr = np.diff(self.peaks[-9:]).astype(float)
        rr = rr[(rr > 0.25 * self.fs) & (rr < 2.5 * self.fs)]
        if rr.size == 0:
            return self.fs * 60.0 / 70.0, 0.1
        med = float(np.median(rr))
        return med, float(np.std(rr) / med) if rr.size > 1 else 0.05

    def _phase_obs(self, k: int, rr_cv: tuple[float, float] | None = None
                   ) -> tuple[float, float]:
        """표본 k 의 위상 관측과 그 **추가 분산** — 마지막 확정 R 에서 외삽한다."""
        import bisect
        rr, cv = rr_cv if rr_cv is not None else self._rr()
        i = bisect.bisect_right(self.peaks, k)
        last = self.peaks[i - 1] if i > 0 else None
        if last is not None and i < len(self.peaks):
            # **다음 R 을 이미 안다** (전진이 lag 만큼 늦게 돈다) — M05 처럼 두 R 사이를
            # 선형 보간한다. 외삽 오차가 없으므로 추가 분산도 없다.
            span = self.peaks[i] - last
            if 0.25 * self.fs < span < 2.5 * self.fs:
                return _wrap1(TWO_PI * (k - last) / span), 0.0
        if last is None:
            return 0.0, (np.pi / 2) ** 2
        frac = (k - last) / rr
        extra = (TWO_PI * min(frac, 2.0) * max(cv, 0.02)) ** 2
        return _wrap1(TWO_PI * frac), extra

    def _maybe_refit(self) -> None:
        if self._job is not None or self.n_in - self.t_refit < self.refit_n:
            return
        self.t_refit = self.n_in
        a = max(self.buf0, self.n_in - self.refit_win)
        pk = np.asarray([q for q in self.peaks if a <= q < self.n_in], dtype=int) - a
        self._submit("refit", _refit_job, self._x_abs(a, self.n_in).copy(), pk, self.fs,
                     self.p["scale"], self.p["kernel"], a)

    def _forward(self) -> None:
        p = self.p
        a_i, b_i, t_i = p["a"], p["b"], p["t"]
        ab = a_i / b_i ** 2
        dt = 1.0 / self.fs
        rr_cv = self._rr()
        w = TWO_PI * self.fs / rr_cv[0]
        # **ω 의 불확실성을 위상 과정잡음에 넣는다.** M05 는 다음 R 을 알고 박동마다 ω 를
        # 정확히 쓰므로 q_th 가 작아도 된다. 여기서는 RR 중앙값으로 외삽하므로, 박동
        # 하나 동안 위상 오차 분산이 (2π·cv)² 까지 쌓인다 — 그것을 샘플마다 나눠 준다.
        q_th = (max(p["q_th"], (TWO_PI * max(rr_cv[1], 0.03)) ** 2 / rr_cv[0])
                if self.q_rr else p["q_th"])
        Q = np.array([[q_th, 0.0], [0.0, p["q_z"]]])
        import bisect
        peaks = self.peaks

        def w_k(k: int) -> float:
            """다음 R 을 알면 그 박동의 ω, 모르면 RR 중앙값의 ω."""
            i = bisect.bisect_right(peaks, k)
            if 0 < i < len(peaks):
                span = peaks[i] - peaks[i - 1]
                if 0.25 * self.fs < span < 2.5 * self.fs:
                    return TWO_PI * self.fs / span
            return TWO_PI * self.fs / rr_cv[0]
        x, P = self._x, self._P
        scale = p["scale"]
        upto = min(self.n_in - self.lag, self.n_det)    # 검출기가 훑은 데까지만
        for k in range(self.n_fwd, upto):
            s = self.buf[k - self.buf0] / scale
            phi, extra = self._phase_obs(k, rr_cv)
            w = w_k(k)
            # ---- predict (M05 와 같은 식)
            dd = _wrap(x[0] - t_i)
            g = np.exp(-(dd ** 2) / (2.0 * b_i ** 2))
            drift = -dt * float(np.sum(ab * w * dd * g))
            xp = np.array([_wrap1(x[0] + w * dt), x[1] + drift])
            dfdth = -dt * float(np.sum(ab * w * g * (1.0 - (dd ** 2) / b_i ** 2)))
            F = np.array([[1.0, 0.0], [dfdth, 1.0]])
            Pp = F @ P @ F.T + Q
            # ---- update
            R = np.array([[p["r_phase"] + extra, 0.0], [0.0, p["r_meas"]]])
            innov = np.array([_wrap1(phi - xp[0]), s - xp[1]])
            K = Pp @ np.linalg.inv(Pp + R)
            x = xp + K @ innov
            x[0] = _wrap1(x[0])
            IK = np.eye(2) - K
            P = IK @ Pp @ IK.T + K @ R @ K.T
            self._hist.append((xp, Pp, F, x.copy(), P))
        self._x, self._P = x, P
        self.n_fwd = max(self.n_fwd, upto)

    def _emit(self, n: int) -> np.ndarray:
        """[emitted, emitted + n) 를 **고정 지연 RTS** 로 평활해 낸다."""
        h, h0 = self._hist, self._hist0
        last = self.n_fwd - 1
        xsm = h[last - h0][3].copy()
        out = np.empty(n)
        stop = self.emitted
        for k in range(last - 1, stop - 1, -1):
            xp1, Pp1, F1, _, _ = h[k + 1 - h0]
            _, _, _, xs, Ps = h[k - h0]
            try:
                C = Ps @ F1.T @ np.linalg.inv(Pp1)
            except np.linalg.LinAlgError:          # pragma: no cover
                xsm = xs.copy()
                continue
            diff = xsm - xp1
            diff[0] = _wrap1(diff[0])
            xsm = xs + C @ diff
            xsm[0] = _wrap1(xsm[0])
            if k < stop + n:
                out[k - stop] = xsm[1]
        if n == self.n_fwd - stop:                 # 끝까지 내는 경우(finish) 마지막 표본
            out[n - 1] = h[last - h0][3][1]
        self.emitted += n
        self.n_runs += 1
        del self._hist[:self.emitted - self._hist0]
        self._hist0 = self.emitted
        return out * self.p["scale"]


class SameniStream(BaseDenoiser):
    """M05S 의 **오프라인** 경로 — 스트리밍 코어를 처음부터 끝까지 흘려 같은 길이로 돌려준다.

    오프라인 평가 틀에서 M05 와 같은 자리에서 비교하기 위한 것이다. 입력을 25 샘플씩
    넣는다 — 실시간 브리지와 같은 내보내기 단위(hop)로 나오게 한다.
    """

    def __init__(self, use_frontend: bool = True, hop: int = 12, d: int = 12,
                 lag: int = 36, q_rr: bool = True, name: str = "M05S"):
        self.fe = FrontEnd() if use_frontend else None
        self.hop, self.d, self.lag, self.q_rr = int(hop), int(d), int(lag), bool(q_rr)
        self.name = name
        self.last_info: dict[str, Any] = {}

    def _run(self, y: np.ndarray, fs: float, ctx: dict[str, Any]) -> np.ndarray:
        x = self.fe(y, fs) if self.fe is not None else y
        core = StreamingSameni(fs, hop=self.hop, d=self.d, lag=self.lag, q_rr=self.q_rr)
        parts = [core.push(x[i:i + 25]) for i in range(0, x.size, 25)]
        parts.append(core.finish())
        out = np.concatenate([q for q in parts if q.size]) if parts else np.zeros(0)
        self.last_info = dict(initialized=core.p is not None, n_refit=core.n_refit,
                              passthrough=core.passthrough_until,
                              r2=core.p["r2"] if core.p else float("nan"))
        if out.size < x.size:                       # 적합 전 앞부분이 비었으면 입력으로
            out = np.concatenate([x[:x.size - out.size], out])
        return out[:x.size]


@register_method("M05S", family="model",
                 label="Sameni EKF · 스트리밍 + 고정 지연 평활 (D-40)")
def _m05s(**kw):
    return SameniStream(**kw)
