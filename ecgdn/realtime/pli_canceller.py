"""전원선 잡음 **고조파 추적 제거기** — 고정 노치의 대체 (D-52).

왜 노치가 아닌가
----------------
D-50 의 노치(Q30)는 −3 dB 폭이 2 Hz 라 한전 주파수가 ±0.2 Hz 만 흔들려도
감쇠가 −14 dB 로 떨어진다. 폭을 넓히면 QRS 가 같이 깎인다. 그리고 250 Hz 에서
접힌 고조파(180 → 70 Hz, 240 → 10 Hz, 300 → 50 Hz …)는 노치 자리에 오지 않는다.

이 모듈은 대역을 깎지 않고 **사인파를 추정해 뺀다.**

    y_n = ECG_n + Σ_k a_k cos(k θ_n) + b_k sin(k θ_n)

- 진폭 (a_k, b_k) 는 랜덤워크 칼만으로 추정한다.
- 공통 위상 θ 와 주파수 f 는 2 차 PLL 로 따라간다. 위상 검출은 «혁신 × ∂예측/∂θ» 다.
- 이산 표본에서 cos(k θ_n) 은 **접힌 고조파의 표본열 그 자체**다. 그래서 fs 를
  바꾸지 않고도 나이퀴스트 위의 고조파까지 뺀다. 클럭 오차도 f 에 흡수된다.
- 혁신이 크면(QRS) 갱신하지 않는다.
- 획득 중(약 3 s)에는 입력을 그대로 낸다 — 덜 맞춘 추정을 빼서 파형을 망가뜨리지 않게. QRS 가 사인파 추정으로 새지 않게 하려는 것이다.
- 획득 모드(시작 시, 또는 잡음이 갑자기 커질 때)에서는 칼만을 빠르게, PLL 을 넓게 쓴다.
  잠기면 PLL 을 좁힌다(기어 변속).
- 기본파가 작으면(전원선 잡음 없음) PLL 을 멈춘다. ECG 를 쫓아가 왜곡을 만들지 않게.

기저선 변동은 혁신을 키워 위 판정을 흔든다 `[측정]` (D-52: 잔류 0.064 → 0.413).
그래서 `PliRemover` 는 0.5 Hz HPF 를 건 **사본**에서 잡음을 추정하고, 그 추정치만
원신호에서 뺀다. 0.5 Hz HPF 는 60 Hz 근처 위상을 거의 바꾸지 않는다. 그리고 각
front-end 모드가 하는 기저선 처리(영위상 · 중앙값)를 건드리지 않는다.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

__all__ = ["HarmonicCanceller", "PliRemover"]


class HarmonicCanceller:
    """전원선 고조파 추적 제거기. `push(block) -> block` (인과 · 표본 단위 · 지연 0)."""

    def __init__(self, fs: float, f0: float = 60.0, n_harm: int = 9, q: float = 1e-6,
                 q_acq: float = 1e-3, gate: float = 4.0, f_lim: tuple[float, float] = (59.0, 61.0),
                 bw_hi: float = 0.5, bw_lo: float = 0.05, acq_s: float = 3.0,
                 lock_pd: float = 0.05, min_amp: float = 0.5):
        self.fs, self.f0, self.K, self.lim = float(fs), float(f0), int(n_harm), f_lim
        self.k = np.arange(1, self.K + 1)
        self.q, self.q_acq, self.gate, self.acq = q, q_acq, gate, int(acq_s * fs)
        self.ramp = int(0.25 * fs)          # 획득이 끝난 뒤 빼기를 0 -> 1 로 올리는 길이
        self.g_hi, self.g_lo = self._gains(bw_hi), self._gains(bw_lo)
        self.lock_pd, self.min_amp = lock_pd, min_amp
        self.reset()

    def _gains(self, bw_hz: float) -> tuple[float, float]:
        """2 차 PLL 의 (kp, ki). 감쇠 0.707, 자연주파수 bw_hz. 위상 검출기 이득은 1 로 정규화했다."""
        wn = 2 * np.pi * bw_hz / self.fs
        return 2 * 0.707 * wn, wn * wn

    def reset(self) -> None:
        self.f = self.f0
        self.x = np.zeros(2 * self.K)
        self.P = np.eye(2 * self.K) * 1e2
        self.th, self.sig, self.fast = 0.0, None, None
        self.acq_left, self.pd_ema, self.on = self.acq, 1.0, 0

    @property
    def amplitudes(self) -> np.ndarray:
        """고조파별 진폭 추정 (입력 단위)."""
        return np.hypot(self.x[:self.K], self.x[self.K:])

    def push(self, y) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel()
        out = np.empty_like(y)
        K, diag = self.K, np.diag_indices(2 * self.K)
        for i, v in enumerate(y):
            self.th += 2 * np.pi * self.f / self.fs
            kt = self.k * self.th
            c, s = np.cos(kt), np.sin(kt)
            h = np.concatenate([c, s])
            pred = h @ self.x
            e = v - pred
            ae = abs(e)
            if self.sig is None:
                self.sig = self.fast = ae + 1e-9
            # 혁신이 느린 척도의 3 배를 넘게 이어지면 새 잡음이 들어온 것이다 → 다시 획득
            self.fast = 0.99 * self.fast + 0.01 * ae
            if self.fast > 3 * self.sig and self.acq_left <= 0:
                self.acq_left = self.acq
            acq = self.acq_left > 0
            # **획득 중에는 입력을 그대로 낸다.** 그동안 추정은 빠르게(큰 q, 문턱 없음) 돌므로
            # ECG 까지 사인파로 맞춘다 — 그 «출력» 을 내보내면 시작 3 초가 망가진다 `[측정]`
            # (깨끗한 입력 0~5 s 왜곡 1.25). 잠긴 뒤 0.25 s 동안 빼기를 0 -> 1 로 올린다.
            self.on = 0 if acq else min(self.ramp, self.on + 1)
            out[i] = v - (self.on / self.ramp) * pred
            R = self.sig ** 2
            self.P[diag] += (self.q_acq if acq else self.q) * R
            if acq or ae < self.gate * 1.4826 * self.sig:
                Ph = self.P @ h
                kg = Ph / (h @ Ph + R)
                self.x += kg * e
                self.P -= np.outer(kg, Ph)
                self.sig = 0.995 * self.sig + 0.005 * ae
                a, b = self.x[:K], self.x[K:]
                if np.hypot(a[0], b[0]) > self.min_amp * self.sig:
                    d = self.k * (-a * s + b * c)
                    g = float(self.k ** 2 @ (a * a + b * b)) / 2 + 1e-12
                    pd = float(np.clip(e * d.sum() / g, -0.5, 0.5))
                    self.pd_ema = 0.998 * self.pd_ema + 0.002 * abs(pd)
                    w = 1.0 if acq else min(1.0, self.pd_ema / self.lock_pd)
                    kp = self.g_lo[0] + w * (self.g_hi[0] - self.g_lo[0])
                    ki = self.g_lo[1] + w * (self.g_hi[1] - self.g_lo[1])
                    self.th += kp * pd
                    self.f = float(np.clip(self.f + ki * pd * self.fs / (2 * np.pi), *self.lim))
            self.acq_left -= 1
            # 획득이 끝나는 순간 기본파가 작으면 잡음은 없었다 — 그동안 ECG 를 맞추며 쌓인
            # 진폭을 지운다. 안 지우면 q 가 작아 수 초간 남아 ECG 를 깎는다 `[측정]` (5~10 s 0.047)
            if self.acq_left == 0 and np.hypot(self.x[0], self.x[K]) < self.min_amp * self.sig:
                self.x[:] = 0.0
                # P 도 줄인다 — 획득 중 큰 q 로 쌓인 P 가 남으면 곧바로 다시 ECG 를 맞춘다.
                # 특히 4 차(240 Hz)는 250 Hz 에서 10 Hz 로 접혀 ECG 대역 한가운데다.
                self.P = np.eye(2 * K) * (self.q * self.sig ** 2 * self.acq)
        return out


class PliRemover:
    """HPF 사본에서 전원선 잡음을 추정해 **원신호에서 그 추정치만** 뺀다. 지연 0."""

    def __init__(self, fs: float, hp_hz: float = 0.5, **kw):
        self.fs = float(fs)
        self._hp = sps.butter(2, hp_hz / (self.fs / 2), btype="highpass", output="sos")
        self._kw = kw
        self.reset()

    def reset(self) -> None:
        self._zi = None
        self.hc = HarmonicCanceller(self.fs, **self._kw)

    def push(self, block) -> np.ndarray:
        x = np.asarray(block, dtype=np.float64).ravel()
        if not x.size:
            return x
        if self._zi is None:                     # 첫 표본으로 상태를 맞춘다 — DC 계단 방지
            self._zi = sps.sosfilt_zi(self._hp) * x[0]
        xh, self._zi = sps.sosfilt(self._hp, x, zi=self._zi)
        pli = xh - self.hc.push(xh)
        return x - pli
