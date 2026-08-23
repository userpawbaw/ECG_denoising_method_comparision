"""잡음 모델 (docs/02_procedure.md STEP 03).

모든 생성 함수는 `(n, fs, gen) -> np.ndarray` 이고, **단위분산으로 정규화해서** 반환한다.
실제 크기 조절은 mixer 가 SNR 로 한다.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import signal as sps

from ..utils import rng as make_rng

__all__ = [
    "unit_var", "awgn", "pli", "baseline_synth", "emg_synth", "motion_synth",
    "impulse", "NOISE_FNS", "make_noise", "mixed_noise",
]


_MIN_STD = 1e-12


def unit_var(x: np.ndarray) -> np.ndarray:
    """평균 0, 분산 1 로 정규화.

    분산이 0 이면 그대로 반환한다. 대신 이 파일의 생성기들이 '항상 분산 > 0' 을
    보장한다 (아래 각 함수의 방어 로직). 그 계약이 깨지면 SNR 스케일링이 불가능해져
    학습이 중단된다.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    s = x.std()
    return x / s if s > _MIN_STD else x


# ------------------------------------------------------------------ 개별 잡음
def awgn(n: int, fs: float, gen: np.random.Generator) -> np.ndarray:
    """가산 백색 가우시안 잡음. ADC quantization / 열잡음의 1차 근사."""
    return unit_var(gen.standard_normal(n))


def pli(n: int, fs: float, gen: np.random.Generator, f0: float = 60.0,
        n_harm: int = 3, drift_hz: float = 0.2) -> np.ndarray:
    """전원선 간섭 (power-line interference).

    한국 전력망 = 60 Hz. 주파수를 f0 +- drift 로 아주 느리게 흔들어 실제와 비슷하게 만든다.
    고조파 진폭비 1 : 0.3 : 0.1.

    주의: fs=250 Hz 이므로 Nyquist 는 125 Hz 다. 3고조파(180 Hz)는 aliasing 되어
    250-180 = 70 Hz 로 접힌다. 이는 실제 취득계에서도 anti-alias 필터가 부족하면
    동일하게 일어나는 현상이라 그대로 둔다. (해석 시 이 점을 명시할 것)
    """
    t = np.arange(n) / fs
    # 매우 느린 주파수 드리프트
    ph_drift = np.cumsum(drift_hz * gen.standard_normal(n) / fs) * 2 * np.pi
    ph_drift = sps.sosfiltfilt(sps.butter(2, 0.05, fs=fs, output="sos"), ph_drift) \
        if n > 30 else ph_drift
    amps = np.array([1.0, 0.3, 0.1])[:n_harm]
    x = np.zeros(n)
    for h, a in enumerate(amps, start=1):
        x += a * np.sin(2 * np.pi * f0 * h * t + h * ph_drift + gen.uniform(0, 2 * np.pi))
    return unit_var(x)


def baseline_synth(n: int, fs: float, gen: np.random.Generator,
                   resp_hz: tuple[float, float] = (0.2, 0.35)) -> np.ndarray:
    """Baseline wander: 0.05-0.5 Hz 대역제한 잡음 + 호흡성 정현파."""
    x = gen.standard_normal(n + 2000)
    sos = sps.butter(4, [0.05, 0.5], btype="bandpass", fs=fs, output="sos")
    x = sps.sosfiltfilt(sos, x)[1000:1000 + n]
    t = np.arange(n) / fs
    fr = gen.uniform(*resp_hz)
    x = unit_var(x) + 1.5 * np.sin(2 * np.pi * fr * t + gen.uniform(0, 2 * np.pi))
    return unit_var(x)


def emg_synth(n: int, fs: float, gen: np.random.Generator,
              band: tuple[float, float] = (20.0, 110.0),
              burst_rate_hz: float = 0.5) -> np.ndarray:
    """근전도(muscle artifact): 대역통과 백색잡음 x 랜덤 burst 포락선.

    ECG 와 주파수가 겹치는 것이 핵심 성질이라, 대역을 QRS 대역까지 내린다.
    """
    hi = min(band[1], 0.45 * fs)
    x = gen.standard_normal(n + 2000)
    sos = sps.butter(4, [band[0], hi], btype="bandpass", fs=fs, output="sos")
    x = sps.sosfiltfilt(sos, x)[1000:1000 + n]

    # burst 포락선: 포아송 시점에 지수 감쇠 + 상승
    env = np.full(n, 0.15)
    n_burst = max(1, gen.poisson(burst_rate_hz * n / fs))
    for _ in range(n_burst):
        c = gen.integers(0, n)
        dur = int(gen.uniform(0.15, 1.2) * fs)
        k = np.arange(dur)
        shape = np.sin(np.pi * k / max(dur, 1)) ** 1.5
        i0, i1 = c, min(c + dur, n)
        env[i0:i1] += gen.uniform(0.6, 1.8) * shape[: i1 - i0]
    return unit_var(x * env)


def motion_synth(n: int, fs: float, gen: np.random.Generator,
                 rate_hz: float = 0.25) -> np.ndarray:
    """전극 움직임(electrode motion): 저주파 대진폭 스텝 + 지수 감쇠.

    ECG morphology 와 주파수가 겹쳐 단순 필터링으로 제거하기 어려운 종류.
    """
    x = np.zeros(n)
    n_ev = max(1, gen.poisson(rate_hz * n / fs))
    # **방어**: 이벤트가 창 끝에 붙으면 상승항 (1-exp(-t/0.03)) 이 0 이라 기여가 없다.
    # n_ev=1 이고 c 가 마지막 샘플이면 전체가 0 이 되어 SNR 스케일링이 불가능해진다.
    # 실측: 약 2600 window 당 1 회 발생해 학습이 에폭 6 에서 중단됐다.
    lead = max(2, int(round(0.12 * fs)))
    hi = max(1, n - lead)
    for _ in range(n_ev):
        c = int(gen.integers(0, hi))
        tau = gen.uniform(0.2, 1.5)
        amp = gen.normal(0, 1.0)
        if abs(amp) < 1e-3:
            amp = 1e-3 if amp >= 0 else -1e-3
        seg = np.arange(n - c) / fs
        x[c:] += amp * (1 - np.exp(-seg / 0.03)) * np.exp(-seg / tau)
    sos = sps.butter(4, 20.0, btype="low", fs=fs, output="sos")
    out = sps.sosfiltfilt(sos, x)
    if float(np.std(out)) <= _MIN_STD:                      # 최후 방어
        out = sps.sosfiltfilt(sos, gen.standard_normal(n))
    return unit_var(out)


def impulse(n: int, fs: float, gen: np.random.Generator,
            rate_hz: float = 0.6) -> np.ndarray:
    """순간적 큰 spike (접촉 불량 / 취득 artifact).

    사용자가 관측한 오실로스코프 화면의 날카로운 spike 를 모사한다.
    백색잡음이 아니라 sparse heavy-tail 이라 별도 조건으로 평가한다. (EXP-B)
    """
    x = np.zeros(n)
    n_ev = max(1, gen.poisson(rate_hz * n / fs))
    for _ in range(n_ev):
        c = int(gen.integers(0, n))
        amp = gen.normal(0, 1.0) * gen.choice([3.0, 6.0, 10.0])
        w = max(2, int(gen.uniform(0.002, 0.012) * fs))
        k = np.arange(-3 * w, 3 * w + 1)
        shape = np.exp(-np.abs(k) / w) * np.cos(np.pi * k / (2 * w))
        i0, i1 = max(0, c + k[0]), min(n, c + k[-1] + 1)
        x[i0:i1] += amp * shape[(i0 - c - k[0]):(i1 - c - k[0])]
    if float(np.std(x)) <= _MIN_STD:                        # 최후 방어
        x = gen.standard_normal(n)
    return unit_var(x)


NOISE_FNS: dict[str, Callable[..., np.ndarray]] = {
    "awgn": awgn,
    "pli": pli,
    "bw_synth": baseline_synth,
    "ma_synth": emg_synth,
    "em_synth": motion_synth,
    "impulse": impulse,
}

# 실제 NSTDB 기록(bw/ma/em)은 nstdb.NoiseBank 가 같은 인터페이스로 제공한다.
SYNTH_KINDS = tuple(NOISE_FNS)


def make_noise(kind: str, n: int, fs: float, gen: np.random.Generator,
               banks: dict[str, "object"] | None = None, **kw) -> np.ndarray:
    """이름으로 잡음을 만든다. banks 가 주어지면 실제 NSTDB 기록을 우선 사용."""
    if banks and kind in banks:
        return unit_var(banks[kind].sample(n, gen))
    if kind not in NOISE_FNS:
        raise KeyError(f"unknown noise kind '{kind}'. available: {sorted(NOISE_FNS)}"
                       + (f" + banks {sorted(banks)}" if banks else ""))
    return NOISE_FNS[kind](n, fs, gen, **kw)


def mixed_noise(n: int, fs: float, gen: np.random.Generator,
                kinds: Sequence[str] | None = None,
                n_pick: tuple[int, int] = (1, 3),
                banks: dict[str, "object"] | None = None) -> tuple[np.ndarray, dict[str, float]]:
    """1~3 종을 랜덤 가중 합성한다. 학습 데이터 생성기의 기본 경로.

    Returns
    -------
    (noise, weights) : noise 는 단위분산. weights 는 각 성분의 파워 비율(합 1).
    """
    pool = list(kinds) if kinds else (list(SYNTH_KINDS) + (list(banks) if banks else []))
    k = int(gen.integers(n_pick[0], n_pick[1] + 1))
    k = min(k, len(pool))
    picked = list(gen.choice(pool, size=k, replace=False))
    # Dirichlet 로 파워 비중 배분 (진폭이 아니라 파워 기준)
    w = gen.dirichlet(np.ones(k) * 1.2)
    out = np.zeros(n)
    for name, wi in zip(picked, w):
        out += np.sqrt(wi) * unit_var(make_noise(name, n, fs, gen, banks=banks))
    if float(np.std(out)) <= _MIN_STD:                      # 계약: 항상 분산 > 0
        out = gen.standard_normal(n)
    return unit_var(out), {str(nm): float(wi) for nm, wi in zip(picked, w)}
