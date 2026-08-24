"""Ground-truth 없는 실측 신호의 SNR 추정 (docs/00_review.md B-3, STEP 08).

**"내 Arduino 신호가 정말 15 dB인가?" 에 답하는 도구.**

세 가지 독립적인 추정기를 제공하고 세 값을 모두 리포트한다.
값이 크게 다르면 그 자체가 정보다 — 어떤 가정이 깨졌는지 알려준다.

가정과 그 실패 모드 (반드시 함께 보고할 것)
--------------------------------------------
(1) sub-sample 정렬 필수
    fs=250 Hz 에서 1 샘플 = 4 ms. QRS sigma 는 10 ms 수준이라 1 샘플 정렬 오차만으로도
    beat 잔차가 부풀어 SNR 을 6 dB 이상 과소평가한다 (실측: 20 dB -> 14 dB).
    -> `aligned_beats` 로 4배 업샘플 후 정렬한다. (해결됨)

(2) beat 간 잡음 무상관 가정
    beat 평균 계열(A, B)은 잡음이 beat 사이에 무상관이라고 가정한다.
    **AWGN 에서는 잘 맞지만, 근전도 burst / 전극 움직임처럼 여러 beat 에 걸쳐
    상관을 갖는 잡음에서는 SNR 을 크게 과대평가한다.**
    -> `beat_resid_lag1_corr` 로 이 위반을 직접 측정해 함께 보고한다.
    -> 실무적으로는 공통 front-end(HPF) 적용 후에 추정하는 것을 권장한다.

(3) 생리학적 beat 변동의 바닥
    잡음이 전혀 없어도 RR 변동에 따라 P/T 위치가 실제로 움직인다.
    정렬을 해도 추정치는 무한대가 되지 않고 25 dB 부근에서 포화한다.
"""
from __future__ import annotations

import numpy as np

from .morphology import aligned_beats, hsc_to_snr_db
from .rpeak import detect_rpeaks
from ..utils import power

__all__ = ["estimate_snr_beat_residual", "estimate_snr_hsc", "estimate_snr_hsc_far",
           "estimate_snr_isoelectric", "beat_resid_lag1_corr", "per_beat_snr_db",
           "estimate_snr_all", "SNR_CEILING_DB", "SNR_CEILING_BY_AXIS"]

# 추정기가 이 값 이상을 보고하면 포화로 본다.
#
# **이 값은 데이터축마다 다르다.** 세 추정기 모두 "beat 가 반복되고 그 잔차가
# 잡음" 이라는 가정 위에 서 있는데, 실제 ECG 는 beat 마다 형태가 달라 그 변동이
# 전부 잡음으로 계산된다. 그래서 잡음이 하나도 없어도 추정치가 유한하다.
# 잡음 없는 신호에 추정기를 돌려 실측한 값이 아래다
# (`docs/04_snr_estimator_calibration{,_d1}.md` 의 '추정기의 천장' 절):
#
#   축   (A) beat-resid  (B) half-sample  (C) isoelectric
#   D0        19.6            19.8             16.8
#   D1        11.9            12.8             15.6
#
# D1 이 8 dB 낮다. 22.0 은 D0 조차 넘지 못하는 값이라 경고가 사실상 뜨지
# 않았다 — 실기록에서는 추정치가 이미 천장인데도 조용했다 (F-18).
SNR_CEILING_BY_AXIS = {"d0": 19.0, "d1": 12.0}
SNR_CEILING_DB = 22.0     # 하위 호환 기본값. 실데이터에는 d1 값을 넘길 것.
LAG1_WARN = 0.15          # beat 간 잔차 상관이 이보다 크면 (A),(B) 는 낙관적


def _get_beats(x, fs, r_peaks, reject: bool = False):
    """SNR 추정용 beat 행렬.

    reject=False (기본): **beat 를 버리지 않는다.**
        형태 기반 이상치 제거는 '잡음이 큰 beat' 를 골라서 버리기 때문에
        SNR 을 크게 과대평가한다 (실측: burst 형 EMG 5 dB -> 15 dB 로 왜곡).
        template 을 만들 때와 SNR 을 잴 때의 요구가 반대라는 점에 주의.
    """
    r = detect_rpeaks(x, fs) if r_peaks is None else np.asarray(r_peaks, dtype=int)
    if r.size < 6:
        return np.empty((0, 0)), r, fs
    return aligned_beats(x, r, fs, corr_thresh=None if reject else -np.inf)


def per_beat_snr_db(beats: np.ndarray) -> np.ndarray:
    """beat 별 SNR [dB] 분포.

    잡음이 burst 형(비정상)이면 SNR 을 단일 스칼라로 말하는 것 자체가 부정확하다.
    분포(중앙값/10퍼센타일)를 함께 보고한다.
    """
    n = beats.shape[0]
    if n < 6:
        return np.empty(0)
    m = beats.mean(axis=0)
    ps = float(np.var(m))
    res = beats - m
    pn = np.var(res, axis=1) * n / (n - 1)
    ps_c = ps - float(np.mean(pn)) / n
    with np.errstate(invalid="ignore", divide="ignore"):
        v = 10 * np.log10(np.maximum(ps_c, 1e-30) / np.maximum(pn, 1e-30))
    return v


def beat_resid_lag1_corr(beats: np.ndarray) -> float:
    """인접 beat 잔차 간 상관 (lag-1).

    0 에 가까우면 '잡음이 beat 간 무상관' 가정이 성립.
    유의하게 양수면 잡음이 여러 beat 에 걸쳐 상관을 가지며,
    beat 평균 계열 SNR 추정치는 **낙관적(과대)** 이다.
    """
    if beats.shape[0] < 8:
        return float("nan")
    res = beats - beats.mean(axis=0)
    a, b = res[:-1], res[1:]
    # 주의: beat 별 평균을 빼면 안 된다. 저주파 드리프트의 상관은 대부분
    #       '잔차의 DC 성분' 으로 나타나므로, 그것을 제거하면 진단이 무력해진다.
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        cc = (a * b).sum(1) / den
    return float(np.nanmedian(cc))


def _beats_or(x, fs, r_peaks, beats):
    return beats if beats is not None else _get_beats(x, fs, r_peaks)[0]


def estimate_snr_beat_residual(x: np.ndarray, fs: float,
                               r_peaks: np.ndarray | None = None,
                               beats: np.ndarray | None = None) -> tuple[float, dict]:
    """(A) beat 평균 잔차 기반.

        m   = mean_i(beat_i)                     <- signal 추정
        r_i = beat_i - m                         <- noise 추정
        P_n = mean_i var(r_i) * N/(N-1)          (평균 자신의 잡음 기여 보정)
        P_s = var(m) - P_n/N
    """
    beats = _beats_or(x, fs, r_peaks, beats)
    n = beats.shape[0]
    info = {"n_beats": int(n)}
    if n < 6:
        return float("nan"), info
    m = beats.mean(axis=0)
    pn = float(np.mean(np.var(beats - m, axis=1))) * n / (n - 1)
    ps = float(np.var(m)) - pn / n
    info.update(p_signal=ps, p_noise=pn)
    if pn <= 0 or ps <= 0:
        return float("nan"), info
    return float(10 * np.log10(ps / pn)), info


def estimate_snr_hsc(x: np.ndarray, fs: float,
                     r_peaks: np.ndarray | None = None,
                     beats: np.ndarray | None = None) -> tuple[float, dict]:
    """(B) half-sample consistency 기반 (docs/00_review.md C-5).

    홀수 beat 평균과 짝수 beat 평균의 상관으로부터 SNR 을 역산한다.
    잡음의 스펙트럼 형태에 대한 가정이 없어 (A) 보다 견고한 경우가 많다.
    """
    beats = _beats_or(x, fs, r_peaks, beats)
    n = beats.shape[0]
    if n < 6:
        return float("nan"), {"n_beats": int(n), "hsc": float("nan")}
    to, te = beats[1::2].mean(axis=0), beats[0::2].mean(axis=0)
    to = to - to.mean(); te = te - te.mean()
    d = float(np.sqrt((to @ to) * (te @ te)))
    hsc = float((to @ te) / d) if d > 0 else float("nan")
    return hsc_to_snr_db(hsc, n), {"n_beats": int(n), "hsc": hsc}


def estimate_snr_hsc_far(x: np.ndarray, fs: float,
                          r_peaks: np.ndarray | None = None,
                          beats: np.ndarray | None = None) -> tuple[float, dict]:
    """(B') 전반부/후반부 분할 HSC.

    (B) 는 인접 beat(홀/짝)를 나누므로, 잡음이 수 beat 규모로 상관을 가지면
    두 template 에 **같은** 잡음이 들어가 상관이 높게 나오고 SNR 이 과대평가된다.
    이 함수는 시간적으로 멀리 떨어진 전/후반부로 나눠 그 누출을 줄인다.

    `snr_hsc_db - snr_hsc_far_db` 가 크면 잡음이 beat 시간척도에서 상관을 갖는다는
    직접적인 증거다. (단, 장시간 morphology 드리프트도 같은 방향으로 작용한다)
    """
    beats = _beats_or(x, fs, r_peaks, beats)
    n = beats.shape[0]
    if n < 12:
        return float("nan"), {"n_beats": int(n)}
    h = n // 2
    t1, t2 = beats[:h].mean(axis=0), beats[h:].mean(axis=0)
    t1 = t1 - t1.mean(); t2 = t2 - t2.mean()
    d = float(np.sqrt((t1 @ t1) * (t2 @ t2)))
    hsc = float((t1 @ t2) / d) if d > 0 else float("nan")
    return hsc_to_snr_db(hsc, n), {"n_beats": int(n), "hsc_far": hsc}


def estimate_snr_isoelectric(x: np.ndarray, fs: float,
                             r_peaks: np.ndarray | None = None,
                             frac: tuple[float, float] = (0.62, 0.85)) -> tuple[float, dict]:
    """(C) 등전위(TP) 구간 기반. **beat 간 무상관 가정이 필요 없다.**

    각 RR 구간의 62~85 % 지점(T 파 종료 후 ~ 다음 P 파 이전)은 생리학적으로 평탄해야
    한다. 이 구간의 (1차 추세 제거 후) 분산을 잡음 파워로 본다.

    한계: 이 구간에 P 파나 잔여 T 파가 걸치면 잡음을 과대평가한다(=SNR 과소평가).
          또한 고주파 잡음에는 정확하지만 저주파 드리프트는 추세 제거로 일부 놓친다.
    """
    r = detect_rpeaks(x, fs) if r_peaks is None else np.asarray(r_peaks, dtype=int)
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(r) < 4:
        return float("nan"), {"n_seg": 0}
    segs = []
    for a, b in zip(r[:-1], r[1:]):
        rr = b - a
        i0, i1 = a + int(rr * frac[0]), a + int(rr * frac[1])
        if i1 - i0 >= 5 and i1 <= x.size:
            s = x[i0:i1]
            t = np.arange(len(s))
            segs.append(float(np.var(s - np.polyval(np.polyfit(t, s, 1), t))))
    if len(segs) < 4:
        return float("nan"), {"n_seg": len(segs)}
    pn = float(np.median(segs))
    ps = power(x) - pn
    info = {"n_seg": len(segs), "p_noise": pn, "p_signal": ps}
    if pn <= 0 or ps <= 0:
        return float("nan"), info
    return float(10 * np.log10(ps / pn)), info


def estimate_snr_all(x: np.ndarray, fs: float,
                     r_peaks: np.ndarray | None = None,
                     ceiling_db: float | None = None) -> dict[str, float]:
    """세 추정기 + 가정 위반 진단을 한 번에.

    해석 지침
    ---------
      - `beat_resid_lag1_corr` > 0.15  -> (A),(B) 는 낙관적. (C) 를 더 신뢰하고,
        잡음이 beat 시간척도에서 상관을 갖는다고 보고한다.
      - `snr_spread_db` 가 크다 (> 5 dB) -> 추정이 불안정. 단일 값으로 인용하지 말 것.
      - `ceiling_warning` = 1 -> "그 값 이상" 으로만 해석. 임계값은 `ceiling_db`
        이고, 생략하면 `SNR_CEILING_DB`(하위 호환 기본값)를 쓴다.
        **실데이터에는 `SNR_CEILING_BY_AXIS["d1"]` 을 넘겨야 한다** — 기본값은
        D0 를 보고 정한 것이라 실기록에서는 절대 뜨지 않는다 (F-18).
    """
    ceil = SNR_CEILING_DB if ceiling_db is None else float(ceiling_db)
    # R-peak 검출과 beat 정렬은 비싸므로 **한 번만** 계산해 모든 추정기에 넘긴다.
    if r_peaks is None:
        r_peaks = detect_rpeaks(x, fs)
    beats, _, _ = _get_beats(x, fs, r_peaks)
    pb = per_beat_snr_db(beats)
    # 형태 이상치 비율 (artifact 지표; 추정 자체에는 사용하지 않음).
    # 정렬은 이미 끝났으므로 상관만 다시 계산한다.
    frac_art = float("nan")
    if beats.shape[0] >= 6:
        tm = np.median(beats, axis=0); tm = tm - tm.mean()
        bc = beats - beats.mean(axis=1, keepdims=True)
        den = np.sqrt((bc * bc).sum(1) * (tm @ tm))
        with np.errstate(invalid="ignore", divide="ignore"):
            cc = (bc @ tm) / den
        fin = cc[np.isfinite(cc)]
        if fin.size >= 6:
            med = float(np.median(fin))
            mad = float(np.median(np.abs(fin - med))) * 1.4826
            frac_art = float(np.mean(fin < med - 5.0 * max(mad, 1e-6)))
    a, ia = estimate_snr_beat_residual(x, fs, r_peaks, beats=beats)
    b, ib = estimate_snr_hsc(x, fs, r_peaks, beats=beats)
    bf, ibf = estimate_snr_hsc_far(x, fs, r_peaks, beats=beats)
    c, ic = estimate_snr_isoelectric(x, fs, r_peaks)
    lag1 = beat_resid_lag1_corr(beats)
    vals = [v for v in (a, b, c) if np.isfinite(v)]
    return {
        "snr_beat_residual_db": a,
        "snr_hsc_db": b,
        "snr_hsc_far_db": bf,
        "snr_hsc_gap_db": float(b - bf) if np.isfinite(b) and np.isfinite(bf) else float("nan"),
        "snr_isoelectric_db": c,
        "snr_median_db": float(np.median(vals)) if vals else float("nan"),
        "snr_spread_db": float(np.max(vals) - np.min(vals)) if len(vals) > 1 else float("nan"),
        "snr_beat_p50_db": float(np.median(pb)) if pb.size else float("nan"),
        "snr_beat_p10_db": float(np.percentile(pb, 10)) if pb.size else float("nan"),
        "snr_beat_iqr_db": float(np.subtract(*np.percentile(pb, [75, 25]))) if pb.size else float("nan"),
        "artifact_beat_frac": float(frac_art),
        "beat_resid_lag1_corr": lag1,
        "n_beats": float(beats.shape[0]),
        "ceiling_warning": float(any(v >= ceil for v in vals)),
        "ceiling_db": ceil,
        "correlated_noise_warning": float(
            (np.isfinite(lag1) and lag1 > LAG1_WARN)
            or (np.isfinite(b) and np.isfinite(bf) and (b - bf) > 3.0)),
    }
