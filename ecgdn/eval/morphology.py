"""ECG morphology 지표 (docs/02_procedure.md STEP 07).

정렬 규약 (docs/00_review.md A-9):
    beat 잘라내기는 **항상 clean reference 의 R-peak 위치**로 양쪽 모두 수행한다.
    denoised 신호의 자체 R-peak 로 정렬하면 timing error 가 상쇄되어 사라진다.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..config import BEAT_POST_MS, BEAT_PRE_MS

__all__ = [
    "beat_matrix", "beat_template", "delineate_qrs", "peak_amplitude",
    "metrics_morph", "half_sample_consistency", "hsc_to_snr_db", "aligned_beats",
]


def beat_matrix(x: np.ndarray, r_peaks: np.ndarray, fs: float,
                pre_ms: float = BEAT_PRE_MS, post_ms: float = BEAT_POST_MS
                ) -> tuple[np.ndarray, np.ndarray]:
    """R-peak 정렬 beat 행렬. 경계에서 잘리는 beat 는 제외.

    Returns
    -------
    (beats, used_peaks) : beats (K, L), used_peaks (K,)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    r = np.asarray(r_peaks, dtype=int).ravel()
    pre = int(round(pre_ms * 1e-3 * fs))
    post = int(round(post_ms * 1e-3 * fs))
    ok = (r - pre >= 0) & (r + post < x.size)
    r = r[ok]
    if r.size == 0:
        return np.empty((0, pre + post + 1)), r
    idx = r[:, None] + np.arange(-pre, post + 1)[None, :]
    return x[idx], r


def beat_template(x: np.ndarray, r_peaks: np.ndarray, fs: float, **kw) -> np.ndarray:
    b, _ = beat_matrix(x, r_peaks, fs, **kw)
    return b.mean(axis=0) if b.size else np.empty(0)


def peak_amplitude(x: np.ndarray, r_peaks: np.ndarray, fs: float,
                   search_ms: float = 40.0) -> np.ndarray:
    """각 R-peak 근방 +-search_ms 에서의 부호 있는 극값.

    미세한 타이밍 이동과 무관하게 '진폭이 눌렸는가' 만 잰다.
    (타이밍은 rpeak_mae_ms 가 따로 담당)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    r = np.asarray(r_peaks, dtype=int).ravel()
    w = max(1, int(round(search_ms * 1e-3 * fs)))
    out = []
    for p in r:
        a, b = max(0, p - w), min(x.size, p + w + 1)
        seg = x[a:b]
        if seg.size == 0:
            out.append(np.nan); continue
        out.append(float(seg[int(np.argmax(np.abs(seg)))]))
    return np.asarray(out, dtype=np.float64)


def delineate_qrs(x: np.ndarray, r_peaks: np.ndarray, fs: float
                  ) -> tuple[np.ndarray, float]:
    """QRS onset~offset 폭 [ms] 배열과 성공률을 반환.

    실패한 beat 는 NaN. 성공률도 반드시 함께 리포트한다 (docs/02_procedure.md STEP 07).
    """
    r = np.asarray(r_peaks, dtype=int).ravel()
    if r.size < 3:
        return np.full(r.size, np.nan), 0.0
    try:
        import neurokit2 as nk
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, info = nk.ecg_delineate(np.asarray(x, dtype=np.float64).ravel(),
                                       rpeaks=r, sampling_rate=int(round(fs)),
                                       method="dwt")
        on = np.asarray(info.get("ECG_R_Onsets", []), dtype=np.float64)
        off = np.asarray(info.get("ECG_R_Offsets", []), dtype=np.float64)
    except Exception:
        return np.full(r.size, np.nan), 0.0

    n = min(on.size, off.size)
    if n == 0:
        return np.full(r.size, np.nan), 0.0
    dur = (off[:n] - on[:n]) / fs * 1e3
    # 생리학적으로 불가능한 값은 실패로 처리 (20~250 ms 밖)
    dur = np.where((dur > 20.0) & (dur < 250.0), dur, np.nan)
    rate = float(np.mean(np.isfinite(dur)))
    return dur, rate


def aligned_beats(x: np.ndarray, r_peaks: np.ndarray, fs: float,
                  upsample: int = 4, max_shift_ms: float = 24.0, n_iter: int = 3,
                  corr_thresh: float | None = None,
                  pre_ms: float = BEAT_PRE_MS, post_ms: float = BEAT_POST_MS
                  ) -> tuple[np.ndarray, np.ndarray, float]:
    """sub-sample 정렬된 beat 행렬.

    **왜 필요한가**
        fs = 250 Hz 에서 1 샘플 = 4 ms 다. QRS 는 sigma 가 10 ms 수준이라
        1 샘플의 정렬 오차만으로도 beat 잔차가 크게 부풀고, beat 평균 계열
        SNR 추정기가 수 dB 를 과소평가한다 (실측 확인됨: 20 dB -> 14 dB).
        따라서 upsample 배로 올린 뒤 정렬한다 (기본 4배 = 1 ms 해상도).

    corr_thresh
        None(기본) = 강건한 이상치 제거만 수행 (median - 5*1.4826*MAD).
        고정 임계값(예: 0.9)은 저 SNR 에서 '깨끗한 beat 만' 남겨 SNR 을 과대평가하므로
        추정기 용도로는 쓰지 않는다.

    Returns
    -------
    (beats, r_up, fs_up) : beats (K, L) @ fs_up, r_up 은 업샘플 인덱스
    """
    from scipy.signal import resample_poly

    x = np.asarray(x, dtype=np.float64).ravel()
    r = np.asarray(r_peaks, dtype=int).ravel()
    U = max(1, int(upsample))
    if U > 1:
        xu = resample_poly(x, U, 1)
        ru = r * U
        fsu = fs * U
    else:
        xu, ru, fsu = x, r.copy(), fs

    pre = int(round(pre_ms * 1e-3 * fsu))
    post = int(round(post_ms * 1e-3 * fsu))
    S = max(1, int(round(max_shift_ms * 1e-3 * fsu)))
    L = pre + post + 1

    ok = (ru - pre - S >= 0) & (ru + post + S < xu.size)
    ru = ru[ok]
    if ru.size < 6:
        b, used = beat_matrix(xu, ru, fsu, pre_ms, post_ms)
        return b, used, fsu

    wide_off = np.arange(-pre - S, post + S + 1)
    wide = xu[ru[:, None] + wide_off[None, :]]          # (K, L+2S)

    def slice_at(shifts: np.ndarray) -> np.ndarray:
        out = np.empty((ru.size, L))
        for i, sh in enumerate(shifts):
            j = S + int(sh)
            out[i] = wide[i, j:j + L]
        return out

    shifts = np.zeros(ru.size, dtype=int)
    cand = np.arange(-S, S + 1)
    for _ in range(max(1, n_iter)):
        beats = slice_at(shifts)
        tmpl = np.median(beats, axis=0)
        tmpl = tmpl - tmpl.mean()
        nt = float(np.sqrt(tmpl @ tmpl))
        if nt <= 0:
            break
        # 모든 후보 시프트에 대한 상관을 한 번에
        best = np.full(ru.size, -np.inf)
        newsh = shifts.copy()
        for sh in cand:
            j = S + sh
            seg = wide[:, j:j + L]
            segc = seg - seg.mean(axis=1, keepdims=True)
            den = np.sqrt((segc * segc).sum(1)) * nt
            with np.errstate(invalid="ignore", divide="ignore"):
                cc = (segc @ tmpl) / den
            upd = cc > best
            best[upd] = cc[upd]
            newsh[upd] = sh
        if np.array_equal(newsh, shifts):
            shifts = newsh
            break
        shifts = newsh

    beats = slice_at(shifts)
    r_al = ru + shifts
    # 형태가 크게 다른 beat 제외
    tmpl = np.median(beats, axis=0); tmpl = tmpl - tmpl.mean()
    bc = beats - beats.mean(axis=1, keepdims=True)
    den = np.sqrt((bc * bc).sum(1) * (tmpl @ tmpl))
    with np.errstate(invalid="ignore", divide="ignore"):
        cc = (bc @ tmpl) / den
    if corr_thresh is None:
        finite = cc[np.isfinite(cc)]
        if finite.size >= 6:
            med = float(np.median(finite))
            mad = float(np.median(np.abs(finite - med))) * 1.4826
            thr = med - 5.0 * max(mad, 1e-6)
        else:
            thr = -np.inf
    else:
        thr = float(corr_thresh)
    keep = np.isfinite(cc) & (cc >= thr)
    if keep.sum() >= 6:
        beats, r_al = beats[keep], r_al[keep]
    return beats, r_al, fsu


def half_sample_consistency(x: np.ndarray, r_peaks: np.ndarray, fs: float,
                            align: bool = True, **kw) -> float:
    """Half-sample consistency (docs/00_review.md C-5). **ground truth 불필요.**

    beat 를 홀/짝으로 나눠 각각 template 을 만들고 두 template 의 상관을 본다.
    잡음은 두 집합에 무상관으로 들어가므로 잡음이 줄수록 1 에 가까워진다.

    align=True 면 sub-sample 정렬을 먼저 한다 (기본). 정렬하지 않으면
    R-peak 의 1 샘플 지터가 그대로 '잡음' 으로 계산되어 값이 크게 낮아진다.
    """
    if align:
        b, _, _ = aligned_beats(x, r_peaks, fs, **kw)
    else:
        b, _ = beat_matrix(x, r_peaks, fs, **kw)
    if b.shape[0] < 6:
        return float("nan")
    to, te = b[1::2].mean(axis=0), b[0::2].mean(axis=0)
    to = to - to.mean(); te = te - te.mean()
    d = float(np.sqrt((to @ to) * (te @ te)))
    return float((to @ te) / d) if d > 0 else float("nan")


def hsc_to_snr_db(hsc: float, n_beats: int) -> float:
    """half-sample consistency 를 단일 beat SNR [dB] 로 환산한다.

    유도
    ----
    template_O = m + e_O,  template_E = m + e_E,  e_O ⟂ e_E,  P(e) = Pn/(N/2)
        corr = Ps / (Ps + Pn') = r' / (1 + r'),   r' = Ps/Pn'
      → r' = corr / (1 - corr)
      → 단일 beat SNR  r = r' / (N/2)

    가정: beat 간 morphology 가 반복되고 잡음이 beat 간 무상관.
    **clean reference 가 없어도 계산되므로 실측 Arduino 데이터에 그대로 쓴다.**
    """
    if not np.isfinite(hsc) or hsc <= 0.0 or hsc >= 1.0 or n_beats < 6:
        return float("nan")
    r_prime = hsc / (1.0 - hsc)
    r = r_prime / (n_beats / 2.0)
    return float(10.0 * np.log10(r)) if r > 0 else float("nan")


def metrics_morph(x_ref: np.ndarray, xhat: np.ndarray, fs: float,
                  r_peaks_ref: np.ndarray, do_delineate: bool = True
                  ) -> dict[str, float]:
    """morphology 지표 묶음. 양쪽 모두 r_peaks_ref 로 정렬."""
    out: dict[str, float] = {}
    x_ref = np.asarray(x_ref, dtype=np.float64).ravel()
    xhat = np.asarray(xhat, dtype=np.float64).ravel()
    r = np.asarray(r_peaks_ref, dtype=int).ravel()

    # --- R-peak 진폭
    a_ref = peak_amplitude(x_ref, r, fs)
    a_hat = peak_amplitude(xhat, r, fs)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(a_hat - a_ref) / np.abs(a_ref)
    out["r_amp_err_pct"] = float(np.nanmean(rel) * 100.0) if rel.size else float("nan")
    out["r_amp_ratio"] = float(np.nanmedian(a_hat / a_ref)) if rel.size else float("nan")

    # --- beat template 상관 (양쪽 동일 R 로 정렬)
    tr = beat_template(x_ref, r, fs)
    th = beat_template(xhat, r, fs)
    if tr.size and th.size == tr.size:
        a = tr - tr.mean(); b = th - th.mean()
        d = float(np.sqrt((a @ a) * (b @ b)))
        out["beat_cc"] = float((a @ b) / d) if d > 0 else float("nan")
    else:
        out["beat_cc"] = float("nan")

    # --- beat 단위 상관의 중앙값 (template 평균이 가리는 개별 beat 손상 포착)
    br, _ = beat_matrix(x_ref, r, fs)
    bh, _ = beat_matrix(xhat, r, fs)
    if br.shape == bh.shape and br.size:
        a = br - br.mean(axis=1, keepdims=True)
        b = bh - bh.mean(axis=1, keepdims=True)
        den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
        with np.errstate(invalid="ignore", divide="ignore"):
            cc = (a * b).sum(1) / den
        out["beat_cc_median"] = float(np.nanmedian(cc))
        out["beat_cc_p05"] = float(np.nanpercentile(cc, 5))
    else:
        out["beat_cc_median"] = out["beat_cc_p05"] = float("nan")

    # --- QRS duration
    if do_delineate:
        d_ref, s_ref = delineate_qrs(x_ref, r, fs)
        d_hat, s_hat = delineate_qrs(xhat, r, fs)
        n = min(d_ref.size, d_hat.size)
        if n:
            e = np.abs(d_hat[:n] - d_ref[:n])
            out["qrs_dur_ref_ms"] = float(np.nanmedian(d_ref[:n]))
            out["qrs_dur_hat_ms"] = float(np.nanmedian(d_hat[:n]))
            out["qrs_dur_err_ms"] = float(np.nanmean(e)) if np.any(np.isfinite(e)) else float("nan")
        else:
            out["qrs_dur_ref_ms"] = out["qrs_dur_hat_ms"] = out["qrs_dur_err_ms"] = float("nan")
        out["delineate_success_rate"] = float(min(s_ref, s_hat))
    else:
        for k in ("qrs_dur_ref_ms", "qrs_dur_hat_ms", "qrs_dur_err_ms",
                  "delineate_success_rate"):
            out[k] = float("nan")

    # --- ground truth 불필요 지표 (실측 평가와 동일 코드 경로 확인용)
    n_beats = float(aligned_beats(xhat, r, fs)[0].shape[0])
    out["n_beats_used"] = n_beats
    out["hsc_hat"] = half_sample_consistency(xhat, r, fs)
    out["hsc_ref"] = half_sample_consistency(x_ref, r, fs)
    out["hsc_snr_hat_db"] = hsc_to_snr_db(out["hsc_hat"], int(n_beats))
    return out
