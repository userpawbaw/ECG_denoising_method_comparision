"""전 파이프라인 공통 규격. 매직넘버는 전부 여기로 올린다.

근거: docs/01_design.md 2장.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------- 신호 규격
FS: float = 250.0          # 통일 샘플링 주파수 [Hz]
WIN: int = 1024            # 윈도우 길이 = 4.096 s (2^10 -> SWT level 5 안전)
HOP: int = 512             # 50% overlap (Hann^0.5 COLA)

FS_MITDB: float = 360.0    # MIT-BIH 원본
RESAMPLE_UP: int = 25      # 360 * 25/36 = 250
RESAMPLE_DOWN: int = 36

# ---------------------------------------------------------------- 실험 그리드
SNR_GRID: tuple[int, ...] = (-5, 0, 5, 10, 15, 20)
SNR_TRAIN_RANGE: tuple[float, float] = (-5.0, 20.0)

# ---------------------------------------------------------------- 평가 규격
# 평가 구간 규약 (실측으로 결정, docs/03_metric_floor.md 참조)
#   0.5 Hz zero-phase HPF 는 경계에서 수 초간 링잉한다. Kalman 은 수렴 시간이 필요하고
#   OLA/wavelet 은 패딩 경계 효과가 있다. 따라서 **모든 방법**에 대해 동일하게
#   구간 양끝 GUARD 초를 지표 계산에서 제외한다. (방법에는 전체 구간을 준다)
#   실측: guard 3 s 에서 40.9 dB, 5 s 에서 41.9 dB 로 포화 -> 5 s 채택.
EVAL_SEG_S: float = 60.0       # 평가 단위 구간 길이
EVAL_GUARD_S: float = 5.0      # 양끝 제외 길이

RPEAK_TOL_MS: float = 75.0     # R-peak 매칭 허용오차
BEAT_PRE_MS: float = 250.0     # beat 잘라내기 (R 기준 앞)
BEAT_POST_MS: float = 400.0    # beat 잘라내기 (R 기준 뒤)
PSD_BAND: tuple[float, float] = (1.0, 100.0)
BAND_EDGES: tuple[tuple[float, float], ...] = (
    (0.5, 4.0),    # baseline / ST
    (4.0, 15.0),   # P, T
    (15.0, 40.0),  # QRS 주 대역
    (40.0, 100.0), # 고주파 (대부분 noise)
)


@dataclass(frozen=True)
class FrontEndCfg:
    """모든 방법에 동일하게 적용되는 공통 front-end (docs/01_design.md 2.1)."""
    hp_hz: float = 0.5
    lp_hz: float = 100.0
    order: int = 4
    notch_hz: tuple[float, ...] = (60.0, 120.0)
    notch_q: float = 30.0
    auto_notch: bool = True      # PSD로 PLI 존재를 판정한 뒤에만 notch 적용
    pli_ratio_thresh: float = 10.0


@dataclass(frozen=True)
class SWTCfg:
    """SWT thresholding (docs/00_review.md B-1).

    k 는 level-dependent threshold 스케일: (D1, D2, D3, D4, D5).
    fs=250 기준 대역: D1 62.5-125, D2 31-62.5, D3 15.6-31, D4 7.8-15.6, D5 3.9-7.8 Hz.
    D3/D4 가 QRS 주 대역이므로 보수적으로 준다.
    """
    wavelet: str = "sym4"
    level: int = 5
    # sigma 추정 출처. **"d1" 또는 "d2" 가 정답이다.**
    #   level 별 MAD 는 D3~D5 에서 ECG 자체가 계수를 지배해 sigma 를 2~5 배 과대추정하고,
    #   그 결과 QRS 대역을 과도하게 잘라낸다 (실측: 18.7 dB -> 14.1 dB 로 악화).
    #   SWT(norm=False) 는 백색잡음의 계수 분산을 level 에 무관하게 보존하므로
    #   잡음이 지배적인 최고주파 band 하나에서 추정한 sigma 를 전 level 에 쓰는 것이 맞다.
    # 아래 기본값은 scripts/tune_swt.py 로 탐색한 결과다 (docs/05_swt_tuning.md).
    # 탐색 seed 와 검증 seed 를 분리했고, 검증 세트에서 +11.4 dB
    # (교과서 기본값 level-MAD + soft + k=1 은 -3.5 dB).
    # MIT-BIH 확보 후에는 TRAIN split 으로 반드시 재탐색할 것.
    sigma_source: str = "d2"          # 'd1' | 'd2' | 'min12' | 'level'
    k: tuple[float, ...] = (2.5, 2.0, 0.6, 0.4, 0.3)
    mode: str = "garrote"          # 'soft' | 'hard' | 'garrote'
    protect_qrs: bool = True
    protect_ms: float = 60.0       # R-peak 주변 보호 반경
    protect_rho: float = 0.3       # 보호 구간에서 threshold 를 rho 배로
    threshold_approx: bool = False # A5(근사계수)는 건드리지 않는다


@dataclass(frozen=True)
class TrainCfg:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 100
    warmup_frac: float = 0.05
    patience: int = 15
    grad_clip: float = 1.0
    seeds: tuple[int, ...] = (0, 1, 2)


# ---------------------------------------------------------------- 합성 ECG 기본 파라미터
# McSharry et al. ODE 모델. (theta[rad], a[amp], b[width])
# 순서: P, Q, R, S, T
@dataclass(frozen=True)
class ECGKernel:
    """Sameni 파라미터화의 ECG kernel.

    z(theta) = sum_i alpha_i * exp( -(wrap(theta - theta_i))^2 / (2 b_i^2) )

    이 형태에서 alpha_i 는 곧 i 번째 파형의 진폭(mV)이다.
    ODE 형태 zdot = -sum_i (alpha_i * w / b_i^2) * dtheta_i * exp(...) 와 동치이며,
    Sameni EKF 의 상태방정식과 정확히 같은 파라미터화다. (docs/02_procedure.md STEP 13)

    기본값: lead II 정상 파형. R 진폭 1.0 mV 기준.
    theta[rad] 0 = R peak. 60 bpm 에서 1 rad = 159 ms.
    """
    name: str = "normal"
    theta: tuple[float, ...] = (-1.0472, -0.2618, 0.0, 0.2618, 1.5708)   # P Q R S T
    alpha: tuple[float, ...] = (0.11, -0.10, 1.00, -0.20, 0.30)          # mV
    b: tuple[float, ...] = (0.20, 0.07, 0.06, 0.07, 0.30)                # rad


PVC_KERNEL = ECGKernel(
    name="pvc",
    # 넓고 반대 방향인 QRS + 반대 극성 T. P 파 없음.
    theta=(-0.35, 0.0, 0.45, 1.45),
    alpha=(-0.35, -0.95, 0.45, -0.35),
    b=(0.16, 0.15, 0.20, 0.35),
)


DEFAULT_FE = FrontEndCfg()
DEFAULT_SWT = SWTCfg()
DEFAULT_KERNEL = ECGKernel()
