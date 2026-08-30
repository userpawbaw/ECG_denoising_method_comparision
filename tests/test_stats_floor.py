"""`compare_methods` 의 분해능 관문(O-14 재발 방지).

유의성 검정과 분해능 검사는 다른 질문이다. 전자는 "이 차이가 우연인가",
후자는 "이 차이를 잴 수는 있는가" 이고 **후자가 먼저다.** floor 의 0.7 배인
차이를 `p = 0.012` 만 보고 "우세" 로 적을 뻔한 사고가 실제로 있었다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ecgdn.eval.stats import compare_methods, load_floor


def _long(delta: float, n: int = 22, metric: str = "psd_logdist", seed: int = 0):
    """baseline 대비 일정한 차이를 갖는 long-format 표를 만든다.

    차이를 상수에 가깝게 두면 Wilcoxon 이 아주 작은 p 를 준다 — 즉
    **유의하지만 분해능 아래인 상황**을 재현할 수 있다.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(4.0, 0.5, n)
    rows = []
    for i in range(n):
        rows += [dict(record=f"R{i:02d}", method="BASE", metric=metric, value=base[i]),
                 dict(record=f"R{i:02d}", method="CAND", metric=metric,
                      value=base[i] + delta + rng.normal(0, abs(delta) * 0.05 + 1e-9))]
    return pd.DataFrame(rows)


def test_floor_absent_leaves_columns_off():
    """floor 를 안 주면 열 자체가 없다 — 기존 호출부가 그대로 돈다."""
    t = compare_methods(_long(0.5), "psd_logdist", "BASE")
    assert "floor_ratio" not in t.columns


def test_significant_but_below_floor_is_not_resolvable():
    """O-14 그 자체. p 는 작은데 분해능 아래인 경우."""
    t = compare_methods(_long(-0.73), "psd_logdist", "BASE",
                        floor={"psd_logdist": 0.9983})
    r = t.iloc[0]
    assert r["p_holm"] < 0.05, "이 테스트는 '유의한데 못 잰다' 를 재현해야 한다"
    assert r["floor_ratio"] < 1.0
    assert not r["resolvable"]


def test_above_floor_is_resolvable():
    t = compare_methods(_long(3.0), "psd_logdist", "BASE",
                        floor={"psd_logdist": 0.9983})
    r = t.iloc[0]
    assert r["floor_ratio"] > 1.0
    assert r["resolvable"]


def test_metric_without_floor_stays_nan_not_zero():
    """floor 부재를 0 으로 채우면 모든 차이가 '분해능 위' 가 된다.

    부재와 0 은 다르다. NaN 으로 남겨 표에서 눈에 띄게 한다.
    """
    t = compare_methods(_long(0.5, metric="snr_imp_scaled"), "snr_imp_scaled",
                        "BASE", floor={"psd_logdist": 0.9983})
    r = t.iloc[0]
    assert np.isnan(r["floor_p95"])
    assert np.isnan(r["floor_ratio"])


def test_axis_name_loads_the_right_floor():
    """축 이름으로 넘기면 그 축의 floor 를 읽는다.

    F-16 — floor 는 축마다 다르다. D1 의 `qrs_dur_err_ms` floor 는 D0 의
    44 배다. 축을 섞으면 판정이 뒤집힌다.
    """
    f0, f1 = load_floor("d0"), load_floor("d1")
    if not f0 or not f1:
        pytest.skip("floor 산출물이 없다")
    assert f1["qrs_dur_err_ms"] > 10 * f0["qrs_dur_err_ms"]


def test_load_floor_missing_axis_is_empty_not_error():
    assert load_floor("d9_nonexistent") == {}


# ---------------------------------------------------------------- 이상값 (F-22)
def test_gain_bias_is_compared_as_distance_from_one():
    """이상값이 1 인 지표를 부호차로 비교하면 판정이 뒤집힌다 (F-22).

    실제로 D0 에서 M06 0.9803 / M10 1.0266 이 1 의 반대편에 거의 대칭인데
    부호차로는 p=7e-06, r=+1.000(22 기록 전부)이 나왔다. 거리로는 p=0.965 다.
    """
    from ecgdn.eval.stats import IDEAL
    assert IDEAL.get("gain_bias") == 1.0

    rows = []
    for i in range(12):
        # 두 방법이 1 의 반대편에 같은 거리로 있다 — 판정은 n.s. 여야 한다
        rows += [dict(record=f"R{i}", method="A", metric="gain_bias", value=0.98),
                 dict(record=f"R{i}", method="B", metric="gain_bias", value=1.02)]
    df = pd.DataFrame(rows)

    signed = compare_methods(df, "gain_bias", "A", use_ideal=False).iloc[0]
    dist = compare_methods(df, "gain_bias", "A").iloc[0]
    assert signed["delta_mean"] == pytest.approx(0.04), "전제: 부호차는 +0.04"
    assert dist["delta_mean"] == pytest.approx(0.0, abs=1e-9), \
        f"거리로는 차이가 없어야 한다: {dist['delta_mean']}"


def test_monotone_ideal_one_metrics_are_left_alone():
    """`cc`·`beat_cc` 는 이상값이 1 이지만 1 을 넘지 못해 단조다 — 건드리면 안 된다."""
    from ecgdn.eval.stats import IDEAL
    for m in ("cc", "beat_cc", "beat_cc_median"):
        assert m not in IDEAL, f"{m} 을 거리 변환하면 오히려 읽기 어려워진다"
