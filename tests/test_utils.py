import numpy as np

from ecgdn.utils import derive_seed, rng, power, power_db, robust_scale


def test_derive_seed_deterministic():
    assert derive_seed("a", 1) == derive_seed("a", 1)
    assert derive_seed("a", 1) != derive_seed("a", 2)
    assert derive_seed("a", 1) != derive_seed("b", 1)
    assert 0 <= derive_seed("x") < 2 ** 32


def test_rng_reproducible():
    a = rng("exp", 100, 3).standard_normal(10)
    b = rng("exp", 100, 3).standard_normal(10)
    c = rng("exp", 100, 4).standard_normal(10)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_power_offset_invariant():
    x = np.random.default_rng(0).standard_normal(1000)
    assert abs(power(x) - power(x + 12.3)) < 1e-12
    assert abs(power(x) - np.var(x)) < 1e-12
    assert power(np.array([])) == 0.0


def test_power_db():
    assert abs(power_db(np.array([1.0, -1.0] * 100)) - 0.0) < 1e-9


def test_robust_scale_outlier_resistant():
    x = np.random.default_rng(1).standard_normal(10000)
    s0 = robust_scale(x)
    x2 = x.copy()
    x2[::500] += 1000.0          # 0.2% 이상치
    s1 = robust_scale(x2)
    assert abs(s1 - s0) / s0 < 0.10
