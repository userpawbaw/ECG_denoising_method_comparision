import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import analyze_seed_sweep as a


def test_holm_known_and_order():
    assert a.holm([0.01, 0.04, 0.03, 0.002]) == pytest.approx([0.03, 0.06, 0.06, 0.008])
    assert a.holm([1, 0, 0.05]) == pytest.approx([1, 0, 0.1])


@pytest.mark.parametrize('n', [4, 10])
def test_exact_wilcoxon_minimum(n):
    assert a.exact_wilcoxon(np.arange(1, n+1)) == 2 / 2**n
    assert a.exact_wilcoxon(np.ones(n)) == 2 / 2**n


def test_tost_boundaries():
    assert a.tost(np.zeros(10)) == 0
    assert a.tost(np.full(10, 0.5)) == 0.5
    assert a.tost(np.full(10, -0.5)) == 0.5
    assert a.tost(np.ones(10)) == 1
    assert a.tost(np.linspace(-0.1, 0.1, 10)) < 0.05
    assert a.tost(np.linspace(-1, 2, 10)) >= 0.05


def test_fixed_family():
    assert len(a.REP_COMPARISONS) == 32
    assert {b for arm, b, m in a.REP_COMPARISONS if arm == 'm06_cond_l6'} == {'m06_l6'}


def test_incomplete_run_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        a.replication_analysis(tmp_path)
    assert not (tmp_path / 'analysis.csv').exists()


def test_complete_analysis_and_environment_guard(tmp_path):
    import json
    import csv
    for arm in ('m06_l1', *a.REP_ARMS):
        for seed in range(10):
            p = tmp_path / f'{arm}__s{seed}' / 'summary.json'
            p.parent.mkdir()
            p.write_text(json.dumps(dict(arm=arm, seed=seed, source='mitdb',
                env=dict(device='cpu', amp=False, torch='test', python='test',
                         platform='test', threads=4, git_commit='test'),
                fixed_snr_imp_scaled=seed, band_15_20=seed)))
    assert a.replication_analysis(tmp_path) == 0
    rows = list(csv.DictReader((tmp_path / 'analysis.csv').open()))
    assert len(rows) == 32
    assert all(r['verdict'] == '동등' for r in rows)
    p = tmp_path / 'm06_l1__s0' / 'summary.json'
    r = json.loads(p.read_text())
    r['env']['torch'] = 'other'
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError, match='mixed runs'):
        a.replication_analysis(tmp_path)
