import numpy as np

from ecgdn.data.mixer import mix_at_snr
from ecgdn.data.noise import awgn
from ecgdn.data.synthetic import synth_ecg
from ecgdn.eval.engine import evaluate, evaluate_many, to_long_frame, trim_guard
from ecgdn.methods.frontend import FrontEnd
from ecgdn.utils import rng


def _case(snr=10.0, dur=90.0):
    s = synth_ecg(dur, seed=21)
    y, _, _ = mix_at_snr(s.x, awgn(len(s.x), s.fs, rng("eng", snr)), snr)
    return s, y


def test_identity_gives_zero_improvement():
    s, y = _case()
    m = evaluate(s.x, y, y, s.fs, r_peaks_ref=s.r_peaks)
    assert abs(m["snr_in"] - 10.0) < 0.5
    assert abs(m["snr_imp_strict"]) < 1e-9


def test_perfect_output():
    s, y = _case()
    m = evaluate(s.x, y, s.x.copy(), s.fs, r_peaks_ref=s.r_peaks)
    assert m["rmse"] == 0.0
    assert abs(m["cc"] - 1.0) < 1e-12
    assert m["se"] > 0.99 and m["ppv"] > 0.99
    assert abs(m["qrs_dur_err_ms"]) < 1e-9


def test_guard_slice():
    sl = trim_guard(30000, 250.0, 5.0)
    assert sl.start == 1250 and sl.stop == 30000 - 1250
    assert trim_guard(100, 250.0, 5.0) == slice(0, 100)   # 너무 짧으면 전체


def test_evaluate_many_and_long_frame():
    s, y = _case()
    methods = {"M00": (lambda a, fs, ctx=None: np.asarray(a, dtype=float)),
               "M_FE": FrontEnd()}
    for k, v in methods.items():
        if not hasattr(v, "name"):
            pass
    res = evaluate_many(s.x, y, methods, s.fs, r_peaks_ref=s.r_peaks, do_morph=False)
    assert set(res) == {"M00", "M_FE"}
    assert res["M_FE"]["snr_imp_scaled"] > res["M00"]["snr_imp_scaled"]
    assert res["M00"]["rtf"] >= 0.0
    df = to_long_frame([{"record": "syn", "method": k, "metrics": v} for k, v in res.items()])
    assert set(df.columns) >= {"record", "method", "metric", "value"}
    assert len(df) > 10
