"""캐시가 **숫자를 바꾸지 않는다**는 것을 고정한다.

학습 한 판이 `FrontEnd` 를 수만 번 부르는데, 종래에는 호출마다 필터를 다시
설계하고(`butter`·`iirnotch`) 참조 창을 다시 필터링했다. 둘 다 **인자만으로
정해지는 순수 계산**이라 캐시할 수 있고, 실측 2.31 배가 빨라졌다.

**그런데 이 종류의 최적화가 정확히 F-9 가 일어난 자리다** — 데이터 파이프라인이
1 비트라도 달라지면 그 전에 학습한 체크포인트가 전부 무효가 되는데, 체크포인트만
봐서는 드러나지 않는다. 그래서 「빨라졌다」가 아니라 **「같은 값이다」**를 잰다.
"""
import numpy as np
import pytest
from scipy import signal as sps

from ecgdn.config import DEFAULT_FE
from ecgdn.data.dataset import ECGDenoiseDataset
from ecgdn.data.nstdb import make_banks
from ecgdn.data.sources import get_source, resolve_source_kind
from ecgdn.methods.frontend import FrontEnd, _butter_sos, _notch_sos


def test_cached_filter_design_equals_a_fresh_design():
    """캐시된 계수가 그때그때 설계한 것과 **비트 단위로** 같다."""
    for order, wn, btype in [(4, 0.5 / 125, "highpass"), (4, 100.0 / 125, "lowpass"),
                             (2, 0.3, "highpass"), (6, 0.8, "lowpass")]:
        fresh = sps.butter(order, wn, btype=btype, output="sos")
        assert np.array_equal(_butter_sos(order, wn, btype), fresh)
    for f0, q, fs in [(60.0, 30.0, 250.0), (120.0, 30.0, 250.0), (50.0, 20.0, 360.0)]:
        b, a = sps.iirnotch(f0, q, fs)
        assert np.array_equal(_notch_sos(f0, q, fs), sps.tf2sos(b, a))


def test_design_cache_hands_out_writable_copies():
    """캐시본을 그대로 주면 두 곳이 같은 배열을 공유한다.

    `sosfiltfilt` 는 쓰기 가능한 버퍼를 요구하므로 읽기 전용으로 잠글 수도 없다.
    그래서 복사본을 준다 — 한쪽을 고쳐도 다음 호출이 오염되지 않아야 한다.
    """
    a = _butter_sos(4, 0.004, "highpass")
    assert a.flags.writeable
    a[0, 0] = 12345.0
    b = _butter_sos(4, 0.004, "highpass")
    assert b[0, 0] != 12345.0
    assert np.array_equal(b, sps.butter(4, 0.004, btype="highpass", output="sos"))


def test_frontend_output_is_stable_across_calls():
    """같은 입력에 같은 출력. 캐시가 상태를 흘리지 않는다."""
    fe = FrontEnd()
    rng = np.random.default_rng(0)
    for n in (256, 1024, 2024, 5000):
        x = rng.standard_normal(n)
        first = fe(x, 250.0)
        assert np.array_equal(first, fe(x, 250.0))
        fe(rng.standard_normal(n), 360.0)          # 다른 fs 를 사이에 끼운다
        assert np.array_equal(first, fe(x, 250.0))


@pytest.mark.skipif(resolve_source_kind("mitdb") != "mitdb",
                    reason="data/raw/mitdb 가 없다")
def test_reference_window_cache_does_not_change_any_sample():
    """참조 창 캐시를 켠 것과 끈 것이 **모든 샘플에서** 같다.

    epoch 을 바꿔 가며 본다 — 참조는 잡음과 무관하므로 `set_epoch` 이 캐시를
    비우지 않는데, 만약 참조가 실제로는 잡음에 의존했다면 여기서 갈라진다.
    0 → 1 → 0 으로 되돌아오는 것도 확인한다.
    """
    src = get_source("mitdb", n_train=18, n_val=4, n_test=22)
    banks = make_banks("train", "data/raw/nstdb")
    kw = dict(banks=banks, salt=("train", 0), win=1024, hop=512,
              snr_range=(-5.0, 20.0), max_per_record=8, frontend=True)
    on = ECGDenoiseDataset(src, "train", cache_ref=True, **kw)
    off = ECGDenoiseDataset(src, "train", cache_ref=False, **kw)
    idx = [0, 1, 7, len(on) // 2, len(on) - 1]
    for ep in (0, 1, 0):
        on.set_epoch(ep)
        off.set_epoch(ep)
        for i in idx:
            y_on, x_on, _ = on[i]
            y_off, x_off, _ = off[i]
            assert np.array_equal(y_on, y_off), f"입력이 갈렸다 (epoch {ep}, i {i})"
            assert np.array_equal(x_on, x_off), f"참조가 갈렸다 (epoch {ep}, i {i})"


@pytest.mark.skipif(resolve_source_kind("mitdb") != "mitdb",
                    reason="data/raw/mitdb 가 없다")
def test_reference_cache_is_keyed_by_window_not_shared():
    """캐시 키가 window 인덱스다 — 서로 다른 window 가 같은 참조를 받으면 안 된다."""
    src = get_source("mitdb", n_train=18, n_val=4, n_test=22)
    ds = ECGDenoiseDataset(src, "train", banks=make_banks("train", "data/raw/nstdb"),
                           salt=("train", 0), win=1024, hop=512,
                           snr_range=(-5.0, 20.0), max_per_record=8, frontend=True)
    a = ds._ref_window(0, *_seg(ds, 0))
    b = ds._ref_window(1, *_seg(ds, 1))
    assert not np.array_equal(a, b)
    assert np.array_equal(ds._ref_window(0, *_seg(ds, 0)), a)


def _seg(ds, i):
    """`raw_item` 이 만드는 것과 같은 seg·fs·margin 을 재현한다."""
    wi = ds.index[i]
    rec = ds.source.get(wi.record)
    m = int(round(ds.fe_margin_s * rec.fs)) if (ds.frontend or ds.ref_frontend) else 0
    lo, hi = wi.start - m, wi.start + ds.win + m
    pad_l, pad_r = max(0, -lo), max(0, hi - rec.x.size)
    seg = rec.x[max(lo, 0):min(hi, rec.x.size)]
    if pad_l or pad_r:
        seg = np.pad(seg, (pad_l, pad_r), mode="edge")
    return seg, rec.fs, m


def test_resunet_default_is_unchanged_by_the_depth_knob():
    """`n_blocks` 를 추가해도 **기본값에서는 종래와 완전히 같은 망**이어야 한다.

    기존 체크포인트가 전부 이 구조로 학습됐다. 파라미터 수나 수용영역이 1 이라도
    달라지면 그 체크포인트들이 무효가 된다(F-9).
    """
    from ecgdn.models.resunet1d import ResUNet1D
    m = ResUNet1D()
    assert m.n_params() == 976_489
    assert m.receptive_field_samples == 887
    assert ResUNet1D(n_blocks=1).n_params() == m.n_params()


def test_depth_knob_deepens_and_widens_the_receptive_field():
    """깊이를 늘리면 수용영역도 같이 는다 — 둘은 conv 망에서 분리되지 않는다."""
    from ecgdn.models.resunet1d import ResUNet1D
    rf = [ResUNet1D(n_blocks=n).receptive_field_samples for n in (1, 2, 3)]
    assert rf == sorted(rf) and len(set(rf)) == 3, rf
