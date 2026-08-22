import numpy as np

from ecgdn.data.dataset import ECGDenoiseDataset, build_eval_set, load_eval_set
from ecgdn.data.sources import SyntheticSource
from ecgdn.data.splits import MITDB_SPLIT, NOISE_SPLIT_FRAC, split_of


def _src():
    return SyntheticSource(n_train=2, n_val=1, n_test=2, dur_s=40.0, pvc_prob=0.1)


def test_split_disjoint():
    seen = set()
    for k, v in MITDB_SPLIT.items():
        assert not (seen & set(v))
        seen |= set(v)
    assert split_of("100") == "test"


def test_noise_split_disjoint():
    r = [NOISE_SPLIT_FRAC[k] for k in ("train", "val", "test")]
    assert r[0][1] <= r[1][0] and r[1][1] <= r[2][0]


def test_dataset_deterministic():
    a = ECGDenoiseDataset(_src(), "train", salt=7)
    b = ECGDenoiseDataset(_src(), "train", salt=7)
    c = ECGDenoiseDataset(_src(), "train", salt=8)
    ya, xa, ma = a[3]
    yb, xb, mb = b[3]
    yc, _, _ = c[3]
    assert np.array_equal(ya, yb) and np.array_equal(xa, xb)
    assert ma["scale"] == mb["scale"] and ma["snr"] == mb["snr"]
    assert not np.array_equal(ya, yc)


def test_dataset_shapes_and_denormalization():
    ds = ECGDenoiseDataset(_src(), "train", salt=0)
    y, x, m = ds[0]
    assert y.shape == (1, 1024) and x.shape == (1, 1024)
    d = ds.raw_item(0)
    # 역정규화하면 원래 mV 스케일로 복원되어야 한다
    rec = ds.source.get(d["record"])
    x_orig = rec.x[d["start"]:d["start"] + 1024]
    assert np.max(np.abs(d["x"] * d["scale"] - x_orig)) < 1e-4


def test_scale_uses_only_noisy():
    """clean 을 바꿔도 scale 이 바뀌면 안 된다 = clean 정보 누설 없음."""
    from ecgdn.utils import robust_scale
    ds = ECGDenoiseDataset(_src(), "train", salt=1)
    d = ds.raw_item(2)
    y_phys = d["y"] * d["scale"]
    assert abs(robust_scale(y_phys) - float(d["scale"])) < 1e-3


def test_eval_set_is_shared_and_reproducible(tmp_path):
    it1 = build_eval_set(_src(), "test", seg_s=20.0, snr_grid=(5, 10), n_seg_per_record=1)
    it2 = build_eval_set(_src(), "test", seg_s=20.0, snr_grid=(5, 10), n_seg_per_record=1)
    assert len(it1) == 2 * 2
    assert all(np.array_equal(a["y"], b["y"]) for a, b in zip(it1, it2))
    p = tmp_path / "ev.npz"
    from ecgdn.data.dataset import save_eval_set
    save_eval_set(it1, p)
    back = load_eval_set(p)
    assert len(back) == len(it1)
    assert np.array_equal(back[0]["y"], it1[0]["y"])
