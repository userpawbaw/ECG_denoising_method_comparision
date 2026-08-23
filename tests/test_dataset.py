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
    """`ref_frontend=False` 면 타깃은 원본 그대로여야 한다."""
    ds = ECGDenoiseDataset(_src(), "train", salt=0, ref_frontend=False)
    y, x, m = ds[0]
    assert y.shape == (1, 1024) and x.shape == (1, 1024)
    d = ds.raw_item(0)
    # 역정규화하면 원래 mV 스케일로 복원되어야 한다
    rec = ds.source.get(d["record"])
    x_orig = rec.x[d["start"]:d["start"] + 1024]
    assert np.max(np.abs(d["x"] * d["scale"] - x_orig)) < 1e-4


def test_reference_is_band_limited_by_default():
    """기본 타깃은 **front-end 를 통과한** 참조다.

    MIT-BIH 는 clean 이 아니라 이미 기저선 변동을 담고 있어서, 원본을 정답으로
    두면 front-end 가 그 성분을 지울수록 정답에서 멀어져 SNR 이 떨어진다.
    실측에서 그 결과 무처리(M00)가 M04 보다 높게 나왔다 — 평가가 아무것도
    구분하지 못한다 (docs/02_procedure.md F-12).

    학습 타깃도 같은 대역이어야 한다. 다르면 신경망이 front-end 를 되돌리는
    법을 배운다.
    """
    from ecgdn.methods.frontend import FrontEnd

    ds = ECGDenoiseDataset(_src(), "train", salt=0)          # 기본 = True
    d = ds.raw_item(0)
    rec = ds.source.get(d["record"])
    x_raw = rec.x[d["start"]:d["start"] + 1024]
    x_got = d["x"] * d["scale"]

    # 원본과 같지 않아야 한다 (필터가 실제로 걸렸다)
    assert np.max(np.abs(x_got - x_raw)) > 1e-6

    # 그리고 front-end 를 통과한 참조와 일치해야 한다.
    # 경계 트랜지언트를 피하려고 margin 을 붙여 필터링하므로 동일 방식으로 비교한다.
    m = int(round(ds.fe_margin_s * rec.fs))
    lo, hi = d["start"] - m, d["start"] + 1024 + m
    pad_l, pad_r = max(0, -lo), max(0, hi - rec.x.size)
    seg = rec.x[max(lo, 0):min(hi, rec.x.size)]
    if pad_l or pad_r:
        seg = np.pad(seg, (pad_l, pad_r), mode="edge")
    x_ref = FrontEnd()(seg, rec.fs)[m:m + 1024]
    assert np.max(np.abs(x_got - x_ref)) < 1e-4


def test_eval_set_keeps_both_reference_and_raw():
    """평가 세트는 참조(x)와 원본(x_raw)을 모두 들고 있어야 한다.

    잡음은 원본에 섞고(취득계는 대역제한 전 신호를 본다), EXP-C 의 왜곡 측정도
    원본을 입력으로 준다 — 참조를 입력으로 주면 front-end 가 두 번 걸린다.
    """
    from ecgdn.data.dataset import build_eval_set

    items = build_eval_set(_src(), "test", seg_s=8.0, snr_grid=(10.0,),
                           n_seg_per_record=1)
    assert items, "평가 세트가 비었다"
    it = items[0]
    assert "x_raw" in it and it["x"].shape == it["x_raw"].shape
    assert np.max(np.abs(it["x"] - it["x_raw"])) > 1e-6, "참조가 대역제한되지 않았다"


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


def test_synthetic_records_have_distinct_morphology():
    """기록마다 morphology 가 달라야 한다.

    회귀 테스트: 모든 합성 기록이 같은 커널을 쓰면 record 단위 split 을 해도
    morphology 가 train/test 에 동일하게 존재해 **leakage 와 같은 효과**가 난다.
    실측으로 신경망이 생성기를 외워 성능이 12 dB 과대평가됐다.
    """
    import itertools

    from ecgdn.eval.morphology import beat_template

    src = SyntheticSource(n_train=4, n_val=1, n_test=4, dur_s=40.0)
    tmpl = {}
    for split in ("train", "test"):
        for nm in src.records(split):
            r = src.get(nm)
            t = beat_template(r.x, r.r_peaks, r.fs)
            tmpl[nm] = t - t.mean()
    ccs = []
    for a, b in itertools.combinations(tmpl, 2):
        ta, tb = tmpl[a], tmpl[b]
        d = np.sqrt((ta @ ta) * (tb @ tb))
        if d > 0:
            ccs.append(abs(float((ta @ tb) / d)))
    assert np.median(ccs) < 0.8, f"기록 간 morphology 가 너무 비슷하다 (median |cc| = {np.median(ccs):.3f})"
