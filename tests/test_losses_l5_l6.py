"""`L5`(SNR 균형) · `L6`(clean 보존) 손실.

두 손실 다 **손실 함수 혼자서는 완결되지 않는다** — L5 는 잡음 입력 `y`,
L6 는 clean 을 통과시킨 출력이 학습 루프에서 와야 한다. 그 연결이 끊긴 채
조용히 도는 것을 막는 것이 이 파일의 절반이다.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ecgdn.models.losses import (LOSS_NAMES, SNR_STRATA_DB, DenoiseLoss,  # noqa: E402
                                make_loss, per_window_loss, realized_snr_db,
                                snr_stratum_weights)


# ---------------------------------------------------------------- L5
def _batch(snrs_db, n=512, seed=0):
    """지정한 실제 SNR 을 갖는 window 배치를 만든다."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(len(snrs_db), 1, n, generator=g)
    x = x / x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()          # 신호 전력 1
    nz = torch.randn(len(snrs_db), 1, n, generator=g)
    nz = nz / nz.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
    amp = torch.tensor([10 ** (-s / 20) for s in snrs_db]).reshape(-1, 1, 1)
    return x, x + nz * amp


def test_stratum_weights_equalise_loss_contribution():
    """각 SNR 구간이 손실의 같은 몫을 갖게 하는 것이 L5 의 정의다."""
    x, y = _batch([-3, -1, 2, 7, 12, 17, 22, 30])
    pw = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.1])
    w = snr_stratum_weights(pw, y, x)
    snr = realized_snr_db(y, x)
    edges = (-1e9,) + SNR_STRATA_DB + (1e9,)
    shares = [float((pw * w)[(snr >= edges[i]) & (snr < edges[i + 1])].sum())
              for i in range(len(edges) - 1)]
    occupied = [s for s in shares if s > 0]
    assert len(occupied) >= 4, "구간이 여러 개 채워져야 의미 있는 검사다"
    assert max(occupied) - min(occupied) < 1e-6, f"기여가 안 맞는다: {occupied}"


def test_stratum_weights_sum_to_one_so_scale_is_preserved():
    """가중 평균이어야 손실 크기가 안 바뀌고 같은 LR 을 쓸 수 있다."""
    x, y = _batch([-3, 2, 7, 12, 17, 25])
    pw = torch.tensor([5.0, 3.0, 2.0, 1.0, 0.6, 0.3])
    w = snr_stratum_weights(pw, y, x)
    assert float(w.sum()) == pytest.approx(1.0, rel=1e-6)
    assert float(pw.min()) <= float((pw * w).sum()) <= float(pw.max())


def test_l5_reduces_to_l3_when_all_windows_share_a_stratum():
    """한 구간에만 있으면 균등 가중 — L5 는 L3 로 정확히 환원돼야 한다."""
    x, y = _batch([7.0, 7.2, 6.8, 7.1])
    xhat = x + 0.05 * torch.randn(x.shape, generator=torch.Generator().manual_seed(1))
    t5, _ = make_loss("L5")(xhat, x, y=y)
    t3, _ = make_loss("L3")(xhat, x)
    assert torch.allclose(t5, t3, rtol=1e-5), f"{float(t5)} vs {float(t3)}"


def test_realized_snr_is_measured_not_taken_from_metadata():
    """메타의 목표 SNR 과 실제 window SNR 은 다르다 (실측 −5~20 목표에 −9~80 실제)."""
    x, y = _batch([-5.0, 0.0, 10.0, 20.0])
    got = realized_snr_db(y, x)
    assert torch.allclose(got, torch.tensor([-5.0, 0.0, 10.0, 20.0]), atol=0.3), got


def test_stratum_edges_match_the_evaluation_grid():
    """학습 무게의 구간과 EXP-A 평가 격자가 어긋나면 결과를 대응시킬 수 없다."""
    assert SNR_STRATA_DB == (0.0, 5.0, 10.0, 15.0, 20.0)


def test_l5_without_y_raises_instead_of_silently_using_plain_mean():
    """조용히 L3 로 되돌아가면 ablation 표에 'L5' 라 적힌 L3 가 실린다."""
    x = torch.randn(4, 1, 128)
    with pytest.raises(ValueError, match="y"):
        make_loss("L5")(x, x)


def test_per_window_loss_batch_mean_matches_the_old_definition():
    """L1~L3 가 비트 단위로 보존돼야 기존 ablation 표를 다시 안 돌려도 된다."""
    torch.manual_seed(0)
    x = torch.randn(8, 1, 512)
    xhat = x + 0.1 * torch.randn(8, 1, 512)
    for name in ("L1", "L2", "L3"):
        loss = make_loss(name)
        ref = torch.nn.functional.mse_loss(xhat, x)
        if loss.w_mae:
            ref = ref + loss.w_mae * torch.nn.functional.l1_loss(xhat, x)
        if loss.w_diff:
            a = xhat[..., 1:] - xhat[..., :-1]
            b = x[..., 1:] - x[..., :-1]
            ref = ref + loss.w_diff * torch.nn.functional.mse_loss(a, b)
        got, _ = loss(xhat, x)
        assert torch.allclose(got, ref, rtol=1e-6), f"{name}: {float(got)} vs {float(ref)}"


# ---------------------------------------------------------------- L6
def test_l6_without_clean_pass_raises():
    loss = make_loss("L6")
    x = torch.randn(4, 1, 128)
    with pytest.raises(ValueError, match="xhat_clean"):
        loss(x, x)


def test_l6_clean_term_is_zero_for_a_perfect_passthrough():
    """clean 을 그대로 통과시키면 벌점이 0 이다 — 그것이 목표 상태다."""
    loss = make_loss("L6")
    x = torch.randn(8, 1, 128)
    _, parts = loss(x, x, xhat_clean=x[:2])
    assert float(parts["clean"]) == pytest.approx(0.0, abs=1e-12)


def test_l6_clean_term_penalises_touching_a_clean_signal():
    loss = make_loss("L6")
    x = torch.randn(8, 1, 128)
    _, quiet = loss(x, x, xhat_clean=x[:2])
    _, noisy = loss(x, x, xhat_clean=x[:2] + 0.1)
    assert float(noisy["total"]) > float(quiet["total"])


def test_l6_uses_only_a_fraction_of_the_batch():
    """clean forward 를 전 배치에 돌리면 학습이 2 배 느려진다."""
    assert 0.0 < make_loss("L6").clean_frac <= 0.5


# ---------------------------------------------------------------- 공통
def test_l6_clean_term_uses_the_same_composite_as_the_main_term():
    """clean 항이 MSE 뿐이면 손실의 1~4 % 밖에 안 된다 — 주 항의 90 % 가 MAE 라서다."""
    loss = make_loss("L6")
    x = torch.randn(8, 1, 256)
    xc = x[:2] + 0.05
    _, parts = loss(x, x, xhat_clean=xc)
    want = per_window_loss(xc, x[:2], loss.w_mae, loss.w_diff).mean()
    assert torch.allclose(parts["clean"], want, rtol=1e-6)


def test_l5_l6_are_l3_plus_one_change():
    """ablation 이 성립하려면 L3 에서 딱 한 가지만 달라야 한다."""
    l3, l5, l6 = (make_loss(n) for n in ("L3", "L5", "L6"))
    for a, b, changed in ((l3, l5, "snr_balance"), (l3, l6, "w_clean")):
        for attr in ("w_mse", "w_mae", "w_diff", "w_band"):
            assert getattr(a, attr) == getattr(b, attr), f"{attr} 이 달라졌다"
        assert getattr(a, changed) != getattr(b, changed)


def test_flags_are_off_for_the_older_presets():
    """L1~L4 에는 추가 비용이 붙지 않아야 한다."""
    for n in ("L1", "L2", "L3", "L4"):
        loss = make_loss(n)
        assert not loss.needs_noisy_input and not loss.needs_clean_pass


def test_every_preset_is_constructible():
    for n in LOSS_NAMES:
        assert isinstance(make_loss(n), DenoiseLoss)
