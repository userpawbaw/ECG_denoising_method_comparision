import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecgdn.data.synthetic import synth_ecg  # noqa: E402
from ecgdn.methods.dl_wrapper import DLDenoiser  # noqa: E402
from ecgdn.models import build_model  # noqa: E402


class _Identity(torch.nn.Module):
    def forward(self, x):
        return x


class _Half(torch.nn.Module):
    def forward(self, x):
        return 0.5 * x


@pytest.mark.parametrize("n", [1024, 3000, 12345])
def test_length_preserved_and_identity(n):
    x = np.random.default_rng(n).standard_normal(n)
    d = DLDenoiser(model=_Identity(), name="M06", frontend=False)
    out = d(x, 250.0)
    assert out.shape == x.shape
    assert np.max(np.abs(out - x)) < 1e-4


def test_scaling_is_undone():
    """모델이 0.5배로 만들면 출력도 정확히 0.5배여야 한다 (정규화가 새는지 확인)."""
    x = synth_ecg(30.0, seed=2).x
    d = DLDenoiser(model=_Half(), name="M06", frontend=False)
    out = d(x, 250.0)
    assert np.max(np.abs(out - 0.5 * x)) < 1e-4


def test_no_boundary_discontinuity():
    """hop 배수 위치에서 1차 차분이 튀지 않아야 한다."""
    s = synth_ecg(40.0, seed=3)
    m = build_model("resunet1d")
    torch.manual_seed(0)
    for p in m.parameters():                   # 무작위 비선형 응답을 만들기 위해
        p.data.add_(0.01 * torch.randn_like(p))
    d = DLDenoiser(model=m, name="M06", batch=8, frontend=False)
    out = d(s.x, s.fs)
    dif = np.abs(np.diff(out))
    hops = np.arange(d.hop, len(dif) - 1, d.hop)
    bg = np.std(dif)
    assert np.max(dif[hops]) < np.max(dif) + 1e-12
    assert np.mean(dif[hops]) < np.mean(dif) + 3 * bg


def test_real_model_shapes():
    s = synth_ecg(20.0, seed=4)
    for name in ("resunet1d", "wavelet_unet"):
        d = DLDenoiser(model=build_model(name), name="M06", batch=4, frontend=False)
        out = d(s.x, s.fs)
        assert out.shape == s.x.shape and np.all(np.isfinite(out))


def test_frontend_is_applied_when_enabled():
    """딥러닝도 고전 기법과 동일한 공통 front-end 를 받아야 한다.

    회귀 테스트: 이 경로가 없으면 DL 만 기저선 제거를 처음부터 학습해야 해서
    baseline wander 조건에서 불공정한 비교가 된다 (FE 단독 +20.3 dB vs FE 없는 U-Net +6.3 dB).
    """
    from ecgdn.methods.frontend import apply_frontend

    s = synth_ecg(30.0, seed=5)
    drift = 0.5 * np.sin(2 * np.pi * 0.1 * np.arange(s.x.size) / s.fs)
    y = s.x + drift
    on = DLDenoiser(model=_Identity(), name="M06", frontend=True)(y, s.fs)
    off = DLDenoiser(model=_Identity(), name="M06", frontend=False)(y, s.fs)
    assert np.max(np.abs(off - y)) < 1e-4                      # FE 없으면 그대로
    assert np.max(np.abs(on - apply_frontend(y, s.fs))) < 1e-4  # FE 있으면 필터를 통과
    g = int(5 * s.fs)
    assert np.std(on[g:-g]) < np.std(off[g:-g])                 # 드리프트가 제거됨


# --------------------------------------------------------------- window 정합
def test_wrapper_follows_the_training_window_recorded_in_the_checkpoint(tmp_path):
    """다른 길이로 학습한 모델을 기본값 1024 로 돌리면 조용히 어긋난다.

    `frontend` 와 같은 계열의 train/inference 불일치다 — 에러가 나지 않고
    표는 정상적으로 생성되며 그 방법만 틀린다 (F-9).
    """
    import torch

    from ecgdn.models import build_model

    ck = tmp_path / "best.pt"
    torch.save({"model": build_model(name="resunet1d").state_dict(),
                "model_name": "resunet1d", "frontend": False,
                "data_win": 2048, "data_hop": 1024}, ck)
    d = DLDenoiser(ckpt=ck, name="M06")
    assert (d.win, d.hop) == (2048, 1024), f"체크포인트를 따르지 않았다: {d.win}/{d.hop}"


def test_explicit_window_overrides_the_checkpoint(tmp_path):
    """명시하면 이긴다 — 진단·탐색을 막지는 않는다."""
    import torch

    from ecgdn.models import build_model

    ck = tmp_path / "best.pt"
    torch.save({"model": build_model(name="resunet1d").state_dict(),
                "model_name": "resunet1d", "frontend": False,
                "data_win": 2048, "data_hop": 1024}, ck)
    d = DLDenoiser(ckpt=ck, name="M06", win=512, hop=256)
    assert (d.win, d.hop) == (512, 256)


def test_old_checkpoint_without_window_falls_back_to_1024(tmp_path):
    """기록이 없는 구형 체크포인트는 전부 1024 로 학습됐다 — 그 값으로 둔다."""
    import torch

    from ecgdn.models import build_model

    ck = tmp_path / "best.pt"
    torch.save({"model": build_model(name="resunet1d").state_dict(),
                "model_name": "resunet1d", "frontend": False}, ck)
    d = DLDenoiser(ckpt=ck, name="M06")
    assert (d.win, d.hop) == (1024, 512)
