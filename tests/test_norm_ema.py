"""D-41 — 정규화 빼기(E8)와 가중치 EMA(E9)의 전제를 고정한다.

둘 다 **기본값에서는 예전과 같아야** 한다. 그래야 지금까지의 모든 판과 견줄 수 있다.
"""
import pytest

torch = pytest.importorskip("torch")

from ecgdn.config import TrainCfg  # noqa: E402
from ecgdn.models.resunet1d import ResUNet1D  # noqa: E402


def _pair(**kw):
    torch.manual_seed(0)
    a = ResUNet1D()
    torch.manual_seed(0)
    b = ResUNet1D(**kw)
    return a, b


def test_default_norm_is_bit_identical():
    a, b = _pair(norm="group")
    assert list(a.state_dict()) == list(b.state_dict())
    x = torch.randn(2, 1, 1024)
    assert torch.equal(a(x), b(x))


def test_none_removes_every_groupnorm_and_keeps_convs():
    a, b = _pair(norm="none")
    assert not any(isinstance(m, torch.nn.GroupNorm) for m in b.modules())
    conv = lambda m: [k for k in m.state_dict() if k.endswith(".weight") and  # noqa: E731
                      m.state_dict()[k].dim() == 3]
    assert conv(a) == conv(b)
    assert torch.isfinite(b(torch.randn(2, 1, 1024))).all()


def test_bad_norm_is_rejected():
    with pytest.raises(ValueError):
        ResUNet1D(norm="layer")


# ---------------------------------------------------------------- EMA
class _DS(torch.utils.data.Dataset):
    def __init__(self, n=8):
        g = torch.Generator().manual_seed(0)
        self.x = [torch.randn(1, 256, generator=g).numpy() for _ in range(n)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i]
        return (x + 0.1).astype("float32"), x.astype("float32"), {"snr": 10.0}


def _trainer(tmp_path, decay):
    from ecgdn.models import make_loss
    from ecgdn.train import Trainer
    torch.manual_seed(0)
    m = ResUNet1D(chs=(8, 8, 8))
    cfg = TrainCfg(lr=1e-3, epochs=2, batch_size=4, patience=50, ema_decay=decay)
    return Trainer(m, make_loss("L1"), _DS(), _DS(4), cfg, out_dir=tmp_path,
                   device="cpu", num_workers=0)


def test_ema_off_creates_no_state(tmp_path):
    t = _trainer(tmp_path, 0.0)
    assert t.ema is None
    with t._ema_weights():                      # 아무것도 안 해야 한다
        pass


def test_ema_update_formula_and_swap_restores(tmp_path):
    t = _trainer(tmp_path, 0.9)
    k = next(k for k, v in t.model.state_dict().items() if v.dtype.is_floating_point)
    old = t.ema[k].clone()
    with torch.no_grad():
        for p in t.model.parameters():
            p.add_(1.0)
    t._ema_update()
    cur = t.model.state_dict()[k]
    assert torch.allclose(t.ema[k], 0.9 * old + 0.1 * cur)
    raw = {kk: v.clone() for kk, v in t.model.state_dict().items()}
    with t._ema_weights():
        assert torch.equal(t.model.state_dict()[k], t.ema[k])
    for kk, v in t.model.state_dict().items():
        assert torch.equal(v, raw[kk]), "평가 뒤 학습 가중치로 안 돌아왔다"


def test_ema_is_saved_and_resumed(tmp_path):
    t = _trainer(tmp_path, 0.99)
    t.fit()
    ck = torch.load(tmp_path / "last.pt", weights_only=False)
    assert "ema" in ck
    best = torch.load(tmp_path / "best.pt", weights_only=False)["model"]
    k = next(iter(best))
    # best.pt 는 EMA 가중치다 (마지막 epoch 이 best 일 때 last 의 EMA 와 같다)
    if t.state.best_epoch == t.state.epoch:
        assert torch.equal(best[k], ck["ema"][k])
    t2 = _trainer(tmp_path, 0.99)
    assert t2.try_resume()
    assert torch.equal(t2.ema[k], ck["ema"][k])
