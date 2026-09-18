"""D-28 4 번 — **조건값을 일부러 틀리게 준다.**

막으려는 사고 셋.
  1) 교란이 학습으로 새는 것 — 학습 중 `cond_offset` 이 0 이 아니면 다른 실험이다
  2) 평가 뒤에 되돌리지 않아 **그 뒤의 모든 수치가 오염**되는 것
  3) 조건 없는 모델에 교란을 걸어 놓고 「영향이 없네」로 읽는 것
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecgdn.config import TrainCfg                      # noqa: E402
from ecgdn.data.dataset import ECGDenoiseDataset       # noqa: E402
from ecgdn.data.sources import get_source              # noqa: E402
from ecgdn.models import make_loss                     # noqa: E402
from ecgdn.models.resunet1d import ResUNet1D           # noqa: E402
from ecgdn.train import Trainer                        # noqa: E402


class Spy(ResUNet1D):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen: list[np.ndarray] = []

    def forward(self, y, snr=None):
        if snr is not None:
            self.seen.append(np.asarray(snr.detach().cpu()))
        return super().forward(y, snr)


def _bits(tmp_path, **kw):
    src = get_source("synthetic", n_records=2, dur_s=20.0)
    ds = ECGDenoiseDataset(src, split="train", win=256, hop=256,
                           max_per_record=4, frontend=False)
    m = Spy(cond=True, chs=(4, 6, 8), **kw)
    t = Trainer(m, make_loss("L1"), ds, ds,
                TrainCfg(epochs=1, batch_size=2, patience=1),
                out_dir=tmp_path, device="cpu", num_workers=0)
    return m, t, ds


def test_default_offset_is_zero_and_true_snr_reaches_the_model(tmp_path):
    m, t, ds = _bits(tmp_path)
    assert t.cond_offset == 0.0
    t.evaluate(ds)
    true = np.array([ds[i][2]["snr"] for i in range(len(ds))], dtype=np.float32)
    got = np.concatenate(m.seen)
    assert np.allclose(np.sort(got), np.sort(true), atol=1e-5)


def test_offset_shifts_every_conditioning_value_by_exactly_that(tmp_path):
    m, t, ds = _bits(tmp_path)
    t.evaluate(ds)
    base = np.sort(np.concatenate(m.seen))
    m.seen.clear()
    t.cond_offset = -5.0
    t.evaluate(ds)
    got = np.sort(np.concatenate(m.seen))
    assert np.allclose(got, base - 5.0, atol=1e-4)


def test_offset_is_not_clipped_at_the_training_range(tmp_path):
    """범위 밖으로 나가야 「추정이 틀렸을 때」를 잴 수 있다."""
    e = ResUNet1D(cond=True, cond_lo=-5.0, cond_hi=20.0).embed
    v = e.normalize(torch.tensor([-10.0, 25.0]))
    assert float(v[0]) < -1.0 and float(v[1]) > 1.0


def test_sweep_restores_the_offset_after_perturbed_eval():
    """되돌리지 않으면 **그 뒤의 모든 판정이 조용히 오염된다.**

    `run_seed_sweep.py` 의 교란 블록이 `trainer.cond_offset = 0.0` 으로
    끝나는지를 소스에서 확인한다 — 실제 판을 돌리지 않고 고정할 수 있는
    성질이라 여기서 싸게 막는다.
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent
           / "scripts" / "run_seed_sweep.py").read_text()
    i = src.index("if cond_perturb and")
    blk = src[i:i + 900]
    assert "trainer.cond_offset = 0.0" in blk, "교란 뒤 원복이 없다"
    assert blk.index("for off in COND_OFFSETS") < blk.index("trainer.cond_offset = 0.0")


def test_offset_grid_excludes_zero():
    """0 은 `fixed_snr_imp_scaled` 가 이미 잰 값이다 — 두 번 재지 않는다."""
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "scripts" / "run_seed_sweep.py"
    src = p.read_text()
    ns: dict = {}
    exec(src[src.index("COND_OFFSETS = "):].split("\n")[0], ns)
    assert 0 not in ns["COND_OFFSETS"]
    assert sorted(ns["COND_OFFSETS"]) == ns["COND_OFFSETS"]


def test_perturbation_needs_a_conditioned_model(tmp_path):
    """조건 없는 모델에는 `snr` 이 아예 안 가므로 교란이 무의미하다 —
    그런 arm 에 `--cond-perturb` 를 걸어도 열이 안 생겨야 한다(조용한 0 금지)."""
    m = ResUNet1D()
    assert getattr(m, "needs_cond", False) is False
