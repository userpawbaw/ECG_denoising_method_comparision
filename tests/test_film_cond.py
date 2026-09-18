"""D-28 2 번 — FiLM 조건화가 **측정 가능한 상태로** 들어갔는지 고정한다.

여기서 막으려는 사고는 세 가지다.
  1) 조건화를 켜면서 기존 체크포인트를 전부 못 읽게 만드는 것 (F-9 계열)
  2) 조건을 넘기는 것을 잊고 「조건화가 안 통하더라」로 결론짓는 것
  3) 조건을 넣었는데 출력이 조건에 반응하지 않는 것 (배선 실수)
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecgdn.models import build_model                     # noqa: E402
from ecgdn.models.resunet1d import ResUNet1D             # noqa: E402


def _y(n=2, L=1024, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 1, L, generator=g)


def test_cond_off_keeps_state_dict_keys():
    """조건을 끄면 **키가 한 글자도 안 바뀐다** — 예전 체크포인트가 그대로 열린다."""
    m = ResUNet1D()
    assert not m.cond and not m.needs_cond
    keys = set(m.state_dict())
    assert any(k.startswith("enc.0.0.") for k in keys), "종래 Sequential 키가 아니다"
    assert not any(".blocks." in k or ".heads." in k for k in keys)
    assert all("embed" not in k for k in keys)
    assert ResUNet1D(n_blocks=1).n_params() == m.n_params()


def test_film_is_identity_at_init():
    """head 를 0 으로 뒀으므로 **초기값에서는 조건 없음과 완전히 같은 함수**다.

    이게 성립해야 「조건화 때문에 좋아졌다/나빠졌다」를 초기화 차이와 섞지 않는다.
    """
    torch.manual_seed(7)
    a = ResUNet1D()
    torch.manual_seed(7)
    b = ResUNet1D(cond=True)
    # 공통 가중치를 그대로 옮긴다 (FiLM head·embed 만 b 에 더 있다)
    sa, sb = a.state_dict(), b.state_dict()
    moved = 0
    for k, v in sa.items():
        kb = k.replace(".0.c", ".blocks.0.c").replace(".0.n", ".blocks.0.n")
        if kb in sb and sb[kb].shape == v.shape:
            sb[kb] = v.clone(); moved += 1
    assert moved > 0
    b.load_state_dict(sb)
    a.eval(); b.eval()
    y = _y()
    with torch.no_grad():
        oa = a(y)
        for s in (-5.0, 0.0, 20.0):
            ob = b(y, torch.full((y.shape[0],), s))
            assert torch.allclose(oa, ob, atol=0, rtol=0), f"snr={s} 에서 달라졌다"


def test_output_depends_on_cond_after_perturbation():
    """head 가 0 이 아니게 되면 **출력이 조건에 따라 달라져야 한다**.

    배선이 끊겨 있으면(emb 를 안 넘기거나 γ 를 안 쓰면) 여기서 잡힌다.
    출력 head 도 함께 흔든다 — 그게 0 이면 망이 무엇을 하든 출력이 y 라서
    조건 의존이 보이지 않는다 (처음 이 검사를 짤 때 실제로 걸린 함정이다).
    """
    torch.manual_seed(3)
    m = ResUNet1D(cond=True).eval()
    torch.nn.init.normal_(m.head.weight, std=0.05)
    for st in m.modules():
        if hasattr(st, "heads"):
            for h in st.heads:
                torch.nn.init.normal_(h.weight, std=0.05)
    y = _y(seed=1)
    with torch.no_grad():
        lo = m(y, torch.full((2,), -5.0))
        hi = m(y, torch.full((2,), 20.0))
    assert not torch.allclose(lo, hi, atol=1e-6)


def test_missing_or_spurious_cond_raises():
    """조용히 0 으로 때우지 않는다 — 둘 다 터진다."""
    with pytest.raises(ValueError, match="snr 이 필요하다"):
        ResUNet1D(cond=True)(_y())
    with pytest.raises(ValueError, match="snr 을 줬다"):
        ResUNet1D()(_y(), torch.zeros(2))


def test_cond_scalar_broadcasts_and_length_is_checked():
    m = ResUNet1D(cond=True).eval()
    y = _y(n=4)
    with torch.no_grad():
        assert m(y, torch.tensor([3.0])).shape == y.shape       # 스칼라 -> 배치로 확장
    with pytest.raises(ValueError, match="배치"):
        m(y, torch.zeros(3))


def test_cond_config_is_iso_param_with_baseline():
    """`configs/m06_cond.yaml` 이 기준과 **파라미터가 맞는지** 확인한다.

    안 맞으면 D-28 2 번은 「조건 덕인지 용량 덕인지」를 못 가른다 (F-42 의 교훈).
    허용 오차는 D-27 의 등파라미터 오차(-0.6 ~ +0.3 %)와 같은 자로 ±1 % 다.
    """
    import yaml
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    base = build_model("resunet1d").n_params()
    c = yaml.safe_load((root / "configs" / "m06_cond.yaml").read_text())
    kw = c["model"]["kwargs"]
    assert kw.get("cond") is True
    n = build_model(c["model"]["name"], **kw).n_params()
    assert abs(n - base) / base < 0.01, f"등파라미터가 아니다: {n} vs {base}"


def test_clean_cond_snr_is_top_of_range():
    """`L6` 의 clean 통과에 줄 조건값. 범위 밖으로 멀리 보내지 않는다."""
    m = ResUNet1D(cond=True, cond_lo=-5.0, cond_hi=20.0)
    assert m.clean_cond_snr == 20.0
    assert np.isnan(ResUNet1D().clean_cond_snr)


def test_cond_normalize_maps_range_to_unit_and_extrapolates():
    """학습 범위는 [-1, 1] 로, 범위 밖은 **자르지 않고** 그대로 넘어간다.

    D-28 4 번이 ±5 dB 교란을 줄 예정이라 여기서 잘리면 그 실험이 무의미해진다.
    """
    e = ResUNet1D(cond=True).embed
    v = e.normalize(torch.tensor([-5.0, 7.5, 20.0, 25.0]))
    assert torch.allclose(v[:3], torch.tensor([-1.0, 0.0, 1.0]), atol=1e-6)
    assert float(v[3]) > 1.0


def test_trainer_feeds_true_snr_to_cond_model():
    """학습기가 배치의 참 SNR 을 실제로 넘기는지 — 넘기다 만 것을 잡는다."""
    from ecgdn.train import Trainer
    seen: list = []

    class Spy(ResUNet1D):
        def forward(self, y, snr=None):
            seen.append(None if snr is None else np.asarray(snr.detach().cpu()))
            return super().forward(y, snr)

    from ecgdn.config import TrainCfg
    from ecgdn.data.dataset import ECGDenoiseDataset
    from ecgdn.data.sources import get_source
    from ecgdn.models import make_loss
    import tempfile

    src = get_source("synthetic", n_records=2, dur_s=20.0)
    ds = ECGDenoiseDataset(src, split="train", win=256, hop=256,
                           max_per_record=4, frontend=False)
    m = Spy(cond=True, chs=(4, 6, 8))
    with tempfile.TemporaryDirectory() as td:
        t = Trainer(m, make_loss("L1"), ds, ds,
                    TrainCfg(epochs=1, batch_size=2, patience=1),
                    out_dir=td, device="cpu", num_workers=0)
        t.evaluate(ds)
    assert seen and all(s is not None and s.size == 2 for s in seen)
    # 합성 SNR 범위 안에 있어야 한다 (0 으로 때운 게 아니다)
    assert len({float(s[0]) for s in seen}) > 1
