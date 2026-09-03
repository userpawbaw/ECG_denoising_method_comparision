"""front-end 를 **입력과 목표에 따로** 걸 수 있어야 한다 (docs/15 §7 목적 B).

`frontend` 하나가 둘 다 껐을 때는 «입력은 날것, 목표는 FE 통과» 를 표현할 수
없었다. 그런데 **D1 에서는 그 조합만이 옳다** — 목표에서 FE 를 빼면 목표가
「MIT-BIH 원본」이 되고 그것은 clean 이 아니다(**F-12**). 모델이 기록 자신의
잡음을 재현하도록 학습된다.

여기서 고정하는 것 셋:
  1. 옛 동작이 그대로다 (`frontend` 만 주면 목표도 따라간다)
  2. 셋째 조합이 실제로 만들어진다
  3. **guard band 가 참조에만 FE 를 걸 때도 붙는다** — 경계 트랜지언트는
     어느 쪽을 거르든 똑같이 생긴다
"""
from __future__ import annotations

import numpy as np
import pytest

from ecgdn.data.dataset import ECGDenoiseDataset
from ecgdn.data.sources import get_source


@pytest.fixture(scope="module")
def src():
    try:
        return get_source("synthetic", dur_s=60.0, n_train=2, n_val=1, n_test=1)
    except Exception as e:                        # pragma: no cover
        pytest.skip(f"소스를 못 만들었다: {e}")


def _ds(src, **kw):
    return ECGDenoiseDataset(src, "train", win=512, hop=512, max_per_record=2, **kw)


def test_reference_frontend_follows_input_by_default(src):
    """`ref_frontend` 를 안 주면 예전과 같아야 한다 — 목표도 함께 꺼진다."""
    off = _ds(src, frontend=False)
    assert off.ref_frontend is False, "기본값이 바뀌면 옛 설정의 의미가 달라진다"
    on = _ds(src, frontend=True)
    assert on.ref_frontend is True


def test_input_raw_but_target_filtered_is_expressible(src):
    """**셋째 조합**: 입력은 날것, 목표는 FE 통과."""
    both = _ds(src, frontend=True, ref_frontend=True).raw_item(0)
    mixed = _ds(src, frontend=False, ref_frontend=True).raw_item(0)
    none = _ds(src, frontend=False, ref_frontend=False).raw_item(0)
    # 입력은 FE 를 안 거쳤으므로 both 와 달라야 한다
    assert not np.allclose(mixed["y"], both["y"]), "입력에 아직 FE 가 걸려 있다"
    # 목표는 FE 를 거쳤으므로 «둘 다 끈 것» 과 달라야 한다
    assert not np.allclose(mixed["x"], none["x"]), "목표에 FE 가 안 걸렸다"


def test_guard_band_is_kept_when_only_the_reference_is_filtered(src):
    """참조에만 FE 를 걸어도 여유 구간이 필요하다.

    0.5 Hz 고역통과는 수 초간 울린다. 여유 없이 창에 그대로 걸면 창 전체가
    트랜지언트가 되고, 그러면 목표 자체가 망가진다.
    """
    ds = _ds(src, frontend=False, ref_frontend=True)
    d = ds.raw_item(0)
    assert d["x"].size == ds.win, "가운데만 잘라내는 규약이 깨졌다"
    # 여유가 0 이면 첫 표본 근처가 크게 튄다 — 창 안이 고르게 유지되는지 본다
    head = float(np.percentile(np.abs(d["x"][:64]), 95))
    body = float(np.percentile(np.abs(d["x"]), 95))
    assert head < 4.0 * body + 1e-9, (
        f"창 앞이 트랜지언트다 (앞 {head:.3f} vs 전체 {body:.3f}) — guard band 가 빠졌다")
