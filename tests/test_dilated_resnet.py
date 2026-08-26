"""M10 (Dilated ResNet) — **M06 과의 공정성 조건을 코드로 고정한다.**

M10 은 "downsampling 을 dilation 으로 바꾸면 QRS 보존이 나아지는가" 하나만
묻는 모델이다. 파라미터 수나 수용영역이 함께 달라지면 결과가 무엇 때문인지
가릴 수 없다 — F-10 에서 겪은 것과 같은 구조의 실수다.

그래서 두 모델의 params 와 RF 가 서로 1 % 안에 있어야 한다는 것을 검사로
남긴다. 나중에 누가 `ch` 나 `dilations` 를 손대면 여기서 걸린다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ecgdn.models import build_model  # noqa: E402
from ecgdn.models.resunet1d import ResUNet1D  # noqa: E402


@pytest.fixture(scope="module")
def pair():
    return ResUNet1D(), build_model("dilated_resnet1d")


def test_params_match_m06_within_1pct(pair):
    a, b = pair
    rel = abs(b.n_params() - a.n_params()) / a.n_params()
    assert rel < 0.01, (
        f"M10 params {b.n_params():,} vs M06 {a.n_params():,} ({rel:.1%}) — "
        "용량이 다르면 '해상도 유지의 효과' 가 아니라 '더 큰 모델의 효과' 가 된다")


def test_receptive_field_matches_m06_within_1pct(pair):
    a, b = pair
    rel = abs(b.receptive_field_samples - a.receptive_field_samples) / a.receptive_field_samples
    assert rel < 0.01, (
        f"M10 RF {b.receptive_field_samples} vs M06 {a.receptive_field_samples} "
        f"({rel:.1%}) — 문맥 폭이 다르면 무엇 때문에 이겼는지 가릴 수 없다")


def test_params_under_1m(pair):
    """STEP 17 DoD — 다른 딥러닝 방법과 같은 예산 안에 둔다."""
    _, b = pair
    assert b.n_params() < 1_000_000, b.n_params()


def test_receptive_field_fits_in_the_training_window(pair):
    """RF 가 window(1024)를 넘으면 경계에서 정보가 잘린다 (models/README.md)."""
    _, b = pair
    assert b.receptive_field_samples <= 1024


def test_length_is_preserved(pair):
    """dilation 의 'same' padding 은 `d * (k // 2)` 다.

    틀리면 출력이 조용히 짧아진다 — U-Net 과 달리 복원 단계가 없어서
    형상 오류가 늦게 드러난다.
    """
    _, b = pair
    for n in (512, 1024, 1000):
        assert b(torch.randn(2, 1, n)).shape == (2, 1, n)


def test_no_downsampling_anywhere(pair):
    """이 모델의 정의 자체 — stride 가 전부 1 이어야 한다."""
    _, b = pair
    bad = [f"{n}: stride={m.stride}" for n, m in b.named_modules()
           if isinstance(m, torch.nn.Conv1d) and tuple(m.stride) != (1,)]
    assert not bad, f"downsampling 이 남아 있다: {bad}"
    ups = [n for n, m in b.named_modules() if isinstance(m, torch.nn.Upsample)]
    assert not ups, f"upsampling 이 남아 있다: {ups}"


def test_starts_as_identity(pair):
    """head 를 0 으로 초기화해 초기 출력이 정확히 입력이어야 한다.

    residual 구조의 hallucination 억제 근거이고, M06 과 같은 조건이다.
    """
    _, b = pair
    b.eval()
    x = torch.randn(2, 1, 512)
    with torch.no_grad():
        assert torch.allclose(b(x), x, atol=1e-6)


def test_is_not_causal_and_that_is_deliberate():
    """M10 은 **의도적으로 비인과**다.

    공통 front-end 가 `sosfiltfilt`(zero-phase, 비인과)이므로 이 모델만
    인과로 만들어도 파이프라인은 여전히 비인과다. 그러면 '해상도 유지의
    효과' 와 '인과 제약의 대가' 가 한 숫자에 섞인다.
    """
    src = (ROOT / "ecgdn" / "models" / "dilated_resnet1d.py").read_text()
    assert "causal" in src.lower(), "비인과인 이유가 문서화돼 있어야 한다"
    m = build_model("dilated_resnet1d")
    m.eval()
    # head 가 0 초기화라 **초기 모델은 정확히 identity 이고 잡음 경로의
    # gradient 가 0 이다** (그것이 설계 의도다 — test_starts_as_identity).
    # 수용영역을 재려면 학습된 상태를 흉내내야 한다.
    torch.nn.init.normal_(m.head.weight, std=0.1)
    x = torch.randn(1, 1, 1024, requires_grad=True)
    y = m(x)
    y[0, 0, 512].backward()
    g = x.grad[0, 0].abs()
    assert g[513:].sum().item() > 0, "미래 샘플에 의존해야 한다 (비인과 설계)"


def test_convolutional_receptive_field_matches_the_formula():
    """계산식이 실제 구조와 맞는지 확인한다.

    **GroupNorm 을 걷어내고 잰다.** GroupNorm 은 시간축 전체의 평균·분산을
    쓰므로 의존 범위가 window 전체가 되어, 그대로 재면 어떤 모델이든
    "RF = window" 가 나온다. 설계 수치로서 의미가 있는 것은 **컨볼루션
    경로의 수용영역**이고 그것이 `receptive_field_samples` 다.
    (M06 의 887 도 같은 의미의 값이다 — `ecgdn/models/README.md`.)
    """
    m = build_model("dilated_resnet1d")
    m.eval()
    for name, mod in list(m.named_modules()):
        for cn, child in list(mod.named_children()):
            if isinstance(child, torch.nn.GroupNorm):
                setattr(mod, cn, torch.nn.Identity())
    torch.nn.init.normal_(m.head.weight, std=0.1)
    x = torch.randn(1, 1, 2048, requires_grad=True)
    y = m(x)
    y[0, 0, 1024].backward()
    nz = (x.grad[0, 0].abs() > 0).nonzero().flatten()
    span = int(nz[-1] - nz[0]) + 1
    assert abs(span - m.receptive_field_samples) <= 2, (
        f"실측 {span} vs 계산 {m.receptive_field_samples}")
