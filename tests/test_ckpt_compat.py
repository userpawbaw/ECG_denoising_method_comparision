"""커밋된 체크포인트가 **지금 코드로 열리는가**.

왜 이 검사가 없어서 뚫렸나
--------------------------
`scripts/check_ckpts.py` 는 **학습 코드의 해시**를 본다 (F-9). 그래서
`ecgdn/train.py` 나 `ecgdn/data/` 가 바뀌면 «재학습되지 않았다» 고 잡는다.
그런데 **모델 정의(`ecgdn/models/`)가 바뀌어 state_dict 키가 어긋나는 것**은
그 목록에 없었다. 실제로 인코더 한 레벨이 블록 하나에서 묶음으로 바뀌면서
(`enc.0.c1` -> `enc.0.0.c1`) **커밋된 체크포인트가 전부 안 열리게 됐는데,
게이트는 조용했다.** `m06_l1` 은 stale 목록에도 없었다.

열리지 않는다는 것은 `results/{d0,d1}/exp_*` 를 **다시 만들 수 없다**는 뜻이고,
그러면 그림도 표도 재생성이 막힌다. 산출물이 문서의 근거인 저장소에서 이것은
조용히 넘어갈 일이 아니다.

여기서 고정하는 것 둘
---------------------
1. **전부 열린다** — 데이터가 없어도 도는 빠른 검사. 이것 하나면 위 결함이
   잡혔다.
2. **같은 수치가 나온다** — 키를 옮겨 실은 것이 「모양만 맞춘 것」이 아니라
   **같은 모델**임을 커밋된 parquet 으로 증명한다. 데이터가 필요해 `slow` 다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _checkpoints() -> list[Path]:
    return sorted(p for tag in ("d0", "d1")
                  for p in (ROOT / "results" / tag).glob("*/best.pt"))


def test_there_are_checkpoints_to_check():
    """빈 목록이면 아래 검사가 **아무것도 안 하면서 통과**한다."""
    assert _checkpoints(), "커밋된 체크포인트가 하나도 없다 — 검사가 헛돈다"


@pytest.mark.parametrize("ckpt", _checkpoints(), ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_every_committed_checkpoint_loads(ckpt: Path):
    """지금 모델 코드로 열려야 한다. 안 열리면 산출물을 다시 못 만든다."""
    pytest.importorskip("torch")
    from ecgdn.methods.dl_wrapper import load_checkpoint
    try:
        load_checkpoint(ckpt, "cpu")
    except Exception as e:                                       # noqa: BLE001
        pytest.fail(
            f"{ckpt.relative_to(ROOT)} 가 안 열린다 — {type(e).__name__}: "
            f"{str(e)[:300]}\n\n모델 정의가 바뀌었는데 체크포인트가 그대로면 "
            "실험을 재생성할 수 없다. 재학습하거나, 가중치가 같은 자리 옮김이면 "
            "`ecgdn/methods/dl_wrapper._lift_old_encoder_keys` 처럼 "
            "**되돌리는 규칙을 명시하고 수치로 증명**할 것.")


def test_the_lift_rule_refuses_what_it_cannot_explain():
    """설명 못 하는 어긋남까지 맞춰 넣으면 **틀린 모델이 조용히 실린다.**"""
    from ecgdn.methods.dl_wrapper import _lift_old_encoder_keys
    want = {"enc.0.0.c1.weight", "head.weight"}
    # 옛 판 — 옮겨진다
    moved, lifted = _lift_old_encoder_keys(
        {"enc.0.c1.weight": 1, "head.weight": 2}, want)
    assert lifted and set(moved) == want
    # 아무 상관 없는 키 — 손대지 않고 그대로 돌려줘 `load_state_dict` 가 터지게 한다
    same, lifted = _lift_old_encoder_keys({"뭔가.다른.키": 1}, want)
    assert not lifted and set(same) == {"뭔가.다른.키"}


@pytest.mark.slow
def test_lifted_checkpoint_reproduces_committed_metrics():
    """옮겨 실은 모델이 **커밋된 값을 그대로** 내놓는가.

    키를 옮기는 것은 모양만 맞추는 일이 될 수도 있다 — 레벨 순서가 뒤바뀌어도
    모양은 맞는다. 그래서 `results/d1/exp_a/metrics.parquet` 의 값과 대조한다.
    """
    pytest.importorskip("torch")
    pytest.importorskip("wfdb")
    if not (ROOT / "data" / "raw" / "mitdb").exists():
        pytest.skip("MIT-BIH 원본이 없다")
    import numpy as np
    import pandas as pd
    import yaml

    sys.path.insert(0, str(ROOT / "scripts"))
    from run_exp import build_methods

    from ecgdn.data.dataset import build_eval_set
    from ecgdn.data.nstdb import make_banks
    from ecgdn.data.sources import get_source, source_tag
    from ecgdn.eval.engine import evaluate

    d = yaml.safe_load((ROOT / "configs" / "exp_a.yaml").read_text())["data"]
    src = get_source("mitdb", dur_s=float(d["dur_s"]), n_test=int(d["n_test"]))
    items = build_eval_set(
        src, "test", seg_s=float(d["seg_s"]), snr_grid=[-5.0],
        noise_conditions=("mixed",),
        banks=make_banks("test", d.get("nstdb_root", "data/raw/nstdb")),
        n_seg_per_record=int(d["n_seg_per_record"]), seed=d.get("seed", "eval"))
    built = build_methods(
        {"methods": [], "frontend": True,
         "dl_methods": {"M06": {"ckpt": "results/{tag}/m06_l1/best.pt"},
                        "M08": {"ckpt": "results/{tag}/m08_l1/best.pt"}}},
        source_tag("mitdb"))

    df = pd.read_parquet(ROOT / "results" / "d1" / "exp_a" / "metrics.parquet")
    df = df[(df.metric == "snr_imp_scaled") & (df.snr_in_target == -5.0)]
    bad = []
    for it in items[:2]:
        x, y, fs = it["x"].astype(float), it["y"].astype(float), float(it["fs"])
        for mid in ("M06", "M08"):
            got = evaluate(x, y, np.asarray(built[mid](y, fs, {}), float).ravel(),
                           fs, r_peaks_ref=np.asarray(it["r_peaks"], int))["snr_imp_scaled"]
            row = df[(df.record == str(it["record"])) & (df.seg == it["seg"])
                     & (df.method == mid)]
            assert len(row), f"parquet 에 {it['record']}/{it['seg']} {mid} 이 없다"
            ref = float(row.value.iloc[0])
            if abs(got - ref) > 1e-6:
                bad.append(f"{it['record']}/{it['seg']} {mid}: {got:.6f} vs {ref:.6f}")
    assert not bad, ("옮겨 실은 모델이 커밋된 값을 재현하지 못한다 — 같은 "
                     "모델이 아니다:\n" + "\n".join(bad))
