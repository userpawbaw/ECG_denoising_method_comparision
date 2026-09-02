#!/usr/bin/env python3
"""C3 카드용 데이터 은행 — **잡음을 고르면 시간축과 스펙트럼이 함께 바뀐다.**

    python3 scripts/build_card_bank.py            # d1, 잡음 7 종
    python3 scripts/build_card_bank.py --snr 0

산출: `demo/card_bank.js` (브라우저가 `<script src>` 로 읽는다. 서버 불필요)

왜 다시 만드는가
---------------
첫 C3 카드는 **잡음 없는 입력**(EXP-C)의 PSD 를 그렸다. 그러면 스펙트럼이
그냥 매끈하게 흘러내리는 곡선이라 **"무엇을 걸러내는 그림인가" 가 안 보인다.**
PSD 가 설득력을 갖는 순간은 **잡음의 봉우리가 보이고 그것이 사라질 때**다 —
60 Hz 전원선이 뾰족하게 서 있다가 notch 하나에 없어지는 그림이 그렇다.

그래서 은행에는 **잡음을 섞은 입력**과 각 방법의 출력을 함께 담고, 같은
색으로 시간축과 PSD 를 나란히 그린다. 잡음 종류를 바꾸면 **양쪽이 동시에**
바뀌므로 "이 방법이 어느 대역을 건드리는가" 가 조작으로 드러난다.

**은행은 `build_demo_bank.py` 와 같은 규약을 따른다** — 조합 하나가 스케일
하나를 공유하고(축이 어긋날 수 없다), int16 + base64 로 파일 하나다.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from ecgdn.config import FS
from ecgdn.data.dataset import build_eval_set
from ecgdn.data.nstdb import make_banks
from ecgdn.data.sources import get_source, source_tag
from ecgdn.eval.spectral import welch_psd
from ecgdn.methods import build as reg_build
from ecgdn.utils import ensure_dir, save_manifest

ROOT = Path(__file__).resolve().parents[1]

CONDS = ["pli", "bw_synth", "ma_synth", "em_synth", "impulse", "awgn", "mixed"]
COND_LABEL = {"pli": "전원선 60 Hz", "bw_synth": "기저선 변동",
              "ma_synth": "근전도(MA)", "em_synth": "전극 움직임(EM)",
              "impulse": "임펄스", "awgn": "백색잡음", "mixed": "혼합"}
# 잡음마다 **어느 대역에 사는가**. 카드가 이것을 PSD 위에 띠로 그린다 —
# "이 잡음은 여기 있다" 를 말로 하지 않고 보이기 위해서다.
COND_BAND = {"pli": (59.0, 61.0), "bw_synth": (0.0, 0.7),
             "ma_synth": (20.0, 100.0), "em_synth": (0.0, 8.0),
             "impulse": (0.0, 125.0), "awgn": (0.0, 125.0), "mixed": (0.0, 125.0)}

# 은행에는 넉넉히 담고 **화면이 셋만 고른다.** dataviz 팔레트의 앞 세 슬롯
# (blue·orange·aqua)만 all-pairs 검증을 통과하기 때문이다 — 네 번째를 겹쳐
# 그리면 정상시야 분리도가 무너진다(M_FE magenta ↔ M04 orange 가 ΔE 12.9).
METHODS = ["M_FE", "M01", "M04", "M06", "M08"]
SHOW_S = 6.0            # 시간축에 그릴 길이
PSD_S = 20.0            # 스펙트럼을 재는 길이 (길수록 매끈하다)


def dl(mid: str, tag: str):
    from ecgdn.methods.dl_wrapper import DLDenoiser
    ck = ROOT / "results" / tag / {"M06": "m06_l1", "M08": "m08_l1"}[mid] / "best.pt"
    if not ck.exists():
        raise FileNotFoundError(ck)
    return DLDenoiser(ckpt=ck, name=mid)


def b64(v: np.ndarray, scale: float) -> str:
    q = np.clip(np.round(np.asarray(v) / scale), -32768, 32767).astype("<i2")
    return base64.b64encode(q.tobytes()).decode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="mitdb", choices=("mitdb", "synthetic"))
    ap.add_argument("--snr", type=float, default=5.0)
    ap.add_argument("--out", default="demo/card_bank.js")
    a = ap.parse_args()

    tag = source_tag(a.source)
    src = get_source(a.source, dur_s=300.0, n_test=22)
    banks = make_banks("test", "data/raw/nstdb")
    methods = {m: (dl(m, tag) if m.startswith(("M06", "M08"))
                   else reg_build(m)) for m in METHODS}

    n_show, n_psd = int(SHOW_S * FS), int(PSD_S * FS)
    scenes = []
    for cond in CONDS:
        # **기록·구간·잡음 실현을 조건마다 한 번만 뽑는다.** seed 에 SNR 을 넣지
        # 않는 것은 `build_demo_bank.py` 와 같은 이유다(SNR 만 바뀐 같은 신호).
        items = build_eval_set(src, "test", seg_s=PSD_S, snr_grid=[a.snr],
                               noise_conditions=(cond,), banks=banks,
                               n_seg_per_record=1, seed=f"card_{cond}")
        it = items[0]
        x, y = it["x"].astype(np.float64), it["y"].astype(np.float64)
        outs = {m: np.asarray(f(y, FS, {}), dtype=np.float64)
                for m, f in methods.items()}

        # 시간축: 가운데 SHOW_S 초. 스케일은 조합이 **하나**를 공유한다.
        o = (len(x) - n_show) // 2
        sl = slice(o, o + n_show)
        # **모든 파형에서 평균을 뺀다.** 방법 출력은 front-end 를 통과해 DC 가
        # 없는데 참값·입력은 있어서, 같은 축에 그리면 참값만 아래로 밀려 마치
        # 다른 신호처럼 보였다. 지표의 DC 규약(3.2.1)과도 이쪽이 맞다.
        tr = {"clean": x[sl], "input": y[sl], **{m: v[sl] for m, v in outs.items()}}
        tr = {k: v - float(np.mean(v)) for k, v in tr.items()}
        scale = max(float(np.max(np.abs(v))) for v in tr.values()) / 32000.0

        # PSD: 긴 구간으로 재고 **참값 최댓값을 0 dB** 로 잡는다. 지수 라벨을
        # 피하고("10^-5" 는 한글 폰트에서 깨진다) dB 가 이 관객에게 더 익숙하다.
        f0, p_ref = welch_psd(x, FS)
        top = float(np.max(p_ref))
        def db(v):
            return 10.0 * np.log10(np.maximum(v, top * 1e-9) / top)
        psd = {"clean": db(p_ref)}
        psd["input"] = db(welch_psd(y, FS)[1])
        for m, v in outs.items():
            psd[m] = db(welch_psd(v, FS)[1])

        scenes.append(dict(
            cond=cond, label=COND_LABEL[cond], band=COND_BAND[cond],
            record=str(it["record"]), snr=a.snr, scale=scale,
            traces={k: b64(v, scale) for k, v in tr.items()},
            psd={k: b64(v, 0.01) for k, v in psd.items()},   # 0.01 dB 해상도
        ))
        print(f"  [{cond:9}] 기록 {it['record']} · 입력 PSD 최대 "
              f"{psd['input'].max():+.1f} dB · M01 {psd['M01'].max():+.1f} dB")

    bank = dict(fs=FS, n=n_show, snr=a.snr, axis=tag, methods=METHODS,
                psd_f=[round(float(v), 3) for v in f0], psd_scale=0.01,
                conds=CONDS, cond_label=COND_LABEL, scenes=scenes)
    out = ROOT / a.out
    ensure_dir(out.parent)
    out.write_text("window.CARD_BANK = " + json.dumps(bank, ensure_ascii=False,
                                                      separators=(",", ":")) + ";\n")
    save_manifest(ROOT / "results" / "card_bank",
                  cfg={"source": a.source, "snr": a.snr, "conds": CONDS,
                       "methods": METHODS}, sources=[__file__])
    print(f"\n{out.relative_to(ROOT)}  {out.stat().st_size/1e6:.2f} MB  "
          f"장면 {len(scenes)} × 파형 {len(METHODS)+2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
