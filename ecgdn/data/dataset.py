"""학습/평가 데이터셋 (docs/02_procedure.md STEP 16).

핵심 규약
--------
  * 난수는 `(salt, record, window_idx)` 로부터 **결정론적으로 유도**한다.
    같은 seed 로 두 번 만들면 비트 단위로 같은 배치가 나와야 한다.
  * 정규화는 **noisy 입력에서만** 계산하고, 출력에서 반드시 되돌린다 (docs/00_review.md A-6).
    clean 을 보고 스케일을 정하면 정보 누설이다.
  * 평가용 데이터는 미리 만들어 디스크에 저장한다. 모든 방법이 **완전히 동일한 y** 를 받아야
    비교가 공정하다.
  * **참조(정답) 신호는 front-end 를 통과한 것이다** (`ref_frontend`).
    MIT-BIH 는 clean 이 아니라 이미 기저선 변동을 담고 있어서, 원본을 정답으로 두면
    front-end 가 그 성분을 제거할수록 정답에서 **멀어져** SNR 이 떨어진다. 실측:
    잡음 없이 front-end 만 통과했을 때 원본과의 SNR 이 D0 는 53.2 dB(무해)인데
    D1 은 9.3 dB 였고, 그 결과 10 dB 조건에서 무처리(M00)가 M04 보다 높게 나왔다.
    즉 평가가 아무것도 구분하지 못한다. 그래서 목표를 **"0.5~100 Hz 대역의 ECG 를
    복원한다"** 로 명시하고 참조도 그 대역으로 맞춘다 (docs/02_procedure.md F-12).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config import EVAL_GUARD_S, HOP, SNR_TRAIN_RANGE, WIN
from ..utils import derive_seed, robust_scale, rng as make_rng
from .mixer import mix_at_snr
from .noise import SYNTH_KINDS, mixed_noise
from .sources import CleanSource, get_source

__all__ = ["WindowIndex", "ECGDenoiseDataset", "build_eval_set", "load_eval_set"]


@dataclass(frozen=True)
class WindowIndex:
    record: str
    start: int


def _index_windows(source: CleanSource, split: str, win: int, hop: int,
                   max_per_record: int | None = None) -> list[WindowIndex]:
    out: list[WindowIndex] = []
    for name in source.records(split):
        n = source.get(name).x.size
        starts = list(range(0, max(n - win + 1, 1), hop))
        if max_per_record is not None and len(starts) > max_per_record:
            step = len(starts) / max_per_record
            starts = [starts[int(i * step)] for i in range(max_per_record)]
        out += [WindowIndex(name, s) for s in starts]
    return out


class ECGDenoiseDataset:
    """on-the-fly 잡음 합성 데이터셋 (torch.utils.data.Dataset 호환).

    torch 가 없어도 순수 numpy 로 동작한다 (테스트/검증용).
    """

    def __init__(self, source: CleanSource | None = None, split: str = "train",
                 win: int = WIN, hop: int = HOP,
                 snr_range: tuple[float, float] = SNR_TRAIN_RANGE,
                 noise_kinds: Sequence[str] | None = None,
                 banks: dict[str, Any] | None = None,
                 salt: Any = 0, max_per_record: int | None = None,
                 normalize: bool = True, pre_denoise: str | None = None,
                 frontend: bool = True, fe_margin_s: float = EVAL_GUARD_S,
                 ref_frontend: bool | None = None):
        self.source = source or get_source("auto")
        self.split = split
        self.win, self.hop = int(win), int(hop)
        self.snr_range = tuple(float(v) for v in snr_range)
        self.noise_kinds = list(noise_kinds) if noise_kinds else list(SYNTH_KINDS)
        self.banks = banks or {}
        self.salt = salt
        self.normalize = bool(normalize)
        # M07(순차 hybrid) 용 전처리. 문자열 ID 로 받아 worker 프로세스에서 각자 만든다.
        self.pre_denoise = pre_denoise
        self._pre = None
        # 공통 front-end. 고전 기법은 전부 내부에서 FE 를 적용하는데 딥러닝만 raw 를 받으면
        # '기저선 제거를 처음부터 학습해야 하는' 불공정이 생긴다 (실측: baseline wander 조건에서
        # FE 단독 +20.3 dB vs FE 없는 U-Net +6.3 dB). 학습·추론 모두에 동일하게 적용한다.
        self.frontend = bool(frontend)
        # 학습 타깃도 참조와 같은 대역이어야 한다. 다르면 신경망이 front-end 를
        # 되돌리는 법을 배우게 된다.
        # **안 주면 `frontend` 를 따라간다.** 여기를 True 로 고정하면 예전에
        # `frontend=False` 로 만들던 모든 호출의 의미가 조용히 바뀐다.
        # 명시할 때만 «입력은 날것, 목표는 FE 통과» 가 된다 (docs/15 7절).
        self.ref_frontend = bool(frontend if ref_frontend is None else ref_frontend)
        self.fe_margin_s = float(fe_margin_s)
        self._fe = None
        self.index = _index_windows(self.source, split, self.win, self.hop, max_per_record)

    def _fe_fn(self):
        """front-end 객체. **입력·참조 어느 한쪽이라도 쓰면** 만든다.

        전에는 `self.frontend` 하나가 둘 다 껐다. 그러면 «입력은 날것,
        목표는 FE 통과» 라는 조합을 표현할 수 없는데, **D1 에서는 그 조합만이
        옳다** — 목표에서 FE 를 빼면 목표가 «MIT-BIH 원본» 이 되고 그것은
        clean 이 아니다(**F-12**). 모델이 기록 자신의 잡음을 재현하도록
        학습된다. 그래서 둘을 갈랐다.
        """
        if not (self.frontend or self.ref_frontend):
            return None
        if self._fe is None:
            from ..methods.frontend import FrontEnd
            self._fe = FrontEnd()
        return self._fe

    def _pre_fn(self):
        if self.pre_denoise is None:
            return None
        if self._pre is None:
            import ecgdn.methods  # noqa: F401  (레지스트리 등록)
            from ..registry import build
            self._pre = build(self.pre_denoise)
        return self._pre

    def __len__(self) -> int:
        return len(self.index)

    def set_epoch(self, epoch: int) -> None:
        """epoch 마다 다른 잡음을 뽑되, epoch 을 고정하면 재현 가능하도록."""
        self.salt = ("epoch", epoch)

    def raw_item(self, i: int) -> dict[str, Any]:
        wi = self.index[i]
        rec = self.source.get(wi.record)

        # front-end 를 쓸 때는 창 양쪽에 **실제 이웃 구간**을 여유로 붙여 필터링한 뒤
        # 가운데만 잘라낸다. 0.5 Hz HPF 는 수 초간 링잉하므로 4.096 s 창에 그대로 걸면
        # 창 전체가 트랜지언트가 된다 (docs/01_design.md 2.0 guard band 측정 참조).
        # 참조에만 FE 를 걸어도 guard band 는 필요하다 — 경계 트랜지언트는
        # 어느 쪽을 거르든 똑같이 생긴다.
        m = (int(round(self.fe_margin_s * rec.fs))
             if (self.frontend or self.ref_frontend) else 0)
        lo, hi = wi.start - m, wi.start + self.win + m
        pad_l, pad_r = max(0, -lo), max(0, hi - rec.x.size)
        seg = rec.x[max(lo, 0):min(hi, rec.x.size)]
        if pad_l or pad_r:
            seg = np.pad(seg, (pad_l, pad_r), mode="edge")
        if seg.size < self.win + 2 * m:             # 방어
            seg = np.pad(seg, (0, self.win + 2 * m - seg.size), mode="edge")

        g = make_rng("ds", self.salt, wi.record, wi.start)
        pool = self.noise_kinds + list(self.banks)
        n, weights = mixed_noise(seg.size, rec.fs, g, kinds=pool, banks=self.banks)
        snr = float(g.uniform(*self.snr_range))
        y_seg, _, _ = mix_at_snr(seg, n, snr)

        fe = self._fe_fn()
        if fe is not None and self.frontend:
            y_seg = fe(y_seg, rec.fs)

        pre = self._pre_fn()
        if pre is not None:
            y_seg = pre(y_seg, rec.fs)   # DSP 전처리 (물리 스케일에서 수행)

        # 가운데 창만 사용. target 은 **참조와 같은 대역** 이어야 한다 (위 규약).
        # y 와 동일하게 margin 을 포함한 채 필터링한 뒤 가운데만 잘라낸다.
        x_seg = fe(seg, rec.fs) if (fe is not None and self.ref_frontend) else seg
        y = y_seg[m:m + self.win]
        x = x_seg[m:m + self.win]

        scale = robust_scale(y) if self.normalize else 1.0
        return dict(y=(y / scale).astype(np.float32),
                    x=(x / scale).astype(np.float32),
                    scale=np.float32(scale), snr=np.float32(snr),
                    record=wi.record, start=wi.start, weights=weights, fs=rec.fs)

    def __getitem__(self, i: int):
        d = self.raw_item(i)
        return (d["y"][None, :], d["x"][None, :],
                {"scale": d["scale"], "snr": d["snr"], "record": d["record"],
                 "start": np.int64(d["start"])})


# ------------------------------------------------------------------ 평가 세트
def build_eval_set(source: CleanSource | None = None, split: str = "test",
                   seg_s: float = 60.0, snr_grid: Sequence[float] = (-5, 0, 5, 10, 15, 20),
                   noise_conditions: Sequence[str] = ("mixed",),
                   banks: dict[str, Any] | None = None,
                   n_seg_per_record: int = 3, seed: Any = "eval",
                   ref_frontend: bool = True,
                   out: str | Path | None = None) -> list[dict[str, Any]]:
    """모든 방법이 공유할 **고정** 평가 세트를 만든다.

    각 항목: 참조 x, noisy y, r_peaks, symbols, 조건 라벨.
    `x` 는 **front-end 를 통과한 참조**이고, `x_raw` 는 원본이다. 잡음은 원본에
    섞는다(실제 취득 상황) — 참조만 대역을 맞춘다.
    `out` 이 주어지면 npz 로 저장한다.
    """
    src = source or get_source("auto")
    _fe = None
    if ref_frontend:
        from ..methods.frontend import FrontEnd
        _fe = FrontEnd()
    items: list[dict[str, Any]] = []
    for name in src.records(split):
        rec = src.get(name)
        n_seg = int(round(seg_s * rec.fs))
        if rec.x.size < n_seg:
            continue
        # 기록 안에서 균등 간격으로 구간을 고른다 (결정론적)
        starts = np.linspace(0, rec.x.size - n_seg, n_seg_per_record).astype(int)
        for si, st in enumerate(starts):
            x_raw = rec.x[st:st + n_seg]
            x = _fe(x_raw, rec.fs) if _fe is not None else x_raw
            rp = rec.r_peaks[(rec.r_peaks >= st) & (rec.r_peaks < st + n_seg)] - st
            sy = rec.symbols[(rec.r_peaks >= st) & (rec.r_peaks < st + n_seg)]
            for cond in noise_conditions:
                # **중요**: 잡음 실현은 SNR 과 무관하게 한 번만 뽑고, SNR 로 스케일만 바꾼다.
                # SNR 마다 다른 잡음을 뽑으면 성능-입력SNR 곡선이 '잡음 조성 변화' 와
                # 뒤섞여 해석 불가능해진다. (EXP-A 의 전제)
                g = make_rng(seed, name, si, cond)
                if cond == "mixed":
                    nz, w = mixed_noise(n_seg, rec.fs, g, banks=banks)
                else:
                    from .noise import make_noise
                    nz, w = make_noise(cond, n_seg, rec.fs, g, banks=banks), {cond: 1.0}
                for snr in snr_grid:
                    # 잡음은 **원본** 에 섞는다 — 취득계는 대역제한 전 신호를 본다
                    y, _, _ = mix_at_snr(x_raw, nz, float(snr))
                    items.append(dict(record=name, seg=si, cond=cond, snr=float(snr),
                                      x=x.astype(np.float32),
                                      x_raw=x_raw.astype(np.float32),
                                      y=y.astype(np.float32),
                                      r_peaks=rp.astype(np.int64), symbols=sy,
                                      fs=float(rec.fs), weights=w))
    if out is not None:
        save_eval_set(items, out)
    return items


def save_eval_set(items: list[dict[str, Any]], out: str | Path) -> Path:
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        x=np.stack([it["x"] for it in items]),
        x_raw=np.stack([it.get("x_raw", it["x"]) for it in items]),
        y=np.stack([it["y"] for it in items]),
        fs=np.array([it["fs"] for it in items]),
        snr=np.array([it["snr"] for it in items]),
        record=np.array([it["record"] for it in items]),
        cond=np.array([it["cond"] for it in items]),
        seg=np.array([it["seg"] for it in items]),
        r_peaks=np.array([it["r_peaks"] for it in items], dtype=object),
        symbols=np.array([it["symbols"] for it in items], dtype=object),
        allow_pickle=True)
    return p


def load_eval_set(path: str | Path) -> list[dict[str, Any]]:
    d = np.load(Path(path), allow_pickle=True)
    n = len(d["x"])
    has_raw = "x_raw" in d
    return [dict(x=d["x"][i], x_raw=(d["x_raw"][i] if has_raw else d["x"][i]),
                 y=d["y"][i], fs=float(d["fs"][i]), snr=float(d["snr"][i]),
                 record=str(d["record"][i]), cond=str(d["cond"][i]), seg=int(d["seg"][i]),
                 r_peaks=d["r_peaks"][i], symbols=d["symbols"][i]) for i in range(n)]
