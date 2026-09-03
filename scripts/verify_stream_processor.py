"""R-4 검증 — `StreamProcessor` 가 오프라인 경로와 얼마나 다른가.

    python scripts/verify_stream_processor.py --axis d1

산출: `results/stream_verify.json`

왜 재는가
--------
R-3(`measure_stream_seam.py`)은 스트리밍을 **측정용으로 흉내** 냈다. 거기서는
front-end 를 기록 전체에 한 번 걸었는데, 실제 스트리밍은 그럴 수 없다 —
가진 것이 링버퍼 `win` 샘플뿐이다. `filtfilt` 는 비인과적이라 이 차이가
가장자리에서 나타난다(F-4 는 이 효과가 11.9 dB 까지 간다고 말한다).

그래서 **추정하지 않고 잰다.** 기준선은 재구현이 아니라 실제 `DLDenoiser`
호출이다 — F-23 에서 배운 것이 그것이다.

**두 가지를 따로 잰다. 섞으면 둘 다 틀린 숫자가 나온다.**

인과 front-end 는 영위상 front-end 와 **다른 신호**를 만든다. 보고서의 D1
참조는 `FE_영위상(원본)` 이므로(D-3), 실시간 출력을 그 참조로 재면 **처리
품질이 아니라 참조 정의의 차이**를 재게 된다 — F-12 · F-15 와 같은 계열이다.
반대로 오프라인 출력을 인과 참조로 재면 오프라인이 부당하게 진다. 실제로
두 방향 다 해 봤고 −13.8 dB 와 +14.0 dB 라는 서로 모순되는 표가 나왔다.

그래서 이렇게 가른다:

- **`plumbing_db`** — front-end 를 **양쪽에서 빼고** 잰다. 링버퍼·OLA·
  "언제 내보내는가" 규칙만 남으므로, **스트리밍 배관 자체의 손해**다.
  0 이어야 정상이다.
- **`pipeline_db`** — 실시간 경로 전체(인과 FE + 방법)를 **자기 참조**
  (인과 FE 를 통과한 참값)로 재고, 오프라인 경로 전체를 **자기 참조**
  (영위상 FE 를 통과한 참값)로 재서 그 차이를 본다. 각 파이프라인을 각자의
  기준으로 재는 것이므로 이것이 **공정한 비교**다.
"""
import _bootstrap  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import numpy as np

import ecgdn.methods  # noqa: F401  레지스트리 등록
from ecgdn.realtime import StreamProcessor

ROOT = Path(__file__).resolve().parents[1]
FS = 250.0


def snr_scaled(x: np.ndarray, xh: np.ndarray) -> float:
    """이득 보정 후 출력 SNR. 보고서의 `snr_out_scaled` 와 같은 규약이다."""
    n = min(x.size, xh.size)
    x, xh = x[:n] - x[:n].mean(), xh[:n] - xh[:n].mean()
    a = float(x @ xh) / max(float(xh @ xh), 1e-20)
    e = x - a * xh
    return 10.0 * np.log10(max(float(x @ x), 1e-20) / max(float(e @ e), 1e-20))


def segment(axis: str, seconds: float, seed: int):
    from ecgdn.data.dataset import ECGDenoiseDataset
    from ecgdn.data.sources import get_source
    n = int(seconds * FS)
    src = get_source("synthetic" if axis == "d0" else "mitdb")
    ds = ECGDenoiseDataset(source=src, split="test", win=n, hop=n, max_per_record=2)
    i = int(np.random.default_rng(seed).integers(len(ds)))
    d = ds.raw_item(i)
    return d["y"].astype(np.float64), d["x"].astype(np.float64)


# 인과 FE 설계를 바꿔 보기 위한 전역. **참값 쪽과 처리 쪽이 반드시 같아야**
# 비교가 성립하므로 한 곳에서 들고 있는다 (F-27).
FE_CFG = [None]
FE_MODE = ["causal"]        # causal / zerophase / median (docs/13 · docs/14)


def rt_fe(x: np.ndarray, hop: int, block: int = 25) -> np.ndarray:
    """**실시간 front-end 를 바깥에서 한 번만** 돌린다 — 브리지와 같은 경로다.

    `StreamProcessor(frontend="causal")` 는 처리기 **안에서** FE 를 돌리는데,
    지연이 있는 모드(`zerophase`·`median`)는 출력 길이가 입력과 달라 그 구조에
    안 맞는다. 그리고 실제 시연 경로(`serial_bridge.py`)가 바깥에서 한 번
    도는 구조이므로, **재는 경로와 도는 경로를 같게** 두는 것이 맞다 (F-25).

    반환은 입력보다 `latency_samples` 만큼 짧고, **표본 번호는 0 부터 그대로**다
    (`frontend_modes` 의 계약) — 그래서 잘라내기만 하면 정렬이 맞는다.
    """
    from ecgdn.realtime.frontend_modes import build_fe
    if FE_MODE[0] == "causal" and FE_CFG[0] is not None:
        return causal_fe(x, block)          # 차단/차수를 바꿔 재는 경로 (D-19)
    fe = build_fe(FE_MODE[0], FS, hop_s=hop / FS)
    return np.concatenate([fe.push(x[i:i + block]) for i in range(0, x.size, block)])


def causal_fe(x: np.ndarray, block: int = 25) -> np.ndarray:
    """참값을 **실시간 경로와 같은 인과 FE** 로 거른다."""
    from ecgdn.realtime.frontend_stream import StreamingFrontEnd
    from ecgdn.config import DEFAULT_FE_CAUSAL
    fe = StreamingFrontEnd(FS, FE_CFG[0] or DEFAULT_FE_CAUSAL)
    return np.concatenate([fe.push(x[i:i + block]) for i in range(0, x.size, block)])


def zp_fe(x: np.ndarray) -> np.ndarray:
    """오프라인과 같은 **영위상** front-end 를 통째로 건다."""
    from ecgdn.methods.frontend import FrontEnd
    return np.asarray(FrontEnd()(x, FS), dtype=np.float64)


# front-end 를 끌 수 없는 방법 — **그 필터가 방법의 정의**다(run_exp.FE_INTRINSIC).
# 이들의 실시간 대응물은 `StreamingFrontEnd` **자신**이므로 여기서 잴 것이 없다.
FE_INTRINSIC = {"M_FE", "M01", "M01d"}


def build_nofe(axis: str, mid: str):
    """방법을 **자체 front-end 없이** 만든다 — 앞단의 인과 FE 와 이중이 된다."""
    from ecgdn.registry import build as reg_build
    if mid.startswith("M06"):
        from ecgdn.methods.dl_wrapper import DLDenoiser
        tag = {"M06": "m06_l1", "M06L6": "m06_l6"}[mid]
        return DLDenoiser(ckpt=ROOT / "results" / axis / tag / "best.pt",
                          name="M06", frontend=False)
    return reg_build(mid, use_frontend=False)


def build(axis: str, mid: str):
    from ecgdn.registry import build as reg_build
    if mid.startswith("M06"):
        from ecgdn.methods.dl_wrapper import DLDenoiser
        tag = {"M06": "m06_l1", "M06L6": "m06_l6"}[mid]
        return DLDenoiser(ckpt=ROOT / "results" / axis / tag / "best.pt", name="M06")
    return reg_build(mid)


def per_window_cost(method, buf: np.ndarray, warmup: int = 3, reps: int = 7) -> float:
    """window 하나를 처리하는 **정상 상태** 비용 [s].

    처음에는 첫 호출 한 번을 그대로 썼는데, **PyTorch 의 첫 호출은 정상 상태의
    약 3.5 배**다(할당자·스레드풀·커널 선택이 그때 일어난다). 그 값으로
    `cpu_frac` 을 만들면 `M06` 이 188 % 로 나와 **"실시간을 못 따라간다"** 는
    결론이 된다. 실제로 20 s 를 흘려 재면 벽시계 RTF 가 0.2 대다 (F-26).

    그래서 몇 번 돌려 예열한 뒤 **중앙값**을 쓴다. 평균이 아니라 중앙값인
    이유는 다른 프로세스에 CPU 를 뺏긴 한 번이 평균을 끌어올리기 때문이다.
    """
    for _ in range(warmup):
        method(buf, FS, {})
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        method(buf, FS, {})
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", nargs="*", default=["d0", "d1"])
    ap.add_argument("--methods", nargs="*", default=["M06", "M06L6", "M04"])
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--segments", type=int, default=3)
    ap.add_argument("--grid", nargs="*", default=["12:12", "12:25", "12:64", "50:128"],
                    help="d:hop 쌍")
    ap.add_argument("--hp-hz", type=float, default=None,
                    help="인과 FE 의 고역통과 차단 [Hz]. 기본은 공통 FE 와 같다(0.5)")
    ap.add_argument("--fe-order", type=int, default=None,
                    help="인과 FE 의 차수. 기본은 공통 FE 와 같다(4)")
    ap.add_argument("--fe-mode", default="causal",
                    choices=["causal", "zerophase", "median"],
                    help="실시간 front-end (docs/13 · docs/14)")
    ap.add_argument("--out", default=None,
                    help="기본은 모드별로 다른 파일이다 — 서로 덮어쓰지 않게 (O-20)")
    a = ap.parse_args()

    FE_MODE[0] = a.fe_mode
    if a.out is None:
        a.out = ("results/stream_verify.json" if a.fe_mode == "causal"
                 else f"results/stream_verify_{a.fe_mode}.json")
    if a.hp_hz is not None or a.fe_order is not None:
        from dataclasses import replace as _rep
        from ecgdn.config import DEFAULT_FE_CAUSAL as _BASE
        kw = {}
        if a.hp_hz is not None: kw["hp_hz"] = a.hp_hz
        if a.fe_order is not None: kw["order"] = a.fe_order
        FE_CFG[0] = _rep(_BASE, **kw)
        print(f"[fe] 인과 FE 를 바꿔 잰다: hp={FE_CFG[0].hp_hz} Hz, "
              f"order={FE_CFG[0].order}  (학습은 0.5 Hz 영위상이다 — F-27)")

    rows = []
    fe_share = {}
    for axis in a.axis:
        segs = [segment(axis, a.seconds, s) for s in range(a.segments)]
        # **front-end 의 몫** (docs/15 §5) — 방법과 무관한 항이라 여기서 한 번 잰다.
        # 「실시간 FE 를 통과한 깨끗한 신호」가 「오프라인 FE 를 통과한 것」에서
        # 얼마나 떨어져 있는가. 이것이 모든 방법에 공통으로 얹히는 손해다.
        sh = []
        for (_, x) in segs:
            xr, xz = rt_fe(x, 12), zp_fe(x)
            n = min(xr.size, xz.size)
            k = int(2 * FS)                      # 앞의 warm-up 은 뺀다
            if n - k < FS * 2:
                continue
            sh.append(snr_scaled(xz[k:n], xr[k:n]))
        fe_share[axis] = round(float(np.mean(sh)), 3) if sh else None
        print(f"[{axis}] front-end 의 몫 ({FE_MODE[0]}): "
              f"{fe_share[axis]} dB — 오프라인 FE 대비, 방법과 무관")
        for mid in a.methods:
            if mid in FE_INTRINSIC:
                print(f"[{axis}/{mid}] 건너뜀 — front-end 자체가 방법이라 "
                      f"인과 FE 와 분리할 수 없다. 실시간 대응물은 "
                      f"StreamingFrontEnd 다.")
                continue
            meth = build(axis, mid)
            # 기준선은 **실제 오프라인 호출**이다. 재구현하지 않는다 (F-23).
            # 인과 FE 를 앞단에 두므로 방법 자신의 FE 는 끈다 (두 번 걸리면 안 된다)
            meth_nofe = build_nofe(axis, mid)
            cost = per_window_cost(meth_nofe, segs[0][0][:1024])
            for spec in a.grid:
                d, hop = (int(v) for v in spec.split(":"))
                plumb, pipe = [], []
                for (y, x) in segs:
                    # (1) 배관만: 양쪽 다 front-end 없음. 미리 걸어 둔 신호를 쓴다.
                    yz = zp_fe(y)
                    sp0 = StreamProcessor(meth_nofe, fs=FS, hop=hop, d=d,
                                          frontend="none")
                    o0 = sp0.run(yz)
                    lo, hi = sp0.origin + sp0.win, sp0.origin + o0.size
                    if hi - lo < FS * 2:
                        continue
                    off0 = np.asarray(meth_nofe(yz, FS, {}))
                    xz = zp_fe(x)
                    plumb.append(snr_scaled(xz[lo:hi], o0[lo - sp0.origin:]) -
                                 snr_scaled(xz[lo:hi], off0[lo:hi]))

                    # (2) 파이프라인 전체: 각자의 참조로 잰다.
                    # front-end 는 **바깥에서 한 번** — 브리지와 같은 경로다.
                    yr = rt_fe(y, hop)
                    sp = StreamProcessor(meth_nofe, fs=FS, hop=hop, d=d,
                                         frontend="none")
                    out = sp.run(yr)
                    lo2, hi2 = sp.origin + sp.win, sp.origin + out.size
                    if hi2 - lo2 < FS * 2:
                        continue
                    xc = rt_fe(x, hop)
                    hi2 = min(hi2, xc.size)
                    if hi2 - lo2 < FS * 2:
                        continue
                    s_rt = snr_scaled(xc[lo2:hi2], out[lo2 - sp.origin:hi2 - sp.origin])
                    off = np.asarray(meth(y, FS, {}))       # 보고서가 쓰는 경로
                    s_off = snr_scaled(x[lo2:hi2], off[lo2:hi2])
                    pipe.append(s_rt - s_off)
                if not plumb or not pipe:
                    continue
                rows.append(dict(axis=axis, method=mid, d=d, hop=hop,
                                 delay_ms=round((d + hop) / FS * 1000, 1),
                                 plumbing_db=round(float(np.mean(plumb)), 3),
                                 pipeline_db=round(float(np.mean(pipe)), 3),
                                 spread_db=round(float(np.std(pipe)), 3),
                                 runs_per_s=round(FS / hop, 1),
                                 cpu_frac=round(FS / hop * cost, 4)))
                r = rows[-1]
                print(f"[{axis}/{mid}] d={d:>3} hop={hop:>3}  "
                      f"지연 {r['delay_ms']:>6.1f} ms  "
                      f"배관 {r['plumbing_db']:+6.3f} dB  "
                      f"파이프라인 {r['pipeline_db']:+6.2f} dB "
                      f"(±{r['spread_db']:.2f})  "
                      f"추론 {r['runs_per_s']:>5.1f}/s  CPU {r['cpu_frac']*100:.0f}%",
                      flush=True)

    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    from ecgdn.config import DEFAULT_FE_CAUSAL
    cfg = FE_CFG[0] or DEFAULT_FE_CAUSAL
    # **어느 FE 로 잰 표인지 산출물이 스스로 말해야 한다.** 안 적으면 두 표를
    # 나란히 놓았을 때 어느 쪽이 어느 설정인지 알 수 없다 (F-27).
    p.write_text(json.dumps(dict(fs=FS, seconds=a.seconds, segments=a.segments,
                                 fe_mode=a.fe_mode, fe_share_db=fe_share,
                                 fe_hp_hz=cfg.hp_hz, fe_order=cfg.order,
                                 rows=rows), indent=1, ensure_ascii=False))
    print(f"\n-> {a.out}  {len(rows)} 행")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
