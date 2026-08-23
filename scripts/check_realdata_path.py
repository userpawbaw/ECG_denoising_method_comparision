#!/usr/bin/env python3
"""실데이터 경로(STEP 15~17) 예행 검증 — PhysioNet 없이.

왜 필요한가
----------
원격 세션에서 PhysioNet 이 egress 정책으로 차단돼 있어, `data/raw/mitdb` 를 읽는
코드 경로(`mitdb.load_record` / `MITDBSource` / `NoiseBank` / `ECGDenoiseDataset`)는
**단 한 번도 실행된 적이 없다**. 실데이터가 도착한 순간 여기서 처음 터지면
디버깅이 사용자 쪽으로 넘어간다.

그래서 WFDB 포맷 fixture 를 직접 만들어(360 Hz, MLII/V1 2채널, .atr annotation)
실데이터와 **같은 코드 경로**를 통과시킨다. 신호의 내용은 합성이지만,
포맷·리샘플·annotation 정렬·split·잡음 bank 는 실물과 동일하게 검증된다.

    python scripts/check_realdata_path.py            # fixture 생성 + 검증
    python scripts/check_realdata_path.py --keep     # fixture 를 지우지 않음

실데이터가 실제로 있으면 `--real` 로 같은 검사를 진짜 데이터에 돌린다.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

from ecgdn.config import FS, FS_MITDB
from ecgdn.data.splits import MITDB_ALL, MITDB_SPLIT

FIX_FS = FS_MITDB          # 360 Hz — MIT-BIH 원본과 동일
FIX_DUR_S = 40.0           # fixture 는 짧게. 포맷 검증이 목적이지 성능 측정이 아니다.
NOISE_DUR_S = 90.0         # split(0.6/0.15/0.25) 을 나눠도 각 구간이 충분히 길도록

# MLII 가 없는 실제 기록 — lead fallback 경로를 fixture 에서도 재현한다.
NO_MLII = {"102", "104"}

_FAILS: list[str] = []
_WARNS: list[str] = []


def check(cond: bool, msg: str) -> bool:
    print(f"  {'OK  ' if cond else 'FAIL'}  {msg}")
    if not cond:
        _FAILS.append(msg)
    return cond


def warn(msg: str) -> None:
    print(f"  WARN  {msg}")
    _WARNS.append(msg)


# ---------------------------------------------------------------- fixture 생성
def _fake_ecg(dur_s: float, fs: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(신호 mV, R-peak 표본 인덱스, beat symbol). 합성기를 fs 로 직접 돌린다."""
    from ecgdn.config import DEFAULT_KERNEL, PVC_KERNEL
    from ecgdn.data.synthetic import jitter_kernel, synth_ecg
    from ecgdn.utils import rng as make_rng

    g = make_rng("fixture", seed)
    s = synth_ecg(dur_s, fs=fs, hr_bpm=60.0 + (seed * 11) % 40,
                  hrv_std=0.03, kernel=jitter_kernel(DEFAULT_KERNEL, g),
                  pvc_kernel=PVC_KERNEL, pvc_prob=0.08, seed=7000 + seed)
    sym = np.asarray(s.beat_labels[:len(s.r_peaks)], dtype=object)
    # synth 의 라벨을 WFDB symbol 로 옮긴다.
    m = {"N": "N", "V": "V", "normal": "N", "pvc": "V"}
    sym = np.asarray([m.get(str(t), "N") for t in sym])
    return s.x, np.asarray(s.r_peaks, dtype=np.int64), sym


def build_fixture(root: Path) -> None:
    """MIT-BIH / NSTDB 와 같은 디렉터리 구조·포맷의 fixture 를 쓴다."""
    import wfdb

    mit, nst = root / "mitdb", root / "nstdb"
    mit.mkdir(parents=True, exist_ok=True)
    nst.mkdir(parents=True, exist_ok=True)

    print(f"[fixture] MIT-BIH 형식 기록 {len(MITDB_ALL)}개 -> {mit}")
    for i, name in enumerate(MITDB_ALL):
        x, rp, sym = _fake_ecg(FIX_DUR_S, FIX_FS, i)
        # 2채널: 실제 MIT-BIH 처럼 MLII + V1. 일부 기록은 MLII 가 없다.
        lead0 = "V5" if name in NO_MLII else "MLII"
        sig = np.stack([x, 0.6 * np.roll(x, 13)], axis=1)
        wfdb.wrsamp(record_name=name, fs=float(FIX_FS), units=["mV", "mV"],
                    sig_name=[lead0, "V1"], p_signal=sig, fmt=["212", "212"],
                    write_dir=str(mit))
        wfdb.wrann(name, "atr", rp, np.asarray(sym), write_dir=str(mit))

    print(f"[fixture] NSTDB 잡음 기록 3개 -> {nst}")
    for j, kind in enumerate(("bw", "ma", "em")):
        g = np.random.default_rng(4242 + j)
        n = int(NOISE_DUR_S * FIX_FS)
        # 종류별로 성격이 다른 잡음 (bw=저주파, ma=광대역, em=혼합)
        t = np.arange(n) / FIX_FS
        if kind == "bw":
            v = np.sin(2 * np.pi * 0.25 * t) + 0.4 * np.sin(2 * np.pi * 0.05 * t)
        elif kind == "ma":
            v = g.standard_normal(n)
        else:
            v = 0.5 * np.sin(2 * np.pi * 0.3 * t) + 0.5 * g.standard_normal(n)
        sig = np.stack([v, np.roll(v, 7)], axis=1) * 0.5
        wfdb.wrsamp(record_name=kind, fs=float(FIX_FS), units=["mV", "mV"],
                    sig_name=["noise1", "noise2"], p_signal=sig, fmt=["16", "16"],
                    write_dir=str(nst))


# ---------------------------------------------------------------- 검사
def check_loader(mit_root: Path) -> None:
    from ecgdn.data.mitdb import available_records, load_record

    print("\n[1] mitdb.load_record — 리샘플 · annotation 정렬 · lead fallback")
    avail = available_records(mit_root)
    check(set(avail) >= set(MITDB_ALL),
          f"48개 기록이 모두 보인다 ({len(avail)}개 발견)")

    name = MITDB_SPLIT["test"][0]
    r = load_record(name, mit_root)
    check(abs(r.fs - FS) < 1e-9, f"{name}: fs 가 {FS:g} Hz 로 리샘플됐다 (원본 {r.fs_orig:g})")

    exp_n = int(round(r.meta["n_orig"] * FS / r.fs_orig))
    check(abs(r.x.size - exp_n) <= 2,
          f"{name}: 길이 {r.x.size} ≈ 기대값 {exp_n} (리샘플 비율 일치)")
    check(r.lead == "MLII", f"{name}: MLII 를 골랐다 (lead={r.lead})")
    check(np.isfinite(r.x).all(), f"{name}: NaN/Inf 없음")
    check(0.05 < float(np.std(r.x)) < 20.0,
          f"{name}: 물리단위 mV 스케일로 보인다 (std={np.std(r.x):.3f})")

    # annotation 이 리샘플 격자 위에서도 R-peak 를 가리키는가?
    # 360 -> 250 Hz 는 정수배가 아니라 반올림 오차가 남는다. 그 오차가
    # **양자화 수준(±1 표본, 4 ms)** 을 넘지 않아야 beat 층화 평가가 성립한다.
    # 이 검사를 빼면 annotation 이 통째로 밀려도 아무도 모른다 (mitdb.py 모듈 주석).
    w = int(round(0.04 * FS))
    offs = []
    for p in r.r_peaks[3:-3]:
        lo, hi = max(int(p) - w, 0), min(int(p) + w + 1, r.x.size)
        if hi - lo < 3:
            continue
        seg = np.abs(r.x[lo:hi] - np.median(r.x))
        offs.append(int(seg.argmax()) + lo - int(p))
    offs = np.asarray(offs)
    check(offs.size >= 20, f"{name}: 정렬 검사에 쓸 beat 가 충분하다 (n={offs.size})")
    if offs.size:
        frac = float(np.mean(np.abs(offs) <= 2))
        bias_ms = float(np.median(offs)) / FS * 1000.0
        check(frac >= 0.90,
              f"{name}: beat 의 {frac:.0%} 가 annotation ±2 표본 안에서 최대 진폭을 갖는다")
        check(abs(bias_ms) <= 8.0,
              f"{name}: 계통적 밀림이 없다 (중앙값 offset {bias_ms:+.1f} ms, 허용 ±8)")

    # lead fallback
    nm = sorted(NO_MLII & set(MITDB_ALL))
    if nm:
        r2 = load_record(nm[0], mit_root)
        check(r2.meta["lead_fallback"] and r2.lead != "MLII",
              f"{nm[0]}: MLII 가 없어 첫 채널({r2.lead})로 대체하고 그 사실을 기록했다")

    # beat symbol 이 살아있는가
    syms = set(np.unique(r.symbols).tolist())
    check(len(syms) >= 1 and syms <= set("NLRejAaJSVEF/fQ"),
          f"{name}: beat symbol 이 유효하다 {sorted(syms)}")
    check(r.r_peaks.size == r.symbols.size, f"{name}: r_peaks 와 symbols 길이가 같다")
    check((np.diff(r.r_peaks) > 0).all(), f"{name}: r_peaks 가 단조증가한다")


def check_source(mit_root: Path) -> None:
    from ecgdn.data.sources import MITDBSource, get_source

    print("\n[2] sources.get_source — auto 가 실데이터를 집는가")
    src = get_source("auto", root=mit_root)
    check(isinstance(src, MITDBSource) and src.kind == "mitdb",
          f"root 에 .hea 가 있으면 MITDBSource 를 고른다 (kind={src.kind})")

    print("\n[3] split × 기록 존재 — DS1/DS2 전 기록이 실제로 읽히는가")
    missing, bad = [], []
    for split in ("train", "val", "test"):
        for name in src.records(split):
            try:
                rec = src.get(name)
                if rec.x.size < int(10 * FS) or rec.r_peaks.size < 5:
                    bad.append(name)
            except Exception as e:                       # noqa: BLE001
                missing.append(f"{name}({type(e).__name__})")
    check(not missing, f"train/val/test 전 기록이 예외 없이 적재된다 (실패 {missing[:5]})")
    check(not bad, f"모든 기록이 최소 길이·beat 수를 만족한다 (미달 {bad[:5]})")


def check_banks(nst_root: Path) -> None:
    from ecgdn.data.nstdb import NSTDB_KINDS, make_banks
    from ecgdn.data.splits import NOISE_SPLIT_FRAC

    print("\n[4] nstdb.NoiseBank — 실잡음 적재와 시간축 disjoint split")
    got = {s: make_banks(s, nst_root) for s in ("train", "val", "test")}
    for s, b in got.items():
        check(set(b) == set(NSTDB_KINDS), f"{s}: bw/ma/em 3종이 모두 만들어졌다 ({sorted(b)})")
    if not all(set(b) == set(NSTDB_KINDS) for b in got.values()):
        return

    # split 구간이 겹치지 않는가 (leakage 방지의 핵심)
    for k in NSTDB_KINDS:
        rs = [got[s][k].range for s in ("train", "val", "test")]
        ok = rs[0][1] <= rs[1][0] and rs[1][1] <= rs[2][0]
        check(ok, f"{k}: train/val/test 구간이 시간축에서 겹치지 않는다 {rs}")

    g = np.random.default_rng(0)
    for k in NSTDB_KINDS:
        v = got["train"][k].sample(1024, g)
        check(v.size == 1024 and np.isfinite(v).all()
              and abs(float(np.std(v)) - 1.0) < 0.05,
              f"{k}: sample() 이 단위분산 crop 을 준다 (std={np.std(v):.4f})")

    # 요청 길이가 구간보다 길 때 타일링 경로
    long = got["val"][NSTDB_KINDS[0]]
    v = long.sample(long.x.size + 5000, g)
    check(v.size == long.x.size + 5000 and np.isfinite(v).all(),
          "구간보다 긴 요청도 타일링으로 처리한다")


def check_dataset(mit_root: Path, nst_root: Path) -> None:
    from ecgdn.data.dataset import ECGDenoiseDataset
    from ecgdn.data.nstdb import make_banks
    from ecgdn.data.sources import get_source

    print("\n[5] ECGDenoiseDataset — 실기록 + 실잡음으로 배치가 만들어지는가")
    src = get_source("auto", root=mit_root)
    banks = make_banks("train", nst_root)
    ds = ECGDenoiseDataset(source=src, split="train", banks=banks,
                           max_per_record=4, salt="fixture")
    check(len(ds) > 0, f"윈도우 인덱스가 만들어졌다 (n={len(ds)})")

    y, x = ds[0][:2]
    from ecgdn.config import WIN
    check(y.shape == x.shape and y.shape == (1, WIN),
          f"(noisy, clean) 이 채널 우선 (1, WIN) 으로 나온다 {y.shape}")
    check(np.isfinite(y).all() and np.isfinite(x).all(), "NaN/Inf 없음")

    # 결정론: 같은 salt 로 두 번 만들면 비트 단위로 같아야 한다
    ds2 = ECGDenoiseDataset(source=src, split="train", banks=banks,
                            max_per_record=4, salt="fixture")
    y2, x2 = ds2[0][:2]
    check(np.array_equal(y, y2) and np.array_equal(x, x2),
          "같은 salt → 비트 단위로 동일한 배치 (결정론)")

    # 전 윈도우를 훑어 zero-power 잡음 같은 사고가 없는지 (F-1 회귀)
    bad = 0
    for i in range(min(len(ds), 200)):
        try:
            a, b = ds[i][:2]
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                bad += 1
        except Exception as e:                            # noqa: BLE001
            print(f"        window {i}: {type(e).__name__}: {e}")
            bad += 1
    check(bad == 0, f"앞 200 윈도우가 예외·NaN 없이 생성된다 (문제 {bad}개)")

    # 실잡음 bank 를 실제로 쓰는가 (합성 잡음으로 조용히 대체되면 의미가 없다)
    ds_nb = ECGDenoiseDataset(source=src, split="train", banks=None,
                              max_per_record=4, salt="fixture")
    check(not np.array_equal(ds_nb[0][0], y),
          "banks 를 주면 잡음이 실제로 달라진다 (실잡음이 무시되지 않는다)")


def check_eval(mit_root: Path, nst_root: Path) -> None:
    print("\n[6] 평가 엔진 — 실기록 한 구간에 전 방법을 통과시켜 본다")
    from ecgdn.data.mixer import mix_at_snr
    from ecgdn.data.nstdb import make_banks
    from ecgdn.data.sources import get_source
    from ecgdn.eval.engine import trim_guard
    from ecgdn.eval.signal_metrics import snr_out_scaled
    from ecgdn.registry import build

    src = get_source("auto", root=mit_root)
    rec = src.get(MITDB_SPLIT["test"][0])
    banks = make_banks("test", nst_root)
    g = np.random.default_rng(1)
    n = banks["ma"].sample(rec.x.size, g)
    y, _, _ = mix_at_snr(rec.x, n, 10.0)

    for mid in ("M_FE", "M01", "M04", "M05"):
        try:
            xh = build(mid)(y, rec.fs)
            sl = trim_guard(rec.x.size, rec.fs)      # 양끝 guard 를 뺀 slice
            s = snr_out_scaled(rec.x[sl], np.asarray(xh).ravel()[sl])
            check(np.isfinite(s), f"{mid}: 실기록에서 동작 (snr_out_scaled={s:+.2f} dB)")
        except Exception as e:                            # noqa: BLE001
            check(False, f"{mid}: {type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="store_true",
                    help="fixture 대신 실제 data/raw 를 검사한다")
    ap.add_argument("--root", default="data/raw", help="--real 일 때의 경로")
    ap.add_argument("--keep", action="store_true", help="fixture 를 지우지 않는다")
    a = ap.parse_args()

    if a.real:
        root = Path(a.root)
        if not (root / "mitdb").glob("*.hea"):
            print(f"실데이터가 없다: {root}/mitdb")
            return 2
        print(f"=== 실데이터 검증: {root} ===")
        tmp = None
    else:
        tmp = Path(tempfile.mkdtemp(prefix="ecgdn_fixture_"))
        root = tmp
        print(f"=== fixture 검증: {root} ===")
        print("(신호는 합성이지만 포맷·경로는 MIT-BIH/NSTDB 와 동일하다)\n")
        build_fixture(root)

    try:
        mit, nst = root / "mitdb", root / "nstdb"
        check_loader(mit)
        check_source(mit)
        check_banks(nst)
        check_dataset(mit, nst)
        check_eval(mit, nst)
    finally:
        if tmp is not None and not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)
        elif tmp is not None:
            print(f"\nfixture 보존: {tmp}")

    print("\n" + "=" * 60)
    if _FAILS:
        print(f"FAIL {len(_FAILS)}건:")
        for m in _FAILS:
            print(f"  - {m}")
        return 1
    print(f"전 항목 통과 ({len(_WARNS)} warn). 실데이터 경로는 코드 상 준비됐다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
