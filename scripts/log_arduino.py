"""STEP 28: Arduino 시리얼 로거 — 규격에 맞는 CSV 를 만든다.

    python scripts/log_arduino.py --port /dev/ttyACM0 --session S1 --dur 300 \
        --fs 500 --gain 1100 --note "seated, still"

헤더 3줄을 반드시 붙인다. fs_hz 를 파일에 안 적으면 나중에 100% 문제가 된다.
"""
import _bootstrap  # noqa: F401

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from ecgdn.data.arduino import SESSIONS
from ecgdn.utils import ensure_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="예: /dev/ttyACM0, COM3")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--session", required=True, choices=sorted(SESSIONS))
    ap.add_argument("--dur", type=float, default=300.0, help="기록 길이 [s]")
    ap.add_argument("--fs", type=float, required=True, help="스케치의 FS_HZ 와 같은 값")
    ap.add_argument("--adc-bits", type=int, default=10)
    ap.add_argument("--vref", type=float, default=5.0)
    ap.add_argument("--gain", type=float, default=1100.0, help="아날로그 프런트엔드 총 이득")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default="data/arduino")
    args = ap.parse_args()

    try:
        import serial
    except ImportError:
        print("pyserial 이 필요하다:  pip install pyserial")
        return 1

    d = ensure_dir(args.out)
    idx = len(list(d.glob(f"arduino_{args.session}_*.csv"))) + 1
    path = d / f"arduino_{args.session}_{datetime.now():%Y%m%d}_{idx:02d}.csv"

    print(f"session {args.session}: {SESSIONS[args.session]}")
    print(f"-> {path}  ({args.dur:.0f} s @ {args.fs:g} Hz)")
    ser = serial.Serial(args.port, args.baud, timeout=2.0)
    time.sleep(2.0)                       # 보드 리셋 대기
    ser.reset_input_buffer()

    n_lo = n = 0
    t0 = time.perf_counter()
    with path.open("w") as f:
        f.write(f"# fs_hz={args.fs:g}\n")
        f.write(f"# adc_bits={args.adc_bits}, vref_v={args.vref:g}, gain={args.gain:g}\n")
        f.write(f"# session={args.session}, note={args.note or SESSIONS[args.session]}\n")
        f.write("t_ms,adc_raw\n")
        while time.perf_counter() - t0 < args.dur:
            ln = ser.readline().decode("ascii", "ignore").strip()
            if not ln or ln.startswith("#") or "," not in ln:
                continue
            f.write(ln + "\n")
            n += 1
            if ln.rsplit(",", 1)[-1] == "-1":
                n_lo += 1
            if n % int(args.fs) == 0:
                el = time.perf_counter() - t0
                sys.stdout.write(f"\r  {el:6.1f}s  {n:8d} samples  "
                                 f"lead-off {100*n_lo/max(n,1):5.1f}%")
                sys.stdout.flush()
    ser.close()
    el = time.perf_counter() - t0
    print(f"\n완료: {n} samples, 실측 {n/el:.1f} Hz (설정 {args.fs:g} Hz), "
          f"lead-off {100*n_lo/max(n,1):.1f}%")
    if abs(n / el - args.fs) / args.fs > 0.05:
        print("[warn] 실측 샘플링률이 설정과 5% 이상 다르다 — Serial 속도나 스케치를 점검할 것.")
    if n_lo / max(n, 1) > 0.02:
        print("[warn] lead-off 구간이 2% 를 넘는다 — 전극 접촉을 다시 확인할 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
