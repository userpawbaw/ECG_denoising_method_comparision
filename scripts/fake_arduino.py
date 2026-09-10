#!/usr/bin/env python3
"""R-7: **가상 아두이노.** 진짜 시리얼 포트 뒤에 펌웨어를 놓는다.

    # 1) 가상 보드를 띄운다 (포트 이름을 찍어 준다)
    python3 scripts/fake_arduino.py --fs 500 --port-file /tmp/fakeuno

    # 2) 다른 창에서, **실제 보드에 쓰는 그 명령 그대로**
    python3 scripts/serial_bridge.py --port $(cat /tmp/fakeuno) \\
            --board-fs 250 --methods M_FE,M04 --serve

왜 이것이 필요한가 — `--replay` 와 무엇이 다른가
------------------------------------------------
`--replay` 는 브리지 **안에서** 바이트를 만든다. 그래서 파서·처리기·화면은
검증되지만 시연 당일 실제로 타는 길, 즉

    pyserial 이 포트를 연다 → 보드가 리셋된다 → 명령을 보낸다 →
    OS 입력 버퍼에서 읽는다

는 **한 줄도 실행되지 않는다.** 이 스크립트는 `os.openpty()` 로 **커널이 만든
진짜 tty 한 쌍**을 열고 한쪽에 `FakeBoard` 를 붙인다. 브리지가 여는 것은
`/dev/pts/N` 이고, 브리지 입장에서 이것은 `/dev/ttyACM0` 과 구별되지 않는다.

> **윈도우에는 `openpty` 가 없다.** com0com(무료) 이나 VSPE 로 `COM8↔COM9`
> 쌍을 만든 뒤 `--attach COM8` 로 붙이면 아래와 같은 일이 된다. 리눅스·맥에서
> `socat -d -d pty,raw,echo=0 pty,raw,echo=0` 을 쓰는 경우도 같다.

무엇을 확인할 수 있게 되는가
---------------------------
1. **명령 규약** — 브리지가 보내는 `'2'`·`'b'` 를 펌웨어가 읽는 순서 그대로 받는다.
2. **부팅 잡음** — 포트를 열면 보드가 리셋되고 `# logger ready` 를 흘린다.
   그것을 버리지 않으면 첫 몇 초가 쓰레기다.
3. **송신 버퍼 포화** — baud·fs·형식이 정하는 진짜 드롭. 확률이 아니다.
4. **포트가 배타 자원이라는 것** — 두 번째 프로세스는 못 붙는다.

`hardware/arduino_ecg_logger/arduino_ecg_logger.ino` 와의 일치는
`tests/test_fake_board.py` 가 고정한다.
"""
import _bootstrap  # noqa: F401

import argparse
import os
import sys
import time
from pathlib import Path

from ecgdn.realtime.fake_board import FakeBoard, synth_counts


def open_virtual_port() -> tuple[int, int, str]:
    """PTY 한 쌍을 연다. 돌려주는 이름이 «가상 아두이노가 꽂힌 포트» 다."""
    import pty
    import tty

    master, slave = pty.openpty()
    name = os.ttyname(slave)
    # **에코를 끈다.** 안 끄면 브리지가 보낸 `'2'`·`'b'` 가 그대로 되돌아오고,
    # 바이너리 파서는 그것을 «깨진 프레임» 으로 센다 — 없는 고장이 보인다.
    tty.setraw(master)
    tty.setraw(slave)
    os.set_blocking(master, False)
    return master, slave, name


def attach_port(port: str, baud: int) -> tuple[int, int, str]:
    """이미 있는 포트(com0com·socat 이 만든 쪽)에 붙는다."""
    import serial

    ser = serial.Serial(port, baud, timeout=0)
    return ser.fileno(), -1, port


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs", type=int, default=500, choices=[250, 500, 1000],
                    help="켜질 때의 fs. 펌웨어 기본은 500 이고 PC 가 명령으로 바꾼다")
    ap.add_argument("--baud", type=int, default=115200,
                    help="스케치의 SERIAL_BAUD. **드롭이 여기서 정해진다**")
    ap.add_argument("--mode", default="ascii", choices=["ascii", "bin"],
                    help="켜질 때의 형식. 펌웨어 기본은 ascii 다")
    ap.add_argument("--drift-ppm", type=float, default=3000.0,
                    help="보드 클럭 오차. Uno 의 세라믹 레조네이터는 약 ±5000 ppm")
    ap.add_argument("--leadoff-every", type=float, default=0.0,
                    help="N 초마다 전극이 떨어진다 [s]")
    ap.add_argument("--leadoff-len", type=float, default=0.7)
    ap.add_argument("--dur", type=float, default=0.0, help="0 이면 무한")
    ap.add_argument("--signal-s", type=float, default=60.0,
                    help="합성 신호 길이 [s]. 끝나면 처음으로 돌아간다")
    ap.add_argument("--snr-db", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--firmware", default="current", choices=["current", "old"],
                    help="old 면 fs·형식 명령을 무시한다 — 구 스케치가 꽂힌 판")
    ap.add_argument("--attach", help="PTY 대신 이 포트에 붙는다 (com0com·socat)")
    ap.add_argument("--port-file", help="포트 이름을 이 파일에 적는다")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    counts = synth_counts(args.signal_s, args.fs, seed=args.seed,
                          snr_db=args.snr_db)
    board = FakeBoard(counts, fs=args.fs, baud=args.baud, mode=args.mode,
                      drift_ppm=args.drift_ppm,
                      leadoff_every_s=args.leadoff_every,
                      leadoff_len_s=args.leadoff_len,
                      accept_commands=(args.firmware == "current"))

    if args.attach:
        fd, slave, name = attach_port(args.attach, args.baud)
    else:
        fd, slave, name = open_virtual_port()
    if args.port_file:
        Path(args.port_file).write_text(name + "\n")
    print(f"가상 아두이노: {name}  (fs {args.fs} Hz · {args.baud} baud · {args.mode})")
    print(f"  브리지에 이렇게 붙인다:\n"
          f"    python3 scripts/serial_bridge.py --port {name} "
          f"--board-fs 250 --methods M_FE,M04 --serve")
    sys.stdout.flush()

    t0 = time.perf_counter()
    t_log = t0
    n_out = 0
    try:
        while True:
            now = time.perf_counter()
            if args.dur and now - t0 > args.dur:
                break
            try:
                cmd = os.read(fd, 256)
            except BlockingIOError:
                cmd = b""
            except OSError:
                cmd = b""
            if cmd:
                board.feed_command(cmd)
                if not args.quiet:
                    print(f"\n  <- 명령 {cmd!r} · fs {board.fs} · {board.mode}")
            data = board.poll(now)
            if data:
                try:
                    n_out += os.write(fd, data)
                except BlockingIOError:
                    # PC 가 안 읽어 OS 버퍼가 찼다. **보드는 기다려 주지 않는다** —
                    # 실제 보드에서도 이 자리는 잃는 자리다.
                    pass
                except OSError:
                    break
            if not args.quiet and now - t_log >= 1.0:
                t_log = now
                print(f"\r  {now - t0:6.1f}s  보낸 샘플 {board.n_sent:8d}  "
                      f"버퍼드롭 {board.dropped:6d}  "
                      f"({100 * board.dropped / max(board.k, 1):5.2f} %)  "
                      f"{n_out / max(now - t0, 1e-9) / 1024:5.2f} kB/s",
                      end="", flush=True)
            time.sleep(0.002)
    except KeyboardInterrupt:
        pass
    finally:
        if slave >= 0:
            os.close(slave)
        os.close(fd)
        if args.port_file:
            Path(args.port_file).unlink(missing_ok=True)
    el = max(time.perf_counter() - t0, 1e-9)
    print(f"\n보드가 뜬 샘플 {board.k} · 선으로 나간 것 {board.n_sent} · "
          f"버퍼가 없어 버린 것 {board.dropped} "
          f"({100 * board.dropped / max(board.k, 1):.2f} %) · "
          f"실측 {board.k / el:.1f} Hz (설정 {board.fs} Hz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
