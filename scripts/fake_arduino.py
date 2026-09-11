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
2. **DTR 리셋** — 포트를 열면 보드가 처음부터 시작한다. 부트로더가 조용한
   1.6 초와 그 뒤의 `# logger ready` 까지(아래 «리셋을 어떻게 흉내내나»).
3. **USB 도착 뭉침** — 바이트는 폴링 간격 단위의 덩어리로 온다(`UsbPipe`).
4. **송신 버퍼 포화** — baud·fs·형식이 정하는 진짜 드롭. 확률이 아니다.
5. **포트가 배타 자원이라는 것** — 두 번째 프로세스는 못 붙는다.

리셋을 어떻게 흉내내나 — **DTR 자체는 못 한다**
----------------------------------------------
PTY 에는 modem control line 이 없다. `TIOCMGET` 이 **ENOTTY** 를 내고,
pyserial 에서 `ser.dtr = False` 를 하면 예외가 난다 `[측정]`. 그래서 「DTR 이
토글된다」는 사건 자체는 흉내낼 수 없다.

**대신 그 결과를 흉내낸다.** PTY master 는 slave 를 아무도 안 열고 있으면
`read` 가 **EIO** 를 내고, 누군가 열면 정상으로 바뀐다. 그 전환이 곧 「포트가
열렸다」이고, 실제 Uno 에서 리셋을 일으키는 사건과 같은 자리다. 그래서 이
스크립트는 slave 를 **붙들지 않고 놓는다**(`--hold-open` 으로 예전처럼 붙들 수
있다) — 그러면 브리지가 열 때마다 보드가 처음부터 시작한다.

여전히 **실보드에서만 볼 수 있는 것**: DTR 을 꺼서 리셋을 막는 경로
(`ser.dtr = False`), 그리고 드라이버·권한(dialout) 문제.

`hardware/arduino_ecg_logger/arduino_ecg_logger.ino` 와의 일치는
`tests/test_fake_board.py` 가 고정한다.
"""
import _bootstrap  # noqa: F401

import argparse
import errno
import os
import sys
import time
from pathlib import Path

from ecgdn.realtime.fake_board import (BOOTLOADER_S, FakeBoard, UsbPipe,
                                       record_counts, synth_counts)


def open_virtual_port(hold_open: bool = False) -> tuple[int, int, str]:
    """PTY 한 쌍을 연다. 돌려주는 이름이 «가상 아두이노가 꽂힌 포트» 다.

    `hold_open=False` 면 slave 를 놓는다 — 그래야 **누가 포트를 여는지 보인다**
    (master 의 EIO 가 풀리는 순간). 그것이 DTR 리셋의 자리다.
    """
    import pty
    import tty

    master, slave = pty.openpty()
    name = os.ttyname(slave)
    # **에코를 끈다.** 안 끄면 브리지가 보낸 `'2'`·`'b'` 가 그대로 되돌아오고,
    # 바이너리 파서는 그것을 «깨진 프레임» 으로 센다 — 없는 고장이 보인다.
    tty.setraw(master)
    tty.setraw(slave)
    os.set_blocking(master, False)
    if not hold_open:
        os.close(slave)
        slave = -1
    return master, slave, name


def attach_port(port: str, baud: int) -> tuple[int, int, str]:
    """이미 있는 포트(com0com·socat 이 만든 쪽)에 붙는다.

    이쪽은 **열림 감지가 안 된다** — 상대가 열든 말든 우리 fd 는 멀쩡하다.
    그래서 `--attach` 에서는 리셋 흉내가 꺼진다.
    """
    import serial

    ser = serial.Serial(port, baud, timeout=0)
    return ser.fileno(), -1, port


def print_choices() -> int:
    """`--list` — **무엇을 고를 수 있는지 한 화면에.**

    이것이 없으면 「`--noise bw2` 가 왜 안 되나」를 스택트레이스로 배우게 된다.
    """
    from ecgdn.data.mitdb import available_records
    from ecgdn.data.noise import NOISE_FNS
    from ecgdn.data.nstdb import NSTDB_KINDS
    from ecgdn.data.splits import MITDB_SPLIT

    have = set(available_records())
    print("== 기록 (--record) ==  디스크에 %d 개" % len(have))
    note = {"test": "**시연은 여기서 고른다** — 모델이 학습에 안 쓴 기록",
            "train": "모델이 학습한 기록. 쓰면 딥러닝이 실제보다 잘 나온다",
            "val": "모델 선택에 쓴 기록",
            "paced": "페이스메이커 — D1 평가에서 제외했다"}
    for split in ("test", "train", "val", "paced"):
        names = [r for r in MITDB_SPLIT.get(split, ()) if r in have]
        miss = len(MITDB_SPLIT.get(split, ())) - len(names)
        print(f"  {split:6s} ({len(names):2d}개{', 없는 것 %d' % miss if miss else ''}) "
              f"{note[split]}")
        print(f"         {' '.join(names)}")

    print("\n== 잡음 (--noise) ==")
    print(f"  NSTDB 실측 : {' '.join(sorted(NSTDB_KINDS))}"
          "        <- 30 분 실측 녹음. 보고서의 D1 이 쓰는 것")
    print(f"  합성       : {' '.join(sorted(NOISE_FNS))}")
    print("  mixed      : 위에서 1~3 종을 랜덤 가중 합성 (학습 파이프라인과 같은 구성)")
    print("  none       : 잡음 없이 기록 그대로 — 「기법이 깨끗한 신호를 망치나」")

    print("\n== 방법 (브리지의 --methods) ==")
    print("  M_FE  front-end 출력 그대로 (필터가 곧 방법)")
    for mid, label in (("M00", "항등 — 배관 점검용"),
                       ("M01", "대역통과 0.5-40 Hz + 자동 notch"),
                       ("M02", "Savitzky-Golay"), ("M03", "DWT soft threshold"),
                       ("M04", "SWT 적응 임계 (고전 최강)"),
                       ("M05", "Sameni EKS")):
        print(f"  {mid:5s} {label}")
    print("  딥러닝: M06 · M06L6 · M08 · M08L6 · M09   "
          "(results/<축>/<태그>/best.pt 가 있어야 한다)")

    print("\n== 이런 것을 보고 싶으면 ==")
    for want, cmd in (
        ("기본 — 실측 잡음에서 방법이 갈리는 것",
         "--record 100 --noise bw --snr-db 6"),
        ("전원선 잡음이 지워지는 것 (M01 의 자동 notch)",
         "--record 103 --noise pli --snr-db 6"),
        ("근전도가 QRS 를 먹는 어려운 판",
         "--record 105 --noise ma --snr-db 0"),
        ("기저선이 크게 흔들리는 판",
         "--record 111 --noise bw --snr-db 0"),
        ("깨끗한 신호를 망치지 않는가",
         "--record 100 --noise none"),
        ("깊은 잡음 — 어디서 무너지나",
         "--record 200 --noise mixed --snr-db -5 --seed 3"),
    ):
        print(f"  {want}\n      … --source d1 {cmd}")
    print("\n  (`mixed` 는 seed 마다 성분이 바뀐다. 같은 그림을 다시 보려면 "
          "--seed 를 고정한다)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs", type=int, default=500, choices=[250, 500, 1000],
                    help="켜질 때의 fs. 펌웨어 기본은 500 이고 PC 가 명령으로 바꾼다")
    ap.add_argument("--baud", type=int, default=115200,
                    help="스케치의 SERIAL_BAUD. **송신 버퍼 드롭이 여기서 정해진다**")
    ap.add_argument("--mode", default="ascii", choices=["ascii", "bin"],
                    help="켜질 때의 형식. 펌웨어 기본은 ascii 다")
    ap.add_argument("--drift-ppm", type=float, default=3000.0,
                    help="보드 클럭 오차. Uno 의 세라믹 레조네이터는 약 ±5000 ppm")
    ap.add_argument("--leadoff-every", type=float, default=0.0,
                    help="N 초마다 전극이 떨어진다 [s]")
    ap.add_argument("--leadoff-len", type=float, default=0.7)
    ap.add_argument("--dur", type=float, default=0.0, help="0 이면 무한")
    ap.add_argument("--signal-s", type=float, default=60.0,
                    help="신호 길이 [s]. 끝나면 처음으로 돌아간다. d1 에서 0 이면 기록 전체")
    ap.add_argument("--snr-db", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=7)
    # ---- 무엇을 흘릴 것인가
    ap.add_argument("--source", default="synth", choices=["synth", "d1"],
                    help="synth = 합성 심전도, d1 = MIT-BIH 기록 + NSTDB 잡음")
    ap.add_argument("--record", default="100",
                    help="--source d1 일 때의 기록 번호 (예: 100 · 105 · 119)")
    ap.add_argument("--noise", default="mixed",
                    help="d1 잡음: mixed · bw · em · ma · pli · none 등")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="NSTDB 잡음 구간. **보고서와 같은 자리를 보려면 test**")
    ap.add_argument("--offset-s", type=float, default=0.0,
                    help="기록에서 몇 초 지점부터 실을 것인가")
    ap.add_argument("--lead", default="MLII")
    ap.add_argument("--gain", type=float, default=1100.0,
                    help="AFE 총 이득. **브리지의 --gain 과 같아야** mV 축이 맞는다")
    ap.add_argument("--vref", type=float, default=5.0)
    ap.add_argument("--firmware", default="current", choices=["current", "old"],
                    help="old 면 fs·형식 명령을 무시한다 — 구 스케치가 꽂힌 판")
    # ---- DTR 리셋 흉내
    ap.add_argument("--boot-s", type=float, default=BOOTLOADER_S,
                    help="리셋 뒤 부트로더가 **조용한** 시간 [s]. 0 이면 즉시 시작")
    ap.add_argument("--hold-open", action="store_true",
                    help="slave 를 붙들어 열림 감지를 끈다 (리셋 흉내도 꺼진다)")
    # ---- USB 도착 뭉침
    ap.add_argument("--usb-poll-ms", type=float, default=1.0,
                    help="호스트 폴링 간격 [ms]. full-speed USB 는 1 ms 프레임이다")
    ap.add_argument("--usb-packet", type=int, default=64,
                    help="bulk 최대 패킷 [B]")
    ap.add_argument("--hiccup-every", type=float, default=0.0,
                    help="N 초마다 호스트가 잠깐 안 가져간다 [s] — 뭉침의 극단")
    ap.add_argument("--hiccup-ms", type=float, default=40.0,
                    help="그 «안 가져가는» 시간 [ms]")
    ap.add_argument("--attach", help="PTY 대신 이 포트에 붙는다 (com0com·socat)")
    ap.add_argument("--list", action="store_true",
                    help="고를 수 있는 기록·잡음·방법을 보여주고 끝낸다")
    ap.add_argument("--port-file", help="포트 이름을 이 파일에 적는다")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.list:
        return print_choices()

    if args.source == "d1":
        counts, info = record_counts(
            args.record, args.fs, noise=args.noise, snr_db=args.snr_db,
            split=args.split, seed=args.seed, offset_s=args.offset_s,
            dur_s=args.signal_s, lead=args.lead, gain=args.gain, vref=args.vref)
        print(f"d1 기록 {info['record']} ({info['lead']}) · {info['n']} 샘플 "
              f"({info['n'] / args.fs:.0f} s @ {args.fs} Hz) · 잡음 {info['noise']}"
              + (f" {info['snr_db']:g} dB" if info["noise"] != "none" else ""))
        if info.get("weights"):
            w = " · ".join(f"{k} {v:.0%}" for k, v in sorted(info["weights"].items()))
            print(f"  잡음 성분: {w}")
        if info.get("banks_error"):
            print(f"  [warn] NSTDB 를 못 읽었다 ({info['banks_error']}) — "
                  "합성 잡음만 썼다. data/raw/nstdb 를 확인할 것")
        print(f"  전극 단 진폭 {info['mv_p2p']:.2f} mV p-p · "
              f"ADC 클리핑 {100 * info['clipped_frac']:.2f} %")
        # **어느 split 인지가 시연에서 중요하다.** `train` 을 쓰면 딥러닝이
        # 자기가 학습한 파형을 보게 되고, 화면에 수치가 안 떠도 파형 품질이
        # 실제보다 좋아 보인다. 조용히 지나가면 안 되는 자리다.
        where = info["split_of_record"]
        if where != "test":
            why = {"train": "**모델이 이 기록으로 학습했다.** 딥러닝이 자기가 본 "
                            "파형을 보게 되므로 실제보다 잘 나온다",
                   "val": "모델 선택(early stopping)에 쓴 기록이다 — 완전히 "
                          "새로운 신호가 아니다",
                   "paced": "페이스메이커 기록이라 **D1 평가에서 제외했다** "
                            "(01_design). 파형 자체가 다른 문제다",
                   }.get(where, "D1 split 밖의 기록이다")
            print(f"  [warn] 기록 {info['record']} 은 **{where}** split 이다 — {why}.\n"
                  "         시연에는 test split 을 쓴다 (`--list` 로 목록을 본다)")
        if info["clipped_frac"] > 0.001:
            print(f"  [warn] R 파가 ADC 레인지를 넘는다 — 이득 {args.gain:g} 에서 "
                  f"잡을 수 있는 폭은 ±{args.vref / args.gain * 1e3 / 2:.2f} mV 다. "
                  "실제 보드에서도 같은 자리가 잘린다")
    else:
        counts = synth_counts(args.signal_s, args.fs, seed=args.seed,
                              snr_db=args.snr_db)

    def new_board(boot_s: float) -> FakeBoard:
        return FakeBoard(counts, fs=args.fs, baud=args.baud, mode=args.mode,
                         drift_ppm=args.drift_ppm,
                         leadoff_every_s=args.leadoff_every,
                         leadoff_len_s=args.leadoff_len,
                         accept_commands=(args.firmware == "current"),
                         boot_s=boot_s)

    if args.attach:
        fd, slave, name = attach_port(args.attach, args.baud)
        can_detect = False
    else:
        fd, slave, name = open_virtual_port(hold_open=args.hold_open)
        can_detect = slave < 0
    # 열림을 못 보는 구성에서는 켜자마자 도는 보드가 맞다 (부트로더도 이미 끝난 것).
    board = new_board(args.boot_s if can_detect else 0.0)
    pipe = UsbPipe(poll_ms=args.usb_poll_ms, packet=args.usb_packet,
                   hiccup_every_s=args.hiccup_every, hiccup_ms=args.hiccup_ms)

    if args.port_file:
        Path(args.port_file).write_text(name + "\n")
    print(f"가상 아두이노: {name}  (fs {args.fs} Hz · {args.baud} baud · {args.mode})")
    if can_detect:
        print(f"  포트를 열면 보드가 리셋된다 — 부트로더 {args.boot_s:.1f} s 는 조용하다")
    else:
        why = "--attach" if args.attach else "--hold-open"
        print(f"  [{why}] 열림을 못 보므로 **리셋 흉내가 꺼진다.** 보드는 계속 돈다")
    if args.hiccup_every > 0:
        print(f"  USB: {args.usb_poll_ms:g} ms 폴링 · {args.usb_packet} B 패킷 · "
              f"{args.hiccup_every:g} s 마다 {args.hiccup_ms:g} ms 멈춤")
    print(f"  브리지에 이렇게 붙인다:\n"
          f"    python3 scripts/serial_bridge.py --port {name} "
          f"--board-fs 250 --methods M_FE,M04 --serve")
    sys.stdout.flush()

    t0 = time.perf_counter()
    t_log = t0
    n_out = 0
    n_reset = 0
    attached = not can_detect            # 지금 누가 포트를 열고 있는가
    try:
        while True:
            now = time.perf_counter()
            if args.dur and now - t0 > args.dur:
                break
            # ---- 명령을 읽으면서 «누가 열고 있는가» 를 함께 본다.
            # PTY master 는 slave 를 아무도 안 열었을 때만 EIO 를 낸다.
            cmd = b""
            live = True
            try:
                cmd = os.read(fd, 256)
            except BlockingIOError:
                pass
            except OSError as e:
                if e.errno == errno.EIO and can_detect:
                    live = False
                elif e.errno not in (errno.EAGAIN, errno.EIO):
                    raise
            if can_detect and live and not attached:
                # **DTR 리셋.** 실제 Uno 가 이 자리에서 처음부터 시작한다.
                n_reset += 1
                board = new_board(args.boot_s)
                board.reset(now)
                pipe = UsbPipe(poll_ms=args.usb_poll_ms, packet=args.usb_packet,
                               hiccup_every_s=args.hiccup_every,
                               hiccup_ms=args.hiccup_ms)
                if not args.quiet:
                    print(f"\n  포트가 열렸다 -> 보드 리셋 #{n_reset} "
                          f"(부트로더 {args.boot_s:.1f} s 는 조용하다)", flush=True)
            elif can_detect and attached and not live and not args.quiet:
                print("\n  포트가 닫혔다 — 다음에 열면 다시 리셋된다", flush=True)
            attached = live

            if cmd:
                board.feed_command(cmd)
                if not args.quiet:
                    print(f"\n  <- 명령 {cmd!r} · fs {board.fs} · {board.mode}")

            # ---- 보드가 UART 로 낸 것을 **USB 관에 넣고**, 호스트가 가져갈
            # 만큼만 꺼낸다. 바이트는 한 개씩이 아니라 덩어리로 도착한다.
            pipe.push(board.poll(now))
            data = pipe.pop(now) if attached else b""
            if data:
                try:
                    n_out += os.write(fd, data)
                except BlockingIOError:
                    # PC 가 안 읽어 OS 버퍼가 찼다. **보드는 기다려 주지 않는다** —
                    # 실제 보드에서도 이 자리는 잃는 자리다.
                    pass
                except OSError as e:
                    if e.errno != errno.EIO:
                        raise
                    attached = False       # 그 사이 닫혔다
            if not args.quiet and now - t_log >= 1.0:
                t_log = now
                state = "연결됨" if attached else "**아무도 안 열었다**"
                print(f"\r  {now - t0:6.1f}s  {state}  보낸 샘플 {board.n_sent:8d}  "
                      f"버퍼드롭 {board.dropped:6d}  "
                      f"({100 * board.dropped / max(board.k, 1):5.2f} %)  "
                      f"USB 대기 {pipe.pending:5d} B  "
                      f"{n_out / max(now - t0, 1e-9) / 1024:5.2f} kB/s",
                      end="", flush=True)
            time.sleep(0.001)
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
          f"리셋 {n_reset} 회 · "
          f"실측 {board.k / el:.1f} Hz (설정 {board.fs} Hz)")
    if pipe.overflow:
        print(f"[warn] 호스트가 안 가져가 USB 관에서 {pipe.overflow} B 를 잃었다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
