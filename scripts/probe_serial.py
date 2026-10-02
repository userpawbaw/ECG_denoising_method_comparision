#!/usr/bin/env python3
"""실보드 점검: 보드가 이 포트·baud 에서 **무슨 바이트를 내보내는지** 본다. 브리지보다 먼저 돌린다.

    python scripts/probe_serial.py --list                  # 어떤 포트가 있나
    python scripts/probe_serial.py --port COM3             # 흔한 baud 를 차례로 시험
    python scripts/probe_serial.py --port COM3 --baud 115200

브리지가 «바이트는 오는데 샘플이 하나도 안 맞는다» 고 할 때, 그 바이트가 **무엇인지**
(줄 모양 · 칸 수 · baud) 를 한 번에 보여 준다. 브리지의 진단은 «안 맞는다» 까지만 말하고
«무엇이 왔는지» 는 말하지 않는다.

포트를 여는 순간 DTR 로 보드가 리셋돼 스케치의 배너(`# logger ready`)가 먼저 나온다.
그래서 baud 마다 약 5 초씩 듣는다 (부팅 1.5 s + 배너 + 스트림).

**pyserial 만 쓴다.** `_bootstrap`(matplotlib 을 부른다)을 일부러 안 부른다 — 점검 도구가
점검 대상 환경(numpy · scipy · torch)에 기대면, 그것이 안 깔린 노트북에서 «왜 안 되는지»
를 못 보여 준다. 브리지 파서와 같은 규칙(`t_ms,adc` 정확히 두 칸 · 동기 바이트 0xA5)을
따로 적어 두었으므로 `tests/test_probe_serial.py` 가 둘을 대조한다.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter

COMMON_BAUDS = [115200, 9600, 19200, 38400, 57600, 230400, 250000, 500000, 1000000]

SYNC = 0xA5          # BINARY 프레임 [0xA5][seq][lo][hi][xor] 의 동기 바이트
FRAME_LEN = 5
MIN_LINES = 20       # 이보다 적게 받았으면 «형식이 맞다» 고 말하지 않는다


# ------------------------------------------------------------------ 판정 (순수 계산)
def bridge_accepts(line: bytes) -> bool:
    """브리지의 `AsciiParser` 가 이 줄을 샘플로 받는가 — **같은 규칙**이다.

    쉼표로 나눠 정확히 두 칸이고 둘째 칸이 정수여야 한다. 첫 칸(`t_ms`)은
    브리지가 읽지 않는다.
    """
    parts = line.strip().split(b",")
    if len(parts) != 2:
        return False
    try:
        int(parts[1])
    except ValueError:
        return False
    return True


def count_frames(buf: bytes) -> int:
    """검사합이 맞는 BINARY 프레임 수 - `BinaryParser` 와 **같은 걸음**으로 센다.

    맞으면 5 바이트를 건너뛰고, 아니면 1 바이트만 민다. 모든 위치를 세면 `seq` 가
    0xA5 인 프레임 안에서 겹치는 가짜 후보가 하나 더 잡힌다.
    """
    n, i, frames = len(buf), 0, 0
    while n - i >= FRAME_LEN:
        if buf[i] == SYNC and (SYNC ^ buf[i + 1] ^ buf[i + 2] ^ buf[i + 3]) == buf[i + 4]:
            frames += 1
            i += FRAME_LEN
        else:
            i += 1
    return frames


def analyse(buf: bytes) -> dict:
    """받은 바이트를 줄 · 칸 수 · 프레임으로 읽는다."""
    n = len(buf)
    printable = sum(1 for b in buf if b in (9, 10, 13) or 32 <= b < 127)
    lines = [ln.rstrip(b"\r") for ln in buf.split(b"\n")[:-1]]      # 마지막 조각은 미완성
    data = [ln for ln in lines if ln.strip() and not ln.strip().startswith(b"#")]
    return {
        "n": n,
        "text": 100.0 * printable / n if n else 0.0,           # 읽을 수 있는 글자 %
        "banner": b"logger ready" in buf or b"ecgstream" in buf,
        "lines": lines,
        "data": data,                                          # 배너(#) 를 뺀 줄
        "ok": sum(1 for ln in data if bridge_accepts(ln)),     # 브리지가 받는 줄
        "fields": Counter(ln.count(b",") + 1 for ln in data),  # 한 줄이 몇 칸인가
        "frames": count_frames(buf),                           # 검사합이 맞는 5 바이트 덩어리
    }


def verdict(a: dict) -> str:
    """`good` · `binary` · `text` · `garbage` · `empty` 중 하나."""
    if a["n"] == 0:
        return "empty"
    if len(a["data"]) >= MIN_LINES and a["ok"] >= 0.8 * len(a["data"]):
        return "good"
    if a["frames"] * FRAME_LEN >= 0.6 * a["n"]:
        return "binary"
    if a["text"] >= 90.0:
        return "text"
    return "garbage"


def rate_hz(data: list[bytes]) -> float | None:
    """줄의 `t_ms` 로 잰 샘플률 - **연속한 줄의 간격의 중앙값**으로 잰다.

    처음과 끝의 차이로 재면 중간에 보드가 리셋돼 `t_ms` 가 되돌아갈 때 틀린다.
    샘플을 버린 자리(간격이 큰 것)도 중앙값은 신경 쓰지 않는다.
    """
    ts = []
    for ln in data:
        head = ln.split(b",")[0].strip()
        if head.isdigit():
            ts.append(int(head))
    gaps = sorted(b - a for a, b in zip(ts, ts[1:]) if b > a)
    if len(gaps) < 10:
        return None
    return 1000.0 / gaps[len(gaps) // 2]


def text_hint(a: dict) -> str:
    """글자는 읽히는데 브리지가 못 받는 이유."""
    if not a["data"]:
        return "줄바꿈이 없는 글자다."
    k = a["fields"].most_common(1)[0][0]
    if k != 2:
        return (f"한 줄이 {k} 칸이다. 브리지는 `t_ms,adc` 정확히 2 칸만 받는다 "
                "(스케치의 ASCII 출력 규격).")
    return "2 칸이지만 둘째 칸이 정수가 아니다. 브리지는 `t_ms,adc` 의 adc 를 정수로 읽는다."


# ------------------------------------------------------------------ 입출력
def listen(serial, port: str, baud: int, seconds: float) -> bytes:
    """포트를 열어 `seconds` 동안 받은 바이트를 돌려준다."""
    s = serial.Serial(port, baud, timeout=0.2)        # 여는 순간 DTR 로 보드가 리셋된다
    buf = bytearray()
    try:
        t_end = time.time() + seconds
        while time.time() < t_end:
            buf += s.read(4096)
    finally:
        s.close()
    return bytes(buf)


def short(b: bytes, n: int = 60) -> str:
    """바이트를 `repr` 로 보여 주되 길면 자른다 (BINARY · 깨진 바이트는 «줄» 이 수백 B 다)."""
    return repr(b[:n]) + (f" ... (+{len(b) - n} B)" if len(b) > n else "")


def show_ports(list_ports) -> list[str]:
    ports = sorted(list_ports.comports(), key=lambda p: p.device)
    if not ports:
        print("  (포트가 하나도 없다 - 케이블이 「충전 전용」 이 아닌지, 드라이버가 깔렸는지 "
              "장치 관리자에서 확인)")
    for p in ports:
        print(f"  {p.device:<12} {p.description}")
    return [p.device for p in ports]


def report(baud: int, buf: bytes, a: dict, v: str) -> None:
    """고른 baud 에서 받은 것을 보여 주고 결론을 말한다."""
    print()
    print(f"-- baud {baud} 에서 받은 것 --")
    shown = [ln for ln in a["lines"] if ln.strip()]
    if v in ("good", "text") and shown:
        print("  처음 3 줄:")
        for ln in shown[:3]:
            print("   ", short(ln))
        print("  끝   3 줄:")
        for ln in shown[-3:]:
            print("   ", short(ln))
        print("  칸 수 분포:", dict(a["fields"].most_common(4)),
              f"  평균 {a['n'] / len(a['lines']):.1f} B/줄")
    elif buf:
        print("  앞 30 바이트(16진):", buf[:30].hex(" "))
    print()
    if v == "good":
        hz = rate_hz(a["data"])
        print(f"결론: baud {baud} 에서 `t_ms,adc` 줄이 읽힌다 -> 보드 · 스케치 · 포트가 정상이다.")
        if hz:
            print(f"  t_ms 로 잰 샘플률은 약 {hz:.0f} Hz (스케치 기본 500 Hz).")
        if baud != 115200:
            print(f"  브리지를 --baud {baud} 로 띄울 것 (스케치의 SERIAL_BAUD 와 같아야 한다).")
        if not a["banner"]:
            print("  배너(`# logger ready`)는 못 봤다 - 포트를 열어도 이 보드는 리셋되지 않았을 수 있다.")
    elif v == "binary":
        print(f"결론: baud {baud} 에서 BINARY 프레임이 읽힌다 -> 보드가 이미 바이너리 모드다.")
        print("  이 보드는 포트를 열어도 리셋되지 않아 앞서 보낸 'b' 가 남아 있다. 브리지를 --ascii "
              "없이 띄우거나, 보드를 한 번 리셋(USB 뽑았다 꽂기)할 것.")
    elif v == "text":
        print(f"결론: baud {baud} 에서 글자는 읽히지만 브리지가 받는 `t_ms,adc` 형식이 아니다.")
        print("  ->", text_hint(a))
        print("  -> 다른 스케치(또는 고친 스케치)가 올라가 있을 가능성이 크다. 위 「처음 3 줄」 이 "
              "그 형식이다. hardware/arduino_ecg_logger 를 다시 업로드할 것.")
    elif v == "garbage":
        print("결론: 읽히는 글자가 없다 -> 보드가 이 baud 들로 말하지 않는다.")
        print("  (스케치가 안 올라갔거나 / 다른 baud 의 스케치이거나 / 보드 클럭이 다르다 - "
              "업로드 완료 메시지부터 확인)")
    else:
        print("결론: 바이트가 하나도 안 온다 -> 포트 이름 · 케이블 · 드라이버를 확인할 것 (--list).")


def main() -> int:
    ap = argparse.ArgumentParser(description="보드가 포트에서 무엇을 말하는지 본다")
    ap.add_argument("--port", help="시리얼 포트. 예: COM3, /dev/ttyACM0 (목록은 --list)")
    ap.add_argument("--baud", type=int, nargs="+", default=COMMON_BAUDS,
                    help="시험할 baud 들. 안 적으면 흔한 값을 차례로 하고, 읽히는 baud 에서 멈춘다. "
                         "스케치의 SERIAL_BAUD 는 115200")
    ap.add_argument("--seconds", type=float, default=4.5,
                    help="baud 마다 듣는 시간 [s]. 보드가 리셋돼 부팅하는 약 1.5 s 가 들어 있으니 "
                         "3 s 밑으로 줄이지 말 것")
    ap.add_argument("--list", action="store_true", help="열 수 있는 포트를 보여 주고 끝낸다")
    args = ap.parse_args()

    try:
        import serial
        from serial.tools import list_ports
    except ImportError:
        raise SystemExit("pyserial 이 필요하다:  pip install pyserial")

    if args.list or not args.port:
        print("열 수 있는 포트:")
        show_ports(list_ports)
        if args.list:
            return 0
        print("\n--port 가 필요하다. 예:  python scripts/probe_serial.py --port <위의 이름>")
        return 2
    if args.seconds < 3.0:
        print("[warn] --seconds 가 3 s 보다 짧으면 부팅과 배너를 놓친다.")

    print(f"포트 {args.port} - 보드가 리셋되며 배너를 낸다. baud 마다 약 {args.seconds:g} 초")
    print(f"{'baud':>8} {'바이트':>7} {'글자':>6} {'배너':>4} {'줄':>5} {'받는줄':>6}  앞부분")
    rows = []
    try:
        for baud in args.baud:
            try:
                buf = listen(serial, args.port, baud, args.seconds)
            except serial.SerialException as e:
                print(f"포트를 못 연다: {e}")
                print("  -> IDE 시리얼 모니터/플로터와 앞서 띄운 브리지를 닫을 것. 열 수 있는 포트:")
                show_ports(list_ports)
                return 1
            a = analyse(buf)
            v = verdict(a)
            print(f"{baud:>8} {a['n']:>7} {a['text']:>5.1f}% {'있음' if a['banner'] else '없음':>4} "
                  f"{len(a['data']):>5} {a['ok']:>6}  {buf[:40]!r}")
            rows.append((baud, buf, a, v))
            if v in ("good", "binary"):
                break
    except KeyboardInterrupt:
        print("\n중단했다.")
        return 130

    hit = [r for r in rows if r[3] in ("good", "binary")]
    baud, buf, a, v = hit[0] if hit else max(rows, key=lambda r: r[2]["text"])
    report(baud, buf, a, v)
    return 0 if hit else 1


if __name__ == "__main__":
    raise SystemExit(main())
