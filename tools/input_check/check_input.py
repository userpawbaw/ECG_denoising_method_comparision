"""AFE 입력 점검 — 팀원 PC 에서 **더블클릭 한 번**으로 돈다 (D-51).

순서: 포트 고르기 -> 10 초 동안 ADC 를 받아 판정 -> 결과를 txt 로 저장 ->
실시간 화면(브리지 + 브라우저)을 띄운다. 파이썬 · 의존성은 묶음에 들어 있다
(`scripts/build_input_check.py` 가 만든다).
"""
from __future__ import annotations

import subprocess
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import serial
from serial.tools import list_ports

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_serial import adc_summary, adc_verdict, read_adc  # noqa: E402

BAUD = 115200
FS = 250.0
SECONDS = 10.0
# 아두이노 정품 · CH340 · FTDI · CP210x
KNOWN_VID = {0x2341: "Arduino", 0x2A03: "Arduino", 0x1A86: "CH340", 0x0403: "FTDI", 0x10C4: "CP210x"}


def pick_port() -> str | None:
    ports = sorted(list_ports.comports(), key=lambda p: p.device)
    if not ports:
        print("포트가 하나도 없다. USB 케이블(충전 전용 아닌 것)과 드라이버를 확인할 것.")
        return None
    for i, p in enumerate(ports, 1):
        tag = KNOWN_VID.get(p.vid or -1, "")
        print(f"  {i}. {p.device}  {p.description}  {('<- ' + tag) if tag else ''}")
    known = [p for p in ports if (p.vid or -1) in KNOWN_VID]
    if len(known) == 1:
        print(f"-> {known[0].device} 를 쓴다")
        return known[0].device
    if len(ports) == 1:
        print(f"-> {ports[0].device} 를 쓴다")
        return ports[0].device
    while True:
        s = input("번호를 고르고 Enter: ").strip()
        if s.isdigit() and 1 <= int(s) <= len(ports):
            return ports[int(s) - 1].device


def tone_counts(x: np.ndarray, f0: float) -> float:
    """f0 성분의 p-p 크기 [ADC 칸]."""
    t = np.arange(x.size) / FS
    return float(4 * abs(np.mean((x - x.mean()) * np.exp(-2j * np.pi * f0 * t))))


def analyse(vals: list[int]) -> list[str]:
    a = adc_summary(vals)
    if not a.get("n"):
        return ["[실패] 샘플을 하나도 못 받았다 — 펌웨어가 올라갔는지, 포트가 맞는지 확인."]
    x = np.asarray(vals, dtype=float)
    lines = [f"샘플 {a['n']} 개 ({a['n'] / FS:.1f} 초), 기대 {int(SECONDS * FS)} 개",
             f"ADC 범위 {a['lo']} ~ {a['hi']} (폭 {a['hi'] - a['lo']} 칸), 중앙값 {a['med']}, "
             f"서로 다른 값 {a['unique']} 개"]
    pp = {f: tone_counts(x, f) for f in (60.0, 120.0)}
    lines.append(f"60 Hz 성분 p-p {pp[60.0]:.1f} 칸, 120 Hz {pp[120.0]:.1f} 칸")
    lines += ["  " + s for s in adc_verdict(a)]
    width = a["hi"] - a["lo"]
    if width <= 2:
        lines.append("[판정] 출력이 거의 고정이다 — AFE 출력이 죽었거나(단락 · 전원) 전극이 안 붙었다.")
    elif a["rail_lo"] > 0.05 or a["rail_hi"] > 0.05:
        lines.append("[판정] 0 또는 1023 에 붙어 있다 — 포화. AFE 기준 전압 · 전극 접촉 확인.")
    elif pp[60.0] > 0.5 * width:
        lines.append("[판정] 60 Hz 가 신호 대부분이다 — 충전기를 빼고 AC 기기에서 떨어져 다시 해 볼 것.")
    else:
        lines.append("[판정] 입력이 들어온다. 화면에서 PQRST 모양을 눈으로 확인할 것.")
    return lines


def main() -> int:
    print("=== AFE 입력 점검 ===\n")
    port = pick_port()
    if not port:
        input("\nEnter 를 누르면 닫힌다")
        return 1
    print(f"\n{SECONDS:.0f} 초 동안 받는다. 전극을 붙이고 가만히 있을 것...")
    try:
        vals = read_adc(serial, port, BAUD, SECONDS)
    except serial.SerialException as e:
        print(f"[실패] 포트를 못 열었다: {e}\n  Arduino IDE 의 시리얼 모니터가 켜져 있으면 끌 것.")
        input("\nEnter 를 누르면 닫힌다")
        return 1
    lines = analyse(vals)
    print("\n" + "\n".join(lines))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT.parent / f"결과_{stamp}.txt"
    out.write_text(f"포트 {port}\n" + "\n".join(lines) + "\n\nADC 원값\n"
                   + ",".join(map(str, vals)) + "\n", encoding="utf-8")
    print(f"\n결과를 저장했다: {out}\n이 파일을 보내 주면 된다.")

    print("\n실시간 화면을 띄운다 (끝내려면 이 창을 닫거나 Ctrl+C)...")
    bridge = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "serial_bridge.py"),
                               "--port", port, "--methods", "M_FE,M01", "--serve"])
    time.sleep(4)
    webbrowser.open("http://127.0.0.1:8765/live.html")
    try:
        bridge.wait()
    except KeyboardInterrupt:
        bridge.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
