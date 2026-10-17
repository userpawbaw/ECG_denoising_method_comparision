"""AFE 입력 점검 묶음(Windows 64 bit, 설치 불필요)을 만든다 (D-51).

    python scripts/build_input_check.py --out <폴더>

Windows 내장형(embeddable) 파이썬 + numpy · scipy · pyserial · PyWavelets 의
win_amd64 휠을 받아 풀고, 브리지 · 화면 · 점검기를 함께 zip 으로 묶는다.
받는 쪽은 압축을 풀고 `START_HERE.bat` 만 더블클릭한다. 리눅스에서도 만들 수 있다.
torch 는 넣지 않는다 — 점검에는 front-end · M01 이면 충분하고, 묶음이 2 GB 가 된다.
"""
from __future__ import annotations

import argparse
import hashlib
import tarfile
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY_VER = "3.11.9"
PY_URL = f"https://www.python.org/ftp/python/{PY_VER}/python-{PY_VER}-embed-amd64.zip"
# 펌웨어 업로드 도구 — Arduino 공식 package_index 의 avrdude 8.0.0-arduino1 Windows 판과 그 SHA-256
AVRDUDE_URL = "https://downloads.arduino.cc/tools/avrdude_8.0-arduino.1_Windows_32bit.tar.gz"
AVRDUDE_SHA = "833aa1a66a8e70cd597fcfdbd7e559a91a00eca1d7aa3be2ce9bcadf7ccb987c"
WHEELS = ["numpy", "scipy", "pyserial", "PyWavelets"]
APP_FILES = ["scripts/serial_bridge.py", "scripts/_bootstrap.py", "scripts/probe_serial.py",
             "tools/input_check/check_input.py"]
APP_DIRS = ["ecgdn", "demo"]

BAT = r"""@echo off
chcp 65001 >nul
cd /d "%~dp0"
python\python.exe app\tools\input_check\check_input.py
pause
"""

README = """AFE 입력 점검

연결: AFE 출력 -> A0, AFE GND -> 아두이노 GND.
      AFE 에 LO+ / LO- (전극 떨어짐) 출력이 있으면 D10 / D11, **없으면 D10 · D11 을 GND 에** 꽂는다
      (떠 있으면 샘플이 무작위로 빠진다).
포트가 안 보이면: 정품이 아닌 우노(CH340 칩)는 드라이버가 필요할 수 있다 — 장치 관리자에서
      '알 수 없는 장치' 로 보이면 'CH340 driver' 를 받아 설치한다.

1. 아두이노(펌웨어 arduino_ecg_logger 가 올라간 것)를 USB 로 꽂고 전극을 붙인다.
2. START_HERE.bat 를 더블클릭한다. (설치할 것 없음. 'Windows 의 PC 보호' 가 뜨면
   '추가 정보' -> '실행')
3. 10 초 동안 가만히 있으면 판정이 나오고, 이 폴더에 결과_날짜.txt 가 생긴다.
   샘플을 못 알아들으면 '측정 펌웨어를 올릴까?' 를 묻는다 — Enter 를 누르면
   알아서 올린다(우노 · 나노). Arduino IDE 는 필요 없지만, 켜져 있으면 끌 것.
4. 이어서 브라우저에 실시간 화면이 뜬다. 끝내려면 검은 창을 닫는다.

결과_날짜.txt 를 보내 주면 된다. (ADC 원값이 들어 있다 — 측정한 사람의 심전도다.)
충전기를 뺀 상태와 꽂은 상태로 한 번씩 해 주면 더 좋다.
"""


def add_firmware(stage: Path, out: Path, cli: str) -> None:
    """측정 펌웨어를 우노용 hex 로 굽고, Windows avrdude 와 함께 넣는다.

    팀원 보드에 다른 스케치가 올라가 있으면 브리지가 «샘플이 하나도 안 맞는다» 로
    끝난다 — 점검기가 그때 이것으로 펌웨어를 올린다 (Arduino IDE 불필요).
    """
    fw = stage / "firmware"
    fw.mkdir()
    build_dir = out / "hex"
    subprocess.run([cli, "compile", "-b", "arduino:avr:uno", "--output-dir", str(build_dir),
                    str(ROOT / "hardware" / "arduino_ecg_logger")], check=True)
    shutil.copy2(build_dir / "arduino_ecg_logger.ino.hex", fw / "arduino_ecg_logger.hex")
    tgz = out / "avrdude_win.tar.gz"
    if not tgz.exists():
        urllib.request.urlretrieve(AVRDUDE_URL, tgz)
    if hashlib.sha256(tgz.read_bytes()).hexdigest() != AVRDUDE_SHA:
        tgz.unlink()
        raise SystemExit("avrdude 체크섬이 다르다 — 받은 파일을 지웠다")
    with tarfile.open(tgz) as t:
        t.extractall(out / "avrdude_x", filter="data")
    shutil.copytree(next((out / "avrdude_x").iterdir()), fw / "avrdude", dirs_exist_ok=True)


def build(out: Path, cli: str) -> Path:
    stage = out / "ecg_input_check"
    if stage.exists():
        shutil.rmtree(stage)
    py = stage / "python"
    py.mkdir(parents=True)
    emb = out / f"python-{PY_VER}-embed-amd64.zip"
    if not emb.exists():
        print("내려받기:", PY_URL)
        urllib.request.urlretrieve(PY_URL, emb)
    zipfile.ZipFile(emb).extractall(py)
    # 내장형 파이썬은 sys.path 를 ._pth 가 **전부** 정한다 — 스크립트 폴더도 안 들어간다.
    pth = next(py.glob("python*._pth"))
    pth.write_text("\n".join([f"{pth.stem}.zip", ".", r"Lib\site-packages",
                              r"..\app", r"..\app\scripts", "import site", ""]))
    whl = out / "wheels"
    whl.mkdir(exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "download", "-q", "-d", str(whl),
                    "--only-binary=:all:", "--platform", "win_amd64",
                    "--python-version", "3.11", "--implementation", "cp", *WHEELS], check=True)
    site = py / "Lib" / "site-packages"
    site.mkdir(parents=True)
    for w in sorted(whl.glob("*.whl")):
        zipfile.ZipFile(w).extractall(site)
    app = stage / "app"
    for f in APP_FILES:
        (app / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, app / f)
    for d in APP_DIRS:
        shutil.copytree(ROOT / d, app / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    add_firmware(stage, out, cli)
    (stage / "START_HERE.bat").write_bytes(BAT.replace("\n", "\r\n").encode())
    (stage / "README.txt").write_text(README.replace("\n", "\r\n"), encoding="utf-8-sig")
    z = out / "ecg_input_check_win64.zip"
    if z.exists():
        z.unlink()
    shutil.make_archive(str(z.with_suffix("")), "zip", out, stage.name)
    return z


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="작업 · 출력 폴더 (git 밖을 권한다)")
    ap.add_argument("--arduino-cli", default="arduino-cli",
                    help="펌웨어 컴파일에 쓸 arduino-cli 경로 (arduino:avr 코어가 깔려 있어야 한다)")
    args = ap.parse_args()
    z = build(Path(args.out).resolve(), args.arduino_cli)
    print(f"만들었다: {z} ({z.stat().st_size / 2**20:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
