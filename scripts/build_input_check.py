"""AFE 입력 점검 묶음(Windows 64 bit, 설치 불필요)을 만든다 (D-51).

    python scripts/build_input_check.py --out <폴더>

Windows 내장형(embeddable) 파이썬 + numpy · scipy · pyserial · PyWavelets 의
win_amd64 휠을 받아 풀고, 브리지 · 화면 · 점검기를 함께 zip 으로 묶는다.
받는 쪽은 압축을 풀고 `START_HERE.bat` 만 더블클릭한다. 리눅스에서도 만들 수 있다.
torch 는 넣지 않는다 — 점검에는 front-end · M01 이면 충분하고, 묶음이 2 GB 가 된다.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY_VER = "3.11.9"
PY_URL = f"https://www.python.org/ftp/python/{PY_VER}/python-{PY_VER}-embed-amd64.zip"
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

1. 아두이노(펌웨어 arduino_ecg_logger 가 올라간 것)를 USB 로 꽂고 전극을 붙인다.
2. START_HERE.bat 를 더블클릭한다. (설치할 것 없음. 'Windows 의 PC 보호' 가 뜨면
   '추가 정보' -> '실행')
3. 10 초 동안 가만히 있으면 판정이 나오고, 이 폴더에 결과_날짜.txt 가 생긴다.
4. 이어서 브라우저에 실시간 화면이 뜬다. 끝내려면 검은 창을 닫는다.

결과_날짜.txt 를 보내 주면 된다. (ADC 원값이 들어 있다 — 측정한 사람의 심전도다.)
충전기를 뺀 상태와 꽂은 상태로 한 번씩 해 주면 더 좋다.
"""


def build(out: Path) -> Path:
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
    z = build(Path(ap.parse_args().out).resolve())
    print(f"만들었다: {z} ({z.stat().st_size / 2**20:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
