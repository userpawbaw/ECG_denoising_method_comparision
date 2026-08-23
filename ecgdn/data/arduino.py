"""실측 Arduino ECG/잡음 적재 (docs/02_procedure.md STEP 28, docs/08_acquisition.md).

CSV 스키마 (헤더 주석 3줄 필수)
------------------------------
    # fs_hz=500
    # adc_bits=10, vref_v=5.0, gain=1100
    # session=S2, note=forearm muscle contraction
    t_ms,adc_raw
    0,512
    2,514
    ...

`fs_hz` 를 파일에 적어두지 않으면 나중에 반드시 문제가 된다.
t_ms 열이 있으면 실제 샘플 간격을 검사해 헤더의 fs_hz 와 대조한다 (드롭 샘플 탐지).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import FS
from .mitdb import resample_to

__all__ = ["ArduinoRecord", "parse_header", "load_csv", "load_dir", "SESSIONS"]

SESSIONS = {
    "S1": "정상 안정 측정 (ECG + 실사용 잡음)",
    "S2": "근육 수축 (EMG artifact, 잡음 전용)",
    "S3": "전극/케이블 움직임 (motion artifact, 잡음 전용)",
    "S4": "호흡/체간 이동 (baseline wander, 잡음 전용)",
    "S5": "전극 미부착 + 더미 저항 (front-end/ADC noise floor, 잡음 전용)",
    "S6": "전원 조건 비교 (USB vs 배터리)",
}
NOISE_ONLY = ("S2", "S3", "S4", "S5")


@dataclass
class ArduinoRecord:
    name: str
    x: np.ndarray                 # 물리단위 [mV] (gain/vref 정보가 있으면), 없으면 ADC LSB
    fs: float
    session: str = ""
    unit: str = "mV"
    header: dict = field(default_factory=dict)
    fs_measured: float | None = None
    note: str = ""

    @property
    def is_noise_only(self) -> bool:
        return self.session in NOISE_ONLY

    def __repr__(self) -> str:
        return (f"<ArduinoRecord {self.name} session={self.session} "
                f"n={self.x.size} fs={self.fs:g} unit={self.unit}>")


def parse_header(lines: list[str]) -> dict:
    """'# key=value, key=value' 형태의 주석 줄을 dict 로."""
    h: dict[str, str] = {}
    for ln in lines:
        ln = ln.strip()
        if not ln.startswith("#"):
            continue
        for kv in re.split(r"[,;]", ln.lstrip("#")):
            if "=" in kv:
                k, v = kv.split("=", 1)
                h[k.strip()] = v.strip()
    return h


def _to_float(h: dict, key: str, default=None):
    try:
        return float(h[key])
    except Exception:
        return default


def load_csv(path: str | Path, fs_out: float | None = FS,
             fs_hz: float | None = None) -> ArduinoRecord:
    """CSV 한 개를 적재하고 (선택적으로) fs_out 으로 리샘플한다."""
    p = Path(path)
    raw = p.read_text().splitlines()
    head = [ln for ln in raw[:20] if ln.strip().startswith("#")]
    h = parse_header(head)

    body = [ln for ln in raw if ln.strip() and not ln.strip().startswith("#")]
    if not body:
        raise ValueError(f"{p}: 데이터가 없다")
    cols = [c.strip().lower() for c in body[0].split(",")]
    has_header_row = not cols[0].replace(".", "", 1).replace("-", "", 1).isdigit()
    if has_header_row:
        body = body[1:]
    else:
        cols = ["t_ms", "adc_raw"][: len(cols)]

    arr = np.array([[float(v) for v in ln.split(",")] for ln in body], dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]

    i_adc = cols.index("adc_raw") if "adc_raw" in cols else (1 if arr.shape[1] > 1 else 0)
    i_t = cols.index("t_ms") if "t_ms" in cols else None
    x = arr[:, i_adc]

    fs_in = fs_hz or _to_float(h, "fs_hz")
    fs_meas = None
    if i_t is not None and arr.shape[0] > 10:
        dt = np.diff(arr[:, i_t]) / 1e3
        dt = dt[(dt > 0) & (dt < 1.0)]
        if dt.size:
            fs_meas = float(1.0 / np.median(dt))
    if fs_in is None:
        fs_in = fs_meas
    if fs_in is None:
        raise ValueError(f"{p}: fs 를 알 수 없다. 헤더에 '# fs_hz=...' 를 넣거나 "
                         "fs_hz 인자로 지정할 것.")
    if fs_meas is not None and abs(fs_meas - fs_in) / fs_in > 0.05:
        print(f"[warn] {p.name}: 헤더 fs_hz={fs_in:g} 이지만 t_ms 로 측정한 값은 "
              f"{fs_meas:.1f} Hz 다 (샘플 드롭 가능). 측정값을 신뢰한다면 fs_hz 를 고칠 것.")

    # 물리단위 변환: mV = (adc/2^bits) * vref * 1000 / gain
    bits = _to_float(h, "adc_bits")
    vref = _to_float(h, "vref_v")
    gain = _to_float(h, "gain")
    unit = "ADC LSB"
    if bits and vref:
        x = (x / (2 ** bits)) * vref * 1e3        # mV (증폭 후)
        unit = "mV (amplified)"
        if gain and gain > 0:
            x = x / gain                          # 전극 단 환산
            unit = "mV (referred to electrodes)"

    if fs_out is not None and abs(fs_out - fs_in) > 1e-9:
        x = resample_to(x, fs_in, fs_out)
        fs = float(fs_out)
    else:
        fs = float(fs_in)

    return ArduinoRecord(name=p.stem, x=np.asarray(x, dtype=np.float64), fs=fs,
                         session=h.get("session", ""), unit=unit, header=h,
                         fs_measured=fs_meas, note=h.get("note", ""))


def load_dir(root: str | Path = "data/arduino", pattern: str = "*.csv",
             fs_out: float | None = FS) -> list[ArduinoRecord]:
    out = []
    for p in sorted(Path(root).glob(pattern)):
        try:
            out.append(load_csv(p, fs_out))
        except Exception as e:
            print(f"[skip] {p.name}: {type(e).__name__}: {e}")
    return out
