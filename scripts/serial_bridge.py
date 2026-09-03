#!/usr/bin/env python3
"""R-5/R-6: 아두이노 -> 실시간 처리 -> 화면. **하드웨어 없이도 끝까지 돈다.**

    # 하드웨어 없이 (모의 보드가 같은 선 규격으로 흘린다)
    python3 scripts/serial_bridge.py --replay synth --methods M_FE,M04,M06 --serve

    # 실제 보드
    python3 scripts/serial_bridge.py --port /dev/ttyACM0 --board-fs 250 --serve
    #   -> 브라우저에서 http://127.0.0.1:8765/live.html

무엇을 잇는가
------------
    [AD8232] -> [Uno ADC] -> USB 시리얼 -> **여기** -> 브라우저 패널
                                            |
                                            +- 인과 front-end (한 번)
                                            +- StreamProcessor × 방법 수
                                            +- SSE 로 밀어내기

설계 근거는 `docs/30_realtime_demo.md` 6.2 다. 요점 넷:

**(1) 시간축의 주인은 보드다.** PC 벽시계로 샘플을 세면 보드 클럭(±0.5 %)과
서서히 어긋난다. 화면은 **도착한 만큼만** 진행하고, x 축은 보드가 붙인
샘플 번호로 그린다.

**(2) front-end 는 한 번만 돈다.** 방법마다 걸면 필터가 방법 수만큼 돌 뿐
아니라, `StreamProcessor` 가 자체 FE 를 켜면 방법 내부 FE 와 이중으로 걸린다
(F-25). 그래서 여기서 `StreamingFrontEnd` 하나를 돌리고 모든 처리기에
`frontend="none"` 을 준다.

**(3) 읽기는 별도 스레드다.** 추론이 100 ms 걸리는 동안 시리얼을 안 읽으면
OS 버퍼(보통 4 kB)가 넘치고 **그때부터는 조용히 샘플이 없어진다.**
읽기 스레드는 오직 읽어서 큐에 넣기만 한다.

**(4) 모드 A 에는 SNR 을 띄우지 않는다.** 실측 신호에는 참값이 없다
(난관 6). 여기서 내보내는 것은 파형과 **처리 상태**(손실·lead-off·지연)뿐이다.
"""
import _bootstrap  # noqa: F401

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

from ecgdn.config import FS
from ecgdn.realtime.frontend_modes import FE_MODES, build_fe, fe_intrinsic
from ecgdn.realtime.serial_link import (SYNC, AsciiParser, BinaryParser,
                                        CausalDecimator, SampleChunk, adc_to_mv)
from ecgdn.realtime.stream import StreamProcessor

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"

# **필터가 곧 방법인 것들.** 실시간 경로에서 이들의 대응물은 앞단의
# front-end **자신**이므로 처리기 출력 대신 그것을 쓴다. 처리기를 쓰면 이미
# 걸린 신호에 같은 필터를 한 번 더 거는 셈이 된다(F-25).
# `scripts/verify_stream_processor.py` 가 이들을 건너뛰는 것과 같은 이유다.
#
# **모드마다 다르다** — `fe_intrinsic(mode)` 가 정한다. 중앙값 front-end 는
# 대역통과가 아니므로 `M01` 을 대체하지 않는다. 그래서 처리기는 `M_FE` 를
# 뺀 **모든** 이름에 대해 만들어 두고, 어느 것을 쓸지는 모드가 고른다.
ALWAYS_INTRINSIC = {"M_FE"}


# --------------------------------------------------------------------- 방법
def build_stream_method(mid: str, axis: str = "d1"):
    """**front-end 를 끈 채로** 방법을 만든다.

    `scripts/verify_stream_processor.py` 의 `build_nofe` 와 같은 규칙이다 —
    실시간 경로에서 FE 는 스트림 앞단에서 한 번만 돈다(F-25).
    """
    if mid.startswith("M06") or mid.startswith("M08") or mid.startswith("M09"):
        from ecgdn.methods.dl_wrapper import DLDenoiser
        tag = {"M06": "m06_l1", "M06L6": "m06_l6",
               "M08": "m08_l1", "M08L6": "m08_l6", "M09": "m09_l1"}[mid]
        ck = ROOT / "results" / axis / tag / "best.pt"
        if not ck.exists():
            raise FileNotFoundError(f"{mid}: 체크포인트가 없다 -> {ck}")
        return DLDenoiser(ckpt=ck, name=mid.rstrip("L6") or "M06", frontend=False)
    from ecgdn.methods import build as reg_build
    return reg_build(mid, use_frontend=False)


# ------------------------------------------------------------------- 입력원
class SerialSource:
    """실제 보드. 읽기만 하는 스레드로 돈다."""

    def __init__(self, port: str, baud: int, board_fs: int, binary: bool):
        try:
            import serial
        except ImportError:                                  # pragma: no cover
            raise SystemExit("pyserial 이 필요하다:  pip install pyserial")
        self.ser = serial.Serial(port, baud, timeout=0.05)
        # 포트를 열면 DTR 이 토글돼 **보드가 리셋된다.** 부팅 + 첫 헤더가
        # 나올 때까지 기다렸다 버린다. 이것을 안 하면 첫 2 초가 쓰레기다.
        time.sleep(2.0)
        self.ser.reset_input_buffer()
        cmd = {250: b"2", 500: b"5", 1000: b"1"}[board_fs]
        self.ser.write(cmd)
        self.ser.write(b"b" if binary else b"a")
        time.sleep(0.2)
        self.ser.reset_input_buffer()

    def read(self) -> bytes:
        return self.ser.read(4096) or b""

    def close(self) -> None:
        self.ser.close()


class ReplaySource:
    """**하드웨어가 없을 때의 보드 흉내.** 같은 선 규격으로, 실시간 속도로 흘린다.

    이것이 있어야 브리지 전체를 하드웨어 없이 검증할 수 있고, 시연 당일
    보드가 죽었을 때의 대체 경로(난관 7)도 같은 코드가 된다.

    일부러 어렵게 만든다 — 실제 선에서 생기는 것들을 재현한다:
    보드 클럭 오차(`--drift`), 도착의 뭉침(USB CDC 는 패킷 단위로 온다),
    송신 버퍼 초과로 인한 샘플 드롭(`--drop`), lead-off 구간(`--leadoff`).
    """

    def __init__(self, kind: str, board_fs: int, binary: bool,
                 drift_ppm: float = 3000.0, drop_rate: float = 0.0,
                 leadoff_every_s: float = 0.0, seed: int = 0,
                 csv: str | None = None):
        self.fs = float(board_fs)
        self.binary = binary
        self.drop_rate = float(drop_rate)
        self.leadoff_every = float(leadoff_every_s)
        self.gen = np.random.default_rng(seed)
        self.counts = self._make(kind, csv, board_fs)
        self.i = 0
        self.seq = 0
        self.t0 = time.perf_counter()
        # 보드 클럭이 PC 보다 조금 빠르거나 느리다. Uno 는 세라믹 레조네이터라
        # 0.3 % 정도가 현실적이다 — 이것이 있으면 PC 벽시계로 샘플을 세는
        # 구현이 곧바로 무너진다.
        self.rate = self.fs * (1.0 + drift_ppm * 1e-6)

    def _make(self, kind: str, csv: str | None, board_fs: int) -> np.ndarray:
        """ADC 카운트(0..1023) 배열. AD8232 출력을 흉내낸다."""
        if kind == "csv":
            from ecgdn.data.arduino import load_csv
            rec = load_csv(csv, fs_out=None)
            x = rec.x
        else:
            from ecgdn.data.mixer import mix_at_snr
            from ecgdn.data.noise import mixed_noise
            from ecgdn.data.synthetic import synth_ecg
            n_s = 120.0
            clean = synth_ecg(duration_s=n_s, fs=board_fs, seed=7).x
            noise, _ = mixed_noise(clean.size, board_fs, self.gen)
            x, _, _ = mix_at_snr(clean, noise, 8.0)
        x = np.asarray(x, dtype=np.float64)
        s = np.percentile(np.abs(x - np.median(x)), 99) or 1.0
        # 10 bit, 중앙 512, 진폭이 ADC 범위의 약 1/3 이 되게 — 실제 AD8232 +
        # 5 V 기준에서 대략 그 정도다.
        return np.clip(512 + (x - np.median(x)) / s * 170, 0, 1023).round()

    def read(self) -> bytes:
        """지금까지 '보드가 보냈어야 할' 샘플을 프레임으로 만들어 준다."""
        el = time.perf_counter() - self.t0
        want = int(el * self.rate)
        if want <= self.i:
            time.sleep(0.002)
            return b""
        out = bytearray()
        for k in range(self.i, min(want, self.i + 2000)):
            lead = (self.leadoff_every > 0
                    and (k / self.fs) % self.leadoff_every < 0.7)
            drop = self.gen.random() < self.drop_rate
            val = 0xFFFF if lead else int(self.counts[k % self.counts.size])
            if drop:
                self.seq = (self.seq + 1) & 0xFF          # 보드가 버린 자리
                continue
            if self.binary:
                lo, hi = val & 0xFF, val >> 8
                out += bytes([SYNC, self.seq, lo, hi, SYNC ^ self.seq ^ lo ^ hi])
            else:
                v = -1 if val == 0xFFFF else val
                out += f"{int(k * 1000 / self.fs)},{v}\n".encode()
            self.seq = (self.seq + 1) & 0xFF
        self.i = min(want, self.i + 2000)
        return bytes(out)

    def close(self) -> None:
        pass


# --------------------------------------------------------------------- 정렬
class Aligner:
    """방법마다 첫 출력 시각이 다르다 — **공통 구간만** 내보낸다.

    처리기마다 `win` 이 다르면 warm-up 도 다르다(딥러닝 1024, 고전 1024 지만
    바뀔 수 있다). 정렬하지 않고 그리면 화면에서 방법끼리 시간이 어긋나고,
    그것은 비교가 아니다.
    """

    def __init__(self, names: list[str]):
        self.names = list(names)
        self.buf: dict[str, list[float]] = {n: [] for n in names}
        self.origin: dict[str, int] = {}
        self.sent = 0

    def add(self, name: str, origin: int, y: np.ndarray) -> None:
        self.origin.setdefault(name, origin)
        self.buf[name].extend(y.tolist())

    def take(self) -> tuple[int, dict[str, list[float]]]:
        if len(self.origin) < len(self.names):
            return 0, {}
        start = max(self.origin.values())
        end = min(self.origin[n] + len(self.buf[n]) for n in self.names)
        lo = max(self.sent, start)
        if end <= lo:
            return 0, {}
        out = {n: self.buf[n][lo - self.origin[n]: end - self.origin[n]]
               for n in self.names}
        self.sent = end
        # **보낸 것은 버린다.** 이것이 없으면 버퍼가 세션 내내 자란다 —
        # 그리고 그것이 CPU 를 먹는다. 파이썬의 세대별 GC 는 추적 대상 컨테이너의
        # **슬롯을 전부 훑으므로**, 원소 수십만 개짜리 리스트가 있으면 gen2 수집
        # 한 번의 비용이 그 길이에 비례한다. 즉 **시간이 갈수록 느려진다** —
        # 초반엔 멀쩡하고 10 분쯤 뒤 RTF 와 큐드롭이 함께 오르는 그 증상이다.
        # `raw` 는 20 s 로 묶여 있었는데(`lo > 20 * FS`) 여기만 빠져 있었다 (F-31).
        for n in self.names:
            k = end - self.origin[n]
            if k > 0:
                del self.buf[n][:k]
                self.origin[n] = end        # buf[n][0] 의 절대 번호를 따라 올린다
        return lo, out


# ----------------------------------------------------------------- SSE 서버
class Hub:
    """지금 붙어 있는 브라우저들. **밀린 클라이언트는 버린다.**

    한 명이 느리다고 처리를 세우면 안 된다 — 그러면 시연 전체가 그 브라우저
    속도로 내려간다. 큐가 차면 그 클라이언트의 프레임을 버리고 센다.
    """

    def __init__(self) -> None:
        self.qs: list[queue.Queue] = []
        self.lock = threading.Lock()
        self.dropped = 0

    def register(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=30)
        with self.lock:
            self.qs.append(q)
        return q

    def unregister(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.qs:
                self.qs.remove(q)

    def publish(self, payload: dict) -> None:
        data = json.dumps(payload, separators=(",", ":"))
        with self.lock:
            targets = list(self.qs)
        for q in targets:
            try:
                q.put_nowait(data)
            except queue.Full:
                self.dropped += 1


class FeSwitch:
    """front-end 전환 요청함. **스레드 하나만 상태를 바꾼다.**

    HTTP 스레드는 `request()` 로 이름만 남기고, 리더 스레드가 `take()` 로
    가져가 자기 시점에 갈아 끼운다. 리더가 도는 중에 HTTP 스레드가 필터
    상태를 건드리면 그 블록의 출력이 반쯤 옛 필터가 된다.
    """

    def __init__(self, mode: str):
        self.mode = mode
        self._want: str | None = None
        self._lock = threading.Lock()

    def request(self, mode: str) -> None:
        with self._lock:
            self._want = mode

    def take(self) -> str | None:
        with self._lock:
            m, self._want = self._want, None
        return m


def make_server(hub: Hub, port: int, switch: "FeSwitch | None" = None):
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    class H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(DEMO), **kw)

        def log_message(self, *a):            # 콘솔을 조용하게
            pass

        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/fe":
                # 화면이 목록을 물어본다 — 이름표를 **코드에서** 받아 가게 해서
                # UI 와 구현이 갈라지지 않게 한다.
                return self._json(200, {"modes": FE_MODES,
                                        "current": switch.mode if switch else None})
            if path != "/stream":
                return super().do_GET()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = hub.register()
            try:
                while True:
                    try:
                        data = q.get(timeout=10.0)
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")   # 프록시 타임아웃 방지
                        self.wfile.flush()
                        continue
                    self.wfile.write(b"data: " + data.encode() + b"\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                hub.unregister(q)

        def do_POST(self):
            if self.path.split("?")[0] != "/fe" or switch is None:
                return self._json(404, {"error": "그런 경로가 없다"})
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                mode = json.loads(self.rfile.read(n) or b"{}").get("mode")
            except json.JSONDecodeError:
                return self._json(400, {"error": "JSON 이 아니다"})
            if mode not in FE_MODES:
                return self._json(400, {"error": f"모드가 아니다: {mode!r}",
                                        "modes": sorted(FE_MODES)})
            switch.request(mode)
            return self._json(200, {"ok": True, "mode": mode})

    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    except OSError as e:
        # 시연 중에 브리지를 다시 띄울 때 실제로 걸린다 — 앞의 것이 아직
        # 살아 있으면 포트를 놓지 않는다. 원인을 그대로 말해 준다.
        raise SystemExit(
            f"포트 {port} 를 열 수 없다 ({e}).\n"
            f"  이전 브리지가 아직 도는지 본다:  ps -ef | grep serial_bridge\n"
            f"  살아 있으면 **PID 로** 끝낸다 (패턴으로 죽이면 자기 셸을 잡는다 — O-18).\n"
            f"  또는 다른 포트로:  --http-port {port + 1}") from None
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ---------------------------------------------------------------------- 본체
def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--port", help="시리얼 포트. 예: /dev/ttyACM0, COM3")
    src.add_argument("--replay", choices=["synth", "csv"],
                     help="하드웨어 없이 같은 선 규격으로 흉내낸다")
    ap.add_argument("--csv", help="--replay csv 일 때 읽을 파일")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--board-fs", type=int, default=250, choices=[250, 500, 1000],
                    help="보드의 샘플링률. **250 이면 리샘플이 없다**")
    ap.add_argument("--ascii", action="store_true",
                    help="ASCII 모드로 읽는다 (손실을 셀 수 없다 — 디버깅용)")
    ap.add_argument("--methods", default="M_FE,M01,M04",
                    help="쉼표로. 딥러닝은 체크포인트가 있어야 한다")
    ap.add_argument("--axis", default="d1", help="딥러닝 체크포인트 축")
    ap.add_argument("--fe", default="zerophase", choices=sorted(FE_MODES),
                    help="front-end 모드. 화면에서 실행 중에도 바꿀 수 있다")
    ap.add_argument("--d", type=int, default=12, help="미래 문맥 [샘플]")
    ap.add_argument("--hop", type=int, default=12, help="추론 간격 [샘플]")
    ap.add_argument("--fps", type=float, default=25.0, help="화면 갱신률")
    ap.add_argument("--serve", action="store_true", help="브라우저용 SSE 서버를 연다")
    ap.add_argument("--http-port", type=int, default=8765)
    ap.add_argument("--dur", type=float, default=0.0, help="0 이면 무한")
    ap.add_argument("--drift-ppm", type=float, default=3000.0, help="--replay 전용")
    ap.add_argument("--drop", type=float, default=0.0, help="--replay 전용 드롭률")
    ap.add_argument("--leadoff-every", type=float, default=0.0, help="--replay 전용 [s]")
    ap.add_argument("--adc-bits", type=int, default=10)
    ap.add_argument("--vref", type=float, default=5.0)
    ap.add_argument("--gain", type=float, default=1100.0,
                    help="아날로그 프런트엔드 총 이득 (AD8232 기본 약 1100)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--diag", type=float, default=0.0,
                    help="N 초마다 버퍼 크기와 RSS 를 적는다 (성능 저하 추적용)")
    args = ap.parse_args()
    # 진행 줄은 `\r` 로 덮어쓴다 — 파일로 넘기면 한 줄이 수만 자가 된다.
    if not sys.stdout.isatty():
        args.quiet = True

    binary = not args.ascii
    names = [m.strip() for m in args.methods.split(",") if m.strip()]

    parser = BinaryParser() if binary else AsciiParser()
    dec = CausalDecimator(args.board_fs, FS)
    if dec.m != 1:
        print(f"[warn] 보드 {args.board_fs} Hz -> {FS:g} Hz 로 {dec.m} 배 데시메이션한다. "
              "보드를 250 Hz 로 돌리면 리샘플이 아예 없다 (권장).")

    # ---- front-end 는 여기서 **한 번만**. 방법은 한 번 만들어 재사용한다
    # (모델 적재가 비싸다) — 전환 때는 `reset()` 만 부른다.
    hop_s = args.hop / FS
    fe = build_fe(args.fe, FS, hop_s=hop_s)
    procs = {}
    for n in names:
        if n in ALWAYS_INTRINSIC:
            continue                       # 출력이 곧 FE 출력이다 (위 주석)
        procs[n] = StreamProcessor(build_stream_method(n, args.axis), fs=FS,
                                   hop=args.hop, d=args.d, frontend="none")
    if not procs:
        raise SystemExit("처리기가 하나도 없다 — front-end 말고 다른 방법을 하나는 넣을 것")

    def split_names(mode: str):
        """이 모드에서 어느 이름을 front-end 출력으로 낼지 고른다."""
        intr = fe_intrinsic(mode)
        return ([n for n in names if n in intr],
                [n for n in names if n not in intr and n in procs])

    fe_names, proc_names = split_names(args.fe)
    lat_ms = 1000.0 * next(iter(procs.values())).latency_s
    warm_s = max(p.warmup_s for p in procs.values())
    print(f"방법 {names} · 지연 {lat_ms:.0f} ms · warm-up {warm_s:.1f} s "
          f"· 추론 {next(iter(procs.values())).runs_per_s:.0f} 회/s")
    if fe_names:
        print(f"  {fe_names} 는 front-end 출력 그대로다 — 필터가 곧 방법이다")
    print(f"  front-end: {args.fe} — {FE_MODES[args.fe]['label']} "
          f"(+{fe.latency_samples / FS * 1000:.0f} ms)")

    # ---- 입력원은 **모델을 다 올린 뒤에** 연다. 먼저 열면 적재에 걸린 몇 초
    # 동안 보드가 보낸 것이 큐에 쌓여, 시작부터 밀린 상태로 출발한다.
    if args.port:
        source = SerialSource(args.port, args.baud, args.board_fs, binary)
    else:
        source = ReplaySource(args.replay, args.board_fs, binary,
                              drift_ppm=args.drift_ppm, drop_rate=args.drop,
                              leadoff_every_s=args.leadoff_every, csv=args.csv)

    hub = Hub()
    switch = FeSwitch(args.fe)
    if args.serve:
        make_server(hub, args.http_port, switch)
        print(f"-> http://127.0.0.1:{args.http_port}/live.html")

    # ---- 읽기 스레드. **읽어서 큐에 넣기만 한다.**
    q: queue.Queue = queue.Queue(maxsize=400)
    stop = threading.Event()
    qdrop = [0]

    def reader():
        while not stop.is_set():
            data = source.read()
            if not data:
                continue
            try:
                q.put_nowait(data)
            except queue.Full:
                # 처리가 못 따라온다. **오래된 것을 버리고 센다** — 조용히
                # 밀리면 화면이 점점 과거를 보이게 된다.
                try:
                    q.get_nowait()
                    qdrop[0] += 1
                except queue.Empty:
                    pass

    th = threading.Thread(target=reader, daemon=True)
    th.start()

    al = Aligner(names)
    raw: list[float] = []
    rawok: list[bool] = []
    raw_base = 0                      # raw[0] 의 절대 인덱스
    stat = dict(lost=0, leadoff=0, bad=0, resync=0)
    t_frame = time.perf_counter()
    t_start = t_diag = t_frame
    cpu = 0.0
    n_in = 0                          # 처리기에 들어간 250 Hz 샘플 (= FE 가 낸 수)
    n_input = 0                       # front-end 에 들어간 250 Hz 샘플. `n_in` 과
                                      # 갈라진다 — 지연 있는 FE 는 늦게 낸다.
    n_board = 0                       # 보드가 보낸 샘플 (손실률의 분모)
    rtf_cpu = rtf_wall = rtf_core = 0.0   # warm-up 이후의 정상 상태
    core = 0.0
    try:
        while not stop.is_set():
            if args.dur and time.perf_counter() - t_start > args.dur:
                break
            try:
                data = q.get(timeout=0.5)
            except queue.Empty:
                continue
            want = switch.take()
            if want and want != switch.mode:
                # **화면을 다시 채운다.** 방법들의 내부 버퍼는 이전 front-end
                # 출력으로 채워져 있어, 그대로 두면 두 필터가 섞인 구간이 나온다.
                # 처리기는 다시 만들지 않고 `reset()` 만 한다 — 모델 적재가 비싸다.
                switch.mode = want
                fe = build_fe(want, FS, hop_s=hop_s)
                # **어느 이름이 «필터가 곧 방법» 인지도 모드마다 다르다.**
                # 중앙값은 대역통과가 아니라 `M01` 을 대체하지 않는다.
                fe_names, proc_names = split_names(want)
                for pr in procs.values():
                    pr.reset()
                al = Aligner(names)
                raw, rawok, raw_base, n_in, n_input = [], [], 0, 0, 0
                fe_lat_ms = fe.latency_samples / FS * 1000.0
                hub.publish(json.dumps({"reset": True, "fe": want,
                                        "fe_label": FE_MODES[want]["label"],
                                        "fe_lat_ms": round(fe_lat_ms, 0)},
                                       ensure_ascii=False))
                if not args.quiet:
                    print(f"\n  front-end -> {want} ({FE_MODES[want]['label']}, "
                          f"+{fe_lat_ms:.0f} ms) · 화면을 다시 채운다")
            pre: SampleChunk = parser.feed(data)
            n_board += pre.x.size          # **보드 기준** 샘플 (채운 것 포함)
            ch: SampleChunk = dec(pre)
            if ch.gap_unknown:
                # 손실이 커서 시간축을 복구할 수 없다. 이어 붙이면 이후 전부가
                # 어긋나므로 **처음부터 다시** 간다 (화면에도 알린다).
                stat["resync"] += 1
                parser.reset()
                fe.reset()
                for p in procs.values():
                    p.reset()
                al = Aligner(names)
                raw, rawok, raw_base = [], [], n_input
                continue
            stat["lost"] += ch.n_lost
            stat["leadoff"] += ch.n_leadoff
            stat["bad"] += ch.n_bad
            if ch.x.size == 0:
                continue

            # **단위를 먼저 mV 로 바꾼다.** 필터는 선형이라 순서가 결과를
            # 바꾸지 않지만, 이렇게 해야 입력과 출력이 **같은 축**에 놓인다.
            mv = adc_to_mv(ch.x, bits=args.adc_bits, vref=args.vref, gain=args.gain)
            # 화면의 '입력' 은 **front-end 이전**이다 — 기저선 흔들림과 60 Hz 가
            # 보여야 무엇이 제거됐는지 눈에 들어온다.
            raw.extend(mv.tolist())
            rawok.extend(ch.ok.tolist())
            n_input += mv.size
            x = fe.push(mv)                # front-end 는 한 번만 (F-25)
            n_in += x.size
            if x.size == 0:                # 지연 있는 모드의 첫 몇 블록
                continue

            t0, c0 = time.perf_counter(), time.process_time()
            for name in proc_names:
                p = procs[name]
                al.add(name, p.origin, p.push(x))
            for name in fe_names:
                al.add(name, 0, x)          # FE 출력은 지연 없이 바로 확정된다
            cpu += time.perf_counter() - t0
            # **벽시계와 CPU 를 따로 센다.** torch 는 코어를 여러 개 쓰므로
            # 벽시계 RTF 가 0.4 라도 CPU 는 1.5 코어일 수 있다. 노트북 하나로
            # 시연할 수 있는가는 **CPU 쪽**이 정한다.
            core += time.process_time() - c0

            now = time.perf_counter()
            if now - t_frame < 1.0 / args.fps:
                continue
            el = now - t_frame
            t_frame = now
            idx, outs = al.take()
            if not outs:
                continue
            k = len(next(iter(outs.values())))
            lo = idx - raw_base
            payload = {
                "i": idx, "n": k, "fs": FS,
                "raw": [round(v, 4) for v in raw[lo:lo + k]],
                "ok": [int(b) for b in rawok[lo:lo + k]],
                "out": {n: [round(v, 4) for v in vs] for n, vs in outs.items()},
                "stat": {**stat, "qdrop": qdrop[0], "sse_drop": hub.dropped,
                         "lat_ms": round(lat_ms + fe.latency_samples / FS * 1000.0, 1),
                         "fe": switch.mode,
                         "rtf": round(cpu / max(el, 1e-9), 3)},
            }
            if idx > (warm_s + 2.0) * FS:      # warm-up 과 첫 적재는 뺀다
                rtf_cpu += cpu
                rtf_core += core
                rtf_wall += el
            cpu = core = 0.0
            hub.publish(payload)
            if lo > 20 * FS:                     # 과거는 버린다 (메모리 상한)
                raw = raw[lo:]
                rawok = rawok[lo:]
                raw_base = idx
            if args.diag and now - t_diag >= args.diag:
                t_diag = now
                rss = 0
                try:
                    with open("/proc/self/status") as fh:
                        for ln in fh:
                            if ln.startswith("VmRSS:"):
                                rss = int(ln.split()[1]) // 1024
                except OSError:
                    pass
                albuf = sum(len(v) for v in al.buf.values())
                print(f"\n[diag] {idx/FS:7.1f}s  RTF {payload['stat']['rtf']:.2f}  "
                      f"큐드롭 {qdrop[0]:4d}  정렬기버퍼 {albuf:8d}  "
                      f"raw {len(raw):7d}  RSS {rss:5d} MB", flush=True)
            if not args.quiet:
                print(f"\r  {idx/FS:7.1f}s  손실 {stat['lost']:5d}  "
                      f"lead-off {stat['leadoff']:5d}  깨짐 {stat['bad']:4d}  "
                      f"재동기 {stat['resync']:2d}  큐드롭 {qdrop[0]:3d}  "
                      f"RTF {payload['stat']['rtf']:.2f}", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        source.close()
    dur = max(n_in / FS, 1e-9)
    # **손실·lead-off 는 보드 기준 샘플 수다.** 데시메이션을 하면 처리 쪽
    # 샘플 수와 단위가 달라지므로 분모를 섞으면 안 된다.
    print(f"\n처리 {n_in} 샘플 ({dur:.1f} s @ {FS:g} Hz) · 보드 {n_board} 샘플 · "
          f"손실 {stat['lost']} ({100*stat['lost']/max(n_board,1):.2f} %) · "
          f"lead-off {stat['leadoff']} ({100*stat['leadoff']/max(n_board,1):.1f} %) · "
          f"재동기 {stat['resync']} · 큐드롭 {qdrop[0]}")
    if rtf_wall > 0:
        print(f"정상 상태 RTF {rtf_cpu/rtf_wall:.3f} (벽시계) · "
              f"{rtf_core/rtf_wall:.2f} 코어 (CPU 시간) · 방법 {len(names)} 개. "
              f"벽시계 RTF 가 1.0 을 넘으면 실시간을 못 따라간다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
