"""R-4: 스트리밍 처리기 — **오프라인과 같은 부품을 쓰고 "언제" 만 바꾼다.**

    sp = StreamProcessor(build("M04"), fs=250.0, hop=12, d=12)
    for block in serial_blocks():          # 아두이노에서 오는 조각들
        out = sp.push(block)               # 확정된 출력 샘플만 돌아온다
        plot(out)

설계 근거는 `docs/30_realtime_demo.md` 2 절이고, 아래 셋이 요점이다.

**(1) window 길이는 지연이 아니다.** 지연을 정하는 것은 두 가지다 —
**`d`**(샘플 하나가 확정되기까지 남겨 두는 미래 문맥)와 **`hop`**(모델이
도는 간격). 샘플을 일정한 지연 `L` 로 내보내려면

    L >= d + hop                     (2.2 에서 유도)

여야 한다. `d` 만 줄이고 `hop` 을 크게 두면 화면이 hop 만큼 멈췄다 뛴다.
`latency_s` 가 이 값을 돌려준다.

**(2) 방법을 다시 구현하지 않는다.** `method` 는 `ecgdn.methods` 의 호출
가능 객체 그대로다. 이 클래스가 하는 일은 **링버퍼를 유지하고, 어느 출력
샘플을 내보내도 되는지 정하는 것**뿐이다. 재구현하면 시연이 보고서와 다른
것을 보이게 되고 그것을 알아챌 방법이 없다 — F-9 · F-23 과 같은 계열이다.

**(3) crossfade 도 러닝 스케일도 없다.** 처음 설계에는 둘 다 있었는데
R-3 에서 재보니 넣을 이유가 없었다(2.3) — 스트리밍 품질이 오프라인의
±0.36 dB 안이고 이음매는 신호 차분 분포의 49~66 백분위다. 겹치는 구간은
오프라인과 같은 **Hann²** 가중으로 더한다.

front-end 는 **밖에서, 한 번만** 돈다 — 이것이 R-4 의 핵심 발견이다
-----------------------------------------------------------------
처음에는 방법을 링버퍼에 그대로 호출했다. `DLDenoiser`·`M04` 는 **내부에서
front-end 를 걸므로**, 그러면 버퍼마다 `filtfilt` 가 다시 돌고 그 결과들이
Hann² 로 겹쳐 더해진다. 오프라인 대비 **−15.0 dB** 가 나왔다 `[측정]`.

배관을 의심했지만 아니었다 — 항등 방법(`M00`)으로 재니 출력이 입력과
**4e-16** 안에서 같았다. 결정적 측정은 이것이다:

| 조건 | 오프라인과의 상관 | 오프라인 대비 |
|---|---|---|
| `M04` (방법이 FE 를 내부에서 건다) | 0.8437 | **−15.00 dB** |
| `M04` (FE 를 **미리** 걸고 방법의 FE 는 끔) | **1.0000** | **−0.02 dB** |

**틀린 것은 필터를 어디서 도느냐 하나였다.** 그리고 고칠 수 없는 이유가
있다 — `FrontEnd` 는 0.5 Hz HPF 의 링잉을 `8 × order / f_c` = **64 초**로
잡는다. 영위상 필터를 100 ms 지연으로 돌릴 방법은 없다.

그래서 실시간 경로는 **인과 front-end**(`StreamingFrontEnd`)를 스트림 앞단에
한 번 두고, 방법에는 `use_frontend=False` 를 준다. 위상 왜곡이라는 대가가
생기고, 그 크기는 `scripts/verify_stream_processor.py` 가 실측해 적는다.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..data.windows import analysis_window


def _has_own_frontend(method: Any) -> bool:
    """방법이 자체 front-end 를 들고 있는가.

    속성 이름이 갈린다 — 고전 방법은 `fe`, `DLDenoiser` 는 `_fe` 다.
    `use_frontend` 는 생성자 인자일 뿐 남지 않으므로 **객체를 봐야 한다.**
    """
    return any(getattr(method, a, None) is not None for a in ("fe", "_fe"))


class StreamProcessor:
    """링버퍼 + "언제 내보낼 수 있는가" 규칙.

    Parameters
    ----------
    method : callable
        `ecgdn.methods` 의 denoiser. `method(y, fs, ctx) -> np.ndarray`.
    fs : float
        표본화 주파수 [Hz].
    win : int | None
        한 번에 방법에 주는 문맥 길이. `None` 이면 `method.win`(딥러닝) 또는
        1024. **문맥이지 지연이 아니다.**
    hop : int
        모델이 도는 간격 [샘플]. 작을수록 지연이 줄고 연산이 는다.
    d : int
        샘플 하나에 남겨 두는 미래 문맥 [샘플]. 2.1 실측에서 12(48 ms)면
        가장자리 손해가 사라진다.
    ctx_pre : int
        방법에 함께 주는 **왼쪽 문맥** [샘플]. 기본 0 — 인과 front-end 를
        앞단에 두면 방법 쪽에는 문맥이 따로 필요 없다. 지연에는 들어가지
        않는다(이미 지나간 표본이다).
    frontend : "causal" | "none"
        `"causal"` 이면 `StreamingFrontEnd` 를 앞단에 두고 방법에는 이미
        걸린 신호를 준다. **방법 자신의 front-end 는 반드시 꺼야 한다** —
        두 번 걸리면 결과가 보고서와 달라진다.
    """

    def __init__(self, method: Callable[..., np.ndarray], fs: float = 250.0,
                 win: int | None = None, hop: int = 12, d: int = 12,
                 ctx_pre: int = 0, frontend: str = "causal", fe_cfg=None):
        if hop < 1:
            raise ValueError(f"hop 은 1 이상이어야 한다: {hop}")
        if d < 0:
            raise ValueError(f"d 는 0 이상이어야 한다: {d}")
        self.method = method
        self.fs = float(fs)
        self.win = int(win if win is not None else getattr(method, "win", 1024))
        self.hop = int(hop)
        self.d = int(d)
        if self.d >= self.win:
            raise ValueError(f"d({self.d}) 가 win({self.win}) 보다 크면 "
                             "확정할 샘플이 없다")
        self.ctx_pre = int(ctx_pre)
        if self.ctx_pre < 0:
            raise ValueError(f"ctx_pre 는 0 이상이어야 한다: {self.ctx_pre}")
        if frontend not in ("causal", "none"):
            raise ValueError(f"frontend 는 'causal' 또는 'none': {frontend!r}")
        self.frontend = frontend
        # 인과 FE 의 설계를 바꿔 볼 수 있게 열어 둔다. **기본값은 공통
        # front-end 와 같다** — 다른 값을 주면 학습 분포에서 멀어지므로,
        # 그 대가를 `verify_stream_processor.py` 로 재고 나서만 쓴다(F-27).
        self.fe_cfg = fe_cfg
        if frontend == "causal" and _has_own_frontend(method):
            raise ValueError(
                "방법이 자체 front-end 를 켜 둔 채로 인과 FE 를 앞단에 두면 "
                "필터가 **두 번** 걸린다. `use_frontend=False`(고전) 또는 "
                "`frontend=False`(DLDenoiser) 로 만들 것.")
        self._w2 = analysis_window(self.win) ** 2
        self.reset()

    # ------------------------------------------------------------------ 상태
    def reset(self) -> None:
        """버퍼와 누적을 비운다. 재생을 처음부터 다시 돌릴 때 쓴다."""
        if self.frontend == "causal":
            from .frontend_stream import StreamingFrontEnd
            from ..config import DEFAULT_FE_CAUSAL
            self._fe = StreamingFrontEnd(self.fs, self.fe_cfg or DEFAULT_FE_CAUSAL)
        else:
            self._fe = None
        self._buf = np.zeros(0)             # **front-end 를 통과한** 표본
        self._acc = np.zeros(0)             # Hann² 가중 누적
        self._wsq = np.zeros(0)
        self._n_in = 0                      # 지금까지 받은 표본 수 (절대 인덱스)
        # 첫 window 는 왼쪽 문맥이 다 찬 뒤에 돈다 (warm-up)
        self._next_run = self.ctx_pre + self.win
        # **아직 아무 window 도 안 돌았다.** 이것을 `_next_run - hop` 으로
        # 대신 쓰면 첫 표본이 들어오자마자 내보내 버린다 (테스트가 잡았다).
        self._last_end = -1
        # `ctx_pre` 앞쪽은 **어떤 window 에도 덮이지 않는다** — 내보내면 0 이
        # 나간다. 처음 구현이 그것을 내보내서 검증이 −17.9 dB 를 냈다.
        self._emitted = self.ctx_pre        # 지금까지 내보낸 절대 인덱스
        self.origin = self.ctx_pre          # 첫 출력 샘플의 절대 인덱스
        self.n_runs = 0                     # 방법 호출 횟수 (처리량 확인용)

    @property
    def warmup_s(self) -> float:
        """첫 출력까지 걸리는 시간 [s]. 문맥과 창이 차는 데 걸리는 시간이다."""
        return (self.ctx_pre + self.win) / self.fs

    @property
    def latency_s(self) -> float:
        """샘플 하나가 화면에 나오기까지의 알고리즘 지연 [s]. `(d + hop)/fs`."""
        return (self.d + self.hop) / self.fs

    @property
    def runs_per_s(self) -> float:
        """초당 방법 호출 횟수. CPU 예산은 이 값 × 1 회 비용이다."""
        return self.fs / self.hop

    # ------------------------------------------------------------------ 입력
    def push(self, block: np.ndarray, ctx: dict[str, Any] | None = None) -> np.ndarray:
        """새 표본을 넣고 **그 사이에 확정된 출력**을 돌려준다.

        블록 크기는 자유다 — `hop` 의 배수일 필요가 없다. 한 번에 여러
        window 를 넘길 만큼 크면 그만큼 여러 번 돈다.
        """
        block = np.asarray(block, dtype=np.float64).ravel()
        if block.size and self._fe is not None:
            block = self._fe.push(block)    # 인과 FE 는 스트림에 **한 번만**
        if block.size:
            self._buf = np.concatenate([self._buf, block])
            self._acc = np.concatenate([self._acc, np.zeros(block.size)])
            self._wsq = np.concatenate([self._wsq, np.zeros(block.size)])
            self._n_in += block.size

        while self._next_run <= self._n_in:
            self._run_window(self._next_run - self.win, ctx)
            self._next_run += self.hop

        if self._last_end < 0:                  # 아직 한 번도 안 돌았다
            return np.zeros(0)
        ready = max(0, min(self._last_end - self.d, self._n_in))
        if ready <= self._emitted:
            return np.zeros(0)
        seg = slice(self._emitted, ready)
        w = self._wsq[seg]
        out = np.where(w > 1e-12, self._acc[seg] / np.maximum(w, 1e-12), 0.0)
        self._emitted = ready
        return out

    def _estimate(self, buf: np.ndarray, ctx: dict[str, Any] | None) -> np.ndarray:
        """버퍼 하나에 대한 방법의 추정. 방법 종류에 따라 **호출 방식이 다르다.**

        **딥러닝(`.model` 이 있는 것)은 모델을 한 번만 통과시킨다.**
        `DLDenoiser.__call__` 을 그대로 쓰면 안 된다 — 그것은 오프라인 전용
        경로라 입력을 reflect 패딩해 다시 프레이밍하고 내부 OLA 를 한다.
        버퍼 하나에 그것을 걸면 **오른쪽 끝 프레임의 절반이 반사된 가짜 신호**
        이고, 그 가장자리가 스트리밍에서 가장 최근 구간이다.

        실측으로 확인했다 `[측정]` — wrapper 를 그대로 호출하면 배관 손해가
        −5.7 dB 였다. 각 호출의 가장자리를 잘라내면 −0.09 dB 까지 줄지만,
        자를 수 있는 폭이 `d` 보다 작아야 해서(그렇지 않으면 쓸 수 있는
        window 가 없다) `d = 12` 에서는 쓸 수 없는 처방이다.

        그래서 R-3(`measure_stream_seam.py`)이 검증한 경로를 쓴다 —
        **같은 모델 · 같은 `robust_scale` · 같은 Hann²**, 프레이밍만 우리가
        한다. 고전 방법은 프레이밍이 없으므로 그대로 호출한다.
        """
        model = getattr(self.method, "model", None)
        if model is None:
            est = np.asarray(self.method(buf, self.fs, dict(ctx or {})),
                             dtype=np.float64).ravel()
            if est.size != buf.size:
                raise ValueError(f"방법이 길이를 바꿨다: {est.size} != {buf.size}")
            return est

        import torch

        from ..utils import robust_scale
        sc = robust_scale(buf) if getattr(self.method, "normalize", True) else 1.0
        with torch.no_grad():
            tt = torch.from_numpy((buf / sc)[None, None].astype(np.float32))
            out = model(tt)
            if isinstance(out, tuple):
                out = out[0]
            return out[0, 0].cpu().numpy().astype(np.float64) * sc

    def _run_window(self, start: int, ctx: dict[str, Any] | None) -> None:
        """`[start, start+win)` 을 추정하고 Hann² 로 누적한다.

        고전 방법에는 **왼쪽 문맥까지 붙여서** 주고(`ctx_pre`) 뒤쪽 `win` 만
        쓴다. 딥러닝은 학습 때와 같은 길이여야 하므로 정확히 `win` 을 준다.
        """
        is_dl = getattr(self.method, "model", None) is not None
        lo = start if is_dl else max(0, start - self.ctx_pre)
        buf = self._buf[lo:start + self.win]
        est = self._estimate(buf, ctx)[-self.win:]
        self._acc[start:start + self.win] += est * self._w2
        self._wsq[start:start + self.win] += self._w2
        self._last_end = start + self.win
        self.n_runs += 1

    # ------------------------------------------------------- 편의: 통째로 재생
    def run(self, y: np.ndarray, block: int = 25,
            ctx: dict[str, Any] | None = None) -> np.ndarray:
        """녹음본을 `block` 샘플씩 흘려 넣어 스트리밍 결과를 얻는다.

        **검증용이지 시연 경로가 아니다.** 시연은 `push` 를 직접 쓴다.
        기본 25 샘플(100 ms)은 시리얼 한 패킷 정도의 크기다.

        돌려주는 배열의 첫 샘플은 원본의 **`origin`(= `ctx_pre`)** 번째다 —
        그 앞은 문맥으로만 쓰이고 출력이 없다. 비교할 때 이 오프셋을 맞추지
        않으면 warm-up 구간의 0 을 성능으로 재게 된다.
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        parts = [self.push(y[i:i + block], ctx) for i in range(0, y.size, block)]
        return np.concatenate([p for p in parts if p.size]) if parts else np.zeros(0)
