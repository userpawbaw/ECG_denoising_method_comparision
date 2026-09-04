"""블록 영위상의 **이음매** — F-36.

블록마다 독립으로 `filtfilt` 를 돌리면 창이 hop 만큼 미끄러질 때마다 같은
표본의 값이 미세하게 달라지고, 그것이 블록 경계에서 **계단**으로 보인다.
T-P 구간처럼 원래 평평한 곳에서 유독 크게 보인다 — 그곳의 내부 차분이
0.1 %R 인데 경계 차분이 2.2 %R, 즉 **23 배**였다.

**눈으로만 보이는 결함이라 기존 테스트가 전부 통과했다.** 스트리밍이
오프라인과 일치하는지(`verify_stream_processor`)도, 지연이 맞는지도,
`Aligner` 가 정렬하는지도 다 맞았다. 그래서 여기서 **경계 대 내부 비**를
직접 고정한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ecgdn.config import FS
from ecgdn.realtime.frontend_modes import BlockZeroPhaseFE, build_fe

HOP_S = 0.096
HOP = int(round(HOP_S * FS))


BEAT_S = 0.8                    # 봉우리 간격 (75 bpm 쯤)


def _signal(seconds=40.0, seed=7):
    """평평한 구간이 넓은 신호 — 이음매는 **평평한 곳에서** 보인다.

    실제 ECG 를 쓰면 T-P 안의 잡음이 분모를 키워 이음매가 묻힌다(D1 에서
    실제로 그렇다). 여기서는 결함 자체를 재야 하므로 잡음을 거의 안 넣는다.
    """
    n = int(seconds * FS)
    t = np.arange(n) / FS
    g = np.random.default_rng(seed)
    x = 0.5 * np.sin(2 * np.pi * 0.13 * t) + 0.3 * np.sin(2 * np.pi * 0.31 * t + 1.0)
    # **T 파가 반드시 있어야 한다.** 처음에는 QRS 만 넣었는데, 그러면 `odd`
    # 확장이 오히려 굴곡이 작게 나와 F-37 을 못 잡았다. 원인이 «창 끝의
    # 기울기» 이므로 **창 끝이 완만한 사면에 떨어질 수 있어야** 재현된다 —
    # QRS 만 있으면 창 끝은 거의 항상 평평한 곳이다. 시험 신호가 기제를
    # 담지 못하면 결함이 있어도 통과한다.
    tw = int(0.16 * FS)                     # T 파 — 넓고 완만한 사면
    for i in _beats(n):
        x[i:i + 10] += np.hanning(10) * 3.0                    # QRS
        j = i + int(0.20 * FS)
        if j + tw < n:
            x[j:j + tw] += np.hanning(tw) * 0.8
    return x + 1e-4 * g.standard_normal(n)


def _beats(n):
    return [i for i in range(int(0.4 * FS), n - 10, int(BEAT_S * FS))]


def _flat_mask(n):
    """봉우리 **사이**의 평평한 구간 — 위치로 고른다.

    처음에는 «차분이 작은 하위 60 %» 로 골랐는데 그것이 틀렸다: 이음매가 큰
    표본은 차분이 크므로 **스스로 걸러진다.** 재려는 것을 빼고 재는 꼴이라
    결함이 있는데도 비가 1.0 근처로 나왔다. 위치 기반이어야 한다.
    """
    m = np.zeros(n, bool)
    for i in _beats(n):
        a, b = i + int(0.40 * FS), i + int(BEAT_S * FS) - int(0.10 * FS)
        if b <= n:
            m[a:b] = True                    # T 파(0.20~0.36 s)가 끝난 뒤부터
    return m


def _seam_ratio(v, hop=HOP):
    """블록 경계의 1 차 차분이 내부보다 몇 배인가 (평평한 구간에서만).

    표본이 모자라면 **skip 이 아니라 실패**다. skip 은 통과처럼 보이는데,
    처리기를 매 블록 새로 만들어 출력이 0 이 된 시험이 실제로 그렇게
    「통과」했다. 시험이 못 돌면 그것을 알려야 한다.
    """
    d = np.abs(np.diff(v))
    flat = _flat_mask(v.size)[:d.size]
    on = np.zeros(d.size, bool)
    on[np.arange(hop - 1, d.size, hop)] = True
    a, b = d[flat & on], d[flat & ~on]
    assert a.size >= 10 and b.size >= 10, (
        f"표본이 모자라 이음매를 못 잰다 (경계 {a.size}, 내부 {b.size}, "
        f"출력 {v.size}) — 시험 자체가 성립 안 한다")
    return float(np.median(a) / max(np.median(b), 1e-12))


def _run(fe, x):
    return np.concatenate([fe.push(x[i:i + HOP]) for i in range(0, x.size, HOP)])


def test_cross_fade_removes_the_block_seam():
    """**이것이 F-36 의 회귀 시험이다.**

    교차 페이드를 끄면 계단이 나타나고, 켜면 사라진다. 둘 다 확인해야
    «시험이 실제로 그 결함을 잡는다» 가 성립한다.
    """
    x = _signal()
    bad = _seam_ratio(_run(BlockZeroPhaseFE(FS, xfade_s=0.0), x))
    good = _seam_ratio(_run(BlockZeroPhaseFE(FS, xfade_s=HOP_S), x))
    assert bad > 3.0, (
        f"교차 페이드를 껐는데 이음매가 안 보인다 (비 {bad:.2f}) — "
        "시험 신호가 결함을 못 드러내고 있다")
    assert good < 1.6, f"교차 페이드를 켰는데 이음매가 남았다 (비 {good:.2f})"
    assert good < bad / 2, f"교차 페이드가 이음매를 못 줄였다 ({bad:.2f} -> {good:.2f})"


def test_default_zerophase_mode_has_the_cross_fade_on():
    """기본값이 꺼져 있으면 화면에서 계단이 그대로 보인다."""
    fe = build_fe("zerophase", FS)
    assert fe.xfade == fe.hop, "기본 `zerophase` 의 교차 페이드가 hop 과 다르다"


def test_median_mode_does_not_pay_for_a_cross_fade_it_does_not_need():
    """**중앙값에는 이음매가 없다** — 켜면 지연만 hop 만큼 는다.

    중앙값은 창 `w1`·`w2` 안만 보는 **국소 연산**이라 창이 미끄러져도 결과가
    거의 안 바뀐다. `filtfilt` 는 **창 전체의 함수**라 바뀐다 — 그 차이가
    교차 페이드가 한쪽에만 필요한 이유다 (F-36).
    """
    fe = build_fe("median", FS)
    assert fe.xfade == 0, "중앙값 모드가 필요 없는 교차 페이드로 지연을 사고 있다"

    from ecgdn.realtime.frontend_modes import MedianBaselineFE
    # **처리기는 한 번만 만든다.** 매 블록마다 새로 만들면 상태가 없어 출력이
    # 0 이 되고, `_seam_ratio` 가 조용히 skip 한다 — 실제로 그렇게 짰다가
    # 「통과」로 보였다. skip 은 통과가 아니다.
    fe = MedianBaselineFE(FS, hop_s=HOP_S, xfade_s=0.0)
    x = _signal()
    v = np.concatenate([fe.push(x[i:i + HOP]) for i in range(0, x.size, HOP)])
    assert v.size > FS * 10, f"출력이 너무 짧다 ({v.size}) — 시험이 성립 안 한다"
    assert _seam_ratio(v) < 2.0, "중앙값에 이음매가 생겼다 — 전제가 바뀌었다"


def test_cross_fade_costs_exactly_its_width_in_latency():
    """**공짜가 아니다.** 지연 = 미리보기 + hop + 교차 페이드.

    지연 표기가 실제보다 짧으면 화면의 «지연» 표시가 거짓이 된다.
    """
    for xf in (0.0, 0.096, 0.25):
        fe = BlockZeroPhaseFE(FS, look_s=0.5, hop_s=HOP_S, xfade_s=xf)
        want = int(round(0.5 * FS)) + HOP + int(round(xf * FS))
        assert fe.latency_samples == want, f"xfade={xf}: {fe.latency_samples} != {want}"
    # 기본값(=hop)일 때도 맞아야 한다
    fe = BlockZeroPhaseFE(FS, look_s=0.5, hop_s=HOP_S)
    assert fe.latency_samples == int(round(0.5 * FS)) + 2 * HOP


def test_smaller_fe_hop_buys_back_the_latency():
    """**이것이 이 수정을 공짜로 만든다** (F-36).

    교차 페이드가 hop 과 같으므로 hop 을 절반으로 줄이면 지연이 그대로다.
    FE 는 블록당 0.6 ms 라 자주 돌려도 예산의 몇 % 다 — 추론 hop 과 분리해
    두었기에 CPU 도 안 는다.
    """
    before = BlockZeroPhaseFE(FS, look_s=0.5, hop_s=0.048, xfade_s=0.0)
    after = BlockZeroPhaseFE(FS, look_s=0.5, hop_s=0.024)
    assert after.latency_samples == before.latency_samples, (
        f"hop 을 절반으로 줄였는데 지연이 달라졌다 "
        f"({before.latency_samples} -> {after.latency_samples})")

    x = _signal()
    def run(fe, hop):
        return np.concatenate([fe.push(x[i:i + hop]) for i in range(0, x.size, hop)])
    bad = _seam_ratio(run(before, 12), 12)
    good = _seam_ratio(run(after, 6), 6)
    assert bad > 3.0 and good < 1.6, f"같은 지연에서 이음매가 안 사라졌다 ({bad:.2f} -> {good:.2f})"


def test_cross_fade_keeps_the_sample_contract():
    """계약: **같은 표본 번호를 더 늦게 낼 뿐** 시간축이 안 밀린다.

    이것이 깨지면 `Aligner` 와 원시 패널이 전부 어긋난다.
    """
    x = _signal(seconds=20.0)
    out = _run(BlockZeroPhaseFE(FS, xfade_s=HOP_S), x)
    assert out.size <= x.size
    # 확정된 출력은 입력의 **앞쪽부터 차례로** 대응해야 한다. 봉우리 하나만
    # 크게 만들어 두고 그 위치가 안 밀렸는지 본다 (봉우리가 다 같으면 argmax
    # 가 어느 것을 고를지 임의라 시험이 못 된다 — 처음에 그렇게 짰다).
    mark = _beats(x.size)[5]
    x[mark:mark + 10] += np.hanning(10) * 6.0
    out = _run(BlockZeroPhaseFE(FS, xfade_s=HOP_S), x)
    pk_out = int(np.argmax(out))
    assert abs(pk_out - (mark + 5)) <= 4, \
        f"봉우리가 {mark + 5} -> {pk_out} 으로 밀렸다 — 표본 번호 계약이 깨졌다"


def _curvature(v, mask=None):
    """평평한 구간 **안**의 굴곡 — 직선을 뺀 잔차의 RMS.

    이음매(구간 **사이**의 불연속)와 다른 양이다. 교차 페이드는 이음매를
    지우지만 굴곡은 거의 못 줄인다 — 원인이 다르기 때문이다 (F-37).
    """
    m = _flat_mask(v.size) if mask is None else mask
    out = []
    for i in _beats(v.size):
        a, b = i + int(0.40 * FS), i + int(BEAT_S * FS) - int(0.10 * FS)
        if b > v.size or b - a < 12:
            continue
        seg = v[a:b]
        t = np.arange(seg.size)
        out.append(np.sqrt(np.mean((seg - np.polyval(np.polyfit(t, seg, 1), t)) ** 2)))
    assert len(out) >= 5, f"박동이 모자라 굴곡을 못 잰다 ({len(out)})"
    return float(np.median(out))


def test_constant_padding_removes_the_curvature_in_flat_segments():
    """**F-37 의 회귀 시험** — 창 확장을 `odd` 로 되돌리면 굴곡이 돌아온다.

    `odd` 확장은 끝점을 중심으로 점대칭이라 **끝점의 기울기를 연장**한다.
    ECG 는 창 끝이 어디에 떨어지든 기울기가 있으므로 늘 가짜 램프가 붙고,
    0.5 Hz 고역통과가 그것을 과도현상으로 바꿔 평활 구간에 굴곡을 남긴다.
    """
    x = _signal()
    good = _curvature(_run(BlockZeroPhaseFE(FS), x))

    class _Odd(BlockZeroPhaseFE):
        def _process(self, w):
            import scipy.signal as sp
            pad = max(1, min(w.size - 1, w.size // 2))
            v = sp.sosfiltfilt(self._hp, w, padtype="odd", padlen=pad)
            return sp.sosfiltfilt(self._lp, v, padtype="odd", padlen=pad)

    bad = _curvature(_run(_Odd(FS), x))
    assert bad > good * 1.8, (
        f"`odd` 로 되돌렸는데 굴곡이 안 돌아온다 ({bad:.4f} vs {good:.4f}) — "
        "시험이 결함을 못 잡고 있다")


def test_the_block_filter_declares_its_padding_choice():
    """`constant` 가 조용히 `odd` 로 돌아가면 굴곡이 되살아난다 (F-37)."""
    src = (Path(__file__).resolve().parent.parent
           / "ecgdn" / "realtime" / "frontend_modes.py").read_text()
    body = src[src.index("class BlockZeroPhaseFE"):]
    body = body[:body.index("class MedianBaselineFE")]
    assert 'padtype="constant"' in body, "블록 영위상이 constant 확장을 안 쓴다"
    assert 'padtype="odd"' not in body, "odd 확장이 되살아났다"


def test_accumulator_stays_bounded():
    """누적기가 자라면 F-31 이 재현된다 — 시간이 갈수록 느려진다."""
    x = _signal(seconds=60.0)
    fe = BlockZeroPhaseFE(FS, xfade_s=HOP_S)
    _run(fe, x)
    assert fe._acc.size <= 4 * HOP, f"누적기가 {fe._acc.size} 로 자랐다"
    assert fe._wsum.size == fe._acc.size
