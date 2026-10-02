"""시리얼 브리지 (`scripts/serial_bridge.py`) — R-5/R-6.

하드웨어가 없으므로 **모의 보드**(`ReplaySource`)가 진짜 보드와 같은 선
규격으로 말하는지, 그리고 화면에 보내기 전 **방법끼리 시각이 맞는지**를
고정한다. 둘 다 틀려도 화면은 그럴듯하게 나오므로 눈으로는 못 잡는다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from ecgdn.realtime.serial_link import AsciiParser, BinaryParser

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "serial_bridge", ROOT / "scripts" / "serial_bridge.py")
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as e:                       # pragma: no cover
        pytest.skip(f"serial_bridge 를 못 읽었다: {e}")
    return m


# ------------------------------------------------------- 모의 보드가 진짜처럼
def test_replay_source_speaks_the_real_wire_format():
    """모의 보드가 실제 파서로 읽혀야 대체 경로로서 의미가 있다."""
    m = _mod()
    src = m.ReplaySource("synth", 250, True, drift_ppm=0.0)
    p = BinaryParser()
    got = 0
    t0 = time.perf_counter()
    while got < 100 and time.perf_counter() - t0 < 5.0:
        ch = p.feed(src.read())
        got += len(ch)
        assert ch.n_bad == 0, "모의 보드가 깨진 프레임을 냈다"
        assert ch.n_lost == 0, "드롭을 안 켰는데 손실이 생겼다"
    assert got >= 100


def test_replay_source_ascii_mode_matches_the_ide_format():
    m = _mod()
    src = m.ReplaySource("synth", 250, False, drift_ppm=0.0)
    p = AsciiParser()
    t0 = time.perf_counter()
    got = 0
    while got < 20 and time.perf_counter() - t0 < 5.0:
        got += len(p.feed(src.read()))
    assert got >= 20


def test_replay_source_drops_show_up_as_sequence_gaps():
    """모의 드롭이 **파서에서 손실로 보여야** 한다 — 그래야 대응을 시험할 수 있다."""
    m = _mod()
    src = m.ReplaySource("synth", 250, True, drift_ppm=0.0, drop_rate=0.2, seed=3)
    p = BinaryParser()
    lost = n = 0
    t0 = time.perf_counter()
    while n < 300 and time.perf_counter() - t0 < 5.0:
        ch = p.feed(src.read())
        n += len(ch)
        lost += ch.n_lost
    assert lost > 0, "드롭을 켰는데 손실이 하나도 안 잡혔다"


# ------------------------------------------------------------------ 정렬
def test_aligner_emits_only_the_range_every_method_has():
    """한 방법이 아직 못 낸 구간을 내보내면 화면에서 시각이 어긋난다."""
    m = _mod()
    al = m.Aligner(["A", "B"])
    al.add("A", 0, np.arange(10.0))
    idx, out = al.take()
    assert out == {}, "B 가 아직 아무것도 안 냈는데 내보냈다"
    al.add("B", 4, np.arange(3.0))
    idx, out = al.take()
    assert idx == 4 and len(out["A"]) == len(out["B"]) == 3
    assert out["A"] == [4.0, 5.0, 6.0]


def test_aligner_never_repeats_or_skips_a_sample():
    m = _mod()
    al = m.Aligner(["A", "B"])
    seen: list[float] = []
    for k in range(5):
        al.add("A", 0, np.arange(k * 7.0, k * 7.0 + 7))
        al.add("B", 0, np.arange(k * 7.0, k * 7.0 + 7))
        idx, out = al.take()
        if out:
            assert idx == len(seen), f"연속이 끊겼다: {idx} vs {len(seen)}"
            seen.extend(out["A"])
    assert seen == list(np.arange(float(len(seen))))


def test_aligner_takes_nothing_twice():
    m = _mod()
    al = m.Aligner(["A"])
    al.add("A", 0, np.arange(5.0))
    assert al.take()[1]["A"] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert al.take()[1] == {}


# --------------------------------------------------- 이중 필터를 막는 규칙
def test_front_end_methods_are_not_given_their_own_processor():
    """`M_FE`·`M01` 은 **필터가 곧 방법**이다 — 단 **front-end 가 대역통과일 때만.**

    실시간 경로는 앞단에 FE 를 이미 한 번 걸었으므로, 대역통과 FE 에서 이들에
    처리기를 또 붙이면 같은 필터가 두 번 걸린다 — R-4 에서 −15 dB 를 만든 그
    실수다 (F-25). `verify_stream_processor.py` 가 이들을 건너뛰는 것과 같은 이유다.

    **중앙값 front-end 는 대역통과가 아니다.** 거기서 `M01` 을 FE 출력으로
    대체하면 «대역통과+노치» 라고 이름 붙인 자리에 **전혀 다른 필터의 출력**이
    나간다. 그래서 대체는 모드에 따라 갈려야 한다.
    """
    from ecgdn.realtime.frontend_modes import FE_MODES, fe_intrinsic
    m = _mod()
    assert m.ALWAYS_INTRINSIC == {"M_FE"}
    for mode, spec in FE_MODES.items():
        got = fe_intrinsic(mode)
        assert "M_FE" in got, f"{mode}: M_FE 는 언제나 FE 출력 그 자체다"
        if spec["bandpass"]:
            assert got == {"M_FE", "M01", "M01d"}, f"{mode}: 이중 필터가 된다 (F-25)"
        else:
            assert "M01" not in got, \
                f"{mode}: 대역통과가 아닌 FE 출력을 M01 이라고 내보내고 있다"
    src = ROOT / "scripts" / "serial_bridge.py"
    assert "split_names(" in src.read_text(), \
        "모드별로 FE 내재 방법을 가르는 분기가 사라졌다"


def test_the_live_page_never_shows_a_performance_number():
    """모드 A 에는 참값이 없다 — SNR 을 띄우면 그것은 지어낸 숫자다(난관 6)."""
    html = (ROOT / "demo" / "live.html").read_text()
    for bad in ("SNR 개선", "dB</", "snr_imp", "축 평균"):
        assert bad not in html, f"실측 화면에 성능 수치가 들어갔다: {bad}"


# ------------------------------------------- 시간이 갈수록 느려지지 않아야 한다
def test_aligner_buffers_stay_bounded():
    """**보낸 것은 버려야 한다** (F-31).

    `Aligner.buf` 가 안 줄어들면 세션 내내 자라고, 파이썬 GC 가 추적 컨테이너의
    슬롯을 전부 훑으므로 **gen2 수집 비용이 그 길이에 비례**한다. 초반엔 멀쩡하고
    10 분쯤 뒤 RTF 와 큐드롭이 함께 오르는 증상이 그것이었다. `raw` 는 20 s 로
    묶여 있었는데 정렬기 버퍼만 빠져 있었다.
    """
    import numpy as np
    m = _mod()
    al = m.Aligner(["a", "b"])
    for step in range(400):                       # 400 x 12 = 4800 표본
        blk = np.zeros(12)
        al.add("a", 0, blk)
        al.add("b", 0, blk)
        al.take()
    total = sum(len(v) for v in al.buf.values())
    assert total <= 4 * 12, (
        f"정렬기 버퍼가 {total} 로 자랐다 — 보낸 것을 안 버리고 있다 (F-31)")
    assert al.sent == 400 * 12, "보낸 위치 계산이 틀렸다"


def test_aligner_still_aligns_after_trimming():
    """트림이 **정렬 자체를 깨뜨리지 않는지** — 값과 시작 번호가 유지돼야 한다."""
    import numpy as np
    m = _mod()
    al = m.Aligner(["a", "b"])
    al.add("a", 0, np.arange(0, 30, dtype=float))
    al.add("b", 10, np.arange(10, 25, dtype=float))   # 늦게 시작하고 짧다
    idx, out = al.take()
    assert idx == 10, f"공통 구간은 10 에서 시작해야 한다 (얻은 값 {idx})"
    assert out["a"] == list(range(10, 25)) and out["b"] == list(range(10, 25))
    # 다음 블록도 이어져야 한다
    al.add("a", 0, np.arange(30, 36, dtype=float))
    al.add("b", 10, np.arange(25, 31, dtype=float))
    idx2, out2 = al.take()
    assert idx2 == 25, f"이어지는 구간은 25 에서 시작해야 한다 (얻은 값 {idx2})"
    assert out2["a"] == list(range(25, 31)) and out2["b"] == list(range(25, 31))

# ------------------------------------------- 화면으로 미는 길 (SSE)
def test_publish_refuses_an_already_serialised_payload():
    """**두 번 직렬화하면 브라우저가 문자열을 받는다** (F-52).

    `JSON.parse` 는 성공하고 결과가 문자열이라 `m.reset` 이 undefined 가 된다.
    화면에서는 «아무 일도 안 일어남» 으로만 보여서 눈으로는 못 잡는다.
    """
    m = _mod()
    hub = m.Hub()
    with pytest.raises(TypeError):
        hub.publish(json.dumps({"reset": True}))


def test_the_reset_notice_arrives_as_an_object_not_a_string():
    m = _mod()
    hub = m.Hub()
    q = hub.register()
    hub.publish({"reset": True, "fe": "median", "fe_label": "중앙값",
                 "fe_lat_ms": 448})
    got = json.loads(q.get_nowait())          # 브라우저의 JSON.parse
    assert isinstance(got, dict), "선에 실린 것이 «JSON 을 담은 JSON» 이다"
    assert got["reset"] is True
    assert got["fe_label"] == "중앙값"


def test_no_call_site_serialises_before_publishing():
    """가드가 있어도 **호출부에 남아 있으면 시연 중에 터진다** — 미리 잡는다."""
    src = (ROOT / "scripts" / "serial_bridge.py").read_text()
    assert "hub.publish(json.dumps(" not in src, \
        "publish 앞에서 json.dumps 를 부르고 있다 (F-52)"


# --------------------------------------- 도착 시각으로 재는 fs · 도착 간격 (D-38)
def _feed(meter, fs, t0, t1, block_s, lag_s=0.0):
    """`fs` 로 오는 선을 `block_s` 덩어리로 받는다. 누적은 처리 시점과 무관하다."""
    t, cum = t0, int(fs * t0)
    while t < t1:
        t += block_s
        cum = int(fs * t)          # 덩어리마다 반올림하면 그 오차를 재게 된다
        meter.add(t, cum)          # 도착 시각 — lag 는 처리 쪽 사정이라 안 들어간다
    return cum


def test_rate_meter_is_not_fooled_by_a_backlog():
    """F-55: 처리가 15 s 밀려도 **도착 시각**으로 재면 250 Hz 다 (전에는 36.5 · 71.5 Hz)."""
    m = _mod()
    r = m.RateMeter(10.0)
    _feed(r, 250, 0.0, 30.0, 0.05)
    assert abs(r.rate() - 250) < 2


def test_rate_meter_is_not_biased_by_big_blocks():
    """읽기가 1.64 s 덩어리여도(윈도우 read(4096)) 기울기는 그대로다."""
    m = _mod()
    r = m.RateMeter(10.0)
    _feed(r, 250, 0.0, 40.0, 1.64)
    assert abs(r.rate() - 250) < 3


def test_rate_meter_needs_a_full_window_and_stays_bounded():
    m = _mod()
    r = m.RateMeter(10.0)
    _feed(r, 250, 0.0, 5.0, 0.05)
    assert r.rate() is None
    _feed(r, 250, 5.0, 600.0, 0.05)
    assert len(r.pts) < 500, "점이 세션 내내 자란다 (F-31 과 같은 모양)"


def test_fs_check_warns_once_after_two_bad_windows_only():
    m = _mod()
    c = m.FsCheck(250, 10.0)
    assert c.step(0.0, 500.0) is None          # 한 창만 어긋남
    assert c.step(5.0, 500.0) is None          # 창이 겹친다 — 판정 안 함
    assert c.step(10.0, 500.0) == 500.0        # 둘째 창 — 경고
    assert c.step(20.0, 500.0) is None         # 한 번만
    ok = m.FsCheck(250, 10.0)
    assert all(ok.step(t, 251.0) is None for t in range(0, 100, 10))


def test_fs_check_forgives_a_single_bad_window():
    m = _mod()
    c = m.FsCheck(250, 10.0)
    assert c.step(0.0, 213.0) is None
    assert c.step(10.0, 250.0) is None
    assert c.step(20.0, 213.0) is None


def test_gap_meter_reports_the_longest_silence():
    m = _mod()
    g = m.GapMeter(2.0)
    for t in (0.0, 0.01, 0.02, 1.66, 1.67):
        g.add(t)
    assert abs(g.max_gap_ms(1.68) - 1640) < 1
    for k in range(400):
        g.add(1.68 + k * 0.01)
    assert g.max_gap_ms(5.68) < 20, "2 s 지난 간격은 잊어야 한다"
    assert g.max_gap_ms(6.68) > 900, "지금 기다리는 중인 것도 간격이다"


def test_the_live_page_shows_the_arrival_gap():
    html = (ROOT / "demo" / "live.html").read_text()
    assert 'id="gap"' in html and "s.gap_ms" in html


# ------------------------------------- 덩어리 크기가 값을 바꾸지 않는다 (F-56)
@pytest.mark.parametrize("mode", ["zerophase", "median", "causal"])
def test_front_end_output_does_not_depend_on_block_size(mode):
    """윈도우 실보드는 372 샘플 덩어리로 왔다 (F-55). 블록 영위상이 그때 1 % 달랐고
    2000 샘플에서는 죽었다 — 버퍼를 «뒤에서 keep 개» 로 잘랐기 때문이다 (F-56)."""
    from ecgdn.realtime.frontend_modes import build_fe
    x = np.random.default_rng(0).standard_normal(250 * 20)

    def run(bs):
        fe = build_fe(mode, 250, hop_s=6 / 250)
        return np.concatenate([fe.push(x[i:i + bs]) for i in range(0, x.size, bs)])

    ref = run(1)
    for bs in (12, 95, 372, 2000):
        np.testing.assert_array_equal(run(bs), ref, err_msg=f"{mode} · 덩어리 {bs}")


def test_bridge_feeds_processors_in_hop_steps():
    """`StreamProcessor` 는 덩어리가 크면 다른 값을 낸다(설계 — 확정된 만큼 바로 낸다).
    브리지는 그래서 hop 씩 나눠 넣는다 (F-56). 그 줄이 빠지면 여기서 걸린다."""
    src = (ROOT / "scripts" / "serial_bridge.py").read_text()
    assert "p.push(x[j:j + args.hop]) for j in range(0, x.size, args.hop)" in src


# ------------------------------- 브리지가 고를 수 있는 딥러닝은 실제로 올라온다 (O-37)
def test_every_bridge_dl_checkpoint_loads_strictly():
    """`n_blocks` 를 넣으며 키가 `enc.0.c1` -> `enc.0.0.c1` 로 바뀌어 옛 `best.pt` 가 전부
    적재에서 죽었다. 학습 테스트는 새로 만든 모델만 저장·적재하므로 못 잡았다 (O-37)."""
    pytest.importorskip("torch")
    from ecgdn.methods.dl_wrapper import load_checkpoint
    m = _mod()
    for mid, tag in m.DL_TAGS.items():
        ck = ROOT / "results" / "d1" / tag / "best.pt"
        assert ck.exists(), f"{mid}: {ck} 가 없다"
        load_checkpoint(ck)                     # strict — 키가 하나라도 어긋나면 죽는다


def test_bridge_names_dl_methods_by_their_own_id():
    m = _mod()
    pytest.importorskip("torch")
    assert m.build_stream_method("M06L6").name == "M06L6"
