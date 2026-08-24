"""산출물 출처 추적 — 낡은 산출물을 재생성 없이 알아낼 수 있어야 한다.

F-9 와 O-11 이 같은 모양으로 두 번 일어났다: **파이프라인 앞단을 고치면 그
뒤로 만들어진 산출물이 한꺼번에 낡는데, 산출물만 보면 드러나지 않는다.**
git 커밋 해시로는 부족하다 — 문서만 고친 커밋에도 해시가 바뀌고, 커밋하지
않은 수정은 해시에 나타나지 않는다.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ecgdn.utils import archive_run, save_manifest, source_digest, stale_sources  # noqa: E402


def test_source_digest_is_content_based(tmp_path):
    """같은 내용이면 같은 해시, 한 글자만 바뀌어도 다른 해시."""
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    h1 = source_digest([f])[str(f)]
    f.write_text("x = 1\n")           # 내용 동일, mtime 만 바뀜
    assert source_digest([f])[str(f)] == h1, "mtime 변화로 해시가 바뀌면 안 된다"
    f.write_text("x = 2\n")
    assert source_digest([f])[str(f)] != h1


def test_missing_source_is_marked_not_crashed(tmp_path):
    d = source_digest([tmp_path / "없는파일.py"])
    assert list(d.values()) == ["missing"]


def test_stale_sources_detects_change(tmp_path):
    src = tmp_path / "gen.py"
    src.write_text("print(1)\n")
    out = tmp_path / "out"
    save_manifest(out, cfg={"a": 1}, sources=[src])

    assert stale_sources(out / "manifest.json") == [], "직후에는 최신이어야 한다"

    src.write_text("print(2)\n")
    assert stale_sources(out / "manifest.json") == [str(src)]


def test_unrecorded_sources_is_not_reported_as_fresh(tmp_path):
    """모른다는 것과 최신이라는 것은 다르다.

    `sources` 없는 산출물을 '최신' 으로 세면, 출처 추적을 붙이기 전의
    산출물이 전부 검증된 것처럼 보인다.
    """
    out = tmp_path / "out"
    save_manifest(out, cfg={"a": 1})          # sources 없이
    assert stale_sources(out / "manifest.json") == ["(sources 미기록)"]


def test_archive_run_preserves_current_outputs(tmp_path):
    """O-10: 재실행이 이전 기록을 덮지 않아야 한다."""
    d = tmp_path / "tune"
    d.mkdir()
    (d / "search.csv").write_text("v\n1\n")
    first = archive_run(d)
    assert (first / "search.csv").read_text() == "v\n1\n"

    (d / "search.csv").write_text("v\n2\n")   # 재실행이 덮어씀
    second = archive_run(d)
    assert (second / "search.csv").read_text() == "v\n2\n"
    assert (first / "search.csv").read_text() == "v\n1\n", \
        "이전 실행 기록이 남아 있어야 한다"
    assert first != second


def test_archive_run_does_not_recurse_into_itself(tmp_path):
    """`runs/` 안의 스냅샷이 다음 스냅샷에 다시 담기면 무한히 커진다."""
    d = tmp_path / "tune"
    d.mkdir()
    (d / "a.txt").write_text("x")
    archive_run(d)
    archive_run(d)
    snaps = sorted((d / "runs").iterdir())
    assert len(snaps) == 2
    for s in snaps:
        assert not (s / "runs").exists()
        assert [f.name for f in s.iterdir()] == ["a.txt"]


def test_archive_run_prunes_old_snapshots(tmp_path):
    d = tmp_path / "tune"
    d.mkdir()
    (d / "a.txt").write_text("x")
    for _ in range(5):
        archive_run(d, keep=3)
    assert len(list((d / "runs").iterdir())) <= 3


@pytest.mark.parametrize("script", ["check_freshness.py", "check_ckpts.py"])
def test_checker_scripts_run(script):
    r = subprocess.run([sys.executable, f"scripts/{script}"] +
                       ([] if script == "check_freshness.py" else ["--source", "synthetic"]),
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode in (0, 2), r.stderr[-800:]


def test_freshness_skips_frozen_archives():
    """`_archive*` 와 `runs/` 는 '낡은 것이 정상' 이므로 검사하지 않는다."""
    src = (ROOT / "scripts" / "check_freshness.py").read_text()
    assert '"runs" not in p.parts' in src
    assert '_archive' in src


def test_generators_declare_their_sources():
    """산출물을 만드는 스크립트는 전부 `sources=` 를 넘겨야 한다.

    하나라도 빠지면 그 산출물만 조용히 추적 밖으로 나간다.
    """
    missing = []
    for p in sorted((ROOT / "scripts").glob("*.py")) + [ROOT / "ecgdn" / "train.py"]:
        t = p.read_text()
        if "save_manifest(" not in t:
            continue
        # save_manifest 호출마다 sources= 가 붙어 있는지
        for chunk in t.split("save_manifest(")[1:]:
            head = chunk[:600]
            if "sources=" not in head:
                missing.append(p.name)
                break
    assert not missing, f"sources= 없이 manifest 를 쓰는 생성기: {missing}"


def test_train_declares_the_pipeline_it_depends_on():
    """F-9 의 재발 방지 — 학습 체크포인트는 데이터 경로 변경에 낡는다."""
    t = (ROOT / "ecgdn" / "train.py").read_text()
    for need in ("ecgdn/data/dataset.py", "ecgdn/methods/frontend.py"):
        assert need in t, f"train.py 가 {need} 를 출처로 선언하지 않는다"


def test_declared_sources_exist():
    """선언한 출처 경로가 실재해야 한다. 오타는 영원히 'missing' 으로 남는다."""
    import re
    bad = []
    for p in sorted((ROOT / "scripts").glob("*.py")) + [ROOT / "ecgdn" / "train.py"]:
        t = p.read_text()
        if "sources=" not in t:
            continue
        for chunk in t.split("sources=")[1:]:
            seg = chunk[:chunk.find("]") + 1] if "]" in chunk[:600] else chunk[:600]
            for m in re.finditer(r'"((?:ecgdn|scripts)/[\w./-]+\.py)"', seg):
                if not (ROOT / m.group(1)).exists():
                    bad.append(f"{p.name} -> {m.group(1)}")
    assert not bad, f"실재하지 않는 출처 경로: {bad}"


# --------------------------------------------------------------------------
# F-18 — 경고 장치의 임계값도 축을 따라가야 한다
# --------------------------------------------------------------------------

def test_snr_ceiling_is_axis_specific_and_lower_on_real_data():
    """추정기의 천장은 실기록에서 8 dB 낮다 (실측, docs/04_*).

    상수 하나로 두면 실기록에서 추정치가 이미 천장인데도 경고가 뜨지 않는다.
    """
    from ecgdn.eval.snr_estimation import SNR_CEILING_BY_AXIS
    assert set(SNR_CEILING_BY_AXIS) == {"d0", "d1"}
    assert SNR_CEILING_BY_AXIS["d1"] < SNR_CEILING_BY_AXIS["d0"], \
        "실기록 천장이 합성보다 낮아야 한다"


def test_ceiling_warning_respects_the_passed_threshold():
    """같은 신호라도 임계값에 따라 경고가 달라져야 한다.

    이 검사가 없으면 `ceiling_db` 를 받기만 하고 쓰지 않아도 통과한다 —
    F-18 의 원래 버그가 정확히 '값은 있는데 아무 일도 하지 않는' 형태였다.
    """
    import numpy as np
    from ecgdn.eval.snr_estimation import estimate_snr_all

    fs = 250.0
    t = np.arange(int(fs * 20)) / fs
    # 아주 깨끗한 주기 신호 — 추정치가 높게 나온다
    x = np.zeros_like(t)
    for k in range(20):
        c = int((0.2 + k) * fs)
        if c + 20 < len(x):
            x[c - 10:c + 10] += np.hanning(20)
    x += 1e-4 * np.random.default_rng(0).standard_normal(len(x))

    lo = estimate_snr_all(x, fs, ceiling_db=-100.0)
    hi = estimate_snr_all(x, fs, ceiling_db=1e6)
    assert lo["ceiling_warning"] == 1.0
    assert hi["ceiling_warning"] == 0.0
    assert lo["ceiling_db"] == -100.0


def test_real_snr_script_uses_the_real_data_ceiling():
    """실측 장비 신호는 실데이터다 — 합성 천장을 쓰면 경고가 뜨지 않는다."""
    t = (ROOT / "scripts" / "estimate_real_snr.py").read_text()
    assert 'SNR_CEILING_BY_AXIS["d1"]' in t, \
        "estimate_real_snr.py 가 실기록 천장을 쓰지 않는다"
    assert "SNR_CEILING_DB" not in t, "합성 기본 상수를 아직 참조한다"


def test_calibration_doc_reports_the_measured_ceiling():
    """교정 문서에 천장 표가 실려 있어야 한다 — 편향의 원인이기 때문이다."""
    for f in ("04_snr_estimator_calibration.md",
              "04_snr_estimator_calibration_d1.md"):
        t = (ROOT / "docs" / f).read_text()
        assert "추정기의 천장" in t, f"{f} 에 천장 절이 없다"
