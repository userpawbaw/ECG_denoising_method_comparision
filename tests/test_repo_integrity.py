"""저장소 자체의 무결성 — 코드가 아니라 **구성과 문서의 정합성**을 잰다.

여기 있는 검사들은 전부 실제로 발생한 결함에서 나왔다.

  * STEP 28/29 산출물 문서명이 이미 쓰이는 번호와 충돌했다.
    문서만 읽어서는 드러나지 않고, 그 STEP 을 실행하는 날 드러난다.
  * loss ablation 이 일부 loss 만 학습된 채 완료 처리됐다 (F-9).
  * config 가 존재하지 않는 방법 ID 를 가리켜도 run_exp 는 조용히 넘어갔다.

이 파일은 데이터도 체크포인트도 요구하지 않는다. `results/` 는 git 밖이므로
"학습 결과물이 있는가" 는 검사하지 않고, **소스 안에서 닫히는 참조**만 본다.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
DOCS = sorted((ROOT / "docs").glob("*.md"))
CONFIGS = sorted((ROOT / "configs").glob("*.yaml"))


def _registry() -> set[str]:
    import ecgdn.methods  # noqa: F401  (등록 부수효과)
    from ecgdn.registry import available
    return set(available())


# --------------------------------------------------------------- 문서 참조
@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_doc_cross_references_resolve(doc: Path):
    """문서가 가리키는 다른 문서가 실제로 있어야 한다.

    없으면 둘 중 하나다: 파일명을 잘못 적었거나, 번호가 충돌했거나.
    아직 생성되지 않은 **산출물** 문서는 예외로 둔다 — 그 STEP 이 외부 조건
    (하드웨어/데이터) 대기 중이라 정상적으로 비어 있을 수 있다.
    """
    produced_later = {"08a_acquisition_log.md", "08b_real_snr.md",
                      "10_loss_ablation.md"}
    text = doc.read_text()
    missing = sorted({m for m in re.findall(r"docs/([0-9A-Za-z_]+\.md)", text)
                      if m not in produced_later and not (ROOT / "docs" / m).exists()})
    assert not missing, f"{doc.name} 이 없는 문서를 가리킨다: {missing}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_doc_code_references_resolve(doc: Path):
    """문서가 가리키는 스크립트/모듈이 실제로 있어야 한다."""
    text = doc.read_text()
    refs = set(re.findall(r"`((?:scripts|ecgdn|configs|tests)/[\w./{}, ]+?)`", text))
    missing = []
    for r in refs:
        if "{" in r or " " in r:        # `configs/m06_l{1,2,3}.yaml` 같은 축약형
            continue
        if not (ROOT / r).exists():
            missing.append(r)
    assert not missing, f"{doc.name} 이 없는 파일을 가리킨다: {sorted(missing)}"


# ----------------------------------------------------------------- config
@pytest.mark.parametrize("cfg_path", CONFIGS, ids=lambda p: p.name)
def test_config_methods_are_registered(cfg_path: Path):
    """실험 config 의 방법 ID 가 전부 레지스트리에 있어야 한다.

    없는 ID 는 build() 단계에서야 터진다 — 즉 실험을 한참 돌린 뒤에.
    """
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    unknown = sorted(set(cfg.get("methods") or []) - _registry())
    assert not unknown, f"{cfg_path.name}: 미등록 방법 {unknown}"


@pytest.mark.parametrize("cfg_path", CONFIGS, ids=lambda p: p.name)
def test_dl_checkpoints_have_a_training_config(cfg_path: Path):
    """dl_methods 가 가리키는 체크포인트마다 그것을 만드는 학습 config 가 있어야 한다.

    `results/` 는 git 밖이라 파일 존재는 검사할 수 없다. 대신 **재현 경로**를
    검사한다: results/<run>/best.pt 를 쓴다면 configs/<run>.yaml 이 있어야
    누구든 그 체크포인트를 다시 만들 수 있다.
    """
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    orphan = []
    for mid, spec in (cfg.get("dl_methods") or {}).items():
        ck = spec if isinstance(spec, str) else (spec or {}).get("ckpt")
        if not ck:
            continue
        run = Path(ck).parent.name
        if not (ROOT / "configs" / f"{run}.yaml").exists():
            orphan.append(f"{mid} -> {ck} (configs/{run}.yaml 없음)")
    assert not orphan, f"{cfg_path.name}: 재현 불가능한 체크포인트 참조 {orphan}"


def test_training_runner_covers_every_referenced_checkpoint():
    """어떤 실험이 쓰는 체크포인트는 전부 기본 학습 러너가 만들어야 한다.

    러너에서 빠진 학습이 있으면, 그 실험은 '체크포인트 없음' 으로 조용히
    건너뛰어진다. F-9 가 정확히 그렇게 발생했다.
    """
    runner = (ROOT / "scripts" / "run_all_training.sh").read_text()
    m = re.search(r'RUNS="\$\{\*:-([^}]+)\}"', runner)
    assert m, "run_all_training.sh 에서 기본 RUNS 를 찾지 못했다"
    runs = set(m.group(1).split())

    needed: set[str] = set()
    for p in CONFIGS:
        cfg = yaml.safe_load(p.read_text()) or {}
        for spec in (cfg.get("dl_methods") or {}).values():
            ck = spec if isinstance(spec, str) else (spec or {}).get("ckpt")
            if ck:
                needed.add(Path(ck).parent.name)
    assert not (needed - runs), \
        f"실험이 쓰지만 러너가 학습하지 않는 체크포인트: {sorted(needed - runs)}"


def test_loss_ablation_covers_every_trained_loss():
    """학습만 되고 어떤 표에도 안 실리는 loss 설정이 없어야 한다.

    러너가 학습하는 m0*_l* 는 전부 ablation config 에 등장해야 한다.
    아니면 계산은 태우고 결과는 버리는 셈이다.
    """
    runner = (ROOT / "scripts" / "run_all_training.sh").read_text()
    m = re.search(r'RUNS="\$\{\*:-([^}]+)\}"', runner)
    runs = {r for r in m.group(1).split() if re.fullmatch(r"m\d+_l\d+", r)}

    referenced: set[str] = set()
    for p in CONFIGS:
        cfg = yaml.safe_load(p.read_text()) or {}
        for spec in (cfg.get("dl_methods") or {}).values():
            ck = spec if isinstance(spec, str) else (spec or {}).get("ckpt")
            if ck:
                referenced.add(Path(ck).parent.name)
    assert not (runs - referenced), \
        f"학습되지만 어떤 실험/표에도 안 쓰이는 설정: {sorted(runs - referenced)}"


def test_status_table_has_no_completed_step_without_artifact():
    """상태표에서 ✅ 인 STEP 의 '산출물' 칸에 적힌 소스 파일이 실제로 있어야 한다.

    F-9 는 산출물이 없는 STEP 이 ✅ 로 남아 있어서 생겼다.
    """
    text = (ROOT / "docs" / "02_procedure.md").read_text()
    bad = []
    for line in text.splitlines():
        if not line.startswith("|") or "✅" not in line:
            continue
        for ref in re.findall(r"`((?:scripts|ecgdn|configs|tests)/[\w./]+)`", line):
            if not (ROOT / ref).exists():
                bad.append(ref)
    assert not bad, f"✅ 로 표시된 STEP 이 없는 산출물을 가리킨다: {sorted(set(bad))}"

def test_every_package_source_file_is_tracked_by_git():
    """디스크에 있는 소스가 전부 git 에 들어 있어야 한다.

    `.gitignore` 에 `data/` 라고 쓰면 git 은 **모든 경로의** data 디렉터리를
    무시한다. 그래서 `ecgdn/data/` 의 10개 파일(mitdb, nstdb, dataset, sources,
    splits, noise, mixer, windows, synthetic, arduino)이 커밋된 적 없이 남아
    있었다 — 저장소를 클론하면 프로젝트가 import 조차 되지 않는 상태였다.

    로컬 작업 디렉터리에서는 파일이 멀쩡히 보이므로 어떤 테스트도, 어떤 실행도
    이걸 잡지 못한다. git 이 무엇을 담고 있는지 직접 물어야만 드러난다.
    """
    import subprocess
    tracked = set(subprocess.run(["git", "ls-files"], cwd=ROOT, check=True,
                                 capture_output=True, text=True).stdout.split())
    on_disk = set()
    for sub in ("ecgdn", "scripts", "tests", "configs"):
        for f in (ROOT / sub).rglob("*"):
            if f.is_file() and f.suffix in (".py", ".yaml", ".sh") \
                    and "__pycache__" not in f.parts:
                on_disk.add(str(f.relative_to(ROOT)))
    missing = sorted(on_disk - tracked)
    assert not missing, f"디스크에 있으나 git 에 없는 소스: {missing}"


def test_gitignore_data_rule_is_anchored_to_repo_root():
    """`data` 무시 규칙은 반드시 선행 슬래시로 루트에 고정해야 한다."""
    lines = [l.strip() for l in (ROOT / ".gitignore").read_text().splitlines()]
    bad = [l for l in lines
           if l and not l.startswith("#") and not l.startswith("!")
           and l.rstrip("/*") == "data"]
    assert not bad, (f"루트에 고정되지 않은 data 규칙 {bad} — 이러면 ecgdn/data/ 같은 "
                     f"하위 패키지가 통째로 무시된다. '/data/*' 로 쓸 것")
