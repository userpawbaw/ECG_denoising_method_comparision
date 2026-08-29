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

import os
import re
import sys
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
        run = Path(str(ck).replace("{tag}", "d0")).parent.name
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
                needed.add(Path(str(ck).replace("{tag}", "d0")).parent.name)
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
                referenced.add(Path(str(ck).replace("{tag}", "d0")).parent.name)
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

def test_experiment_configs_use_tag_templated_checkpoints():
    """실험 config 의 체크포인트 경로는 `{tag}` 로 데이터축을 따라가야 한다.

    `results/m06_l1/best.pt` 처럼 축이 고정된 경로를 쓰면, D0 로 학습한 모델을
    D1 평가에 그대로 먹이는 사고가 조용히 일어난다. manifest 를 열어보기 전에는
    드러나지 않는다 (docs/99_status.md 2.1).
    """
    bad = []
    for p in CONFIGS:
        cfg = yaml.safe_load(p.read_text()) or {}
        for mid, spec in (cfg.get("dl_methods") or {}).items():
            ck = spec if isinstance(spec, str) else (spec or {}).get("ckpt")
            if ck and "{tag}" not in str(ck):
                bad.append(f"{p.name}:{mid} -> {ck}")
    assert not bad, f"데이터축이 고정된 체크포인트 경로: {bad}"


def test_runner_scripts_require_an_explicit_data_axis():
    """러너는 데이터축을 인자로 받아야 한다 — auto 에만 기대면 재현되지 않는다."""
    for name in ("run_all_training.sh", "run_all_experiments.sh"):
        t = (ROOT / "scripts" / name).read_text()
        assert "SOURCE=" in t and "synthetic|mitdb" in t, \
            f"{name} 에 데이터축 인자 처리가 없다"

def test_record_keeping_convention_is_followed():
    """기록 규약(docs/19_record_keeping.md)이 실제로 지켜지는지.

    규약을 만들어 놓고 새 항목을 기존 방식으로 적으면 의미가 없다. 실제로
    이 검사를 넣기 전에는 F-1~F-9 대부분에 **"놓쳤다면"** 이 빠져 있었다 —
    보고서에서 인용 가치가 가장 높은 항목인데도.

    잠정([잠정]) 항목은 아직 못 채운 것이 정상이므로 예외로 둔다.
    """
    import subprocess
    r = subprocess.run([sys.executable, "scripts/check_records.py"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"기록 규약 미충족:\n{r.stdout[-1500:]}"



def test_record_checker_still_detects_missing_records():
    """누락 검사 3종이 살아 있고 **실제로 잡는지** 확인한다.

    검사를 만들어 두고 나중에 지워도 스위트는 초록으로 남는다 — 그것이
    이 프로젝트가 반복해서 겪은 형태다. 그래서 존재만 보지 않고 **가짜
    누락을 넣어 잡히는지**까지 본다.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_records", ROOT / "scripts" / "check_records.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for fn in ("check_design_choices_have_a_decision",
               "check_findings_reach_the_report",
               "check_handoff_sections_name_a_destination"):
        assert hasattr(mod, fn), f"누락 검사 {fn} 이 없어졌다"

    # main 이 실제로 호출하는지 (정의만 하고 안 부르면 무의미)
    src = (ROOT / "scripts" / "check_records.py").read_text()
    main_body = src[src.index("def main("):]
    for fn in ("check_design_choices_have_a_decision",
               "check_findings_reach_the_report",
               "check_handoff_sections_name_a_destination"):
        assert fn in main_body, f"{fn} 이 정의만 되고 main 에서 호출되지 않는다"

    # 가짜 모델을 끼워 넣으면 잡아야 한다
    try:
        from ecgdn.models import MODELS
    except Exception:
        pytest.skip("torch 없음")
    sentinel = "zzz_model_with_no_decision_record"
    MODELS[sentinel] = lambda **kw: None
    try:
        bad = mod.check_design_choices_have_a_decision()
        assert any(sentinel in b for b in bad), \
            f"D 기록 없는 모델을 잡지 못했다: {bad}"
    finally:
        MODELS.pop(sentinel, None)


def test_report_body_matches_regenerated_artifacts():
    """보고서 본문이 자동 생성 문서와 어긋나면 잡아야 한다.

    O-16 — 참조 정의가 바뀌어 D0 산출물이 재생성됐는데 보고서 5.3 절을
    사흘 동안 안 맞췄다. 숫자만이 아니라 **해석이 틀렸다**: 걷힌 천장을
    여전히 있다고 서술하고 있었다.

    검사가 존재하는지가 아니라 **실제로 잡는지**를 본다. 처음 만든 판은
    생성 문서의 표 앵커가 틀려서 19 dB 어긋난 값도 통과시켰다 (F-18 계열 —
    울리지 않는 경고).
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_records", ROOT / "scripts" / "check_records.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rep = ROOT / "docs" / "91_report.md"
    if not rep.exists() or not (ROOT / "docs" / "90_results_d0.md").exists():
        pytest.skip("산출물 없음")

    assert mod.check_report_matches_generated_docs() == [], \
        "현재 보고서가 이미 산출물과 어긋나 있다"

    original = rep.read_text()
    try:
        # 정본과 다른 값을 심는다 (작은 어긋남도 잡혀야 한다)
        broken = re.sub(r"(\|\s*`M04`\s*\|\s*\*\*)([0-9.]+)(\*\*)",
                        r"\g<1>99.99\g<3>", original, count=1)
        assert broken != original, "5.3 절의 M04 행을 찾지 못했다 (표 형식이 바뀌었나)"
        rep.write_text(broken)
        bad = mod.check_report_matches_generated_docs()
        assert any("M04" in b for b in bad), \
            f"어긋난 값을 잡지 못했다 — 울리지 않는 경고다: {bad}"
    finally:
        rep.write_text(original)


def test_report_summary_exemptions_carry_a_reason():
    """면제는 이유와 함께여야 한다. 빈 면제는 '귀찮아서 뺐다' 와 구분되지 않는다."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_records", ROOT / "scripts" / "check_records.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fid, why in mod.F_SUMMARY_EXEMPT.items():
        assert len(why.strip()) > 20, f"{fid} 의 면제 사유가 너무 짧다: {why!r}"


# --------------------------------------------------------------------------
# 체크포인트 게이트 (scripts/check_ckpts.py)
# --------------------------------------------------------------------------

def test_training_gate_ignores_processes_that_merely_mention_the_script():
    """게이트가 감시 스크립트를 학습으로 오인하면 실험이 영구히 막힌다 (O-17).

    초판은 `/proc` 의 cmdline 에서 "scripts/train.py" 를 **부분 문자열**로
    찾았다. 그래서 그 문자열을 명령줄에 담은 완료 대기 스크립트 하나가
    게이트를 3 시간 넘게 막았고, 겉으로는 "아직 학습 중" 과 구별되지 않았다.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_ckpts", ROOT / "scripts" / "check_ckpts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # 명령줄에 문자열만 담은 프로세스는 학습이 아니다
    assert not mod._is_training_pid(os.getpid()), \
        "이 테스트 프로세스 자신이 학습으로 잡힌다 — 부분 문자열 매칭이다"

    # argv 원소로 들어오면 학습이다 (러너를 흉내낸다)
    import subprocess
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert not mod._is_training_pid(proc.pid)
    finally:
        proc.kill(); proc.wait()


def test_training_lock_records_its_pid():
    """락이 PID 를 남겨야 게이트가 그 하나만 보고 판정할 수 있다 (O-17)."""
    runner = (ROOT / "scripts" / "run_all_training.sh").read_text()
    assert '$$ > "$LOCK/pid"' in runner, "러너가 락에 PID 를 기록하지 않는다"
    assert 'rm -f "$LOCK/pid"' in runner, "종료 시 PID 파일을 지우지 않는다"


def test_experiment_runner_gates_on_checkpoints():
    """실험 러너는 첫 실험 전에 체크포인트 게이트를 통과해야 한다.

    run_exp.py 는 체크포인트가 없으면 `[skip]` 한 줄만 찍고 그 방법을 뺀 채
    표를 만든다. 학습이 덜 끝난 상태로 실험을 돌리는 사고가 실제로 가능하고,
    그 결과물은 정상적으로 보인다. 게이트 호출이 사라지면 그 방어가 사라진다.
    """
    t = (ROOT / "scripts" / "run_all_experiments.sh").read_text()
    assert "check_ckpts.py" in t, "실험 러너에 체크포인트 게이트 호출이 없다"
    gate = t.index("check_ckpts.py")
    # 주석에도 run_exp.py 가 나오므로 실제 호출 위치를 본다.
    first = t.index("python3 scripts/run_exp.py")
    assert gate < first, "게이트가 첫 실험보다 뒤에 있다 — 막지 못한다"


def test_ckpt_gate_covers_every_config_the_runner_runs():
    """게이트의 기본 config 목록이 실험 러너가 실제로 도는 것을 전부 덮어야 한다.

    러너에 실험을 추가하면서 게이트 목록을 안 고치면, 그 실험만 조용히
    검사 밖으로 빠진다.
    """
    runner = (ROOT / "scripts" / "run_all_experiments.sh").read_text()
    ran = set(re.findall(r"configs/(\w+)\.yaml", runner))
    ran |= {m for m in re.findall(r"for c in ([\w ]+); do", runner)
            for m in m.split()}
    # run_safety_probe.py 는 config 이름을 인자로 받지 않고 exp_e 를 직접 읽는다.
    ran.add("exp_e")
    ran = {c for c in ran if (ROOT / "configs" / f"{c}.yaml").exists()}

    src = (ROOT / "scripts" / "check_ckpts.py").read_text()
    m = re.search(r"DEFAULT_CONFIGS = \(([^)]+)\)", src)
    assert m, "check_ckpts.py 에서 DEFAULT_CONFIGS 를 찾지 못했다"
    covered = set(re.findall(r'"(\w+)"', m.group(1)))

    # dl_methods 가 있는 config 만 게이트 대상이다.
    needs_ckpt = {c for c in ran
                  if (yaml.safe_load((ROOT / "configs" / f"{c}.yaml").read_text())
                      or {}).get("dl_methods")}
    assert not (needs_ckpt - covered), \
        f"러너가 돌지만 게이트가 검사하지 않는 config: {sorted(needs_ckpt - covered)}"


def test_ckpt_gate_refuses_while_training_runs():
    """학습 중에는 게이트가 막아야 한다.

    `best.pt` 는 epoch 마다 갱신되므로 학습 도중에도 파일은 존재한다. 파일
    존재만 보면 게이트를 통과하고, 덜 학습된 모델이 그대로 표에 들어간다.
    """
    src = (ROOT / "scripts" / "check_ckpts.py").read_text()
    assert "def training_in_progress(" in src, \
        "게이트에 학습 진행 중 검사가 없다"
    assert ".train.lock" in src, "게이트가 학습 락을 보지 않는다"


# --------------------------------------------------------------------------
# 산출물이 저장소에 실제로 남아 있는가 (O-13)
# --------------------------------------------------------------------------

def _tracked() -> set[str]:
    import subprocess
    return set(subprocess.run(["git", "ls-files"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.split())


def test_deployment_artifacts_are_tracked_by_git():
    """실시간 처리에 그대로 쓰이는 산출물은 저장소에 남아야 한다.

    `.gitignore` 에 `results/` 한 줄이 있어서 **SWT 튜닝값과 지표 교정표가
    저장소에 하나도 올라가지 않았다**(O-13). 컨테이너가 회수되면 탐색을
    다시 해야 하고, 무엇보다 이 값들은 장비에 그대로 이식할 산출물이다.

    `a893f2b`(`data/` 규칙이 `ecgdn/data/` 패키지를 통째로 삼킨 건)와 같은
    원인이다 — 규칙이 의도보다 넓었고, 산출물만 봐서는 드러나지 않았다.
    """
    tracked = _tracked()
    need = []
    for tag in ("d0", "d1"):
        need += [f"results/{tag}/tune_swt/best.json",
                 f"results/{tag}/tune_swt/manifest.json",
                 f"results/{tag}/metric_floor/floor.csv"]
    missing = [p for p in need if p not in tracked and (ROOT / p).exists()]
    assert not missing, (
        f"디스크에 있으나 git 에 없는 배포 산출물: {missing}")


def test_trained_checkpoints_are_tracked_with_their_provenance():
    """체크포인트는 **출처와 함께** 추적해야 쓸모가 있다.

    `best.pt` 만 있고 `manifest.json`(학습 조건·코드 해시)이 없으면
    그 가중치가 어느 파이프라인의 산물인지 알 수 없다 (F-9).
    """
    tracked = _tracked()
    bad = []
    for p in sorted(ROOT.glob("results/d[01]/*/best.pt")):
        rel = p.relative_to(ROOT)
        if str(rel) not in tracked:
            bad.append(f"{rel} (체크포인트 미추적)")
        elif str(rel.parent / "manifest.json") not in tracked:
            bad.append(f"{rel.parent}/manifest.json (출처 미추적)")
    assert not bad, bad


def test_gitignore_never_reintroduces_a_blanket_results_rule():
    """`results/` 한 줄로 되돌리면 O-13 이 그대로 재발한다."""
    for line in (ROOT / ".gitignore").read_text().splitlines():
        t = line.split("#")[0].strip()
        assert t not in ("results/", "results", "/results/", "/results"), (
            "`results/` 통째 무시는 배포 산출물까지 지운다 — "
            "큰 파일만 패턴으로 제외할 것 (O-13)")


def test_resume_only_state_is_not_tracked():
    """`last.pt` 는 재개 전용이고 182 MB 다. 이력에 쌓으면 안 된다."""
    tracked = _tracked()
    bad = [p for p in tracked if p.endswith("last.pt")]
    assert not bad, f"재개 전용 파일이 추적되고 있다: {bad[:5]}"
