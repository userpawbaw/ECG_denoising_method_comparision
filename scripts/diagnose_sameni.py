"""STEP 13 자가진단 — docs/00_review.md A-2 의 6개 실패 원인 점검.

**목적**
    "Sameni 방식이 잘 안 된다" 는 결론을 내리기 전에, 그 구현이 공정한 상태인지 먼저 증명한다.
    이 스크립트를 통과하지 못한 구현을 비교 실험에 넣으면 그 비교 결과 자체가 무효다.

산출: docs/06_sameni_diagnosis.md, results/fig/sameni_*.png
"""
import _bootstrap  # noqa: F401

import argparse

import numpy as np
import matplotlib.pyplot as plt

from ecgdn.config import DEFAULT_KERNEL, FS
from ecgdn.data.mixer import mix_at_snr
from ecgdn.data.noise import awgn, mixed_noise
from ecgdn.data.sources import real_clean_segments
from ecgdn.data.synthetic import synth_ecg
from ecgdn.eval.engine import evaluate, trim_guard
from ecgdn.eval.rpeak import detect_rpeaks
from ecgdn.methods.kalman_sameni import SameniKalman, assign_phase
from ecgdn.utils import ensure_dir, rng, save_manifest

PASS, FAIL = "PASS", "FAIL"


def _imp(s, y, d):
    m = evaluate(s.x, y, d(y, s.fs), s.fs, r_peaks_ref=s.r_peaks,
                 do_morph=False, do_spectral=False)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", "--source", dest="data", default="synthetic",
                    choices=["synthetic", "mitdb"], help="config 없이 도는 보조 스크립트다. synthetic 이 기본 — D0 결과(이미 보고서에 인용된 값)와의 연속성을 지킨다")
    ap.add_argument("--dur", type=float, default=90.0)
    ap.add_argument("--snr", type=float, default=5.0)
    args = ap.parse_args()
    tag = "d0" if args.data == "synthetic" else "d1"
    suffix = "" if tag == "d0" else "_d1"

    if args.data == "mitdb":
        s = real_clean_segments(1, args.dur, fs=FS, split="train")[0]
    else:
        s = synth_ecg(args.dur, fs=FS, seed=5)
    _xr = getattr(s, "x_raw", s.x)
    y, _, _ = mix_at_snr(_xr, awgn(len(_xr), s.fs, rng("diag")), args.snr)  # F-12
    checks: list[tuple[str, str, str, str]] = []   # (id, 항목, 결과, 상세)

    # ---------- C1  위상
    det = detect_rpeaks(y, s.fs)
    theta, omega = assign_phase(det, len(y), s.fs)
    # theta 의 0-crossing (상승) 위치
    zc = np.where((theta[:-1] < 0) & (theta[1:] >= 0))[0] + 1
    if zc.size and det.size:
        err_ms = np.array([np.min(np.abs(det - z)) for z in zc]) / s.fs * 1e3
        ok = float(np.median(err_ms)) <= 10.0
        checks.append(("C1", "위상 0-crossing 이 R-peak 와 일치",
                       PASS if ok else FAIL,
                       f"median |Δ| = {np.median(err_ms):.2f} ms (≤ 10 ms), "
                       f"zero-crossing {zc.size} 개 / R-peak {det.size} 개"))
    else:
        checks.append(("C1", "위상 0-crossing", FAIL, "0-crossing 없음"))

    # ---------- C2  baseline (front-end 유무)
    d_on = SameniKalman(smooth=True, use_frontend=True)
    d_off = SameniKalman(smooth=True, use_frontend=False)
    n_bw, _ = mixed_noise(len(s.x), s.fs, rng("diag_bw"), kinds=["bw_synth", "awgn"])
    y_bw, _, _ = mix_at_snr(s.x, n_bw, args.snr)
    a = _imp(s, y_bw, d_on)["snr_imp_scaled"]
    b = _imp(s, y_bw, d_off)["snr_imp_scaled"]
    checks.append(("C2", "front-end HPF 를 EKF 앞단에 적용", PASS if a > b else FAIL,
                   f"baseline wander 포함 조건: FE-ON {a:+.2f} dB vs FE-OFF {b:+.2f} dB "
                   f"(차이 {a - b:+.2f} dB)"))

    # ---------- C3  커널 적합 품질
    d = SameniKalman(smooth=True, n_kernels=7)
    m = _imp(s, y, d)
    r2 = d.last_info["r2"]
    checks.append(("C3", "phase-averaged template 에 Gaussian 적합", PASS if r2 > 0.98 else FAIL,
                   f"R² = {r2:.4f} (> 0.98), 커널 7개"))
    fit = d.last_info["fit"]

    # ---------- C4  R 추정 정확도 (합성이라 참값을 안다)
    sl = trim_guard(len(y), s.fs)
    scale = d.last_info["scale"]
    true_noise_var = float(np.var((y - s.x)[sl])) / scale ** 2
    est_r = d.last_info["r_meas"]
    ratio_db = 10 * np.log10(est_r / max(true_noise_var, 1e-30))
    checks.append(("C4", "측정잡음 R 추정", PASS if abs(ratio_db) <= 3.0 else FAIL,
                   f"추정/참값 = {ratio_db:+.2f} dB (|·| ≤ 3 dB)"))

    # ---------- C5  EKS > EKF
    ekf = SameniKalman(smooth=False, n_kernels=7)
    eks = SameniKalman(smooth=True, n_kernels=7)
    v_f = _imp(s, y, ekf)["snr_imp_scaled"]
    v_s = _imp(s, y, eks)["snr_imp_scaled"]
    checks.append(("C5", "EKS(평활) 가 EKF(전방) 보다 우수", PASS if v_s > v_f else FAIL,
                   f"EKF {v_f:+.2f} dB → EKS {v_s:+.2f} dB (차이 {v_s - v_f:+.2f} dB)"))

    # ---------- C6  진폭 정규화 (gain bias)
    g = m["gain_bias"]
    checks.append(("C6", "진폭 정규화/역정규화", PASS if 0.9 <= g <= 1.1 else FAIL,
                   f"gain_bias = {g:.4f} (0.9 ~ 1.1)"))

    # ---------- DoD: 생성 커널 주입 시 성능 — **합성축에서만 의미가 있다**
    #
    # `DEFAULT_KERNEL` 은 `synthetic.py` 가 신호를 만들 때 쓰는 바로 그 커널이다.
    # 그래서 D0 에서는 이것을 넣는 것이 "참 파라미터를 준다" 에 해당하고,
    # 파라미터 추정 오차를 배제한 **방법 자체의 잠재력**을 잰다.
    #
    # D1(실기록)에서는 그렇지 않다. 실제 ECG 에 대응하는 McSharry 참 파라미터라는
    # 것은 존재하지 않고, `DEFAULT_KERNEL` 은 그 환자와 아무 관계가 없는 남의
    # 커널일 뿐이다. 그것을 넣은 값을 D0 의 값과 나란히 놓으면 **서로 다른 양을
    # 비교하는 것**이 된다 (F-17). 실제로 D1 에서는 적합 커널(+11.23)이
    # 주입 커널(+5.91)보다 낫다 — 주입값이 천장이 아니라는 직접 증거다.
    d_true = SameniKalman(smooth=True, kernel=DEFAULT_KERNEL)
    v_true = _imp(s, y, d_true)["snr_imp_scaled"]
    v_fit = v_s
    if args.data == "synthetic":
        dod = v_true >= 8.0
        checks.append(("DoD", f"생성 커널 주입 + EKS, 입력 {args.snr:.0f} dB 에서 개선 ≥ 8 dB",
                       PASS if dod else FAIL, f"{v_true:+.2f} dB"))
    else:
        checks.append(("DoD", "생성 커널 주입 (합성축 전용 검사)", "— N/A",
                       f"{v_true:+.2f} dB — 실기록에는 '참 커널' 이 없어 "
                       f"이 값은 천장이 아니다 (적합 커널 {v_fit:+.2f} dB 가 더 낫다)"))

    # ---------- 그림
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    t = np.arange(len(y)) / s.fs
    m0, m1 = int(20 * s.fs), int(23 * s.fs)
    ax[0].plot(t[m0:m1], theta[m0:m1], lw=1.0, color="#2ca02c", label="assigned phase θ(t)")
    for p in det[(det >= m0) & (det < m1)]:
        ax[0].axvline(t[p], color="#d62728", lw=0.8, ls="--")
    ax[0].set_title("C1: assigned phase (red dashed = detected R-peak; must cross 0)")
    ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("rad"); ax[0].legend(fontsize=8)
    ax[1].plot(fit.template_grid, fit.template, lw=1.4, label="phase-averaged template")
    ax[1].plot(fit.template_grid, fit.fitted, lw=1.2, ls="--", label=f"fitted (R²={fit.r2:.4f})")
    ax[1].set_title("C3: Gaussian kernel fit to phase-averaged template"); ax[1].set_xlabel("phase [rad]")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    figd = ensure_dir(f"results/{tag}/fig")
    fig.savefig(figd / "sameni_diagnosis.png", dpi=130); plt.close(fig)

    # ---------- 문서
    # N/A(그 축에 해당 없음)는 분모에서도 뺀다. 실패로 세면 구현에 문제가
    # 있는 것처럼 읽힌다 — D1 의 DoD 가 정확히 그런 경우다.
    n_pass = sum(1 for c in checks if c[2] == PASS)
    n_applicable = sum(1 for c in checks if c[2] in (PASS, FAIL))
    n_na = len(checks) - n_applicable
    # 이 버그 수정 이력은 D0 에서 일어난 일이다. D1 문서에 "수정 전 +5.9" 를
    # 그대로 실으면 **다른 축의 값과 나란히 놓여 수정이 무효였던 것처럼 보인다**
    # (실제로 D1 의 v_true 가 우연히 +5.9 라 그렇게 읽혔다).
    if args.data == "synthetic":
        fix_history = [
            f"수정 전후 (합성 ECG, 입력 {args.snr:.0f} dB, EKS + 생성 커널 주입):",
            "",
            "| | `snr_imp_scaled` |", "|---|---|",
            "| 수정 전 | +5.9 dB |", f"| 수정 후 | **{v_true:+.1f} dB** |",
            "",
        ]
    else:
        fix_history = [
            "> 위 두 버그의 수정 전후 대조는 합성축에서 이뤄졌다. "
            "`docs/06_sameni_diagnosis.md` 를 볼 것.",
            "",
        ]

    if v_true >= v_fit:
        conclusion_gap = [
            f"- 그 차이 **{v_true - v_fit:+.2f} dB** 가 '파라미터를 추정해야 한다' 는 "
            "조건이 만드는 실용상 한계다.",
        ]
    else:
        conclusion_gap = [
            f"- **주입 커널이 적합 커널보다 {v_fit - v_true:.2f} dB 나쁘다.** "
            "이 축에서는 `DEFAULT_KERNEL` 이 '참 파라미터' 가 아니기 때문이다 — "
            "실제 ECG 에 대응하는 McSharry 참 파라미터는 존재하지 않는다. "
            "따라서 주입값을 성능의 천장으로 읽으면 안 된다 (F-17).",
        ]

    md = ["# 06. Sameni EKF/EKS 자가진단",
          "", "> 자동 생성: `python scripts/diagnose_sameni.py`", "",
          "## 왜 이 문서가 필요한가",
          "",
          "프로젝트 초기에 \"Sameni 방식이 기대만큼 개선되지 않는다\" 는 관찰이 있었다.",
          "그러나 **약하게 구현된 baseline 을 두고 다른 방법이 이겼다고 결론내면 그 비교는 무효다.**",
          "그래서 비교 실험에 넣기 전에 아래 6개 항목을 자동으로 점검한다.",
          "",
          f"**결과: {n_pass}/{n_applicable} 통과**"
          + (f" (그 외 {n_na} 항목은 이 데이터축에 해당 없음)" if n_na else ""), "",
          "| # | 항목 | 결과 | 상세 |", "|---|---|---|---|"]
    for cid, name, res, detail in checks:
        md.append(f"| `{cid}` | {name} | **{res}** | {detail} |")
    md += ["", f"![diagnosis](../results/{tag}/fig/sameni_diagnosis.png)", "",
           "## 구현 과정에서 실제로 발견된 버그 2개",
           "",
           "이 진단 절차가 없었다면 놓쳤을 것들이다. 둘 다 성능을 10 dB 이상 떨어뜨렸다.",
           "",
           "### (a) phase-averaged template 의 상수 오프셋",
           "",
           "상태방정식은 `dz/dθ` 만 규정하므로 **z 의 상수 오프셋은 동역학과 무관**하다.",
           "그런데 front-end HPF 가 DC 를 제거한 신호와 오프셋이 있는 커널 합을 그대로 비교하면",
           "적합 R² 가 0.99 → 0.82 로 떨어지고, 그 잔차가 그대로 `q_z` 과대추정으로 이어진다.",
           "→ template 과 커널 합 양쪽의 평균을 제거하고 적합한다.",
           "",
           "### (b) `q_z` 를 누적 분산으로 잘못 계산",
           "",
           "`q_z` 는 **매 샘플의** 상태잡음 분산인데, 관측되는 모델 잔차는",
           "beat 한 개 길이 `L` 동안 **누적된** 편차다 (위상이 beat 마다 재고정되므로).",
           "random walk 의 누적 분산은 `L · q_z` 이므로 `L` 로 나눠야 한다.",
           "이 정규화를 빼면 `q_z` 가 수백 배 과대추정되어 필터가 사실상 평활화를 하지 않는다.",
           "",
           *fix_history,
           "## 결론",
           "",
           f"- 생성 커널 주입 시 {v_true:+.2f} dB, template 적합 시 {v_fit:+.2f} dB.",
           *conclusion_gap,
           "- EKS 는 EKF 대비 " f"{v_s - v_f:+.2f} dB 우수하다. **EKF 만 쓰면 안 된다.**",
           "",
           "## 재현", "", "```bash",
           f"python scripts/diagnose_sameni.py --data {args.data}", "```"]
    doc = ensure_dir("docs") / f"06_sameni_diagnosis{suffix}.md"
    doc.write_text("\n".join(md) + "\n")

    print("=" * 78)
    for cid, name, res, detail in checks:
        print(f"[{res}] {cid:4s} {name:38s} {detail}")
    print("=" * 78)
    print(f"{n_pass}/{n_applicable} passed"
          + (f" (+{n_na} N/A)" if n_na else "") + f"  ->  {doc}")
    save_manifest(f"results/{tag}/sameni_diagnosis", cfg=vars(args), sources=[
        "scripts/diagnose_sameni.py", "ecgdn/methods/kalman_sameni.py", "ecgdn/eval/engine.py", "ecgdn/eval/signal_metrics.py", "ecgdn/eval/morphology.py", "ecgdn/eval/rpeak.py"])
    return 0 if n_pass == n_applicable else 1


if __name__ == "__main__":
    raise SystemExit(main())
