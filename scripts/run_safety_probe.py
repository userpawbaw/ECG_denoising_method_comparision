"""STEP 26 / EXP-E — 안전성 프로브 (docs/00_review.md C-3, C-4).

**질문**: denoiser 가 잡음을 제거하는 것이 아니라, 없는 파형을 **만들어내지는 않는가?**

의료 신호에서 이건 SNR 보다 중요한 질문이다. 특히 생성형/딥러닝 계열은
"이 환자 ECG 라면 이렇게 생겼을 것" 을 그럴듯하게 그려낼 수 있다.

세 가지 프로브
  P1 (beat dropout) : clean 에서 1 beat 구간(R ± 200 ms)을 등전위선으로 지우고 잡음을 덮는다.
                      -> 그 자리에 QRS 를 만들어내면 실패.
  P2 (asystole)     : 3 초 구간을 등전위선으로 치환하고 잡음을 덮는다. -> 상동.
  P3 (ectopic)      : 부정맥(PVC, 'V') beat 만 골라 형태 보존을 정상 beat 와 비교한다.
                      -> PVC 가 정상 QRS 모양으로 '교정' 되면 실패.

산출: results/{tag}/exp_e/{probe.csv, summary.md}, docs/07_safety_probe_{tag}.md
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import ecgdn.methods  # noqa: F401
from ecgdn.data.mixer import mix_at_snr
from ecgdn.data.noise import mixed_noise
from ecgdn.data.nstdb import make_banks
from ecgdn.data.sources import get_source, resolve_source_kind, source_tag
from ecgdn.eval.morphology import beat_matrix
from ecgdn.registry import build
from ecgdn.utils import ensure_dir, rng, save_manifest

# main() 이 --source 로 채운다. _methods 가 ckpt 의 {tag} 치환에 쓴다.
_TAG = ["d0"]


def _methods(cfg):
    ms = {mid: build(mid) for mid in cfg.get("methods", [])}
    tag = _TAG[0]
    for mid, spec in (cfg.get("dl_methods") or {}).items():
        from ecgdn.methods.dl_wrapper import DLDenoiser
        spec = {"ckpt": spec} if isinstance(spec, str) else spec
        spec = dict(spec, ckpt=str(spec["ckpt"]).replace("{tag}", tag))
        if not Path(spec["ckpt"]).exists():
            print(f"[skip] {mid}: no checkpoint"); continue
        ms[mid] = DLDenoiser(ckpt=spec["ckpt"], name=mid, pre=spec.get("pre"), batch=32)
    return ms


def _energy_ratio(xhat, x_ref_clean, i0, i1):
    """지운 구간에서 출력이 만들어낸 에너지 / 원래 그 구간의 에너지."""
    seg = xhat[i0:i1] - np.median(xhat[max(0, i0 - 200):i0 + 1] if i0 > 0 else xhat[i0:i1])
    ref = x_ref_clean[i0:i1] - np.median(x_ref_clean[i0:i1])
    pr = float(np.mean(ref ** 2))
    return float(np.mean(seg ** 2) / max(pr, 1e-30))


def probe_p1_p2(items, methods, snr_db, banks, out_rows, mode="p1"):
    for it in items:
        x, fs = it["x"].astype(np.float64), it["fs"]
        rp = np.asarray(it["r_peaks"], dtype=int)
        if rp.size < 8:
            continue
        x_mod = x.copy()
        if mode == "p1":
            j = rp[len(rp) // 2]
            half = int(round(0.20 * fs))
            i0, i1 = max(0, j - half), min(len(x), j + half)
        else:
            i0 = int(len(x) * 0.4)
            i1 = min(len(x), i0 + int(3.0 * fs))
        base = float(np.median(x[max(0, i0 - int(0.3 * fs)):i0])) if i0 > 0 else 0.0
        x_mod[i0:i1] = base                              # 등전위선으로 치환

        g = rng("probe", mode, it["record"], it["seg"])
        n, _ = mixed_noise(len(x), fs, g, banks=banks)
        y, _, _ = mix_at_snr(x_mod, n, snr_db)
        for mid, fn in methods.items():
            ctx = {"x_clean": x_mod} if getattr(fn, "needs_clean", False) else {}
            xh = fn(y, fs, ctx)
            out_rows.append(dict(probe=mode, record=it["record"], seg=it["seg"],
                                 method=mid,
                                 halluc_energy=_energy_ratio(xh, x, i0, i1),
                                 gap_s=(i1 - i0) / fs))


def probe_p3(items, methods, snr_db, banks, out_rows):
    for it in items:
        x, fs = it["x"].astype(np.float64), it["fs"]
        rp = np.asarray(it["r_peaks"], dtype=int)
        sym = np.asarray(it["symbols"])
        if rp.size < 8 or not np.any(sym == "V"):
            continue
        g = rng("probe", "p3", it["record"], it["seg"])
        n, _ = mixed_noise(len(x), fs, g, banks=banks)
        y, _, _ = mix_at_snr(x, n, snr_db)
        for mid, fn in methods.items():
            ctx = {"x_clean": x} if getattr(fn, "needs_clean", False) else {}
            xh = fn(y, fs, ctx)
            for want in ("N", "V"):
                m = sym == want
                if m.sum() < 2:
                    continue
                br, used = beat_matrix(x, rp[m], fs)
                bh, _ = beat_matrix(xh, rp[m], fs)
                if br.shape != bh.shape or br.size == 0:
                    continue
                a = br - br.mean(1, keepdims=True)
                b = bh - bh.mean(1, keepdims=True)
                den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
                with np.errstate(invalid="ignore", divide="ignore"):
                    cc = (a * b).sum(1) / den
                out_rows.append(dict(probe="p3", record=it["record"], seg=it["seg"],
                                     method=mid, beat_type=want,
                                     beat_cc=float(np.nanmean(cc)), n_beats=int(m.sum())))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/exp_e.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--source", default=None, choices=("auto", "synthetic", "mitdb"),
                    help="config 의 data.source 를 덮어쓴다")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.source is not None:
        cfg.setdefault("data", {})["source"] = args.source
    d = cfg.get("data", {})
    snr = float(cfg.get("snr_db", 5.0))

    requested = d.get("source", "auto")
    kind = resolve_source_kind(requested)
    _TAG[0] = source_tag(requested)
    if requested == "auto":
        print(f"[exp_e] source=auto -> {kind!r} 로 해석됨. --source {kind} 명시 권장.")

    src = get_source(d.get("source", "auto"), dur_s=float(d.get("dur_s", 300.0)),
                     n_test=int(d.get("n_test", 22)))
    banks = make_banks("test", d.get("nstdb_root", "data/raw/nstdb"))
    from ecgdn.data.dataset import build_eval_set
    items = build_eval_set(src, "test", seg_s=float(d.get("seg_s", 60.0)),
                           snr_grid=[snr], noise_conditions=("mixed",), banks=banks,
                           n_seg_per_record=int(d.get("n_seg_per_record", 1)),
                           seed="probe")
    if args.limit:
        items = items[: args.limit]
    methods = _methods(cfg)
    print(f"[exp_e] tag={_TAG[0]} items={len(items)} methods={list(methods)} snr={snr} dB")

    rows: list[dict] = []
    probe_p1_p2(items, methods, snr, banks, rows, "p1")
    probe_p1_p2(items, methods, snr, banks, rows, "p2")
    probe_p3(items, methods, snr, banks, rows)
    df = pd.DataFrame(rows)
    out = ensure_dir(f"results/{_TAG[0]}/exp_e")
    df.to_csv(out / "probe.csv", index=False)
    save_manifest(out, cfg=cfg)

    axis = "D0 — 합성 ECG" if _TAG[0] == "d0" else "D1 — MIT-BIH + NSTDB"
    md = [f"# 07. 안전성 프로브 (EXP-E, {_TAG[0].upper()})", "",
          f"> 자동 생성: `python scripts/run_safety_probe.py --source {kind}`", "",
          f"> **데이터축: {axis}**", "",
          "**질문: denoiser 가 없는 파형을 만들어내지는 않는가?**", "",
          "의료 신호에서는 SNR 보다 중요한 질문이다. 딥러닝/생성형 계열은",
          "\"이 환자 ECG 라면 이렇게 생겼을 것\" 을 그럴듯하게 그려낼 수 있다.", "",
          f"조건: 입력 SNR {snr:.0f} dB, 혼합 잡음, 평가 구간 {len(items)} 개.", ""]

    for pr, name, desc, col, good in [
        ("p1", "P1 — beat dropout",
         "clean 에서 **1 beat 구간(R ± 200 ms)** 을 등전위선으로 지우고 잡음을 덮었다. "
         "출력이 그 자리에 만들어낸 에너지를, 원래 그 구간이 갖고 있던 에너지로 나눈 값. "
         "**작을수록 안전**하다. 1.0 에 가까우면 사라진 beat 를 그대로 복원(=생성)했다는 뜻이다.",
         "halluc_energy", "낮을수록 안전"),
        ("p2", "P2 — asystole (3 s)",
         "3 초 구간을 통째로 등전위선으로 치환했다. 심정지 구간에 beat 를 생성하면 "
         "임상적으로 치명적이다.", "halluc_energy", "낮을수록 안전"),
    ]:
        sub = df[df.probe == pr]
        if sub.empty:
            continue
        t = sub.groupby("method")[col].agg(["mean", "std", "max"]).round(4)
        md += [f"## {name}", "", desc, "", f"| method | mean | std | max |", "|---|---|---|---|"]
        for m, r in t.iterrows():
            md.append(f"| `{m}` | {r['mean']:.4f} | {r['std']:.4f} | {r['max']:.4f} |")
        md.append("")

    sub = df[df.probe == "p3"]
    if not sub.empty:
        piv = sub.pivot_table(index="method", columns="beat_type", values="beat_cc",
                              aggfunc="mean").round(4)
        if "N" in piv and "V" in piv:
            piv["V - N"] = (piv["V"] - piv["N"]).round(4)
        md += ["## P3 — ectopic beat (PVC) 형태 보존", "",
               "부정맥 beat 만 골라 형태 보존(beat template correlation)을 정상 beat 와 비교했다.",
               "**`V - N` 이 크게 음수면 그 방법은 정상 beat 에 편향되어 부정맥을 훼손한다.**",
               "", "| method | " + " | ".join(str(c) for c in piv.columns) + " |",
               "|---" * (len(piv.columns) + 1) + "|"]
        for m, r in piv.iterrows():
            md.append(f"| `{m}` | " + " | ".join(f"{v:.4f}" for v in r.to_numpy()) + " |")
        md.append("")

    md += ["## 해석 지침", "",
           "- `halluc_energy` 는 **낮을수록** 좋다. residual 구조(출력 = 입력 − 예측잡음)를 쓰는",
           "  모델은 원리적으로 이 값이 낮아야 한다. 높게 나오면 그 모델은 '복원' 이 아니라 '생성' 을 하고 있다.",
           "- 다만 잡음이 그대로 남아도 이 값은 올라간다. 따라서 **잡음 제거량(EXP-A)과 함께** 봐야 한다.",
           "- P3 의 `V - N` 이 0 근처면 부정맥에 대해 편향이 없다는 뜻이다.",
           "", "## 재현", "", "```bash", "python scripts/run_safety_probe.py", "```"]
    doc = ensure_dir("docs") / f"07_safety_probe_{_TAG[0]}.md"
    doc.write_text("\n".join(md) + "\n")
    print(df.groupby(["probe", "method"]).size().to_string())
    print(f"-> {doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
