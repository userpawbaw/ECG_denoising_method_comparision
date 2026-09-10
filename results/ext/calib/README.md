# results/ext/calib — Colab T4 캘리브레이션 (2026-09-09)

**이 `sweep.csv` 는 러너가 쓴 것이 아니라 실행 로그에서 옮긴 것이다** — Colab
연결이 끊겨 산출물 폴더가 `s2` 한 줄만 남았고, 사용자가 로그를 전사해 왔다.
그래서 `history.json` · `log.csv` · `manifest.json` 이 없다. 근거 등급은
`[로그]` 이지 `[측정]` 이 아니다 (`docs/19_record_keeping.md` 8절).

담긴 값: `best_metric` · `best_epoch` · `n_epochs` · `fixed_snr_imp_scaled` ·
환경 지문. 판정에 쓴 것은 그게 전부라 결론(F-41)은 이 표만으로 재현된다.

    python3 scripts/analyze_seed_sweep.py results/ext/calib

전문은 `docs/20_findings.md` F-41, 실행 안내는 `docs/35_external_compute.md` §2-1.
