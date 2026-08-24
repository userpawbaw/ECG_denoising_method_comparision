#!/usr/bin/env python3
"""산출물이 낡았는지 검사한다 — 생성 코드가 그 뒤로 바뀌었는가.

    python scripts/check_freshness.py            # 전체 검사
    python scripts/check_freshness.py --strict   # 낡은 것이 있으면 종료코드 2

**왜 필요한가.** 파이프라인 앞단을 고치면 그 뒤로 만들어진 산출물이 한꺼번에
낡는데, **산출물만 보면 그 사실이 드러나지 않는다.** 이 프로젝트에서 두 번
일어났다.

- F-9: front-end 를 고치자 그 전에 학습한 체크포인트가 전부 무효가 됐다.
  결과표는 정상적으로 생성됐고, 표만 봐서는 조건이 섞인 것을 알 수 없었다.
- O-11: `measure_metric_floor.py` 를 두 번 고쳤는데 D0 산출물만 재생성되지
  않았다. `r_amp_err_pct` floor 가 6.5 % 어긋난 채로 커밋돼 있었다.

**git 커밋 해시로는 부족하다.** 문서만 고친 커밋에도 해시는 바뀌고, 반대로
커밋하지 않은 수정은 해시에 나타나지 않는다. 그래서 `save_manifest(sources=...)`
가 **생성 코드 파일의 내용 해시**를 남기고, 이 스크립트가 그것을 대조한다.

`sources` 가 없는 산출물은 **판정 불가**로 따로 센다 — 모른다는 것과 최신이라는
것은 다르다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecgdn.utils import stale_sources  # noqa: E402


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--strict", action="store_true",
                    help="낡은 산출물이 있으면 종료코드 2")
    a = ap.parse_args()

    # `runs/` 는 보존 스냅샷, `_archive*` 는 의도적으로 동결한 과거 결과다.
    # 둘 다 "낡은 것이 정상" 이므로 검사하지 않는다.
    mans = sorted(p for p in Path(a.root).rglob("manifest.json")
                  if "runs" not in p.parts
                  and not any(x.startswith("_archive") for x in p.parts))
    if not mans:
        print(f"[freshness] {a.root} 에 manifest 가 없다.")
        return 0

    fresh, stale, unknown = [], [], []
    for m in mans:
        try:
            bad = stale_sources(m)
        except Exception as e:                     # noqa: BLE001
            unknown.append((m, [f"(읽기 실패: {e})"])); continue
        if not bad:
            fresh.append(m)
        elif bad == ["(sources 미기록)"]:
            unknown.append((m, bad))
        else:
            stale.append((m, bad))

    print(f"[freshness] {len(mans)} 개 산출물 — "
          f"최신 {len(fresh)} / 낡음 {len(stale)} / 판정불가 {len(unknown)}")

    if stale:
        print("\n낡은 산출물 — 생성 코드가 바뀌었는데 재생성되지 않았다:")
        for m, bad in stale:
            print(f"  {m.parent}")
            for b in bad:
                print(f"      바뀐 코드: {b}")
    if unknown:
        print("\n판정 불가 — `save_manifest(sources=...)` 가 없다:")
        for m, _ in unknown:
            print(f"  {m.parent}")

    if stale and a.strict:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
