"""STEP 14: PhysioNet 데이터 다운로드 + 검증.

    python scripts/download_data.py --db mitdb --out data/raw
    python scripts/download_data.py --db nstdb --out data/raw

원격 세션에서 physionet.org 가 차단되면 **로컬에서 실행**한 뒤 data/raw 를 옮겨오면 된다.
"""
import _bootstrap  # noqa: F401

import argparse

from ecgdn.data.download import DBS, download, verify


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", nargs="+", default=["mitdb", "nstdb"], choices=sorted(DBS))
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    rc = 0
    for db in args.db:
        try:
            if not args.verify_only:
                print(f"[download] {db} ({DBS[db]}) ...")
                download(db, args.out)
            info = verify(db, args.out)
            print(f"[ok] {db}: {info['n_records']} records -> {info['dir']}")
        except Exception as e:
            print(f"[FAIL] {db}: {type(e).__name__}: {e}")
            print("       physionet.org 접근이 막힌 환경일 수 있다. 로컬에서 실행 후 "
                  "data/raw 를 복사해 오면 된다.")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
