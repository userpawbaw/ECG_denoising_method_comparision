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
    ap.add_argument("--db", nargs="+", default=["mitdb", "nstdb"], choices=sorted(DBS),
        help="받을 데이터베이스 (mitdb · nstdb). 여러 개를 나열할 수 있다")
    ap.add_argument("--out", default="data/raw",
        help="받은 것을 둘 곳")
    ap.add_argument("--verify-only", action="store_true",
        help="받지 않고 **이미 있는 것만** 검사한다")
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
