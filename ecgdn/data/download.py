"""PhysioNet database download and verification helpers.

This module provides the API expected by scripts/download_data.py:
    DBS
    download(db, out)
    verify(db, out)
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict

import wfdb


# PhysioNet database names and their canonical versions.
DBS: Dict[str, str] = {
    "mitdb": "1.0.0",
    "nstdb": "1.0.0",
}

# RECORDS is the authoritative list of records in each database.
BASE_URL = "https://physionet.org/files"


def _get_records(db: str) -> list[str]:
    """Read the database RECORDS file from PhysioNet."""
    version = DBS[db]
    url = f"{BASE_URL}/{db}/{version}/RECORDS"
    with urllib.request.urlopen(url) as response:
        text = response.read().decode("utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        destination.write_bytes(response.read())


def download(db: str, out: str | Path) -> Path:
    """Download one PhysioNet database into out/<db>.

    The download is intentionally explicit and database-specific. It uses
    PhysioNet's RECORDS file and downloads the WFDB files belonging to each
    listed record.
    """
    if db not in DBS:
        raise ValueError(f"Unsupported database: {db!r}. Choose from {sorted(DBS)}")

    out = Path(out)
    db_dir = out / db
    db_dir.mkdir(parents=True, exist_ok=True)

    version = DBS[db]
    records = _get_records(db)

    # Download the files needed by WFDB for each record.
    # MITDB/NSTDB records may have .dat/.hea/.atr files.
    # The RECORDS file gives record names; the directory listing is not
    # relied upon, so the common WFDB extensions are requested explicitly.
    extensions = (".hea", ".dat", ".atr")

    for i, record in enumerate(records, 1):
        print(f"[download] {db}: {i}/{len(records)} {record}")
        for ext in extensions:
            url = f"{BASE_URL}/{db}/{version}/{record}{ext}"
            destination = db_dir / f"{record}{ext}"

            try:
                _download_file(url, destination)
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    # Some records may legitimately lack one of the
                    # optional extensions.
                    continue
                raise

    return db_dir


def verify(db: str, out: str | Path) -> dict:
    """Verify downloaded WFDB records and return a compact summary."""
    if db not in DBS:
        raise ValueError(f"Unsupported database: {db!r}. Choose from {sorted(DBS)}")

    db_dir = Path(out) / db
    if not db_dir.exists():
        raise FileNotFoundError(f"Database directory does not exist: {db_dir}")

    expected = _get_records(db)
    verified = []
    failures = []

    for record in expected:
        try:
            # read_header validates that the WFDB header is parseable.
            record_path = db_dir / record
            wfdb.rdheader(str(record_path))
            verified.append(record)
        except Exception as exc:
            failures.append((record, f"{type(exc).__name__}: {exc}"))

    if failures:
        preview = "; ".join(f"{r}: {e}" for r, e in failures[:5])
        raise RuntimeError(
            f"{db}: {len(failures)} record(s) failed verification. {preview}"
        )

    return {
        "db": db,
        "n_records": len(verified),
        "dir": str(db_dir),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, choices=sorted(DBS))
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if not args.verify_only:
        download(args.db, args.out)

    print(verify(args.db, args.out))
