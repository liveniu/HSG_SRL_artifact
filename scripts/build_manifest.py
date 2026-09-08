#!/usr/bin/env python3
"""Write manifests/ARTIFACT_MANIFEST.csv with sha256 for published files."""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import ARTIFACT_ROOT  # noqa: E402

SKIP_DIR_NAMES = {
    ".git",
    ".idea",
    ".vscode",
    ".cursor",
    "__pycache__",
    ".venv",
    "work",
    "external",
    "runs",
    "smoke",
    "frozen_checkpoints",
    "training",
}
SKIP_SUFFIXES = {".pyc", ".pyo", ".pt", ".log", ".db-wal", ".db-shm"}
SKIP_FILE_NAMES = {"train_fab2_site_frozen_run_metadata.json"}
KEEP_DB = {"data/init/smoke_pair.db"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_files() -> list[Path]:
    files: list[Path] = []
    for path in ARTIFACT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(ARTIFACT_ROOT).parts
        if any(part in SKIP_DIR_NAMES for part in rel_parts):
            continue
        rel = path.relative_to(ARTIFACT_ROOT).as_posix()
        if path.suffix in SKIP_SUFFIXES:
            continue
        if path.suffix == ".db" and rel not in KEEP_DB:
            continue
        if path.name in SKIP_FILE_NAMES:
            continue
        if path.name == "ARTIFACT_MANIFEST.csv":
            continue
        files.append(path)
    return sorted(files)


def main() -> int:
    dest = ARTIFACT_ROOT / "manifests" / "ARTIFACT_MANIFEST.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in _iter_files():
        rel = path.relative_to(ARTIFACT_ROOT).as_posix()
        rows.append(
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    with dest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {dest} ({len(rows)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
