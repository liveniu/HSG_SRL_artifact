#!/usr/bin/env python3
"""Fail if published artifact files contain CJK characters."""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import ARTIFACT_ROOT  # noqa: E402

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]")

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
SKIP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".pdf",
    ".pt",
    ".pth",
    ".db",
    ".pyc",
    ".pyo",
    ".bin",
    ".npz",
}


def main() -> int:
    hits: list[str] = []
    for path in sorted(ARTIFACT_ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(ARTIFACT_ROOT).parts
        if any(part in SKIP_DIR_NAMES for part in rel_parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if path.name.startswith("_tmp_"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if CJK_RE.search(line):
                rel = path.relative_to(ARTIFACT_ROOT).as_posix()
                hits.append(f"{rel}:{i}: {line.strip()[:160]}")
    if hits:
        print(f"CJK characters found in {len(hits)} line(s):", file=sys.stderr)
        for row in hits:
            print(row, file=sys.stderr)
        return 1
    print("English check passed (no CJK in published files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
