#!/usr/bin/env python3
"""Locate the isolated artifact root and put vendored packages on sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ARTIFACT_ROOT / "src"


def ensure_import_path() -> Path:
    src = str(SRC_ROOT)
    if src not in sys.path:
        sys.path.insert(0, src)
    return ARTIFACT_ROOT
