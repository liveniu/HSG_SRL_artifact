# Vendored HSG-SRL artifact copy. Isolated from the development repo;
# edit this file only. See src/SOURCE_MAP.csv.
"""Residual invoke gate: which cameras receive frozen MLP ΔP.

Coverage uses the overlap graph stored in the checkpoint (the graph the MLP
was trained on), not the yaml fusion triangle. Val-mean quality is applied
offline by pipeline/fit_residual_acceptor.py and stored in a JSON artifact
next to the checkpoint, analogous to T_c JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _norm_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))  # type: ignore[return-value]


def overlap_from_meta(meta: dict[str, Any] | None) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for scene in (meta or {}).get("scenes") or []:
        for pair in scene.get("overlap_pairs") or []:
            if len(pair) == 2:
                out.add(_norm_pair(pair[0], pair[1]))
    return out


def cameras_on_overlap(overlap: set[tuple[str, str]]) -> set[str]:
    cams: set[str] = set()
    for a, b in overlap:
        cams.add(a)
        cams.add(b)
    return cams


def load_residual_acceptor_json(path: str | Path) -> dict[str, Any]:
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    cameras = {str(c) for c in blob.get("residual_cameras") or []}
    blob["_residual_cameras"] = cameras
    return blob
