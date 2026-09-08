# Vendored HSG-SRL artifact copy. Isolated from the development repo;
# edit this file only. See src/SOURCE_MAP.csv.
"""Resolve paths against the published artifact root.

YAML paths, default output directories, and ``data/`` are relative to the
artifact root. Scripts should not rely on ``../`` or the process cwd.
"""

from __future__ import annotations

from pathlib import Path

# Artifact isolation: this file lives at src/utils/project_paths.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path, *, base: Path | None = None) -> Path:
    """Resolve a relative path against the artifact root (or ``base``)."""
    p = Path(path)
    if p.is_absolute():
        return p
    root = PROJECT_ROOT if base is None else base
    return (root / p).resolve()
