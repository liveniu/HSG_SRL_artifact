# Vendored HSG-SRL artifact copy. Isolated from the development repo;
# edit this file only. See src/SOURCE_MAP.csv.
"""Image write helper.

Some Windows OpenCV builds (including 4.13) let ``cv.imwrite`` fail silently
(False, no file) while ``cv.imencode`` still works. This module always uses
imencode plus ``Path.write_bytes``.
"""

from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np


def _encode_params_for_suffix(suffix: str, params: list[int] | None) -> list[int]:
    if params:
        return list(params)
    if suffix in {".jpg", ".jpeg"}:
        return [int(cv.IMWRITE_JPEG_QUALITY), 95]
    if suffix == ".png":
        return [int(cv.IMWRITE_PNG_COMPRESSION), 3]
    return []


def imwrite(path: str | Path, image: np.ndarray, params: list[int] | None = None) -> bool:
    """Write ``image`` to ``path``. Return True on success."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return False

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if not image.flags["C_CONTIGUOUS"]:
        image = np.ascontiguousarray(image)

    suffix = out.suffix.lower() or ".png"
    encode_params = _encode_params_for_suffix(suffix, params)
    ok, buf = cv.imencode(suffix, image, encode_params)
    if not ok:
        return False

    out.write_bytes(buf.tobytes())
    return out.is_file() and out.stat().st_size > 0
