#!/usr/bin/env python3
# Vendored HSG-SRL artifact copy. Isolated from the development repo;
# edit this file only. See src/SOURCE_MAP.csv.
"""
Multi-camera bird's-eye-view stitching from ground-plane ChArUco boards
(supports an arbitrary number of cameras).

Pipeline
────────
1. Load each camera image (optional undistortion).
2. Place one ChArUco board in each adjacent-camera overlap to form a chain:
     cam0 ─[board01]─ cam1 ─[board12]─ cam2 ─[board23]─ cam3 …
   - The first board is tagged world_reference: true. Its outer top-left corner
     becomes the world origin, board width is the X axis, and board height is
     the Y axis. No measured coordinates are required.
   - Remaining boards are tagged auto_locate: true. A BFS chain estimates their
     world coordinates: find a camera that sees both the new board and an
     already located board, estimate a homography from known corners, then
     project the new board's pixel coordinates into world space. No numeric
     surveying is required.
3. For each camera, estimate an image-to-BEV perspective homography
   (cv.findHomography) from visible ChArUco inner corners
   (image coordinates ↔ world coordinates).
4. Warp every camera image onto a shared world-frame BEV canvas.
   - image_roi limits the valid region (full frame if omitted). Ordinary
     prime/zoom lenses (FOV ≤ 90°) can usually omit it; set it for wide-angle
     lenses or when edge distortion is obvious.
5. Luminance gain compensation (compute_luminance_gains):
   - Use the first camera as the reference and estimate per-channel median
     luminance ratios in adjacent-camera BEV overlaps.
   - Clamp the adjustment with tune_gain() to avoid over-compensation.
   - Gains are persisted to SQLite with the homographies and loaded under
     --use-cal.
6. Seam-weight blending (build_blend_weights):
   - For each camera, compute its exclusive region (BEV pixels not covered by
     any other camera).
   - Vectorize per-pixel distance to each exclusive-region boundary with
     distanceTransform.
   - Smooth only adjacent pairs (cam[i] ↔ cam[i+1]) in a band of width
     2 × seam_blend_px around the overlap midline; non-overlap regions are
     copied.
   - This reduces ghosting of 3D objects (vehicles) without amplifying seam
     color mismatch from a wide feather.

Calibration persistence (SQLite)
────────────────────────────────
    # First calibration (boards must be in view); save homographies and gains
    py pipeline/multi_camera_bev_stitch.py --config config/my_scene.yaml --save-cal --db cals.db

    # Later stitch any new images (boards not required)
    py pipeline/multi_camera_bev_stitch.py --config config/my_scene.yaml --use-cal --db cals.db

    # List saved calibration records
    py pipeline/multi_camera_bev_stitch.py --list-cal --db cals.db

Calibration-point detection priority (per camera)
─────────────────────────────────────────────────
  ① YAML point_pairs (≥4 pairs used directly; useful for debug / fine-tuning)
  ② manual_charuco_corners (YAML hand-labeled corners when the image is too
     blurry for automatic detection)
  ③ Automatic ChArUco board detection (recommended: tens of sub-pixel corners)
  ④ Single ArUco marker detection (legacy fallback; 4 corners per marker)

Install
───────
    pip install -r requirements.txt
    # Requires opencv-contrib-python, not plain opencv-python

Real cameras (no surveying)
───────────────────────────
    # Copy and fill config/real_images_template.yaml with board square specs
    # and image paths
    py pipeline/multi_camera_bev_stitch.py --config config/my_scene.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2 as cv
import numpy as np
import yaml

from utils.project_paths import resolve_project_path
from utils.image_io import imwrite

# fab1: calibration schema version bound to the metric-homography-Jacobian
# sensitivity plus metric-box feature geometry contract. Train/eval loaders use
# it as a breaking version gate and reject older (cla1) calibration DBs that
# lack or mismatch this version, so pixel-distance-era calibrations cannot be
# treated as a fab1 P_geo base.
GEOMETRY_VERSION = "fab1_metric_jacobian_v1"
CALIBRATION_DIAGNOSTICS_DIR = "outputs/calibration_diagnostics"


ARUCO_DICTS = {
    name: getattr(cv.aruco, name)
    for name in dir(cv.aruco)
    if name.startswith("DICT_")
}


def load_camera_matrix(path: str | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load camera intrinsics from npz/yml/yaml; return None if not configured."""
    if not path:
        return None, None

    p = resolve_project_path(path)
    if p.suffix.lower() == ".npz":
        data = np.load(str(p))
        camera_matrix = data["camera_matrix"] if "camera_matrix" in data else data["mtx"]
        dist_coeffs = data["dist_coeffs"] if "dist_coeffs" in data else data["dist"]
        return camera_matrix.astype(np.float64), dist_coeffs.astype(np.float64)

    fs = cv.FileStorage(str(p), cv.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calibration file: {path}")
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    if dist_coeffs is None:
        dist_coeffs = fs.getNode("dist_coefficients").mat()
    if dist_coeffs is None:
        dist_coeffs = fs.getNode("dist").mat()
    fs.release()
    if camera_matrix is None or dist_coeffs is None:
        raise ValueError(f"{path} must contain camera_matrix and dist_coeffs")
    return camera_matrix.astype(np.float64), dist_coeffs.astype(np.float64)


def marker_world_corners(marker_cfg: dict[str, Any]) -> np.ndarray:
    """Return board world corners: top-left, top-right, bottom-right, bottom-left, in meters."""
    if "corners" in marker_cfg:
        return np.asarray(marker_cfg["corners"], dtype=np.float32)

    center_x, center_y = marker_cfg["center"]
    size = float(marker_cfg["size"])
    yaw = math.radians(float(marker_cfg.get("yaw_deg", 0.0)))
    half = size / 2.0
    local = np.asarray(
        [[-half, half], [half, half], [half, -half], [-half, -half]],
        dtype=np.float32,
    )
    rot = np.asarray(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=np.float32,
    )
    return local @ rot.T + np.asarray([center_x, center_y], dtype=np.float32)


def charuco_inner_corner_count(squares_x: int, squares_y: int) -> int:
    """ChArUco inner-corner count = (squares_x - 1) * (squares_y - 1)."""
    return max(0, (int(squares_x) - 1) * (int(squares_y) - 1))


def load_manual_charuco_corners(
    cam_cfg: dict[str, Any],
) -> dict[str, dict[int, list[float]]] | None:
    """Read the camera YAML ``manual_charuco_corners`` field.

    Expected format:
      manual_charuco_corners:
        AB_overlap:
          - [corner_id, image_x, image_y]

    Returns {board_name: {corner_id: [x, y]}}; None if unset.
    When a board has hand-labeled corners, stitch / auto_locate uses them and
    skips ArUco detection.
    """
    raw = cam_cfg.get("manual_charuco_corners")
    if not raw:
        return None
    result: dict[str, dict[int, list[float]]] = {}
    for board_name, entries in raw.items():
        if entries:
            result[str(board_name)] = {
                int(e[0]): [float(e[1]), float(e[2])] for e in entries
            }
    return result or None


def validate_charuco_board_cfg(name: str, board_cfg: dict[str, Any]) -> None:
    """Validate board specs at startup; warn if corners are too few or markers too large."""
    sx = int(board_cfg["squares_x"])
    sy = int(board_cfg["squares_y"])
    sq_len = float(board_cfg["square_length"])
    mk_len = float(board_cfg["marker_length"])
    corners = charuco_inner_corner_count(sx, sy)
    if corners < 4:
        raise ValueError(
            f"Board '{name}' size {sx}×{sy} provides only {corners} ChArUco inner corners; "
            f"homography estimation needs at least 4. Increase the square count, e.g. 4×5 (12 corners) or 6×8 (35 corners)."
        )
    border = (sq_len - mk_len) * 0.5
    min_border = mk_len * 0.15  # OpenCV-suggested lower bound: marker white border ≳ 70% of pin size
    if border < min_border:
        suggested = round(sq_len * 0.75, 4)
        print(
            f"  [warn] board '{name}': marker white border {border:.4f} m is too small (square={sq_len}, marker={mk_len}). "
            f"Suggested marker_length ≈ {suggested} m (about square_length × 0.75), otherwise detection is unstable."
        )


def validate_charuco_board_topology(boards_cfg: dict[str, dict[str, Any]]) -> None:
    """Validate multi-board world-coordinate setup before homography fitting."""
    if not boards_cfg:
        return

    world_refs = [
        name for name, cfg in boards_cfg.items()
        if bool(cfg.get("world_reference", False))
    ]
    auto_located = [
        name for name, cfg in boards_cfg.items()
        if bool(cfg.get("auto_locate", False))
    ]

    if len(world_refs) > 1:
        joined = ", ".join(world_refs)
        raise ValueError(
            "ChArUco config error: only one board may set world_reference: true.\n"
            f"  Current world_reference boards: {joined}\n"
            "Every world_reference board is interpreted as the same world origin and "
            "orientation. Multiple physical boards configured this way create conflicting "
            "point pairs, so RANSAC may silently keep only one board.\n"
            "Keep exactly one world_reference board. Configure the other boards with "
            "auto_locate: true, or set world_origin / world_yaw_deg for their measured "
            "world poses."
        )

    if auto_located and not world_refs:
        joined = ", ".join(auto_located)
        raise ValueError(
            "ChArUco config error: auto_locate boards require one world_reference board.\n"
            f"  auto_locate boards: {joined}\n"
            "Set one board as world_reference: true to anchor the world coordinate system."
        )

    for name, cfg in boards_cfg.items():
        modes = [
            "world_reference" if cfg.get("world_reference", False) else None,
            "auto_locate" if cfg.get("auto_locate", False) else None,
            "manual_world" if "world_origin" in cfg else None,
        ]
        active_modes = [m for m in modes if m is not None]
        if len(active_modes) > 1:
            raise ValueError(
                f"ChArUco config error: board '{name}' has multiple pose modes: "
                f"{', '.join(active_modes)}.\n"
                "Use exactly one of: world_reference, auto_locate, or "
                "world_origin/world_yaw_deg."
            )
        if not active_modes:
            raise ValueError(
                f"ChArUco config error: board '{name}' has no world pose mode.\n"
                "Set world_reference: true, auto_locate: true, or "
                "world_origin / world_yaw_deg."
            )


def world_to_bev_points(world_xy: np.ndarray, scale_px_per_meter: float) -> np.ndarray:
    """Convert ground-plane world coordinates (meters) to intermediate BEV pixels
    (without the final canvas translation).

    Convention:
        bev_x = world_x * scale   (X axis agrees)
        bev_y = world_y * scale   (Y axis agrees; row-0 of the world_reference
                                    board faces the far side of the scene,
                                    world_y grows toward the cameras, and the
                                    bottom of the BEV image is the camera side)
    """
    xy = np.asarray(world_xy, dtype=np.float32).copy()
    xy[:, 0] *= scale_px_per_meter
    xy[:, 1] *= scale_px_per_meter
    return xy


def load_manual_point_pairs(
    cam_cfg: dict[str, Any],
    scale_px_per_meter: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Read configured manual point pairs; return image points and BEV pixels."""
    image_pts = []
    world_pts = []
    for pair in cam_cfg.get("point_pairs", []):
        image_pts.append(pair["image"])
        world_pts.append(pair["world"])
    if not image_pts:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    return (
        np.asarray(image_pts, dtype=np.float32),
        world_to_bev_points(np.asarray(world_pts, dtype=np.float32), scale_px_per_meter),
    )


def detect_aruco_points(
    image: np.ndarray,
    marker_world_by_id: dict[int, np.ndarray],
    dictionary_name: str,
    scale_px_per_meter: float,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Detect single ArUco markers; return image points, BEV pixels, and used IDs.

    Each marker contributes 4 corners. Only markers listed in marker_world_by_id
    are used. This is the priority-③ legacy fallback; ChArUco is preferred
    (higher accuracy, more corners).
    """
    if dictionary_name not in ARUCO_DICTS:
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")

    dictionary = cv.aruco.getPredefinedDictionary(ARUCO_DICTS[dictionary_name])
    params = cv.aruco.DetectorParameters()
    if hasattr(cv.aruco, "ArucoDetector"):
        detector = cv.aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(image)
    else:
        corners, ids, _ = cv.aruco.detectMarkers(image, dictionary, parameters=params)

    if ids is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), []

    image_pts = []
    bev_pts = []
    used_ids: list[int] = []
    for marker_corners, marker_id_arr in zip(corners, ids.flatten()):
        marker_id = int(marker_id_arr)
        if marker_id not in marker_world_by_id:
            continue
        image_pts.extend(marker_corners.reshape(4, 2))
        bev_pts.extend(world_to_bev_points(marker_world_by_id[marker_id], scale_px_per_meter))
        used_ids.append(marker_id)

    return (
        np.asarray(image_pts, dtype=np.float32),
        np.asarray(bev_pts, dtype=np.float32),
        used_ids,
    )


def _create_charuco_board(board_cfg: dict[str, Any]) -> Any:
    """Create a CharucoBoard from config; first_marker_id can match printed ArUco IDs."""
    squares_x = int(board_cfg["squares_x"])
    squares_y = int(board_cfg["squares_y"])
    sq_len = float(board_cfg["square_length"])
    mk_len = float(board_cfg["marker_length"])
    dict_name = board_cfg.get("aruco_dictionary", "DICT_4X4_50")
    if dict_name not in ARUCO_DICTS:
        raise ValueError(f"Unknown ArUco dictionary: {dict_name}")

    dictionary = cv.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
    marker_ids: np.ndarray | None = None
    if "marker_ids" in board_cfg:
        marker_ids = np.asarray(board_cfg["marker_ids"], dtype=np.int32).reshape(-1, 1)
    elif "first_marker_id" in board_cfg:
        first_id = int(board_cfg["first_marker_id"])
        default_board = cv.aruco.CharucoBoard((squares_x, squares_y), sq_len, mk_len, dictionary)
        count = int(default_board.getIds().size)  # type: ignore[union-attr]
        marker_ids = np.arange(first_id, first_id + count, dtype=np.int32).reshape(-1, 1)

    try:
        if marker_ids is not None:
            board = cv.aruco.CharucoBoard(
                (squares_x, squares_y), sq_len, mk_len, dictionary, marker_ids
            )
        else:
            board = cv.aruco.CharucoBoard((squares_x, squares_y), sq_len, mk_len, dictionary)
    except TypeError:
        if marker_ids is not None:
            raise RuntimeError(
                "This OpenCV build is too old for custom marker_ids; upgrade opencv-contrib-python"
            ) from None
        board = cv.aruco.CharucoBoard_create(  # type: ignore[attr-defined]
            squares_x, squares_y, sq_len, mk_len, dictionary
        )
    return board


def charuco_marker_id_grid(board_cfg: dict[str, Any]) -> np.ndarray:
    """Return the ChArUco marker-ID grid, shape=(squares_y, squares_x); empty squares are -1.

    OpenCV CharucoBoard places a marker where (row+col)%2==1 and IDs increase
    row-major (consistent with first_marker_id / marker_ids).
    """
    sx = int(board_cfg["squares_x"])
    sy = int(board_cfg["squares_y"])
    board = _create_charuco_board(board_cfg)
    ids = board.getIds().flatten()
    grid = np.full((sy, sx), -1, dtype=np.int32)
    idx = 0
    for row in range(sy):
        for col in range(sx):
            if (row + col) % 2 == 1:
                grid[row, col] = int(ids[idx])
                idx += 1
    return grid


def _make_aruco_detector_params() -> cv.aruco.DetectorParameters:
    """Return ArUco detector parameters tuned for low-quality / low-light images.

    Changes versus defaults:
      · Wider adaptive-threshold window (3–53) for marker-size variation and
        local illumination changes;
      · Smaller min marker perimeter (minMarkerPerimeterRate 0.01) for distant
        small markers;
      · Sub-pixel corner refinement (CORNER_REFINE_SUBPIX) for better accuracy;
      · Slightly looser error correction (0.6) to reduce misses from print or
        image deviation.
    """
    p = cv.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 53
    p.adaptiveThreshWinSizeStep = 4
    p.adaptiveThreshConstant = 7
    p.minMarkerPerimeterRate = 0.01
    p.maxMarkerPerimeterRate = 4.0
    p.polygonalApproxAccuracyRate = 0.03
    p.errorCorrectionRate = 0.6
    p.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 5
    p.cornerRefinementMaxIterations = 30
    return p


def _detection_variants(image: np.ndarray) -> list[tuple[str, np.ndarray, float]]:
    """Build preprocessing variants for multi-pass detection.

    Each list item is (variant_name, processed_image, coord_scale).
    coord_scale > 1 means the image was upsampled: detected pixels must be
    divided by this factor to recover original-image coordinates. Used only
    inside _detect_charuco_corners_raw / detect_aruco_enhanced; callers always
    receive original-image coordinates.
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image

    def to_bgr(g: np.ndarray) -> np.ndarray:
        return cv.cvtColor(g, cv.COLOR_GRAY2BGR) if image.ndim == 3 else g

    variants: list[tuple[str, np.ndarray, float]] = [
        ("original", image, 1.0),
    ]

    # ① CLAHE — adaptive contrast; usually best for uneven lighting
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g_clahe = clahe.apply(gray)
    variants.append(("CLAHE", to_bgr(g_clahe), 1.0))

    # ② CLAHE + mild sharpen — stronger edges, helps slight blur
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    g_sharp = np.clip(
        cv.filter2D(g_clahe.astype(np.float32), -1, kernel), 0, 255
    ).astype(np.uint8)
    variants.append(("CLAHE+sharpen", to_bgr(g_sharp), 1.0))

    # ③ denoise + CLAHE — Gaussian blur then contrast; for noisy images
    g_blur_clahe = clahe.apply(cv.GaussianBlur(gray, (3, 3), 0))
    variants.append(("denoise+CLAHE", to_bgr(g_blur_clahe), 1.0))

    # ④ 2× upsample + CLAHE — more marker pixels so distant ArUco can be found
    #    After detection, divide corner coordinates by 2 to recover original pixels
    up_scale = 2.0
    h, w = gray.shape[:2]
    g_up = cv.resize(gray, (int(w * up_scale), int(h * up_scale)), interpolation=cv.INTER_CUBIC)
    g_up_clahe = clahe.apply(g_up)
    variants.append(("2x-upsample+CLAHE", to_bgr(g_up_clahe), up_scale))

    return variants


def _detect_charuco_corners_raw(
    image: np.ndarray,
    board_cfg: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Detect inner corners of one ChArUco board; return (corners [N,2], ids [N]).

    Tries preprocessing variants in order (original → CLAHE → CLAHE+sharpen →
    denoise+CLAHE → 2x-upsample+CLAHE) and keeps the result with the most inner
    corners. Returned coordinates are always in original-image pixels.
    """
    board = _create_charuco_board(board_cfg)
    total_expected = charuco_inner_corner_count(
        int(board_cfg["squares_x"]), int(board_cfg["squares_y"])
    )
    det_params = _make_aruco_detector_params()

    best_corners: np.ndarray | None = None
    best_ids: np.ndarray | None = None
    best_n = 0

    for _variant_name, variant_img, coord_scale in _detection_variants(image):
        corners: np.ndarray | None = None
        ids: np.ndarray | None = None

        if hasattr(cv.aruco, "CharucoDetector"):
            # OpenCV ≥ 4.7 new API
            charuco_params = cv.aruco.CharucoParameters()
            charuco_params.minMarkers = 1          # allow interpolating from a single ArUco marker
            detector = cv.aruco.CharucoDetector(board, charuco_params, det_params)
            corners, ids, _, _ = detector.detectBoard(variant_img)
        else:
            # Legacy API compatibility path
            gray_v = (
                cv.cvtColor(variant_img, cv.COLOR_BGR2GRAY)
                if variant_img.ndim == 3
                else variant_img
            )
            dict_name = board_cfg.get("aruco_dictionary", "DICT_4X4_50")
            dictionary = cv.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
            mc, mi, _ = cv.aruco.detectMarkers(  # type: ignore[attr-defined]
                gray_v, dictionary, parameters=det_params
            )
            if mi is not None and len(mi) > 0:
                _, corners, ids = cv.aruco.interpolateCornersCharuco(  # type: ignore[attr-defined]
                    mc, mi, gray_v, board
                )

        if corners is None or ids is None or len(ids) == 0:
            continue

        n = len(ids)
        # Map upsampled coordinates back to original-image pixels
        if coord_scale != 1.0:
            corners = corners / coord_scale

        if n > best_n:
            best_n = n
            best_corners = corners
            best_ids = ids

        if best_n >= total_expected:
            break   # all inner corners found; no need to try more variants

    if best_corners is None or best_ids is None:
        return None, None
    return best_corners.reshape(-1, 2), best_ids.flatten()


def charuco_corner_local_xy(idx: int, squares_x: int, square_length: float) -> tuple[float, float]:
    """ChArUco inner-corner position in board-local meters; origin at the outer top-left."""
    cols_per_row = int(squares_x) - 1
    row = int(idx) // cols_per_row
    col = int(idx) % cols_per_row
    return (col + 1) * square_length, (row + 1) * square_length


def extrapolate_charuco_board_world_corners(
    board_cfg: dict[str, Any],
    partial_corner_world: dict[int, list[float]],
) -> dict[int, list[float]]:
    """Complete all inner-corner world coordinates from a partial set via 2D affine.

    auto_locate on a bridge camera often sees only part of the board. Saving
    only those corners yields zero matches if another camera detects different
    IDs. After completion, any camera that sees ≥4 corners on the board can
    calibrate.
    """
    sx = int(board_cfg["squares_x"])
    sy = int(board_cfg["squares_y"])
    sq_len = float(board_cfg["square_length"])
    total = charuco_inner_corner_count(sx, sy)
    if len(partial_corner_world) >= total:
        return dict(partial_corner_world)
    if len(partial_corner_world) < 3:
        return dict(partial_corner_world)

    src_pts: list[list[float]] = []
    dst_pts: list[list[float]] = []
    for idx, wxy in partial_corner_world.items():
        lx, ly = charuco_corner_local_xy(int(idx), sx, sq_len)
        src_pts.append([lx, ly])
        dst_pts.append(wxy)

    src = np.asarray(src_pts, dtype=np.float32)
    dst = np.asarray(dst_pts, dtype=np.float32)
    if len(src) == 3:
        affine = cv.getAffineTransform(src, dst)
    else:
        affine, _ = cv.estimateAffine2D(src, dst)
        if affine is None:
            return dict(partial_corner_world)

    full: dict[int, list[float]] = {}
    for idx in range(total):
        lx, ly = charuco_corner_local_xy(idx, sx, sq_len)
        wx, wy = affine @ np.array([lx, ly, 1.0], dtype=np.float64)
        full[idx] = [float(wx), float(wy)]
    return full


def describe_charuco_detection_gaps(
    image: np.ndarray,
    boards_cfg: dict[str, dict[str, Any]],
    camera_board_names: list[str],
    board_corner_world_override: dict[str, dict[int, list[float]]] | None,
) -> list[str]:
    """Summarize per-board detection/match counts for calibration-failure diagnostics."""
    lines: list[str] = []
    for board_name in camera_board_names:
        if board_name not in boards_cfg:
            continue
        cfg = boards_cfg[board_name]
        corners, ids = _detect_charuco_corners_raw(image, cfg)
        detected = 0 if ids is None else len(ids)
        need = charuco_inner_corner_count(int(cfg["squares_x"]), int(cfg["squares_y"]))
        matched = detected
        if cfg.get("auto_locate", False) and board_corner_world_override:
            pre = board_corner_world_override.get(board_name, {})
            if ids is not None and pre:
                matched = sum(1 for i in ids if int(i) in pre)
        id_sample = ""
        if ids is not None and len(ids) > 0:
            flat = [int(x) for x in ids.flatten()[:6]]
            id_sample = f" id={flat}{'...' if len(ids) > 6 else ''}"
        lines.append(
            f"board '{board_name}': detected {detected}/{need} corners, usable matches {matched}{id_sample}"
        )
    return lines


def detect_charuco_points(
    image: np.ndarray,
    boards_cfg: dict[str, dict[str, Any]],
    camera_board_names: list[str],
    scale_px_per_meter: float,
    board_corner_world_override: dict[str, dict[int, list[float]]] | None = None,
    manual_charuco_corners: dict[str, dict[int, list[float]]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect visible ChArUco inner corners; return image coordinates and BEV pixels.

    Three board modes (selected by YAML fields):

    ① world_reference: true
        This board defines the world frame: outer top-left = origin (0,0),
        board-width direction = X. Do not set world_origin / world_yaw_deg.
        Place the board; no surveying is required.

    ② auto_locate: true
        Corner world coordinates are inferred by a bridge camera that sees both
        a known board and this board. Also measurement-free. Call
        auto_locate_charuco_boards() first.

    ③ Manual pose (default, backward compatible)
        Set world_origin: [x, y] and world_yaw_deg. Use when a reference board
        is unavailable or a fixed absolute world frame is required.

    manual_charuco_corners takes precedence over automatic detection: if a board
    has hand-labeled corners in YAML, ArUco detection is skipped.
    """
    all_image_pts: list[np.ndarray] = []
    all_world_pts: list[list[float]] = []

    for board_name in camera_board_names:
        if board_name not in boards_cfg:
            continue
        cfg = boards_cfg[board_name]

        # Prefer hand-labeled corners when ArUco cannot detect a blurry image
        if manual_charuco_corners and board_name in manual_charuco_corners:
            manual = manual_charuco_corners[board_name]
            corners = np.array([[v[0], v[1]] for v in manual.values()], dtype=np.float32)
            ids = np.array(list(manual.keys()), dtype=np.int32)
            print(f"  [manual] board '{board_name}' using hand-labeled corners ({len(corners)})")
        else:
            corners, ids = _detect_charuco_corners_raw(image, cfg)
        if corners is None or ids is None:
            continue

        sq_len = float(cfg["square_length"])

        if cfg.get("world_reference", False):
            # Reference board: outer top-left = world origin (0,0); width = +X, height = +Y
            for corner, idx in zip(corners, ids):
                wx, wy = charuco_corner_local_xy(int(idx), int(cfg["squares_x"]), sq_len)
                all_image_pts.append(corner)
                all_world_pts.append([wx, wy])

        elif cfg.get("auto_locate", False):
            # auto_locate board: use world corners precomputed by auto_locate_charuco_boards()
            if board_corner_world_override is None or board_name not in board_corner_world_override:
                continue
            precomputed = board_corner_world_override[board_name]
            for corner, idx in zip(corners, ids):
                if int(idx) in precomputed:
                    all_image_pts.append(corner)
                    all_world_pts.append(precomputed[int(idx)])

        else:
            # Manual pose: world corners from world_origin + world_yaw_deg
            ox = float(cfg["world_origin"][0])
            oy = float(cfg["world_origin"][1])
            yaw = math.radians(float(cfg.get("world_yaw_deg", 0.0)))
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            for corner, idx in zip(corners, ids):
                lx, ly = charuco_corner_local_xy(int(idx), int(cfg["squares_x"]), sq_len)
                wx = ox + lx * cos_y - ly * sin_y
                wy = oy + lx * sin_y + ly * cos_y
                all_image_pts.append(corner)
                all_world_pts.append([wx, wy])

    if not all_image_pts:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    image_arr = np.asarray(all_image_pts, dtype=np.float32)
    world_arr = np.asarray(all_world_pts, dtype=np.float32)
    return image_arr, world_to_bev_points(world_arr, scale_px_per_meter)


def auto_locate_charuco_boards(
    all_images: dict[str, np.ndarray],
    boards_cfg: dict[str, dict[str, Any]],
    cam_board_names_map: dict[str, list[str]],
    scale_px_per_meter: float,
    ransac_reproj_threshold: float,
    manual_corners_by_cam: dict[str, dict[str, dict[int, list[float]]]] | None = None,
) -> dict[str, dict[int, list[float]]]:
    """BFS-chain world coordinates for boards with auto_locate=true.

    Supports an arbitrary N-camera chain, e.g.:
      cam0 ─[board01]─ cam1 ─[board12]─ cam2 ─[board23]─ cam3 …

    Algorithm (BFS):
      1. Treat every world_reference=true board as a known anchor.
      2. Iterate. Each round, for every remaining auto_locate board:
           find a camera that sees that board and at least one known board;
           estimate that camera's homography H from known corners;
           project the unknown board's pixels through H into world coordinates.
      3. At least one new board must be located each round (else raise: topology
         is disconnected).
      4. Return when every board is located.

    Returns: {board_name: {corner_id: [wx_meters, wy_meters]}}
    """
    ref_names: set[str] = {
        name for name, cfg in boards_cfg.items() if cfg.get("world_reference", False)
    }
    auto_names = [name for name, cfg in boards_cfg.items() if cfg.get("auto_locate", False)]

    if not ref_names or not auto_names:
        return {}

    # known_overrides: world corners of already located auto_locate boards
    known_overrides: dict[str, dict[int, list[float]]] = {}
    remaining = list(auto_names)

    for _iteration in range(len(auto_names) + 1):
        if not remaining:
            break
        progress = False
        failure_notes: list[str] = []

        for auto_name in remaining[:]:
            # Find a camera that sees auto_name and at least one known board
            for cam_name, cam_boards in cam_board_names_map.items():
                if auto_name not in cam_boards:
                    continue
                known_visible = [
                    b for b in cam_boards
                    if b in ref_names or b in known_overrides
                ]
                if not known_visible:
                    continue  # this camera sees no known board; try the next

                bridge_img = all_images[cam_name]
                bridge_manual = (
                    manual_corners_by_cam.get(cam_name)
                    if manual_corners_by_cam else None
                )

                # Estimate the bridge-camera homography from known-board corners (manual labels first)
                ref_img_pts, ref_bev_pts = detect_charuco_points(
                    bridge_img,
                    {b: boards_cfg[b] for b in known_visible},
                    known_visible,
                    scale_px_per_meter,
                    board_corner_world_override=known_overrides,
                    manual_charuco_corners=bridge_manual,
                )
                if len(ref_img_pts) < 4:
                    per_board: list[str] = []
                    for bname in known_visible:
                        corners, ids = _detect_charuco_corners_raw(
                            bridge_img, boards_cfg[bname]
                        )
                        n = 0 if ids is None else len(ids)
                        need = charuco_inner_corner_count(
                            boards_cfg[bname]["squares_x"],
                            boards_cfg[bname]["squares_y"],
                        )
                        per_board.append(f"{bname} detected {n}/{need} corners")
                    failure_notes.append(
                        f"Camera '{cam_name}' as bridge: known-board corners < 4 "
                        f"(got {len(ref_img_pts)}): {'; '.join(per_board)}"
                    )
                    continue  # not enough corners; try another camera

                try:
                    H_bridge, _ = estimate_image_to_bev_homography(
                        ref_img_pts, ref_bev_pts, ransac_reproj_threshold
                    )
                except RuntimeError:
                    failure_notes.append(
                        f"Camera '{cam_name}' failed to estimate homography from bridge boards {known_visible}"
                    )
                    continue

                # Detect the unknown board in the bridge camera (manual labels first)
                if bridge_manual and auto_name in bridge_manual:
                    manual_data = bridge_manual[auto_name]
                    corners = np.array(
                        [[v[0], v[1]] for v in manual_data.values()], dtype=np.float32
                    )
                    ids = np.array(list(manual_data.keys()), dtype=np.int32)
                    print(
                        f"  [manual] auto_locate board '{auto_name}' in bridge camera "
                        f"'{cam_name}' using hand-labeled corners ({len(corners)})"
                    )
                else:
                    corners, ids = _detect_charuco_corners_raw(bridge_img, boards_cfg[auto_name])
                if corners is None or ids is None:
                    failure_notes.append(
                        f"Camera '{cam_name}' did not detect ChArUco corners of board '{auto_name}'"
                    )
                    continue

                # pixel coords → BEV pixels → world meters
                corner_world: dict[int, list[float]] = {}
                for corner, idx in zip(corners, ids):
                    pt_h = np.array([corner[0], corner[1], 1.0], dtype=np.float64)
                    bev_h = H_bridge @ pt_h
                    bev_x = bev_h[0] / bev_h[2]
                    bev_y = bev_h[1] / bev_h[2]
                    corner_world[int(idx)] = [
                        bev_x / scale_px_per_meter,
                        bev_y / scale_px_per_meter,
                    ]

                known_overrides[auto_name] = extrapolate_charuco_board_world_corners(
                    boards_cfg[auto_name],
                    corner_world,
                )
                remaining.remove(auto_name)
                progress = True
                n_partial = len(corner_world)
                n_full = len(known_overrides[auto_name])
                print(
                    f"[auto_locate] board '{auto_name}' located via camera '{cam_name}' "
                    f"(known boards: {known_visible}); bridge detected {n_partial} corners, "
                    f"completed to {n_full} corner world coordinates."
                )
                break  # found a bridge camera; move to the next board

        if not progress and remaining:
            # Boards with ruler_measurements may fail auto_locate (ruler locate later)
            boards_needing_ruler = [b for b in remaining if boards_cfg[b].get("ruler_measurements")]
            boards_must_fix = [b for b in remaining if b not in boards_needing_ruler]
            if boards_needing_ruler:
                print(
                    f"  [auto_locate] the following boards will use ruler_measurements "
                    f"(auto_locate could not bridge): {boards_needing_ruler}"
                )
                for b in boards_needing_ruler:
                    remaining.remove(b)
                if not remaining:
                    break
            if boards_must_fix:
                detail = "\n  ".join(dict.fromkeys(failure_notes)) if failure_notes else "no usable bridge camera"
                board_hints = []
                for bname in boards_must_fix:
                    if bname in boards_cfg:
                        bc = boards_cfg[bname]
                        sx, sy = int(bc["squares_x"]), int(bc["squares_y"])
                        nc = charuco_inner_corner_count(sx, sy)
                        board_hints.append(f"{bname} is {sx}×{sy} squares ({nc} inner corners)")
                hint_boards = "; ".join(board_hints) if board_hints else "see YAML charuco_boards"
                raise RuntimeError(
                    f"Cannot auto-locate these calibration boards: {boards_must_fix}.\n"
                    f"Please confirm:\n"
                    f"  1) each board has ≥ 4 ChArUco inner corners ({hint_boards});\n"
                    f"  2) a bridge camera sees both a known board and the unknown board, "
                    f"and both boards are fully visible and sharp;\n"
                    f"  3) site-YAML ChArUco board parameters "
                    f"(square_length / marker_length / first_marker_id) match the printed boards;\n"
                    f"  4) inspect per-camera detected corner counts and ArUco IDs;\n"
                    f"  5) or add a ruler_measurements field (trilateration as a backup).\n"
                    f"Diagnostics:\n  {detail}"
                )

    return known_overrides


# ──────────────────────────────────────────────────────────────────────────────
# Ruler assist: tape-measure corner distances, then trilateration + Procrustes
# ──────────────────────────────────────────────────────────────────────────────

# Corner-reference notes (YAML from_corner / to_corner)
# ──────────────────────────────────────────────────────────────────────────────
# Either form is accepted:
#
#  ① Outer corners (physical board corners; easiest for a tape):
#       "outer_tl"  outer top-left (world origin on the world_reference board)
#       "outer_tr"  outer top-right
#       "outer_bl"  outer bottom-left
#       "outer_br"  outer bottom-right
#
#  ② Inner-corner index (integer):
#       idx = row × (squares_x - 1) + col   (row/col from 0; top-left is (0,0))
#
#       Example for squares_x=4, squares_y=5 (3 cols × 4 rows = 12 inner corners):
#
#         outer_tl ┌───┬───┬───┬───┐ outer_tr
#                  │   0   1   2   │ ← row-0 inner corners
#                  │   3   4   5   │
#                  │   6   7   8   │
#                  │   9  10  11   │ ← last row
#         outer_bl └───┴───┴───┴───┘ outer_br
#
#       Common: top-left inner=0, top-right inner=squares_x-2,
#               bottom-left inner=(squares_y-2)*(squares_x-1)
#               bottom-right inner=(squares_y-1)*(squares_x-1)-1
#
# Prefer outer_tl/outer_tr/outer_bl/outer_br as anchors and targets because
# physical corners are easier to identify and measure.


_OUTER_CORNER_NAMES = {"outer_tl", "outer_tr", "outer_bl", "outer_br"}


def _resolve_corner_ref(
    corner_ref: int | str,
    board_cfg: dict[str, Any],
) -> tuple[float, float]:
    """Convert a corner ref (inner-corner index or outer-corner name) to board-local meters.

    Local origin = outer top-left (outer_tl); +X along board width; +Y along board height.
    """
    sq_len = float(board_cfg["square_length"])
    sx = int(board_cfg["squares_x"])
    sy = int(board_cfg["squares_y"])

    if isinstance(corner_ref, str):
        ref = corner_ref.lower()
        if ref == "outer_tl":
            return 0.0, 0.0
        if ref == "outer_tr":
            return sx * sq_len, 0.0
        if ref == "outer_bl":
            return 0.0, sy * sq_len
        if ref == "outer_br":
            return sx * sq_len, sy * sq_len
        raise ValueError(
            f"Unknown corner name '{corner_ref}'; supported: outer_tl / outer_tr / outer_bl / outer_br "
            f"or an integer inner-corner index (0 ~ {(sx-1)*(sy-1)-1})"
        )
    return charuco_corner_local_xy(int(corner_ref), sx, sq_len)


def _board_local_to_world(
    lx: float,
    ly: float,
    board_name: str,
    boards_cfg: dict[str, dict[str, Any]],
    known_overrides: dict[str, dict[int, list[float]]],
) -> np.ndarray | None:
    """Convert board-local (lx, ly) to world coordinates.

    Supports:
      · world_reference board: local coords are world coords (origin outer_tl, no rotation).
      · Located boards (auto_locate / ruler_measurements): fit pose from inner-corner overrides.
    """
    cfg = boards_cfg.get(board_name)
    if cfg is None:
        return None

    if cfg.get("world_reference", False):
        return np.array([lx, ly], dtype=np.float64)

    # Fit board pose (origin + yaw) from known inner corners
    overrides = known_overrides.get(board_name, {})
    if len(overrides) < 2:
        return None

    sq_len = float(cfg["square_length"])
    sq_x = int(cfg["squares_x"])
    ids = sorted(overrides.keys())[:8]          # at most 8 inner corners; enough for pose
    l_pts = np.array([charuco_corner_local_xy(i, sq_x, sq_len) for i in ids])
    w_pts = np.array([overrides[i] for i in ids])
    ox, oy, yaw_deg = _fit_board_pose_2d(l_pts, w_pts)

    yaw_rad = math.radians(yaw_deg)
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    wx = ox + lx * cos_y - ly * sin_y
    wy = oy + lx * sin_y + ly * cos_y
    return np.array([wx, wy], dtype=np.float64)


def _get_board_corner_world(
    board_name: str,
    corner_ref: int | str,
    boards_cfg: dict[str, dict[str, Any]],
    known_overrides: dict[str, dict[int, list[float]]],
) -> np.ndarray | None:
    """World coordinates (meters) of a named board corner.

    corner_ref may be:
      · int: inner-corner index (0 = top-left inner)
      · str: outer_tl / outer_tr / outer_bl / outer_br (physical outer corners)
    """
    cfg = boards_cfg.get(board_name)
    if cfg is None:
        return None
    try:
        lx, ly = _resolve_corner_ref(corner_ref, cfg)
    except (ValueError, TypeError) as e:
        print(f"  [ruler] failed to resolve corner ref ({board_name}/{corner_ref}): {e}")
        return None
    return _board_local_to_world(lx, ly, board_name, boards_cfg, known_overrides)


def _trilaterate_2d(
    anchors: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    """Linearized least-squares 2D target from N≥3 known anchors and distances.

    Algorithm
    ────
    Subtracting equation pairs (i, 0) cancels tx²+ty² and yields A·t = b,
    where A and b depend only on anchor coordinates and distances:
        2*(x_i - x_0)*tx + 2*(y_i - y_0)*ty
            = (x_i² - x_0²) + (y_i² - y_0²) - (d_i² - d_0²)
    Exact for N=3; np.linalg.lstsq least squares for N>3.

    Error: measurement noise σ_d (typically ≤ 5mm); estimate ~ σ_d / sqrt(N).
    """
    if len(anchors) < 3:
        raise ValueError(
            f"Trilateration needs at least 3 distinct anchors (got {len(anchors)}). "
            "Not enough ruler_measurements for this to_corner; add more."
        )
    A = 2.0 * (anchors[1:] - anchors[0])         # (N-1, 2)
    b = (
        np.sum(anchors[1:] ** 2, axis=1)
        - np.sum(anchors[0] ** 2)
        - (distances[1:] ** 2 - distances[0] ** 2)
    )                                              # (N-1,)
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return result  # (2,) = [tx, ty]


def _fit_board_pose_2d(
    local_pts: np.ndarray,  # (N, 2) inner-corner positions in board-local frame
    world_pts: np.ndarray,  # (N, 2) corresponding world coordinates (trilateration)
) -> tuple[float, float, float]:
    """Procrustes analysis: fit board pose from known local↔world pairs.

    Returns (world_origin_x, world_origin_y, world_yaw_deg).
    world_origin is the world position of the local origin (0,0) (outer top-left).
    """
    L_bar = local_pts.mean(axis=0)
    W_bar = world_pts.mean(axis=0)
    dL = local_pts - L_bar
    dW = world_pts - W_bar
    H = dL.T @ dW                                 # (2, 2) covariance
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:                      # remove reflection ambiguity
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    T = W_bar - R @ L_bar                         # world origin = local origin in world
    yaw_deg = math.degrees(math.atan2(float(R[1, 0]), float(R[0, 0])))
    return float(T[0]), float(T[1]), yaw_deg


def locate_boards_from_ruler_measurements(
    boards_cfg: dict[str, dict[str, Any]],
    accumulated_overrides: dict[str, dict[int, list[float]]],
) -> dict[str, dict[int, list[float]]]:
    """Infer corner world coordinates from ruler_measurements via trilateration.

    Each measurement: put one end of a steel tape on a known-board corner
    (from_board / from_corner) and the other on this board (to_corner); record
    the ground distance (distance_m, meters).

    Corner refs:
        · "outer_tl" / "outer_tr" / "outer_bl" / "outer_br" — physical outer
          corners (easiest to measure)
        · integers 0, 1, 2, … — inner-corner indices

    Minimum measurements:
        · each to_corner must come from ≥3 non-collinear from_corners
          (unique 3-circle intersection)
        · at least 2 distinct to_corners (to determine yaw)

    If a board has both auto_locate: true and ruler_measurements, ruler results
    override auto_locate (ruler is usually more accurate).
    """
    result: dict[str, dict[int, list[float]]] = {}

    for board_name, board_cfg in boards_cfg.items():
        measurements = board_cfg.get("ruler_measurements")
        if not measurements:
            continue

        # Merge all known corner coordinates (world_reference + already located boards)
        all_known = {**accumulated_overrides, **result}

        # Group anchors by to_corner (int/str dict keys)
        groups: dict[int | str, list[tuple[np.ndarray, float]]] = {}
        for m in measurements:
            from_board = str(m["from_board"])
            from_corner = m["from_corner"]   # int or "outer_tl", etc.
            to_corner = m["to_corner"]       # int or "outer_tl", etc.
            dist_m = float(m["distance_m"])

            # Normalize strings to lowercase
            if isinstance(from_corner, str):
                from_corner = from_corner.lower()
            if isinstance(to_corner, str):
                to_corner = to_corner.lower()

            anchor = _get_board_corner_world(from_board, from_corner, boards_cfg, all_known)
            if anchor is None:
                print(
                    f"  [ruler] warn: world coordinates of board '{from_board}' corner '{from_corner}' are unknown, "
                    f"skipping this measurement (confirm from_board is world_reference or already located)"
                )
                continue
            groups.setdefault(to_corner, []).append((anchor, dist_m))

        # Trilaterate each to_corner → world coordinates
        located_local: list[tuple[float, float]] = []
        located_world: list[np.ndarray] = []

        for to_corner, pairs in groups.items():
            anchors = np.array([p[0] for p in pairs])
            dists = np.array([p[1] for p in pairs])
            try:
                world_pos = _trilaterate_2d(anchors, dists)
                lx, ly = _resolve_corner_ref(to_corner, board_cfg)
                located_local.append((lx, ly))
                located_world.append(world_pos)
                print(
                    f"  [ruler] board '{board_name}' corner '{to_corner}' → "
                    f"world ({world_pos[0]:.4f}, {world_pos[1]:.4f}) m  "
                    f"({len(pairs)} measurements)"
                )
            except ValueError as e:
                print(f"  [ruler] warn: board '{board_name}' corner '{to_corner}' — {e}")

        if len(located_local) < 2:
            print(
                f"  [ruler] board '{board_name}': only located {len(located_local)} corner(s); "
                f"need at least 2 distinct to_corner values to determine yaw."
            )
            continue

        # Procrustes board-pose fit
        local_pts = np.array(located_local)
        world_pts = np.array(located_world)
        ox, oy, yaw_deg = _fit_board_pose_2d(local_pts, world_pts)
        print(
            f"  [ruler] board '{board_name}' pose fit: "
            f"world_origin=[{ox:.4f}, {oy:.4f}], world_yaw_deg={yaw_deg:.2f}°"
        )

        # Build partial_corner_world from located world coords and complete inner corners
        # Trilaterated corners (outer corners are not in overrides; pass inner corners to extrapolate)
        sq_len = float(board_cfg["square_length"])
        sq_x = int(board_cfg["squares_x"])
        yaw_rad = math.radians(yaw_deg)
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)

        # World coordinates of every inner corner from the fitted pose
        total = charuco_inner_corner_count(sq_x, int(board_cfg["squares_y"]))
        full_corners: dict[int, list[float]] = {}
        for idx in range(total):
            ilx, ily = charuco_corner_local_xy(idx, sq_x, sq_len)
            wx = ox + ilx * cos_y - ily * sin_y
            wy = oy + ilx * sin_y + ily * cos_y
            full_corners[idx] = [wx, wy]

        result[board_name] = full_corners
        print(f"  [ruler] board '{board_name}' generated all {total} inner-corner world coordinates.")

    return result


def estimate_image_to_bev_homography(
    image_pts: np.ndarray,
    bev_pts: np.ndarray,
    ransac_reproj_threshold: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Estimate a 3×3 image-to-BEV homography from point pairs.

    RANSAC rejects outliers when there are more than 4 pairs; exactly 4 uses
    the exact solution (no inlier mask). Returns (H_3x3, inliers_mask);
    inliers_mask is None for the exact 4-point case.
    """
    if len(image_pts) < 4:
        raise ValueError(f"Need at least 4 point pairs, got {len(image_pts)}")
    h, inliers = cv.findHomography(
        image_pts,
        bev_pts,
        method=cv.RANSAC if len(image_pts) > 4 else 0,
        ransacReprojThreshold=ransac_reproj_threshold,
    )
    if h is None:
        raise RuntimeError("findHomography failed")
    return h.astype(np.float64), inliers

# ──────────────────────────────────────────────────────────────────────────────
# Optional H refine/diagnostics: after BFS/RANSAC init, refine the same H with
# reprojection + horizon constraints
# ──────────────────────────────────────────────────────────────────────────────

def normalize_horizon_line(
    line: np.ndarray | list[float] | tuple[float, float, float],
    frame_wh: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return line ``ax+by+c=0`` with ``sqrt(a^2+b^2)=1`` and bottom side positive."""
    arr = np.asarray(line, dtype=np.float64).reshape(3)
    n = float(np.hypot(arr[0], arr[1]))
    if not np.isfinite(n) or n <= 1e-12:
        raise ValueError("horizon line must have a non-zero normal")
    arr = arr / n
    if frame_wh is not None:
        fw, fh = frame_wh
        probe = np.array([float(fw) * 0.5, max(float(fh) - 1.0, 0.0), 1.0])
        if float(arr @ probe) < 0.0:
            arr = -arr
    return arr


def horizon_line_from_points(
    p0: tuple[float, float] | list[float],
    p1: tuple[float, float] | list[float],
    frame_wh: tuple[int, int] | None = None,
) -> np.ndarray:
    """Build a normalized horizon line from two manually clicked image points."""
    a = np.array([float(p0[0]), float(p0[1]), 1.0], dtype=np.float64)
    b = np.array([float(p1[0]), float(p1[1]), 1.0], dtype=np.float64)
    if np.linalg.norm(a[:2] - b[:2]) < 2.0:
        raise ValueError("manual horizon points are too close")
    return normalize_horizon_line(np.cross(a, b), frame_wh)


def homography_horizon_line(
    homography: np.ndarray,
    frame_wh: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return the image horizon induced by an image->BEV homography."""
    h = np.asarray(homography, dtype=np.float64).reshape(3, 3)
    return normalize_horizon_line(h[2, :], frame_wh)


def horizon_line_to_segment(
    line: np.ndarray | list[float],
    frame_wh: tuple[int, int],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Clip an infinite image line to the image rectangle for drawing."""
    a, b, c = normalize_horizon_line(line, frame_wh)
    fw, fh = frame_wh
    candidates: list[tuple[float, float]] = []
    if abs(b) > 1e-12:
        for x in (0.0, float(fw - 1)):
            y = -(a * x + c) / b
            if -1.0 <= y <= float(fh):
                candidates.append((x, y))
    if abs(a) > 1e-12:
        for y in (0.0, float(fh - 1)):
            x = -(b * y + c) / a
            if -1.0 <= x <= float(fw):
                candidates.append((x, y))
    unique: list[tuple[float, float]] = []
    for p in candidates:
        if not any((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < 1.0 for q in unique):
            unique.append(p)
    if len(unique) < 2:
        return None
    best = (unique[0], unique[1])
    best_d2 = -1.0
    for i, p in enumerate(unique):
        for q in unique[i + 1:]:
            d2 = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
            if d2 > best_d2:
                best = (p, q)
                best_d2 = d2
    return best


def load_manual_horizon_line(
    cam_cfg: dict[str, Any],
    frame_wh: tuple[int, int],
) -> dict[str, Any] | None:
    """Load camera-level ``manual_horizon_line`` and normalize it to bottom-positive sign."""
    raw = cam_cfg.get("manual_horizon_line")
    if not isinstance(raw, dict):
        return None
    points_raw = raw.get("points")
    points: list[list[float]] | None = None
    if isinstance(points_raw, list) and len(points_raw) >= 2:
        points = [
            [float(points_raw[0][0]), float(points_raw[0][1])],
            [float(points_raw[1][0]), float(points_raw[1][1])],
        ]
        line = horizon_line_from_points(points[0], points[1], frame_wh)
    elif raw.get("line") is not None:
        line = normalize_horizon_line(raw["line"], frame_wh)
    else:
        return None
    return {
        "line": [float(v) for v in line.tolist()],
        "points": points,
        "ground_side_positive": True,
    }


def _project_homography_points(homography: np.ndarray, image_pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
    h = np.asarray(homography, dtype=np.float64).reshape(3, 3)
    src = np.c_[pts, np.ones(len(pts), dtype=np.float64)]
    dst = (h @ src.T).T
    denom = dst[:, 2:3]
    out = np.full((len(pts), 2), np.nan, dtype=np.float64)
    valid = np.abs(denom[:, 0]) > 1e-12
    out[valid] = dst[valid, :2] / denom[valid]
    return out


def homography_reprojection_rms(
    homography: np.ndarray,
    image_pts: np.ndarray,
    bev_pts: np.ndarray,
) -> float:
    proj = _project_homography_points(homography, image_pts)
    target = np.asarray(bev_pts, dtype=np.float64).reshape(-1, 2)
    valid = np.all(np.isfinite(proj), axis=1)
    if not np.any(valid):
        return float("inf")
    err2 = np.sum((proj[valid] - target[valid]) ** 2, axis=1)
    return float(np.sqrt(np.mean(err2)))


def _sample_line_points(line: np.ndarray, frame_wh: tuple[int, int]) -> np.ndarray:
    a, b, c = normalize_horizon_line(line, frame_wh)
    fw, fh = frame_wh
    pts: list[tuple[float, float]] = []
    if abs(b) >= abs(a) and abs(b) > 1e-12:
        for x in (0.0, float(fw - 1) * 0.5, float(fw - 1)):
            pts.append((x, -(a * x + c) / b))
    elif abs(a) > 1e-12:
        for y in (0.0, float(fh - 1) * 0.5, float(fh - 1)):
            pts.append((-(b * y + c) / a, y))
    else:
        pts = [(0.0, 0.0), (float(fw - 1), 0.0), (float(fw - 1) * 0.5, float(fh - 1) * 0.5)]
    return np.asarray(pts, dtype=np.float64)


def horizon_line_delta_px(
    line_a: np.ndarray | list[float],
    line_b: np.ndarray | list[float],
    frame_wh: tuple[int, int],
) -> float:
    """Symmetric mean distance between two normalized image lines, measured in pixels."""
    la = normalize_horizon_line(line_a, frame_wh)
    lb = normalize_horizon_line(line_b, frame_wh)
    pts_a = _sample_line_points(la, frame_wh)
    pts_b = _sample_line_points(lb, frame_wh)
    da = np.abs(np.c_[pts_a, np.ones(len(pts_a))] @ lb)
    db = np.abs(np.c_[pts_b, np.ones(len(pts_b))] @ la)
    return float(0.5 * (np.mean(da) + np.mean(db)))


def _pack_homography(homography: np.ndarray) -> np.ndarray:
    h = np.asarray(homography, dtype=np.float64).reshape(3, 3).copy()
    if abs(float(h[2, 2])) <= 1e-12:
        raise ValueError("cannot refine homography with near-zero h22")
    h /= h[2, 2]
    return np.array([h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1]], dtype=np.float64)


def _unpack_homography(params: np.ndarray) -> np.ndarray:
    p = np.asarray(params, dtype=np.float64).reshape(8)
    return np.array(
        [[p[0], p[1], p[2]], [p[3], p[4], p[5]], [p[6], p[7], 1.0]],
        dtype=np.float64,
    )


def refine_homography(
    image_pts: np.ndarray,
    bev_pts: np.ndarray,
    initial_h: np.ndarray,
    *,
    frame_wh: tuple[int, int],
    manual_horizon_line: np.ndarray | list[float] | None = None,
    lambda_horizon: float = 1.0,
    max_iters: int = 40,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refine one image->BEV homography by LM on reprojection and optional horizon residuals.

    This is intentionally self-contained (no SciPy dependency).  ``manual_horizon_line`` is a
    camera-level, independently clicked line; ``manual_homography_horizon`` must not be passed here.
    """
    image_arr = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
    bev_arr = np.asarray(bev_pts, dtype=np.float64).reshape(-1, 2)
    if len(image_arr) < 4:
        raise ValueError("Need at least 4 point pairs to refine homography")

    manual_line = None
    if manual_horizon_line is not None and lambda_horizon > 0.0:
        manual_line = normalize_horizon_line(manual_horizon_line, frame_wh)

    def residual(params: np.ndarray) -> np.ndarray:
        h = _unpack_homography(params)
        proj = _project_homography_points(h, image_arr)
        r = (proj - bev_arr).reshape(-1)
        r[~np.isfinite(r)] = 1e6
        if manual_line is not None:
            try:
                h_line = homography_horizon_line(h, frame_wh)
                h_pts = _sample_line_points(h_line, frame_wh)
                h_dist = np.c_[h_pts, np.ones(len(h_pts), dtype=np.float64)] @ manual_line
                r = np.concatenate([r, math.sqrt(lambda_horizon) * h_dist])
            except Exception:
                r = np.concatenate([r, np.full(3, 1e6, dtype=np.float64)])
        return r.astype(np.float64)

    p = _pack_homography(initial_h)
    best_p = p.copy()
    best_r = residual(best_p)
    best_cost = 0.5 * float(best_r @ best_r)
    damping = 1e-3
    accepted_steps = 0

    for iteration in range(max(0, int(max_iters))):
        r0 = residual(p)
        cost0 = 0.5 * float(r0 @ r0)
        jac = np.zeros((len(r0), len(p)), dtype=np.float64)
        for j in range(len(p)):
            step = 1e-6 * max(1.0, abs(float(p[j])))
            pp = p.copy()
            pp[j] += step
            jac[:, j] = (residual(pp) - r0) / step
        diag = np.diag(jac.T @ jac)
        lhs = jac.T @ jac + damping * np.diag(np.maximum(diag, 1.0))
        rhs = -(jac.T @ r0)
        try:
            delta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            damping *= 10.0
            continue
        if not np.all(np.isfinite(delta)):
            damping *= 10.0
            continue
        candidate = p + delta
        rc = residual(candidate)
        cost_c = 0.5 * float(rc @ rc)
        if cost_c + 1e-9 < cost0:
            p = candidate
            damping = max(damping * 0.3, 1e-9)
            accepted_steps += 1
            if cost_c < best_cost:
                best_cost = cost_c
                best_p = candidate.copy()
                best_r = rc
            if float(np.linalg.norm(delta)) < 1e-10 * (1.0 + float(np.linalg.norm(p))):
                break
        else:
            damping = min(damping * 10.0, 1e12)

    h_before = np.asarray(initial_h, dtype=np.float64).reshape(3, 3)
    h_after = _unpack_homography(best_p)
    diag: dict[str, Any] = {
        "point_count": int(len(image_arr)),
        "iterations": int(max(0, int(max_iters))),
        "accepted_steps": int(accepted_steps),
        "cost_before": float(0.5 * (residual(_pack_homography(h_before)) @ residual(_pack_homography(h_before)))),
        "cost_after": float(best_cost),
        "reproj_rms_before_px": homography_reprojection_rms(h_before, image_arr, bev_arr),
        "reproj_rms_after_px": homography_reprojection_rms(h_after, image_arr, bev_arr),
        "used_horizon_constraint": bool(manual_line is not None),
    }
    if manual_line is not None:
        diag["horizon_delta_before_px"] = horizon_line_delta_px(homography_horizon_line(h_before, frame_wh), manual_line, frame_wh)
        diag["horizon_delta_after_px"] = horizon_line_delta_px(homography_horizon_line(h_after, frame_wh), manual_line, frame_wh)
    diag["improved"] = bool(diag["cost_after"] <= diag["cost_before"] + 1e-9)
    diag["homography_before"] = [[float(v) for v in row] for row in h_before.tolist()]
    diag["homography_after"] = [[float(v) for v in row] for row in h_after.tolist()]
    return h_after, diag


def _backproject_bev_to_image(homography: np.ndarray, bev_pts: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(np.asarray(homography, dtype=np.float64).reshape(3, 3))
    dst = np.c_[np.asarray(bev_pts, dtype=np.float64).reshape(-1, 2), np.ones(len(bev_pts), dtype=np.float64)]
    src = (inv @ dst.T).T
    return src[:, :2] / src[:, 2:3]


def render_calibration_refine_overlay(
    image: np.ndarray,
    image_pts: np.ndarray,
    bev_pts: np.ndarray,
    h_before: np.ndarray,
    h_after: np.ndarray,
    *,
    frame_wh: tuple[int, int],
    manual_horizon_line: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """Render before/after point residuals and horizon lines for visual QA."""
    vis = image.copy()
    pts = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
    before_img = _backproject_bev_to_image(h_before, bev_pts)
    after_img = _backproject_bev_to_image(h_after, bev_pts)
    for p, b, a in zip(pts, before_img, after_img):
        pi = tuple(np.round(p).astype(int))
        bi = tuple(np.round(b).astype(int))
        ai = tuple(np.round(a).astype(int))
        cv.circle(vis, pi, 4, (0, 255, 255), -1, cv.LINE_AA)
        cv.arrowedLine(vis, pi, bi, (255, 80, 220), 1, cv.LINE_AA, tipLength=0.2)
        cv.arrowedLine(vis, pi, ai, (60, 220, 60), 1, cv.LINE_AA, tipLength=0.2)
    for line, color, label in (
        (homography_horizon_line(h_before, frame_wh), (255, 80, 220), "H before"),
        (homography_horizon_line(h_after, frame_wh), (60, 220, 60), "H refined"),
    ):
        seg = horizon_line_to_segment(line, frame_wh)
        if seg is not None:
            p0, p1 = (tuple(np.round(seg[0]).astype(int)), tuple(np.round(seg[1]).astype(int)))
            cv.line(vis, p0, p1, color, 2, cv.LINE_AA)
            cv.putText(vis, label, p0, cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv.LINE_AA)
    if manual_horizon_line is not None:
        seg = horizon_line_to_segment(manual_horizon_line, frame_wh)
        if seg is not None:
            p0, p1 = (tuple(np.round(seg[0]).astype(int)), tuple(np.round(seg[1]).astype(int)))
            cv.line(vis, p0, p1, (255, 255, 0), 2, cv.LINE_AA)
            cv.putText(vis, "manual horizon", p0, cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv.LINE_AA)
    return vis


def _collect_camera_calibration_points(
    image: np.ndarray,
    cam: dict[str, Any],
    *,
    marker_world_by_id: dict[int, np.ndarray],
    dictionary_name: str,
    charuco_boards_cfg: dict[str, dict[str, Any]],
    board_corner_world_override: dict[str, dict[int, list[float]]],
    manual_corners_by_cam: dict[str, dict[str, dict[int, list[float]]]],
    scale: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    image_pts, bev_pts = load_manual_point_pairs(cam, scale)
    source = "point_pairs"
    if len(image_pts) < 4:
        cam_board_names: list[str] = cam.get("charuco_boards", [])
        cam_manual = manual_corners_by_cam.get(cam["name"])
        if cam_board_names and charuco_boards_cfg:
            image_pts, bev_pts = detect_charuco_points(
                image,
                charuco_boards_cfg,
                cam_board_names,
                scale,
                board_corner_world_override=board_corner_world_override,
                manual_charuco_corners=cam_manual,
            )
            if len(image_pts) >= 4:
                has_manual = cam_manual and any(b in cam_manual for b in cam_board_names)
                source = f"charuco(manual) boards={cam_board_names}" if has_manual else f"charuco boards={cam_board_names}"
    if len(image_pts) < 4:
        aruco_img, aruco_bev, used_ids = detect_aruco_points(image, marker_world_by_id, dictionary_name, scale)
        if len(aruco_img) >= 4:
            image_pts, bev_pts = aruco_img, aruco_bev
            source = f"aruco markers ids={used_ids}"
    return image_pts, bev_pts, source


def run_bfs_joint_refine_preview(
    config_path: str,
    *,
    scene_id: str = "parking",
    diagnostics_dir: str | None = None,
    enable_horizon_refine: bool = False,
    lambda_horizon: float = 1.0,
    max_iters: int = 40,
) -> dict[str, Any]:
    """Dry-run BFS/RANSAC vs refined H and export console/JSON/overlay diagnostics."""
    config_path = str(resolve_project_path(config_path))
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    scale = float(cfg.get("scale_px_per_meter", 100.0))
    dictionary_name = cfg.get("aruco_dictionary", "DICT_4X4_50")
    ransac_threshold = float(cfg.get("ransac_reproj_threshold_px", 4.0))
    out_dir = resolve_project_path(diagnostics_dir or CALIBRATION_DIAGNOSTICS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded_images: dict[str, np.ndarray] = {}
    for cam in cfg["cameras"]:
        name = cam["name"]
        image_path = str(resolve_project_path(cam["image"]))
        img = cv.imread(image_path, cv.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image for camera {name}: {image_path}")
        camera_matrix, dist_coeffs = load_camera_matrix(cam.get("calibration"))
        if camera_matrix is not None:
            img = cv.undistort(img, camera_matrix, dist_coeffs)
        loaded_images[name] = img

    marker_world_by_id = {int(marker_id): marker_world_corners(marker_cfg) for marker_id, marker_cfg in cfg.get("markers", {}).items()}
    charuco_boards_cfg: dict[str, dict[str, Any]] = cfg.get("charuco_boards", {})
    for board_name, board_cfg in charuco_boards_cfg.items():
        validate_charuco_board_cfg(board_name, board_cfg)
    validate_charuco_board_topology(charuco_boards_cfg)
    cam_board_names_map = {cam["name"]: cam.get("charuco_boards", []) for cam in cfg["cameras"]}
    manual_corners_by_cam: dict[str, dict[str, dict[int, list[float]]]] = {}
    for cam in cfg["cameras"]:
        mc = load_manual_charuco_corners(cam)
        if mc:
            manual_corners_by_cam[cam["name"]] = mc
    board_corner_world_override = auto_locate_charuco_boards(
        loaded_images,
        charuco_boards_cfg,
        cam_board_names_map,
        scale,
        ransac_threshold,
        manual_corners_by_cam=manual_corners_by_cam or None,
    )
    ruler_overrides = locate_boards_from_ruler_measurements(charuco_boards_cfg, board_corner_world_override)
    if ruler_overrides:
        board_corner_world_override.update(ruler_overrides)

    diagnostics: dict[str, Any] = {
        "scene_id": scene_id,
        "config": config_path,
        "scale_px_per_meter": scale,
        "enable_horizon_refine": bool(enable_horizon_refine),
        "lambda_horizon": float(lambda_horizon),
        "cameras": {},
    }
    print("\n[refine-preview] camera       pts  inl  rms_before  rms_after   dH_rel    horizon_before  horizon_after")
    for cam in cfg["cameras"]:
        name = cam["name"]
        image = loaded_images[name]
        frame_wh = (image.shape[1], image.shape[0])
        image_pts, bev_pts, source = _collect_camera_calibration_points(
            image,
            cam,
            marker_world_by_id=marker_world_by_id,
            dictionary_name=dictionary_name,
            charuco_boards_cfg=charuco_boards_cfg,
            board_corner_world_override=board_corner_world_override,
            manual_corners_by_cam=manual_corners_by_cam,
            scale=scale,
        )
        if len(image_pts) < 4:
            raise RuntimeError(f"Camera {name}: not enough calibration points for refine preview")
        h0, inliers = estimate_image_to_bev_homography(image_pts, bev_pts, ransac_threshold)
        inlier_mask = inliers.ravel().astype(bool) if inliers is not None else np.ones(len(image_pts), dtype=bool)
        manual = load_manual_horizon_line(cam, frame_wh) if enable_horizon_refine else None
        h1, diag = refine_homography(
            image_pts[inlier_mask],
            bev_pts[inlier_mask],
            h0,
            frame_wh=frame_wh,
            manual_horizon_line=manual["line"] if manual else None,
            lambda_horizon=lambda_horizon,
            max_iters=max_iters,
        )
        d_h = float(np.linalg.norm((h1 / h1[2, 2]) - (h0 / h0[2, 2])) / max(np.linalg.norm(h0 / h0[2, 2]), 1e-12))
        diag.update({
            "source": source,
            "point_count_total": int(len(image_pts)),
            "inlier_count": int(inlier_mask.sum()),
            "delta_h_rel": d_h,
            "manual_horizon_line": manual,
        })
        diagnostics["cameras"][name] = diag
        overlay = render_calibration_refine_overlay(
            image,
            image_pts[inlier_mask],
            bev_pts[inlier_mask],
            h0,
            h1,
            frame_wh=frame_wh,
            manual_horizon_line=manual["line"] if manual else None,
        )
        imwrite(out_dir / f"{scene_id}_{name}_refine_overlay.jpg", overlay)
        hb = diag.get("horizon_delta_before_px")
        ha = diag.get("horizon_delta_after_px")
        print(
            f"[refine-preview] {name:<12} {len(image_pts):>4} {int(inlier_mask.sum()):>4} "
            f"{diag['reproj_rms_before_px']:>10.3f} {diag['reproj_rms_after_px']:>10.3f} "
            f"{d_h:>8.2e} "
            f"{hb if hb is not None else '-':>14} {ha if ha is not None else '-':>14}"
        )
    json_path = out_dir / f"{scene_id}_refine_preview.json"
    json_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[refine-preview] diagnostics written to {json_path}")
    return diagnostics

def camera_roi_points(image: np.ndarray, cam_cfg: dict[str, Any]) -> np.ndarray:
    """Return the camera's valid-region polygon vertices (pixel coordinates).

    - image_roi configured: use that polygon (≥ 3 vertices, any winding).
    - image_roi omitted: the four image corners (full frame).

    Ordinary prime/zoom lenses (FOV ≤ 90°) can usually omit image_roi.
    For wide-angle lenses (FOV > 90°) or strong edge distortion/occlusion,
    supply a valid ground-region polygon so warped edge pixels are dropped and
    calibration/stitching stay cleaner.
    """
    if "image_roi" in cam_cfg:
        roi = np.asarray(cam_cfg["image_roi"], dtype=np.float32)
        if roi.ndim != 2 or roi.shape[1] != 2 or len(roi) < 3:
            raise ValueError(f"{cam_cfg['name']}: image_roi must be a polygon of [x, y] points")
        return roi

    height, width = image.shape[:2]
    return np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )


def transform_camera_roi(image: np.ndarray, cam_cfg: dict[str, Any], h: np.ndarray) -> np.ndarray:
    """Project the ROI polygon through homography H into BEV pixels for canvas bounds."""
    roi = camera_roi_points(image, cam_cfg).reshape(-1, 1, 2)
    return cv.perspectiveTransform(roi, h).reshape(-1, 2)


def draw_image_roi_overlay(
    image: np.ndarray,
    cam_cfg: dict[str, Any],
    *,
    line_color: tuple[int, int, int] = (0, 220, 80),
    fill_color: tuple[int, int, int] = (0, 180, 60),
    fill_alpha: float = 0.22,
    point_color: tuple[int, int, int] = (0, 60, 255),
) -> np.ndarray:
    """Overlay the image_roi polygon on a debug image.

    Drawn only when YAML defines image_roi; otherwise returns a copy unchanged.
    Debug output only; does not affect calibration or stitching.
    """
    canvas = image.copy()
    if "image_roi" not in cam_cfg:
        return canvas

    roi = camera_roi_points(image, cam_cfg)
    pts = np.round(roi).astype(np.int32).reshape(-1, 1, 2)

    overlay = canvas.copy()
    cv.fillPoly(overlay, [pts], fill_color)
    cv.addWeighted(overlay, fill_alpha, canvas, 1.0 - fill_alpha, 0, canvas)
    cv.polylines(canvas, [pts], True, line_color, 2, cv.LINE_AA)

    for i, (x, y) in enumerate(roi):
        px, py = int(round(x)), int(round(y))
        cv.circle(canvas, (px, py), 5, point_color, -1, cv.LINE_AA)
        cv.circle(canvas, (px, py), 5, (255, 255, 255), 1, cv.LINE_AA)
        cv.putText(
            canvas,
            str(i + 1),
            (px + 8, py - 6),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv.LINE_AA,
        )

    cv.putText(
        canvas,
        "image_roi",
        (pts[0, 0, 0] + 4, max(pts[0, 0, 1] - 10, 16)),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        line_color,
        2,
        cv.LINE_AA,
    )
    return canvas


def draw_bev_roi_overlay(
    bev_image: np.ndarray,
    roi_bev_pts: np.ndarray,
    *,
    line_color: tuple[int, int, int] = (0, 220, 80),
    point_color: tuple[int, int, int] = (0, 60, 255),
) -> np.ndarray:
    """Overlay the projected ROI polygon on a BEV debug image.

    roi_bev_pts are vertices in canvas coordinates (same as warpPerspective).
    Debug output only; does not affect the stitch.
    """
    canvas = bev_image.copy()
    if len(roi_bev_pts) < 3:
        return canvas

    pts = np.round(roi_bev_pts).astype(np.int32).reshape(-1, 1, 2)
    cv.polylines(canvas, [pts], True, line_color, 2, cv.LINE_AA)

    for i, (x, y) in enumerate(roi_bev_pts):
        px, py = int(round(x)), int(round(y))
        cv.circle(canvas, (px, py), 4, point_color, -1, cv.LINE_AA)
        cv.circle(canvas, (px, py), 4, (255, 255, 255), 1, cv.LINE_AA)
        cv.putText(
            canvas,
            str(i + 1),
            (px + 6, py - 4),
            cv.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv.LINE_AA,
        )

    label_x = int(np.clip(pts[:, 0, 0].min() + 4, 0, canvas.shape[1] - 1))
    label_y = int(np.clip(pts[:, 0, 1].min() - 8, 16, canvas.shape[0] - 1))
    cv.putText(
        canvas,
        "image_roi (BEV)",
        (label_x, label_y),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        line_color,
        2,
        cv.LINE_AA,
    )
    return canvas


# Canvas size cap: a degenerate homography can project the full frame to huge
# coordinates and OOM.
DEFAULT_MAX_CANVAS_SIDE_PX = 50_000
DEFAULT_MAX_CANVAS_PIXELS = 500_000_000  # ~120 MP; RGB ~360 MB


def validate_projected_roi(
    camera_name: str,
    image: np.ndarray,
    cam_cfg: dict[str, Any],
    h: np.ndarray,
    *,
    inlier_count: int,
    total_points: int,
    max_side_px: float,
) -> np.ndarray:
    """Check that the ROI projection is sane; raise RuntimeError with diagnostics if not."""
    corners = transform_camera_roi(image, cam_cfg, h)
    if not np.all(np.isfinite(corners)):
        raise RuntimeError(
            f"Camera '{camera_name}' ROI projection contains NaN/Inf; homography is invalid.\n"
            f"  inliers {inlier_count}/{total_points}; check that square_length matches the "
            f"printed board, or increase the number of visible calibration corners."
        )

    span = corners.max(axis=0) - corners.min(axis=0)
    max_abs = float(np.max(np.abs(corners)))
    if max_side_px > 0 and (span[0] > max_side_px or span[1] > max_side_px or max_abs > max_side_px * 2):
        h_img, w_img = image.shape[:2]
        raise RuntimeError(
            f"Camera '{camera_name}' ROI projection span is abnormal: "
            f"width {span[0]:.0f}px, height {span[1]:.0f}px (cap {max_side_px:.0f}px).\n"
            f"  projected corner BEV coords: {np.round(corners).astype(int).tolist()}\n"
            f"  only {inlier_count}/{total_points} inliers (frame {w_img}×{h_img}).\n"
            f"Common causes:\n"
            f"  1) site-YAML ChArUco board parameters "
            f"(square_length / marker_length) do not match the printed boards;\n"
            f"  2) too few or collinear calibration corners, RANSAC left only 4 inliers;\n"
            f"  3) scale_px_per_meter is too large (try 100–300 for this scene).\n"
            f"Suggestion: correct board sizes → recapture images → confirm each camera "
            f"has ≥6 corners."
        )
    return corners


def validate_canvas_size(
    canvas_width: int,
    canvas_height: int,
    *,
    max_side_px: int,
    max_pixels: int,
    camera_corners: list[tuple[str, np.ndarray]],
) -> None:
    pixels = int(canvas_width) * int(canvas_height)
    if canvas_width <= 0 or canvas_height <= 0:
        raise RuntimeError(f"Invalid BEV canvas size: {canvas_width}×{canvas_height}")
    if canvas_width > max_side_px or canvas_height > max_side_px or pixels > max_pixels:
        detail = "\n".join(
            f"  {name}: ROI projection {np.round(c.min(axis=0)).astype(int).tolist()}"
            f" ~ {np.round(c.max(axis=0)).astype(int).tolist()}"
            for name, c in camera_corners
        )
        raise RuntimeError(
            f"BEV canvas too large: {canvas_width}×{canvas_height} = {pixels / 1e6:.1f} MP "
            f"(cap {max_side_px}px side / {max_pixels / 1e6:.0f} MP).\n"
            f"Continuing warpPerspective may exhaust memory (~{pixels * 3 / 1e9:.1f} GB).\n"
            f"Per-camera ROI projection spans:\n{detail}\n"
            f"Fix calibration (board size, corner count, scale_px_per_meter) and re-run --save-cal."
        )


def feather_mask(mask: np.ndarray, feather_radius: int) -> np.ndarray:
    """Convert a binary mask to a [0,1] feathered weight map (legacy, kept for compatibility).

    Distance transform: each foreground pixel weight = min(distance_to_boundary / feather_radius, 1.0).
    feather_radius=0 falls back to a hard 0/1 boundary.
    Prefer build_blend_weights() for new work.
    """
    if feather_radius <= 0:
        return (mask > 0).astype(np.float32)
    binary = (mask > 0).astype(np.uint8)
    dist = cv.distanceTransform(binary, cv.DIST_L2, 3)
    alpha = np.clip(dist / float(feather_radius), 0.0, 1.0)
    return alpha.astype(np.float32)


def build_blend_weights(
    warped_masks: list[np.ndarray],
    seam_blend_px: int = 15,
) -> list[np.ndarray]:
    """Per-camera blend weights from pairwise overlap seams.

    Replaces the older feather_mask approach:
    - Seam location: midline of the true adjacent-camera overlap in BEV
      (equidistant from both exclusive regions), not an inward feather from
      each camera's own ROI boundary.
    - Blend width: exactly seam_blend_px on each side of the seam; everything
      else is hard-assigned to one camera, removing wide-feather ghosting of
      3D objects.

    Algorithm:
    ① Exclusive-region mask per camera (BEV pixels covered only by that camera).
    ② distanceTransform from each exclusive-region boundary.
    ③ For each adjacent overlapping pair: seam_signed = d_i - d_j is the seam
       coordinate; linearly blend where |seam_signed| ≤ seam_blend_px; hard
       assign elsewhere.
    ④ Normalize per pixel so weights sum to 1.

    seam_blend_px=0  → hard cut, zero ghosting, possible 1px stair at the seam
    seam_blend_px=15 → 15 px on each side (30 px total); recommended
    """
    n = len(warped_masks)
    if n == 0:
        return []
    shape = warped_masks[0].shape[:2]

    coverages = [m > 0 for m in warped_masks]

    # Exclusive region: BEV pixels covered only by this camera
    excl_masks: list[np.ndarray] = []
    for i, cov_i in enumerate(coverages):
        others = np.zeros(shape, dtype=bool)
        for j, cov_j in enumerate(coverages):
            if j != i:
                others |= cov_j
        excl_masks.append(cov_i & ~others)

    # Distance from each pixel to each camera's exclusive-region boundary
    # If a camera has no exclusive region (fully overlapped), fall back to its coverage boundary
    dist_to_excl: list[np.ndarray] = []
    for i, excl in enumerate(excl_masks):
        ref = excl if excl.any() else coverages[i]
        src = (~ref).astype(np.uint8) * 255
        d = cv.distanceTransform(src, cv.DIST_L2, 5).astype(np.float32)
        dist_to_excl.append(d)

    # Start from a hard assignment (covered=1, uncovered=0)
    weights: list[np.ndarray] = [cov.astype(np.float32) for cov in coverages]

    # Blend only adjacent pairs in the chain (cam[i] ↔ cam[i+1]).
    # Skip non-adjacent pairs (e.g. A-C) so a false seam is not created where B does not cover.
    for i in range(n - 1):
        j = i + 1
        ov = coverages[i] & coverages[j]
        if not ov.any():
            continue

        d_i = dist_to_excl[i]
        d_j = dist_to_excl[j]
        # seam_signed < 0 → closer to camera i exclusive region → camera i dominates
        seam_signed = (d_i - d_j).astype(np.float32)

        if seam_blend_px <= 0:
            # Hard cut: assign by which exclusive region is closer (d_i == d_j on the midline)
            G_i = (d_i <= d_j).astype(np.float32)
        else:
            # Linear blend: t ∈ [-1, +1]; t=-1 all i, t=+1 all j
            t = np.clip(seam_signed / float(seam_blend_px), -1.0, 1.0)
            G_i = ((1.0 - t) / 2.0).astype(np.float32)

        weights[i] = np.where(ov, G_i, weights[i])
        weights[j] = np.where(ov, 1.0 - G_i, weights[j])

    # Safe normalize: np.divide where= avoids divide-by-zero from eager np.where eval.
    # Adjacent-only blending already sums to 1 on covered pixels; this is a safety net
    # for rare triple overlaps with three cameras.
    total = np.zeros(shape, dtype=np.float32)
    for w in weights:
        total += w
    covered = total > 1e-6
    weights = [
        np.divide(w, total, out=np.zeros_like(w), where=covered).astype(np.float32)
        for w in weights
    ]
    return weights


def tune_gain(x: float) -> float:
    """Clamp gain adjustments to avoid over-compensation (from surround-view-system-introduction).

    Gains > 1 (brighten) use a looser upper bound; gains < 1 (darken) are tighter.
    """
    if x >= 1.0:
        return x * math.exp((1.0 - x) * 0.5)
    else:
        return x * math.exp((1.0 - x) * 0.8)


def compute_luminance_gains(
    camera_names: list[str],
    warped_images: list[np.ndarray],
    warped_masks: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """Chain luminance compensation: per-channel brightness ratios in adjacent BEV overlaps.

    The first camera is the luminance reference (gain=[1,1,1]). Each following
    camera is aligned to the previous (already compensated) median in the overlap.
    tune_gain() keeps the adjustment stable.

    Assumes adjacent cameras in camera_names actually overlap in BEV (chain topology).
    Returns: {camera_name: np.array([gain_B, gain_G, gain_R], dtype=float32)}
    """
    n = len(camera_names)
    gains: dict[str, np.ndarray] = {camera_names[0]: np.ones(3, dtype=np.float32)}

    for i in range(1, n):
        prev_name = camera_names[i - 1]
        curr_name = camera_names[i]

        overlap = (warped_masks[i - 1] > 0) & (warped_masks[i] > 0)
        if not overlap.any():
            gains[curr_name] = np.ones(3, dtype=np.float32)
            print(f"  [gain] {prev_name}↔{curr_name}: no overlap; using unity gain.")
            continue

        # Use the already-compensated previous camera as the reference
        prev_gain = gains[prev_name]
        prev_adj = np.clip(
            warped_images[i - 1].astype(np.float32) * prev_gain.reshape(1, 1, 3),
            0, 255,
        )
        curr_img = warped_images[i].astype(np.float32)

        gain = np.ones(3, dtype=np.float32)
        for c in range(3):  # B, G, R
            med_ref = float(np.median(prev_adj[overlap, c]))
            med_cur = float(np.median(curr_img[overlap, c]))
            if med_cur > 2.0:  # skip near-black channels (avoid dividing by noise)
                raw = med_ref / med_cur
                gain[c] = float(tune_gain(raw))

        gains[curr_name] = gain

    return gains


def blend_images(
    canvas_size: tuple[int, int],
    warped_images: list[np.ndarray],
    warped_masks: list[np.ndarray],
    feather_radius: int = 0,
    *,
    blend_weights: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Weighted-average merge of images already warped onto one canvas.

    Prefer blend_weights from build_blend_weights(); if omitted, fall back to
    the legacy feather_mask path (feather_radius). Uncovered pixels are black.
    """
    width, height = canvas_size
    accum = np.zeros((height, width, 3), dtype=np.float32)

    if blend_weights is not None:
        covered = np.zeros((height, width), dtype=np.float32)
        for image, bw in zip(warped_images, blend_weights):
            accum += image.astype(np.float32) * bw[..., None]
            covered += bw
        result = np.where(covered[..., None] > 1e-6, accum, 0.0)
    else:
        weights_sum = np.zeros((height, width, 1), dtype=np.float32)
        for image, mask in zip(warped_images, warped_masks):
            alpha = feather_mask(mask, feather_radius)[..., None]
            accum += image.astype(np.float32) * alpha
            weights_sum += alpha
        result = accum / np.maximum(weights_sum, 1e-6)
        result[weights_sum[..., 0] <= 1e-6] = 0

    return np.clip(result, 0, 255).astype(np.uint8)


# ── SQLite calibration database ───────────────────────────────────────────────

_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS camera_calibrations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_name         TEXT    NOT NULL,
    scene_id            TEXT    NOT NULL DEFAULT 'parking',
    scale_px_per_meter  REAL    NOT NULL,
    homography          BLOB    NOT NULL,
    image_pts           BLOB,
    world_pts           BLOB,
    source              TEXT,
    point_count         INTEGER,
    inlier_count        INTEGER,
    reproj_rms          REAL,
    geometry_version    TEXT,
    calibrated_at       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cam_scene
    ON camera_calibrations(camera_name, scene_id, calibrated_at DESC);

CREATE TABLE IF NOT EXISTS camera_gains (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_name  TEXT    NOT NULL,
    scene_id     TEXT    NOT NULL DEFAULT 'parking',
    gain_b       REAL    NOT NULL DEFAULT 1.0,
    gain_g       REAL    NOT NULL DEFAULT 1.0,
    gain_r       REAL    NOT NULL DEFAULT 1.0,
    computed_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gains_cam_scene
    ON camera_gains(camera_name, scene_id, computed_at DESC);
"""
# homography: 9 × float64 row-major, 72 bytes
# image_pts:  N × 2 float32, pixel coordinates
# world_pts:  N × 2 float32, world coordinates (meters)
# camera_gains: per-channel (BGR) luminance gains estimated in overlap at calibration time


def init_db(db_path: str) -> sqlite3.Connection:
    """Open (or create) the SQLite calibration DB, ensure schema exists, return the connection."""
    conn = sqlite3.connect(db_path)
    conn.executescript(_DB_SCHEMA)
    return conn


def save_calibration(
    conn: sqlite3.Connection,
    camera_name: str,
    scene_id: str,
    scale: float,
    H: np.ndarray,
    image_pts: np.ndarray,
    world_pts: np.ndarray,
    source: str,
    inlier_count: int,
    reproj_rms: float,
    *,
    geometry_version: str = GEOMETRY_VERSION,
    commit: bool = True,
) -> None:
    """Append one camera's calibration to the database (history is kept).

    ``commit=False`` is for callers that must write calibration and other tables in one transaction.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO camera_calibrations
            (camera_name, scene_id, scale_px_per_meter, homography,
             image_pts, world_pts, source, point_count, inlier_count,
             reproj_rms, geometry_version, calibrated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            camera_name,
            scene_id,
            scale,
            H.astype(np.float64).tobytes(),
            image_pts.astype(np.float32).tobytes() if image_pts is not None and len(image_pts) else None,
            world_pts.astype(np.float32).tobytes() if world_pts is not None and len(world_pts) else None,
            source,
            len(image_pts) if image_pts is not None else 0,
            inlier_count,
            reproj_rms,
            geometry_version,
            now,
        ),
    )
    if commit:
        conn.commit()
    print(f"  [DB] camera '{camera_name}' (scene '{scene_id}') calibration saved.")


def load_calibration(
    conn: sqlite3.Connection,
    camera_name: str,
    scene_id: str,
    *,
    require_geometry_version: str | None = None,
) -> tuple[np.ndarray, float] | None:
    """Load the latest calibration; return (H_3x3, scale_px_per_meter) or None.

    ``require_geometry_version``: train/eval entry points that must consume only
    fab1-semantic calibration should pass this (usually ``GEOMETRY_VERSION``).
    A missing column or version mismatch raises instead of silently fitting
    fab1 features from a cla1 pixel-distance-era calibration.
    Live tracking/replay/visualization that only does geometric projection may
    omit it (lenient).
    """
    row = conn.execute(
        """
        SELECT homography, scale_px_per_meter, geometry_version
          FROM camera_calibrations
         WHERE camera_name = ? AND scene_id = ?
         ORDER BY calibrated_at DESC
         LIMIT 1
        """,
        (camera_name, scene_id),
    ).fetchone()
    if row is None:
        return None
    if require_geometry_version is not None and row[2] != require_geometry_version:
        raise RuntimeError(
            f"Calibration geometry_version mismatch: camera={camera_name!r} scene={scene_id!r} "
            f"record={row[2]!r} required={require_geometry_version!r}. "
            "Older (cla1 or earlier) calibration cannot be used for fab1 train/eval; regenerate with the current pipeline."
        )
    H = np.frombuffer(row[0], dtype=np.float64).reshape(3, 3).copy()
    scale = float(row[1])
    return H, scale


def save_gains(
    conn: sqlite3.Connection,
    gains: dict[str, np.ndarray],
    scene_id: str,
) -> None:
    """Write per-camera BGR luminance gains to the database (history is kept)."""
    now = datetime.now(timezone.utc).isoformat()
    for camera_name, g in gains.items():
        conn.execute(
            """
            INSERT INTO camera_gains
                (camera_name, scene_id, gain_b, gain_g, gain_r, computed_at)
            VALUES (?,?,?,?,?,?)
            """,
            (camera_name, scene_id, float(g[0]), float(g[1]), float(g[2]), now),
        )
    conn.commit()
    print(f"  [DB] luminance gains for {len(gains)} camera(s) saved (scene '{scene_id}').")


def load_gains(
    conn: sqlite3.Connection,
    camera_names: list[str],
    scene_id: str,
) -> dict[str, np.ndarray]:
    """Load the latest luminance gain per camera; missing cameras get unity [1,1,1]."""
    result: dict[str, np.ndarray] = {}
    for name in camera_names:
        row = conn.execute(
            """
            SELECT gain_b, gain_g, gain_r
              FROM camera_gains
             WHERE camera_name = ? AND scene_id = ?
             ORDER BY computed_at DESC
             LIMIT 1
            """,
            (name, scene_id),
        ).fetchone()
        if row is not None:
            result[name] = np.array([row[0], row[1], row[2]], dtype=np.float32)
        else:
            result[name] = np.ones(3, dtype=np.float32)
    return result


def list_calibrations(conn: sqlite3.Connection) -> None:
    """Print a summary of all calibrated cameras in the database."""
    rows = conn.execute(
        """
        SELECT camera_name, scene_id, scale_px_per_meter,
               inlier_count, reproj_rms, geometry_version, calibrated_at
          FROM camera_calibrations
         ORDER BY camera_name, scene_id, calibrated_at DESC
        """
    ).fetchall()
    if not rows:
        print("No calibration records in the database.")
        return
    header = (
        f"{'camera':<12} {'scene':<12} {'scale(px/m)':<14} {'inliers':<8} "
        f"{'reproj RMS(px)':<14} {'geometry_version':<26} calibrated_at(UTC)"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        inliers = r[3] if r[3] is not None else "-"
        rms = f"{r[4]:.2f}" if r[4] is not None else "-"
        geom_ver = r[5] if r[5] is not None else "-"
        print(
            f"{r[0]:<12} {r[1]:<12} {r[2]:<14.1f} {str(inliers):<8} "
            f"{rms:<14} {geom_ver:<26} {r[6]}"
        )


def run(
    config_path: str,
    *,
    db_path: str | None = None,
    save_cal: bool = False,
    use_cal: bool = False,
    scene_id: str = "parking",
    enable_joint_refine: bool = False,
    enable_horizon_refine: bool = False,
    refine_lambda_horizon: float = 1.0,
    refine_max_iters: int = 40,
    refine_diagnostics_dir: str | None = None,
) -> None:
    """Run multi-camera BEV stitching in one of three modes.

    Default      ─ compute homographies from board images and stitch (no DB I/O).
    --save-cal   ─ same, then save homographies + luminance gains to SQLite.
    --use-cal    ─ load homographies + gains from the DB and stitch new images (no boards).
    """
    config_path = str(resolve_project_path(config_path))
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    scale = float(cfg.get("scale_px_per_meter", 100.0))
    dictionary_name = cfg.get("aruco_dictionary", "DICT_4X4_50")
    ransac_threshold = float(cfg.get("ransac_reproj_threshold_px", 4.0))
    # seam_blend_px: smooth transition of N px on each side of the seam (recommended 10–20).
    # Legacy: if only feather_radius_px is set, reuse that value.
    seam_blend_px = int(cfg.get("seam_blend_px", cfg.get("feather_radius_px", 15)))
    output_path = resolve_project_path(cfg.get("output", "bev_mosaic.png"))
    debug_dir = resolve_project_path(cfg.get("debug_dir", "debug_bev"))
    if db_path:
        db_path = str(resolve_project_path(db_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: load and undistort all camera images ──────────────────────────
    loaded_images: dict[str, np.ndarray] = {}
    for cam in cfg["cameras"]:
        name = cam["name"]
        image_path = str(resolve_project_path(cam["image"]))
        img = cv.imread(image_path, cv.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image for camera {name}: {image_path}")
        camera_matrix, dist_coeffs = load_camera_matrix(cam.get("calibration"))
        if camera_matrix is not None:
            img = cv.undistort(img, camera_matrix, dist_coeffs)
        loaded_images[name] = img

    images: list[np.ndarray] = []
    homographies: list[np.ndarray] = []
    camera_cfgs: list[dict[str, Any]] = []
    camera_names: list[str] = []

    if use_cal:
        # ── Run mode: load homographies from DB, skip board detection ──────────
        if db_path is None:
            raise ValueError("--use-cal requires --db to specify the database path.")
        conn = init_db(db_path)
        for cam in cfg["cameras"]:
            name = cam["name"]
            result = load_calibration(conn, name, scene_id)
            if result is None:
                raise RuntimeError(
                    f"No calibration for camera '{name}' (scene '{scene_id}') in the database.\n"
                    f"Run --save-cal first to calibrate and save."
                )
            H, db_scale = result
            if abs(db_scale - scale) > 1e-6:
                print(
                    f"  [warn] camera '{name}': DB scale={db_scale} vs config scale={scale} "
                    f"mismatch; using the database value."
                )
                scale = db_scale
            print(f"  [DB] camera '{name}' calibration loaded.")
            camera_names.append(name)
            camera_cfgs.append(cam)
            images.append(loaded_images[name])
            homographies.append(H)
        conn.close()

    else:
        # ── Calibration mode (default or --save-cal): estimate homographies from boards ─
        marker_world_by_id = {
            int(marker_id): marker_world_corners(marker_cfg)
            for marker_id, marker_cfg in cfg.get("markers", {}).items()
        }
        charuco_boards_cfg: dict[str, dict[str, Any]] = cfg.get("charuco_boards", {})
        for board_name, board_cfg in charuco_boards_cfg.items():
            validate_charuco_board_cfg(board_name, board_cfg)
        validate_charuco_board_topology(charuco_boards_cfg)

        # Step 2: BFS-chain auto-locate of auto_locate boards (arbitrary N cameras)
        cam_board_names_map: dict[str, list[str]] = {
            cam["name"]: cam.get("charuco_boards", [])
            for cam in cfg["cameras"]
        }
        # Collect per-camera hand-labeled corners (for auto_locate and per-camera calibration)
        manual_corners_by_cam: dict[str, dict[str, dict[int, list[float]]]] = {}
        for cam in cfg["cameras"]:
            mc = load_manual_charuco_corners(cam)
            if mc:
                manual_corners_by_cam[cam["name"]] = mc
        if manual_corners_by_cam:
            cams_with_manual = [n for n, mc in manual_corners_by_cam.items()]
            print(f"  [manual] cameras with hand-labeled corners loaded: {cams_with_manual}")

        board_corner_world_override = auto_locate_charuco_boards(
            loaded_images, charuco_boards_cfg, cam_board_names_map, scale, ransac_threshold,
            manual_corners_by_cam=manual_corners_by_cam or None,
        )

        # Step 2 (extra): overlay/supplement auto_locate with ruler_measurements trilateration
        ruler_overrides = locate_boards_from_ruler_measurements(
            charuco_boards_cfg, board_corner_world_override
        )
        if ruler_overrides:
            board_corner_world_override.update(ruler_overrides)
            print(
                f"[ruler] updated coordinates for {len(ruler_overrides)} board(s) from tape measurements: "
                f"{list(ruler_overrides.keys())}"
            )

        # Step 3: estimate a homography per camera
        conn = init_db(db_path) if (save_cal and db_path) else None

        for cam in cfg["cameras"]:
            name = cam["name"]
            image = loaded_images[name]

            # Calibration-point detection priority:
            # ① point_pairs (manual, highest priority when ≥4)
            # ② ChArUco boards (manual_charuco_corners first; else automatic detection)
            # ③ single ArUco markers (legacy)
            image_pts, bev_pts = load_manual_point_pairs(cam, scale)
            source = "point_pairs"

            if len(image_pts) < 4:
                cam_board_names: list[str] = cam.get("charuco_boards", [])
                cam_manual = manual_corners_by_cam.get(name)
                if cam_board_names and charuco_boards_cfg:
                    image_pts, bev_pts = detect_charuco_points(
                        image, charuco_boards_cfg, cam_board_names, scale,
                        board_corner_world_override=board_corner_world_override,
                        manual_charuco_corners=cam_manual,
                    )
                    if len(image_pts) >= 4:
                        has_manual = cam_manual and any(
                            b in cam_manual for b in cam_board_names
                        )
                        source = (
                            f"charuco(manual) boards={cam_board_names}"
                            if has_manual
                            else f"charuco boards={cam_board_names}"
                        )

            if len(image_pts) < 4:
                aruco_img, aruco_bev, used_ids = detect_aruco_points(
                    image, marker_world_by_id, dictionary_name, scale
                )
                if len(aruco_img) >= 4:
                    image_pts, bev_pts = aruco_img, aruco_bev
                    source = f"aruco markers ids={used_ids}"

            if len(image_pts) < 4:
                hints = describe_charuco_detection_gaps(
                    image,
                    charuco_boards_cfg,
                    cam.get("charuco_boards", []),
                    board_corner_world_override,
                )
                detail = "\n  ".join(hints) if hints else "charuco_boards not configured"
                raise RuntimeError(
                    f"Camera '{name}' has fewer than 4 calibration points "
                    f"(got {len(image_pts)}, source={source}).\n"
                    f"Ensure the calibration board is fully visible and sharp, "
                    f"or add YAML manual_charuco_corners if ArUco cannot be detected.\n"
                    f"ChArUco detection details:\n  {detail}"
                )

            h_img_to_bev, inliers = estimate_image_to_bev_homography(
                image_pts, bev_pts, ransac_threshold
            )
            inlier_count = int(inliers.sum()) if inliers is not None else len(image_pts)
            if inlier_count < 4:
                raise RuntimeError(f"Camera {name}: too few homography inliers: {inlier_count}")

            min_inliers = int(cfg.get("min_homography_inliers", 6))
            uses_full_frame = "image_roi" not in cam
            if uses_full_frame and inlier_count < min_inliers:
                raise RuntimeError(
                    f"Camera '{name}' homography inliers={inlier_count} "
                    f"(full-frame calibration recommends ≥ {min_inliers}).\n"
                    f"  calibration points={len(image_pts)}, source={source}.\n"
                    f"Make the board more complete and sharp, or set image_roi to shrink "
                    f"the valid ground region."
                )

            max_roi_side = float(cfg.get("max_canvas_side_px", DEFAULT_MAX_CANVAS_SIDE_PX))
            validate_projected_roi(
                name,
                image,
                cam,
                h_img_to_bev,
                inlier_count=inlier_count,
                total_points=len(image_pts),
                max_side_px=max_roi_side,
            )
            if inlier_count <= 4 and len(image_pts) > 4:
                print(
                    f"  [warn] {name}: RANSAC inliers only {inlier_count}/{len(image_pts)}; "
                    f"homography may be unstable; cover a larger ground area with the board."
                )

            inlier_mask = (
                inliers.ravel().astype(bool)
                if inliers is not None
                else np.ones(len(image_pts), dtype=bool)
            )
            src_h = np.c_[image_pts[inlier_mask], np.ones(inlier_mask.sum(), dtype=np.float64)]
            proj = (h_img_to_bev @ src_h.T).T
            proj = proj[:, :2] / proj[:, 2:3]
            reproj_rms = float(
                np.sqrt(np.mean(np.sum((proj - bev_pts[inlier_mask]) ** 2, axis=1)))
            )

            refine_requested = bool(enable_joint_refine or enable_horizon_refine)
            if refine_requested:
                frame_wh = (image.shape[1], image.shape[0])
                manual_horizon = load_manual_horizon_line(cam, frame_wh) if enable_horizon_refine else None
                if enable_horizon_refine and manual_horizon is None:
                    print(
                        f"  [refine] {name}: manual_horizon_line not configured; "
                        "this pass is reprojection joint refine only."
                    )
                h_before_refine = h_img_to_bev.copy()
                h_refined, refine_diag = refine_homography(
                    image_pts[inlier_mask],
                    bev_pts[inlier_mask],
                    h_img_to_bev,
                    frame_wh=frame_wh,
                    manual_horizon_line=manual_horizon["line"] if manual_horizon else None,
                    lambda_horizon=refine_lambda_horizon,
                    max_iters=refine_max_iters,
                )
                h_img_to_bev = h_refined
                reproj_rms = float(refine_diag["reproj_rms_after_px"])
                source += "+joint_refine"
                if manual_horizon is not None:
                    source += "+horizon_refine"
                validate_projected_roi(
                    name,
                    image,
                    cam,
                    h_img_to_bev,
                    inlier_count=inlier_count,
                    total_points=len(image_pts),
                    max_side_px=max_roi_side,
                )
                overlay_dir = resolve_project_path(refine_diagnostics_dir) if refine_diagnostics_dir else debug_dir
                overlay_dir.mkdir(parents=True, exist_ok=True)
                overlay = render_calibration_refine_overlay(
                    image,
                    image_pts[inlier_mask],
                    bev_pts[inlier_mask],
                    h_before_refine,
                    h_img_to_bev,
                    frame_wh=frame_wh,
                    manual_horizon_line=manual_horizon["line"] if manual_horizon else None,
                )
                imwrite(overlay_dir / f"{name}_refine_overlay.jpg", overlay)
                msg = (
                    f"  [refine] {name}: RMS "
                    f"{refine_diag['reproj_rms_before_px']:.3f} -> "
                    f"{refine_diag['reproj_rms_after_px']:.3f}px, "
                    f"accepted_steps={refine_diag['accepted_steps']}"
                )
                if "horizon_delta_before_px" in refine_diag:
                    msg += (
                        f", horizon_delta "
                        f"{refine_diag['horizon_delta_before_px']:.2f} -> "
                        f"{refine_diag['horizon_delta_after_px']:.2f}px"
                    )
                print(msg)

            vis = draw_image_roi_overlay(image, cam)
            for pt in image_pts.reshape(-1, 2):
                cv.circle(vis, tuple(np.round(pt).astype(int)), 5, (0, 255, 255), -1)
            imwrite(debug_dir / f"{name}_detected_points.jpg", vis)

            print(
                f"{name}: source={source}, points={len(image_pts)}, "
                f"inliers={inlier_count}, reproj_rms={reproj_rms:.2f}px"
            )

            # Save to the database (--save-cal mode)
            if conn is not None:
                world_pts_m = bev_pts / scale
                save_calibration(
                    conn, name, scene_id, scale, h_img_to_bev,
                    image_pts, world_pts_m, source, inlier_count, reproj_rms,
                )

            camera_names.append(name)
            camera_cfgs.append(cam)
            images.append(image)
            homographies.append(h_img_to_bev)

        if conn is not None:
            conn.close()

    # ── Step 4: canvas bounds → warp → luminance → seam weights → save ────────
    projected_corners: list[tuple[str, np.ndarray]] = []
    for image, cam, h in zip(images, camera_cfgs, homographies):
        projected_corners.append(
            (cam["name"], transform_camera_roi(image, cam, h))
        )
    all_corners = np.vstack([c for _, c in projected_corners])
    min_xy = np.floor(all_corners.min(axis=0)).astype(int)
    max_xy = np.ceil(all_corners.max(axis=0)).astype(int)
    margin = int(cfg.get("canvas_margin_px", 50))
    min_xy -= margin
    max_xy += margin

    canvas_width = int(max_xy[0] - min_xy[0] + 1)
    canvas_height = int(max_xy[1] - min_xy[1] + 1)
    max_side = int(cfg.get("max_canvas_side_px", DEFAULT_MAX_CANVAS_SIDE_PX))
    max_pixels = int(cfg.get("max_canvas_pixels", DEFAULT_MAX_CANVAS_PIXELS))
    validate_canvas_size(
        canvas_width,
        canvas_height,
        max_side_px=max_side,
        max_pixels=max_pixels,
        camera_corners=projected_corners,
    )
    print(f"  [BEV] canvas {canvas_width}×{canvas_height} px "
          f"({canvas_width * canvas_height / 1e6:.1f} MP)")
    translate = np.asarray(
        [[1.0, 0.0, -min_xy[0]], [0.0, 1.0, -min_xy[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    warped_images: list[np.ndarray] = []
    warped_masks: list[np.ndarray] = []
    for name, image, cam, h in zip(camera_names, images, camera_cfgs, homographies):
        h_canvas = translate @ h
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        roi = np.round(camera_roi_points(image, cam)).astype(np.int32)
        cv.fillPoly(mask, [roi], 255)
        warped = cv.warpPerspective(
            image,
            h_canvas,
            (canvas_width, canvas_height),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        warped_mask = cv.warpPerspective(
            mask,
            h_canvas,
            (canvas_width, canvas_height),
            flags=cv.INTER_NEAREST,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=0,
        )
        roi_bev = transform_camera_roi(image, cam, h_canvas)
        warped_dbg = draw_bev_roi_overlay(warped, roi_bev)
        imwrite(debug_dir / f"{name}_bev.jpg", warped_dbg)
        warped_images.append(warped)
        warped_masks.append(warped_mask)

    # ── Luminance gain compensation ───────────────────────────────────────────
    gains: dict[str, np.ndarray]
    if use_cal and db_path:
        # Load saved gains from the database
        _gc = init_db(db_path)
        gains = load_gains(_gc, camera_names, scene_id)
        _gc.close()
        any_nontrivial = any(not np.allclose(g, 1.0, atol=0.02) for g in gains.values())
        if any_nontrivial:
            for name, g in gains.items():
                print(f"  [gain] {name}: B={g[0]:.3f} G={g[1]:.3f} R={g[2]:.3f}")
        else:
            print("  [gain] no luminance-gain records in the database; using unity (1.0).")
    else:
        # Estimate from the current images (default / save-cal)
        gains = compute_luminance_gains(camera_names, warped_images, warped_masks)
        for name, g in gains.items():
            if not np.allclose(g, 1.0, atol=0.02):
                print(f"  [gain] {name}: B={g[0]:.3f} G={g[1]:.3f} R={g[2]:.3f}")
        if save_cal and db_path:
            _gc = init_db(db_path)
            save_gains(_gc, gains, scene_id)
            _gc.close()

    # Apply gains to each camera BEV image
    adjusted_images: list[np.ndarray] = []
    for name, img in zip(camera_names, warped_images):
        g = gains.get(name, np.ones(3, dtype=np.float32))
        if np.allclose(g, 1.0, atol=1e-4):
            adjusted_images.append(img)
        else:
            adj = np.clip(
                img.astype(np.float32) * g.reshape(1, 1, 3), 0, 255
            ).astype(np.uint8)
            adjusted_images.append(adj)

    # ── Seam weights (pairwise overlap bands; replaces ROI-boundary feather) ──
    blend_weights = build_blend_weights(warped_masks, seam_blend_px)
    for name, bw in zip(camera_names, blend_weights):
        imwrite(
            debug_dir / f"{name}_blend_weight.png",
            (bw * 255).astype(np.uint8),
        )

    mosaic = blend_images(
        (canvas_width, canvas_height),
        adjusted_images,
        warped_masks,
        blend_weights=blend_weights,
    )
    imwrite(output_path, mosaic)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-camera BEV stitching (N cameras + SQLite calibration persistence)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes
─────
  default     compute homographies and stitch (no DB I/O; same as the old CLI)
  --save-cal  compute homographies, save them to the database, and stitch
  --use-cal   load homographies from the database and stitch (boards not required)
  --list-cal  list all calibration records and exit

Examples
────────
  # First calibration (boards must be in view); save results
  py pipeline/multi_camera_bev_stitch.py --config config/my_scene.yaml --save-cal --db cals.db

  # Later stitch any new images without boards
  py pipeline/multi_camera_bev_stitch.py --config config/my_scene.yaml --use-cal --db cals.db

  # List saved calibration records
  py pipeline/multi_camera_bev_stitch.py --list-cal --db cals.db

  # Multiple sites: distinguish cameras with --scene
  py pipeline/multi_camera_bev_stitch.py --config config/parking_lot.yaml --save-cal --db cals.db --scene parking_lot
  py pipeline/multi_camera_bev_stitch.py --config config/warehouse.yaml   --save-cal --db cals.db --scene warehouse
        """,
    )
    parser.add_argument("--config", help="Path to the YAML config file")
    parser.add_argument(
        "--db",
        default="camera_calibrations.db",
        metavar="PATH",
        help="SQLite database path (default: camera_calibrations.db)",
    )
    parser.add_argument(
        "--scene",
        default="parking",
        metavar="ID",
        help="Scene ID to distinguish sites in one database (default: parking)",
    )
    parser.add_argument(
        "--save-cal",
        action="store_true",
        help="Calibration mode: compute homographies and save them to the database",
    )
    parser.add_argument(
        "--use-cal",
        action="store_true",
        help="Run mode: load homographies from the database; boards not required",
    )
    parser.add_argument(
        "--list-cal",
        action="store_true",
        help="List all calibration records and exit",
    )
    parser.add_argument(
        "--preview-refine",
        action="store_true",
        help="Dry-run BFS/RANSAC vs refined H only; write JSON and overlay, do not save the DB",
    )
    parser.add_argument(
        "--enable-joint-refine",
        action="store_true",
        help="Enable reprojection joint refine before save/stitch (off by default)",
    )
    parser.add_argument(
        "--enable-horizon-refine",
        action="store_true",
        help="Enable manual_horizon_line soft-constraint H refine (implies joint refine; off by default)",
    )
    parser.add_argument(
        "--refine-lambda-horizon",
        type=float,
        default=1.0,
        help="Manual-horizon residual weight; larger stays closer to manual_horizon_line (default 1.0)",
    )
    parser.add_argument(
        "--refine-max-iters",
        type=int,
        default=40,
        help="Max LM iterations for H refine (default 40)",
    )
    parser.add_argument(
        "--refine-diagnostics-dir",
        default=None,
        help=f"Directory for refine diagnostic JSON/overlays (default {CALIBRATION_DIAGNOSTICS_DIR}; run mode uses debug_dir if omitted)",
    )
    args = parser.parse_args()
    if args.list_cal:
        conn = init_db(args.db)
        list_calibrations(conn)
        conn.close()
        return

    if args.config is None:
        parser.error("--config is required (except --list-cal)")

    if args.save_cal and args.use_cal:
        parser.error("--save-cal and --use-cal cannot be used together")

    if args.preview_refine and args.use_cal:
        parser.error("--preview-refine must recompute from images/boards and cannot be used with --use-cal")

    if args.preview_refine:
        run_bfs_joint_refine_preview(
            args.config,
            scene_id=args.scene,
            diagnostics_dir=args.refine_diagnostics_dir,
            enable_horizon_refine=args.enable_horizon_refine,
            lambda_horizon=args.refine_lambda_horizon,
            max_iters=args.refine_max_iters,
        )
        return

    run(
        args.config,
        db_path=args.db if (args.save_cal or args.use_cal) else None,
        save_cal=args.save_cal,
        use_cal=args.use_cal,
        scene_id=args.scene,
        enable_joint_refine=args.enable_joint_refine or args.enable_horizon_refine,
        enable_horizon_refine=args.enable_horizon_refine,
        refine_lambda_horizon=args.refine_lambda_horizon,
        refine_max_iters=args.refine_max_iters,
        refine_diagnostics_dir=args.refine_diagnostics_dir,
    )


if __name__ == "__main__":
    main()
