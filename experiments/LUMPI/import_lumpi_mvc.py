#!/usr/bin/env python3
# Vendored HSG-SRL artifact copy. Isolated from the development repo;
# edit this file only. See src/SOURCE_MAP.csv.
"""Import the LUMPI real-world dataset into this project's SQLite DB
for MVC self-supervised training of the geometry-guided residual MLP
and for offline tracking evaluation.

Data conventions
----------------
- The dataset has 7 Measurements (indexed 0-6) with two camera groups:
    Measurements 0-3: camera devices 8, 9, 10 (experiment IDs 0-3)
    Measurements 4-6: camera devices 5, 6, 7 (experiment IDs 4-6)
- Camera calibration comes from meta.json: standard OpenCV pinhole model
  (Z=forward), including distortion coefficients
  - extrinsic = camera->world; world->camera is composed from rvec/tvec
    (do not invert extrinsic as w2c)
- Labels come from Label/{Measurement}/Label.csv (one file shared per Measurement)
  - Fields: time, object_id, 2d_x, 2d_y, 2d_w, 2d_h, score, class_id, visibility,
           3d_cx, 3d_cy, 3d_cz, length, width, height, heading, [extra...]
  - The 2D rectangle fields are world BEV coordinates, not pixels; pixel bboxes
    are computed by projecting the 3D box
  - object_id is globally unique across cameras within a Measurement and is
    used directly as global_id
  - Label rate is 10 Hz; camera video FPS varies by device (25-50 fps)
- Ground height z_ground ~ -2.0 m (LUMPI world coordinates; override with --z-ground)
- Measurement 0's Label folder has a spelling error (Measurment0, missing 'e');
  the script handles this internally
- Resolution is probed from video; if the video is missing it is estimated from
  the principal point cx/cy

Usage
-----
  # Paper wrapper (sets data/work outputs and licensed-data env roots)
  python scripts/import_paper_sites.py --site lumpi_M6

  # Import a single Measurement (training-set labels)
  py experiments/LUMPI/import_lumpi_mvc.py import \\
      --measurement 1 \\
      --db data/work/lumpi_M1.db \\
      --batch lumpi_import_001

  # Import all 7 Measurements (each scene gets scene_id=lumpi_M{N} automatically;
  # do not combine with --scene-id)
  py experiments/LUMPI/import_lumpi_mvc.py import --all \\
      --db data/work/lumpi_all.db \\
      --batch lumpi_import_001

  # Import a small LiDAR-equipped sample from test_data
  py experiments/LUMPI/import_lumpi_mvc.py import \\
      --measurement 0 --test-split \\
      --db data/work/lumpi_test_M0.db \\
      --batch lumpi_test_001

  # Validate import quality
  py experiments/LUMPI/import_lumpi_mvc.py validate \\
      --db data/work/lumpi_M1.db \\
      --scene lumpi_M1 --batch lumpi_import_001

  # Quick database inspection
  py experiments/LUMPI/import_lumpi_mvc.py inspect \\
      --db data/work/lumpi_M1.db \\
      --scene lumpi_M1 --batch lumpi_import_001
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_ARTIFACT_ROOT = Path(__file__).resolve().parents[2]
_ROOT = _ARTIFACT_ROOT
_SRC = _ARTIFACT_ROOT / "src"
_LUMPI_DIR = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline.multi_camera_bev_stitch import save_calibration  # noqa: E402
from pipeline.multi_camera_trajectory_fusion import (  # noqa: E402
    DEFAULT_ANCHOR_SHAPE_ETA,
    box_anchor,
    image_point_to_world,
    init_trajectory_db,
    is_bbox_clipped,
    utc_now,
)
from utils.scene_geometry_db import (  # noqa: E402
    load_batch_camera_frame_wh,
    load_import_audit_metadata,
    load_scene_overlap_pairs,
    save_batch_camera_frame_wh,
    save_import_audit_metadata,
    save_scene_overlap_pairs,
)

try:
    from pipeline.geometry_guided_residual_mlp import GeometryDataView  # noqa: E402
except ImportError:
    GeometryDataView = None  # type: ignore[misc, assignment]

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_DATA_ROOT = _ROOT / "data" / "external" / "LUMPI"
LABEL_FPS = 10.0          # Label.csv timestamp step (10 Hz LiDAR labels)
DEFAULT_Z_GROUND = -2.0   # Ground Z coordinate (LUMPI world, meters)
DEFAULT_SCALE = 4.0
DEFAULT_S0_M = 60.0
ANCHOR_MODE = "multiplicative_gating_offsetted"
MIN_DEPTH = 0.5           # Minimum camera depth (Z_cam); below this = behind camera / not visible

# LUMPI class_id → (COCO class_id, class_name)
# Per the LUMPI dataset labeling protocol (intersection multi-object tracking)
# class_id meanings pending official PDF confirmation; reasonable inferences, may be corrected from data
LUMPI_CLASS_MAP: dict[int, tuple[int, str]] = {
    0: (0, "pedestrian"),
    1: (2, "car"),
    2: (1, "bicycle"),
    3: (3, "motorcycle"),
    4: (7, "truck"),
    5: (2, "van"),
    6: (5, "bus"),
}
DEFAULT_CLASS = (2, "vehicle")

# Measurement index → list of camera device IDs
MEASUREMENT_CAMERAS: dict[int, list[int]] = {
    0: [8, 9, 10], 1: [8, 9, 10], 2: [8, 9, 10], 3: [8, 9, 10],
    4: [5, 6, 7],  5: [5, 6, 7],  6: [5, 6, 7],
}

# Label folder names (Measurement 0 has a spelling error)
LABEL_DIR_NAMES: dict[int, str] = {
    0: "Measurment0",   # Spelling error in the original dataset (missing 'e')
    **{i: f"Measurement{i}" for i in range(1, 7)},
}

# Measurement index = experimentId (one-to-one)
MEASUREMENT_TO_EXPERIMENT: dict[int, int] = {i: i for i in range(7)}

ALL_MEASUREMENTS = list(range(7))


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class CameraCalib:
    name: str
    device_id: int
    K: np.ndarray              # 3×3 intrinsics
    dist: np.ndarray           # (5,) distortion coefficients
    rvec: np.ndarray           # (3,1) Rodrigues rotation vector
    tvec: np.ndarray           # (3,1) translation vector (world→camera)
    T_world_to_cam: np.ndarray # 4×4 world→camera (composed from rvec/tvec)
    T_cam_to_world: np.ndarray # 4×4 camera→world (meta.json extrinsic)
    fps: float
    frame_wh: tuple[int, int]
    homography: np.ndarray     # image → world BEV (includes origin offset and scale)
    reproj_rms: float


@dataclass
class ObservationDraft:
    global_id: int
    camera_name: str
    frame_index: int
    timestamp_sec: float
    image_x: float
    image_y: float
    world_x: float
    world_y: float
    outside_reference_x: float | None
    outside_reference_y: float | None
    outside_reference_image_x: float | None
    outside_reference_image_y: float | None
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    confidence: float
    class_id: int
    class_name: str


# ── Calibration loading ───────────────────────────────────────────────────────


def load_meta(data_root: Path) -> dict[str, Any]:
    """Load meta.json; prefer Label/meta.json (full calibration for all Measurements)."""
    for candidate in [data_root / "Label" / "meta.json", data_root / "meta.json"]:
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"meta.json not found: {data_root}")


def _detect_frame_wh(
    video_path: Path | None,
    K: np.ndarray,
) -> tuple[int, int]:
    """Probe camera resolution: prefer the video; on failure estimate from cx/cy (2×cx, 2×cy, rounded)."""
    if video_path is not None and video_path.is_file():
        cap = cv2.VideoCapture(str(video_path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            return w, h
    # Estimate from intrinsics (even integers)
    cx, cy = float(K[0, 2]), float(K[1, 2])
    w = int(round(2 * cx / 2)) * 2
    h = int(round(2 * cy / 2)) * 2
    return max(w, 640), max(h, 480)


def load_camera_calib(
    meta: dict[str, Any],
    measurement_idx: int,
    device_id: int,
    data_root: Path,
    *,
    z_ground: float = DEFAULT_Z_GROUND,
    use_test_split: bool = False,
) -> CameraCalib:
    """Load one camera's calibration from meta.json and fit a ground homography."""
    experiment_id = MEASUREMENT_TO_EXPERIMENT[measurement_idx]
    exp_map: dict[str, Any] = meta["experiment"][str(experiment_id)]
    session_id = exp_map[str(device_id)]
    sess: dict[str, Any] = meta["session"][str(session_id)]

    K = np.array(sess["intrinsic"], dtype=np.float64)          # 3×3
    dist = np.array(sess["distortion"], dtype=np.float64).flatten()  # (5,)
    rvec = np.array(sess["rvec"], dtype=np.float64).reshape(3, 1)
    tvec = np.array(sess["tvec"], dtype=np.float64).reshape(3, 1)
    # LUMPI meta.json extrinsic is camera→world ([R.T | cam_pos]),
    # not the usual OpenCV world→camera; world→cam is composed from rvec/tvec.
    T_cam_to_world = np.array(sess["extrinsic"], dtype=np.float64)
    R, _ = cv2.Rodrigues(rvec)
    T_world_to_cam = np.eye(4, dtype=np.float64)
    T_world_to_cam[:3, :3] = R
    T_world_to_cam[:3, 3] = tvec.ravel()
    fps = float(sess["fps"])
    cam_name = f"C{device_id:02d}"

    # Probe video path (used to read resolution)
    meas_folder = f"Measurement{measurement_idx}"
    if use_test_split:
        video_path = data_root / "test_data" / meas_folder / "cam" / str(device_id) / "video.mp4"
    else:
        video_path = data_root / meas_folder / "cam" / str(device_id) / "video.mp4"

    frame_wh = _detect_frame_wh(video_path, K)

    try:
        h_mat, reproj = fit_ground_homography(
            T_cam_to_world, K, dist, rvec, tvec,
            frame_wh[0], frame_wh[1], z_ground=z_ground,
        )
    except ValueError:
        h_mat = np.eye(3, dtype=np.float64)
        reproj = float("inf")

    return CameraCalib(
        name=cam_name,
        device_id=device_id,
        K=K,
        dist=dist,
        rvec=rvec,
        tvec=tvec,
        T_world_to_cam=T_world_to_cam,
        T_cam_to_world=T_cam_to_world,
        fps=fps,
        frame_wh=frame_wh,
        homography=h_mat,
        reproj_rms=reproj,
    )


def load_all_calibs(
    meta: dict[str, Any],
    measurement_idx: int,
    data_root: Path,
    *,
    z_ground: float = DEFAULT_Z_GROUND,
    use_test_split: bool = False,
) -> dict[str, CameraCalib]:
    """Load all camera calibrations for a Measurement; return {camera_name: CameraCalib}."""
    device_ids = MEASUREMENT_CAMERAS[measurement_idx]
    calibs: dict[str, CameraCalib] = {}
    for dev_id in device_ids:
        c = load_camera_calib(
            meta, measurement_idx, dev_id, data_root,
            z_ground=z_ground, use_test_split=use_test_split,
        )
        calibs[c.name] = c
    return calibs


# ── Ground homography ─────────────────────────────────────────────────────────


def fit_ground_homography(
    T_cam_to_world: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    width: int,
    height: int,
    *,
    z_ground: float = DEFAULT_Z_GROUND,
    grid: int = 9,
) -> tuple[np.ndarray, float]:
    """Sample the ground plane (z=z_ground) with the standard OpenCV pinhole
    model (Z=forward) and fit an image→world homography."""
    R = T_cam_to_world[:3, :3]          # camera→world rotation
    o_world = T_cam_to_world[:3, 3]     # camera position in the world

    img_pts: list[list[float]] = []
    world_pts: list[list[float]] = []

    for v in np.linspace(0.08 * height, 0.92 * height, grid):
        for u in np.linspace(0.08 * width, 0.92 * width, grid):
            # Standard pinhole ray (camera-local coordinates, Z=1 unit depth)
            ray_cam = np.array(
                [(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0],
                dtype=np.float64,
            )
            d_world = R @ ray_cam
            norm = np.linalg.norm(d_world)
            if norm < 1e-9:
                continue
            d_world /= norm
            if abs(d_world[2]) < 1e-9:
                continue
            s = (z_ground - o_world[2]) / d_world[2]
            if s <= 0:
                continue
            p = o_world + s * d_world
            img_pts.append([u, v])
            world_pts.append([p[0], p[1]])

    if len(img_pts) < 4:
        raise ValueError("not enough ground-homography samples (camera may not look down at the ground)")

    src = np.array(img_pts, dtype=np.float32)
    dst = np.array(world_pts, dtype=np.float32)
    h_mat, _ = cv2.findHomography(src, dst, method=0)
    if h_mat is None:
        raise ValueError("findHomography failed")
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), h_mat).reshape(-1, 2)
    reproj = float(np.sqrt(np.mean(np.linalg.norm(proj - dst, axis=1) ** 2)))
    return h_mat.astype(np.float64), reproj


def fit_image_to_world_homographies(
    samples: dict[str, list[tuple[tuple[float, float], tuple[float, float]]]],
    fallback_calibs: dict[str, CameraCalib],
    *,
    ransac_thresh_m: float = 2.0,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, int], dict[str, str]]:
    """Data-driven image→world H fit from GT bottom-center points; fall back to the ground model if too few points."""
    h_by_cam: dict[str, np.ndarray] = {}
    reproj_by_cam: dict[str, float] = {}
    inliers_by_cam: dict[str, int] = {}
    source_by_cam: dict[str, str] = {}
    for cam_name, calib in fallback_calibs.items():
        pairs = samples.get(cam_name, [])
        if len(pairs) >= 4:
            src = np.asarray([p[0] for p in pairs], dtype=np.float32)
            dst = np.asarray([p[1] for p in pairs], dtype=np.float32)
            method = cv2.RANSAC if len(pairs) > 4 else 0
            h_mat, mask = cv2.findHomography(
                src, dst, method=method, ransacReprojThreshold=ransac_thresh_m
            )
            if h_mat is not None:
                pred = cv2.perspectiveTransform(src.reshape(-1, 1, 2), h_mat).reshape(-1, 2)
                inlier_mask = (
                    mask.reshape(-1).astype(bool) if mask is not None
                    else np.ones(len(src), dtype=bool)
                )
                if not np.any(inlier_mask):
                    inlier_mask = np.ones(len(src), dtype=bool)
                err = np.linalg.norm(pred[inlier_mask] - dst[inlier_mask], axis=1)
                h_by_cam[cam_name] = h_mat.astype(np.float64)
                reproj_by_cam[cam_name] = float(np.sqrt(np.mean(err**2))) if len(err) else 0.0
                inliers_by_cam[cam_name] = int(inlier_mask.sum())
                source_by_cam[cam_name] = "lumpi_bottom_center_fit"
                continue
        h_by_cam[cam_name] = calib.homography
        reproj_by_cam[cam_name] = calib.reproj_rms
        inliers_by_cam[cam_name] = 0
        source_by_cam[cam_name] = "lumpi_ground_plane_fallback"
    return h_by_cam, reproj_by_cam, inliers_by_cam, source_by_cam


def localize_homographies(
    h_base_by_cam: dict[str, np.ndarray],
    sample_world_xy: list[tuple[float, float]],
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    """Translate the world origin to the sample median, then multiply by scale_px_per_meter."""
    if sample_world_xy:
        origin = np.median(np.array(sample_world_xy, dtype=np.float64), axis=0)
    else:
        origin = np.zeros(2, dtype=np.float64)
    T_trans = np.array(
        [[1.0, 0.0, -origin[0]], [0.0, 1.0, -origin[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    T_scale = np.array(
        [[DEFAULT_SCALE, 0.0, 0.0], [0.0, DEFAULT_SCALE, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    h_local = {name: T_scale @ T_trans @ h for name, h in h_base_by_cam.items()}
    return h_local, origin, DEFAULT_SCALE


def compute_overlap_pairs(
    calibs: dict[str, CameraCalib],
    h_local: dict[str, np.ndarray],
    *,
    scale: float,
) -> set[tuple[str, str]]:
    """Infer overlapping camera pairs from AABB intersection of BEV image corners."""
    world_bounds: dict[str, tuple[float, float, float, float]] = {}
    for name, calib in calibs.items():
        w, h = calib.frame_wh
        corners_img = np.array([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=np.float32)
        pts = cv2.perspectiveTransform(
            corners_img, h_local[name].astype(np.float64)
        ).reshape(-1, 2)
        pts_m = pts / scale
        world_bounds[name] = (
            float(pts_m[:, 0].min()),
            float(pts_m[:, 1].min()),
            float(pts_m[:, 0].max()),
            float(pts_m[:, 1].max()),
        )
    names = sorted(world_bounds)
    pairs: set[tuple[str, str]] = set()
    for i, a in enumerate(names):
        ax0, ay0, ax1, ay1 = world_bounds[a]
        for b in names[i + 1:]:
            bx0, by0, bx1, by1 = world_bounds[b]
            if ax0 <= bx1 and bx0 <= ax1 and ay0 <= by1 and by0 <= ay1:
                pairs.add(tuple(sorted((a, b))))  # type: ignore[arg-type]
    return pairs


# ── Projection helpers ────────────────────────────────────────────────────────


def _point_depth_in_cam(E: np.ndarray, world_xyz: np.ndarray) -> float:
    """Return the world point's Z (depth) in the camera frame."""
    p_cam = E @ np.append(world_xyz, 1.0)
    return float(p_cam[2])


def project_world_to_image(
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    world_xyz: np.ndarray,
    frame_wh: tuple[int, int],
    *,
    E: np.ndarray | None = None,
) -> tuple[float, float] | None:
    """Project a world 3D point to pixel coordinates (standard OpenCV with distortion); return None if outside the image."""
    # Depth check (fast Z_cam from the extrinsic matrix)
    if E is not None and _point_depth_in_cam(E, world_xyz) < MIN_DEPTH:
        return None
    pts_img, _ = cv2.projectPoints(
        world_xyz.reshape(1, 1, 3).astype(np.float64), rvec, tvec, K, dist
    )
    u, v = float(pts_img[0, 0, 0]), float(pts_img[0, 0, 1])
    w, h = frame_wh
    if 0 <= u < w and 0 <= v < h:
        return u, v
    return None


def compute_pixel_bbox_from_3d(
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    E: np.ndarray,
    cx: float, cy: float, cz: float,
    length: float, width: float, height: float, heading: float,
    frame_wh: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    """Compute a camera-image pixel bbox (x, y, w, h) from 3D box parameters.

    heading is in radians; length=along heading, width=lateral, height=vertical.
    Returns None if the box is fully behind the camera or projects outside the image.
    """
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    hl, hw, hh = length / 2, width / 2, height / 2

    # 8 corners (rotate about z by heading, then translate to the world center)
    lx = np.array([ hl,  hl, -hl, -hl,  hl,  hl, -hl, -hl])
    ly = np.array([ hw, -hw, -hw,  hw,  hw, -hw, -hw,  hw])
    lz = np.array([ hh,  hh,  hh,  hh, -hh, -hh, -hh, -hh])

    rx = lx * cos_h - ly * sin_h + cx
    ry = lx * sin_h + ly * cos_h + cy
    rz = lz + cz
    corners_world = np.stack([rx, ry, rz], axis=1).astype(np.float64)  # (8, 3)

    # Drop corners behind the camera (Z_cam ≤ MIN_DEPTH)
    z_cam = (E[:3, :3] @ corners_world.T + E[:3, 3:4]).T[:, 2]
    visible_mask = z_cam > MIN_DEPTH
    if not np.any(visible_mask):
        return None

    corners_vis = corners_world[visible_mask]
    pts_img, _ = cv2.projectPoints(
        corners_vis.reshape(-1, 1, 3), rvec, tvec, K, dist
    )
    pts_2d = pts_img.reshape(-1, 2)

    fw, fh = frame_wh
    x1 = max(0.0, float(pts_2d[:, 0].min()))
    y1 = max(0.0, float(pts_2d[:, 1].min()))
    x2 = min(float(fw), float(pts_2d[:, 0].max()))
    y2 = min(float(fh), float(pts_2d[:, 1].max()))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


# ── Label CSV parsing ─────────────────────────────────────────────────────────


def parse_label_csv(
    label_path: Path,
) -> dict[float, list[dict[str, Any]]]:
    """Parse LUMPI Label.csv; return {timestamp_sec: [observation_dict, ...]}.

    Fields per record:
        time, object_id, 2d_x, 2d_y, 2d_w, 2d_h, score, class_id, visibility,
        cx, cy, cz, length, width, height, heading, [extra...]
    """
    result: dict[float, list[dict[str, Any]]] = defaultdict(list)
    with label_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("time"):   # skip header
                continue
            cols = line.split(",")
            if len(cols) < 16:
                continue
            try:
                ts = float(cols[0])
                obj_id = int(cols[1])
                class_id = int(cols[7])
                score = float(cols[6])
                visibility = float(cols[8])
                cx = float(cols[9])
                cy = float(cols[10])
                cz = float(cols[11])
                length = float(cols[12])
                width = float(cols[13])
                height = float(cols[14])
                heading = float(cols[15])
            except (ValueError, IndexError):
                continue

            result[ts].append({
                "object_id": obj_id,
                "class_id": class_id,
                "score": score,
                "visibility": visibility,
                "cx": cx, "cy": cy, "cz": cz,
                "length": length, "width": width, "height": height,
                "heading": heading,
            })
    return dict(result)


def label_path_for_measurement(
    data_root: Path,
    measurement_idx: int,
    *,
    use_test_split: bool = False,
) -> Path:
    """Return the Label.csv path for a Measurement (handles the Measurement0 spelling error)."""
    dir_name = LABEL_DIR_NAMES[measurement_idx]
    if use_test_split:
        return data_root / "test_data" / f"Measurement{measurement_idx}" / "Label.csv"
    return data_root / "Label" / dir_name / "Label.csv"


# ── Scene ID helpers ──────────────────────────────────────────────────────────


def measurement_to_scene_id(measurement_idx: int, *, use_test_split: bool = False) -> str:
    prefix = "lumpi_test" if use_test_split else "lumpi"
    return f"{prefix}_M{measurement_idx}"


# ── Core import ───────────────────────────────────────────────────────────────


def import_scene(
    measurement_idx: int,
    data_root: Path,
    meta: dict[str, Any],
    *,
    db_path: Path,
    scene_id: str,
    batch_id: str,
    z_ground: float,
    overwrite: bool,
    use_test_split: bool,
    anchor_shape_eta: float = DEFAULT_ANCHOR_SHAPE_ETA,
    max_bbox_clip_px: int = 3,
    max_bbox_clip_ratio: float = 0.01,
    min_visibility: float = 0.0,
    h_source_mode: str = "data",
) -> dict[str, Any]:
    """Import a single Measurement into SQLite.

    Uses 3D labels from Label.csv; pixel bboxes are computed by projecting the
    3D box into each camera; object_id is unique across cameras within a
    Measurement and is used directly as global_id.
    """
    calibs = load_all_calibs(
        meta, measurement_idx, data_root,
        z_ground=z_ground, use_test_split=use_test_split,
    )
    camera_names = sorted(calibs.keys())

    label_path = label_path_for_measurement(data_root, measurement_idx, use_test_split=use_test_split)
    if not label_path.is_file():
        raise FileNotFoundError(f"Label.csv not found: {label_path}")

    labels_by_time = parse_label_csv(label_path)
    sorted_times = sorted(labels_by_time.keys())

    drafts: list[ObservationDraft] = []
    homography_samples: dict[str, list[tuple[tuple[float, float], tuple[float, float]]]] = defaultdict(list)
    stats: Counter = Counter()

    for ts in sorted_times:
        # frame_index: 1-based, aligned to LABEL_FPS (10 Hz)
        frame_index = int(round(ts * LABEL_FPS)) + 1
        objs = labels_by_time[ts]

        # Count which global_ids appear on multiple cameras (for mvc_candidates)
        gid_cam_count: Counter = Counter()

        for obj in objs:
            if obj["visibility"] < min_visibility:
                stats["skip_low_visibility"] += 1
                continue

            gid = obj["object_id"]
            cx, cy_w, cz = obj["cx"], obj["cy"], obj["cz"]
            ln, wid, ht = obj["length"], obj["width"], obj["height"]
            heading = obj["heading"]
            class_lumpi = obj["class_id"]
            coco_class_id, class_name = LUMPI_CLASS_MAP.get(class_lumpi, DEFAULT_CLASS)

            # 3D bottom center (ground-truth outside_reference)
            bottom_z = cz - ht / 2.0
            bottom_xyz = np.array([cx, cy_w, bottom_z], dtype=np.float64)
            world_ref_xy = (cx, cy_w)   # world XY (bottom-center projection on the ground)

            for cam_name, calib in calibs.items():
                # Project the bottom center to pixels (outside_reference_image)
                ref_img = project_world_to_image(
                    calib.rvec, calib.tvec, calib.K, calib.dist,
                    bottom_xyz, calib.frame_wh, E=calib.T_world_to_cam,
                )
                if ref_img is None:
                    stats["skip_not_visible"] += 1
                    continue

                # Collect H-fit samples
                homography_samples[cam_name].append((ref_img, world_ref_xy))

                # 3D box projection → pixel bbox
                bbox_xywh = compute_pixel_bbox_from_3d(
                    calib.rvec, calib.tvec, calib.K, calib.dist, calib.T_world_to_cam,
                    cx, cy_w, cz, ln, wid, ht, heading, calib.frame_wh,
                )
                if bbox_xywh is None:
                    stats["skip_no_bbox"] += 1
                    continue

                bx, by, bw, bh = bbox_xywh
                xyxy_arr = np.array([bx, by, bx + bw, by + bh], dtype=np.float64)
                if is_bbox_clipped(xyxy_arr, calib.frame_wh, max_bbox_clip_px, max_bbox_clip_ratio):
                    stats["skip_clipped"] += 1
                    continue

                gid_cam_count[gid] += 1
                drafts.append(ObservationDraft(
                    global_id=gid,
                    camera_name=cam_name,
                    frame_index=frame_index,
                    timestamp_sec=ts,
                    image_x=0.0, image_y=0.0,
                    world_x=0.0, world_y=0.0,
                    outside_reference_x=cx,
                    outside_reference_y=cy_w,
                    outside_reference_image_x=float(ref_img[0]),
                    outside_reference_image_y=float(ref_img[1]),
                    bbox_x=bx, bbox_y=by, bbox_w=bw, bbox_h=bh,
                    confidence=float(obj["score"]) if obj["score"] > 0 else 1.0,
                    class_id=coco_class_id,
                    class_name=class_name,
                ))
                stats["obs"] += 1

        mvc_candidates = sum(1 for cnt in gid_cam_count.values() if cnt >= 2)
        stats["mvc_candidates"] += mvc_candidates

    if not drafts:
        raise RuntimeError(
            f"Measurement {measurement_idx} produced no observations  stats={dict(stats)}"
        )

    # ── Fit homography ────────────────────────────────────────────────────
    sample_world_xy = [
        (d.outside_reference_x, d.outside_reference_y)
        for d in drafts
        if d.outside_reference_x is not None
    ]
    if h_source_mode == "model":
        # Paper protocol: principal homography is the calibrated ground model,
        # not a vehicle-GT fit.
        h_samples_for_fit: dict[str, list] = {}
    else:
        h_samples_for_fit = {
            cam: [(ip, wxy) for ip, wxy in pts] for cam, pts in homography_samples.items()
        }
    h_base, reproj_rms, h_inliers, h_source = fit_image_to_world_homographies(
        h_samples_for_fit,
        calibs,
    )
    h_local, origin, scale = localize_homographies(h_base, sample_world_xy)

    # Subtract origin from outside_reference
    for d in drafts:
        if d.outside_reference_x is not None:
            d.outside_reference_x -= float(origin[0])
            d.outside_reference_y -= float(origin[1])

    overlap_pairs = compute_overlap_pairs(calibs, h_local, scale=scale)

    # ── Compute anchor (image_x/y) → world (P_geo) ───────────────────────────
    for d in drafts:
        calib = calibs[d.camera_name]
        xyxy = np.array(
            [d.bbox_x, d.bbox_y, d.bbox_x + d.bbox_w, d.bbox_y + d.bbox_h],
            dtype=np.float64,
        )
        ax, ay = box_anchor(
            xyxy,
            ANCHOR_MODE,
            homography=h_local[d.camera_name],
            frame_wh=calib.frame_wh,
            scale=scale,
            risk_scalar_mode="metric_jacobian",
            s0_m=DEFAULT_S0_M,
            shape_eta=anchor_shape_eta,
        )
        d.image_x, d.image_y = ax, ay
        p_geo = image_point_to_world(h_local[d.camera_name], (ax, ay), scale)
        d.world_x, d.world_y = float(p_geo[0]), float(p_geo[1])

    # ── Write SQLite ────────────────────────────────────────────────────────
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = init_trajectory_db(str(db_path), batch_id)
    obs_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(trajectory_observations)").fetchall()
    }
    required_ref_img = {"outside_reference_image_x", "outside_reference_image_y"}
    if not required_ref_img.issubset(obs_cols):
        conn.close()
        raise RuntimeError(
            f"{db_path} trajectory_observations is missing {sorted(required_ref_img)}. "
            "Delete this DB and re-run import."
        )

    if overwrite:
        for tbl in (
            "trajectory_observations", "trajectory_global_tracks",
            "trajectory_merges", "trajectory_runtime_stats", "trajectory_camera_config",
        ):
            conn.execute(f"DELETE FROM {tbl} WHERE scene_id=? AND batch_id=?", (scene_id, batch_id))
        for tbl in ("camera_calibrations", "scene_canvas", "scene_overlap_pairs"):
            conn.execute(f"DELETE FROM {tbl} WHERE scene_id=?", (scene_id,))
    else:
        existing = conn.execute(
            "SELECT COUNT(*) FROM trajectory_observations WHERE scene_id=? AND batch_id=?",
            (scene_id, batch_id),
        ).fetchone()
        n_existing = int(existing[0]) if existing else 0
        if n_existing > 0:
            conn.close()
            raise RuntimeError(
                f"scene_id={scene_id} batch_id={batch_id} already has {n_existing} observations; "
                "refusing to append duplicates when --no-overwrite is set. Drop --no-overwrite to overwrite, "
                "or change --scene-id / --batch."
            )

    for cam_name, calib in calibs.items():
        save_calibration(
            conn, cam_name, scene_id, scale, h_local[cam_name],
            image_pts=np.zeros((0, 2), dtype=np.float32),
            world_pts=np.zeros((0, 2), dtype=np.float32),
            source=f"{h_source.get(cam_name, 'lumpi')}:M{measurement_idx}",
            inlier_count=h_inliers.get(cam_name, 0),
            reproj_rms=reproj_rms.get(cam_name, calib.reproj_rms),
            commit=False,
        )

    now = utc_now()
    insert_sql = """
        INSERT INTO trajectory_observations
            (scene_id, batch_id, camera_name, local_track_id, global_id, frame_index,
             timestamp_sec, image_x, image_y, world_x, world_y,
             outside_reference_x, outside_reference_y,
             outside_reference_image_x, outside_reference_image_y,
             bbox_x, bbox_y, bbox_w, bbox_h, confidence,
             class_id, class_name, recorded_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """
    local_track_counter: dict[str, int] = defaultdict(int)
    local_track_map: dict[tuple[str, int], int] = {}
    global_ids: set[int] = set()
    world_pts_list: list[tuple[float, float]] = []

    for d in drafts:
        key = (d.camera_name, d.global_id)
        if key not in local_track_map:
            local_track_counter[d.camera_name] += 1
            local_track_map[key] = local_track_counter[d.camera_name]
        global_ids.add(d.global_id)
        world_pts_list.append((d.world_x, d.world_y))
        conn.execute(insert_sql, (
            scene_id, batch_id,
            d.camera_name, local_track_map[key], d.global_id,
            d.frame_index, d.timestamp_sec,
            d.image_x, d.image_y, d.world_x, d.world_y,
            d.outside_reference_x, d.outside_reference_y,
            d.outside_reference_image_x, d.outside_reference_image_y,
            d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
            d.confidence, d.class_id, d.class_name, now,
        ))

    for gid in sorted(global_ids):
        conn.execute(
            """INSERT OR IGNORE INTO trajectory_global_tracks
               (scene_id, batch_id, global_id, created_at, source)
               VALUES (?,?,?,?,?)""",
            (scene_id, batch_id, gid, now, f"lumpi:M{measurement_idx}"),
        )

    if world_pts_list:
        wpts_arr = np.array(world_pts_list, dtype=np.float64)
        margin_m = 5.0
        min_xy = wpts_arr.min(axis=0) - margin_m
        max_xy = wpts_arr.max(axis=0) + margin_m
        cw = int(np.ceil((max_xy[0] - min_xy[0]) * scale)) + 1
        ch = int(np.ceil((max_xy[1] - min_xy[1]) * scale)) + 1
        conn.execute(
            """INSERT OR REPLACE INTO scene_canvas
               (scene_id, scale, min_x, min_y, canvas_w, canvas_h, saved_at)
               VALUES (?,?,?,?,?,?,?)""",
            (scene_id, scale, float(min_xy[0]), float(min_xy[1]), cw, ch, now),
        )

    frame_wh_by_camera = {cam: calibs[cam].frame_wh for cam in camera_names}
    save_scene_overlap_pairs(conn, scene_id, overlap_pairs, replace=True)
    save_batch_camera_frame_wh(conn, scene_id, batch_id, frame_wh_by_camera)

    meta_out: dict[str, Any] = {
        "scene_id": scene_id,
        "measurement_idx": measurement_idx,
        "use_test_split": use_test_split,
        "label_path": str(label_path),
        "cameras": camera_names,
        "observation_count": len(drafts),
        "global_id_count": len(global_ids),
        "mvc_candidates": int(stats["mvc_candidates"]),
        "overlap_pairs": [list(p) for p in sorted(overlap_pairs)],
        "origin_world": origin.tolist(),
        "scale": scale,
        "z_ground": z_ground,
        "label_fps": LABEL_FPS,
        "frame_wh_by_camera": {cam: list(wh) for cam, wh in frame_wh_by_camera.items()},
        "coordinate_frame": "lumpi_world_minus_origin",
        "world_source": "p_geo",
        "reference_source": "gt_3d_bottom_center",
        "reference_image_source": "opencv_projection_bottom_center",
        "reproj_rms": reproj_rms,
        "homography_source": h_source,
        "homography_inliers": h_inliers,
        "import_stats": dict(stats),
    }
    save_import_audit_metadata(conn, scene_id, batch_id, meta_out)
    conn.commit()
    conn.close()
    return meta_out


# ── Validation ────────────────────────────────────────────────────────────────


def validate_import(
    db_path: Path,
    scene_id: str,
    batch_id: str,
    *,
    max_time_gap: float = 0.35,
    bad_dist_m: float = 80.0,
) -> dict[str, Any]:
    if not db_path.is_file() or db_path.stat().st_size == 0:
        raise FileNotFoundError(f"database missing or empty: {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM trajectory_observations WHERE scene_id=? AND batch_id=?",
            (scene_id, batch_id),
        ).fetchone()
        if row is None or int(row[0]) == 0:
            raise RuntimeError(f"scene_id={scene_id} batch_id={batch_id} has no observations; import first")
    except Exception:
        conn.close()
        raise

    obs_count = conn.execute(
        "SELECT COUNT(*) FROM trajectory_observations WHERE scene_id=? AND batch_id=?",
        (scene_id, batch_id),
    ).fetchone()[0]
    cameras = [
        r[0] for r in conn.execute(
            "SELECT DISTINCT camera_name FROM trajectory_observations "
            "WHERE scene_id=? AND batch_id=? ORDER BY camera_name",
            (scene_id, batch_id),
        )
    ]
    gid_count = conn.execute(
        "SELECT COUNT(DISTINCT global_id) FROM trajectory_observations WHERE scene_id=? AND batch_id=?",
        (scene_id, batch_id),
    ).fetchone()[0]
    multi_gid = conn.execute(
        """SELECT COUNT(*) FROM (
               SELECT global_id FROM trajectory_observations WHERE scene_id=? AND batch_id=?
               GROUP BY global_id HAVING COUNT(DISTINCT camera_name) >= 2
           )""",
        (scene_id, batch_id),
    ).fetchone()[0]

    overlap_pairs = load_scene_overlap_pairs(conn, scene_id, cameras)
    warnings_list: list[str] = []
    if not overlap_pairs and len(cameras) >= 2:
        overlap_pairs = {
            tuple(sorted((cameras[i], cameras[j])))
            for i in range(len(cameras)) for j in range(i + 1, len(cameras))
        }
        warnings_list.append("DB has no scene_overlap_pairs; temporarily using all camera pairs")

    mvc_count = 0
    spatial_count = 0
    dists: list[float] = []

    if GeometryDataView is None:
        warnings_list.append("cannot import GeometryDataView; skipping MVC pair stats")
    else:
        view = GeometryDataView()
        calibrations = view.load_camera_calibrations(conn, scene_id, cameras)
        db_wh = load_batch_camera_frame_wh(conn, scene_id, batch_id, cameras)
        observations = list(
            view.iter_observations_from_db(
                conn, scene_id, batch_id, cameras, calibrations,
                frame_wh_by_camera=db_wh,
            )
        )
        mvc_pairs = view.mine_mvc_pairs(
            observations, overlap_pairs, max_time_gap_sec=max_time_gap
        )
        mvc_count = len(mvc_pairs)
        for pair in mvc_pairs:
            ax, ay = pair.obs_a.world_geo
            bx, by = pair.obs_b.world_geo
            d = float(np.hypot(ax - bx, ay - by))
            dists.append(d)
            if d <= bad_dist_m:
                spatial_count += 1
        if mvc_count == 0:
            warnings_list.append("MVC pair count is 0")
        elif spatial_count < 32:
            warnings_list.append(
                f"spatially consistent MVC (≤{bad_dist_m}m) has only {spatial_count} pairs; sample is small"
            )

    conn.close()
    return {
        "scene_id": scene_id,
        "observation_count": int(obs_count),
        "camera_count": len(cameras),
        "global_id_count": int(gid_count),
        "multi_camera_global_ids": int(multi_gid),
        "overlap_pairs": sorted(overlap_pairs),
        "mvc_pair_count": mvc_count,
        "spatial_mvc_pair_count": spatial_count,
        "mvc_median_dist_m": float(np.median(dists)) if dists else None,
        "warnings": warnings_list,
    }


def print_validation_report(report: dict[str, Any]) -> None:
    print(f"\n=== LUMPI import validation: {report['scene_id']} ===")
    print(f"  observations     : {report['observation_count']}")
    print(f"  cameras          : {report['camera_count']}")
    print(f"  global IDs       : {report['global_id_count']}")
    print(f"  cross-camera IDs : {report['multi_camera_global_ids']}")
    print(f"  overlap pairs    : {report['overlap_pairs']}")
    print(f"  MVC pairs        : {report['mvc_pair_count']}")
    print(f"  MVC (≤80m)       : {report['spatial_mvc_pair_count']}")
    if report.get("mvc_median_dist_m") is not None:
        print(f"  MVC median d  : {report['mvc_median_dist_m']:.3f} m")
    for w in report.get("warnings", []):
        print(f"  [WARN] {w}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def cmd_import(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    meta = load_meta(data_root)

    if args.all and args.scene_id:
        raise SystemExit(
            "error: --all and --scene-id cannot be used together. "
            "--all auto-generates a distinct scene_id (lumpi_M{N}) per Measurement; "
            "forcing the same scene_id with default overwrite would keep only the last Measurement."
        )

    if args.all:
        measurements = ALL_MEASUREMENTS
    else:
        measurements = [args.measurement]

    db_path = Path(args.db)
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    failures: list[tuple[int, str]] = []

    for m_idx in measurements:
        sid = args.scene_id or measurement_to_scene_id(m_idx, use_test_split=args.test_split)
        print(f"\n>>> importing Measurement{m_idx} -> scene_id={sid}")
        try:
            meta_out = import_scene(
                m_idx, data_root, meta,
                db_path=db_path,
                scene_id=sid,
                batch_id=args.batch,
                z_ground=args.z_ground,
                overwrite=not args.no_overwrite,
                use_test_split=args.test_split,
                anchor_shape_eta=args.anchor_shape_eta,
                min_visibility=args.min_visibility,
                h_source_mode=args.h_source,
            )
            results.append(meta_out)
            print(
                f"  obs={meta_out['observation_count']}  "
                f"global_id={meta_out['global_id_count']}  "
                f"mvc_candidates={meta_out['mvc_candidates']}  "
                f"overlap={meta_out['overlap_pairs']}"
            )
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            failures.append((m_idx, str(exc)))
            if args.fail_fast:
                raise SystemExit(1) from exc

    elapsed = time.perf_counter() - t0
    print(f"\nimport done ({elapsed:.1f}s)  DB={db_path}  "
          f"ok={len(results)} fail={len(failures)}")
    for r in results:
        sid = r["scene_id"]
        print(f"  {sid}: obs={r['observation_count']} mvc={r['mvc_candidates']}")
        print(
            f"    validate: py experiments/LUMPI/import_lumpi_mvc.py validate "
            f"--db {db_path} --scene {sid} --batch {args.batch}"
        )
    if failures:
        for m_idx, msg in failures:
            print(f"  [FAIL] Measurement{m_idx}: {msg}")
        raise SystemExit(1)


def cmd_validate(args: argparse.Namespace) -> None:
    report = validate_import(
        Path(args.db), args.scene, args.batch,
        max_time_gap=args.max_time_gap,
        bad_dist_m=args.bad_dist_m,
    )
    print_validation_report(report)
    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  JSON report -> {args.json}")


def cmd_inspect(args: argparse.Namespace) -> None:
    db_path = Path(args.db)
    if not db_path.is_file():
        raise FileNotFoundError(f"DB not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    print(f"=== inspect (LUMPI) scene={args.scene} batch={args.batch} ===")
    for r in conn.execute(
        """SELECT camera_name, COUNT(*), MIN(timestamp_sec), MAX(timestamp_sec),
                  MIN(world_x), MAX(world_x), MIN(world_y), MAX(world_y), AVG(confidence)
             FROM trajectory_observations
            WHERE scene_id=? AND batch_id=?
            GROUP BY camera_name ORDER BY camera_name""",
        (args.scene, args.batch),
    ):
        print(
            f"  {r[0]}: n={r[1]} t=[{r[2]:.1f},{r[3]:.1f}] "
            f"wx=[{r[4]:.1f},{r[5]:.1f}] wy=[{r[6]:.1f},{r[7]:.1f}] conf={r[8]:.3f}"
        )
    pairs = load_scene_overlap_pairs(conn, args.scene)
    audit = load_import_audit_metadata(conn, args.scene, args.batch)
    print(f"  overlap_pairs : {sorted(pairs)}")
    if audit:
        print(f"  mvc_candidates: {audit.get('mvc_candidates')}")
        print(f"  import_stats  : {audit.get('import_stats')}")
    conn.close()


def _configure_stdio_utf8() -> None:
    """Best-effort: let a Windows GBK console print Unicode in help/logs."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main() -> None:
    _configure_stdio_utf8()
    parser = argparse.ArgumentParser(
        description="LUMPI dataset -> this project's SQLite (MVC residual training)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── import ──────────────────────────────────────────────────────────────
    p_import = sub.add_parser("import", help="import one or all Measurements")
    p_import.add_argument(
        "--data-root", default=str(DEFAULT_DATA_ROOT),
        help=f"LUMPI data root (default: {DEFAULT_DATA_ROOT})",
    )
    grp = p_import.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--measurement", type=int, choices=ALL_MEASUREMENTS, metavar="N",
        help="Measurement index (0-6)",
    )
    grp.add_argument("--all", action="store_true", help="import all 7 Measurements")
    p_import.add_argument("--db", required=True, help="output SQLite path")
    p_import.add_argument("--batch", required=True, help="batch_id, must match the pipeline")
    p_import.add_argument(
        "--scene-id", default=None,
        help="scene_id in the DB (default lumpi_M{N}); cannot be used with --all",
    )
    p_import.add_argument(
        "--z-ground", type=float, default=DEFAULT_Z_GROUND,
        help=f"ground Z coordinate (LUMPI world, default {DEFAULT_Z_GROUND})",
    )
    p_import.add_argument(
        "--test-split", action="store_true",
        help="use the test_data/ subdirectory (small sample with LiDAR frames)",
    )
    p_import.add_argument(
        "--min-visibility", type=float, default=0.0,
        help="minimum visibility threshold (Label.csv visibility column; default 0.0 = no filter)",
    )
    p_import.add_argument(
        "--no-overwrite", action="store_true",
        help="refuse import if scene_id+batch_id already has observations (overwrite by default)",
    )
    p_import.add_argument(
        "--anchor-shape-eta", type=float, default=DEFAULT_ANCHOR_SHAPE_ETA,
        dest="anchor_shape_eta", metavar="ETA",
        help="anchor shape mix coefficient (0=bottom center, 1=slimness shape term)",
    )
    p_import.add_argument(
        "--fail-fast", action="store_true",
        help="stop on the first failure in a batch import (default: continue remaining Measurements; exit code 1 if any failed)",
    )
    p_import.add_argument(
        "--h-source", choices=("data", "model"), default="data",
        help="image→world H source: data=GT bottom-center fit (legacy, diagnostics only); "
             "model=calibrated ground model (paper protocol: principal homography is the calibrated ground model, not a vehicle-GT fit)",
    )
    p_import.set_defaults(func=cmd_import)

    # ── validate ─────────────────────────────────────────────────────────────
    p_val = sub.add_parser("validate", help="validate import quality")
    p_val.add_argument("--db", required=True)
    p_val.add_argument("--scene", required=True)
    p_val.add_argument("--batch", required=True)
    p_val.add_argument("--max-time-gap", type=float, default=0.35)
    p_val.add_argument("--bad-dist-m", type=float, default=80.0)
    p_val.add_argument("--json", default=None, help="path for JSON report output")
    p_val.set_defaults(func=cmd_validate)

    # ── inspect ──────────────────────────────────────────────────────────────
    p_ins = sub.add_parser("inspect", help="briefly inspect database contents")
    p_ins.add_argument("--db", required=True)
    p_ins.add_argument("--scene", required=True)
    p_ins.add_argument("--batch", required=True)
    p_ins.set_defaults(func=cmd_inspect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
