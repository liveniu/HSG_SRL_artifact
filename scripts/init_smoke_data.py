#!/usr/bin/env python3
"""Build the synthetic init/smoke SQLite package.

The package contains no LUMPI or Synthehicle frames, labels, or coordinates.
It only exercises HSG features, MVC mining, and a short residual train.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import ARTIFACT_ROOT, ensure_import_path  # noqa: E402

ensure_import_path()

from pipeline.geometry_guided_residual_mlp import compute_geometry_base  # noqa: E402
from pipeline.multi_camera_bev_stitch import (  # noqa: E402
    GEOMETRY_VERSION,
    save_calibration,
)
from pipeline.multi_camera_trajectory_fusion import init_trajectory_db  # noqa: E402
from utils.scene_geometry_db import save_scene_overlap_pairs  # noqa: E402
from utils.trajectory_batches import start_batch_stage  # noqa: E402

SCENE = "smoke_pair"
BATCH = "smoke_001"
SCALE = 4.0
FRAME_WH = (1920, 1080)
DT = 0.10
T_TRAIN = 24.0
T_VAL = 28.0
T_END = 40.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild the published synthetic init package even when it already exists",
    )
    return parser.parse_args()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _homography(img_pts: np.ndarray, world_m: np.ndarray) -> np.ndarray:
    world_px = (world_m * SCALE).astype(np.float32)
    H, _ = cv.findHomography(img_pts.astype(np.float32), world_px)
    if H is None:
        raise RuntimeError("failed to fit smoke homography")
    return H.astype(np.float64)


def _project_world_to_image(H: np.ndarray, world_xy: tuple[float, float]) -> tuple[float, float]:
    pt = np.asarray([[[world_xy[0] * SCALE, world_xy[1] * SCALE]]], dtype=np.float32)
    img = cv.perspectiveTransform(pt, np.linalg.inv(H)).reshape(2)
    return float(img[0]), float(img[1])


def _box_at(u: float, v: float, near: bool) -> tuple[float, float, float, float]:
    w = 90.0 if near else 56.0
    h = 70.0 if near else 120.0
    x = u - 0.5 * w
    y = v - 0.72 * h
    return x, y, w, h


def _xywh_inside(box: tuple[float, float, float, float], margin: float = 24.0) -> bool:
    x, y, w, h = box
    return (
        x >= margin
        and y >= margin
        and x + w <= FRAME_WH[0] - margin
        and y + h <= FRAME_WH[1] - margin
    )


def _remove_sqlite_file(path: Path) -> None:
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        if candidate.exists():
            candidate.unlink()


def main() -> int:
    args = _parse_args()
    data_dir = ARTIFACT_ROOT / "data" / "init"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "smoke_pair.db"
    split_path = ARTIFACT_ROOT / "splits" / "smoke_split_manifest.csv"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists() and split_path.exists() and not args.force:
        print(
            f"synthetic init package already exists at {db_path}; "
            "use --force to rebuild it"
        )
        return 0
    if db_path.exists():
        _remove_sqlite_file(db_path)

    cam_h = {
        "CA": _homography(
            np.array([[160, 360], [1760, 360], [80, 1000], [1840, 1000]], dtype=np.float32),
            np.array([[-12, 18], [12, 18], [-12, 4], [12, 4]], dtype=np.float32),
        ),
        "CB": _homography(
            np.array([[240, 180], [1680, 180], [40, 1040], [1880, 1040]], dtype=np.float32),
            np.array([[-18, 45], [18, 45], [-18, 6], [18, 6]], dtype=np.float32),
        ),
    }

    conn = init_trajectory_db(str(db_path), BATCH)
    now = _utc()
    for cam, H in cam_h.items():
        img_pts = np.array([[200, 400], [1720, 400], [200, 900], [1720, 900]], dtype=np.float32)
        world_pts = np.array([[-8, 16], [8, 16], [-8, 6], [8, 6]], dtype=np.float32)
        save_calibration(
            conn,
            cam,
            SCENE,
            SCALE,
            H,
            img_pts,
            world_pts,
            source="artifact_smoke_synthetic",
            inlier_count=4,
            reproj_rms=0.0,
            geometry_version=GEOMETRY_VERSION,
            commit=False,
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO trajectory_camera_config
                (scene_id, batch_id, camera_name, infer_w, infer_h, calib_w, calib_h, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (SCENE, BATCH, cam, FRAME_WH[0], FRAME_WH[1], FRAME_WH[0], FRAME_WH[1], now),
        )
    save_scene_overlap_pairs(conn, SCENE, [("CA", "CB")], replace=True)
    start_batch_stage(conn, [SCENE], BATCH, "import", metadata={"kind": "synthetic_smoke"})

    vehicles = [
        {"gid": 1, "x0": -6.0, "y0": 8.0, "vx": 0.05, "vy": 0.22},
        {"gid": 2, "x0": -2.0, "y0": 9.0, "vx": -0.02, "vy": 0.20},
        {"gid": 3, "x0": 2.5, "y0": 8.5, "vx": 0.03, "vy": 0.18},
        {"gid": 4, "x0": 6.0, "y0": 10.0, "vx": -0.04, "vy": 0.16},
        {"gid": 5, "x0": -4.5, "y0": 7.5, "vx": 0.06, "vy": 0.19},
        {"gid": 6, "x0": 0.0, "y0": 11.0, "vx": 0.01, "vy": 0.15},
        {"gid": 7, "x0": 4.0, "y0": 7.8, "vx": -0.03, "vy": 0.21},
        {"gid": 8, "x0": -1.0, "y0": 8.2, "vx": 0.04, "vy": 0.17},
    ]

    split_rows = []
    n_obs = 0
    t = 0.0
    frame = 0
    while t < T_END - 1e-9:
        if t < T_TRAIN:
            split = "train"
        elif t < T_VAL:
            split = "val"
        else:
            split = "test"
        for veh in vehicles:
            wx = veh["x0"] + veh["vx"] * t
            wy = veh["y0"] + veh["vy"] * t
            for cam, H in cam_h.items():
                u, v = _project_world_to_image(H, (wx, wy))
                box = _box_at(u, v, near=(cam == "CA"))
                if not _xywh_inside(box):
                    continue
                xyxy = np.array(
                    [box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float64,
                )
                anchor_xy, world_geo, feat, risk = compute_geometry_base(
                    xyxy, H, SCALE, FRAME_WH
                )
                if not (risk.valid_ground_side and risk.valid_projection):
                    continue
                sid = f"{SCENE}|{BATCH}|{cam}|{veh['gid']}|{frame}"
                conn.execute(
                    """
                    INSERT INTO trajectory_observations (
                        scene_id, batch_id, camera_name, local_track_id, global_id,
                        frame_index, timestamp_sec, image_x, image_y, world_x, world_y,
                        bbox_x, bbox_y, bbox_w, bbox_h, confidence, class_id, class_name,
                        recorded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        SCENE,
                        BATCH,
                        cam,
                        veh["gid"],
                        veh["gid"],
                        frame,
                        t,
                        float(anchor_xy[0]),
                        float(anchor_xy[1]),
                        float(world_geo[0]),
                        float(world_geo[1]),
                        box[0],
                        box[1],
                        box[2],
                        box[3],
                        0.92,
                        2,
                        "car",
                        now,
                    ),
                )
                n_obs += 1
                split_rows.append(
                    {
                        "dataset": "synthetic",
                        "site": SCENE,
                        "recording": SCENE,
                        "split": split,
                        "scene_id": SCENE,
                        "batch_id": BATCH,
                        "timestamp": f"{t:.3f}",
                        "frame_index": frame,
                        "camera_id": cam,
                        "source_detection_id": sid,
                        "local_track_id": veh["gid"],
                        "gt_global_id": "",
                        "gt_world_x": "",
                        "gt_world_y": "",
                        "has_gt": 0,
                    }
                )
        t += DT
        frame += 1

    conn.commit()
    conn.close()

    fields = [
        "dataset",
        "site",
        "recording",
        "split",
        "scene_id",
        "batch_id",
        "timestamp",
        "frame_index",
        "camera_id",
        "source_detection_id",
        "local_track_id",
        "gt_global_id",
        "gt_world_x",
        "gt_world_y",
        "has_gt",
    ]
    with split_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(split_rows)

    meta = {
        "scene_id": SCENE,
        "batch_id": BATCH,
        "description": "Synthetic two-camera overlap for the artifact smoke test.",
        "n_observations": n_obs,
        "n_split_rows": len(split_rows),
        "db": str(db_path.relative_to(ARTIFACT_ROOT)).replace("\\", "/"),
        "split_manifest": str(split_path.relative_to(ARTIFACT_ROOT)).replace("\\", "/"),
        "geometry_version": GEOMETRY_VERSION,
        "contains_licensed_data": False,
    }
    (data_dir / "smoke_pair_metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {db_path} ({n_obs} observations) and {split_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
