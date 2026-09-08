#!/usr/bin/env python3
# Vendored HSG-SRL artifact copy. Isolated from the development repo;
# edit this file only. See src/SOURCE_MAP.csv.
"""Profile FAB2 residual localization and summarize edge pipeline runtime."""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.fab2_common import (  # noqa: E402
    FAB2_DEFAULT_RESULTS,
    connect_db,
    ensure_dir,
    finite_float,
    infer_dataset_labels,
    read_csv_rows,
    scene_camera_names,
    selected_scenes,
    stats_from_errors,
    table_exists,
    write_csv,
    write_json,
    write_run_metadata,
)
from pipeline.geometry_guided_residual_mlp import (  # noqa: E402
    GeometryDataView,
    MVCResidualController,
    compute_geometry_base,
    xywh_to_xyxy,
    _infer_and_calib_wh_from_db,
)
from pipeline.multi_camera_trajectory_fusion import adapt_homography_to_resolution  # noqa: E402


MICRO_FIELDS = [
    "dataset",
    "site",
    "recording",
    "scene_id",
    "batch_id",
    "method",
    "checkpoint",
    "parameter_count",
    "sample_count",
    "warmup",
    "p50_ms",
    "p90_ms",
    "p99_ms",
    "mean_ms",
    "targets_per_sec",
    "device_note",
]

SYSTEM_FIELDS = [
    "dataset",
    "site",
    "recording",
    "scene_id",
    "batch_id",
    "frames",
    "camera_count",
    "scene_duration_sec",
    "wall_duration_sec",
    "aggregate_fps",
    "per_stream_fps",
    "mean_active_local_tracks",
    "mean_active_global_tracks",
    "mean_merge_candidate_comparisons",
    "mean_db_rows_written",
    "mean_storage_rows_per_sec",
    "mean_power_w",
    "energy_j_per_frame",
    "device_note",
]


def _param_count(checkpoint: str | None) -> int:
    if not checkpoint:
        return 0
    try:
        import torch

        blob = torch.load(str(Path(checkpoint)), map_location="cpu")
        state = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        return int(sum(v.numel() for v in state.values() if hasattr(v, "numel")))
    except Exception:
        return 0


def _sample_rows(conn: sqlite3.Connection, scene_id: str, batch_id: str, limit: int) -> list[sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT camera_name, bbox_x, bbox_y, bbox_w, bbox_h
          FROM trajectory_observations
         WHERE scene_id=? AND batch_id=?
         ORDER BY timestamp_sec, frame_index, camera_name
         LIMIT ?
        """,
        (scene_id, batch_id, limit),
    ).fetchall()
    return rows


def _calibrations(conn: sqlite3.Connection, scene_id: str, batch_id: str) -> dict[str, tuple[np.ndarray, float, tuple[int, int]]]:
    view = GeometryDataView()
    cameras = scene_camera_names(conn, scene_id, batch_id)
    cal = view.load_camera_calibrations(conn, scene_id, cameras)
    infer_wh, calib_wh = _infer_and_calib_wh_from_db(conn, scene_id, batch_id, cameras)
    out: dict[str, tuple[np.ndarray, float, tuple[int, int]]] = {}
    for cam, (h_cal, scale) in cal.items():
        wh = infer_wh.get(cam) or (1920, 1080)
        cwh = calib_wh.get(cam)
        if cwh and cwh != wh:
            h_cal = adapt_homography_to_resolution(h_cal, cwh, wh)
        out[cam] = (h_cal, scale, wh)
    return out


def _micro_for_scene(conn: sqlite3.Connection, scene_id: str, args: argparse.Namespace) -> dict[str, Any]:
    labels = infer_dataset_labels(scene_id, args.dataset)
    rows = _sample_rows(conn, scene_id, args.batch, args.samples + args.warmup)
    if not rows:
        raise RuntimeError(f"No observations for scene={scene_id} batch={args.batch}")
    cals = _calibrations(conn, scene_id, args.batch)
    controller = MVCResidualController.from_checkpoint(Path(args.checkpoint)) if args.checkpoint else None
    elapsed_ms: list[float] = []
    for idx, row in enumerate(rows):
        cam = str(row["camera_name"])
        if cam not in cals:
            continue
        h_mat, scale, wh = cals[cam]
        xyxy = xywh_to_xyxy((float(row["bbox_x"]), float(row["bbox_y"]), float(row["bbox_w"]), float(row["bbox_h"])))
        t0 = time.perf_counter()
        if controller is not None:
            controller.predict_residual(xyxy, h_mat, scale, wh, camera_name=cam, scene_id=scene_id)
        else:
            compute_geometry_base(xyxy, h_mat, scale, wh)
        dt = (time.perf_counter() - t0) * 1000.0
        if idx >= args.warmup:
            elapsed_ms.append(dt)
    st = stats_from_errors(elapsed_ms)
    mean_ms = st["mae_m"]
    return {
        "dataset": labels.dataset,
        "site": labels.site,
        "recording": labels.recording,
        "scene_id": scene_id,
        "batch_id": args.batch,
        "method": "HSG-SRL residual head" if args.checkpoint else "Homography+HSG geometry",
        "checkpoint": args.checkpoint or "",
        "parameter_count": _param_count(args.checkpoint),
        "sample_count": len(elapsed_ms),
        "warmup": args.warmup,
        "p50_ms": st["median_m"],
        "p90_ms": st["p90_m"],
        "p99_ms": float(np.percentile(np.asarray(elapsed_ms), 99)) if elapsed_ms else "",
        "mean_ms": mean_ms,
        "targets_per_sec": (1000.0 / mean_ms) if mean_ms and math.isfinite(mean_ms) and mean_ms > 0 else "",
        "device_note": args.device_note,
    }


def _parse_recorded_at(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _runtime_spans(rows: list[sqlite3.Row]) -> tuple[int, float, float]:
    """Aggregate processed sync ticks, source-timeline span and wall-clock span.

    ``trajectory_runtime_stats`` is sampled every ``--stats-every-frames`` ticks,
    so the row count is not the number of processed frames; ``frame_index`` is the
    global sync-tick counter and is what must be spanned.  ``--resume`` restarts
    ``frame_index`` at 0 while ``timestamp_sec`` continues from the resume point, so
    spans are accumulated per session instead of taking a global max-min, which
    would otherwise fold the idle gap between two sessions into the wall clock.

    ``rows`` must be in insertion order (``ORDER BY rowid``); a session boundary is
    a ``frame_index`` that fails to advance.  Ordering by ``timestamp_sec`` cannot
    separate sessions because a resumed run starts at the timestamp the previous
    one stopped at.
    """
    sessions: list[list[sqlite3.Row]] = []
    for row in rows:
        fi = int(row["frame_index"])
        if not sessions or fi <= int(sessions[-1][-1]["frame_index"]):
            sessions.append([row])
        else:
            sessions[-1].append(row)

    frames = 0
    scene_span = 0.0
    wall_span = 0.0
    for session in sessions:
        first, last = session[0], session[-1]
        frames += int(last["frame_index"]) - int(first["frame_index"]) + 1
        scene_span += max(0.0, float(last["timestamp_sec"]) - float(first["timestamp_sec"]))
        t_start = _parse_recorded_at(first["recorded_at"])
        t_end = _parse_recorded_at(last["recorded_at"])
        if t_start is not None and t_end is not None:
            wall_span += max(0.0, (t_end - t_start).total_seconds())
    return frames, scene_span, wall_span


def _mean_power(power_csv: str | None) -> float | None:
    if not power_csv:
        return None
    values: list[float] = []
    for row in read_csv_rows(power_csv):
        p = finite_float(row.get("power_w", row.get("power", "")))
        if p is not None:
            values.append(p)
    return float(np.mean(values)) if values else None


def _system_for_scene(conn: sqlite3.Connection, scene_id: str, args: argparse.Namespace) -> dict[str, Any]:
    labels = infer_dataset_labels(scene_id, args.dataset)
    empty_row = {
        "dataset": labels.dataset,
        "site": labels.site,
        "recording": labels.recording,
        "scene_id": scene_id,
        "batch_id": args.batch,
        "frames": 0,
        "camera_count": "",
        "scene_duration_sec": "",
        "wall_duration_sec": "",
        "aggregate_fps": "",
        "per_stream_fps": "",
        "mean_active_local_tracks": "",
        "mean_active_global_tracks": "",
        "mean_merge_candidate_comparisons": "",
        "mean_db_rows_written": "",
        "mean_storage_rows_per_sec": "",
        "mean_power_w": _mean_power(args.power_csv) or "",
        "energy_j_per_frame": "",
        "device_note": args.device_note,
    }
    if not table_exists(conn, "trajectory_runtime_stats"):
        return empty_row
    rows = conn.execute(
        """
        SELECT frame_index, timestamp_sec, active_local_tracks, active_global_tracks,
               merge_candidate_comparisons, db_rows_written, storage_rows_per_sec,
               recorded_at
          FROM trajectory_runtime_stats
         WHERE scene_id=? AND batch_id=?
         ORDER BY rowid
        """,
        (scene_id, args.batch),
    ).fetchall()
    if not rows:
        return empty_row
    frames, scene_duration, wall_duration = _runtime_spans(rows)
    camera_count = max(1, len(scene_camera_names(conn, scene_id, args.batch)))
    # One delivered frame per camera per sync tick: per-stream rate; aggregate multiplies by camera count.
    per_stream_fps = frames / wall_duration if wall_duration > 0 else ""
    aggregate_fps = (per_stream_fps * camera_count) if per_stream_fps else ""
    power = _mean_power(args.power_csv)
    energy = (power / aggregate_fps) if power is not None and aggregate_fps else ""
    return {
        "dataset": labels.dataset,
        "site": labels.site,
        "recording": labels.recording,
        "scene_id": scene_id,
        "batch_id": args.batch,
        "frames": frames,
        "camera_count": camera_count,
        "scene_duration_sec": scene_duration,
        "wall_duration_sec": wall_duration,
        "aggregate_fps": aggregate_fps,
        "per_stream_fps": per_stream_fps,
        "mean_active_local_tracks": float(np.mean([float(r["active_local_tracks"]) for r in rows])),
        "mean_active_global_tracks": float(np.mean([float(r["active_global_tracks"]) for r in rows])),
        "mean_merge_candidate_comparisons": float(np.mean([float(r["merge_candidate_comparisons"]) for r in rows])),
        "mean_db_rows_written": float(np.mean([float(r["db_rows_written"]) for r in rows])),
        "mean_storage_rows_per_sec": float(np.mean([float(r["storage_rows_per_sec"]) for r in rows])),
        "mean_power_w": power or "",
        "energy_j_per_frame": energy,
        "device_note": args.device_note,
    }


def run(args: argparse.Namespace) -> None:
    out_root = ensure_dir(Path(args.out_dir))
    ivf = ensure_dir(out_root / "iv_f")
    conn = connect_db(args.db)
    try:
        scenes = selected_scenes(conn, args.batch, args.scene)
        micro_rows = []
        system_rows = []
        if args.mode in ("micro", "both"):
            micro_rows = [_micro_for_scene(conn, scene, args) for scene in scenes]
        if args.mode in ("system", "both"):
            system_rows = [_system_for_scene(conn, scene, args) for scene in scenes]
    finally:
        conn.close()
    write_csv(ivf / "orin_residual_microbenchmark.csv", micro_rows, MICRO_FIELDS)
    write_csv(ivf / "orin_4stream_pipeline.csv", system_rows, SYSTEM_FIELDS)
    write_json(ivf / "edge_profile_summary.json", {"mode": args.mode, "scenes": scenes, "micro_rows": len(micro_rows), "system_rows": len(system_rows)})
    write_run_metadata(out_root, "profile_fab2_edge", args, {"scenes": scenes})
    print(f"Edge profile tables written under {ivf}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--out-dir", default=str(FAB2_DEFAULT_RESULTS))
    parser.add_argument("--mode", choices=("micro", "system", "both"), default="both")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--power-csv", default=None, help="Optional CSV with power_w column")
    parser.add_argument("--device-note", default="Jetson Orin Nano 40W expected; host run if not on Orin")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
