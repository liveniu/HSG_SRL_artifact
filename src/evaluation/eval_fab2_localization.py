#!/usr/bin/env python3
# Vendored HSG-SRL artifact copy. Isolated from the development repo;
# edit this file only. See src/SOURCE_MAP.csv.
"""Evaluate FAB2 BEV localization metrics."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.fab2_common import (  # noqa: E402
    CANONICAL_PREDICTION_FIELDS,
    FAB2_DEFAULT_RESULTS,
    assign_spatial_slices,
    connect_db,
    distance_bin,
    ensure_dir,
    finite_float,
    group_errors,
    load_db_prediction_rows,
    load_gt_index,
    load_prediction_csv,
    load_slice_rules,
    paired_bootstrap_delta,
    selected_scenes,
    split_csv_arg,
    upsert_csv_rows,
    write_csv,
    write_json,
    write_run_metadata,
)


LOCALIZATION_FIELDS = [
    "method",
    "evaluation_regime",
    "metric_scope",
    "dataset",
    "site",
    "recording",
    "scene_id",
    "camera_id",
    "spatial_slice",
    "distance_bin",
    "n",
    "matched_count",
    "rmse_m",
    "mae_m",
    "median_m",
    "p90_m",
    "p95_m",
    "max_m",
    "paired_n",
    "baseline_method",
    "delta_rmse_m",
    "ci95_low_m",
    "ci95_high_m",
]


def _error_by_source(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        px, py = finite_float(row.get("world_x")), finite_float(row.get("world_y"))
        gx, gy = finite_float(row.get("gt_world_x")), finite_float(row.get("gt_world_y"))
        if None in (px, py, gx, gy):
            continue
        key = str(row.get("source_detection_id") or row.get("event_key") or "")
        if not key:
            continue
        out[key] = math.hypot(float(px) - float(gx), float(py) - float(gy))
    return out


def _annotate_distance(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["distance_bin"] = distance_bin(row)


def _filter_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    class_ids = {int(x) for x in split_csv_arg(args.class_ids)} if args.class_ids else None
    out: list[dict[str, Any]] = []
    for row in rows:
        if args.split is not None and str(row.get("split", "")) != args.split:
            continue
        if class_ids is not None:
            cid = row.get("class_id", "")
            try:
                if int(float(cid)) not in class_ids:
                    continue
            except (TypeError, ValueError):
                continue
        out.append(row)
    return out


def _metric_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    grouped: list[tuple[str, list[str]]] = [
        ("overall", ["method", "evaluation_regime"]),
        ("per_scene", ["method", "evaluation_regime", "dataset", "site", "recording", "scene_id"]),
        ("per_camera", ["method", "evaluation_regime", "dataset", "site", "recording", "scene_id", "camera_id"]),
        ("spatial_slice", ["method", "evaluation_regime", "dataset", "site", "recording", "scene_id", "spatial_slice"]),
        ("distance_bin", ["method", "evaluation_regime", "dataset", "site", "recording", "scene_id", "distance_bin"]),
    ]
    out: list[dict[str, Any]] = []
    baseline_errors: dict[str, float] = {}
    baseline_method = ""
    if args.baseline_predictions:
        base_rows = _filter_rows(load_prediction_csv(args.baseline_predictions, require_gt=True), args)
        base_rows = assign_spatial_slices(
            base_rows,
            slice_rules=load_slice_rules(args.slice_csv),
            handoff_window_sec=args.handoff_window_sec,
            infer_from_tracks=args.infer_spatial_slices,
        )
        _annotate_distance(base_rows)
        baseline_errors = _error_by_source(base_rows)
        if base_rows:
            baseline_method = str(base_rows[0].get("method", "baseline"))

    current_errors = _error_by_source(rows)
    paired = paired_bootstrap_delta(baseline_errors, current_errors) if baseline_errors else {}

    for scope, fields in grouped:
        for row in group_errors(rows, fields):
            result = {
                "method": row.get("method", args.method),
                "evaluation_regime": row.get("evaluation_regime", args.evaluation_regime),
                "metric_scope": scope,
                "dataset": row.get("dataset", ""),
                "site": row.get("site", ""),
                "recording": row.get("recording", ""),
                "scene_id": row.get("scene_id", ""),
                "camera_id": row.get("camera_id", ""),
                "spatial_slice": row.get("spatial_slice", ""),
                "distance_bin": row.get("distance_bin", ""),
                "n": row.get("n", 0),
                "matched_count": row.get("matched_count", 0),
                "rmse_m": row.get("rmse_m", ""),
                "mae_m": row.get("mae_m", ""),
                "median_m": row.get("median_m", ""),
                "p90_m": row.get("p90_m", ""),
                "p95_m": row.get("p95_m", ""),
                "max_m": row.get("max_m", ""),
                "paired_n": "",
                "baseline_method": "",
                "delta_rmse_m": "",
                "ci95_low_m": "",
                "ci95_high_m": "",
            }
            if scope == "overall" and paired:
                result.update(
                    {
                        "paired_n": paired.get("paired_n", ""),
                        "baseline_method": baseline_method,
                        "delta_rmse_m": paired.get("delta_rmse_m", ""),
                        "ci95_low_m": paired.get("ci95_low_m", ""),
                        "ci95_high_m": paired.get("ci95_high_m", ""),
                    }
                )
            out.append(result)
    return out


def _load_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    if args.predictions_csv:
        rows = load_prediction_csv(args.predictions_csv, require_gt=True)
        return rows, sorted({str(r.get("scene_id", "")) for r in rows if r.get("scene_id")})
    conn = connect_db(args.db)
    try:
        scenes = selected_scenes(conn, args.batch, args.scene)
        gt_source, gt_event = load_gt_index(args.gt_index)
        rows = load_db_prediction_rows(
            conn,
            scenes=scenes,
            batch_id=args.batch,
            method=args.method,
            evaluation_regime=args.evaluation_regime,
            world_coords=args.world_coords,
            dataset=args.dataset,
            gt_index_by_source=gt_source,
            gt_index_by_event=gt_event,
            require_gt=True,
            class_ids={int(x) for x in split_csv_arg(args.class_ids)} if args.class_ids else None,
        )
        return rows, scenes
    finally:
        conn.close()


def evaluate(args: argparse.Namespace) -> None:
    out_root = ensure_dir(Path(args.out_dir))
    pred_dir = ensure_dir(out_root / "comparability" / "predictions_by_method")
    loc_dir = ensure_dir(out_root / "comparability")
    ivc_dir = ensure_dir(out_root / "iv_c")

    rows, scenes = _load_rows(args)
    rows = _filter_rows(rows, args)
    rows = assign_spatial_slices(
        rows,
        slice_rules=load_slice_rules(args.slice_csv),
        handoff_window_sec=args.handoff_window_sec,
        infer_from_tracks=args.infer_spatial_slices,
    )
    _annotate_distance(rows)

    if not rows:
        raise SystemExit("No localization rows with GT were available.")

    method_slug = args.method.lower().replace(" ", "_").replace("-", "_")
    pred_path = write_csv(pred_dir / f"predictions_{method_slug}.csv", rows, CANONICAL_PREDICTION_FIELDS)
    metric_rows = _metric_rows(rows, args)
    loc_path = upsert_csv_rows(
        loc_dir / "localization_by_method.csv",
        metric_rows,
        LOCALIZATION_FIELDS,
        key_fields=[
            "method",
            "evaluation_regime",
            "metric_scope",
            "dataset",
            "site",
            "recording",
            "scene_id",
            "camera_id",
            "spatial_slice",
            "distance_bin",
        ],
    )
    write_csv(ivc_dir / f"localization_{method_slug}.csv", metric_rows, LOCALIZATION_FIELDS)
    write_json(
        ivc_dir / f"localization_{method_slug}_summary.json",
        {
            "method": args.method,
            "world_coords": args.world_coords,
            "prediction_rows": len(rows),
            "scenes": scenes,
            "predictions_csv": str(pred_path),
            "metrics_csv": str(loc_path),
        },
    )
    write_run_metadata(out_root, "eval_fab2_localization", args, {"scenes": scenes, "prediction_rows": len(rows)})
    print(f"Localization predictions -> {pred_path}")
    print(f"Localization metrics -> {loc_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--db", help="Trajectory SQLite DB")
    src.add_argument("--predictions-csv", help="Canonical predictions CSV")
    parser.add_argument("--batch", default="", help="Batch id when reading DB")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--gt-index", default=None, help="manifest/gt_index.csv")
    parser.add_argument("--method", default="Baseline-IPM")
    parser.add_argument("--evaluation-regime", default="A:FROZEN")
    parser.add_argument("--world-coords", choices=("p_geo", "final", "auto"), default="p_geo")
    parser.add_argument("--split", default=None, help="Optional split filter, e.g. test")
    parser.add_argument("--class-ids", default="", help="Comma-separated class IDs")
    parser.add_argument("--slice-csv", default=None)
    parser.add_argument("--infer-spatial-slices", action="store_true", help="Diagnostic only: infer slices from test GT/predictions when no preregistered slice is present")
    parser.add_argument("--handoff-window-sec", type=float, default=1.0)
    parser.add_argument("--baseline-predictions", default=None, help="Optional baseline predictions CSV for paired CI")
    parser.add_argument("--out-dir", default=str(FAB2_DEFAULT_RESULTS))
    args = parser.parse_args()
    if args.split is not None:
        args.split = str(args.split).strip()
        if not args.split:
            parser.error("--split must be non-empty when provided")
    if args.db and not args.batch:
        parser.error("--batch is required with --db")
    evaluate(args)


if __name__ == "__main__":
    main()
