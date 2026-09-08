#!/usr/bin/env python3
# Vendored HSG-SRL artifact copy. Isolated from the development repo;
# edit this file only. See src/SOURCE_MAP.csv.
"""Evaluate FAB2 tracking metrics from canonical predictions or trajectory DB.

This is a diagnostic ``hota_proxy`` (greedy ID assignment, no LocA).
Table IV BEV-HOTA / LocA is ``evaluation/eval_fab2_bev_hota.py``.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
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
    finite_int,
    load_db_prediction_rows,
    load_gt_index,
    load_prediction_csv,
    load_slice_rules,
    read_csv_rows,
    selected_scenes,
    split_csv_arg,
    time_key,
    upsert_csv_rows,
    write_csv,
    write_json,
    write_run_metadata,
)


TRACKING_FIELDS = [
    "method",
    "evaluation_regime",
    "metric_scope",
    "dataset",
    "site",
    "recording",
    "scene_id",
    "spatial_slice",
    "distance_bin",
    "hota_proxy",
    "deta",
    "assa_proxy",
    "idf1",
    "mota",
    "id_switches",
    "fragments",
    "tp",
    "fp",
    "fn",
    "gt_count",
    "pred_count",
    "match_distance_m",
    "match_mode",
]

EVENT_FIELDS = [
    "dataset",
    "site",
    "recording",
    "scene_id",
    "timestamp",
    "camera_id",
    "source_detection_id",
    "matched_gt_id",
    "global_track_id",
    "distance_m",
    "is_tp",
    "match_mode",
    "spatial_slice",
    "distance_bin",
]


def _row_distance(row: dict[str, Any]) -> float | None:
    px, py = finite_float(row.get("world_x")), finite_float(row.get("world_y"))
    gx, gy = finite_float(row.get("gt_world_x")), finite_float(row.get("gt_world_y"))
    if None in (px, py, gx, gy):
        return None
    return math.hypot(float(px) - float(gx), float(py) - float(gy))


def _filter_prediction_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    class_ids = {int(x) for x in split_csv_arg(args.class_ids)} if args.class_ids else None
    out: list[dict[str, Any]] = []
    for row in rows:
        if args.split is not None and str(row.get("split", "")) != args.split:
            continue
        if class_ids is not None:
            cid = finite_int(row.get("class_id"))
            if cid not in class_ids:
                continue
        out.append(row)
    return out


def _load_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    if args.predictions_csv:
        rows = load_prediction_csv(args.predictions_csv, require_gt=False)
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
            require_gt=False,
            class_ids={int(x) for x in split_csv_arg(args.class_ids)} if args.class_ids else None,
        )
        return rows, scenes
    finally:
        conn.close()


def _gt_rows(args: argparse.Namespace, scenes: set[str]) -> list[dict[str, Any]]:
    if not args.gt_index:
        return []
    class_ids = {int(x) for x in split_csv_arg(args.class_ids)} if args.class_ids else None
    out: list[dict[str, Any]] = []
    for row in read_csv_rows(args.gt_index):
        if scenes and row.get("scene_id") not in scenes:
            continue
        if args.split is not None and str(row.get("split", "")) != args.split:
            continue
        if class_ids is not None:
            cid = finite_int(row.get("class_id"))
            if cid not in class_ids:
                continue
        row = dict(row)
        if not str(row.get("spatial_slice", "")).strip():
            row["spatial_slice"] = "unregistered"
        row["distance_bin"] = _gt_distance_bin(row)
        out.append(row)
    return out


def _gt_distance_bin(row: dict[str, Any]) -> str:
    return distance_bin(row)


def _gt_event_id(row: dict[str, Any]) -> str:
    return str(row.get("source_detection_id") or row.get("event_key") or "")


def _pred_event_id(row: dict[str, Any]) -> str:
    return str(row.get("source_detection_id") or row.get("event_key") or "")


def _is_tp_row(
    row: dict[str, Any],
    args: argparse.Namespace,
    *,
    gt_event_ids: set[str] | None = None,
) -> bool:
    if not str(row.get("matched_gt_id", "")):
        return False
    if args.match_mode == "source":
        if gt_event_ids:
            event_id = _pred_event_id(row)
            return bool(event_id and event_id in gt_event_ids)
        return True
    dist = _row_distance(row)
    return dist is not None and dist <= args.match_distance_m


def _group_key(row: dict[str, Any], fields: list[str]) -> tuple[Any, ...]:
    return tuple(row.get(f, "") for f in fields)


def _greedy_id_assignment(counts: dict[tuple[str, str], int]) -> tuple[int, dict[str, str]]:
    used_pred: set[str] = set()
    used_gt: set[str] = set()
    mapping: dict[str, str] = {}
    idtp = 0
    for (pred, gt), count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        if pred in used_pred or gt in used_gt:
            continue
        used_pred.add(pred)
        used_gt.add(gt)
        mapping[pred] = gt
        idtp += int(count)
    return idtp, mapping


def _count_id_switches(tp_rows: list[dict[str, Any]]) -> int:
    by_gt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tp_rows:
        gt = str(row.get("matched_gt_id", ""))
        if gt:
            by_gt[gt].append(row)
    switches = 0
    for items in by_gt.values():
        items.sort(key=lambda r: (float(r.get("timestamp", 0.0)), str(r.get("camera_id", ""))))
        prev = None
        for row in items:
            pred = str(row.get("global_track_id", ""))
            if not pred:
                continue
            if prev is not None and pred != prev:
                switches += 1
            prev = pred
    return switches


def _count_fragments(tp_rows: list[dict[str, Any]], gt_rows: list[dict[str, Any]], *, time_quantum_sec: float) -> int:
    matched = {_pred_event_id(r) for r in tp_rows}
    by_gt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gt_rows:
        gt = str(row.get("gt_global_id") or row.get("matched_gt_id") or "")
        if gt:
            by_gt[gt].append(row)
    fragments = 0
    for items in by_gt.values():
        items.sort(key=lambda r: (float(r.get("timestamp", 0.0) or 0.0), str(r.get("camera_id", ""))))
        seen_segment = False
        in_segment = False
        for row in items:
            is_match = _gt_event_id(row) in matched
            if is_match and not in_segment:
                if seen_segment:
                    fragments += 1
                seen_segment = True
                in_segment = True
            elif not is_match:
                in_segment = False
    return fragments


def _metrics_for_group(
    rows: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    pred_count = len(rows)
    gt_event_ids = {_gt_event_id(r) for r in gt_rows if _gt_event_id(r)}
    gt_count = len(gt_event_ids)
    if gt_count == 0:
        matched_ids = {_pred_event_id(r) for r in rows if str(r.get("matched_gt_id", ""))}
        gt_count = len(matched_ids)

    tp_rows: list[dict[str, Any]] = []
    fp = 0
    for row in rows:
        if _is_tp_row(row, args, gt_event_ids=gt_event_ids):
            tp_rows.append(row)
        else:
            fp += 1
    tp = len(tp_rows)
    matched_gt_events = {_pred_event_id(r) for r in tp_rows if _pred_event_id(r)}
    fn = max(0, gt_count - len(matched_gt_events))

    denom_det = tp + fp + fn
    deta = tp / denom_det if denom_det else 0.0

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in tp_rows:
        pred = str(row.get("global_track_id", ""))
        gt = str(row.get("matched_gt_id", ""))
        if pred and gt:
            counts[(pred, gt)] += 1
    idtp, _mapping = _greedy_id_assignment(counts)
    idfp = max(0, pred_count - idtp)
    idfn = max(0, gt_count - idtp)
    idf1 = (2 * idtp) / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) else 0.0
    assa = idtp / (idtp + idfp + idfn) if (idtp + idfp + idfn) else 0.0
    idsw = _count_id_switches(tp_rows)
    fragments = _count_fragments(tp_rows, gt_rows, time_quantum_sec=args.time_quantum_sec) if gt_rows else 0
    mota = 1.0 - ((fn + fp + idsw) / gt_count) if gt_count else 0.0
    hota = math.sqrt(max(0.0, deta) * max(0.0, assa))
    return {
        "hota_proxy": hota,
        "deta": deta,
        "assa_proxy": assa,
        "idf1": idf1,
        "mota": mota,
        "id_switches": idsw,
        "fragments": fragments,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_count": gt_count,
        "pred_count": pred_count,
    }


def _event_rows(rows: list[dict[str, Any]], args: argparse.Namespace, gt_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    gt_event_ids = {_gt_event_id(r) for r in gt_rows if _gt_event_id(r)}
    for row in rows:
        dist = _row_distance(row)
        is_tp = int(_is_tp_row(row, args, gt_event_ids=gt_event_ids))
        out.append(
            {
                "dataset": row.get("dataset", ""),
                "site": row.get("site", ""),
                "recording": row.get("recording", ""),
                "scene_id": row.get("scene_id", ""),
                "timestamp": row.get("timestamp", ""),
                "camera_id": row.get("camera_id", ""),
                "source_detection_id": row.get("source_detection_id", ""),
                "matched_gt_id": row.get("matched_gt_id", ""),
                "global_track_id": row.get("global_track_id", ""),
                "distance_m": dist if dist is not None else "",
                "is_tp": is_tp,
                "match_mode": args.match_mode,
                "spatial_slice": row.get("spatial_slice", ""),
                "distance_bin": row.get("distance_bin", ""),
            }
        )
    return out


def _metric_rows(rows: list[dict[str, Any]], gt_rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    scopes = [
        ("overall", ["method", "evaluation_regime"]),
        ("per_scene", ["method", "evaluation_regime", "dataset", "site", "recording", "scene_id"]),
        ("spatial_slice", ["method", "evaluation_regime", "dataset", "site", "recording", "scene_id", "spatial_slice"]),
        ("distance_bin", ["method", "evaluation_regime", "dataset", "site", "recording", "scene_id", "distance_bin"]),
    ]
    out: list[dict[str, Any]] = []
    for scope, fields in scopes:
        row_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        gt_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            row_groups[_group_key(row, fields)].append(row)
        for gt in gt_rows:
            # GT rows do not have method/evaluation fields; blank them to match
            # the prediction grouping key positions.
            gt_aug = dict(gt)
            gt_aug.setdefault("method", rows[0].get("method", args.method) if rows else args.method)
            gt_aug.setdefault("evaluation_regime", rows[0].get("evaluation_regime", args.evaluation_regime) if rows else args.evaluation_regime)
            gt_groups[_group_key(gt_aug, fields)].append(gt_aug)
        for key in sorted(row_groups):
            sample = row_groups[key][0]
            metrics = _metrics_for_group(row_groups[key], gt_groups.get(key, []), args)
            out.append(
                {
                    "method": sample.get("method", args.method),
                    "evaluation_regime": sample.get("evaluation_regime", args.evaluation_regime),
                    "metric_scope": scope,
                    "dataset": sample.get("dataset", ""),
                    "site": sample.get("site", ""),
                    "recording": sample.get("recording", ""),
                    "scene_id": sample.get("scene_id", ""),
                    "spatial_slice": sample.get("spatial_slice", ""),
                    "distance_bin": sample.get("distance_bin", ""),
                    "match_distance_m": args.match_distance_m,
                    "match_mode": args.match_mode,
                    **metrics,
                }
            )
    return out


def evaluate(args: argparse.Namespace) -> None:
    out_root = ensure_dir(Path(args.out_dir))
    pred_dir = ensure_dir(out_root / "comparability" / "predictions_by_method")
    comp_dir = ensure_dir(out_root / "comparability")
    ivc_dir = ensure_dir(out_root / "iv_c")

    rows, scenes = _load_rows(args)
    rows = _filter_prediction_rows(rows, args)
    rows = assign_spatial_slices(
        rows,
        slice_rules=load_slice_rules(args.slice_csv),
        handoff_window_sec=args.handoff_window_sec,
        infer_from_tracks=args.infer_spatial_slices,
    )
    for row in rows:
        row["distance_bin"] = distance_bin(row)

    if not rows:
        raise SystemExit("No tracking prediction rows were available.")
    gt_rows = _gt_rows(args, set(scenes))

    method_slug = args.method.lower().replace(" ", "_").replace("-", "_")
    pred_path = write_csv(pred_dir / f"predictions_{method_slug}.csv", rows, CANONICAL_PREDICTION_FIELDS)
    events_path = write_csv(comp_dir / f"tracking_events_{method_slug}.csv", _event_rows(rows, args, gt_rows), EVENT_FIELDS)
    metric_rows = _metric_rows(rows, gt_rows, args)
    tracking_path = upsert_csv_rows(
        comp_dir / "tracking_by_method.csv",
        metric_rows,
        TRACKING_FIELDS,
        key_fields=[
            "method",
            "evaluation_regime",
            "metric_scope",
            "dataset",
            "site",
            "recording",
            "scene_id",
            "spatial_slice",
            "distance_bin",
            "match_mode",
            "match_distance_m",
        ],
    )
    write_csv(ivc_dir / f"tracking_{method_slug}.csv", metric_rows, TRACKING_FIELDS)
    write_json(
        ivc_dir / f"tracking_{method_slug}_summary.json",
        {
            "method": args.method,
            "world_coords": args.world_coords,
            "match_mode": args.match_mode,
            "metric_note": (
                "hota_proxy is sqrt(DetA * AssA_proxy), not Table IV. "
                "Use eval_fab2_bev_hota.py for Hungarian BEV HOTA/LocA."
            ),
            "prediction_rows": len(rows),
            "gt_rows": len(gt_rows),
            "scenes": scenes,
            "predictions_csv": str(pred_path),
            "events_csv": str(events_path),
            "metrics_csv": str(tracking_path),
        },
    )
    write_run_metadata(out_root, "eval_fab2_tracking", args, {"scenes": scenes, "prediction_rows": len(rows)})
    print(f"Tracking events -> {events_path}")
    print(f"Tracking metrics -> {tracking_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--db")
    src.add_argument("--predictions-csv")
    parser.add_argument("--batch", default="")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--gt-index", default=None)
    parser.add_argument("--method", default="Baseline-IPM")
    parser.add_argument("--evaluation-regime", default="A:FROZEN")
    parser.add_argument("--world-coords", choices=("p_geo", "final", "auto"), default="p_geo")
    parser.add_argument("--split", default=None)
    parser.add_argument("--class-ids", default="")
    parser.add_argument("--slice-csv", default=None)
    parser.add_argument("--infer-spatial-slices", action="store_true", help="Diagnostic only: infer slices from test GT/predictions when no preregistered slice is present")
    parser.add_argument("--handoff-window-sec", type=float, default=1.0)
    parser.add_argument("--time-quantum-sec", type=float, default=0.05)
    parser.add_argument("--match-distance-m", type=float, default=3.0)
    parser.add_argument(
        "--match-mode",
        choices=("source", "distance"),
        default="source",
        help="source uses matched_gt_id/source_detection_id; distance also requires localization distance <= threshold.",
    )
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
