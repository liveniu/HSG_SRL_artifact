#!/usr/bin/env python3
# Vendored HSG-SRL artifact copy. Isolated from the development repo;
# edit this file only. See src/SOURCE_MAP.csv.
"""Evaluate FAB2 MVC mining quality from trajectory SQLite data."""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from collections import defaultdict
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
    load_gt_index,
    scene_camera_names,
    selected_scenes,
    split_csv_arg,
    stats_from_errors,
    time_key,
    write_csv,
    write_json,
    write_run_metadata,
)
from pipeline.geometry_guided_residual_mlp import (  # noqa: E402
    DEFAULT_ANCHOR_ASPECT_RATIO_MAX,
    DEFAULT_ANCHOR_ASPECT_RATIO_MIN,
    DEFAULT_ANCHOR_SHAPE_ETA,
    GeometryDataView,
    MVCPair,
    _filter_observations_for_training_protocol,
    _gt_identity_leakage_report,
    _infer_and_calib_wh_from_db,
    _load_training_split_filter,
    _parse_overlap_pairs,
)
from pipeline.multi_camera_trajectory_fusion import (  # noqa: E402
    RISK_SCALAR_MODES,
    adapt_homography_to_resolution,
)
from utils.scene_geometry_db import load_scene_overlap_pairs  # noqa: E402


AUDIT_FIELDS = [
    "dataset",
    "site",
    "recording",
    "scene_id",
    "batch_id",
    "camera_count",
    "cameras",
    "overlap_pairs",
    "total_db_observations",
    "feature_observations",
    "invalid_or_clipped_observations",
    "nearest_raw_pairs",
    "linear_raw_pairs",
    "distance_qualified_pairs",
    "teacher_qualified_pairs",
    "accepted_pairs",
    "unique_pair_events",
    "eligible_unique_events",
    "event_coverage",
    "mvc_precision",
    "audit_gt_source",
    "gt_identity_compared",
    "gt_identity_match_fraction",
    "accepted_mean_dist_m",
    "accepted_p90_dist_m",
    "accepted_p95_dist_m",
]

EDGE_FIELDS = [
    "dataset",
    "site",
    "recording",
    "scene_id",
    "batch_id",
    "camera_a",
    "camera_b",
    "raw_pairs",
    "accepted_pairs",
    "accepted_percent",
    "unique_pair_events",
    "mvc_precision",
    "mean_dist_m",
    "p90_dist_m",
    "p95_dist_m",
]

MOTION_FIELDS = [
    "dataset",
    "site",
    "recording",
    "scene_id",
    "batch_id",
    "motion_compensation",
    "pair_count",
    "mean_time_gap_sec",
    "p90_time_gap_sec",
    "mean_pair_dist_m",
    "p90_pair_dist_m",
    "p95_pair_dist_m",
]

REJECT_FIELDS = [
    "dataset",
    "site",
    "recording",
    "scene_id",
    "batch_id",
    "stage",
    "count",
    "percent_of_previous",
]


def _pair_dist(pair: MVCPair) -> float:
    ax, ay = pair.obs_a.world_geo
    bx, by = pair.obs_b.world_geo
    return math.hypot(float(ax) - float(bx), float(ay) - float(by))


def _pair_time_gap(pair: MVCPair) -> float:
    return abs(float(pair.obs_a.timestamp_sec) - float(pair.obs_b.timestamp_sec))


def _teacher_select(pairs: list[MVCPair], *, min_conf: float, min_gap: float) -> list[MVCPair]:
    return [
        p
        for p in pairs
        if max(float(p.obs_a.horizon_confidence), float(p.obs_b.horizon_confidence)) >= min_conf
        and abs(float(p.obs_a.horizon_confidence) - float(p.obs_b.horizon_confidence)) >= min_gap
    ]


def _pair_learning_score(pair: MVCPair) -> float:
    ka = float(pair.obs_a.horizon_confidence)
    kb = float(pair.obs_b.horizon_confidence)
    return max(ka, kb) * (0.25 + abs(ka - kb)) + 0.10 * min(ka, kb)


def _top_learning_pairs(pairs: list[MVCPair], max_pairs: int | None) -> list[MVCPair]:
    ranked = sorted(pairs, key=_pair_selection_key)
    if max_pairs is None or len(ranked) <= max_pairs:
        return ranked
    return ranked[:max_pairs]


def _pair_selection_key(pair: MVCPair) -> tuple[Any, ...]:
    scene = str(pair.obs_a.scene_id or pair.obs_b.scene_id or "")
    t_mid = 0.5 * (float(pair.obs_a.timestamp_sec) + float(pair.obs_b.timestamp_sec))
    cam_a, cam_b = sorted((str(pair.obs_a.camera_name), str(pair.obs_b.camera_name)))
    return (
        -_pair_learning_score(pair),
        scene,
        round(t_mid, 6),
        cam_a,
        cam_b,
        int(pair.obs_a.local_track_id),
        int(pair.obs_b.local_track_id),
        int(pair.obs_a.frame_index or -1),
        int(pair.obs_b.frame_index or -1),
    )


def _distance_select(pairs: list[MVCPair], max_dist_m: float | None) -> list[MVCPair]:
    if max_dist_m is None:
        return list(pairs)
    return [p for p in pairs if _pair_dist(p) <= max_dist_m]


def _pair_edge(pair: MVCPair) -> tuple[str, str]:
    return tuple(sorted((pair.obs_a.camera_name, pair.obs_b.camera_name)))  # type: ignore[return-value]


def _gt_maps_from_db_or_index(
    conn: sqlite3.Connection,
    scene_id: str,
    batch_id: str,
    gt_index: str | None,
    *,
    allow_db_identity_audit: bool = False,
) -> tuple[dict[tuple[str, str, int, int], str], dict[tuple[str, str, int], list[tuple[float, str]]], str]:
    by_exact: dict[tuple[str, str, int, int], str] = {}
    by_track: dict[tuple[str, str, int], list[tuple[float, str]]] = defaultdict(list)
    by_source, by_event = load_gt_index(gt_index)
    if by_source or by_event:
        for row in list(by_source.values()) + list(by_event.values()):
            if row.get("scene_id") != scene_id:
                continue
            cam = row.get("camera_id", "")
            local = row.get("local_track_id", "")
            t = finite_float(row.get("timestamp"))
            gt = row.get("gt_global_id", "")
            if not cam or local == "" or t is None or not gt:
                continue
            key = (scene_id, cam, int(float(local)), time_key(t))
            by_exact[key] = str(gt)
            by_track[(scene_id, cam, int(float(local)))].append((t, str(gt)))
        return by_exact, by_track, "gt_index"

    if not allow_db_identity_audit:
        return by_exact, by_track, "none"

    rows = conn.execute(
        """
        SELECT camera_name, local_track_id, timestamp_sec, global_id
          FROM trajectory_observations
         WHERE scene_id=? AND batch_id=?
        """,
        (scene_id, batch_id),
    ).fetchall()
    for r in rows:
        gt = str(r["global_id"])
        cam = str(r["camera_name"])
        local = int(r["local_track_id"])
        t = float(r["timestamp_sec"])
        by_exact[(scene_id, cam, local, time_key(t))] = gt
        by_track[(scene_id, cam, local)].append((t, gt))
    return by_exact, by_track, "db_global_id_not_protocol_gt"


def _gt_for_obs(obs: Any, by_exact: dict[tuple[str, str, int, int], str], by_track: dict[tuple[str, str, int], list[tuple[float, str]]]) -> str:
    scene_id = str(getattr(obs, "scene_id", "") or "")
    cam = str(obs.camera_name)
    local = int(obs.local_track_id)
    t = float(obs.timestamp_sec)
    exact = by_exact.get((scene_id, cam, local, time_key(t)))
    if exact is not None:
        return exact
    candidates = by_track.get((scene_id, cam, local), [])
    if not candidates:
        return ""
    best_t, best_gt = min(candidates, key=lambda item: abs(item[0] - t))
    if abs(best_t - t) <= 0.20:
        return best_gt
    return ""


def _pair_precision(
    pairs: list[MVCPair],
    by_exact: dict[tuple[str, str, int, int], str],
    by_track: dict[tuple[str, str, int], list[tuple[float, str]]],
) -> float | None:
    if not pairs:
        return None
    ok = 0
    seen = 0
    for pair in pairs:
        ga = _gt_for_obs(pair.obs_a, by_exact, by_track)
        gb = _gt_for_obs(pair.obs_b, by_exact, by_track)
        if ga == "" or gb == "":
            continue
        seen += 1
        ok += int(ga == gb)
    if seen == 0:
        return None
    return ok / seen


def _unique_events(
    pairs: list[MVCPair],
    by_exact: dict[tuple[str, str, int, int], str],
    by_track: dict[tuple[str, str, int], list[tuple[float, str]]],
) -> set[tuple[str, str, int]]:
    out: set[tuple[str, str, int]] = set()
    for pair in pairs:
        gt = _gt_for_obs(pair.obs_a, by_exact, by_track)
        if not gt:
            continue
        scene = str(pair.obs_a.scene_id or pair.obs_b.scene_id or "")
        t = 0.5 * (float(pair.obs_a.timestamp_sec) + float(pair.obs_b.timestamp_sec))
        out.add((scene, gt, time_key(t)))
    return out


def _eligible_events_from_observations(
    observations: list[Any],
    overlap_pairs: set[tuple[str, str]],
    by_exact: dict[tuple[str, str, int, int], str],
    by_track: dict[tuple[str, str, int], list[tuple[float, str]]],
) -> set[tuple[str, str, int]]:
    by_gt_time: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for obs in observations:
        gt = _gt_for_obs(obs, by_exact, by_track)
        if not gt:
            continue
        by_gt_time[(str(obs.scene_id), gt, time_key(float(obs.timestamp_sec)))].add(str(obs.camera_name))
    out: set[tuple[str, str, int]] = set()
    overlap_set = {tuple(sorted(p)) for p in overlap_pairs}
    for (scene, gid, tk), cams in by_gt_time.items():
        cams_list = sorted(cams)
        is_overlap = any(tuple(sorted((a, b))) in overlap_set for i, a in enumerate(cams_list) for b in cams_list[i + 1 :])
        if is_overlap:
            out.add((scene, str(gid), tk))
    return out


def _motion_row(labels: Any, scene_id: str, batch_id: str, mode: str, pairs: list[MVCPair]) -> dict[str, Any]:
    dists = [_pair_dist(p) for p in pairs]
    gaps = [_pair_time_gap(p) for p in pairs]
    dstat = stats_from_errors(dists)
    gstat = stats_from_errors(gaps)
    return {
        "dataset": labels.dataset,
        "site": labels.site,
        "recording": labels.recording,
        "scene_id": scene_id,
        "batch_id": batch_id,
        "motion_compensation": mode,
        "pair_count": len(pairs),
        "mean_time_gap_sec": gstat["mae_m"],
        "p90_time_gap_sec": gstat["p90_m"],
        "mean_pair_dist_m": dstat["mae_m"],
        "p90_pair_dist_m": dstat["p90_m"],
        "p95_pair_dist_m": dstat["p95_m"],
    }


def _load_scene_observations(
    conn: sqlite3.Connection,
    scene_id: str,
    args: argparse.Namespace,
) -> tuple[list[Any], set[tuple[str, str]], list[str], int]:
    view = GeometryDataView()
    explicit_cameras = split_csv_arg(args.cameras)
    cameras = explicit_cameras or scene_camera_names(conn, scene_id, args.batch)
    calibrations = view.load_camera_calibrations(conn, scene_id, cameras)
    db_infer_wh, db_calib_wh = _infer_and_calib_wh_from_db(conn, scene_id, args.batch, cameras)
    frame_wh = (int(args.frame_width), int(args.frame_height))
    frame_wh_by_camera = {cam: db_infer_wh.get(cam) or frame_wh for cam in cameras}
    adapted: dict[str, tuple[np.ndarray, float]] = {}
    for cam, (h_cal, scale) in calibrations.items():
        infer_wh = frame_wh_by_camera.get(cam)
        calib_wh = db_calib_wh.get(cam)
        if infer_wh and calib_wh and infer_wh != calib_wh:
            h_cal = adapt_homography_to_resolution(h_cal, calib_wh, infer_wh)
        adapted[cam] = (h_cal, scale)
    observations = list(
        view.iter_observations_from_db(
            conn,
            scene_id,
            args.batch,
            cameras,
            adapted,
            frame_wh_by_camera=frame_wh_by_camera,
            risk_scalar_mode=args.risk_scalar,
            s0_m=args.s0_m,
            feature_l0_m=args.feature_l0_m,
            pixel_grazing_d0=args.pixel_grazing_d0,
            anchor_mode=args.anchor_mode,
            anchor_shape_eta=args.anchor_shape_eta,
            anchor_aspect_ratio_min=args.anchor_aspect_ratio_min,
            anchor_aspect_ratio_max=args.anchor_aspect_ratio_max,
            sensitivity_clip_m=args.sensitivity_clip_m,
            extent_clip_m=args.extent_clip_m,
            clip_margin_px=args.clip_margin,
            clip_margin_ratio=args.clip_margin_ratio,
        )
    )
    if args.overlap_pairs:
        overlap = _parse_overlap_pairs(args.overlap_pairs)
    else:
        overlap = load_scene_overlap_pairs(conn, scene_id, cameras)
    if not overlap:
        overlap = GeometryDataView.default_overlap_pairs(cameras)
    total_db = int(
        conn.execute(
            f"SELECT COUNT(*) FROM trajectory_observations WHERE scene_id=? AND batch_id=? "
            f"AND camera_name IN ({','.join('?' for _ in cameras)})",
            (scene_id, args.batch, *cameras),
        ).fetchone()[0]
    )
    return observations, overlap, cameras, total_db


def evaluate(args: argparse.Namespace) -> None:
    if args.max_pairs is not None and args.max_pairs < 1:
        raise SystemExit("--max-pairs must be >= 1 when provided")
    if args.max_pair_dist_m is not None and args.max_pair_dist_m <= 0:
        raise SystemExit("--max-pair-dist-m must be > 0")
    if not split_csv_arg(args.audit_splits):
        raise SystemExit("--audit-splits must contain at least one split name")
    if args.time_start is not None and args.time_end is not None and args.time_start >= args.time_end:
        raise SystemExit("--time-start must be smaller than --time-end")
    if not args.gt_index and not args.allow_db_identity_audit:
        raise SystemExit(
            "--gt-index is required for protocol MVC audit. "
            "DB global_id is the mined pseudo identity and cannot audit itself; "
            "use --allow-db-identity-audit only for legacy diagnostics."
        )
    out_root = ensure_dir(Path(args.out_dir))
    ivb = ensure_dir(out_root / "iv_b")
    conn = connect_db(args.db)
    try:
        scenes = selected_scenes(conn, args.batch, args.scene)
        if not scenes:
            raise SystemExit("No scenes found. Pass --scene or check --batch.")
        split_manifest = args.split_manifest or args.gt_index
        split_filter = _load_training_split_filter(
            split_manifest,
            args.audit_splits,
            scene_ids=scenes,
            batch_id=args.batch,
        )
        if split_filter is not None and split_filter.row_count == 0:
            raise SystemExit(
                "FAB2 audit split manifest contains no rows for "
                f"scene={args.scene or ','.join(scenes)} batch={args.batch} split={','.join(split_filter.split_names)}"
            )
        audit_rows: list[dict[str, Any]] = []
        edge_rows: list[dict[str, Any]] = []
        motion_rows: list[dict[str, Any]] = []
        reject_rows: list[dict[str, Any]] = []
        for scene_id in scenes:
            labels = infer_dataset_labels(scene_id, args.dataset)
            observations, overlap, cameras, total_db = _load_scene_observations(conn, scene_id, args)
            protocol_stats, observations = _filter_observations_for_training_protocol(
                observations,
                scene_id=scene_id,
                batch_id=args.batch,
                split_filter=split_filter,
                time_start=args.time_start,
                time_end=args.time_end,
            )
            leak_report = _gt_identity_leakage_report(
                observations,
                scene_id=scene_id,
                batch_id=args.batch,
                split_filter=split_filter,
            )
            if (
                split_filter is not None
                and not args.allow_gt_identity_mining
                and leak_report["compared"] >= max(1, min(32, len(observations)))
                and leak_report["match_fraction"] >= 0.95
            ):
                raise SystemExit(
                    "FAB2 protocol guard: MVC audit batch appears to be using GT identities "
                    f"for scene={scene_id} ({int(leak_report['matches'])}/"
                    f"{int(leak_report['compared'])} observations have global_id == gt_global_id). "
                    "Run re-merge with pseudo identities before audit/training, or pass "
                    "--allow-gt-identity-mining only for supervised smoke tests."
                )
            by_exact, by_track, gt_source = _gt_maps_from_db_or_index(
                conn,
                scene_id,
                args.batch,
                args.gt_index,
                allow_db_identity_audit=args.allow_db_identity_audit,
            )
            nearest = GeometryDataView.mine_mvc_pairs(
                observations,
                overlap,
                max_time_gap_sec=args.max_time_gap,
                motion_compensation="nearest",
                risk_scalar_mode=args.risk_scalar,
                s0_m=args.s0_m,
                feature_l0_m=args.feature_l0_m,
                pixel_grazing_d0=args.pixel_grazing_d0,
                anchor_mode=args.anchor_mode,
                anchor_shape_eta=args.anchor_shape_eta,
                anchor_aspect_ratio_min=args.anchor_aspect_ratio_min,
                anchor_aspect_ratio_max=args.anchor_aspect_ratio_max,
                sensitivity_clip_m=args.sensitivity_clip_m,
                extent_clip_m=args.extent_clip_m,
            )
            linear = GeometryDataView.mine_mvc_pairs(
                observations,
                overlap,
                max_time_gap_sec=args.max_time_gap,
                motion_compensation=args.motion_compensation,
                risk_scalar_mode=args.risk_scalar,
                s0_m=args.s0_m,
                feature_l0_m=args.feature_l0_m,
                pixel_grazing_d0=args.pixel_grazing_d0,
                anchor_mode=args.anchor_mode,
                anchor_shape_eta=args.anchor_shape_eta,
                anchor_aspect_ratio_min=args.anchor_aspect_ratio_min,
                anchor_aspect_ratio_max=args.anchor_aspect_ratio_max,
                sensitivity_clip_m=args.sensitivity_clip_m,
                extent_clip_m=args.extent_clip_m,
            )
            dist_pairs = _distance_select(linear, args.max_pair_dist_m)
            teacher_pairs = _teacher_select(
                dist_pairs,
                min_conf=args.min_teacher_confidence,
                min_gap=args.min_teacher_confidence_gap,
            )
            accepted = _top_learning_pairs(teacher_pairs, args.max_pairs)
            accepted_dists = [_pair_dist(p) for p in accepted]
            dstat = stats_from_errors(accepted_dists)
            events = _unique_events(accepted, by_exact, by_track)
            eligible_events = _eligible_events_from_observations(observations, overlap, by_exact, by_track)
            precision = _pair_precision(accepted, by_exact, by_track)
            audit_rows.append(
                {
                    "dataset": labels.dataset,
                    "site": labels.site,
                    "recording": labels.recording,
                    "scene_id": scene_id,
                    "batch_id": args.batch,
                    "camera_count": len(cameras),
                    "cameras": cameras,
                    "overlap_pairs": sorted(overlap),
                    "total_db_observations": total_db,
                    "feature_observations": len(observations),
                    "invalid_or_clipped_observations": total_db - protocol_stats["observations_before_protocol_filter"],
                    "nearest_raw_pairs": len(nearest),
                    "linear_raw_pairs": len(linear),
                    "distance_qualified_pairs": len(dist_pairs),
                    "teacher_qualified_pairs": len(teacher_pairs),
                    "accepted_pairs": len(accepted),
                    "unique_pair_events": len(events),
                    "eligible_unique_events": len(eligible_events),
                    "event_coverage": (len(events) / len(eligible_events)) if eligible_events else "",
                    "mvc_precision": precision if precision is not None else "",
                    "audit_gt_source": gt_source,
                    "gt_identity_compared": int(leak_report["compared"]),
                    "gt_identity_match_fraction": leak_report["match_fraction"],
                    "accepted_mean_dist_m": dstat["mae_m"],
                    "accepted_p90_dist_m": dstat["p90_m"],
                    "accepted_p95_dist_m": dstat["p95_m"],
                }
            )

            for stage, prev, count in [
                ("invalid_or_clipped_observations", total_db, total_db - protocol_stats["observations_before_protocol_filter"]),
                (
                    "protocol_split_or_time_rejected",
                    protocol_stats["observations_before_protocol_filter"],
                    protocol_stats["observations_before_protocol_filter"] - len(observations),
                ),
                ("time_or_bracket_rejected", len(nearest), max(0, len(nearest) - len(linear))),
                ("distance_rejected", len(linear), max(0, len(linear) - len(dist_pairs))),
                ("teacher_rejected", len(dist_pairs), max(0, len(dist_pairs) - len(teacher_pairs))),
                ("budget_trimmed", len(teacher_pairs), max(0, len(teacher_pairs) - len(accepted))),
            ]:
                reject_rows.append(
                    {
                        "dataset": labels.dataset,
                        "site": labels.site,
                        "recording": labels.recording,
                        "scene_id": scene_id,
                        "batch_id": args.batch,
                        "stage": stage,
                        "count": count,
                        "percent_of_previous": (count / prev) if prev else "",
                    }
                )

            motion_rows.append(_motion_row(labels, scene_id, args.batch, "nearest", nearest))
            motion_rows.append(_motion_row(labels, scene_id, args.batch, args.motion_compensation, linear))

            raw_by_edge: dict[tuple[str, str], list[MVCPair]] = defaultdict(list)
            acc_by_edge: dict[tuple[str, str], list[MVCPair]] = defaultdict(list)
            for p in linear:
                raw_by_edge[_pair_edge(p)].append(p)
            for p in accepted:
                acc_by_edge[_pair_edge(p)].append(p)
            for edge in sorted(set(raw_by_edge) | set(acc_by_edge) | set(overlap)):
                pairs = acc_by_edge.get(edge, [])
                dists = [_pair_dist(p) for p in pairs]
                st = stats_from_errors(dists)
                eprec = _pair_precision(pairs, by_exact, by_track)
                edge_rows.append(
                    {
                        "dataset": labels.dataset,
                        "site": labels.site,
                        "recording": labels.recording,
                        "scene_id": scene_id,
                        "batch_id": args.batch,
                        "camera_a": edge[0],
                        "camera_b": edge[1],
                        "raw_pairs": len(raw_by_edge.get(edge, [])),
                        "accepted_pairs": len(pairs),
                        "accepted_percent": (len(pairs) / len(accepted)) if accepted else "",
                        "unique_pair_events": len(_unique_events(pairs, by_exact, by_track)),
                        "mvc_precision": eprec if eprec is not None else "",
                        "mean_dist_m": st["mae_m"],
                        "p90_dist_m": st["p90_m"],
                        "p95_dist_m": st["p95_m"],
                    }
                )
            print(
                f"{scene_id}: obs={len(observations)}/{total_db} "
                f"pairs={len(accepted)}/{len(linear)} precision={precision if precision is not None else 'NA'}"
            )
    finally:
        conn.close()

    write_csv(ivb / "mvc_mining_audit.csv", audit_rows, AUDIT_FIELDS)
    write_csv(ivb / "mvc_edge_distribution.csv", edge_rows, EDGE_FIELDS)
    write_csv(ivb / "mvc_motion_compensation_audit.csv", motion_rows, MOTION_FIELDS)
    write_csv(ivb / "mvc_reject_reasons.csv", reject_rows, REJECT_FIELDS)
    write_json(
        ivb / "mvc_mining_summary.json",
        {
            "db": str(args.db),
            "batch_id": args.batch,
            "scenes": scenes,
            "accepted_pairs_total": sum(int(r["accepted_pairs"]) for r in audit_rows),
        },
    )
    write_run_metadata(out_root, "eval_fab2_mining_quality", args, {"scenes": scenes})
    print(f"MVC mining quality written under {ivb}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--out-dir", default=str(FAB2_DEFAULT_RESULTS))
    parser.add_argument("--gt-index", default=None, help="Required manifest/gt_index.csv for protocol MVC precision")
    parser.add_argument("--split-manifest", default=None, help="Optional split manifest; defaults to --gt-index")
    parser.add_argument("--audit-splits", default="train", help="Comma-separated split names to audit from split manifest")
    parser.add_argument("--time-start", type=float, default=None)
    parser.add_argument("--time-end", type=float, default=None)
    parser.add_argument("--allow-db-identity-audit", action="store_true", help="Legacy diagnostic only: audit against DB global_id")
    parser.add_argument("--allow-gt-identity-mining", action="store_true", help="Bypass GT identity leakage guard for supervised smoke tests")
    parser.add_argument("--cameras", default="")
    parser.add_argument("--overlap-pairs", default="")
    parser.add_argument("--frame-width", type=int, default=1920)
    parser.add_argument("--frame-height", type=int, default=1080)
    parser.add_argument("--clip-margin", type=int, default=3)
    parser.add_argument("--clip-margin-ratio", type=float, default=0.01)
    parser.add_argument("--max-time-gap", type=float, default=0.35)
    parser.add_argument("--motion-compensation", choices=("off", "nearest", "linear_interpolate"), default="linear_interpolate")
    parser.add_argument("--max-pair-dist-m", type=float, default=3.0)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional diagnostic cap on retained MVC pairs after teacher filters.",
    )
    parser.add_argument("--min-teacher-confidence", type=float, default=0.20)
    parser.add_argument("--min-teacher-confidence-gap", type=float, default=0.05)
    parser.add_argument("--risk-scalar", choices=RISK_SCALAR_MODES, default="metric_jacobian")
    parser.add_argument("--s0-m", type=float, default=60.0)
    parser.add_argument("--feature-l0-m", type=float, default=10.0)
    parser.add_argument("--pixel-grazing-d0", type=float, default=50.0 / 1080.0)
    parser.add_argument("--anchor-mode", choices=("multiplicative_gating_offsetted", "bottom_center", "center"), default="multiplicative_gating_offsetted")
    parser.add_argument("--anchor-shape-eta", type=float, default=DEFAULT_ANCHOR_SHAPE_ETA)
    parser.add_argument("--anchor-aspect-ratio-min", type=float, default=DEFAULT_ANCHOR_ASPECT_RATIO_MIN)
    parser.add_argument("--anchor-aspect-ratio-max", type=float, default=DEFAULT_ANCHOR_ASPECT_RATIO_MAX)
    parser.add_argument("--sensitivity-clip-m", type=float, default=None)
    parser.add_argument("--extent-clip-m", type=float, default=None)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
