#!/usr/bin/env python3
# Vendored HSG-SRL artifact copy. Isolated from the development repo;
# edit this file only. See src/SOURCE_MAP.csv.
"""Fit label-free fixed-translation projection artifacts for G2 residual training.

This is the production counterpart of the earlier experiment helper. It mines
the same MVC pairs as ``geometry_guided_residual_mlp.py train`` and fits an
accepted-gated, gauge-constrained per-camera 2D translation projection before
the residual MLP is trained.
The same JSON may be passed explicitly at runtime when final coordinates should
include that deterministic translation; checkpoints never auto-apply it.

The tool writes a JSON artifact. It does not create a low-order DB copy and does
not rewrite ``trajectory_observations.world_x/y``; those columns remain raw
``P_geo`` by contract.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.geometry_guided_residual_mlp import (  # noqa: E402
    DEFAULT_ANCHOR_ASPECT_RATIO_MAX,
    DEFAULT_ANCHOR_ASPECT_RATIO_MIN,
    DEFAULT_ANCHOR_SHAPE_ETA,
    GeometryDataView,
    MVCPair,
    _filter_mvc_pairs_by_world_distance,
    _filter_observations_for_training_protocol,
    _gt_identity_leakage_report,
    _infer_and_calib_wh_from_db,
    _load_training_split_filter,
    _parse_overlap_pairs,
    _parse_time_offsets,
)
from pipeline.multi_camera_trajectory_fusion import (  # noqa: E402
    RISK_SCALAR_MODES,
    adapt_homography_to_resolution,
)
from utils.g2_preflight_diagnostic import diagnose_low_order_bias  # noqa: E402
from utils.scene_geometry_db import load_scene_overlap_pairs  # noqa: E402
from utils.trajectory_batches import (  # noqa: E402
    finish_batch_stage,
    require_trajectory_batch_schema,
    start_batch_stage,
)
from utils.project_paths import resolve_project_path  # noqa: E402

# Affine-explained fraction threshold: if the affine model explains this much
# more variance than the fixed-translation projection, the error field has a
# significant linear component that T_c alone cannot address.
_AFFINE_DOMINANCE_MARGIN = 0.15


@dataclass(frozen=True)
class PairMeasurement:
    scene_id: str
    camera_a: str
    camera_b: str
    diff_xy: np.ndarray
    kappa_a: float
    kappa_b: float
    weight: float


def _fmt_xy(xy: np.ndarray | tuple[float, float] | list[float]) -> str:
    return f"[{float(xy[0]):+.4f}, {float(xy[1]):+.4f}]"


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "n": float(values.size),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def _vector_stats(vectors: np.ndarray) -> dict[str, Any]:
    vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 2)
    if len(vectors) == 0:
        return {
            "n": 0,
            "mean_xy": [0.0, 0.0],
            "median_xy": [0.0, 0.0],
            "mean_norm": 0.0,
            "norm": _stats(np.zeros(0, dtype=np.float64)),
        }
    norms = np.linalg.norm(vectors, axis=1)
    return {
        "n": int(len(vectors)),
        "mean_xy": np.mean(vectors, axis=0).tolist(),
        "median_xy": np.median(vectors, axis=0).tolist(),
        "mean_norm": float(np.linalg.norm(np.mean(vectors, axis=0))),
        "norm": _stats(norms),
    }


def _pair_learning_score(pair: MVCPair) -> float:
    ka = float(pair.obs_a.horizon_confidence)
    kb = float(pair.obs_b.horizon_confidence)
    return max(ka, kb) * (0.25 + abs(ka - kb)) + 0.10 * min(ka, kb)


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


def _pair_weight(pair: MVCPair, mode: str) -> float:
    ka = float(pair.obs_a.horizon_confidence)
    kb = float(pair.obs_b.horizon_confidence)
    if mode == "uniform":
        return 1.0
    if mode == "kappa-min":
        return min(ka, kb)
    if mode == "kappa-max":
        return max(ka, kb)
    if mode == "kappa-product":
        return ka * kb
    if mode == "learning-score":
        return _pair_learning_score(pair)
    raise ValueError(f"unknown pair weight mode: {mode}")


def _select_teacher_pairs(
    pairs: list[MVCPair],
    *,
    min_teacher_confidence: float,
    min_teacher_confidence_gap: float,
    max_pairs: int | None,
) -> list[MVCPair]:
    out = [
        p
        for p in pairs
        if max(p.obs_a.horizon_confidence, p.obs_b.horizon_confidence)
        >= min_teacher_confidence
        and abs(p.obs_a.horizon_confidence - p.obs_b.horizon_confidence)
        >= min_teacher_confidence_gap
    ]
    out.sort(key=_pair_selection_key)
    if max_pairs is not None and len(out) > max_pairs:
        out = out[:max_pairs]
    return out


def _scene_camera_names(
    conn: sqlite3.Connection,
    scene_id: str,
    batch_id: str,
    explicit: list[str],
) -> list[str]:
    if explicit:
        return explicit
    return [
        str(r[0])
        for r in conn.execute(
            """
            SELECT DISTINCT camera_name FROM trajectory_observations
             WHERE scene_id=? AND batch_id=? ORDER BY camera_name
            """,
            (scene_id, batch_id),
        ).fetchall()
    ]


def _mine_scene_pairs(
    conn: sqlite3.Connection,
    db_path: Path,
    scene_id: str,
    args: argparse.Namespace,
    split_filter: Any,
) -> tuple[list[MVCPair], dict[str, Any]]:
    view = GeometryDataView()
    frame_wh = (int(args.frame_width), int(args.frame_height))
    explicit_cameras = [s.strip() for s in args.cameras.split(",") if s.strip()]
    camera_names = _scene_camera_names(conn, scene_id, args.batch, explicit_cameras)
    if len(camera_names) < 2:
        return [], {"scene_id": scene_id, "skip": "fewer_than_two_cameras"}

    try:
        calibrations = view.load_camera_calibrations(conn, scene_id, camera_names)
    except RuntimeError as exc:
        return [], {"scene_id": scene_id, "skip": f"calibration_missing: {exc}"}

    db_infer_wh, db_calib_wh = _infer_and_calib_wh_from_db(
        conn, scene_id, args.batch, camera_names
    )
    frame_wh_by_camera = {
        cam: db_infer_wh.get(cam) or frame_wh
        for cam in camera_names
    }

    adapted_calibrations: dict[str, tuple[np.ndarray, float]] = {}
    for cam, (h_cal, scale) in calibrations.items():
        infer_wh_cam = frame_wh_by_camera.get(cam)
        calib_wh_cam = db_calib_wh.get(cam)
        if infer_wh_cam and calib_wh_cam and calib_wh_cam != infer_wh_cam:
            h_cal = adapt_homography_to_resolution(h_cal, calib_wh_cam, infer_wh_cam)
        adapted_calibrations[cam] = (h_cal, scale)

    observations = list(
        view.iter_observations_from_db(
            conn,
            scene_id,
            args.batch,
            camera_names,
            adapted_calibrations,
            frame_wh_by_camera=frame_wh_by_camera,
            risk_scalar_mode=args.risk_scalar,
            s0_m=args.s0_m,
            feature_l0_m=args.feature_l0_m,
            pixel_grazing_d0=args.pixel_grazing_d0,
            anchor_mode=args.anchor_mode,
            anchor_shape_eta=args.anchor_shape_eta,
            anchor_aspect_ratio_min=args.anchor_aspect_ratio_min,
            anchor_aspect_ratio_max=args.anchor_aspect_ratio_max,
            sensitivity_clip_m=args.metric_sensitivity_clip_m,
            extent_clip_m=args.extent_clip_m,
            clip_margin_px=args.clip_margin,
            clip_margin_ratio=args.clip_margin_ratio,
        )
    )
    total_db_obs = int(
        conn.execute(
            f"""
            SELECT COUNT(*) FROM trajectory_observations
             WHERE scene_id=? AND batch_id=? AND camera_name IN ({','.join('?' for _ in camera_names)})
            """,
            (scene_id, args.batch, *camera_names),
        ).fetchone()[0]
    )
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
            "FAB2 protocol guard: low-order MVC mining appears to be using GT identities "
            f"for scene={scene_id} ({int(leak_report['matches'])}/"
            f"{int(leak_report['compared'])} observations have global_id == gt_global_id). "
            "Run re-merge with pseudo identities before fitting, or pass "
            "--allow-gt-identity-mining only for supervised smoke tests."
        )

    if args.overlap_pairs:
        overlap = _parse_overlap_pairs(args.overlap_pairs)
        overlap_source = "--overlap-pairs"
    else:
        overlap = load_scene_overlap_pairs(conn, scene_id, camera_names)
        overlap_source = "db:scene_overlap_pairs"
    if not overlap:
        return [], {
            "scene_id": scene_id,
            "skip": "no_overlap_pairs",
            "hint": "re-import / re-track to fill scene_overlap_pairs, or pass --overlap-pairs",
        }

    raw_pairs = view.mine_mvc_pairs(
        observations,
        overlap,
        max_time_gap_sec=args.max_time_gap,
        motion_compensation=args.motion_compensation,
        time_offsets_sec=_parse_time_offsets(json.loads(args.time_offsets_json) if args.time_offsets_json else {}),
        risk_scalar_mode=args.risk_scalar,
        s0_m=args.s0_m,
        feature_l0_m=args.feature_l0_m,
        pixel_grazing_d0=args.pixel_grazing_d0,
        anchor_mode=args.anchor_mode,
        anchor_shape_eta=args.anchor_shape_eta,
        anchor_aspect_ratio_min=args.anchor_aspect_ratio_min,
        anchor_aspect_ratio_max=args.anchor_aspect_ratio_max,
        sensitivity_clip_m=args.metric_sensitivity_clip_m,
        extent_clip_m=args.extent_clip_m,
    )
    dist_pairs = _filter_mvc_pairs_by_world_distance(raw_pairs, args.max_pair_dist_m)
    selected_pairs = _select_teacher_pairs(
        dist_pairs,
        min_teacher_confidence=args.min_teacher_confidence,
        min_teacher_confidence_gap=args.min_teacher_confidence_gap,
        max_pairs=args.max_pairs,
    )
    pairs = selected_pairs if args.pair_source == "selected" else dist_pairs
    summary = {
        "scene_id": scene_id,
        "camera_names": camera_names,
        "overlap_pairs": [list(p) for p in sorted(overlap)],
        "overlap_source": overlap_source,
        "observations": len(observations),
        "total_db_observations": total_db_obs,
        "protocol_stats": protocol_stats,
        "gt_identity_guard": leak_report,
        "raw_mvc_pairs": len(raw_pairs),
        "distance_filtered_pairs": len(dist_pairs),
        "selected_pairs": len(selected_pairs),
        "used_pairs": len(pairs),
        "frame_wh_by_camera": {cam: list(wh) for cam, wh in frame_wh_by_camera.items()},
    }
    return pairs, summary


def mine_pairs_by_scene(args: argparse.Namespace) -> tuple[dict[str, list[MVCPair]], list[dict[str, Any]]]:
    db_path = resolve_project_path(args.db)
    scene_ids = [s.strip() for s in args.scene.split(",") if s.strip()]
    if not scene_ids:
        raise SystemExit("--scene must not be empty")

    conn = sqlite3.connect(str(db_path))
    try:
        require_trajectory_batch_schema(conn)
        split_filter = _load_training_split_filter(
            args.split_manifest,
            args.train_splits,
            scene_ids=scene_ids,
            batch_id=args.batch,
        )
        if split_filter is not None and split_filter.row_count == 0:
            raise SystemExit(
                "FAB2 split manifest contains no rows for "
                f"scene={args.scene} batch={args.batch} split={','.join(split_filter.split_names)}"
            )
        by_scene: dict[str, list[MVCPair]] = {}
        summaries: list[dict[str, Any]] = []
        for scene_id in scene_ids:
            pairs, summary = _mine_scene_pairs(conn, db_path, scene_id, args, split_filter)
            by_scene[scene_id] = pairs
            summaries.append(summary)
        return by_scene, summaries
    finally:
        conn.close()


def measurements_from_pairs(
    scene_id: str,
    pairs: list[MVCPair],
    *,
    pair_weight_mode: str,
) -> list[PairMeasurement]:
    measurements: list[PairMeasurement] = []
    for pair in pairs:
        diff = (
            np.asarray(pair.obs_a.world_geo, dtype=np.float64)
            - np.asarray(pair.obs_b.world_geo, dtype=np.float64)
        )
        weight = max(float(_pair_weight(pair, pair_weight_mode)), 1e-9)
        measurements.append(
            PairMeasurement(
                scene_id=scene_id,
                camera_a=pair.obs_a.camera_name,
                camera_b=pair.obs_b.camera_name,
                diff_xy=diff,
                kappa_a=float(pair.obs_a.horizon_confidence),
                kappa_b=float(pair.obs_b.horizon_confidence),
                weight=weight,
            )
        )
    return measurements


def _components(cameras: list[str], measurements: list[PairMeasurement]) -> list[list[str]]:
    adj: dict[str, set[str]] = {c: set() for c in cameras}
    for m in measurements:
        adj.setdefault(m.camera_a, set()).add(m.camera_b)
        adj.setdefault(m.camera_b, set()).add(m.camera_a)
    seen: set[str] = set()
    comps: list[list[str]] = []
    for cam in sorted(adj):
        if cam in seen:
            continue
        q: deque[str] = deque([cam])
        seen.add(cam)
        comp: list[str] = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nxt in sorted(adj[cur]):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        comps.append(sorted(comp))
    return comps


def _camera_gauge_weights(
    cameras: list[str],
    measurements: list[PairMeasurement],
    mode: str,
) -> dict[str, float]:
    if mode == "uniform":
        return {cam: 1.0 for cam in cameras}
    values: dict[str, list[float]] = {cam: [] for cam in cameras}
    for m in measurements:
        values.setdefault(m.camera_a, []).append(m.kappa_a)
        values.setdefault(m.camera_b, []).append(m.kappa_b)
    return {
        cam: max(float(np.mean(values.get(cam) or [1.0])), 1e-6)
        for cam in cameras
    }


def fit_translations(
    measurements: list[PairMeasurement],
    *,
    gauge_mode: str,
    gauge_weight_mode: str,
    reference_camera: str | None,
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    cameras = sorted({m.camera_a for m in measurements} | {m.camera_b for m in measurements})
    if not cameras:
        raise ValueError("no MVC pair measurements to fit")
    index = {cam: i for i, cam in enumerate(cameras)}
    a_mat = np.zeros((len(measurements), len(cameras)), dtype=np.float64)
    b_mat = np.zeros((len(measurements), 2), dtype=np.float64)
    weights = np.zeros(len(measurements), dtype=np.float64)
    for row, m in enumerate(measurements):
        a_mat[row, index[m.camera_a]] = 1.0
        a_mat[row, index[m.camera_b]] = -1.0
        b_mat[row] = -m.diff_xy
        weights[row] = m.weight

    gauge_weights = _camera_gauge_weights(cameras, measurements, gauge_weight_mode)
    constraints: list[np.ndarray] = []
    for comp in _components(cameras, measurements):
        row = np.zeros(len(cameras), dtype=np.float64)
        if gauge_mode == "reference" and reference_camera in comp:
            row[index[str(reference_camera)]] = 1.0
        else:
            for cam in comp:
                row[index[cam]] = gauge_weights.get(cam, 1.0)
            denom = float(np.sum(np.abs(row)))
            if denom > 0:
                row /= denom
        constraints.append(row)

    w_col = weights[:, None]
    normal = a_mat.T @ (w_col * a_mat)
    rhs_all = a_mat.T @ (w_col * b_mat)
    c_mat = np.vstack(constraints)
    kkt = np.block(
        [
            [normal, c_mat.T],
            [c_mat, np.zeros((c_mat.shape[0], c_mat.shape[0]), dtype=np.float64)],
        ]
    )

    solution = np.zeros((len(cameras), 2), dtype=np.float64)
    for dim in range(2):
        rhs = np.concatenate([rhs_all[:, dim], np.zeros(c_mat.shape[0], dtype=np.float64)])
        sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
        solution[:, dim] = sol[: len(cameras)]

    corrections = {
        cam: (float(solution[index[cam], 0]), float(solution[index[cam], 1]))
        for cam in cameras
    }
    return corrections, gauge_weights


def corrected_diffs(
    measurements: list[PairMeasurement],
    corrections: dict[str, tuple[float, float]],
) -> np.ndarray:
    out = []
    for m in measurements:
        ca = np.asarray(corrections.get(m.camera_a, (0.0, 0.0)), dtype=np.float64)
        cb = np.asarray(corrections.get(m.camera_b, (0.0, 0.0)), dtype=np.float64)
        out.append(m.diff_xy + ca - cb)
    return np.asarray(out, dtype=np.float64)


def pair_summaries(
    measurements: list[PairMeasurement],
    corrections: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[PairMeasurement]] = defaultdict(list)
    for m in measurements:
        grouped[(m.camera_a, m.camera_b)].append(m)

    out: list[dict[str, Any]] = []
    for (cam_a, cam_b), items in sorted(grouped.items()):
        diffs = np.asarray([m.diff_xy for m in items], dtype=np.float64)
        cdiffs = corrected_diffs(items, corrections)
        before = _vector_stats(diffs)
        after = _vector_stats(cdiffs)
        out.append(
            {
                "camera_a": cam_a,
                "camera_b": cam_b,
                "count": len(items),
                "before": before,
                "after": after,
                "mean_geo_a_minus_geo_b_m": before["mean_xy"],
                "mean_corrected_geo_a_minus_geo_b_m": after["mean_xy"],
                "mean_kappa_a": float(np.mean([m.kappa_a for m in items])),
                "mean_kappa_b": float(np.mean([m.kappa_b for m in items])),
            }
        )
    return out


def _assess_correction(
    before: dict[str, Any],
    after: dict[str, Any],
    pairs_out: list[dict[str, Any]],
    *,
    min_pairs: int,
) -> dict[str, Any]:
    warnings: list[str] = []
    before_mean = float(before["norm"]["mean"])
    after_mean = float(after["norm"]["mean"])
    before_p90 = float(before["norm"]["p90"])
    after_p90 = float(after["norm"]["p90"])
    before_bias = float(before["mean_norm"])
    after_bias = float(after["mean_norm"])
    improvement_frac = 0.0 if before_mean <= 1e-9 else (before_mean - after_mean) / before_mean
    bias_reduction_frac = 0.0 if before_bias <= 1e-9 else (before_bias - after_bias) / before_bias

    if int(before["n"]) < min_pairs:
        warnings.append(f"pair_count {int(before['n'])} < min_pairs {min_pairs}")
    if after_mean > before_mean * 1.02:
        warnings.append("mean pair distance would increase after correction")
    if after_p90 > before_p90 * 1.02:
        warnings.append("p90 pair distance would increase after correction")
    if before_bias >= 0.30 and bias_reduction_frac < 0.30:
        warnings.append("translation does not remove most of the mean P_geo bias")
    for item in pairs_out:
        b = float(item["before"]["norm"]["mean"])
        a = float(item["after"]["norm"]["mean"])
        if b > 1e-9 and a > b * 1.05:
            warnings.append(
                f"{item['camera_a']}:{item['camera_b']} mean distance worsens "
                f"{b:.3f}m -> {a:.3f}m"
            )

    if any("would increase" in msg or "worsens" in msg for msg in warnings):
        status = "not_recommended"
    elif warnings or improvement_frac < 0.05:
        status = "caution"
    else:
        status = "recommended"

    return {
        "status": status,
        "improvement_frac": float(improvement_frac),
        "bias_reduction_frac": float(bias_reduction_frac),
        "warnings": warnings,
    }


def _write_db_metadata(
    db_path: Path,
    *,
    correction_id: str,
    payload: dict[str, Any],
) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS g2_low_order_corrections (
                correction_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                correction_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO g2_low_order_corrections
                (correction_id, created_at, correction_json)
            VALUES (?, ?, ?)
            """,
            (
                correction_id,
                str(payload.get("created_at", "")),
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def cmd_fit(args: argparse.Namespace) -> None:
    if args.max_pair_dist_m is not None and args.max_pair_dist_m <= 0:
        raise SystemExit("--max-pair-dist-m must be > 0")
    if args.max_pairs is not None and args.max_pairs < 1:
        raise SystemExit("--max-pairs must be >= 1 when provided")
    if args.time_start is not None and args.time_end is not None and args.time_start >= args.time_end:
        raise SystemExit("--time-start must be smaller than --time-end")
    if args.min_pairs < 1:
        raise SystemExit("--min-pairs must be >= 1")
    if args.gauge == "reference" and not args.reference_camera:
        raise SystemExit("--gauge reference requires --reference-camera")
    if not args.split_manifest and not args.allow_unsplit_fitting:
        raise SystemExit(
            "--split-manifest is required for FAB2 low-order preflight so split "
            "filtering and the GT identity guard are active. Pass "
            "--allow-unsplit-fitting only for legacy diagnostics."
        )

    scene_ids = [s.strip() for s in args.scene.split(",") if s.strip()]
    audit_conn = sqlite3.connect(str(resolve_project_path(args.db)))
    require_trajectory_batch_schema(audit_conn)
    stage_run_id = start_batch_stage(
        audit_conn,
        scene_ids,
        args.batch,
        "g2_low_order_fit",
        {"out": str(resolve_project_path(args.out)), "pair_source": args.pair_source},
    )
    audit_conn.close()

    pairs_by_scene, scene_summaries = mine_pairs_by_scene(args)
    corrections_by_scene: dict[str, dict[str, dict[str, float]]] = {}
    diagnostics_by_scene: dict[str, Any] = {}
    global_warnings: list[str] = []

    for scene_id, pairs in pairs_by_scene.items():
        measurements = measurements_from_pairs(
            scene_id,
            pairs,
            pair_weight_mode=args.pair_weight,
        )
        if len(measurements) < args.min_pairs:
            msg = (
                f"scene={scene_id}: not enough MVC pairs after filtering "
                f"{len(measurements)} < --min-pairs {args.min_pairs}"
            )
            print(f"[g2 low-order][skip] {msg}")
            global_warnings.append(msg)
            continue

        # Run the preflight diagnostic on the raw pairs to detect whether the
        # error field has a dominant linear (affine) component that a constant
        # per-camera translation cannot capture.  Warn before fitting so the
        # operator can decide to go back to recalibration rather than accepting
        # a fixed-translation projection that leaves most of the structure in place.
        preflight = diagnose_low_order_bias(
            pairs,
            pair_source="g2_low_order_correction_fit",
            max_residual_m=args.max_pair_dist_m or 3.0,
        )
        affine_dominated_pairs: list[str] = []
        for item in preflight.get("camera_pairs", []):
            t_exp = float(item.get("translation_explained_frac") or 0.0)
            a_exp = item.get("affine_explained_frac")
            if a_exp is not None and float(a_exp) - t_exp > _AFFINE_DOMINANCE_MARGIN:
                affine_dominated_pairs.append(
                    f"{item['camera_a']}:{item['camera_b']} "
                    f"(translation_explained={t_exp*100:.1f}% "
                    f"affine_explained={float(a_exp)*100:.1f}%)"
                )
        if affine_dominated_pairs:
            print(
                f"[g2 low-order][warning] scene={scene_id}: the error field has a "
                "significant linear (affine) component that a fixed-translation "
                "projection cannot address.  The translation projection will leave "
                "residual structure; "
                "consider recalibrating H before relying on this JSON:\n  "
                + "\n  ".join(affine_dominated_pairs)
            )
            global_warnings.extend(
                f"scene={scene_id} affine-dominated pair: {p}"
                for p in affine_dominated_pairs
            )

        corrections, gauge_weights = fit_translations(
            measurements,
            gauge_mode=args.gauge,
            gauge_weight_mode=args.gauge_weight,
            reference_camera=args.reference_camera,
        )
        diffs = np.asarray([m.diff_xy for m in measurements], dtype=np.float64)
        before = _vector_stats(diffs)
        after = _vector_stats(corrected_diffs(measurements, corrections))
        pairs_out = pair_summaries(measurements, corrections)
        decision = _assess_correction(before, after, pairs_out, min_pairs=args.min_pairs)

        print(f"\n=== scene={scene_id} ===")
        for item in pairs_out:
            print(
                f"{item['camera_a']}:{item['camera_b']} n={item['count']} "
                f"mean_before={_fmt_xy(item['before']['mean_xy'])} "
                f"mean_after={_fmt_xy(item['after']['mean_xy'])} "
                f"dist_mean={item['before']['norm']['mean']:.4f}->"
                f"{item['after']['norm']['mean']:.4f}m "
                f"kappa=({item['mean_kappa_a']:.3f},{item['mean_kappa_b']:.3f})"
            )
        print("corrections (meters):")
        for cam, delta in sorted(corrections.items()):
            print(f"  {cam}: {_fmt_xy(delta)}  gauge_weight={gauge_weights.get(cam, 0.0):.4f}")
        print(
            "residual norm mean/p50/p90/max: "
            f"{before['norm']['mean']:.4f}/{before['norm']['p50']:.4f}/"
            f"{before['norm']['p90']:.4f}/{before['norm']['max']:.4f}m -> "
            f"{after['norm']['mean']:.4f}/{after['norm']['p50']:.4f}/"
            f"{after['norm']['p90']:.4f}/{after['norm']['max']:.4f}m"
        )
        print(
            f"decision={decision['status']} "
            f"improvement={decision['improvement_frac'] * 100:.1f}% "
            f"bias_reduction={decision['bias_reduction_frac'] * 100:.1f}%"
        )
        if decision["warnings"]:
            print("[g2 low-order][warning] correction may make some metric worse:")
            for warning in decision["warnings"]:
                print(f"  - {warning}")
            print(
                "  Use the JSON only after replay/held-out validation; otherwise train without "
                "--camera-translation-json, do not pass --g2-camera-translation-json, and keep "
                "this as a diagnostic artifact."
            )

        corrections_by_scene[scene_id] = {
            cam: {
                "dx": float(delta[0]),
                "dy": float(delta[1]),
                "gauge_weight": float(gauge_weights.get(cam, 0.0)),
            }
            for cam, delta in sorted(corrections.items())
        }
        diagnostics_by_scene[scene_id] = {
            "pair_count": len(measurements),
            "before": before,
            "after": after,
            "pair_stats": pairs_out,
            "decision": decision,
        }

    if not corrections_by_scene:
        raise SystemExit("no scene produced a usable correction")

    overall_status = "recommended"
    all_decisions = [d["decision"]["status"] for d in diagnostics_by_scene.values()]
    if "not_recommended" in all_decisions:
        overall_status = "not_recommended"
    elif "caution" in all_decisions or global_warnings:
        overall_status = "caution"

    out = resolve_project_path(args.out)
    result: dict[str, Any] = {
        "version": "g2_camera_translation_correction_v2",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "db": str(resolve_project_path(args.db)),
        "scenes": [s.strip() for s in args.scene.split(",") if s.strip()],
        "batch_id": args.batch,
        "unit": "meter",
        "correction_model": "per_camera_2d_translation",
        "fit_scope": "per_scene",
        "pair_source": args.pair_source,
        "pair_weight": args.pair_weight,
        "gauge": args.gauge,
        "gauge_weight": args.gauge_weight,
        "reference_camera": args.reference_camera,
        "max_pair_dist_m": args.max_pair_dist_m,
        "min_teacher_confidence": args.min_teacher_confidence,
        "min_teacher_confidence_gap": args.min_teacher_confidence_gap,
        "decision": {
            "status": overall_status,
            "warnings": global_warnings,
        },
        "corrections_m_by_scene": corrections_by_scene,
        "diagnostics_by_scene": diagnostics_by_scene,
        "scene_summaries": scene_summaries,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {out}")
    if overall_status == "recommended":
        print(
            "next: pass this file to train with --camera-translation-json after your normal "
            "replay/held-out checks. Pass the same file at runtime with "
            "--g2-camera-translation-json only if final_world_x/y should include the "
            "accepted fixed-translation projection."
        )
    else:
        print(
            f"[g2 low-order][{overall_status}] JSON saved for audit, but do not treat it as "
            "automatic. Review warnings before using --camera-translation-json or "
            "--g2-camera-translation-json."
        )

    if args.write_db_metadata:
        correction_id = args.correction_id or out.stem
        _write_db_metadata(resolve_project_path(args.db), correction_id=correction_id, payload=result)
        print(f"db metadata recorded -> g2_low_order_corrections/{correction_id}")

    audit_conn = sqlite3.connect(str(resolve_project_path(args.db)))
    finish_batch_stage(
        audit_conn,
        stage_run_id,
        "completed",
        {"artifact": str(out), "decision": overall_status},
    )
    audit_conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit label-free fixed-translation projection artifacts from MVC pairs."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit", help="Fit a scene-aware fixed-translation projection JSON")
    p.add_argument("--db", required=True)
    p.add_argument(
        "--batch",
        required=True,
        help="MVC trajectory batch id; must match tracking / train / remerge",
    )
    p.add_argument("--scene", required=True, help="Comma-separated scene ids")
    p.add_argument("--out", required=True, help="Output correction JSON")
    p.add_argument("--cameras", default="", help="Comma-separated cameras; default infers per scene")
    p.add_argument(
        "--overlap-pairs",
        default="",
        help="Manual overlap pairs, e.g. A:B,B:C; overrides DB scene_overlap_pairs",
    )
    p.add_argument("--pair-source", choices=("selected", "dist-filtered"), default="selected")
    p.add_argument(
        "--split-manifest",
        default=None,
        help="FAB2 gt_index.csv or dataset_split_manifest.csv; limits fitting to --train-splits.",
    )
    p.add_argument(
        "--allow-unsplit-fitting",
        action="store_true",
        help="Legacy diagnostic only: permit fitting without --split-manifest.",
    )
    p.add_argument(
        "--train-splits",
        default="train",
        help="Comma-separated split names to use from --split-manifest (default: train).",
    )
    p.add_argument("--time-start", type=float, default=None)
    p.add_argument("--time-end", type=float, default=None)
    p.add_argument(
        "--allow-gt-identity-mining",
        action="store_true",
        help="Bypass FAB2 guard that rejects MVC mining on GT global_id values; use only for smoke tests.",
    )
    p.add_argument(
        "--pair-weight",
        choices=("learning-score", "uniform", "kappa-min", "kappa-max", "kappa-product"),
        default="learning-score",
    )
    p.add_argument("--min-pairs", type=int, default=32)
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--max-pair-dist-m", type=float, default=3.0)
    p.add_argument("--min-teacher-confidence", type=float, default=0.20)
    p.add_argument("--min-teacher-confidence-gap", type=float, default=0.05)
    p.add_argument("--max-time-gap", type=float, default=0.35)
    p.add_argument(
        "--motion-compensation",
        choices=("off", "nearest", "linear_interpolate"),
        default="linear_interpolate",
    )
    p.add_argument("--time-offsets-json", default=None)
    p.add_argument(
        "--gauge",
        choices=("weighted-zero", "reference"),
        default="weighted-zero",
        help="Fix translation gauge per connected camera component",
    )
    p.add_argument(
        "--gauge-weight",
        choices=("kappa-mean", "uniform"),
        default="kappa-mean",
    )
    p.add_argument("--reference-camera", default=None)
    p.add_argument(
        "--risk-scalar",
        choices=RISK_SCALAR_MODES,
        default="metric_jacobian",
    )
    p.add_argument("--s0-m", type=float, default=60.0)
    p.add_argument("--feature-l0-m", type=float, default=10.0)
    p.add_argument("--anchor-mode", default="multiplicative_gating_offsetted")
    p.add_argument("--anchor-shape-eta", type=float, default=DEFAULT_ANCHOR_SHAPE_ETA)
    p.add_argument("--anchor-aspect-ratio-min", type=float, default=DEFAULT_ANCHOR_ASPECT_RATIO_MIN)
    p.add_argument("--anchor-aspect-ratio-max", type=float, default=DEFAULT_ANCHOR_ASPECT_RATIO_MAX)
    p.add_argument("--pixel-grazing-d0", type=float, default=50.0 / 1080.0)
    p.add_argument("--metric-sensitivity-clip-m", type=float, default=None)
    p.add_argument("--extent-clip-m", type=float, default=None)
    p.add_argument("--frame-width", type=int, default=1920)
    p.add_argument("--frame-height", type=int, default=1080)
    p.add_argument("--clip-margin", type=int, default=3)
    p.add_argument("--clip-margin-ratio", type=float, default=0.01)
    p.add_argument(
        "--write-db-metadata",
        action="store_true",
        help="Record the correction JSON in a metadata table; does not rewrite observation rows",
    )
    p.add_argument("--correction-id", default=None)
    p.set_defaults(func=cmd_fit)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
