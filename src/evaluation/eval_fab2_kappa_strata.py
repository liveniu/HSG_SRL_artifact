#!/usr/bin/env python3
# Vendored HSG-SRL artifact copy. Isolated from the development repo;
# edit this file only. See src/SOURCE_MAP.csv.
"""FAB2 kappa-stratified localization report (plan sections 6.3 / 9.x).

The theory predicts residual correction gains concentrate where the geometric
baseline is least reliable (low kappa). This tool evaluates, on the test
split of the frozen manifest, the localization error of

* ``baseline_pgeo``   - geometric baseline P_geo;
* ``final``           - P_geo + frozen-checkpoint residual (per seed);

stratified by kappa tertiles (boundaries fixed on the evaluated subset), and
writes ``<out-dir>/iv_c/localization_kappa_strata.csv``. Kappa is a
self-supervised quantity; it is used only to slice reporting, never to select
checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.fab2_common import (  # noqa: E402
    FAB2_DEFAULT_RESULTS,
    ensure_dir,
    split_csv_arg,
    upsert_csv_rows,
    write_run_metadata,
)
from pipeline.geometry_guided_residual_mlp import (  # noqa: E402
    GeometryDataView,
    _infer_and_calib_wh_from_db,
    load_camera_translation_corrections,
    load_checkpoint,
)
from pipeline.multi_camera_trajectory_fusion import adapt_homography_to_resolution  # noqa: E402

FIELDS = [
    "scene",
    "batch_id",
    "splits",
    "method",
    "seed",
    "stratum",
    "n",
    "rmse_m",
    "median_m",
    "p95_m",
    "delta_rmse_vs_baseline_m",
]


def load_manifest(path: Path) -> dict[tuple[str, int, int], tuple[str, float, float]]:
    out: dict[tuple[str, int, int], tuple[str, float, float]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            try:
                key = (r["camera_id"], int(r["local_track_id"]), int(r["frame_index"]))
            except (KeyError, ValueError):
                continue
            gx = float(r["gt_world_x"]) if r.get("gt_world_x") else float("nan")
            gy = float(r["gt_world_y"]) if r.get("gt_world_y") else float("nan")
            out[key] = (r.get("split", ""), gx, gy)
    return out


def load_data(args: argparse.Namespace, meta: dict) -> dict[str, np.ndarray]:
    conn = sqlite3.connect(str(args.db))
    view = GeometryDataView()
    cameras = sorted(
        r[0] for r in conn.execute(
            "SELECT DISTINCT camera_name FROM trajectory_observations WHERE scene_id=? AND batch_id=?",
            (args.scene, args.batch),
        )
    )
    cal = view.load_camera_calibrations(conn, args.scene, cameras)
    infer_wh, calib_wh = _infer_and_calib_wh_from_db(conn, args.scene, args.batch, cameras)
    adapted = {}
    for cam, (h, s) in cal.items():
        iw, cw = infer_wh.get(cam), calib_wh.get(cam)
        if iw and cw and iw != cw:
            h = adapt_homography_to_resolution(h, cw, iw)
        adapted[cam] = (h, s)
    obs = view.iter_observations_from_db(
        conn, args.scene, args.batch, cameras, adapted,
        frame_wh_by_camera={c: infer_wh[c] for c in cameras},
        risk_scalar_mode=str(meta.get("risk_scalar_mode", "metric_jacobian")),
        s0_m=float(meta.get("s0_m", 60.0)),
        feature_l0_m=float(meta.get("feature_l0_m", 10.0)),
        anchor_mode=str(meta.get("anchor_mode", "multiplicative_gating_offsetted")),
        anchor_shape_eta=float(meta.get("anchor_shape_eta", 0.75)),
        sensitivity_clip_m=meta.get("sensitivity_clip_m"),
        extent_clip_m=meta.get("extent_clip_m"),
    )
    manifest = load_manifest(Path(args.split_manifest))
    wanted = set(split_csv_arg(args.splits))
    kappa, feats, p_geo, gt, cams = [], [], [], [], []
    for o in obs:
        key = (o.camera_name, int(o.local_track_id), int(o.frame_index or -1))
        sp, gx, gy = manifest.get(key, ("", float("nan"), float("nan")))
        if sp not in wanted or not (np.isfinite(gx) and np.isfinite(gy)):
            continue
        kappa.append(float(o.horizon_confidence))
        feats.append(np.asarray(o.feature, dtype=np.float32))
        p_geo.append(np.asarray(o.world_geo, dtype=np.float64))
        gt.append((gx, gy))
        cams.append(o.camera_name)
    conn.close()
    return {
        "kappa": np.asarray(kappa),
        "features": np.stack(feats),
        "p_geo": np.stack(p_geo),
        "gt": np.asarray(gt, dtype=np.float64),
        "camera": np.asarray(cams),
    }


def strata_rows(
    args: argparse.Namespace, method: str, seed: Any,
    pred: np.ndarray, data: dict[str, np.ndarray],
    baseline_stats: dict[str, float] | None,
) -> list[dict[str, Any]]:
    k = data["kappa"]
    q1, q2 = np.quantile(k, [1.0 / 3.0, 2.0 / 3.0])
    masks = {
        "all": np.ones_like(k, dtype=bool),
        "kappa_low": k <= q1,
        "kappa_mid": (k > q1) & (k <= q2),
        "kappa_high": k > q2,
    }
    rows = []
    for stratum, m in masks.items():
        err = np.hypot(pred[m, 0] - data["gt"][m, 0], pred[m, 1] - data["gt"][m, 1])
        rmse = float(np.sqrt(np.mean(err ** 2)))
        rows.append({
            "scene": args.scene,
            "batch_id": args.batch,
            "splits": args.splits,
            "method": method,
            "seed": seed,
            "stratum": stratum,
            "n": int(err.size),
            "rmse_m": round(rmse, 4),
            "median_m": round(float(np.median(err)), 4),
            "p95_m": round(float(np.percentile(err, 95)), 4),
            "delta_rmse_vs_baseline_m": (
                round(rmse - baseline_stats[stratum], 4) if baseline_stats else ""
            ),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split-manifest", required=True, help="dataset_split_manifest.csv with GT columns")
    parser.add_argument("--splits", default="test")
    parser.add_argument("--checkpoints", default="", help="Comma-separated frozen checkpoint paths (seed inferred from _seedN suffix)")
    parser.add_argument("--camera-translation-json", default=None, help="Accepted T_c correction JSON (part of the label-free chain)")
    parser.add_argument("--camera-translation-acceptance", choices=("recommended", "caution", "any"), default="recommended")
    parser.add_argument("--out-dir", default=str(FAB2_DEFAULT_RESULTS))
    args = parser.parse_args()

    import torch

    ckpts = [Path(p) for p in split_csv_arg(args.checkpoints)]
    meta: dict = {}
    if ckpts:
        _, meta = load_checkpoint(ckpts[0])
    data = load_data(args, meta)

    # Accepted per-camera T_c translations are part of the label-free chain
    # (applied at train and remerge time). A pure world-plane translation
    # leaves kappa and the feature vector unchanged, so adding it to P_geo
    # reproduces the runtime chain exactly.
    tc_offset = np.zeros_like(data["p_geo"])
    if args.camera_translation_json:
        corr = load_camera_translation_corrections(
            args.camera_translation_json, acceptance=args.camera_translation_acceptance
        )
        scene_map = corr.by_scene.get(args.scene, {})
        for cam, (dx, dy) in scene_map.items():
            m = data["camera"] == cam
            tc_offset[m, 0] += dx
            tc_offset[m, 1] += dy

    rows: list[dict[str, Any]] = []
    base_rows = strata_rows(args, "Baseline-IPM", "", data["p_geo"], data, None)
    baseline_stats = {r["stratum"]: r["rmse_m"] for r in base_rows}
    rows.extend(base_rows)

    for ckpt in ckpts:
        model, _ = load_checkpoint(ckpt)
        model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(data["features"]), 65536):
                t = torch.from_numpy(data["features"][start:start + 65536].astype(np.float32))
                preds.append(model(t).cpu().numpy().astype(np.float64))
        delta = np.concatenate(preds, axis=0)
        seed = ckpt.stem.split("seed")[-1] if "seed" in ckpt.stem else ckpt.stem
        rows.extend(strata_rows(
            args, "Ours-HSG-SRL-Frozen", seed,
            data["p_geo"] + tc_offset + delta, data, baseline_stats,
        ))

    out_dir = ensure_dir(Path(args.out_dir) / "iv_c")
    out_csv = out_dir / "localization_kappa_strata.csv"
    upsert_csv_rows(out_csv, rows, FIELDS, key_fields=["scene", "batch_id", "splits", "method", "seed", "stratum"])
    write_run_metadata(Path(args.out_dir), "eval_fab2_kappa_strata", args, {"checkpoints": [str(c) for c in ckpts]})
    for r in rows:
        print(
            f"{r['method']:<22} seed={r['seed'] or '-':<3} {r['stratum']:<10} n={r['n']:<7} "
            f"rmse={r['rmse_m']:.4f} med={r['median_m']:.4f} p95={r['p95_m']:.4f} d_rmse={r['delta_rmse_vs_baseline_m']}"
        )
    print(f"kappa strata report -> {out_csv}")


if __name__ == "__main__":
    main()
