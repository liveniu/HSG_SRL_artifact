#!/usr/bin/env python3
# Vendored HSG-SRL artifact copy. Isolated from the development repo;
# edit this file only. See src/SOURCE_MAP.csv.
"""FAB2 premise audit: kappa-vs-baseline-error monotonicity (plan section 6.3).

The asymmetric teacher assumes kappa ranks the absolute error of the geometric
baseline. This tool computes, per scene and on the requested splits of the
identity-annotated audit subset:

* Spearman correlation rho between kappa and ||P_geo - P_GT||;
* kappa-tertile median errors (tertile boundaries fixed on the same subset);
* the preregistered continuation verdict (rho <= threshold and monotone
  decreasing tertile medians).

Sites failing the audit are excluded from A-level attribution and routed to
the controlled perturbation-recovery protocol (plan section 9.4). Audit
identities are read only from the frozen manifest and never flow back into
training.

Output: <out-dir>/iv_b/premise_kappa_error_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
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
    split_csv_arg,
    upsert_csv_rows,
    write_run_metadata,
)
from pipeline.geometry_guided_residual_mlp import (  # noqa: E402
    GeometryDataView,
    _infer_and_calib_wh_from_db,
)
from pipeline.multi_camera_trajectory_fusion import adapt_homography_to_resolution  # noqa: E402

AUDIT_FIELDS = [
    "scene",
    "batch_id",
    "splits",
    "n",
    "spearman_rho",
    "rho_threshold",
    "tertile_low_kappa_median_m",
    "tertile_mid_kappa_median_m",
    "tertile_high_kappa_median_m",
    "tertile_monotone_decreasing",
    "premise_pass",
    "baseline_rmse_m",
    "baseline_median_m",
]


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (average ranks for ties)."""
    def _rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(x), dtype=np.float64)
        # average ties
        sx = x[order]
        i = 0
        while i < len(sx):
            j = i
            while j + 1 < len(sx) and sx[j + 1] == sx[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = 0.5 * (i + j)
            i = j + 1
        return ranks

    ra, rb = _rank(a), _rank(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def load_manifest(path: Path) -> dict[tuple[str, int, int], tuple[str, float, float]]:
    """(camera_id, local_track_id, frame_index) -> (split, gt_x, gt_y)."""
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


def audit_scene(args: argparse.Namespace, scene: str, manifest: dict) -> dict[str, Any]:
    conn = connect_db(args.db)
    view = GeometryDataView()
    cameras = sorted(
        r[0] for r in conn.execute(
            "SELECT DISTINCT camera_name FROM trajectory_observations WHERE scene_id=? AND batch_id=?",
            (scene, args.batch),
        )
    )
    cal = view.load_camera_calibrations(conn, scene, cameras)
    infer_wh, calib_wh = _infer_and_calib_wh_from_db(conn, scene, args.batch, cameras)
    adapted = {}
    for cam, (h, s) in cal.items():
        iw, cw = infer_wh.get(cam), calib_wh.get(cam)
        if iw and cw and iw != cw:
            h = adapt_homography_to_resolution(h, cw, iw)
        adapted[cam] = (h, s)

    obs = view.iter_observations_from_db(
        conn, scene, args.batch, cameras, adapted,
        frame_wh_by_camera={c: infer_wh[c] for c in cameras},
        risk_scalar_mode=args.risk_scalar,
        s0_m=args.s0_m,
        feature_l0_m=args.feature_l0_m,
        anchor_mode=args.anchor_mode,
        anchor_shape_eta=args.anchor_shape_eta,
    )

    wanted_splits = set(split_csv_arg(args.splits))
    kappa: list[float] = []
    err: list[float] = []
    for o in obs:
        key = (o.camera_name, int(o.local_track_id), int(o.frame_index or -1))
        sp, gx, gy = manifest.get(key, ("", float("nan"), float("nan")))
        if sp not in wanted_splits or not (np.isfinite(gx) and np.isfinite(gy)):
            continue
        kappa.append(float(o.horizon_confidence))
        err.append(float(np.hypot(o.world_geo[0] - gx, o.world_geo[1] - gy)))
    conn.close()

    k = np.asarray(kappa)
    e = np.asarray(err)
    row: dict[str, Any] = {
        "scene": scene,
        "batch_id": args.batch,
        "splits": args.splits,
        "n": int(e.size),
        "rho_threshold": args.rho_threshold,
    }
    if e.size < args.min_samples:
        row.update({
            "spearman_rho": "",
            "tertile_low_kappa_median_m": "",
            "tertile_mid_kappa_median_m": "",
            "tertile_high_kappa_median_m": "",
            "tertile_monotone_decreasing": "",
            "premise_pass": 0,
            "baseline_rmse_m": "",
            "baseline_median_m": "",
        })
        return row

    rho = spearman_rho(k, e)
    q1, q2 = np.quantile(k, [1.0 / 3.0, 2.0 / 3.0])
    med_low = float(np.median(e[k <= q1]))
    med_mid = float(np.median(e[(k > q1) & (k <= q2)]))
    med_high = float(np.median(e[k > q2]))
    monotone = med_low >= med_mid >= med_high
    row.update({
        "spearman_rho": round(rho, 4),
        "tertile_low_kappa_median_m": round(med_low, 4),
        "tertile_mid_kappa_median_m": round(med_mid, 4),
        "tertile_high_kappa_median_m": round(med_high, 4),
        "tertile_monotone_decreasing": int(monotone),
        "premise_pass": int(rho <= args.rho_threshold and monotone),
        "baseline_rmse_m": round(float(np.sqrt(np.mean(e ** 2))), 4),
        "baseline_median_m": round(float(np.median(e)), 4),
    })
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--scene", required=True, help="Comma-separated scene ids")
    parser.add_argument("--split-manifest", required=True, help="FAB2 dataset_split_manifest.csv with GT columns")
    parser.add_argument("--splits", default="train", help="Comma-separated splits used for the audit")
    parser.add_argument("--out-dir", default=str(FAB2_DEFAULT_RESULTS))
    parser.add_argument("--rho-threshold", type=float, default=-0.3)
    parser.add_argument("--min-samples", type=int, default=200)
    parser.add_argument("--risk-scalar", default="metric_jacobian")
    parser.add_argument("--anchor-mode", default="multiplicative_gating_offsetted")
    parser.add_argument("--s0-m", type=float, default=60.0)
    parser.add_argument("--feature-l0-m", type=float, default=10.0)
    parser.add_argument("--anchor-shape-eta", type=float, default=0.75)
    args = parser.parse_args()

    manifest = load_manifest(Path(args.split_manifest))
    out_root = ensure_dir(Path(args.out_dir))
    out_dir = ensure_dir(out_root / "iv_b")
    rows = []
    for scene in split_csv_arg(args.scene):
        row = audit_scene(args, scene, manifest)
        rows.append(row)
        print(
            f"[premise-audit] scene={row['scene']} n={row['n']} rho={row['spearman_rho']} "
            f"tertile medians (low/mid/high kappa)={row['tertile_low_kappa_median_m']}/"
            f"{row['tertile_mid_kappa_median_m']}/{row['tertile_high_kappa_median_m']} "
            f"pass={row['premise_pass']}"
        )
    out_csv = out_dir / "premise_kappa_error_audit.csv"
    upsert_csv_rows(out_csv, rows, AUDIT_FIELDS, key_fields=["scene", "batch_id", "splits"])
    write_run_metadata(out_root, "eval_fab2_premise_audit", args, {"scenes": [r["scene"] for r in rows]})
    print(f"premise audit written -> {out_csv}")


if __name__ == "__main__":
    main()
