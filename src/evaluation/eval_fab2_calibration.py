#!/usr/bin/env python3
# Vendored HSG-SRL artifact copy. Isolated from the development repo;
# edit this file only. See src/SOURCE_MAP.csv.
"""Build FAB2 calibration robustness audit tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.fab2_common import (  # noqa: E402
    FAB2_DEFAULT_RESULTS,
    command_to_str,
    ensure_dir,
    finite_float,
    read_csv_rows,
    run_command,
    split_csv_arg,
    write_csv,
    write_json,
    write_run_metadata,
)


BFS_FIELDS = [
    "dataset",
    "site",
    "recording",
    "scene_id",
    "bfs_hop",
    "joint_refinement",
    "camera_count",
    "mean_reproj_rms",
    "max_reproj_rms",
    "bev_rmse_m",
    "hota_proxy",
    "idf1",
]

PREFLIGHT_FIELDS = [
    "scene_id",
    "status",
    "R_T",
    "R_A",
    "R_A_minus_R_T",
    "raw_mean_m",
    "translation_mean_m",
    "raw_p90_m",
    "translation_p90_m",
    "camera",
    "tx_m",
    "ty_m",
    "warnings",
]

PERTURB_FIELDS = [
    "perturbation_id",
    "axis",
    "degrees",
    "method",
    "bev_rmse_m",
    "hota_proxy",
    "idf1",
    "mota",
    "id_switches",
    "localization_csv",
    "tracking_csv",
]

PLAN_FIELDS = ["stage", "artifact", "command", "status"]


def _overall_metric(csv_path: str | None, metric_name: str) -> float | str:
    if not csv_path:
        return ""
    for row in read_csv_rows(csv_path):
        if row.get("metric_scope") == "overall" and row.get(metric_name, "") != "":
            v = finite_float(row.get(metric_name))
            return v if v is not None else ""
    return ""


def _build_bfs_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.topology_csv:
        return []
    topo = [r for r in read_csv_rows(args.topology_csv) if r.get("record_type") == "camera_calibration"]
    by_scene_hop: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in topo:
        hop = row.get("bfs_hop") or "unknown"
        by_scene_hop.setdefault((row.get("scene_id", ""), hop), []).append(row)
    loc_rmse = _overall_metric(args.localization_csv, "rmse_m")
    hota = _overall_metric(args.tracking_csv, "hota_proxy")
    idf1 = _overall_metric(args.tracking_csv, "idf1")
    out: list[dict[str, Any]] = []
    for (_scene, hop), rows in sorted(by_scene_hop.items()):
        rms = [finite_float(r.get("reproj_rms")) for r in rows]
        rms = [r for r in rms if r is not None]
        sample = rows[0]
        out.append(
            {
                "dataset": sample.get("dataset", ""),
                "site": sample.get("site", ""),
                "recording": sample.get("recording", ""),
                "scene_id": sample.get("scene_id", ""),
                "bfs_hop": hop,
                "joint_refinement": args.joint_refinement,
                "camera_count": len(rows),
                "mean_reproj_rms": sum(rms) / len(rms) if rms else "",
                "max_reproj_rms": max(rms) if rms else "",
                "bev_rmse_m": loc_rmse,
                "hota_proxy": hota,
                "idf1": idf1,
            }
        )
    return out


def _nested_get(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _build_preflight_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.preflight_json:
        return []
    data = json.loads(Path(args.preflight_json).read_text(encoding="utf-8"))
    diagnostics = data.get("diagnostics_by_scene", {}) if isinstance(data, dict) else {}
    corrections = data.get("corrections_m_by_scene", {}) if isinstance(data, dict) else {}
    rows: list[dict[str, Any]] = []
    for scene_id, diag in diagnostics.items():
        decision = diag.get("decision", {}) if isinstance(diag, dict) else {}
        status = decision.get("status", data.get("decision", {}).get("status", "unknown"))
        warnings = "; ".join(str(w) for w in decision.get("warnings", []))
        rt = _nested_get(diag, "explained_fraction", "translation")
        ra = _nested_get(diag, "explained_fraction", "affine")
        raw_mean = _nested_get(diag, "raw", "norm", "mean")
        trans_mean = _nested_get(diag, "translation_corrected", "norm", "mean")
        raw_p90 = _nested_get(diag, "raw", "norm", "p90")
        trans_p90 = _nested_get(diag, "translation_corrected", "norm", "p90")
        scene_corr = corrections.get(scene_id, {}) if isinstance(corrections, dict) else {}
        if not scene_corr:
            rows.append(
                {
                    "scene_id": scene_id,
                    "status": status,
                    "R_T": rt,
                    "R_A": ra,
                    "R_A_minus_R_T": (float(ra) - float(rt)) if rt is not None and ra is not None else "",
                    "raw_mean_m": raw_mean,
                    "translation_mean_m": trans_mean,
                    "raw_p90_m": raw_p90,
                    "translation_p90_m": trans_p90,
                    "camera": "",
                    "tx_m": "",
                    "ty_m": "",
                    "warnings": warnings,
                }
            )
        for cam, value in scene_corr.items():
            if isinstance(value, dict):
                tx, ty = value.get("dx", value.get("x", 0.0)), value.get("dy", value.get("y", 0.0))
            else:
                tx, ty = value[0], value[1]
            rows.append(
                {
                    "scene_id": scene_id,
                    "status": status,
                    "R_T": rt,
                    "R_A": ra,
                    "R_A_minus_R_T": (float(ra) - float(rt)) if rt is not None and ra is not None else "",
                    "raw_mean_m": raw_mean,
                    "translation_mean_m": trans_mean,
                    "raw_p90_m": raw_p90,
                    "translation_p90_m": trans_p90,
                    "camera": cam,
                    "tx_m": tx,
                    "ty_m": ty,
                    "warnings": warnings,
                }
            )
    return rows


def _build_perturb_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.perturbation_runs:
        return []
    out: list[dict[str, Any]] = []
    for row in read_csv_rows(args.perturbation_runs):
        loc = row.get("localization_csv", "")
        trk = row.get("tracking_csv", "")
        out.append(
            {
                "perturbation_id": row.get("perturbation_id", row.get("id", "")),
                "axis": row.get("axis", ""),
                "degrees": row.get("degrees", row.get("perturb_deg", "")),
                "method": row.get("method", ""),
                "bev_rmse_m": _overall_metric(loc, "rmse_m"),
                "hota_proxy": _overall_metric(trk, "hota_proxy"),
                "idf1": _overall_metric(trk, "idf1"),
                "mota": _overall_metric(trk, "mota"),
                "id_switches": _overall_metric(trk, "id_switches"),
                "localization_csv": loc,
                "tracking_csv": trk,
            }
        )
    return out


def _preflight_plan(args: argparse.Namespace, out_json: Path) -> list[dict[str, Any]]:
    if not args.db or not args.batch or not args.scene:
        return []
    cmd = [
        sys.executable,
        str(_ROOT / "pipeline" / "g2_low_order_correction.py"),
        "fit",
        "--db",
        args.db,
        "--batch",
        args.batch,
        "--scene",
        args.scene,
        "--out",
        str(out_json),
    ]
    if args.cameras:
        cmd.extend(["--cameras", args.cameras])
    if args.overlap_pairs:
        cmd.extend(["--overlap-pairs", args.overlap_pairs])
    if args.split_manifest:
        cmd.extend(["--split-manifest", args.split_manifest])
    if args.train_splits:
        cmd.extend(["--train-splits", args.train_splits])
    if args.time_start is not None:
        cmd.extend(["--time-start", str(args.time_start)])
    if args.time_end is not None:
        cmd.extend(["--time-end", str(args.time_end)])
    if args.allow_gt_identity_mining:
        cmd.append("--allow-gt-identity-mining")
    return [
        {
            "stage": "low_order_preflight_fit",
            "artifact": str(out_json),
            "command": command_to_str(cmd),
            "_cmd": cmd,
            "status": "dry_run" if args.dry_run else "planned",
        }
    ]


def run(args: argparse.Namespace) -> None:
    out_root = ensure_dir(Path(args.out_dir))
    ive = ensure_dir(out_root / "iv_e")
    plan_rows = _preflight_plan(args, ive / "preflight_Tc.json")
    if args.execute_preflight:
        for row in plan_rows:
            run_command(row["_cmd"], dry_run=args.dry_run, cwd=_ROOT)
            row["status"] = "completed" if not args.dry_run else "dry_run"

    bfs_rows = _build_bfs_rows(args)
    preflight_rows = _build_preflight_rows(args)
    perturb_rows = _build_perturb_rows(args)
    write_csv(ive / "bfs_hop_refinement.csv", bfs_rows, BFS_FIELDS)
    write_csv(ive / "preflight_control.csv", preflight_rows, PREFLIGHT_FIELDS)
    write_csv(ive / "frozen_homography_perturbation.csv", perturb_rows, PERTURB_FIELDS)
    write_csv(ive / "calibration_command_plan.csv", plan_rows, PLAN_FIELDS)
    write_json(
        ive / "calibration_summary.json",
        {
            "bfs_rows": len(bfs_rows),
            "preflight_rows": len(preflight_rows),
            "perturbation_rows": len(perturb_rows),
        },
    )
    write_run_metadata(out_root, "eval_fab2_calibration", args)
    print(f"Calibration audit tables written under {ive}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(FAB2_DEFAULT_RESULTS))
    parser.add_argument("--db", default=None)
    parser.add_argument("--batch", default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--cameras", default="")
    parser.add_argument("--overlap-pairs", default="")
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--train-splits", default="train")
    parser.add_argument("--time-start", type=float, default=None)
    parser.add_argument("--time-end", type=float, default=None)
    parser.add_argument("--allow-gt-identity-mining", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-preflight", action="store_true")
    parser.add_argument("--topology-csv", default=None)
    parser.add_argument("--localization-csv", default=None)
    parser.add_argument("--tracking-csv", default=None)
    parser.add_argument("--joint-refinement", choices=("off", "on", "unknown"), default="unknown")
    parser.add_argument("--preflight-json", default=None)
    parser.add_argument("--perturbation-runs", default=None, help="CSV listing perturbation_id,axis,degrees,localization_csv,tracking_csv")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
