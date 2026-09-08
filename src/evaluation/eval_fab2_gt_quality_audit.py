#!/usr/bin/env python3
# Vendored HSG-SRL artifact copy. Isolated from the development repo;
# edit this file only. See src/SOURCE_MAP.csv.
"""FAB2 GT quality audit: identity-mismatch exclusion and cleaned localization.

Ground-truth matching faults (a local track matched to the wrong GT identity,
or a mis-annotated GT trajectory) show up as (camera, gt_id) units whose
matched error is large and *constant over tens of seconds* -- unlike geometric
error, which varies smoothly with position. Left in place they dominate RMSE
(lumpi_M5: 2.85% of observations carry the baseline from 1.87 m to 4.38 m of
RMSE) and invert kappa strata, because the observations are geometrically
confident and kappa cannot flag identity mismatch.

Protocol: units are flagged on the *baseline* predictions only (median matched
error > ``--median-threshold-m``), then the frozen exclusion set is applied to
every method's predictions, keeping the comparison paired. The exclusion list
is written next to the cleaned metrics so every excluded identity is auditable.

Outputs under <site-dir>/iv_c/:
* ``gt_quality_exclusions.csv``  - one row per excluded (camera, gt_id) unit;
* ``localization_cleaned_by_method.csv`` - full vs cleaned metrics per method.
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
    ensure_dir,
    upsert_csv_rows,
    write_run_metadata,
)

EXCLUSION_FIELDS = [
    "scene",
    "camera_id",
    "matched_gt_id",
    "class_name",
    "n",
    "median_err_m",
    "p95_err_m",
    "max_err_m",
    "time_start",
    "time_end",
    "median_threshold_m",
]

METRIC_FIELDS = [
    "scene",
    "method",
    "variant",
    "n",
    "rmse_m",
    "median_m",
    "p95_m",
    "p99_m",
    "n_excluded",
    "excluded_units",
    "median_threshold_m",
]


def load_errors(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            try:
                err = float(np.hypot(
                    float(r["world_x"]) - float(r["gt_world_x"]),
                    float(r["world_y"]) - float(r["gt_world_y"]),
                ))
            except (KeyError, ValueError):
                continue
            out.append({
                "scene": r.get("scene_id", ""),
                "camera_id": r.get("camera_id", ""),
                "gt_id": r.get("matched_gt_id", ""),
                "class_name": r.get("class_name", ""),
                "timestamp": r.get("timestamp", ""),
                "err": err,
            })
    return out


def flag_units(baseline: list[dict[str, Any]], threshold_m: float) -> dict[tuple[str, str], dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in baseline:
        groups.setdefault((row["camera_id"], row["gt_id"]), []).append(row)
    flagged: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in groups.items():
        errs = np.asarray([r["err"] for r in rows])
        med = float(np.median(errs))
        if med <= threshold_m:
            continue
        times = sorted(float(r["timestamp"]) for r in rows if r["timestamp"] not in ("", None))
        flagged[key] = {
            "scene": rows[0]["scene"],
            "camera_id": key[0],
            "matched_gt_id": key[1],
            "class_name": rows[0]["class_name"],
            "n": int(errs.size),
            "median_err_m": round(med, 4),
            "p95_err_m": round(float(np.percentile(errs, 95)), 4),
            "max_err_m": round(float(errs.max()), 4),
            "time_start": round(times[0], 2) if times else "",
            "time_end": round(times[-1], 2) if times else "",
            "median_threshold_m": threshold_m,
        }
    return flagged


def metrics(errs: np.ndarray) -> dict[str, Any]:
    if errs.size == 0:
        return {"n": 0, "rmse_m": "", "median_m": "", "p95_m": "", "p99_m": ""}
    return {
        "n": int(errs.size),
        "rmse_m": round(float(np.sqrt(np.mean(errs ** 2))), 4),
        "median_m": round(float(np.median(errs)), 4),
        "p95_m": round(float(np.percentile(errs, 95)), 4),
        "p99_m": round(float(np.percentile(errs, 99)), 4),
    }


def method_name_from_file(path: Path) -> str:
    stem = path.stem
    return stem[len("predictions_"):] if stem.startswith("predictions_") else stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", required=True, help="Site results dir, e.g. research/results/fab2/sites/lumpi_M5")
    parser.add_argument("--baseline-file", default="predictions_baseline_ipm.csv")
    parser.add_argument("--median-threshold-m", type=float, default=5.0)
    args = parser.parse_args()

    site_dir = Path(args.site_dir)
    pred_dir = site_dir / "comparability" / "predictions_by_method"
    baseline_path = pred_dir / args.baseline_file
    if not baseline_path.is_file():
        raise SystemExit(f"baseline predictions not found: {baseline_path}")

    baseline = load_errors(baseline_path)
    flagged = flag_units(baseline, args.median_threshold_m)
    scene = baseline[0]["scene"] if baseline else site_dir.name
    out_dir = ensure_dir(site_dir / "iv_c")

    excl_rows = sorted(flagged.values(), key=lambda r: -r["median_err_m"])
    excl_csv = out_dir / "gt_quality_exclusions.csv"
    upsert_csv_rows(excl_csv, excl_rows, EXCLUSION_FIELDS, key_fields=["scene", "camera_id", "matched_gt_id"])
    n_excl_obs = sum(r["n"] for r in excl_rows)
    total = len(baseline)
    print(
        f"[gt-audit] {scene}: {len(excl_rows)} excluded (camera, gt_id) units, "
        f"{n_excl_obs}/{total} observations ({100.0 * n_excl_obs / max(total, 1):.2f}%) "
        f"at median threshold {args.median_threshold_m} m"
    )

    metric_rows: list[dict[str, Any]] = []
    for pred_file in sorted(pred_dir.glob("predictions_*.csv")):
        rows = baseline if pred_file == baseline_path else load_errors(pred_file)
        errs_all = np.asarray([r["err"] for r in rows])
        keep = np.asarray([(r["camera_id"], r["gt_id"]) not in flagged for r in rows])
        method = method_name_from_file(pred_file)
        for variant, errs in (("full", errs_all), ("cleaned", errs_all[keep])):
            row: dict[str, Any] = {
                "scene": scene,
                "method": method,
                "variant": variant,
                "n_excluded": int(errs_all.size - errs.size) if variant == "cleaned" else 0,
                "excluded_units": len(excl_rows) if variant == "cleaned" else 0,
                "median_threshold_m": args.median_threshold_m,
            }
            row.update(metrics(errs))
            metric_rows.append(row)
        cleaned = next(r for r in metric_rows if r["method"] == method and r["variant"] == "cleaned")
        full = next(r for r in metric_rows if r["method"] == method and r["variant"] == "full")
        print(
            f"[gt-audit] {method:<36} rmse {full['rmse_m']} -> {cleaned['rmse_m']}  "
            f"med {full['median_m']} -> {cleaned['median_m']}  p95 {full['p95_m']} -> {cleaned['p95_m']}"
        )

    out_csv = out_dir / "localization_cleaned_by_method.csv"
    upsert_csv_rows(out_csv, metric_rows, METRIC_FIELDS, key_fields=["scene", "method", "variant"])
    write_run_metadata(site_dir, "eval_fab2_gt_quality_audit", args, {
        "excluded_units": len(excl_rows),
        "excluded_observations": n_excl_obs,
    })
    print(f"exclusions -> {excl_csv}")
    print(f"cleaned metrics -> {out_csv}")


if __name__ == "__main__":
    main()
