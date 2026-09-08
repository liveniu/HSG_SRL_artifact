#!/usr/bin/env python3
# Vendored HSG-SRL artifact copy. Isolated from the development repo;
# edit this file only. See src/SOURCE_MAP.csv.
"""Generate and optionally run FAB2 Appendix-F hyperparameter scans."""

from __future__ import annotations

import argparse
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
    run_command,
    split_csv_arg,
    write_csv,
    write_json,
    write_run_metadata,
)


FIELDS = ["group", "parameter", "value", "seed", "status", "command", "artifact_dir"]


def _scan_defs() -> list[tuple[str, str, list[str], str]]:
    return [
        ("s0", "--s0-m", ["30", "60", "120"], "s0_m"),
        ("rmax", "--max-residual-m", ["0.5", "1.0", "1.5", "2.0"], "r_max"),
        ("lambda_smooth", "--lambda-smooth", ["0.0", "0.02", "0.05"], "lambda_s"),
        ("teacher_confidence", "--min-teacher-confidence", ["0.1", "0.2", "0.3"], "tau"),
        ("teacher_gap", "--min-teacher-confidence-gap", ["0.0", "0.05", "0.10"], "tau_delta"),
        ("anchor_shape_eta", "--anchor-shape-eta", ["0.0", "0.5", "0.75", "1.0"], "eta_shape"),
        ("motion_compensation", "--motion-compensation", ["off", "nearest", "linear_interpolate"], "motion"),
    ]


def _cmd(args: argparse.Namespace, flag: str, value: str, seed: int, artifact: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(_ROOT / "evaluation" / "train_fab2_site_frozen.py"),
        "--db",
        args.db,
        "--batch",
        args.batch,
        "--scene",
        args.scene,
        "--out-dir",
        str(artifact),
        "--seeds",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--max-pair-dist-m",
        str(args.max_pair_dist_m),
        "--min-mvc-pairs",
        str(args.min_mvc_pairs),
        flag,
        value,
    ]
    if args.max_pairs is not None:
        cmd.extend(["--max-pairs", str(args.max_pairs)])
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
    if args.cameras:
        cmd.extend(["--cameras", args.cameras])
    if args.skip_low_order:
        cmd.append("--skip-low-order")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def run(args: argparse.Namespace) -> None:
    out_root = ensure_dir(Path(args.out_dir))
    app = ensure_dir(out_root / "appendix")
    seeds = [int(s) for s in split_csv_arg(args.seeds)] or [0]
    selected = set(split_csv_arg(args.only_groups))
    rows: list[dict[str, Any]] = []
    for group, flag, values, parameter in _scan_defs():
        if selected and group not in selected and parameter not in selected:
            continue
        for value in values:
            for seed in seeds:
                artifact = out_root / "hyperparams" / f"{group}_{value}_seed{seed}".replace(".", "p")
                cmd = _cmd(args, flag, value, seed, artifact)
                rows.append(
                    {
                        "group": group,
                        "parameter": parameter,
                        "value": value,
                        "seed": seed,
                        "status": "dry_run" if args.dry_run else ("completed" if args.execute else "planned"),
                        "command": command_to_str(cmd),
                        "artifact_dir": str(artifact),
                    }
                )
                if args.execute:
                    run_command(cmd, dry_run=args.dry_run, cwd=_ROOT)
    write_csv(app / "hyperparameter_sensitivity_plan.csv", rows, FIELDS)
    # The final metric table is intentionally a separate file so aggregation
    # from completed runs cannot be confused with the command plan.
    write_csv(app / "hyperparameter_sensitivity.csv", [], FIELDS + ["rmse_m", "hota_proxy", "idf1"])
    write_json(app / "hyperparameter_sensitivity_summary.json", {"jobs": len(rows), "execute": args.execute, "seeds": seeds})
    write_run_metadata(out_root, "eval_fab2_hyperparams", args, {"seeds": seeds})
    print(f"Hyperparameter scan plan written under {app}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out-dir", default=str(FAB2_DEFAULT_RESULTS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--max-pair-dist-m", type=float, default=3.0)
    parser.add_argument("--min-mvc-pairs", type=int, default=64)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional diagnostic cap on retained MVC pairs after teacher filters.",
    )
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--train-splits", default="train")
    parser.add_argument("--time-start", type=float, default=None)
    parser.add_argument("--time-end", type=float, default=None)
    parser.add_argument("--allow-gt-identity-mining", action="store_true")
    parser.add_argument("--cameras", default="")
    parser.add_argument("--skip-low-order", action="store_true")
    parser.add_argument("--only-groups", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
