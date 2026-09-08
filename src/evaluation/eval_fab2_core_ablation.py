#!/usr/bin/env python3
# Vendored HSG-SRL artifact copy. Isolated from the development repo;
# edit this file only. See src/SOURCE_MAP.csv.
"""Generate and optionally run FAB2 core ablation jobs."""

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


PLAN_FIELDS = [
    "ablation_group",
    "variant",
    "status",
    "reason",
    "seed",
    "command",
    "expected_artifact",
]


def _train_cmd(args: argparse.Namespace, variant: dict[str, Any], seed: int, out_dir: Path) -> list[str]:
    out = out_dir / "training" / "frozen_checkpoints" / f"{args.scene.replace(',', '_')}_{variant['variant']}_seed{seed}.pt"
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
        str(out_dir),
        "--seeds",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--max-pair-dist-m",
        str(args.max_pair_dist_m),
        "--min-mvc-pairs",
        str(args.min_mvc_pairs),
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
    if args.dry_run:
        cmd.append("--dry-run")
    if args.skip_low_order:
        cmd.append("--skip-low-order")
    if args.cameras:
        cmd.extend(["--cameras", args.cameras])
    for flag, value in variant.get("flags", []):
        cmd.extend([flag, str(value)])
    # The train wrapper chooses its own checkpoint path. Keep this artifact
    # field as a stable run-directory hint for the ablation manifest.
    return cmd


def _variants() -> list[dict[str, Any]]:
    return [
        {
            "ablation_group": "risk_scalar",
            "variant": "metric_jacobian_default",
            "status": "ready",
            "reason": "FAB2 default C1 risk scalar.",
            "flags": [("--risk-scalar", "metric_jacobian")],
        },
        {
            "ablation_group": "risk_scalar",
            "variant": "pixel_orthogonal",
            "status": "ready",
            "reason": "Pixel horizon-distance risk baseline.",
            "flags": [("--risk-scalar", "pixel_orthogonal")],
        },
        {
            "ablation_group": "anchor_gate",
            "variant": "bottom_center",
            "status": "ready",
            "reason": "Fixed IPM bottom-center anchor.",
            "flags": [("--anchor-mode", "bottom_center")],
        },
        {
            "ablation_group": "anchor_gate",
            "variant": "center",
            "status": "ready",
            "reason": "Box center anchor.",
            "flags": [("--anchor-mode", "center")],
        },
        {
            "ablation_group": "teacher",
            "variant": "symmetric_weighted",
            "status": "ready",
            "reason": "Symmetric MVC consistency baseline.",
            "flags": [("--loss-mode", "symmetric_weighted")],
        },
        {
            "ablation_group": "motion_compensation",
            "variant": "motion_off",
            "status": "ready",
            "reason": "Nearest/off MVC pair construction.",
            "flags": [("--motion-compensation", "off")],
        },
        {
            "ablation_group": "residual_bound",
            "variant": "unbounded_residual",
            "status": "requires_unbounded_head",
            "reason": "Current residual head always applies tanh*r_max; no strict unbounded CLI exists.",
            "flags": [],
        },
    ]


def run(args: argparse.Namespace) -> None:
    out_root = ensure_dir(Path(args.out_dir))
    ivd = ensure_dir(out_root / "iv_d")
    seeds = [int(s) for s in split_csv_arg(args.seeds)] or [0]
    selected = set(split_csv_arg(args.only_variants))
    rows: list[dict[str, Any]] = []
    for variant in _variants():
        if selected and variant["variant"] not in selected and variant["ablation_group"] not in selected:
            continue
        for seed in seeds:
            if variant["status"] != "ready":
                rows.append(
                    {
                        "ablation_group": variant["ablation_group"],
                        "variant": variant["variant"],
                        "status": variant["status"],
                        "reason": variant["reason"],
                        "seed": seed,
                        "command": "",
                        "expected_artifact": "",
                    }
                )
                continue
            artifact_dir = out_root / variant["variant"] / f"seed{seed}"
            cmd = _train_cmd(args, variant, seed, artifact_dir)
            rows.append(
                {
                    "ablation_group": variant["ablation_group"],
                    "variant": variant["variant"],
                    "status": "dry_run" if args.dry_run else ("queued" if not args.execute else "completed"),
                    "reason": variant["reason"],
                    "seed": seed,
                    "command": command_to_str(cmd),
                    "expected_artifact": str(artifact_dir),
                }
            )
            if args.execute:
                run_command(cmd, dry_run=args.dry_run, cwd=_ROOT)
    write_csv(ivd / "core_ablation_plan.csv", rows, PLAN_FIELDS)
    write_json(ivd / "core_ablation_summary.json", {"seeds": seeds, "execute": args.execute, "variants": len(rows)})
    write_run_metadata(out_root, "eval_fab2_core_ablation", args, {"seeds": seeds})
    print(f"Core ablation plan written under {ivd}")


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
    parser.add_argument("--only-variants", default="", help="Comma-separated variant or group names")
    parser.add_argument("--execute", action="store_true", help="Run commands instead of only writing a plan")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
