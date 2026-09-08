#!/usr/bin/env python3
# Vendored HSG-SRL artifact copy. Isolated from the development repo;
# edit this file only. See src/SOURCE_MAP.csv.
"""Run FAB2 site-level preflight, offline training and frozen checkpoint export."""

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
    checkpoint_meta,
    command_to_str,
    ensure_dir,
    run_command,
    split_csv_arg,
    write_csv,
    write_json,
    write_run_metadata,
)


COMMAND_FIELDS = [
    "stage",
    "seed",
    "artifact",
    "command",
    "status",
]

CHECKPOINT_FIELDS = [
    "method",
    "scene",
    "batch_id",
    "seed",
    "checkpoint_path",
    "exists",
    "sha256",
    "bytes",
    "feature_version",
    "risk_scalar_mode",
    "anchor_mode",
    "max_residual_m",
    "max_pairs",
    "mvc_pair_count",
    "raw_mvc_pair_count",
]


def _scene_slug(scene: str) -> str:
    return scene.replace(",", "_").replace(" ", "").replace("/", "_").replace("\\", "_")


def _append_if_value(cmd: list[str], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    cmd.extend([flag, str(value)])


def _base_train_args(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(_ROOT / "pipeline" / "geometry_guided_residual_mlp.py"),
        "train",
        "--db",
        args.db,
        "--batch",
        args.batch,
        "--scene",
        args.scene,
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--lambda-reg",
        str(args.lambda_reg),
        "--lambda-smooth",
        str(args.lambda_smooth),
        "--lambda-mean-delta",
        str(args.lambda_mean_delta),
        "--lambda-pair-center",
        str(args.lambda_pair_center),
        "--max-residual-m",
        str(args.max_residual_m),
        "--max-time-gap",
        str(args.max_time_gap),
        "--risk-scalar",
        args.risk_scalar,
        "--anchor-mode",
        args.anchor_mode,
        "--s0-m",
        str(args.s0_m),
        "--feature-l0-m",
        str(args.feature_l0_m),
        "--anchor-shape-eta",
        str(args.anchor_shape_eta),
        "--motion-compensation",
        args.motion_compensation,
        "--loss-mode",
        args.loss_mode,
        "--min-teacher-confidence",
        str(args.min_teacher_confidence),
        "--min-teacher-confidence-gap",
        str(args.min_teacher_confidence_gap),
        "--max-pair-dist-m",
        str(args.max_pair_dist_m),
        "--min-mvc-pairs",
        str(args.min_mvc_pairs),
        "--frame-width",
        str(args.frame_width),
        "--frame-height",
        str(args.frame_height),
        "--clip-margin",
        str(args.clip_margin),
        "--clip-margin-ratio",
        str(args.clip_margin_ratio),
    ]
    _append_if_value(cmd, "--cameras", args.cameras)
    _append_if_value(cmd, "--overlap-pairs", args.overlap_pairs)
    _append_if_value(cmd, "--time-offsets-json", args.time_offsets_json)
    _append_if_value(cmd, "--split-manifest", args.split_manifest)
    _append_if_value(cmd, "--train-splits", args.train_splits)
    _append_if_value(cmd, "--time-start", args.time_start)
    _append_if_value(cmd, "--time-end", args.time_end)
    _append_if_value(cmd, "--max-pairs", args.max_pairs)
    _append_if_value(cmd, "--metric-sensitivity-clip-m", args.sensitivity_clip_m)
    _append_if_value(cmd, "--extent-clip-m", args.extent_clip_m)
    if args.allow_gt_identity_mining:
        cmd.append("--allow-gt-identity-mining")
    if args.allow_unsplit_training:
        cmd.append("--allow-unsplit-training")
    return cmd


def _low_order_command(args: argparse.Namespace, out_json: Path) -> list[str]:
    iterative = args.preflight_iters > 1 or args.preflight_wide_window_m is not None
    if iterative:
        return _low_order_iterative_command(args, out_json)
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
        "--min-pairs",
        str(args.preflight_min_pairs),
        "--max-pair-dist-m",
        str(args.max_pair_dist_m),
        "--min-teacher-confidence",
        str(args.min_teacher_confidence),
        "--min-teacher-confidence-gap",
        str(args.min_teacher_confidence_gap),
        "--max-time-gap",
        str(args.max_time_gap),
        "--motion-compensation",
        args.motion_compensation,
        "--risk-scalar",
        args.risk_scalar,
        "--anchor-mode",
        args.anchor_mode,
        "--s0-m",
        str(args.s0_m),
        "--feature-l0-m",
        str(args.feature_l0_m),
        "--anchor-shape-eta",
        str(args.anchor_shape_eta),
    ]
    _append_if_value(cmd, "--cameras", args.cameras)
    _append_if_value(cmd, "--overlap-pairs", args.overlap_pairs)
    _append_if_value(cmd, "--time-offsets-json", args.time_offsets_json)
    _append_if_value(cmd, "--split-manifest", args.split_manifest)
    _append_if_value(cmd, "--train-splits", args.train_splits)
    _append_if_value(cmd, "--time-start", args.time_start)
    _append_if_value(cmd, "--time-end", args.time_end)
    _append_if_value(cmd, "--max-pairs", args.max_pairs)
    if args.allow_gt_identity_mining:
        cmd.append("--allow-gt-identity-mining")
    if args.allow_unsplit_training:
        cmd.append("--allow-unsplit-fitting")
    if args.write_db_metadata:
        cmd.append("--write-db-metadata")
    return cmd


def _low_order_iterative_command(args: argparse.Namespace, out_json: Path) -> list[str]:
    """Coarse-to-fine preflight via evaluation/fit_fab2_low_order.py (plan 9.2)."""
    if args.write_db_metadata:
        raise SystemExit("--write-db-metadata is not supported with iterative preflight")
    cmd = [
        sys.executable,
        str(_ROOT / "evaluation" / "fit_fab2_low_order.py"),
        "--db",
        args.db,
        "--batch",
        args.batch,
        "--scene",
        args.scene,
        "--out",
        str(out_json),
        "--iters",
        str(args.preflight_iters),
        "--converge-m",
        str(args.preflight_converge_m),
        "--min-pairs",
        str(args.preflight_min_pairs),
        "--max-pair-dist-m",
        str(args.max_pair_dist_m),
        "--min-teacher-confidence",
        str(args.min_teacher_confidence),
        "--min-teacher-confidence-gap",
        str(args.min_teacher_confidence_gap),
        "--max-time-gap",
        str(args.max_time_gap),
        "--motion-compensation",
        args.motion_compensation,
        "--risk-scalar",
        args.risk_scalar,
        "--anchor-mode",
        args.anchor_mode,
        "--s0-m",
        str(args.s0_m),
        "--feature-l0-m",
        str(args.feature_l0_m),
        "--anchor-shape-eta",
        str(args.anchor_shape_eta),
    ]
    _append_if_value(cmd, "--wide-window-m", args.preflight_wide_window_m)
    _append_if_value(cmd, "--cameras", args.cameras)
    _append_if_value(cmd, "--overlap-pairs", args.overlap_pairs)
    _append_if_value(cmd, "--time-offsets-json", args.time_offsets_json)
    _append_if_value(cmd, "--split-manifest", args.split_manifest)
    _append_if_value(cmd, "--train-splits", args.train_splits)
    _append_if_value(cmd, "--time-start", args.time_start)
    _append_if_value(cmd, "--time-end", args.time_end)
    _append_if_value(cmd, "--max-pairs", args.max_pairs)
    if args.allow_gt_identity_mining:
        cmd.append("--allow-gt-identity-mining")
    if args.allow_unsplit_training:
        cmd.append("--allow-unsplit-fitting")
    return cmd


def run(args: argparse.Namespace) -> None:
    if args.max_pairs is not None and args.max_pairs < 1:
        raise SystemExit("--max-pairs must be >= 1 when provided")
    if args.time_start is not None and args.time_end is not None and args.time_start >= args.time_end:
        raise SystemExit("--time-start must be smaller than --time-end")
    if not args.split_manifest and not args.allow_unsplit_training:
        raise SystemExit(
            "--split-manifest is required for FAB2 frozen training. "
            "Pass --allow-unsplit-training only for legacy diagnostics."
        )
    out_root = ensure_dir(Path(args.out_dir))
    train_dir = ensure_dir(out_root / "training")
    ckpt_dir = ensure_dir(train_dir / "frozen_checkpoints")
    scene_slug = _scene_slug(args.scene)
    commands: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []

    correction_json = Path(args.camera_translation_json) if args.camera_translation_json else train_dir / f"{scene_slug}_preflight_Tc.json"
    if not args.skip_low_order and not args.camera_translation_json:
        cmd = _low_order_command(args, correction_json)
        status = "dry_run" if args.dry_run else "completed"
        commands.append({"stage": "low_order_preflight", "seed": "", "artifact": str(correction_json), "command": command_to_str(cmd), "status": status})
        run_command(cmd, dry_run=args.dry_run, cwd=_ROOT)
    elif args.camera_translation_json:
        commands.append({"stage": "low_order_preflight", "seed": "", "artifact": str(correction_json), "command": "external_json", "status": "external"})
    else:
        commands.append({"stage": "low_order_preflight", "seed": "", "artifact": "", "command": "skipped", "status": "skipped"})

    seeds = [int(s) for s in split_csv_arg(args.seeds)] or [0]
    base = _base_train_args(args)
    for seed in seeds:
        ckpt = ckpt_dir / f"{scene_slug}_seed{seed}.pt"
        cmd = list(base)
        cmd.extend(["--seed", str(seed), "--out", str(ckpt)])
        if not args.skip_low_order and not args.no_apply_low_order:
            cmd.extend(["--camera-translation-json", str(correction_json)])
            cmd.extend(["--camera-translation-acceptance", args.camera_translation_acceptance])
        if args.skip_train_preflight_diagnostic:
            cmd.append("--skip-preflight-diagnostic")
        status = "dry_run" if args.dry_run else "completed"
        commands.append({"stage": "train_frozen_checkpoint", "seed": seed, "artifact": str(ckpt), "command": command_to_str(cmd), "status": status})
        run_command(cmd, dry_run=args.dry_run, cwd=_ROOT)
        meta = checkpoint_meta(ckpt) if not args.dry_run else {"exists": False, "path": str(ckpt)}
        checkpoint_rows.append(
            {
                "method": "Ours-HSG-SRL-Frozen",
                "scene": args.scene,
                "batch_id": args.batch,
                "seed": seed,
                "checkpoint_path": meta.get("path", str(ckpt)),
                "exists": int(bool(meta.get("exists"))),
                "sha256": meta.get("sha256", ""),
                "bytes": meta.get("bytes", ""),
                "feature_version": meta.get("feature_version", ""),
                "risk_scalar_mode": meta.get("risk_scalar_mode", ""),
                "anchor_mode": meta.get("anchor_mode", ""),
                "max_residual_m": meta.get("max_residual_m", ""),
                "max_pairs": meta.get("max_pairs", ""),
                "mvc_pair_count": meta.get("mvc_pair_count", ""),
                "raw_mvc_pair_count": meta.get("raw_mvc_pair_count", ""),
            }
        )

    write_csv(train_dir / "train_fab2_site_frozen_commands.csv", commands, COMMAND_FIELDS)
    write_csv(train_dir / "frozen_checkpoint_manifest.csv", checkpoint_rows, CHECKPOINT_FIELDS)
    write_json(
        train_dir / "train_fab2_site_frozen_summary.json",
        {
            "scene": args.scene,
            "batch_id": args.batch,
            "seeds": seeds,
            "correction_json": str(correction_json) if not args.skip_low_order else "",
            "checkpoints": [r["checkpoint_path"] for r in checkpoint_rows],
        },
    )
    write_run_metadata(out_root, "train_fab2_site_frozen", args, {"seeds": seeds})
    print(f"FAB2 frozen training artifacts written under {train_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--scene", required=True, help="Comma-separated scene ids for one site model")
    parser.add_argument("--out-dir", default=str(FAB2_DEFAULT_RESULTS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-low-order", action="store_true")
    parser.add_argument("--no-apply-low-order", action="store_true")
    parser.add_argument("--camera-translation-json", default=None, help="Use an existing correction JSON")
    parser.add_argument("--camera-translation-acceptance", choices=("recommended", "caution", "any"), default="recommended")
    parser.add_argument("--write-db-metadata", action="store_true")
    parser.add_argument("--skip-train-preflight-diagnostic", action="store_true")
    parser.add_argument("--split-manifest", default=None, help="FAB2 gt_index.csv or dataset_split_manifest.csv")
    parser.add_argument("--train-splits", default="train")
    parser.add_argument("--allow-unsplit-training", action="store_true", help="Legacy diagnostic only: allow preflight/train without split manifest")
    parser.add_argument("--time-start", type=float, default=None)
    parser.add_argument("--time-end", type=float, default=None)
    parser.add_argument("--allow-gt-identity-mining", action="store_true", help="Only for supervised smoke tests")
    parser.add_argument("--cameras", default="")
    parser.add_argument("--overlap-pairs", default="")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-reg", type=float, default=0.05)
    parser.add_argument("--lambda-smooth", type=float, default=0.02)
    parser.add_argument(
        "--lambda-mean-delta",
        type=float,
        default=1.0,
        help="Appendix E R_mean weight. Trainer argparse default is 0 (off); this wrapper forwards the paper value.",
    )
    parser.add_argument(
        "--lambda-pair-center",
        type=float,
        default=0.5,
        help="Appendix E R_pair weight. Trainer argparse default is 0 (off); this wrapper forwards the paper value.",
    )
    parser.add_argument("--max-residual-m", type=float, default=1.5)
    parser.add_argument("--max-time-gap", type=float, default=0.35)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional diagnostic cap on retained MVC pairs after confidence/gap filtering.",
    )
    parser.add_argument("--risk-scalar", choices=("metric_jacobian", "pixel_orthogonal", "pixel_vertical", "grazing_angle"), default="metric_jacobian")
    parser.add_argument("--anchor-mode", choices=("multiplicative_gating_offsetted", "bottom_center", "center"), default="multiplicative_gating_offsetted")
    parser.add_argument("--s0-m", type=float, default=60.0)
    parser.add_argument("--feature-l0-m", type=float, default=10.0)
    parser.add_argument("--anchor-shape-eta", type=float, default=0.75)
    parser.add_argument("--motion-compensation", choices=("off", "nearest", "linear_interpolate"), default="linear_interpolate")
    parser.add_argument("--loss-mode", choices=("asymmetric_teacher", "symmetric_weighted"), default="asymmetric_teacher")
    parser.add_argument("--min-teacher-confidence", type=float, default=0.20)
    parser.add_argument("--min-teacher-confidence-gap", type=float, default=0.05)
    parser.add_argument("--max-pair-dist-m", type=float, default=3.0)
    parser.add_argument("--min-mvc-pairs", type=int, default=64)
    parser.add_argument("--preflight-min-pairs", type=int, default=32)
    parser.add_argument(
        "--preflight-iters", type=int, default=8,
        help="Outer fit-apply-refit rounds for the low-order preflight; 1 = legacy single fit",
    )
    parser.add_argument(
        "--preflight-wide-window-m", type=float, default=None,
        help="Round-0 pairing window in meters (coarse-to-fine); use when large disagreement is suspected",
    )
    parser.add_argument("--preflight-converge-m", type=float, default=0.02)
    parser.add_argument("--frame-width", type=int, default=1920)
    parser.add_argument("--frame-height", type=int, default=1080)
    parser.add_argument("--clip-margin", type=int, default=3)
    parser.add_argument("--clip-margin-ratio", type=float, default=0.01)
    parser.add_argument("--time-offsets-json", default=None)
    parser.add_argument("--sensitivity-clip-m", type=float, default=None)
    parser.add_argument("--extent-clip-m", type=float, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
