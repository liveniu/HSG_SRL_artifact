#!/usr/bin/env python3
"""Train the paper principal frozen residual for one site.

This isolated wrapper follows the recorded principal-run command, not the
library argparse defaults (r_max 1.5 / max_pair_dist 3.0).

Per seed, after the frozen checkpoint is written, this script also:
  1. Fits the label-free residual acceptor (Appendix C selector) on the *val* split
     only (no test keys, no GT identities) via
     ``pipeline/fit_residual_acceptor.py``, including the deployed-mask
     D_S recheck, writing
     ``results/runs/<site>/acceptor/<site>_seed{N}.json``.
  2. Remerges a per-seed evaluable copy of the detection DB with
     ``pipeline/multi_camera_trajectory_fusion.py --remerge``, applying the
     seed checkpoint + accepted T_c + that seed's acceptor JSON so
     ``final_world_x/y`` = P_final is populated for
     ``scripts/eval_paper_tables.py``.
  3. Remerges one geometric baseline copy ``<site>_baseline.db`` (no
     checkpoint) so Table IV can score HSG-Geo identities on ``world``.

Use --skip-acceptor / --skip-remerge to stop after the checkpoint if you
only want the training step (e.g. for a from-scratch reproduction check).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import ARTIFACT_ROOT, SRC_ROOT  # noqa: E402


def _load_json(rel: str) -> dict:
    return json.loads((ARTIFACT_ROOT / rel).read_text(encoding="utf-8"))


def _run(cmd: list[str], log: Path, dry_run: bool) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(" ".join(cmd) + "\n\n", encoding="utf-8")
    print(" ".join(cmd), flush=True)
    if dry_run:
        return 0
    with log.open("a", encoding="utf-8") as f:
        return subprocess.call(cmd, cwd=str(SRC_ROOT), stdout=f, stderr=subprocess.STDOUT)


def main() -> int:
    catalog = _load_json("configs/paper_sites.json")
    offsets = _load_json("configs/paper_time_offsets.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", choices=catalog["principal_sites"], required=True)
    parser.add_argument("--db", required=True, help="Detection/remerge SQLite (no GT mining)")
    parser.add_argument("--batch", default=catalog["batch_id"])
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--skip-tc", action="store_true")
    parser.add_argument(
        "--skip-acceptor", action="store_true",
        help="Stop after the checkpoint; do not fit the residual acceptor.",
    )
    parser.add_argument(
        "--skip-remerge", action="store_true",
        help="Fit the acceptor but do not write a per-seed evaluable DB copy.",
    )
    parser.add_argument(
        "--min-val-pairs", type=int, default=32,
        help="N_min for the acceptor thin-val rule (paper default: 32).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    site_cfg = catalog["sites"][args.site]
    out_dir = ARTIFACT_ROOT / "results" / "runs" / args.site
    ckpt_dir = out_dir / "training" / "frozen_checkpoints"
    log_dir = out_dir / "logs"
    for path in (out_dir, ckpt_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    offset_path = out_dir / "time_offsets.json"
    offset_path.write_text(json.dumps(offsets[args.site], indent=2) + "\n", encoding="utf-8")
    tc_json = out_dir / "training" / f"{args.site}_preflight_Tc.json"
    commands: list[dict[str, str]] = []

    if not args.skip_tc:
        tc_cmd = [
            sys.executable,
            str(SRC_ROOT / "evaluation" / "fit_fab2_low_order.py"),
            "--db",
            args.db,
            "--batch",
            args.batch,
            "--scene",
            site_cfg["scene_id"],
            "--split-manifest",
            args.split_manifest,
            "--train-splits",
            "train",
            "--out",
            str(tc_json),
            "--work-dir",
            str(out_dir / "work" / "tc"),
            "--time-offsets-json",
            str(offset_path),
            "--cameras",
            ",".join(site_cfg["cameras"]),
        ]
        commands.append(
            {
                "stage": "low_order_preflight",
                "seed": "",
                "artifact": str(tc_json.relative_to(ARTIFACT_ROOT)).replace("\\", "/"),
                "command": " ".join(tc_cmd),
                "status": "dry_run" if args.dry_run else "queued",
            }
        )
        rc = _run(tc_cmd, log_dir / f"tc_{args.site}.log", args.dry_run)
        commands[-1]["status"] = "dry_run" if args.dry_run else ("completed" if rc == 0 else "failed")
        if rc != 0:
            _write_command_csv(out_dir, commands)
            return rc

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for seed in seeds:
        ckpt = ckpt_dir / f"{args.site}_seed{seed}.pt"
        train_cmd = [
            sys.executable,
            str(SRC_ROOT / "pipeline" / "geometry_guided_residual_mlp.py"),
            "train",
            "--db",
            args.db,
            "--batch",
            args.batch,
            "--scene",
            site_cfg["scene_id"],
            "--seed",
            str(seed),
            "--split-manifest",
            args.split_manifest,
            "--train-splits",
            "train",
            "--epochs",
            str(args.epochs),
            "--max-residual-m",
            "1.0",
            "--max-pair-dist-m",
            "2",
            "--lambda-reg",
            "0.10",
            "--lambda-smooth",
            "0.02",
            "--min-teacher-confidence",
            "0.20",
            "--min-teacher-confidence-gap",
            "0.05",
            "--teacher-confidence-power",
            "1.0",
            "--student-weight-floor",
            "0.05",
            "--reg-confidence-floor",
            "0.25",
            "--lambda-mean-delta",
            "1.0",
            "--lambda-pair-center",
            "0.5",
            "--risk-scalar",
            "metric_jacobian",
            "--anchor-mode",
            "multiplicative_gating_offsetted",
            "--motion-compensation",
            "linear_interpolate",
            "--loss-mode",
            "asymmetric_teacher",
            "--time-offsets-json",
            str(offset_path),
            "--cameras",
            ",".join(site_cfg["cameras"]),
            "--out",
            str(ckpt),
        ]
        if tc_json.exists() or args.dry_run:
            train_cmd.extend(["--camera-translation-json", str(tc_json)])
        commands.append(
            {
                "stage": "train_frozen_checkpoint",
                "seed": str(seed),
                "artifact": str(ckpt.relative_to(ARTIFACT_ROOT)).replace("\\", "/"),
                "command": " ".join(train_cmd),
                "status": "dry_run" if args.dry_run else "queued",
            }
        )
        rc = _run(train_cmd, log_dir / f"train_{args.site}_seed{seed}.log", args.dry_run)
        commands[-1]["status"] = "dry_run" if args.dry_run else ("completed" if rc == 0 else "failed")
        if rc != 0:
            _write_command_csv(out_dir, commands)
            return rc

        if args.skip_acceptor:
            continue

        acceptor_json = out_dir / "acceptor" / f"{args.site}_seed{seed}.json"
        acceptor_cmd = [
            sys.executable,
            str(SRC_ROOT / "pipeline" / "fit_residual_acceptor.py"),
            "--db", args.db,
            "--scene", site_cfg["scene_id"],
            "--batch", args.batch,
            "--checkpoint", str(ckpt),
            "--split-manifest", args.split_manifest,
            "--tc-json", str(tc_json),
            "--out", str(acceptor_json),
            "--min-pairs", str(args.min_val_pairs),
            "--seed", str(seed),
        ]
        commands.append(
            {
                "stage": "fit_residual_acceptor",
                "seed": str(seed),
                "artifact": str(acceptor_json.relative_to(ARTIFACT_ROOT)).replace("\\", "/"),
                "command": " ".join(acceptor_cmd),
                "status": "dry_run" if args.dry_run else "queued",
            }
        )
        rc = _run(acceptor_cmd, log_dir / f"acceptor_{args.site}_seed{seed}.log", args.dry_run)
        commands[-1]["status"] = "dry_run" if args.dry_run else ("completed" if rc == 0 else "failed")
        if rc != 0:
            _write_command_csv(out_dir, commands)
            return rc

        if args.skip_remerge:
            continue

        eval_db = out_dir / "eval" / f"{args.site}_seed{seed}.db"
        eval_db.parent.mkdir(parents=True, exist_ok=True)
        if not args.dry_run:
            shutil.copy2(args.db, eval_db)
        remerge_cmd = [
            sys.executable,
            str(SRC_ROOT / "pipeline" / "multi_camera_trajectory_fusion.py"),
            "--config", str(ARTIFACT_ROOT / site_cfg["config"]),
            "--db", str(eval_db),
            "--scene", site_cfg["scene_id"],
            "--batch", args.batch,
            "--remerge",
            "--g2-checkpoint", str(ckpt),
            "--remerge-g2-for-merge",
        ]
        if tc_json.exists() or args.dry_run:
            remerge_cmd += ["--g2-camera-translation-json", str(tc_json)]
        remerge_cmd += ["--g2-acceptor-json", str(acceptor_json)]
        commands.append(
            {
                "stage": "remerge_seed_eval_db",
                "seed": str(seed),
                "artifact": str(eval_db.relative_to(ARTIFACT_ROOT)).replace("\\", "/"),
                "command": " ".join(remerge_cmd),
                "status": "dry_run" if args.dry_run else "queued",
            }
        )
        rc = _run(remerge_cmd, log_dir / f"remerge_{args.site}_seed{seed}.log", args.dry_run)
        commands[-1]["status"] = "dry_run" if args.dry_run else ("completed" if rc == 0 else "failed")
        if rc != 0:
            _write_command_csv(out_dir, commands)
            return rc

    if not (args.skip_acceptor or args.skip_remerge):
        baseline_db = out_dir / "eval" / f"{args.site}_baseline.db"
        baseline_db.parent.mkdir(parents=True, exist_ok=True)
        if not args.dry_run:
            shutil.copy2(args.db, baseline_db)
        baseline_cmd = [
            sys.executable,
            str(SRC_ROOT / "pipeline" / "multi_camera_trajectory_fusion.py"),
            "--config", str(ARTIFACT_ROOT / site_cfg["config"]),
            "--db", str(baseline_db),
            "--scene", site_cfg["scene_id"],
            "--batch", args.batch,
            "--remerge",
        ]
        commands.append(
            {
                "stage": "remerge_geo_baseline_eval_db",
                "seed": "",
                "artifact": str(baseline_db.relative_to(ARTIFACT_ROOT)).replace("\\", "/"),
                "command": " ".join(baseline_cmd),
                "status": "dry_run" if args.dry_run else "queued",
            }
        )
        rc = _run(baseline_cmd, log_dir / f"remerge_{args.site}_baseline.log", args.dry_run)
        commands[-1]["status"] = "dry_run" if args.dry_run else ("completed" if rc == 0 else "failed")
        if rc != 0:
            _write_command_csv(out_dir, commands)
            return rc

    _write_command_csv(out_dir, commands)
    print(f"command log -> {out_dir / 'train_fab2_site_frozen_commands.csv'}")
    if not args.skip_acceptor:
        print(f"acceptor JSONs -> {out_dir / 'acceptor'}")
    if not (args.skip_acceptor or args.skip_remerge):
        print(f"per-seed evaluable DBs (P_final populated) -> {out_dir / 'eval'}")
        print(f"geometric baseline eval DB (P_geo identities) -> {out_dir / 'eval' / (args.site + '_baseline.db')}")
    return 0


def _write_command_csv(out_dir: Path, rows: list[dict[str, str]]) -> None:
    dest = out_dir / "train_fab2_site_frozen_commands.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["stage", "seed", "artifact", "command", "status"]
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
