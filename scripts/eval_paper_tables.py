#!/usr/bin/env python3
"""Print the core evaluation commands for the paper tables.

This wrapper does not invent a second evaluation path. It calls the vendored
evaluation modules with paper-explicit flags for Tables III, IV, V/VII, and
IX. Table VI, Table VIII, and Table X use the specialized entries listed in
results/tables/TABLE_TO_SCRIPT.csv.

Table III needs two localization passes, both appended (via upsert) into
``localization_by_method.csv``:

  * HSG-Geo   -- ``--world-coords p_geo``, plain ``--db`` (world_x/y always
    stores P_geo, so this needs no checkpoint or remerge).
  * HSG-SRL-Frozen (\"ours\") -- ``--world-coords final`` against the
    per-seed *remerged* DB from ``scripts/train_paper_sites.py``
    (``results/runs/<site>/eval/<site>_seed{N}.db``), which has
    ``final_world_x/y`` = P_final populated from the frozen checkpoint +
    accepted T_c + that seed's residual acceptor JSON.

Table IV is ``eval_fab2_bev_hota.py`` (Hungarian DetA/AssA/LocA/HOTA on the
label-clock grid). It needs the origin GT sqlite and the det-gt match
report in addition to the remerged DBs. ``eval_fab2_tracking.py`` is a
coarser ``hota_proxy`` and is not invoked here.

Baseline-IPM (bottom-center anchor, no homography-sensitivity guidance) is
not produced by this generic evaluator: it needs the anchor_mode=bottom_center
recomputation path documented in docs/CODE_TO_PAPER.md, not a stored
DB column.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import ARTIFACT_ROOT, SRC_ROOT  # noqa: E402


def _load_sites() -> dict:
    return json.loads(
        (ARTIFACT_ROOT / "configs" / "paper_sites.json").read_text(encoding="utf-8")
    )


def _print_and_maybe_run(cmd: list[str], execute: bool) -> int:
    print(" ".join(cmd), flush=True)
    if not execute:
        return 0
    return subprocess.call(cmd, cwd=str(SRC_ROOT))


def _loc_cmd(*, args, site, out, world_coords: str, method: str, db: str) -> list[str]:
    return [
        sys.executable,
        str(SRC_ROOT / "evaluation" / "eval_fab2_localization.py"),
        "--db", db,
        "--batch", args.batch,
        "--scene", site["scene_id"],
        "--gt-index", args.gt_index,
        "--split", "test",
        "--world-coords", world_coords,
        "--method", method,
        "--out-dir", str(out / "table_iii"),
    ]


def _hota_cmd(*, args, site, out, coord: str, method: str, pred_db: str,
              cameras: str = "") -> list[str]:
    cmd = [
        sys.executable,
        str(SRC_ROOT / "evaluation" / "eval_fab2_bev_hota.py"),
        "--pred-db", pred_db,
        "--gt-db", args.origin_db,
        "--scene", site["scene_id"],
        "--pred-batch", args.batch,
        "--gt-batch", args.gt_batch,
        "--split-manifest", args.gt_index,
        "--match-report", args.match_report,
        "--coord", coord,
        "--method", method,
        "--out-dir", str(out / "table_iv"),
    ]
    if args.reference_verdict:
        cmd += ["--reference-verdict", args.reference_verdict]
    max_t = args.max_t
    if max_t is None:
        max_t = site.get("identity_max_t_sec")
    if max_t is not None:
        cmd += ["--max-t", str(max_t)]
    if cameras:
        cmd += ["--cameras", cameras]
    return cmd


def main() -> int:
    catalog = _load_sites()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", choices=catalog["principal_sites"], required=True)
    parser.add_argument(
        "--db", required=True,
        help="Detection DB (world_x/y = P_geo). Used as-is for the HSG-Geo row.",
    )
    parser.add_argument(
        "--eval-db", default=None,
        help=(
            "Per-seed remerged DB with final_world_x/y (P_final) populated, "
            "i.e. results/runs/<site>/eval/<site>_seed{N}.db from "
            "scripts/train_paper_sites.py. Required for the HSG-SRL-Frozen "
            "row unless --skip-ours is passed. Defaults to "
            "results/runs/<site>/eval/<site>_seed{seed}.db."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used to default --eval-db.")
    parser.add_argument("--batch", default=catalog["batch_id"])
    parser.add_argument("--gt-index", required=True, help="Frozen gt_index.csv or split manifest")
    parser.add_argument(
        "--origin-db", default=None,
        help="Origin GT sqlite from the importer (Table IV). Typically data/work/origin_<site>.db.",
    )
    parser.add_argument(
        "--match-report", default=None,
        help="det_gt_match_report.json from eval_fab2_det_gt_match.py (Table IV time maps).",
    )
    parser.add_argument("--gt-batch", default="alpha_gt_001")
    parser.add_argument("--reference-verdict", default=None)
    parser.add_argument("--max-t", type=float, default=None)
    parser.add_argument(
        "--geo-eval-db", default=None,
        help=(
            "Geometric-remerge DB for the Table IV HSG-Geo row "
            "(results/runs/<site>/eval/<site>_baseline.db). Defaults next to --eval-db."
        ),
    )
    parser.add_argument("--skip-geo", action="store_true", help="Skip the HSG-Geo (P_geo) row.")
    parser.add_argument(
        "--skip-ours", action="store_true",
        help="Skip the HSG-SRL-Frozen (P_final) row; use when only --db is available.",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    site = catalog["sites"][args.site]
    out = ARTIFACT_ROOT / "results" / "runs" / args.site
    out.mkdir(parents=True, exist_ok=True)

    eval_db = args.eval_db or str(
        ARTIFACT_ROOT / "results" / "runs" / args.site / "eval" / f"{args.site}_seed{args.seed}.db"
    )
    geo_eval_db = args.geo_eval_db or str(
        ARTIFACT_ROOT / "results" / "runs" / args.site / "eval" / f"{args.site}_baseline.db"
    )

    loc_cmds: list[tuple[str, list[str]]] = []
    hota_cmds: list[tuple[str, list[str]]] = []
    table_iv_ready = bool(args.origin_db and args.match_report)
    if not args.skip_geo:
        loc_cmds.append(("HSG-Geo", _loc_cmd(args=args, site=site, out=out, world_coords="p_geo", method="HSG-Geo", db=args.db)))
        if table_iv_ready:
            hota_cmds.append((
                "HSG-Geo",
                _hota_cmd(args=args, site=site, out=out, coord="world",
                          method="HSG-Geo", pred_db=geo_eval_db),
            ))
    if not args.skip_ours:
        loc_cmds.append(("HSG-SRL-Frozen", _loc_cmd(args=args, site=site, out=out, world_coords="final", method="HSG-SRL-Frozen", db=eval_db)))
        if table_iv_ready:
            hota_cmds.append((
                "HSG-SRL-Frozen",
                _hota_cmd(args=args, site=site, out=out, coord="final_world",
                          method="HSG-SRL-Frozen", pred_db=eval_db),
            ))
    subgraph = site.get("mvc_supported_component") or []
    if (
        table_iv_ready
        and not args.skip_ours
        and subgraph
        and set(subgraph) != set(site.get("cameras") or [])
    ):
        hota_cmds.append((
            "HSG-SRL-Frozen-subgraph",
            _hota_cmd(
                args=args, site=site, out=out, coord="final_world",
                method="HSG-SRL-Frozen-subgraph", pred_db=eval_db,
                cameras=",".join(subgraph),
            ),
        ))
        if not args.skip_geo:
            hota_cmds.append((
                "HSG-Geo-subgraph",
                _hota_cmd(
                    args=args, site=site, out=out, coord="world",
                    method="HSG-Geo-subgraph", pred_db=geo_eval_db,
                    cameras=",".join(subgraph),
                ),
            ))

    abl = [
        sys.executable,
        str(SRC_ROOT / "evaluation" / "eval_fab2_core_ablation.py"),
        "--db",
        args.db,
        "--batch",
        args.batch,
        "--scene",
        site["scene_id"],
        "--split-manifest",
        args.gt_index,
        "--train-splits",
        "train",
        "--seeds",
        "0,1,2",
        "--max-pair-dist-m",
        "2",
        "--out-dir",
        str(out / "table_v_vii"),
    ]
    rec = [
        sys.executable,
        str(SRC_ROOT / "evaluation" / "run_fab2_perturbation_recovery.py"),
        "--db",
        args.db,
        "--batch",
        args.batch,
        "--scene",
        site["scene_id"],
        "--split-manifest",
        args.gt_index,
        "--gt-index",
        args.gt_index,
        "--seeds",
        "0,1,2",
        "--max-residual-m",
        "1.0",
        "--max-pair-dist-m",
        "2",
        "--out-dir",
        str(out / "table_ix"),
    ]
    print(f"# Table mapping for {args.site}")
    print(
        "# Baseline-IPM is not produced here; see docs/CODE_TO_PAPER.md "
        "(anchor_mode=bottom_center recomputation)."
    )
    print("# III localization -> eval_fab2_localization.py (upserts one row per --method)")
    for method, cmd in loc_cmds:
        print(f"#   method={method}")
        rc = _print_and_maybe_run(cmd, args.execute)
        if rc:
            return rc
    if table_iv_ready:
        print("# IV tracking -> eval_fab2_bev_hota.py (Hungarian DetA/AssA/LocA/HOTA)")
        for method, cmd in hota_cmds:
            print(f"#   method={method}")
            rc = _print_and_maybe_run(cmd, args.execute)
            if rc:
                return rc
    else:
        print(
            "# IV tracking skipped: pass --origin-db and --match-report "
            "(eval_fab2_bev_hota.py). Do not use eval_fab2_tracking.py hota_proxy. "
            "Recorded numbers: results/tables/table_iv_tracking.csv"
        )
    print("# V/VII ablations -> eval_fab2_core_ablation.py")
    rc = _print_and_maybe_run(abl, args.execute)
    if rc:
        return rc
    print("# VIII/IX translation gate -> recorded preflight JSON + run_fab2_perturbation_recovery.py")
    print(
        "# IX affine is the principal class (no Synthehicle camera_info). "
        "scale/rotation need --synthehicle-root or HSGSRL_SYNTHEHICLE_ROOT."
    )
    rc = _print_and_maybe_run(rec, args.execute)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
