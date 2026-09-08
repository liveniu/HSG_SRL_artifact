#!/usr/bin/env python3
# Vendored HSG-SRL artifact copy. Isolated from the development repo;
# edit this file only. See src/SOURCE_MAP.csv.
"""Table IV scorer: Hungarian BEV HOTA / LocA on a remerged trajectory DB.

This is the paper Table IV identity metric. It is *not*
``eval_fab2_tracking.py`` (that file's ``hota_proxy``).

Required inputs:

  * ``--pred-db``  geometrically or residual-remerged detection DB
  * ``--gt-db``    origin GT sqlite (importer batch, typically ``alpha_gt_001``)
  * ``--split-manifest``  test keys (``dataset_split_manifest.csv`` or ``gt_index.csv``)
  * ``--match-report``    ``det_gt_match_report.json`` (camera time maps)

Optional: ``--reference-verdict`` (bias applied only if the file starts with
``REFERENCE OFFSET``), ``--max-t`` (M6 recorded window is 360 s),
``--cameras`` (M6 C05--C06 footnote).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.bev_hota import (  # noqa: E402
    GATES,
    MAX_T,
    identity_metrics,
    load_gt_bins,
    load_predictions,
    load_split_keys,
    load_time_maps_from_match_report,
    mean_bev_hota,
    parse_reference_bias,
)
from evaluation.fab2_common import (  # noqa: E402
    FAB2_DEFAULT_RESULTS,
    ensure_dir,
    upsert_csv_rows,
    write_json,
    write_run_metadata,
)

IDENTITY_FIELDS = [
    "site",
    "method",
    "gate",
    "tp",
    "fp",
    "fn",
    "idsw",
    "deta",
    "assa",
    "hota",
    "loca",
    "idf1",
    "mota",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-db", required=True)
    parser.add_argument("--gt-db", required=True, help="Origin GT sqlite from the importer")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--pred-batch", required=True, help="Detection/remerge batch_id")
    parser.add_argument("--gt-batch", default="alpha_gt_001")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--match-report", required=True, help="det_gt_match_report.json")
    parser.add_argument(
        "--coord",
        choices=("world", "final_world"),
        default="world",
        help="world = P_geo after geometric remerge; final_world = P_final",
    )
    parser.add_argument("--method", default="HSG-Geo")
    parser.add_argument("--cameras", default="", help="Optional camera subset, e.g. C05,C06")
    parser.add_argument("--reference-verdict", default=None)
    parser.add_argument("--gt-bias", default=None, help="x,y override; wins over the verdict file")
    parser.add_argument("--max-t", type=float, default=None)
    parser.add_argument("--out-dir", default=str(FAB2_DEFAULT_RESULTS))
    args = parser.parse_args()

    cameras = {c.strip() for c in args.cameras.split(",") if c.strip()} or None
    test_keys = load_split_keys(Path(args.split_manifest), args.split)
    tm = load_time_maps_from_match_report(Path(args.match_report))

    bias = np.zeros(2)
    if args.gt_bias:
        parts = [p.strip() for p in args.gt_bias.split(",")]
        if len(parts) != 2:
            parser.error("--gt-bias must be x,y")
        bias = np.array([float(parts[0]), float(parts[1])])
    elif args.reference_verdict:
        bias = parse_reference_bias(Path(args.reference_verdict).read_text(encoding="utf-8"))

    pred, t0 = load_predictions(
        Path(args.pred_db),
        args.scene,
        args.coord,
        tm,
        test_keys,
        pred_batch=args.pred_batch,
        cameras=cameras,
    )
    max_t = args.max_t if args.max_t is not None else MAX_T.get(args.scene)
    gt_bins = load_gt_bins(
        Path(args.gt_db),
        args.scene,
        bias,
        t0,
        gt_batch=args.gt_batch,
        cameras=cameras,
        max_t=max_t,
    )

    rows = []
    header = (
        f"{'gate':>5} {'method':>16} {'DetA':>8} {'AssA':>8} "
        f"{'LocA':>8} {'HOTA':>8} {'IDF1':>8} {'MOTA':>8} {'IDsw':>6}"
    )
    print(header)
    for gate in GATES:
        rec = identity_metrics(gt_bins, pred, gate)
        rec_out = {"site": args.scene, "method": args.method, **rec}
        rows.append(rec_out)
        print(
            f"{gate:5.1f} {args.method:>16} {rec['deta']:8.4f} {rec['assa']:8.4f} "
            f"{rec['loca']:8.4f} {rec['hota']:8.4f} {rec['idf1']:8.4f} "
            f"{rec['mota']:8.4f} {rec['idsw']:6d}"
        )
    bev = mean_bev_hota(rows)
    print(f"{args.method:>16}  BEV-HOTA = {bev:.4f}")

    out_root = ensure_dir(Path(args.out_dir))
    csv_path = upsert_csv_rows(
        out_root / "identity_metrics.csv",
        rows,
        IDENTITY_FIELDS,
        key_fields=["site", "method", "gate"],
    )
    write_json(
        out_root / f"identity_{args.method.lower().replace(' ', '_').replace('-', '_')}.json",
        {
            "site": args.scene,
            "method": args.method,
            "coord": args.coord,
            "cameras": sorted(cameras) if cameras else "all",
            "t_min_label": t0,
            "max_t": max_t,
            "bev_hota": bev,
            "bias": [float(bias[0]), float(bias[1])],
            "n_test_keys": len(test_keys),
        },
    )
    write_run_metadata(
        out_root,
        "eval_fab2_bev_hota",
        args,
        {"bev_hota": bev, "n_rows": len(rows)},
    )
    print(f"identity metrics -> {csv_path}")


if __name__ == "__main__":
    main()
