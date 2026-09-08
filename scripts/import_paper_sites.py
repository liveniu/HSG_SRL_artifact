#!/usr/bin/env python3
"""Import the four paper sites after the licensed datasets are downloaded.

GT identities and 3D coordinates stay in the evaluation interface. This wrapper
only builds origin databases from the vendor importers.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import ARTIFACT_ROOT  # noqa: E402


def _load_sites() -> dict:
    return json.loads(
        (ARTIFACT_ROOT / "configs" / "paper_sites.json").read_text(encoding="utf-8")
    )


def _data_root(site_cfg: dict) -> Path:
    dataset = site_cfg["dataset"]
    if dataset == "LUMPI":
        raw = os.environ.get("HSGSRL_LUMPI_ROOT", str(ARTIFACT_ROOT / "data" / "external" / "LUMPI"))
    else:
        raw = os.environ.get(
            "HSGSRL_SYNTHEHICLE_ROOT",
            str(ARTIFACT_ROOT / "data" / "external" / "synthehicle_core"),
        )
    return Path(raw)


def main() -> int:
    catalog = _load_sites()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", choices=catalog["principal_sites"], required=True)
    parser.add_argument("--db", default=None, help="Output SQLite path")
    parser.add_argument("--batch", default="paper_gt_001")
    args = parser.parse_args()

    site_cfg = catalog["sites"][args.site]
    data_root = _data_root(site_cfg)
    if not data_root.exists():
        print(
            f"Dataset root not found: {data_root}\n"
            "Download the licensed dataset and set HSGSRL_LUMPI_ROOT or "
            "HSGSRL_SYNTHEHICLE_ROOT. See docs/DATASETS.md.",
            file=sys.stderr,
        )
        return 2

    db = Path(args.db) if args.db else ARTIFACT_ROOT / "data" / "work" / f"origin_{args.site}.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    importer = [sys.executable, str(ARTIFACT_ROOT / site_cfg["importer"][0])]
    importer.extend(site_cfg["importer"][1:])
    importer.extend(["--db", str(db), "--batch", args.batch, "--data-root", str(data_root)])
    print(" ".join(importer), flush=True)
    return subprocess.call(importer, cwd=str(ARTIFACT_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
