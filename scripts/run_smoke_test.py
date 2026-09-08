#!/usr/bin/env python3
"""Minimal smoke test aligned with Algorithm 1 and Sections III-D--III-F.

Checks HSG feature construction, MVC pair mining, a short residual train
(paper-explicit ``--lambda-mean-delta 1.0 --lambda-pair-center 0.5``),
and the label-free residual acceptor (Appendix C selector), writing a short report
under results/smoke/. The acceptor's status (recommended / thin_val /
not_recommended) depends on the tiny fixture and the 3-epoch seed-0
checkpoint; any of the three is a valid outcome of the selector rule --
only a script crash counts as a smoke-test failure.

Does not score Table IV (needs origin GT + time maps). Run
``python -m evaluation.bev_hota`` from ``src/`` for the HOTA/LocA self-check.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import ARTIFACT_ROOT, SRC_ROOT, ensure_import_path  # noqa: E402

ensure_import_path()

from pipeline.geometry_guided_residual_mlp import (  # noqa: E402
    GeometryDataView,
    compute_geometry_base,
)
from pipeline.multi_camera_bev_stitch import load_calibration  # noqa: E402
from pipeline.multi_camera_trajectory_fusion import init_trajectory_db  # noqa: E402

SCENE = "smoke_pair"
BATCH = "smoke_001"


def _remove_sqlite_file(path: Path) -> None:
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        if candidate.exists():
            candidate.unlink()


def _fail(msg: str) -> int:
    print(f"SMOKE FAIL: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    init_db = ARTIFACT_ROOT / "data" / "init" / "smoke_pair.db"
    split = ARTIFACT_ROOT / "splits" / "smoke_split_manifest.csv"
    if not init_db.is_file() or not split.is_file():
        rc = subprocess.call(
            [sys.executable, str(ARTIFACT_ROOT / "scripts" / "init_smoke_data.py")],
            cwd=str(ARTIFACT_ROOT),
        )
        if rc != 0:
            return _fail("could not build the synthetic init package")

    out_dir = ARTIFACT_ROOT / "results" / "smoke"
    work_dir = out_dir / "work"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    db = work_dir / "smoke_pair_runtime.db"
    _remove_sqlite_file(db)
    shutil.copy2(init_db, db)

    conn = init_trajectory_db(str(db), BATCH)
    cal_a = load_calibration(conn, "CA", SCENE)
    cal_b = load_calibration(conn, "CB", SCENE)
    if cal_a is None or cal_b is None:
        return _fail("missing smoke homographies")
    H_a, scale_a = cal_a
    H_b, scale_b = cal_b

    box_a = np.array([860.0, 620.0, 960.0, 720.0], dtype=np.float64)
    box_b = np.array([900.0, 280.0, 970.0, 430.0], dtype=np.float64)
    _, _, feat_a, risk_a = compute_geometry_base(box_a, H_a, scale_a, (1920, 1080))
    _, _, feat_b, risk_b = compute_geometry_base(box_b, H_b, scale_b, (1920, 1080))
    if feat_a.shape != (7,) or feat_b.shape != (7,):
        return _fail("feature F must be length 7")
    if abs((1.0 - risk_a.alpha) - risk_a.kappa) > 1e-6:
        return _fail("kappa must equal 1-alpha")
    if not (risk_a.valid_ground_side and risk_b.valid_ground_side):
        return _fail("smoke boxes must lie on the valid ground side")
    if risk_a.kappa <= risk_b.kappa:
        return _fail("camera CA should be the higher-kappa teacher in this fixture")

    cals = GeometryDataView.load_camera_calibrations(conn, SCENE, ["CA", "CB"])
    obs = list(
        GeometryDataView.iter_observations_from_db(
            conn, SCENE, BATCH, ["CA", "CB"], cals
        )
    )
    pairs = GeometryDataView.mine_mvc_pairs(
        obs, {("CA", "CB")}, max_time_gap_sec=0.35
    )
    conn.close()
    if len(pairs) < 32:
        return _fail(f"expected at least 32 MVC pairs, got {len(pairs)}")

    report = {
        "status": "hsg_ok",
        "n_observations": len(obs),
        "n_mvc_pairs": len(pairs),
        "feature_dim": 7,
        "kappa_ca": float(risk_a.kappa),
        "kappa_cb": float(risk_b.kappa),
        "alpha_kappa_coupled": True,
        "smoke_db_source": str(init_db.relative_to(ARTIFACT_ROOT)).replace("\\", "/"),
        "runtime_db": str(db.relative_to(ARTIFACT_ROOT)).replace("\\", "/"),
    }

    ckpt = out_dir / "smoke_seed0.pt"
    train_py = SRC_ROOT / "evaluation" / "train_fab2_site_frozen.py"
    cmd = [
        sys.executable,
        str(train_py),
        "--db",
        str(db),
        "--batch",
        BATCH,
        "--scene",
        SCENE,
        "--split-manifest",
        str(split),
        "--train-splits",
        "train",
        "--seeds",
        "0",
        "--epochs",
        "3",
        "--max-residual-m",
        "1.0",
        "--max-pair-dist-m",
        "2",
        "--lambda-reg",
        "0.10",
        "--lambda-smooth",
        "0.02",
        "--lambda-mean-delta",
        "1.0",
        "--lambda-pair-center",
        "0.5",
        "--min-teacher-confidence",
        "0.20",
        "--min-teacher-confidence-gap",
        "0.05",
        "--min-mvc-pairs",
        "16",
        "--out-dir",
        str(out_dir),
    ]
    train_rc = subprocess.call(cmd, cwd=str(SRC_ROOT))
    generated_metadata = out_dir / "train_fab2_site_frozen_run_metadata.json"
    if generated_metadata.exists():
        generated_metadata.unlink()
    report["train_status"] = "ok" if train_rc == 0 else f"failed_rc_{train_rc}"
    seed0_ckpt = out_dir / "training" / "frozen_checkpoints" / "smoke_pair_seed0.pt"
    report["checkpoint_exists"] = ckpt.exists() or seed0_ckpt.exists()
    if train_rc != 0:
        (out_dir / "smoke_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return _fail("frozen training wrapper failed; HSG geometry still passed")

    # Appendix C selector: label-free residual acceptor, val split only (no test keys,
    # no GT identities). Exercises pipeline/fit_residual_acceptor.py end to
    # end on the seed-0 checkpoint that was just trained.
    tc_json = out_dir / "training" / "smoke_pair_preflight_Tc.json"
    acceptor_json = out_dir / "smoke_acceptor_seed0.json"
    acceptor_cmd = [
        sys.executable,
        str(SRC_ROOT / "pipeline" / "fit_residual_acceptor.py"),
        "--db", str(db),
        "--scene", SCENE,
        "--batch", BATCH,
        "--checkpoint", str(seed0_ckpt),
        "--split-manifest", str(split),
        "--tc-json", str(tc_json),
        "--out", str(acceptor_json),
        "--min-pairs", "32",
        "--seed", "0",
    ]
    acceptor_rc = subprocess.call(acceptor_cmd, cwd=str(SRC_ROOT))
    report["acceptor_status"] = "ok" if acceptor_rc == 0 else f"failed_rc_{acceptor_rc}"
    if acceptor_rc == 0 and acceptor_json.is_file():
        acceptor_blob = json.loads(acceptor_json.read_text(encoding="utf-8"))
        report["acceptor_residual_cameras"] = acceptor_blob.get("residual_cameras", [])
        report["acceptor_camera_status"] = {
            cam: rec.get("status") for cam, rec in (acceptor_blob.get("cameras") or {}).items()
        }
    if acceptor_rc != 0:
        (out_dir / "smoke_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return _fail("residual acceptor wrapper failed; HSG geometry + training still passed")

    (out_dir / "smoke_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("SMOKE PASS")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
