#!/usr/bin/env python3
# Vendored HSG-SRL artifact copy. Isolated from the development repo;
# edit this file only. See src/SOURCE_MAP.csv.
"""Coarse-to-fine fixed-translation projection fitting for FAB2 preflight.

Single-shot fitting inside the default pairing window converges in one round
when the camera-level projection candidate is well below the window. When the
candidate is comparable to or larger than the window, surviving pairs form a
biased subset and the fit stalls at a wrong fixed point. This wrapper therefore
runs an outer fit-apply-refit loop around ``pipeline/g2_low_order_correction.py``:

* round 0 optionally uses a widened pairing window (``--wide-window-m``);
* the fitted fixed-translation projection is applied to a throwaway working
  copy of the DB (both ``world_x/world_y`` and the homographies, the latter
  via scale-conjugated composition so metric translations are expressed in BEV
  pixels);
* subsequent rounds use the default window until the per-round update norm
  falls below ``--converge-m``.

The source DB is never modified. The output JSON keeps the schema of a single
``g2 fit`` (v2) so ``--camera-translation-json`` consumers work unchanged:
cumulative translations under ``corrections_m_by_scene``, acceptance decision
taken from round 0 (the fit against the uncorrected data), and a
``fit_procedure`` block recording every round.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.fab2_common import (  # noqa: E402
    ensure_dir,
    utc_now,
)


def _update_homographies(conn: sqlite3.Connection, scene: str, cam: str, dx: float, dy: float) -> None:
    """Apply a metric world-plane translation to the camera's stored H.

    DB homographies output BEV pixels (meters x scale_px_per_meter), so the
    metric translation T must be conjugated: H' = S @ T @ S^-1 @ H with
    S = diag(scale, scale, 1); i.e. the translation column is scaled.
    """
    rows = conn.execute(
        "SELECT rowid, homography, scale_px_per_meter FROM camera_calibrations "
        "WHERE camera_name=? AND scene_id=?",
        (cam, scene),
    ).fetchall()
    for rowid, blob, scale in rows:
        s = float(scale)
        H = np.frombuffer(blob, dtype=np.float64).reshape(3, 3).copy()
        H[0] += dx * s * H[2]
        H[1] += dy * s * H[2]
        conn.execute(
            "UPDATE camera_calibrations SET homography=? WHERE rowid=?",
            (H.astype(np.float64).tobytes(), rowid),
        )


def apply_corrections(db: Path, batch: str, by_scene: dict[str, dict[str, tuple[float, float]]]) -> None:
    conn = sqlite3.connect(str(db))
    try:
        for scene, cams in by_scene.items():
            for cam, (dx, dy) in cams.items():
                conn.execute(
                    """UPDATE trajectory_observations
                       SET world_x = world_x + ?, world_y = world_y + ?
                     WHERE scene_id=? AND batch_id=? AND camera_name=?""",
                    (dx, dy, scene, batch, cam),
                )
                _update_homographies(conn, scene, cam, dx, dy)
        conn.commit()
    finally:
        conn.close()


def parse_corrections(blob: dict[str, Any]) -> dict[str, dict[str, tuple[float, float]]]:
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for scene, cams in (blob.get("corrections_m_by_scene") or {}).items():
        out[scene] = {c: (float(v["dx"]), float(v["dy"])) for c, v in cams.items()}
    return out


def fit_command(args: argparse.Namespace, db: Path, out_json: Path, window_m: float) -> list[str]:
    cmd = [
        sys.executable,
        str(_ROOT / "pipeline" / "g2_low_order_correction.py"),
        "fit",
        "--db", str(db),
        "--batch", args.batch,
        "--scene", args.scene,
        "--out", str(out_json),
        "--min-pairs", str(args.min_pairs),
        "--max-pair-dist-m", str(window_m),
        "--min-teacher-confidence", str(args.min_teacher_confidence),
        "--min-teacher-confidence-gap", str(args.min_teacher_confidence_gap),
        "--max-time-gap", str(args.max_time_gap),
        "--motion-compensation", args.motion_compensation,
        "--risk-scalar", args.risk_scalar,
        "--anchor-mode", args.anchor_mode,
        "--s0-m", str(args.s0_m),
        "--feature-l0-m", str(args.feature_l0_m),
        "--anchor-shape-eta", str(args.anchor_shape_eta),
    ]
    for flag, value in (
        ("--cameras", args.cameras),
        ("--overlap-pairs", args.overlap_pairs),
        ("--time-offsets-json", args.time_offsets_json),
        ("--split-manifest", args.split_manifest),
        ("--train-splits", args.train_splits),
        ("--time-start", args.time_start),
        ("--time-end", args.time_end),
        ("--max-pairs", args.max_pairs),
    ):
        if value is not None and value != "":
            cmd.extend([flag, str(value)])
    if args.allow_gt_identity_mining:
        cmd.append("--allow-gt-identity-mining")
    if args.allow_unsplit_fitting:
        cmd.append("--allow-unsplit-fitting")
    return cmd


def run(args: argparse.Namespace) -> None:
    src_db = Path(args.db)
    out_json = Path(args.out)
    ensure_dir(out_json.parent)
    work_dir = ensure_dir(Path(args.work_dir) if args.work_dir else out_json.parent / (out_json.stem + "_work"))
    work_db = work_dir / (src_db.stem + "_tcwork.db")
    shutil.copy2(src_db, work_db)

    cumulative: dict[str, dict[str, np.ndarray]] = {}
    round_log: list[dict[str, Any]] = []
    template: dict[str, Any] | None = None
    try:
        for it in range(args.iters):
            window = args.wide_window_m if (it == 0 and args.wide_window_m) else args.max_pair_dist_m
            round_json = work_dir / f"round{it}_Tc.json"
            cmd = fit_command(args, work_db, round_json, window)
            proc = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True)
            if proc.returncode != 0:
                sys.stderr.write(proc.stdout + proc.stderr)
                raise SystemExit(f"g2 fit failed at round {it} (exit {proc.returncode})")
            blob = json.loads(round_json.read_text(encoding="utf-8"))
            if template is None:
                template = blob
            corr = parse_corrections(blob)
            steps = [float(np.hypot(dx, dy)) for cams in corr.values() for dx, dy in cams.values()]
            max_step = max(steps, default=0.0)
            statuses = {
                scene: (blob.get("diagnostics_by_scene", {}).get(scene, {}).get("decision", {}) or {}).get("status", "")
                for scene in corr
            }
            round_log.append({
                "round": it,
                "window_m": window,
                "max_step_m": round(max_step, 4),
                "status_by_scene": statuses,
            })
            print(f"[tc round {it}] window={window}m max_step={max_step:.3f}m status={statuses}")
            apply_corrections(work_db, args.batch, corr)
            for scene, cams in corr.items():
                slot = cumulative.setdefault(scene, {})
                for cam, (dx, dy) in cams.items():
                    slot[cam] = slot.get(cam, np.zeros(2)) + np.array([dx, dy])
            if max_step < args.converge_m:
                break
    finally:
        if not args.keep_work_db:
            work_db.unlink(missing_ok=True)

    assert template is not None
    payload = dict(template)
    payload["db"] = str(src_db)
    payload["created_at"] = utc_now()
    payload["correction_model"] = "per_camera_2d_translation_cumulative"
    template_corr = template.get("corrections_m_by_scene") or {}
    payload["corrections_m_by_scene"] = {
        scene: {
            cam: {
                "dx": float(vec[0]),
                "dy": float(vec[1]),
                "gauge_weight": (template_corr.get(scene, {}).get(cam, {}) or {}).get("gauge_weight"),
            }
            for cam, vec in cams.items()
        }
        for scene, cams in cumulative.items()
    }
    payload["fit_procedure"] = {
        "mode": "coarse_to_fine_outer_loop",
        "wide_window_m": args.wide_window_m,
        "default_window_m": args.max_pair_dist_m,
        "converge_m": args.converge_m,
        "max_iters": args.iters,
        "rounds": round_log,
        "note": "decision/diagnostics taken from round 0 (fit against uncorrected data)",
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    totals = {
        scene: {cam: round(float(np.hypot(*vec)), 3) for cam, vec in cams.items()}
        for scene, cams in cumulative.items()
    }
    print(f"cumulative |T_c| per camera: {totals}")
    print(f"correction JSON written -> {out_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--scene", required=True, help="Comma-separated scene ids")
    parser.add_argument("--out", required=True, help="Output cumulative correction JSON")
    parser.add_argument("--work-dir", default=None, help="Directory for working DB and per-round JSONs")
    parser.add_argument("--keep-work-db", action="store_true")
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--converge-m", type=float, default=0.02)
    parser.add_argument("--wide-window-m", type=float, default=None, help="Round-0 pairing window (coarse stage)")
    parser.add_argument("--max-pair-dist-m", type=float, default=3.0)
    parser.add_argument("--min-pairs", type=int, default=32)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional diagnostic cap on retained MVC pairs after teacher filters.",
    )
    parser.add_argument("--min-teacher-confidence", type=float, default=0.20)
    parser.add_argument("--min-teacher-confidence-gap", type=float, default=0.05)
    parser.add_argument("--max-time-gap", type=float, default=0.35)
    parser.add_argument("--motion-compensation", default="linear_interpolate")
    parser.add_argument("--risk-scalar", default="metric_jacobian")
    parser.add_argument("--anchor-mode", default="multiplicative_gating_offsetted")
    parser.add_argument("--s0-m", type=float, default=60.0)
    parser.add_argument("--feature-l0-m", type=float, default=10.0)
    parser.add_argument("--anchor-shape-eta", type=float, default=0.75)
    parser.add_argument("--cameras", default="")
    parser.add_argument("--overlap-pairs", default="")
    parser.add_argument("--time-offsets-json", default=None)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--train-splits", default="train")
    parser.add_argument("--time-start", type=float, default=None)
    parser.add_argument("--time-end", type=float, default=None)
    parser.add_argument("--allow-gt-identity-mining", action="store_true")
    parser.add_argument("--allow-unsplit-fitting", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
