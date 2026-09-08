#!/usr/bin/env python3
# Vendored HSG-SRL artifact copy. Isolated from the development repo;
# edit this file only. See src/SOURCE_MAP.csv.
"""FAB2 controlled perturbation-recovery protocol (plan section 9.4).

For sites that fail the kappa premise audit (typically model-H synthetic data
whose homographies are exact, so the residual error is anchor bias rather
than calibration error), this tool injects a *known* per-camera world-plane
error before any label-free correction runs, then measures how much of it the
correction chain (coarse-to-fine T_c + frozen HSG-SRL) recovers.

Error classes (``--dose-type``):

* ``affine``  - per-camera translation (zero mean across cameras, mean norm =
  dose) plus alternating +/-0.4 deg/m rotation about the scene centroid.
  ``--rot-scale 0`` gives the pure-translation class.
* ``scale``   - camera-height-like radial scaling about each camera's ground
  point (gamma = dose/50, alternating sign). Error grows with distance and is
  kappa-correlated, i.e. inside the MLP's feature class (synthehicle only:
  ground points come from the CARLA camera_info extrinsics).
* ``rotation`` - out-of-class control: world-plane rotation about each
  camera's ground point. Angle is dose/50 rad so the equivalent
  displacement at 50 m equals the dose (synthehicle only).

The perturbation is written consistently to ``world_x/world_y`` *and* to the
homographies (scale-conjugated composition), because mining recomputes
anchors from H. GT is never touched. ``--clean-anchor`` replaces anchors with
GT bottom-center projections to isolate injected error from anchor bias.

Variants reported per dose x seed: ``unperturbed_baseline`` (reference),
``perturbed_baseline``, ``plus_tc``, ``plus_tc_mlp``; recovery fractions are
relative to the unperturbed baseline. Output:
``<out-dir>/iv_e/perturbation_recovery.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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
    FAB2_DEFAULT_RESULTS,
    ensure_dir,
    upsert_csv_rows,
    write_run_metadata,
)
from evaluation.fit_fab2_low_order import parse_corrections  # noqa: E402
from pipeline.geometry_guided_residual_mlp import (  # noqa: E402
    GeometryDataView,
    _infer_and_calib_wh_from_db,
    load_checkpoint,
)
from pipeline.multi_camera_trajectory_fusion import adapt_homography_to_resolution  # noqa: E402

ROT_DEG_PER_METER_DOSE = 0.4

RESULT_FIELDS = [
    "scene",
    "tag",
    "dose_type",
    "dose_m",
    "rot_scale",
    "clean_anchor",
    "seed",
    "variant",
    "n",
    "rmse_m",
    "median_m",
    "p95_m",
    "rmse_recovery_frac",
    "median_recovery_frac",
    "tc_rounds",
    "tc_round0_status",
    "mean_tc_undo_err_m",
]


# ---------------------------------------------------------------------------
# perturbation construction
# ---------------------------------------------------------------------------

def _synthehicle_dataset_root(explicit: str | Path | None = None) -> Path:
    """Locate Synthehicle core (camera_info). Artifact and development layouts differ."""
    if explicit and str(explicit).strip():
        return Path(explicit)
    env = os.environ.get("HSGSRL_SYNTHEHICLE_ROOT", "").strip()
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    roots = [here.parents[2]] if here.parents[1].name == "src" else [here.parents[1]]
    for root in roots:
        for rel in (
            Path("data") / "external" / "synthehicle_core",
            Path("train-test_data") / "synthehicle_core",
        ):
            cand = root / rel
            if (cand / "calibration").is_dir():
                return cand
    return roots[0] / "data" / "external" / "synthehicle_core"

def _update_homographies(conn: sqlite3.Connection, scene: str, cam: str, A: np.ndarray) -> None:
    """Apply a metric world-plane affine A to the camera's stored H.

    DB homographies output BEV pixels (meters x scale_px_per_meter):
    H' = S @ A @ S^-1 @ H with S = diag(scale, scale, 1).
    """
    rows = conn.execute(
        "SELECT rowid, homography, scale_px_per_meter FROM camera_calibrations "
        "WHERE camera_name=? AND scene_id=?",
        (cam, scene),
    ).fetchall()
    for rowid, blob, scale in rows:
        s = float(scale)
        S = np.diag([s, s, 1.0])
        S_inv = np.diag([1.0 / s, 1.0 / s, 1.0])
        H = np.frombuffer(blob, dtype=np.float64).reshape(3, 3).copy()
        conn.execute(
            "UPDATE camera_calibrations SET homography=? WHERE rowid=?",
            ((S @ A @ S_inv @ H).astype(np.float64).tobytes(), rowid),
        )


def make_affines(
    cameras: list[str], dose_m: float, rng: np.random.Generator, center: np.ndarray,
    *, rot_scale: float,
) -> dict[str, np.ndarray]:
    """Per-camera 3x3 world-plane affine: rotation about *center* (+/- theta *
    rot_scale) plus zero-mean translations with mean norm = dose_m."""
    n = len(cameras)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n)
    b = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    b = b - b.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(b, axis=1)
    b = b * (dose_m / max(norms.mean(), 1e-9))

    theta_deg = ROT_DEG_PER_METER_DOSE * dose_m * rot_scale
    out: dict[str, np.ndarray] = {}
    for i, cam in enumerate(cameras):
        th = np.deg2rad(theta_deg) * (1 if i % 2 == 0 else -1)
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        t = center - R @ center + b[i]
        W = np.eye(3)
        W[:2, :2] = R
        W[:2, 2] = t
        out[cam] = W
    return out


def camera_ground_points_from_calib(
    conn: sqlite3.Connection, scene: str, cameras: list[str],
    *, synthehicle_root: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Per-camera ground point in the DB world frame (synthehicle only).

    The image vanishing point of the world down direction (0,0,-1), using the
    CARLA camera convention (X forward, Y right, Z up), maps through the DB
    homography to the camera's ground projection.
    """
    town = scene.split("_")[1]
    info_dir = (
        _synthehicle_dataset_root(synthehicle_root) / "calibration"
        / "overlapping" / town / "camera_info"
    )
    if not info_dir.is_dir():
        raise FileNotFoundError(
            "Synthehicle camera_info is required for --dose-type scale|rotation. "
            f"Missing: {info_dir}. Set --synthehicle-root or HSGSRL_SYNTHEHICLE_ROOT "
            "to the dataset root (expected calibration/overlapping/<Town>/camera_info). "
            "Table IX affine doses do not need this directory."
        )
    out: dict[str, np.ndarray] = {}
    for cam in cameras:
        n = int(cam.lstrip("C"))
        blob = json.loads((info_dir / f"camera_{n}.txt").read_text(encoding="utf-8"))
        K = np.array(blob["intrinsic_matrix"], dtype=np.float64)
        E = np.array(blob["extrinsic_matrix"], dtype=np.float64)
        d_cam = E[:3, :3] @ np.array([0.0, 0.0, -1.0])
        if d_cam[0] < 0:
            d_cam = -d_cam
        fx, fy, cx_, cy_ = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        v_px = np.array([
            fx * d_cam[1] / d_cam[0] + cx_,
            fy * (-d_cam[2]) / d_cam[0] + cy_,
        ])
        row = conn.execute(
            """SELECT homography, scale_px_per_meter FROM camera_calibrations
               WHERE camera_name=? AND scene_id=? ORDER BY calibrated_at DESC LIMIT 1""",
            (cam, scene),
        ).fetchone()
        H = np.frombuffer(row[0], dtype=np.float64).reshape(3, 3)
        w = H @ np.array([v_px[0], v_px[1], 1.0])
        out[cam] = (w[:2] / w[2]) / float(row[1])
    return out


def make_perturbed_db(
    args: argparse.Namespace, dose_m: float, tag: str, work_dir: Path
) -> tuple[Path, dict[str, Any]]:
    scene, batch = args.scene, args.batch
    dst = work_dir / f"{tag}.db"
    shutil.copy2(args.db, dst)

    conn = sqlite3.connect(str(dst))
    cameras = sorted(
        r[0] for r in conn.execute(
            "SELECT DISTINCT camera_name FROM trajectory_observations WHERE scene_id=? AND batch_id=?",
            (scene, batch),
        )
    )
    cx, cy = conn.execute(
        "SELECT AVG(world_x), AVG(world_y) FROM trajectory_observations WHERE scene_id=? AND batch_id=?",
        (scene, batch),
    ).fetchone()

    if args.clean_anchor:
        conn.execute(
            """UPDATE trajectory_observations
               SET bbox_x = outside_reference_image_x - bbox_w / 2.0,
                   bbox_y = outside_reference_image_y - bbox_h,
                   image_x = outside_reference_image_x,
                   image_y = outside_reference_image_y,
                   world_x = outside_reference_x,
                   world_y = outside_reference_y
             WHERE scene_id=? AND batch_id=?
               AND outside_reference_image_x IS NOT NULL
               AND outside_reference_x IS NOT NULL""",
            (scene, batch),
        )

    rng = np.random.default_rng(args.pert_seed)
    cam_grounds: dict[str, np.ndarray] = {}
    if args.dose_type == "scale":
        cam_grounds = camera_ground_points_from_calib(
            conn, scene, cameras, synthehicle_root=args.synthehicle_root or None,
        )
        gamma = dose_m / 50.0
        affines = {}
        for i, cam in enumerate(cameras):
            g = gamma * (1 if i % 2 == 0 else -1)
            cgx, cgy = cam_grounds[cam]
            W = np.eye(3)
            W[0, 0] = W[1, 1] = 1.0 + g
            W[0, 2] = -g * cgx
            W[1, 2] = -g * cgy
            affines[cam] = W
    elif args.dose_type == "rotation":
        cam_grounds = camera_ground_points_from_calib(
            conn, scene, cameras, synthehicle_root=args.synthehicle_root or None,
        )
        theta = dose_m / 50.0
        affines = {}
        for i, cam in enumerate(cameras):
            th = theta * (1 if i % 2 == 0 else -1)
            c, s = np.cos(th), np.sin(th)
            cgx, cgy = cam_grounds[cam]
            W = np.eye(3)
            W[0, 0], W[0, 1] = c, -s
            W[1, 0], W[1, 1] = s, c
            W[0, 2] = cgx - c * cgx + s * cgy
            W[1, 2] = cgy - s * cgx - c * cgy
            affines[cam] = W
    else:
        affines = make_affines(cameras, dose_m, rng, np.array([cx, cy]), rot_scale=args.rot_scale)

    for cam, W in affines.items():
        a, b_, tx = W[0]
        c, d, ty = W[1]
        conn.execute(
            """UPDATE trajectory_observations
               SET world_x = ? * world_x + ? * world_y + ?,
                   world_y = ? * world_x + ? * world_y + ?
             WHERE scene_id=? AND batch_id=? AND camera_name=?""",
            (a, b_, tx, c, d, ty, scene, batch, cam),
        )
        _update_homographies(conn, scene, cam, W)
    conn.commit()
    conn.close()

    spec = {
        "scene": scene,
        "dose_m": dose_m,
        "dose_type": args.dose_type,
        "rot_deg": (
            ROT_DEG_PER_METER_DOSE * dose_m * args.rot_scale
            if args.dose_type == "affine"
            else (float(np.degrees(dose_m / 50.0)) if args.dose_type == "rotation" else 0.0)
        ),
        "gamma": dose_m / 50.0 if args.dose_type == "scale" else None,
        "pert_seed": args.pert_seed,
        "clean_anchor": bool(args.clean_anchor),
        "center": [float(cx), float(cy)],
        "camera_ground_points": {c_: g.tolist() for c_, g in cam_grounds.items()} or None,
        "affines": {cam: W.tolist() for cam, W in affines.items()},
        "injected_shift_per_camera_m": {
            cam: [float(W[0, 2] + (W[0, 0] - 1) * cx + W[0, 1] * cy),
                  float(W[1, 2] + W[1, 0] * cx + (W[1, 1] - 1) * cy)]
            for cam, W in affines.items()
        },
    }
    (work_dir / f"{tag}_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return dst, spec


# ---------------------------------------------------------------------------
# pipeline steps
# ---------------------------------------------------------------------------

def run_cmd(cmd: list[str], log: Path) -> None:
    ensure_dir(log.parent)
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(_ROOT))
    if r.returncode != 0:
        raise SystemExit(f"command failed (exit {r.returncode}), see {log}")


def fit_tc_coarse_to_fine(
    args: argparse.Namespace, pert_db: Path, tag: str, work_dir: Path, log_dir: Path
) -> tuple[Path, dict[str, tuple[float, float]], dict[str, Any]]:
    """Run the formal coarse-to-fine fitter, apply cumulative T_c to a fresh
    working DB copy, and return (corrected_db, cumulative corrections, log)."""
    out_json = work_dir / f"{tag}_Tc.json"
    cmd = [
        sys.executable, str(_ROOT / "evaluation" / "fit_fab2_low_order.py"),
        "--db", str(pert_db), "--batch", args.batch, "--scene", args.scene,
        "--out", str(out_json),
        "--work-dir", str(work_dir / f"{tag}_tcfit"),
        "--iters", str(args.tc_iters),
        "--split-manifest", args.gt_index, "--train-splits", "train",
        "--anchor-mode", args.anchor_mode,
        "--min-pairs", str(args.tc_min_pairs),
        "--min-teacher-confidence", str(args.min_teacher_confidence),
        "--min-teacher-confidence-gap", str(args.min_teacher_confidence_gap),
    ]
    if args.max_pairs is not None:
        cmd += ["--max-pairs", str(args.max_pairs)]
    if args.tc_wide_window_m:
        cmd += ["--wide-window-m", str(args.tc_wide_window_m)]
    for flag, value in (
        ("--time-offsets-json", args.time_offsets_json),
        ("--cameras", args.cameras),
        ("--overlap-pairs", args.overlap_pairs),
    ):
        if value:
            cmd += [flag, str(value)]
    run_cmd(cmd, log_dir / f"tcfit_{tag}.log")

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    corr = parse_corrections(payload).get(args.scene, {})
    corrected_db = work_dir / f"{tag}_corrected.db"
    shutil.copy2(pert_db, corrected_db)
    conn = sqlite3.connect(str(corrected_db))
    for cam, (dx, dy) in corr.items():
        conn.execute(
            """UPDATE trajectory_observations
               SET world_x = world_x + ?, world_y = world_y + ?
             WHERE scene_id=? AND batch_id=? AND camera_name=?""",
            (dx, dy, args.scene, args.batch, cam),
        )
        T = np.eye(3)
        T[0, 2], T[1, 2] = dx, dy
        _update_homographies(conn, args.scene, cam, T)
    conn.commit()
    conn.close()
    return corrected_db, corr, payload.get("fit_procedure", {})


def train_mlp(args: argparse.Namespace, db: Path, tag: str, seed: int, ckpt_dir: Path, log_dir: Path) -> Path:
    ckpt = ckpt_dir / f"{tag}_seed{seed}.pt"
    if args.reuse_checkpoints and ckpt.exists():
        print(f"[{tag} seed{seed}] reuse checkpoint -> {ckpt}")
        return ckpt
    cmd = [
        sys.executable, str(_ROOT / "pipeline" / "geometry_guided_residual_mlp.py"), "train",
        "--db", str(db), "--batch", args.batch, "--scene", args.scene,
        "--split-manifest", args.gt_index, "--train-splits", "train",
        "--seed", str(seed), "--epochs", str(args.epochs),
        "--min-mvc-pairs", str(args.min_mvc_pairs),
        "--anchor-mode", args.anchor_mode,
        "--max-residual-m", str(args.max_residual_m),
        "--max-pair-dist-m", str(args.max_pair_dist_m),
        "--lambda-reg", str(args.lambda_reg),
        "--lambda-smooth", str(args.lambda_smooth),
        "--lambda-mean-delta", str(args.lambda_mean_delta),
        "--lambda-pair-center", str(args.lambda_pair_center),
        "--min-teacher-confidence", str(args.min_teacher_confidence),
        "--min-teacher-confidence-gap", str(args.min_teacher_confidence_gap),
        "--teacher-confidence-power", str(args.teacher_confidence_power),
        "--student-weight-floor", str(args.student_weight_floor),
        "--reg-confidence-floor", str(args.reg_confidence_floor),
        "--out", str(ckpt),
    ]
    if args.max_pairs is not None:
        cmd += ["--max-pairs", str(args.max_pairs)]
    for flag, value in (
        ("--time-offsets-json", args.time_offsets_json),
        ("--cameras", args.cameras),
        ("--overlap-pairs", args.overlap_pairs),
    ):
        if value:
            cmd += [flag, str(value)]
    run_cmd(cmd, log_dir / f"train_{tag}_seed{seed}.log")
    return ckpt


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def load_split_gt(path: Path) -> dict[tuple[str, int, int], tuple[str, float, float]]:
    out: dict[tuple[str, int, int], tuple[str, float, float]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            try:
                key = (r["camera_id"], int(r["local_track_id"]), int(r["frame_index"]))
            except (KeyError, ValueError):
                continue
            gx = float(r["gt_world_x"]) if r.get("gt_world_x") else float("nan")
            gy = float(r["gt_world_y"]) if r.get("gt_world_y") else float("nan")
            out[key] = (r.get("split", ""), gx, gy)
    return out


def load_observations(
    db: Path, batch: str, scene: str, split_gt: dict, meta: dict
) -> dict[str, np.ndarray]:
    conn = sqlite3.connect(str(db))
    view = GeometryDataView()
    cameras = sorted(
        r[0] for r in conn.execute(
            "SELECT DISTINCT camera_name FROM trajectory_observations WHERE scene_id=? AND batch_id=?",
            (scene, batch),
        )
    )
    cal = view.load_camera_calibrations(conn, scene, cameras)
    infer_wh, calib_wh = _infer_and_calib_wh_from_db(conn, scene, batch, cameras)
    adapted = {}
    for cam, (h, s) in cal.items():
        iw, cw = infer_wh.get(cam), calib_wh.get(cam)
        if iw and cw and iw != cw:
            h = adapt_homography_to_resolution(h, cw, iw)
        adapted[cam] = (h, s)
    obs = view.iter_observations_from_db(
        conn, scene, batch, cameras, adapted,
        frame_wh_by_camera={c: infer_wh[c] for c in cameras},
        risk_scalar_mode=str(meta.get("risk_scalar_mode", "metric_jacobian")),
        s0_m=float(meta.get("s0_m", 60.0)),
        feature_l0_m=float(meta.get("feature_l0_m", 10.0)),
        anchor_mode=str(meta.get("anchor_mode", "multiplicative_gating_offsetted")),
        anchor_shape_eta=float(meta.get("anchor_shape_eta", 0.75)),
        sensitivity_clip_m=meta.get("sensitivity_clip_m"),
        extent_clip_m=meta.get("extent_clip_m"),
    )
    split, feats, p_geo, gt, kappa = [], [], [], [], []
    for o in obs:
        key = (o.camera_name, int(o.local_track_id), int(o.frame_index or -1))
        sp, gx, gy = split_gt.get(key, ("", float("nan"), float("nan")))
        split.append(sp)
        feats.append(np.asarray(o.feature, dtype=np.float32))
        p_geo.append(np.asarray(o.world_geo, dtype=np.float64))
        gt.append((gx, gy))
        kappa.append(float(getattr(o, "horizon_confidence", np.nan)))
    conn.close()
    return {
        "split": np.asarray(split),
        "features": np.stack(feats),
        "p_geo": np.stack(p_geo),
        "gt": np.asarray(gt, dtype=np.float64),
        "kappa": np.asarray(kappa, dtype=np.float64),
    }


def predict_residuals(ckpt: Path, features: np.ndarray) -> tuple[np.ndarray, dict]:
    import torch

    model, meta = load_checkpoint(Path(ckpt))
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(features), 65536):
            t = torch.from_numpy(features[start:start + 65536].astype(np.float32))
            out.append(model(t).cpu().numpy().astype(np.float64))
    return np.concatenate(out, axis=0), meta


def loc_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(gt).all(axis=1)
    err = np.hypot(pred[ok, 0] - gt[ok, 0], pred[ok, 1] - gt[ok, 1])
    if err.size == 0:
        return {"n": 0, "rmse_m": float("nan"), "median_m": float("nan"), "p95_m": float("nan")}
    return {
        "n": int(err.size),
        "rmse_m": float(np.sqrt(np.mean(err ** 2))),
        "median_m": float(np.median(err)),
        "p95_m": float(np.percentile(err, 95)),
    }


def _recovery_ref(name: str, prefix: str) -> str:
    for band in ("kappa_low", "kappa_mid", "kappa_high"):
        if name.endswith(f"_{band}"):
            return f"{prefix}_{band}"
    return prefix


def recovery(pert: float, var: float, clean: float) -> float | str:
    denom = pert - clean
    if not np.isfinite(denom) or abs(denom) < 1e-9:
        return ""
    return round(float((pert - var) / denom), 4)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    out_root = ensure_dir(Path(args.out_dir))
    out_dir = ensure_dir(out_root / "iv_e")
    work_dir = ensure_dir(Path(args.work_dir) if args.work_dir else out_dir / "work")
    ckpt_dir = ensure_dir(work_dir / "checkpoints")
    log_dir = ensure_dir(work_dir / "logs")

    split_gt = load_split_gt(Path(args.split_manifest))
    doses = [float(d) for d in args.doses.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows: list[dict[str, Any]] = []

    # Reference DB: dose-0 construction (identity affines, same --clean-anchor
    # handling), so the unperturbed baseline shares the error composition of
    # the perturbed chain instead of including anchor bias the chain never saw.
    ref_tag = f"{args.scene}_{args.dose_type}_ref" + ("_clean" if args.clean_anchor else "")
    ref_db, _ = make_perturbed_db(args, 0.0, ref_tag, work_dir)
    data_clean: dict[str, np.ndarray] | None = None

    for dose in doses:
        tag = (
            f"{args.scene}_{args.dose_type}{dose:g}m"
            + (f"_rot{args.rot_scale:g}" if args.dose_type == "affine" and args.rot_scale != 1.0 else "")
            + ("_clean" if args.clean_anchor else "")
            + f"_pseed{args.pert_seed}"
        )
        pert_db, spec = make_perturbed_db(args, dose, tag, work_dir)
        print(f"[{tag}] perturbed DB made")
        corrected_db, corr, fit_log = fit_tc_coarse_to_fine(args, pert_db, tag, work_dir, log_dir)
        tc_rounds = len(fit_log.get("rounds", []))
        tc_round0_status = ""
        if fit_log.get("rounds"):
            statuses = fit_log["rounds"][0].get("status_by_scene", {})
            tc_round0_status = statuses.get(args.scene, "")

        inj = spec["injected_shift_per_camera_m"]
        undo = [
            float(np.hypot(bx + corr.get(cam, (0.0, 0.0))[0], by + corr.get(cam, (0.0, 0.0))[1]))
            for cam, (bx, by) in sorted(inj.items())
        ]
        mean_undo = float(np.mean(undo)) if undo else float("nan")
        print(f"[{tag}] Tc rounds={tc_rounds} mean_undo_err={mean_undo:.3f}m")

        for seed in seeds:
            ckpt = train_mlp(args, corrected_db, tag, seed, ckpt_dir, log_dir)
            _, meta = predict_residuals(ckpt, np.zeros((1, 7), dtype=np.float32))
            if data_clean is None:
                data_clean = load_observations(ref_db, args.batch, args.scene, split_gt, meta)
            data_pert = load_observations(pert_db, args.batch, args.scene, split_gt, meta)
            data_corr = load_observations(corrected_db, args.batch, args.scene, split_gt, meta)
            delta, _ = predict_residuals(ckpt, data_corr["features"])

            test = data_corr["split"] == "test"
            variants = {
                "unperturbed_baseline": (data_clean["p_geo"], data_clean["split"] == "test", data_clean["gt"]),
                "perturbed_baseline": (data_pert["p_geo"], data_pert["split"] == "test", data_pert["gt"]),
                "plus_tc": (data_corr["p_geo"], test, data_corr["gt"]),
                "plus_tc_mlp": (data_corr["p_geo"] + delta, test, data_corr["gt"]),
            }
            if args.dose_type == "scale":
                k = data_corr["kappa"]
                finite = test & np.isfinite(k)
                if int(np.count_nonzero(finite)) >= 30:
                    cuts = np.quantile(k[finite], [1.0 / 3.0, 2.0 / 3.0])
                    bands = {
                        "kappa_low": finite & (k <= cuts[0]),
                        "kappa_mid": finite & (k > cuts[0]) & (k <= cuts[1]),
                        "kappa_high": finite & (k > cuts[1]),
                    }
                    for band, mask in bands.items():
                        variants[f"plus_tc_{band}"] = (data_corr["p_geo"], mask, data_corr["gt"])
                        variants[f"plus_tc_mlp_{band}"] = (
                            data_corr["p_geo"] + delta, mask, data_corr["gt"])
                        variants[f"perturbed_baseline_{band}"] = (
                            data_pert["p_geo"], mask, data_pert["gt"])
                        variants[f"unperturbed_baseline_{band}"] = (
                            data_clean["p_geo"], mask, data_clean["gt"])
            stats = {name: loc_metrics(pred[mask], gt[mask]) for name, (pred, mask, gt) in variants.items()}
            for name, st in stats.items():
                rows.append({
                    "scene": args.scene,
                    "tag": tag,
                    "dose_type": args.dose_type,
                    "dose_m": dose,
                    "rot_scale": args.rot_scale if args.dose_type == "affine" else "",
                    "clean_anchor": int(args.clean_anchor),
                    "seed": seed,
                    "variant": name,
                    "n": st["n"],
                    "rmse_m": round(st["rmse_m"], 4),
                    "median_m": round(st["median_m"], 4),
                    "p95_m": round(st["p95_m"], 4),
                    "rmse_recovery_frac": recovery(
                        stats[_recovery_ref(name, "perturbed_baseline")]["rmse_m"],
                        st["rmse_m"],
                        stats[_recovery_ref(name, "unperturbed_baseline")]["rmse_m"],
                    ) if not name.startswith("unperturbed_baseline")
                    and not name.startswith("perturbed_baseline") else "",
                    "median_recovery_frac": recovery(
                        stats[_recovery_ref(name, "perturbed_baseline")]["median_m"],
                        st["median_m"],
                        stats[_recovery_ref(name, "unperturbed_baseline")]["median_m"],
                    ) if not name.startswith("unperturbed_baseline")
                    and not name.startswith("perturbed_baseline") else "",
                    "tc_rounds": tc_rounds,
                    "tc_round0_status": tc_round0_status,
                    "mean_tc_undo_err_m": round(mean_undo, 4),
                })
                print(
                    f"[{tag} seed{seed}] {name:<22} rmse={st['rmse_m']:.4f} "
                    f"med={st['median_m']:.4f} p95={st['p95_m']:.4f}"
                )

        if not args.keep_work_dbs:
            pert_db.unlink(missing_ok=True)
            corrected_db.unlink(missing_ok=True)

    if not args.keep_work_dbs:
        ref_db.unlink(missing_ok=True)

    out_csv = out_dir / "perturbation_recovery.csv"
    upsert_csv_rows(out_csv, rows, RESULT_FIELDS, key_fields=["scene", "tag", "seed", "variant"])
    write_run_metadata(out_root, "run_fab2_perturbation_recovery", args, {"doses": doses, "seeds": seeds})
    print(f"perturbation recovery results -> {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Unperturbed source DB (never modified)")
    parser.add_argument("--batch", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split-manifest", required=True, help="dataset_split_manifest.csv with GT columns")
    parser.add_argument("--gt-index", required=True, help="gt_index.csv used to restrict fitting/training to train split")
    parser.add_argument("--out-dir", default=str(FAB2_DEFAULT_RESULTS))
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--keep-work-dbs", action="store_true")
    parser.add_argument("--doses", default="0,0.5,1.0,2.0")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--pert-seed", type=int, default=1234)
    parser.add_argument("--dose-type", choices=("affine", "scale", "rotation"), default="affine")
    parser.add_argument(
        "--synthehicle-root",
        default="",
        help=(
            "Synthehicle core root (calibration/overlapping/<Town>/camera_info). "
            "Needed for --dose-type scale|rotation unless HSGSRL_SYNTHEHICLE_ROOT is set. "
            "Table IX affine doses do not read this directory."
        ),
    )
    parser.add_argument("--rot-scale", type=float, default=1.0, help="Rotation component scale; 0 = pure translation")
    parser.add_argument("--clean-anchor", action="store_true", help="Replace anchors with GT bottom-center projections")
    parser.add_argument("--anchor-mode", default=None, help="Defaults to bottom_center with --clean-anchor, else multiplicative_gating_offsetted")
    parser.add_argument("--tc-iters", type=int, default=8)
    parser.add_argument("--tc-wide-window-m", type=float, default=None)
    parser.add_argument("--tc-min-pairs", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--max-residual-m", type=float, default=1.5)
    parser.add_argument("--max-pair-dist-m", type=float, default=3.0)
    parser.add_argument("--lambda-reg", type=float, default=0.05)
    parser.add_argument("--lambda-smooth", type=float, default=0.02)
    parser.add_argument("--lambda-mean-delta", type=float, default=0.0)
    parser.add_argument("--lambda-pair-center", type=float, default=0.0)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional diagnostic cap on retained MVC pairs after teacher filters.",
    )
    parser.add_argument("--min-mvc-pairs", type=int, default=64)
    parser.add_argument("--min-teacher-confidence", type=float, default=0.20)
    parser.add_argument("--min-teacher-confidence-gap", type=float, default=0.05)
    parser.add_argument("--teacher-confidence-power", type=float, default=1.0)
    parser.add_argument("--student-weight-floor", type=float, default=0.05)
    parser.add_argument("--reg-confidence-floor", type=float, default=0.25)
    parser.add_argument("--time-offsets-json", default="")
    parser.add_argument("--cameras", default="")
    parser.add_argument("--overlap-pairs", default="")
    parser.add_argument(
        "--reuse-checkpoints",
        action="store_true",
        help="Reuse an existing per-tag/per-seed checkpoint if present; missing checkpoints are trained normally.",
    )
    args = parser.parse_args()
    if args.anchor_mode is None:
        args.anchor_mode = "bottom_center" if args.clean_anchor else "multiplicative_gating_offsetted"
    run(args)


if __name__ == "__main__":
    main()
