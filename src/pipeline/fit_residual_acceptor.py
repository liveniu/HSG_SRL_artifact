#!/usr/bin/env python3
# Vendored HSG-SRL artifact copy. Isolated from the development repo;
# edit this file only. See src/SOURCE_MAP.csv.
"""Label-free per-camera residual acceptor (val split only).

Does not read test keys, GT coordinates, or GT identities. Pair mining uses
the detection DB's geometric global_id and the overlap graph recorded in the
checkpoint (the graph the MLP was actually trained on).

Rule (same for every camera, no name list):
  * Cameras with fewer than N_min val MVC pairs: apply=false. Unsupervised
    cameras (not on the training overlap graph) are status=insufficient;
    cameras on the graph with a thin val window are status=thin_val.
    No-evidence is not treated as a license to apply ΔP.
  * Cameras with enough val pairs: apply ΔP unless the residual field
    *worsens mean* pair distance vs T_c-only (+2% band). p90/median are
    diagnostics so a fatter tail cannot turn off a camera whose bulk MVC
    gap shrinks.
  * After the per-camera tests, recompute mean pair distance under the
    deployed on/off mask (ΔP only on selected cameras). If that D_S
    worsens vs T_c-only, drop cameras from the set until the mask passes.

Writes residual_cameras + per-camera status JSON. Pass that JSON to fusion as
--g2-acceptor-json. Without it, fusion still zeros ΔP on cameras that are
not on the training overlap graph.

Usage:
  py pipeline/fit_residual_acceptor.py \\
      --db data/work/lumpi_M6.db \\
      --scene lumpi_M6 --batch paper_yolo_001 \\
      --checkpoint results/runs/lumpi_M6/training/frozen_checkpoints/lumpi_M6_seed0.pt \\
      --split-manifest splits/lumpi_M6/manifest/dataset_split_manifest.csv \\
      --tc-json results/runs/lumpi_M6/training/lumpi_M6_preflight_Tc.json \\
      --out results/runs/lumpi_M6/acceptor/lumpi_M6_seed0.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.geometry_guided_residual_mlp import (  # noqa: E402
    GeometryDataView,
    MVCResidualController,
    _filter_mvc_pairs_by_world_distance,
    _filter_observations_for_training_protocol,
    _gt_identity_leakage_report,
    _infer_and_calib_wh_from_db,
    _load_training_split_filter,
    _parse_time_offsets,
    apply_camera_translation_to_pairs,
    load_camera_translation_corrections,
)
from pipeline.multi_camera_trajectory_fusion import (  # noqa: E402
    adapt_homography_to_resolution,
)
from pipeline.residual_invoke import (  # noqa: E402
    cameras_on_overlap,
    overlap_from_meta,
)
from utils.scene_geometry_db import load_scene_overlap_pairs  # noqa: E402

WorsenTol = 1.02
MIN_PAIRS = 32


def _repo_rel(path: Path) -> str:
    """Store artifact paths as repo-relative POSIX strings for reproduction."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _norm_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))  # type: ignore[return-value]


def dist_stats(d: np.ndarray) -> dict[str, float]:
    if d.size == 0:
        return {"n": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0}
    return {
        "n": float(d.size),
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "p90": float(np.percentile(d, 90)),
    }


def worsens_mean(before: dict[str, float], after: dict[str, float]) -> tuple[bool, list[str]]:
    """Hard reject uses mean pair distance only.

    Residual training already accepts a minority of pairs getting worse (the
    M6 train summary is ~83% improved / ~17% worse). A T_c-style p90 gate
    would turn off C05/C06 when the bulk MVC gap shrinks but the tail fattens
    slightly. p90/median are recorded as diagnostics, not kill switches.
    """
    notes = []
    hit = False
    if after["n"] <= 0:
        return False, notes
    if after["mean"] > before["mean"] * WorsenTol:
        notes.append(
            f"mean pair distance {before['mean']:.4f}->{after['mean']:.4f} m"
        )
        hit = True
    return hit, notes


def tail_notes(before: dict[str, float], after: dict[str, float]) -> list[str]:
    notes = []
    if after["n"] <= 0:
        return notes
    if after["median"] > before["median"] * WorsenTol:
        notes.append(
            f"median pair distance {before['median']:.4f}->{after['median']:.4f} m (diagnostic)"
        )
    if after["p90"] > before["p90"] * WorsenTol:
        notes.append(
            f"p90 pair distance {before['p90']:.4f}->{after['p90']:.4f} m (diagnostic)"
        )
    return notes


def mine_split_pairs(
    *,
    db: Path,
    site: str,
    batch_id: str,
    split_name: str,
    manifest: Path,
    ctrl: MVCResidualController,
    overlap: set[tuple[str, str]],
    corrections,
):
    cfg = ctrl.train_cfg
    meta = ctrl._meta
    clip_px = int(meta.get("clip_margin_px", 3) or 3)
    clip_ratio = float(meta.get("clip_margin_ratio", 0.01) or 0.01)
    split_filter = _load_training_split_filter(
        manifest, split_name, scene_ids=[site], batch_id=batch_id,
    )
    if split_filter is None or split_filter.row_count < 1:
        raise SystemExit(f"split={split_name} empty in {manifest}")

    conn = sqlite3.connect(str(db))
    camera_names = [
        r[0]
        for r in conn.execute(
            """
            SELECT DISTINCT camera_name FROM trajectory_observations
             WHERE scene_id=? AND batch_id=? ORDER BY camera_name
            """,
            (site, batch_id),
        ).fetchall()
    ]
    view = GeometryDataView()
    calibrations = view.load_camera_calibrations(conn, site, camera_names)
    db_infer_wh, db_calib_wh = _infer_and_calib_wh_from_db(
        conn, site, batch_id, camera_names
    )
    frame_wh_by_camera = {
        cam: db_infer_wh.get(cam) or (1920, 1080) for cam in camera_names
    }
    adapted = {}
    for cam, (H_cal, sc) in calibrations.items():
        infer_wh = frame_wh_by_camera.get(cam)
        calib_wh = db_calib_wh.get(cam)
        if infer_wh and calib_wh and calib_wh != infer_wh:
            H_cal = adapt_homography_to_resolution(H_cal, calib_wh, infer_wh)
        adapted[cam] = (H_cal, sc)

    observations = list(
        view.iter_observations_from_db(
            conn,
            site,
            batch_id,
            camera_names,
            adapted,
            frame_wh_by_camera=frame_wh_by_camera,
            risk_scalar_mode=cfg.risk_scalar_mode,
            s0_m=cfg.s0_m,
            feature_l0_m=cfg.feature_l0_m,
            pixel_grazing_d0=cfg.pixel_grazing_d0,
            anchor_mode=cfg.anchor_mode,
            anchor_shape_eta=cfg.anchor_shape_eta,
            anchor_aspect_ratio_min=cfg.anchor_aspect_ratio_min,
            anchor_aspect_ratio_max=cfg.anchor_aspect_ratio_max,
            sensitivity_clip_m=cfg.sensitivity_clip_m,
            extent_clip_m=cfg.extent_clip_m,
            clip_margin_px=clip_px,
            clip_margin_ratio=clip_ratio,
        )
    )
    protocol_stats, observations = _filter_observations_for_training_protocol(
        observations,
        scene_id=site,
        batch_id=batch_id,
        split_filter=split_filter,
        time_start=None,
        time_end=None,
    )
    leak = _gt_identity_leakage_report(
        observations, scene_id=site, batch_id=batch_id, split_filter=split_filter,
    )
    if (
        leak["compared"] >= max(1, min(32, len(observations)))
        and leak["match_fraction"] >= 0.95
    ):
        conn.close()
        raise SystemExit(
            "acceptor abort: MVC mining looks like GT identities "
            f"({int(leak['matches'])}/{int(leak['compared'])} global_id==gt). "
            "Use the detection DB, not the GT-attached eval DB."
        )

    db_overlap = load_scene_overlap_pairs(conn, site, camera_names)
    conn.close()
    use_overlap = overlap or db_overlap
    if not use_overlap:
        raise SystemExit("no overlap graph in checkpoint meta or DB")

    mvc = view.mine_mvc_pairs(
        observations,
        use_overlap,
        max_time_gap_sec=float(cfg.max_time_gap_sec),
        motion_compensation=cfg.motion_compensation,
        time_offsets_sec=_parse_time_offsets(cfg.time_offsets_sec) or None,
        risk_scalar_mode=cfg.risk_scalar_mode,
        s0_m=cfg.s0_m,
        feature_l0_m=cfg.feature_l0_m,
        pixel_grazing_d0=cfg.pixel_grazing_d0,
        anchor_mode=cfg.anchor_mode,
        anchor_shape_eta=cfg.anchor_shape_eta,
        anchor_aspect_ratio_min=cfg.anchor_aspect_ratio_min,
        anchor_aspect_ratio_max=cfg.anchor_aspect_ratio_max,
        sensitivity_clip_m=cfg.sensitivity_clip_m,
        extent_clip_m=cfg.extent_clip_m,
    )
    raw = len(mvc)
    mvc = apply_camera_translation_to_pairs(mvc, corrections)
    mvc = _filter_mvc_pairs_by_world_distance(mvc, cfg.max_pair_dist_m)
    dist_n = len(mvc)
    mvc = ctrl.model.select_mvc_pairs(mvc)
    return {
        "camera_names": camera_names,
        "overlap": sorted(use_overlap),
        "db_overlap": sorted(db_overlap),
        "protocol_stats": protocol_stats,
        "gt_identity_guard": leak,
        "raw_mvc": raw,
        "dist_mvc": dist_n,
        "selected_mvc": len(mvc),
        "pairs": mvc,
        "n_obs": len(observations),
    }


def pair_distances(pairs, model):
    """T_c-only vs both-on residual distances, plus per-side leave-one-on."""
    if not pairs:
        empty = np.zeros(0)
        return empty, empty, {}
    feat_a = np.stack([p.obs_a.feature for p in pairs]).astype(np.float32)
    feat_b = np.stack([p.obs_b.feature for p in pairs]).astype(np.float32)
    geo_a = np.asarray([p.obs_a.world_geo for p in pairs], dtype=np.float64)
    geo_b = np.asarray([p.obs_b.world_geo for p in pairs], dtype=np.float64)
    da = model.predict_delta(feat_a).astype(np.float64)
    db = model.predict_delta(feat_b).astype(np.float64)
    d0 = np.linalg.norm(geo_a - geo_b, axis=1)
    d_both = np.linalg.norm((geo_a + da) - (geo_b + db), axis=1)
    cams_a = [p.obs_a.camera_name for p in pairs]
    cams_b = [p.obs_b.camera_name for p in pairs]
    return d0, d_both, {
        "da": da, "db": db, "geo_a": geo_a, "geo_b": geo_b,
        "cams_a": cams_a, "cams_b": cams_b,
    }


def distances_under_mask(pack: dict[str, Any], selected: set[str]) -> np.ndarray:
    """Pair distances when ΔP is applied only on cameras in ``selected``."""
    if not pack:
        return np.zeros(0)
    geo_a = np.asarray(pack["geo_a"], dtype=np.float64)
    geo_b = np.asarray(pack["geo_b"], dtype=np.float64)
    da = np.asarray(pack["da"], dtype=np.float64)
    db = np.asarray(pack["db"], dtype=np.float64)
    sel_a = np.array(
        [str(c) in selected for c in pack["cams_a"]], dtype=np.float64
    )[:, None]
    sel_b = np.array(
        [str(c) in selected for c in pack["cams_b"]], dtype=np.float64
    )[:, None]
    return np.linalg.norm((geo_a + da * sel_a) - (geo_b + db * sel_b), axis=1)


def close_deployed_mask(
    d0: np.ndarray,
    pack: dict[str, Any],
    selected: list[str],
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Recheck mean MVC gap under the actual on/off set; drop until it passes.

    D_1(c) is both-on. Deployment may leave a partner off, so this pass
    evaluates D_S with Delta P only on the selected cameras.
    """
    chosen: set[str] = {str(c) for c in selected}
    dropped: list[str] = []
    before = dist_stats(d0)
    d_init = distances_under_mask(pack, chosen)
    init_hit, init_notes = worsens_mean(before, dist_stats(d_init))

    while chosen and d0.size:
        d_s = distances_under_mask(pack, chosen)
        hit, _notes = worsens_mean(before, dist_stats(d_s))
        if not hit:
            break
        best_cam: str | None = None
        best_mean: float | None = None
        for cam in sorted(chosen):
            trial = chosen - {cam}
            d_trial = distances_under_mask(pack, trial)
            mean_trial = float(np.mean(d_trial)) if d_trial.size else 0.0
            if best_mean is None or mean_trial < best_mean:
                best_mean = mean_trial
                best_cam = cam
        if best_cam is None:
            break
        chosen.remove(best_cam)
        dropped.append(best_cam)

    d_final = distances_under_mask(pack, chosen)
    after = dist_stats(d_final)
    final_hit, final_notes = worsens_mean(before, after)
    return sorted(chosen), dropped, {
        "n": int(d0.size),
        "before": before,
        "after_initial_mask": dist_stats(d_init),
        "after_final_mask": after,
        "initial_mean_worsens": bool(init_hit),
        "initial_notes": init_notes + tail_notes(before, dist_stats(d_init)),
        "final_mean_worsens": bool(final_hit),
        "final_notes": final_notes + tail_notes(before, after),
        "dropped": list(dropped),
    }


def camera_subset(d0, d_both, pack, camera: str):
    idx = [i for i, (a, b) in enumerate(zip(pack["cams_a"], pack["cams_b"]))
           if a == camera or b == camera]
    if not idx:
        z = np.zeros(0)
        return z, z, z
    ii = np.asarray(idx)
    d_loo = np.empty(len(ii), dtype=np.float64)
    for k, i in enumerate(ii):
        pa = pack["geo_a"][i].copy()
        pb = pack["geo_b"][i].copy()
        if pack["cams_a"][i] == camera:
            pa = pa + pack["da"][i]
        if pack["cams_b"][i] == camera:
            pb = pb + pack["db"][i]
        d_loo[k] = float(np.linalg.norm(pa - pb))
    return d0[ii], d_both[ii], d_loo


def edge_stats(d0, d_both, pack) -> dict[str, Any]:
    """Per-edge mean/median/p90, independent of the camera allowlist."""
    if not pack:
        return {}
    buckets: dict[tuple[str, str], list[int]] = {}
    for i, (a, b) in enumerate(zip(pack["cams_a"], pack["cams_b"])):
        e = _norm_pair(a, b)
        buckets.setdefault(e, []).append(i)
    out: dict[str, Any] = {}
    for e, idx in sorted(buckets.items()):
        ii = np.asarray(idx)
        b = dist_stats(d0[ii])
        a = dist_stats(d_both[ii])
        hit, notes = worsens_mean(b, a)
        out[f"{e[0]}:{e[1]}"] = {
            "n": int(ii.size),
            "before": b,
            "after_both_on": a,
            "mean_worsens": hit,
            "notes": notes + tail_notes(b, a),
        }
    return out


def decide_camera(
    *,
    camera: str,
    n_pairs: int,
    on_train_overlap: bool,
    before: dict[str, float],
    after: dict[str, float],
    loo: dict[str, float],
    min_pairs: int,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "camera": camera,
        "n_val_pairs": n_pairs,
        "on_train_overlap": on_train_overlap,
        "before": before,
        "after_both_on": after,
        "after_leave_one_on": loo,
        "apply": False,
        "status": "insufficient",
        "notes": [],
    }
    if n_pairs < min_pairs:
        rec["apply"] = False
        if on_train_overlap:
            rec["status"] = "thin_val"
            rec["notes"].append(
                f"val pairs {n_pairs} < {min_pairs}; too thin to test, ΔP=0"
            )
        else:
            rec["status"] = "insufficient"
            rec["notes"].append(
                f"val pairs {n_pairs} < {min_pairs} and camera is not on the "
                "training overlap graph; ΔP=0"
            )
        return rec

    hit, notes = worsens_mean(before, after)
    rec["notes"].extend(notes)
    rec["notes"].extend(tail_notes(before, after))
    if hit:
        rec["status"] = "not_recommended"
        rec["apply"] = False
        rec["notes"].append("val MVC mean pair distance worsens with residual; ΔP=0")
        return rec
    rec["status"] = "recommended"
    rec["apply"] = True
    rec["notes"].append("val MVC mean pair distance does not worsen; keep ΔP")
    loo_hit, loo_notes = worsens_mean(before, loo)
    rec["leave_one_on_worsens"] = loo_hit
    rec["leave_one_on_notes"] = loo_notes + tail_notes(before, loo)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--tc-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--min-pairs", type=int, default=MIN_PAIRS)
    ap.add_argument("--seed", type=int, default=None,
                    help="optional seed tag written into the JSON")
    args = ap.parse_args()
    min_pairs = int(args.min_pairs)

    yolo = Path(args.db)
    manifest = Path(args.split_manifest)
    ckpt = Path(args.checkpoint)
    tc_json = Path(args.tc_json)
    dest = Path(args.out)
    for p in (yolo, manifest, ckpt, tc_json):
        if not p.exists():
            raise SystemExit(f"missing {p}")

    corrections = load_camera_translation_corrections(
        tc_json, acceptance="recommended",
    )
    ctrl = MVCResidualController.from_checkpoint(
        ckpt, camera_translation_corrections=corrections,
    )
    overlap = overlap_from_meta(ctrl._meta)
    train_cams = cameras_on_overlap(overlap)
    print(
        f"[acceptor] {args.scene}  "
        f"train_overlap={sorted(overlap)}  train_cameras={sorted(train_cams)}",
        flush=True,
    )

    mined = mine_split_pairs(
        db=yolo, site=args.scene, batch_id=args.batch, split_name=args.split,
        manifest=manifest, ctrl=ctrl, overlap=overlap, corrections=corrections,
    )
    pairs = mined["pairs"]
    print(
        f"[acceptor] val obs={mined['n_obs']}  raw_mvc={mined['raw_mvc']}  "
        f"dist={mined['dist_mvc']}  selected={mined['selected_mvc']}  "
        f"gt_id_match={mined['gt_identity_guard']['match_fraction']:.4f}",
        flush=True,
    )

    d0, d_both, pack = pair_distances(pairs, ctrl.model)
    cameras = list(mined["camera_names"])
    by_cam: dict[str, Any] = {}
    for cam in cameras:
        b, a, lo = camera_subset(d0, d_both, pack, cam) if pack else (
            np.zeros(0), np.zeros(0), np.zeros(0)
        )
        rec = decide_camera(
            camera=cam,
            n_pairs=int(b.size),
            on_train_overlap=cam in train_cams,
            before=dist_stats(b),
            after=dist_stats(a),
            loo=dist_stats(lo),
            min_pairs=min_pairs,
        )
        by_cam[cam] = rec
        print(
            f"[acceptor] {cam}: status={rec['status']} apply={rec['apply']} "
            f"n={rec['n_val_pairs']} overlap={rec['on_train_overlap']}",
            flush=True,
        )

    edges = edge_stats(d0, d_both, pack)
    for name, e in edges.items():
        print(
            f"[acceptor] edge {name}: n={e['n']} mean "
            f"{e['before']['mean']:.3f}->{e['after_both_on']['mean']:.3f} "
            f"worsens={e['mean_worsens']}",
            flush=True,
        )

    residual_cameras = sorted(c for c, rec in by_cam.items() if rec["apply"])
    initial_residual_cameras = list(residual_cameras)
    residual_cameras, dropped, mask_rec = close_deployed_mask(
        d0, pack, residual_cameras,
    )
    for cam in dropped:
        rec = by_cam.get(cam)
        if rec is None:
            continue
        rec["apply"] = False
        rec["status"] = "not_recommended"
        rec["dropped_by_mask_recheck"] = True
        rec["notes"].append(
            "deployed-mask D_S worsens vs T_c-only; dropped in closed-loop recheck"
        )
    if pack:
        d_s = distances_under_mask(pack, set(residual_cameras))
        for cam, rec in by_cam.items():
            idx = [
                i for i, (a, b) in enumerate(zip(pack["cams_a"], pack["cams_b"]))
                if a == cam or b == cam
            ]
            if not idx:
                rec["after_deployed_mask"] = dist_stats(np.zeros(0))
                continue
            ii = np.asarray(idx)
            rec["after_deployed_mask"] = dist_stats(d_s[ii])
    print(
        f"[acceptor] deployed_mask n={mask_rec['n']} mean "
        f"{mask_rec['before']['mean']:.3f}->{mask_rec['after_final_mask']['mean']:.3f} "
        f"initial_worsens={mask_rec['initial_mean_worsens']} "
        f"dropped={dropped}",
        flush=True,
    )

    out = {
        "version": "mlp_percam_acceptor_v1",
        "site": args.scene,
        "seed": args.seed,
        "split": args.split,
        "uses_test_gt": False,
        "uses_gt_identity": False,
        "min_pairs": min_pairs,
        "worsen_tolerance": WorsenTol,
        "train_overlap_pairs": [list(p) for p in sorted(overlap)],
        "mine_overlap_pairs": [list(p) for p in sorted(overlap)],
        "db_overlap_pairs": [list(p) for p in mined["db_overlap"]],
        "edges": edges,
        "val_mining": {
            "n_obs": mined["n_obs"],
            "raw_mvc": mined["raw_mvc"],
            "dist_mvc": mined["dist_mvc"],
            "selected_mvc": mined["selected_mvc"],
            "protocol_stats": mined["protocol_stats"],
            "gt_identity_guard": mined["gt_identity_guard"],
        },
        "initial_residual_cameras": initial_residual_cameras,
        "residual_cameras": residual_cameras,
        "deployed_mask": mask_rec,
        "cameras": by_cam,
        "checkpoint": _repo_rel(ckpt),
        "detection_db": _repo_rel(yolo),
        "split_manifest": _repo_rel(manifest),
        "tc_json": _repo_rel(tc_json),
        "rule": (
            "ΔP only if val MVC count >= N_min and mean pair distance does "
            "not worsen vs T_c-only, then recheck D_S under the deployed on/off "
            "mask and drop cameras if the combination worsens. Thin-val and "
            "unsupervised cameras get ΔP=0. T_c is not gated here."
        ),
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"[acceptor] residual_cameras={residual_cameras}")
    print(f"[acceptor] wrote {dest}")


if __name__ == "__main__":
    main()
