#!/usr/bin/env python3
# Vendored HSG-SRL artifact copy. Isolated from the development repo;
# edit this file only. See src/SOURCE_MAP.csv.
"""Table IV BEV identity metrics (Hungarian HOTA / LocA).

This is the recorded Table IV scorer (Hungarian TP at a BEV gate).
Ground-plane tracking has no image IoU; the HOTA true-positive criterion is
a BEV metric gate. LocA is the mean of ``1 - d/gate`` on matched TPs.
BEV-HOTA is the mean HOTA over 0.5 / 1 / 1.5 / 2 m.

``evaluation/eval_fab2_tracking.py`` is a coarser ``hota_proxy`` and must
not be treated as Table IV.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

GATES = (0.5, 1.0, 1.5, 2.0)
VEHICLE_CLASSES = ("car", "bus", "truck", "van")
# Label-clock cutoff used by the recorded M6 Table IV window (cam5 video is
# black after ~362 s). Other paper sites leave this unset.
MAX_T = {"lumpi_M6": 360.0}

BinMap = dict[int, dict[int, tuple[float, float]]]
TimeMap = dict[str, tuple[float, float]]
SplitKey = tuple[str, int, int]


def identity_metrics(gt_by_bin: BinMap, pred_by_bin: BinMap, gate: float) -> dict[str, Any]:
    """DetA/AssA/LocA/HOTA/IDF1/MOTA/IDsw at one BEV metric gate.

    HOTA follows the standard decomposition sqrt(DetA * AssA) with the
    image-IoU localization threshold replaced by the BEV metric gate; the
    scalar BEV-HOTA reported outside is the mean over the gate sweep.
    LocA is the metric analogue of HOTA's localization component:
    mean over TP matches of (1 - d/gate).
    """
    tp = fp = fn = 0
    loca_sum = 0.0
    pair_counts: Counter = Counter()
    gt_frames: Counter = Counter()
    pred_frames: Counter = Counter()
    pred_total = gt_total = 0
    last_pid: dict[int, int] = {}
    idsw = 0
    for bn in sorted(set(gt_by_bin) | set(pred_by_bin)):
        gts = gt_by_bin.get(bn, {})
        preds = pred_by_bin.get(bn, {})
        gt_total += len(gts)
        pred_total += len(preds)
        for g in gts:
            gt_frames[g] += 1
        for p in preds:
            pred_frames[p] += 1
        if not gts or not preds:
            fp += len(preds)
            fn += len(gts)
            continue
        gids, pids = list(gts), list(preds)
        G = np.asarray([gts[g] for g in gids])
        P = np.asarray([preds[p] for p in pids])
        D = np.hypot(G[:, None, 0] - P[None, :, 0],
                     G[:, None, 1] - P[None, :, 1])
        ri, ci = linear_sum_assignment(np.where(D <= gate, D, 1e6))
        matched = set()
        for i, j in zip(ri, ci):
            if D[i, j] <= gate:
                tp += 1
                loca_sum += 1.0 - D[i, j] / gate
                g, p = gids[i], pids[j]
                pair_counts[(p, g)] += 1
                matched.add(g)
                if g in last_pid and last_pid[g] != p:
                    idsw += 1
                last_pid[g] = p
        fn += len(gids) - len(matched)
        fp += len(pids) - len(matched)
    pids_all = sorted({p for p, _ in pair_counts})
    gids_all = sorted({g for _, g in pair_counts})
    M = np.zeros((len(pids_all), len(gids_all)))
    pi = {p: i for i, p in enumerate(pids_all)}
    gi = {g: i for i, g in enumerate(gids_all)}
    for (p, g), n in pair_counts.items():
        M[pi[p], gi[g]] = n
    ri, ci = linear_sum_assignment(-M) if M.size else ((), ())
    idtp = M[ri, ci].sum() if M.size else 0.0
    # AssA: TP-weighted mean of TPA / (gt_frames + pred_frames - TPA)
    # over matched (pred, gt) pairs (standard HOTA association accuracy).
    assa = 0.0
    if tp:
        acc = 0.0
        for (p, g), tpa in pair_counts.items():
            acc += tpa * (tpa / (gt_frames[g] + pred_frames[p] - tpa))
        assa = acc / tp
    deta = tp / max(tp + fn + fp, 1)
    return dict(gate=gate, tp=tp, fp=fp, fn=fn, idsw=idsw,
                deta=deta, assa=assa, hota=float(np.sqrt(deta * assa)),
                loca=loca_sum / tp if tp else 0.0,
                idf1=2 * idtp / max(pred_total + gt_total, 1),
                mota=1.0 - (fp + fn + idsw) / max(gt_total, 1))


def mean_bev_hota(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return float(np.mean([float(r["hota"]) for r in rows]))


def load_split_keys(path: Path, split: str = "test") -> set[SplitKey]:
    if not path.is_file():
        raise FileNotFoundError(f"missing split manifest {path}")
    keys: set[SplitKey] = set()
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("split") != split:
                continue
            cam = r.get("camera_id") or r.get("camera_name") or ""
            keys.add((cam, int(r["local_track_id"]), int(r["frame_index"])))
    if not keys:
        raise ValueError(f"split manifest has no {split!r} rows: {path}")
    return keys


def load_time_maps_from_match_report(path: Path) -> TimeMap:
    blob = json.loads(path.read_text(encoding="utf-8"))
    cameras = blob.get("cameras", blob)
    tm: TimeMap = {}
    for cam, rec in cameras.items():
        if not isinstance(rec, dict) or "error" in rec:
            continue
        tm[cam] = (
            float(rec.get("dt_applied_sec", 0.0))
            + float(rec.get("drift_const_ms", 0.0)) / 1000.0,
            float(rec.get("drift_slope_ms_per_min", 0.0)) / 60000.0,
        )
    return tm


def parse_reference_bias(text: str) -> np.ndarray:
    """Apply common GT offset only when the verdict starts with REFERENCE OFFSET.

    Matches the recorded ``eval_site.py`` rule. A ``NO COMMON OFFSET`` file may
    still contain a ``common_bias=`` line; that value is not applied.
    """
    if not text.startswith("REFERENCE OFFSET"):
        return np.zeros(2)
    m = re.search(r"common_bias=\(([-+0-9.]+),([-+0-9.]+)\)", text)
    if not m:
        return np.zeros(2)
    return np.array([float(m.group(1)), float(m.group(2))])


def load_gt_bins(
    origin_db: Path,
    site: str,
    bias: np.ndarray,
    t_min: float,
    *,
    gt_batch: str = "alpha_gt_001",
    cameras: set[str] | None = None,
    max_t: float | None = None,
    vehicle_classes: tuple[str, ...] = VEHICLE_CLASSES,
) -> BinMap:
    conn = sqlite3.connect(str(origin_db))
    ph = ",".join("?" * len(vehicle_classes))
    cam_sql = ""
    cam_bind: list[Any] = []
    if cameras:
        cam_sql = " AND camera_name IN (" + ",".join("?" * len(cameras)) + ")"
        cam_bind = list(sorted(cameras))
    sql = (
        f"""SELECT global_id, timestamp_sec, outside_reference_x,
                   outside_reference_y FROM trajectory_observations
            WHERE scene_id=? AND batch_id=?
              AND class_name IN ({ph}){cam_sql}
              AND outside_reference_x IS NOT NULL"""
    )
    rows = conn.execute(sql, [site, gt_batch, *vehicle_classes, *cam_bind]).fetchall()
    if not rows:
        sql2 = (
            f"""SELECT global_id, timestamp_sec, outside_reference_x,
                       outside_reference_y FROM trajectory_observations
                WHERE scene_id=? AND batch_id=?
                  AND outside_reference_x IS NOT NULL{cam_sql}"""
        )
        rows = conn.execute(sql2, [site, gt_batch, *cam_bind]).fetchall()
    conn.close()
    if max_t is None:
        max_t = MAX_T.get(site)
    gt_by_bin: BinMap = defaultdict(dict)
    for gid, t, x, y in rows:
        if t < t_min or (max_t is not None and t > max_t):
            continue
        gt_by_bin[int(round(t * 10))][gid] = (x + bias[0], y + bias[1])
    return gt_by_bin


def load_predictions(
    db: Path,
    site: str,
    coord: str,
    tm: TimeMap,
    test_keys: set[SplitKey],
    *,
    pred_batch: str,
    cameras: set[str] | None = None,
) -> tuple[BinMap, float]:
    if coord not in {"world", "final_world"}:
        raise ValueError("coord must be 'world' or 'final_world'")
    conn = sqlite3.connect(str(db))
    acc: dict[tuple[int, int], list] = defaultdict(list)
    t_min_label = float("inf")
    for cam, tid, fr, gidp, t, x, y in conn.execute(
        f"""SELECT camera_name, local_track_id, frame_index, global_id,
                   timestamp_sec, {coord}_x, {coord}_y
            FROM trajectory_observations
            WHERE scene_id=? AND batch_id=? AND {coord}_x IS NOT NULL""",
        (site, pred_batch),
    ):
        if cameras is not None and cam not in cameras:
            continue
        if (cam, int(tid), int(fr)) not in test_keys:
            continue
        a, b = tm.get(cam, (0.0, 0.0))
        lt = t + a + b * t
        t_min_label = min(t_min_label, lt)
        acc[(int(round(lt * 10)), gidp)].append((x, y))
    conn.close()
    out: BinMap = defaultdict(dict)
    for (bn, gidp), pts in acc.items():
        arr = np.asarray(pts)
        out[bn][gidp] = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
    if t_min_label == float("inf"):
        t_min_label = 0.0
    return out, t_min_label


def self_check() -> None:
    """Tiny deterministic check that does not need licensed data."""
    gt: BinMap = {0: {1: (0.0, 0.0), 2: (10.0, 0.0)}}
    pred: BinMap = {0: {10: (0.1, 0.0), 20: (10.2, 0.0)}}
    r = identity_metrics(gt, pred, 1.0)
    assert r["tp"] == 2 and r["fp"] == 0 and r["fn"] == 0
    assert abs(r["loca"] - ((1.0 - 0.1) + (1.0 - 0.2)) / 2.0) < 1e-12
    assert abs(r["deta"] - 1.0) < 1e-12
    empty = identity_metrics({0: {1: (0.0, 0.0)}}, {}, 2.0)
    assert empty["tp"] == 0 and empty["fn"] == 1 and empty["loca"] == 0.0
    assert np.all(parse_reference_bias("NO COMMON OFFSET\ncommon_bias=(-0.1,+0.2)\n") == 0)
    applied = parse_reference_bias("REFERENCE OFFSET\ncommon_bias=(-0.0385,+0.0393)\n")
    assert abs(applied[0] + 0.0385) < 1e-12 and abs(applied[1] - 0.0393) < 1e-12


if __name__ == "__main__":
    self_check()
    print("bev_hota self-check ok")
