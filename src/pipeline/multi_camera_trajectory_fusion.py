#!/usr/bin/env python3
# Vendored HSG-SRL artifact copy. Isolated from the development repo;
# edit this file only. See src/SOURCE_MAP.csv.
"""Multi-camera vehicle trajectory drawing, BEV mapping, and global-ID fusion.

This module reads the calibration database produced by
``pipeline/multi_camera_bev_stitch.py`` and runs a detect/track backend on each
camera (default Ultralytics YOLO+BoT-SORT; optional MMDet YOLOX + standalone
BoT-SORT). Local track points are projected from image coordinates into a shared
world frame; in camera overlap regions, candidate tracks are matched by time,
distance, and trajectory shape and merged into stable global IDs.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import copy
import json
import math
import os
import signal
import sqlite3
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import cv2 as cv
import numpy as np
import yaml

from utils.image_text import draw_label, warn_if_no_pil
from utils.project_paths import resolve_project_path
from utils.image_io import imwrite
from pipeline.multi_camera_bev_stitch import (
    GEOMETRY_VERSION,
    camera_roi_points,
    init_db as init_calibration_db,
    load_calibration,
    transform_camera_roi,
)
from utils.scene_geometry_db import save_scene_overlap_pairs
from utils.trajectory_batches import (
    finish_batch_stage,
    initialize_trajectory_batch_schema,
    start_batch_stage,
    validate_batch_id,
)
from detection.detect_track_backends import (
    backend_class_names,
    load_track_backends,
)


VEHICLE_COCO_CLASSES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
DEFAULT_ANCHOR_SHAPE_ETA = 0.75
DEFAULT_ANCHOR_ASPECT_RATIO_MIN = 0.5   # rho=h/w
DEFAULT_ANCHOR_ASPECT_RATIO_MAX = 3.0

_ORIN_STAGE_PROFILE = (
    _PROJECT_ROOT / "experiments_delta_orin" / "step4_stage" / "stage_profile.py"
)


def _load_orin_stage_profile():
    """IV-F CSV sinks live with experiments_delta_orin, not in pipeline/."""
    import importlib.util

    path = _ORIN_STAGE_PROFILE
    if not path.is_file():
        raise SystemExit(
            "--stage-profile / --remerge-profile are Orin IV-F instrumentation; "
            f"missing {path}"
        )
    spec = importlib.util.spec_from_file_location("orin_ivf_stage_profile", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_TRACK_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS scene_canvas (
    scene_id   TEXT    NOT NULL,
    scale      REAL    NOT NULL,
    min_x      REAL    NOT NULL,
    min_y      REAL    NOT NULL,
    canvas_w   INTEGER NOT NULL,
    canvas_h   INTEGER NOT NULL,
    saved_at   TEXT    NOT NULL,
    PRIMARY KEY (scene_id)
);

CREATE TABLE IF NOT EXISTS trajectory_global_tracks (
    scene_id      TEXT    NOT NULL DEFAULT 'parking',
    batch_id      TEXT    NOT NULL,
    global_id     INTEGER NOT NULL,
    created_at    TEXT    NOT NULL,
    source        TEXT,
    PRIMARY KEY (scene_id, batch_id, global_id)
);

CREATE TABLE IF NOT EXISTS trajectory_observations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    scene_id         TEXT    NOT NULL DEFAULT 'parking',
    batch_id         TEXT    NOT NULL,
    camera_name      TEXT    NOT NULL,
    local_track_id   INTEGER NOT NULL,
    global_id        INTEGER NOT NULL,
    frame_index      INTEGER NOT NULL,
    timestamp_sec    REAL    NOT NULL,
    image_x          REAL    NOT NULL,
    image_y          REAL    NOT NULL,
    world_x          REAL    NOT NULL,
    world_y          REAL    NOT NULL,
    final_world_x    REAL,
    final_world_y    REAL,
    outside_reference_x  REAL,
    outside_reference_y  REAL,
    outside_reference_image_x  REAL,
    outside_reference_image_y  REAL,
    bbox_x           REAL    NOT NULL,
    bbox_y           REAL    NOT NULL,
    bbox_w           REAL    NOT NULL,
    bbox_h           REAL    NOT NULL,
    confidence       REAL,
    class_id         INTEGER,
    class_name       TEXT,
    recorded_at      TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS trajectory_merges (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    scene_id               TEXT    NOT NULL DEFAULT 'parking',
    batch_id               TEXT    NOT NULL,
    merge_kind             TEXT    NOT NULL DEFAULT 'cross_camera',
    kept_global_id         INTEGER NOT NULL,
    merged_global_id       INTEGER NOT NULL,
    camera_a               TEXT    NOT NULL,
    local_track_id_a       INTEGER NOT NULL,
    camera_b               TEXT    NOT NULL,
    local_track_id_b       INTEGER NOT NULL,
    score                  REAL    NOT NULL,
    time_score             REAL    NOT NULL,
    distance_score         REAL    NOT NULL,
    shape_score            REAL    NOT NULL,
    mean_distance_m        REAL    NOT NULL,
    time_gap_sec           REAL    NOT NULL,
    comparison_count       INTEGER NOT NULL,
    decided_at             TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS trajectory_runtime_stats (
    id                            INTEGER PRIMARY KEY AUTOINCREMENT,
    scene_id                       TEXT    NOT NULL DEFAULT 'parking',
    batch_id                       TEXT    NOT NULL,
    frame_index                    INTEGER NOT NULL,
    timestamp_sec                  REAL    NOT NULL,
    active_local_tracks            INTEGER NOT NULL,
    active_global_tracks           INTEGER NOT NULL,
    merge_candidate_comparisons    INTEGER NOT NULL,
    merge_complexity               TEXT    NOT NULL,
    db_rows_written                INTEGER NOT NULL,
    storage_rows_per_sec           REAL    NOT NULL,
    storage_bytes_per_sec          REAL    NOT NULL,
    recorded_at                    TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS trajectory_camera_config (
    scene_id    TEXT    NOT NULL,
    batch_id    TEXT    NOT NULL,
    camera_name TEXT    NOT NULL,
    infer_w     INTEGER NOT NULL,
    infer_h     INTEGER NOT NULL,
    calib_w     INTEGER,
    calib_h     INTEGER,
    updated_at  TEXT    NOT NULL,
    PRIMARY KEY (scene_id, batch_id, camera_name)
);
CREATE TABLE IF NOT EXISTS scene_overlap_pairs (
    scene_id   TEXT NOT NULL,
    camera_a   TEXT NOT NULL,
    camera_b   TEXT NOT NULL,
    PRIMARY KEY (scene_id, camera_a, camera_b)
);
"""


@dataclass
class TrackSample:
    timestamp_sec: float
    frame_index: int
    image_xy: tuple[float, float]
    world_xy: tuple[float, float]
    bbox_xywh: tuple[float, float, float, float]
    confidence: float | None
    class_id: int | None
    class_name: str | None
    # Defensive flag: the main loop drops edge-clipped bboxes before building
    # TrackSample; if another caller passes clipped=True, the point still
    # does not take part in trajectory matching.
    clipped: bool = False


@dataclass
class LocalTrack:
    camera_name: str
    local_track_id: int
    global_id: int
    max_history: int
    samples: deque[TrackSample] = field(init=False)
    # Keep the track-head samples separately: samples is a bounded deque, so
    # long tracks drop early points, but same-camera stitching needs the new
    # track's start position / initial velocity.
    head_samples: list[TrackSample] = field(init=False)
    last_seen_frame: int = 0
    last_seen_time: float = 0.0
    first_seen_time: float = 0.0
    sample_count: int = 0

    _HEAD_KEEP = 5

    def __post_init__(self) -> None:
        self.samples = deque(maxlen=self.max_history)
        self.head_samples = []

    def add_sample(self, sample: TrackSample) -> None:
        if self.sample_count == 0:
            self.first_seen_time = sample.timestamp_sec
        self.sample_count += 1
        self.samples.append(sample)
        if not sample.clipped and len(self.head_samples) < self._HEAD_KEEP:
            self.head_samples.append(sample)
        self.last_seen_frame = sample.frame_index
        self.last_seen_time = sample.timestamp_sec


@dataclass
class ObservationRow:
    scene_id: str
    batch_id: str
    camera_name: str
    local_track_id: int
    global_id: int
    frame_index: int
    timestamp_sec: float
    image_xy: tuple[float, float]
    world_xy: tuple[float, float]
    bbox_xywh: tuple[float, float, float, float]
    confidence: float | None
    class_id: int | None
    class_name: str | None
    # Final coordinates after the g2 residual MLP; None means g2 was not used.
    # world_xy always stores the pure-homography projection P_geo (fixed meaning):
    #   · MVC offline training always reads world_x/y from the DB (i.e. P_geo),
    #     independent of whether g2 was running;
    #   · during online tracking, TrackSample.world_xy (used by merge/BEV) =
    #     final_world_xy if g2 else p_geo_xy; the DB stores P_geo in world_xy
    #     and the g2 result in final_world_xy, in separate columns;
    #   · remerge defaults to DB P_geo; if final is chosen explicitly, only
    #     existing final_world_x/y values are used.
    final_world_xy: tuple[float, float] | None = None


@dataclass
class MatchScores:
    score: float
    time_score: float
    distance_score: float
    shape_score: float
    mean_distance_m: float
    time_gap_sec: float


@dataclass
class MergeEvent:
    merge_kind: str
    kept_global_id: int
    merged_global_id: int
    track_a: LocalTrack
    track_b: LocalTrack
    scores: MatchScores
    comparison_count: int


@dataclass
class MatchConfig:
    score_threshold: float = 0.78
    max_time_gap_sec: float = 1.2
    max_distance_m: float = 1.5
    min_samples: int = 5
    sample_count: int = 12
    active_timeout_sec: float = 2.0
    time_sigma: float = 0.6
    distance_sigma: float = 0.75
    shape_sigma: float = 0.35
    time_weight: float = 0.25
    distance_weight: float = 0.45
    shape_weight: float = 0.30
    # ── Same-camera tracklet stitching (temporal reconnect of broken tracks) ─
    # Cross-camera merge can only repair a break when another view provides a
    # bridge; same-camera stitching handles fragments that abut in time after
    # occlusion or missed detections. Evidence is motion extrapolation
    # (tail velocity × time gap), not appearance, so the method stays
    # scene- and training-data agnostic.
    # Off by default: enable only after validating on the target scene / public
    # benchmarks as an optional unified tracking backend.
    stitch_enabled: bool = False
    # finalize=0 means auto-derive from tracker_buffer_frames × track sample
    # period + margin.
    stitch_finalize_sec: float = 0.0
    tracker_buffer_frames: int = 30
    stitch_finalize_margin_sec: float = 0.5
    stitch_default_sample_period_sec: float = 0.1
    stitch_max_gap_sec: float = 3.0          # max gap from old track end to new track start
    # overlap=0 means auto-derive from both sample periods; base/rate>0 is
    # kept only for legacy config compatibility.
    stitch_max_overlap_sec: float = 0.0
    stitch_base_tolerance_m: float = 0.0
    stitch_tolerance_rate_mps: float = 0.0
    stitch_min_samples: int = 3              # min samples on both sides of a stitch
    stitch_direction_min_cos: float = 0.25   # min direction cosine when both are moving
    stitch_min_speed_mps: float = 0.5        # below this speed treat as stationary; skip direction check
    stitch_speed_sigma_mps: float = 5.0      # exponential decay scale for the speed-difference score
    stitch_score_threshold: float = 0.85     # min accepted score for a Hungarian real edge
    stitch_uncertainty_floor_m: float = 0.25
    stitch_process_noise_mps2: float = 0.5
    stitch_max_normalized_error: float = 3.0
    # Abstain when predicted std is too large vs the combined two-end
    # uncertainty floor.
    stitch_max_uncertainty_ratio: float = 4.0
    stitch_require_mutual_best: bool = True
    stitch_min_score_margin: float = 0.05
    # ── Same-camera spatial exclusion guard ────────────────────────────────
    # The tracker may emit time-overlapping duplicate tracklets for one vehicle;
    # temporal overlap alone is not a conflict. Only persistently well-separated
    # near-simultaneous BEV positions make the same identity physically impossible.
    guard_sync_tolerance_sec: float = 0.0
    guard_sync_period_factor: float = 1.5
    guard_max_separation_m: float = 0.0       # >0 enables the legacy metric gate
    guard_position_sigma_floor_m: float = 0.5
    guard_position_sigma_cap_m: float = 0.75
    guard_max_normalized_separation: float = 3.0
    guard_min_conflicting_samples: int = 2


DEFAULT_MATCH_CONFIG = MatchConfig()


def _linking_config_summary(cfg: MatchConfig) -> str:
    finalize = (
        f"{cfg.stitch_finalize_sec:.2f}s"
        if cfg.stitch_finalize_sec > 0
        else f"auto({cfg.tracker_buffer_frames} frames + {cfg.stitch_finalize_margin_sec:.2f}s)"
    )
    gate = (
        f"{cfg.stitch_base_tolerance_m:.2f}m"
        f"+{cfg.stitch_tolerance_rate_mps:.2f}m/s"
        if cfg.stitch_base_tolerance_m + cfg.stitch_tolerance_rate_mps > 0
        else f"normalized<={cfg.stitch_max_normalized_error:.2f}"
    )
    guard = (
        f"{cfg.guard_max_separation_m:.2f}m"
        if cfg.guard_max_separation_m > 0
        else f"normalized>{cfg.guard_max_normalized_separation:.2f}"
    )
    return (
        f"finalize={finalize}  gap<={cfg.stitch_max_gap_sec:.1f}s  "
        f"position_gate={gate}  uncertainty_ratio<={cfg.stitch_max_uncertainty_ratio:.1f}  "
        f"mutual={cfg.stitch_require_mutual_best} margin={cfg.stitch_min_score_margin:.2f}  "
        f"guard={guard}"
    )


def _track_sample_period_sec(track: LocalTrack, cfg: MatchConfig) -> float:
    """Robustly estimate sample period from timestamps; a low quantile damps long gaps from missed detections."""
    times = np.array(
        sorted({s.timestamp_sec for s in track.samples if not s.clipped}),
        dtype=np.float64,
    )
    if len(times) < 2:
        return cfg.stitch_default_sample_period_sec
    diffs = np.diff(times)
    diffs = diffs[diffs > 1e-6]
    if len(diffs) == 0:
        return cfg.stitch_default_sample_period_sec
    period = float(np.quantile(diffs, 0.25))
    return min(1.0, max(1.0 / 120.0, period))


def _track_finalize_sec(track: LocalTrack, cfg: MatchConfig) -> float:
    if cfg.stitch_finalize_sec > 0:
        return cfg.stitch_finalize_sec
    return (
        cfg.tracker_buffer_frames * _track_sample_period_sec(track, cfg)
        + cfg.stitch_finalize_margin_sec
    )


def _track_overlap_tolerance_sec(
    track_a: LocalTrack,
    track_b: LocalTrack,
    cfg: MatchConfig,
) -> float:
    if cfg.stitch_max_overlap_sec > 0:
        return cfg.stitch_max_overlap_sec
    return 1.5 * max(
        _track_sample_period_sec(track_a, cfg),
        _track_sample_period_sec(track_b, cfg),
    )


def _track_position_sigma_m(track: LocalTrack, cfg: MatchConfig) -> float:
    """Estimate same-camera track position noise from local linear-fit residuals."""
    samples = [s for s in track.samples if not s.clipped][-8:]
    if len(samples) < 3:
        return cfg.guard_position_sigma_floor_m
    times = np.array([s.timestamp_sec for s in samples], dtype=np.float64)
    points = np.array([s.world_xy for s in samples], dtype=np.float64)
    dt = times - float(np.mean(times))
    design = np.column_stack((dt, np.ones_like(dt)))
    fitted = design @ np.linalg.lstsq(design, points, rcond=None)[0]
    residual = float(np.sqrt(np.mean(np.sum((points - fitted) ** 2, axis=1))))
    return min(
        cfg.guard_position_sigma_cap_m,
        max(cfg.guard_position_sigma_floor_m, residual),
    )


def _same_camera_tracks_conflict(
    track_a: LocalTrack,
    track_b: LocalTrack,
    cfg: MatchConfig,
) -> bool:
    """Decide whether two same-camera tracklets give physically exclusive co-timed position evidence.

    Overlapping time intervals may just be duplicate tracks after a tracker ID
    switch and must not by themselves reject a union. A monotonic two-pointer
    matches near-simultaneous samples; only several pairs of co-timed BEV
    positions that are too far apart prove the tracks cannot be the same
    vehicle. With no co-timed samples, stay conservative and do not reject.
    """
    if track_a.camera_name != track_b.camera_name:
        return False
    samples_a = [s for s in track_a.samples if not s.clipped]
    samples_b = [s for s in track_b.samples if not s.clipped]
    sync_tolerance = cfg.guard_sync_tolerance_sec
    if sync_tolerance <= 0:
        sync_tolerance = min(
            0.5,
            cfg.guard_sync_period_factor
            * max(
                _track_sample_period_sec(track_a, cfg),
                _track_sample_period_sec(track_b, cfg),
            ),
        )
    sigma = math.hypot(
        _track_position_sigma_m(track_a, cfg),
        _track_position_sigma_m(track_b, cfg),
    )
    i = 0
    j = 0
    conflicts = 0
    while i < len(samples_a) and j < len(samples_b):
        sa = samples_a[i]
        sb = samples_b[j]
        dt = sa.timestamp_sec - sb.timestamp_sec
        if abs(dt) <= sync_tolerance:
            distance = math.dist(sa.world_xy, sb.world_xy)
            is_conflict = (
                distance > cfg.guard_max_separation_m
                if cfg.guard_max_separation_m > 0
                else distance / max(sigma, 1e-6)
                > cfg.guard_max_normalized_separation
            )
            if is_conflict:
                conflicts += 1
                if conflicts >= cfg.guard_min_conflicting_samples:
                    return True
            i += 1
            j += 1
        elif dt < 0:
            i += 1
        else:
            j += 1
    return False


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def make(self, item: int) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: int) -> int:
        self.make(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union_keep_smallest(self, a: int, b: int) -> tuple[int, int] | None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return None
        kept, merged = (ra, rb) if ra < rb else (rb, ra)
        self.parent[merged] = kept
        return kept, merged


@dataclass
class CameraRuntime:
    name: str
    cfg: dict[str, Any]           # original YAML camera config (main-stream resolution coords)
    effective_cfg: dict[str, Any] # inference config (image_roi already scaled to infer-stream resolution)
    source: str | int
    cap: cv.VideoCapture
    first_frame: np.ndarray | None
    homography: np.ndarray        # image coords (infer-stream resolution) → BEV pixels
    calib_homography: np.ndarray  # image coords (main/calib resolution) → BEV pixels (same as bev_stitch)
    scale: float
    frame_index: int = 0
    fps: float = 25.0
    last_timestamp_sec: float = 0.0  # last timestamp handed to inference (offline timeline)
    writer: VideoRecorder | None = None
    # RTSP reconnect state
    consecutive_failures: int = 0
    last_good_frame: np.ndarray | None = None
    rtsp_transport: str = "tcp"         # "tcp" or "udp"; tcp is more stable
    # Multi-stream info (filled for sub_stream; used in logs and debugging)
    calib_wh: tuple[int, int] | None = None   # main-stream W×H at calibration time
    infer_wh: tuple[int, int] | None = None   # infer-stream (sub-stream) W×H
    # Debug recording: async read + write path (decoupled from YOLO to cut lag and stale pads)
    writer_path: str | None = None
    read_lock: threading.Lock = field(default_factory=threading.Lock)
    latest_frame: np.ndarray | None = None
    latest_frame_time: float = 0.0
    latest_frame_seq: int = 0
    last_consumed_seq: int = -1
    async_reader_stop: threading.Event | None = None
    async_reader_thread: threading.Thread | None = None
    stale_write_count: int = 0
    last_track_ms: float = 0.0


@dataclass
class CanvasGeometry:
    camera_names: list[str]
    scale: float
    min_xy: np.ndarray
    translate: np.ndarray
    size: tuple[int, int]
    coverage_masks: dict[str, np.ndarray]
    overlap_pairs: set[tuple[str, str]]

    def world_to_canvas(self, world_xy: tuple[float, float] | np.ndarray) -> tuple[int, int]:
        xy = np.asarray(world_xy, dtype=np.float64)
        px = xy * float(self.scale) - self.min_xy.astype(np.float64)
        return int(round(px[0])), int(round(px[1]))

    def point_in_pair_overlap(
        self,
        camera_a: str,
        camera_b: str,
        world_xy: tuple[float, float] | np.ndarray,
    ) -> bool:
        pair = tuple(sorted((camera_a, camera_b)))
        if pair not in self.overlap_pairs:
            return False
        x, y = self.world_to_canvas(world_xy)
        width, height = self.size
        if x < 0 or y < 0 or x >= width or y >= height:
            return False
        return (
            self.coverage_masks[camera_a][y, x] > 0
            and self.coverage_masks[camera_b][y, x] > 0
        )


class TrajectoryFusion:
    def __init__(
        self,
        conn: sqlite3.Connection,
        scene_id: str,
        batch_id: str,
        geometry: CanvasGeometry,
        match_cfg: MatchConfig,
        max_history: int,
        auto_commit_merge_events: bool = True,
    ) -> None:
        self.conn = conn
        self.scene_id = scene_id
        self.batch_id = validate_batch_id(batch_id)
        self.geometry = geometry
        self.match_cfg = match_cfg
        self.max_history = max_history
        self.auto_commit_merge_events = auto_commit_merge_events
        self.uf = UnionFind()
        self.local_tracks: dict[tuple[str, int], LocalTrack] = {}
        # root global_id -> local track keys in that connected component. Avoids
        # scanning the full track history on every consistency guard so online
        # cost scales with the two groups being merged.
        self.group_members: dict[int, set[tuple[str, int]]] = {}
        self.next_global_id = self._load_next_global_id()
        # Deferred global_id rewrite: enqueue during merge_active and flush in
        # flush_rewrites() so each merge is not its own UPDATE + commit.
        self.pending_rewrites: list[tuple[int, int]] = []

    def _load_next_global_id(self) -> int:
        row = self.conn.execute(
            """
            SELECT MAX(global_id)
              FROM trajectory_global_tracks
             WHERE scene_id = ? AND batch_id = ?
            """,
            (self.scene_id, self.batch_id),
        ).fetchone()
        return int(row[0] or 0) + 1

    def _register_global_track(self, global_id: int, source: str) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO trajectory_global_tracks
                (scene_id, batch_id, global_id, created_at, source)
            VALUES (?,?,?,?,?)
            """,
            (self.scene_id, self.batch_id, global_id, utc_now(), source),
        )

    def add_sample(
        self,
        camera_name: str,
        local_track_id: int,
        sample: TrackSample,
    ) -> int:
        key = (camera_name, local_track_id)
        if key not in self.local_tracks:
            global_id = self.next_global_id
            self.next_global_id += 1
            self.uf.make(global_id)
            self._register_global_track(global_id, f"{camera_name}:{local_track_id}")
            self.local_tracks[key] = LocalTrack(
                camera_name=camera_name,
                local_track_id=local_track_id,
                global_id=global_id,
                max_history=self.max_history,
            )
            self.group_members[global_id] = {key}
        track = self.local_tracks[key]
        track.add_sample(sample)
        return self.uf.find(track.global_id)

    def active_tracks(self, now_sec: float) -> list[LocalTrack]:
        timeout = self.match_cfg.active_timeout_sec
        return [
            t for t in self.local_tracks.values()
            if t.samples and now_sec - t.last_seen_time <= timeout
        ]

    def active_global_count(self, now_sec: float) -> int:
        return len({self.uf.find(t.global_id) for t in self.active_tracks(now_sec)})

    def merge_active(self, now_sec: float) -> tuple[int, list[MergeEvent]]:
        active_by_camera: dict[str, list[LocalTrack]] = defaultdict(list)
        for track in self.active_tracks(now_sec):
            if len(track.samples) >= self.match_cfg.min_samples:
                active_by_camera[track.camera_name].append(track)

        comparisons = 0
        events: list[MergeEvent] = []

        # ── Stage 1: same-camera tracklet stitching (temporal reconnect) ──
        if self.match_cfg.stitch_enabled:
            stitch_comparisons, stitch_events = self._stitch_same_camera(now_sec)
            comparisons += stitch_comparisons
            events.extend(stitch_events)

        # ── Stage 2: cross-camera co-timed track merge ────────────────────
        for camera_a, camera_b in sorted(self.geometry.overlap_pairs):
            tracks_a = active_by_camera.get(camera_a, [])
            tracks_b = active_by_camera.get(camera_b, [])
            for track_a in tracks_a:
                for track_b in tracks_b:
                    if self.uf.find(track_a.global_id) == self.uf.find(track_b.global_id):
                        continue
                    if not self._overlap_gate(track_a, track_b):
                        continue
                    comparisons += 1
                    scores = trajectory_similarity(track_a, track_b, self.match_cfg)
                    if scores is None:
                        continue
                    if scores.score < self.match_cfg.score_threshold:
                        continue
                    if not self._union_allowed(track_a, track_b):
                        continue
                    result = self.uf.union_keep_smallest(track_a.global_id, track_b.global_id)
                    if result is None:
                        continue
                    kept, merged = result
                    self._merge_member_groups(kept, merged)
                    self.pending_rewrites.append((kept, merged))
                    event = MergeEvent(
                        merge_kind="cross_camera",
                        kept_global_id=kept,
                        merged_global_id=merged,
                        track_a=track_a,
                        track_b=track_b,
                        scores=scores,
                        comparison_count=comparisons,
                    )
                    self._save_merge_event(event)
                    events.append(event)
        if events and self.auto_commit_merge_events:
            self.conn.commit()  # commit merge-event INSERTs immediately
        return comparisons, events

    def _stitch_same_camera(self, now_sec: float) -> tuple[int, list[MergeEvent]]:
        """Same-camera tracklet linking in a sliding window (Hungarian 1-to-1).

        An old track must outlive the finalize period derived from
        track_buffer × measured sample period, so we do not irreversibly union
        a track that may still recover and grow. A new track must have enough
        samples but may already have ended briefly, so online finalize delay
        does not drop short tracklets. Candidates stay in a bounded time
        window so long-run cost does not grow with full history.
        """
        cfg = self.match_cfg
        comparisons = 0
        events: list[MergeEvent] = []
        by_camera: dict[str, list[LocalTrack]] = defaultdict(list)
        for track in self.local_tracks.values():
            if track.sample_count < cfg.stitch_min_samples:
                continue
            # Keep only recent tracklets that can still take part in linking.
            track_horizon = _track_finalize_sec(track, cfg) + cfg.stitch_max_gap_sec
            if now_sec - track.last_seen_time > track_horizon:
                continue
            by_camera[track.camera_name].append(track)

        for tracks in by_camera.values():
            old_tracks = [
                t for t in tracks
                if now_sec - t.last_seen_time > _track_finalize_sec(t, cfg)
            ]
            new_tracks = [
                t for t in tracks
                if now_sec - t.first_seen_time
                <= _track_finalize_sec(t, cfg) + cfg.stitch_max_gap_sec
            ]
            if not old_tracks or not new_tracks:
                continue

            scores_by_pair: dict[tuple[int, int], MatchScores] = {}
            # One private dummy column per row; unmatched cost equals the threshold boundary.
            unmatched_cost = 1.0 - cfg.stitch_score_threshold
            cost = np.full(
                (len(old_tracks), len(new_tracks) + len(old_tracks)),
                unmatched_cost,
                dtype=np.float64,
            )
            cost[:, :len(new_tracks)] = 1e6
            for i, old_track in enumerate(old_tracks):
                for j, new_track in enumerate(new_tracks):
                    if old_track is new_track:
                        continue
                    if self.uf.find(old_track.global_id) == self.uf.find(new_track.global_id):
                        continue
                    gap = new_track.first_seen_time - old_track.last_seen_time
                    overlap_tolerance = _track_overlap_tolerance_sec(
                        old_track, new_track, cfg
                    )
                    if gap < -overlap_tolerance or gap > cfg.stitch_max_gap_sec:
                        continue
                    comparisons += 1
                    scores = stitch_compatibility(old_track, new_track, cfg)
                    if scores is None or scores.score < cfg.stitch_score_threshold:
                        continue
                    if not self._union_allowed(old_track, new_track):
                        continue
                    scores_by_pair[(i, j)] = scores
                    cost[i, j] = 1.0 - scores.score

            for i, j in _linear_sum_assignment(cost):
                if j >= len(new_tracks) or (i, j) not in scores_by_pair:
                    continue
                if not _assignment_pair_is_confident(i, j, scores_by_pair, cfg):
                    continue
                old_track = old_tracks[i]
                new_track = new_tracks[j]
                # An earlier camera's union may have changed the root; re-check before commit.
                if self.uf.find(old_track.global_id) == self.uf.find(new_track.global_id):
                    continue
                if not self._union_allowed(old_track, new_track):
                    continue
                result = self.uf.union_keep_smallest(
                    old_track.global_id, new_track.global_id
                )
                if result is None:
                    continue
                kept, merged = result
                self._merge_member_groups(kept, merged)
                self.pending_rewrites.append((kept, merged))
                event = MergeEvent(
                    merge_kind="same_camera_link",
                    kept_global_id=kept,
                    merged_global_id=merged,
                    track_a=old_track,
                    track_b=new_track,
                    scores=scores_by_pair[(i, j)],
                    comparison_count=comparisons,
                )
                self._save_merge_event(event)
                events.append(event)
        return comparisons, events

    def _merge_member_groups(self, kept: int, merged: int) -> None:
        kept_members = self.group_members.setdefault(kept, set())
        kept_members.update(self.group_members.pop(merged, set()))

    def _union_allowed(
        self,
        track_a: LocalTrack,
        track_b: LocalTrack,
    ) -> bool:
        """Same-camera spatial exclusion guard: check co-timed position conflict before merging two global groups.

        One physical vehicle cannot occupy two well-separated BEV positions at
        the same time, but the tracker may emit time-overlapping duplicate
        tracklets. Reject a union only when there is repeated co-timed spatial
        conflict evidence, not because of interval overlap or ID activity.
        """
        root_a = self.uf.find(track_a.global_id)
        root_b = self.uf.find(track_b.global_id)
        members_a: dict[str, list[LocalTrack]] = defaultdict(list)
        members_b: list[LocalTrack] = []
        for key in self.group_members.get(root_a, ()):
            track = self.local_tracks[key]
            members_a[track.camera_name].append(track)
        for key in self.group_members.get(root_b, ()):
            members_b.append(self.local_tracks[key])
        for tb in members_b:
            for ta in members_a.get(tb.camera_name, ()):
                if _same_camera_tracks_conflict(ta, tb, self.match_cfg):
                    return False
        return True

    def flush_rewrites(self) -> int:
        """Apply deferred global_id UPDATE statements in bulk.

        The caller commits after this returns. Deferral avoids UPDATE + commit
        on every merge during merge_active; rewrites flush together on the
        stats period (default every 30 frames).
        """
        if not self.pending_rewrites:
            return 0
        count = len(self.pending_rewrites)
        for kept, merged in self.pending_rewrites:
            self.conn.execute(
                """
                UPDATE trajectory_observations
                   SET global_id = ?
                 WHERE scene_id = ? AND batch_id = ? AND global_id = ?
                """,
                (kept, self.scene_id, self.batch_id, merged),
            )
        self.pending_rewrites.clear()
        return count

    def _overlap_gate(self, track_a: LocalTrack, track_b: LocalTrack) -> bool:
        # The vehicle may already have left the overlap when we compare; check
        # whether any historical sample from either track ever fell in this
        # camera pair's overlap, not just the midpoint of the two latest points.
        cam_a = track_a.camera_name
        cam_b = track_b.camera_name
        in_overlap = any(
            self.geometry.point_in_pair_overlap(cam_a, cam_b, s.world_xy)
            for s in list(track_a.samples) + list(track_b.samples)
        )
        if not in_overlap:
            return False
        # Secondary gate: world distance of the two latest points.
        latest_dist = math.dist(track_a.samples[-1].world_xy, track_b.samples[-1].world_xy)
        return latest_dist <= self.match_cfg.max_distance_m

    def _save_merge_event(self, event: MergeEvent) -> None:
        self.conn.execute(
            """
            INSERT INTO trajectory_merges
                (scene_id, batch_id, merge_kind, kept_global_id, merged_global_id,
                 camera_a, local_track_id_a, camera_b, local_track_id_b,
                 score, time_score, distance_score, shape_score,
                 mean_distance_m, time_gap_sec, comparison_count, decided_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                self.scene_id,
                self.batch_id,
                event.merge_kind,
                event.kept_global_id,
                event.merged_global_id,
                event.track_a.camera_name,
                event.track_a.local_track_id,
                event.track_b.camera_name,
                event.track_b.local_track_id,
                event.scores.score,
                event.scores.time_score,
                event.scores.distance_score,
                event.scores.shape_score,
                event.scores.mean_distance_m,
                event.scores.time_gap_sec,
                event.comparison_count,
                utc_now(),
            ),
        )

    def grouped_recent_samples(
        self,
        now_sec: float,
        history_sec: float,
        max_points: int = 0,
    ) -> dict[int, list[TrackSample]]:
        """Return samples grouped by global_id and sorted by time.

        Args:
            max_points: if > 0, keep only the most recent max_points samples
                per global ID to cut draw_bev_overlay sort and polyline cost
                on large history windows.
        """
        groups: dict[int, list[TrackSample]] = defaultdict(list)
        for track in self.local_tracks.values():
            root = self.uf.find(track.global_id)
            for sample in track.samples:
                if now_sec - sample.timestamp_sec <= history_sec:
                    groups[root].append(sample)
        for gid, samples in list(groups.items()):
            # A merged global ID collects samples from several cameras; insert
            # order is not necessarily chronological, so still sort.
            samples.sort(key=lambda s: s.timestamp_sec)
            if max_points > 0 and len(samples) > max_points:
                groups[gid] = samples[-max_points:]
        return groups


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_trajectory_db(db_path: str, batch_id: str) -> sqlite3.Connection:
    conn = init_calibration_db(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_TRACK_DB_SCHEMA)
    merge_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(trajectory_merges)").fetchall()
    }
    if "merge_kind" not in merge_columns:
        conn.execute(
            "ALTER TABLE trajectory_merges "
            "ADD COLUMN merge_kind TEXT NOT NULL DEFAULT 'cross_camera'"
        )
    initialize_trajectory_batch_schema(conn)
    validate_batch_id(batch_id)
    return conn


def parse_source(raw: str) -> str | int:
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    return raw


def parse_sources(items: list[str] | None) -> dict[str, str | int]:
    result: dict[str, str | int] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError("--source format must be NAME=PATH_OR_INDEX")
        name, raw_source = item.split("=", 1)
        result[name.strip()] = parse_source(raw_source)
    return result


_VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}


def _is_video_file(source: str | int) -> bool:
    if isinstance(source, int):
        return False
    return Path(source).suffix.lower() in _VIDEO_SUFFIXES


def resolve_tracking_sources(
    name: str,
    cam_cfg: dict[str, Any],
    overrides: dict[str, str | int],
) -> tuple[str | int, str]:
    """Resolve the inference source for trajectory fusion / YOLO.

    Returns (infer_source, mode_label).

    Priority
    --------
    1. ``--sub-source NAME=URL`` → sub-stream inference (homography rescaled
       from the ``image`` main-stream resolution)
    2. ``--source NAME=PATH``   → CLI override (MP4 or RTSP)
    3. ``cameras.video``        → **offline MP4**; ignore YAML ``stream`` /
       ``sub_stream`` so we do not accidentally open RTSP
    4. ``cameras.sub_stream``   → live sub-stream (``image`` must be the
       main-stream calibration still)
    5. ``cameras.stream`` / ``source`` → main stream

    Note: with only ``video`` set and no ``sub_stream``, the file is opened
    with **main-stream logic**. If the MP4 resolution is lower than ``image``
    (a sub-stream recording), the program **automatically rescales** the
    homography and ``image_roi`` from ``image`` vs the first video frame
    (same as a live sub-stream).
    """
    sub_key = f"sub:{name}"
    if sub_key in overrides:
        return overrides[sub_key], "sub-stream(--sub-source)"
    if name in overrides:
        src = overrides[name]
        if _is_video_file(src):
            return src, "offline-video(--source)"
        return src, "cli(--source)"
    video = cam_cfg.get("video")
    if video:
        return parse_source(str(video)), "offline-video(video)"
    sub = cam_cfg.get("sub_stream")
    stream = cam_cfg.get("stream", cam_cfg.get("source"))
    if sub is not None and stream is not None:
        return parse_source(str(sub)), "sub-stream(sub_stream)"
    if stream is not None:
        return parse_source(str(stream)) if isinstance(stream, str) else stream, "main-stream(stream)"
    if cam_cfg.get("source") is not None:
        raw = cam_cfg["source"]
        return parse_source(str(raw)) if isinstance(raw, str) else raw, "source"
    raise ValueError(
        f"camera '{name}': no inference input configured. "
        "Set video (offline), stream/sub_stream (live), or --source / --sub-source."
    )


_CLAHE_INSTANCES: dict[tuple[float, int], cv.CLAHE] = {}


def apply_clahe_bgr(
    frame: np.ndarray,
    clip_limit: float,
    tile_grid: int = 8,
) -> np.ndarray:
    """Apply CLAHE on the LAB luminance channel for YOLO input only; overlays/saves still use the original frame."""
    key = (clip_limit, tile_grid)
    clahe = _CLAHE_INSTANCES.get(key)
    if clahe is None:
        clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
        _CLAHE_INSTANCES[key] = clahe
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    l = clahe.apply(l)
    return cv.cvtColor(cv.merge([l, a, b]), cv.COLOR_LAB2BGR)


def parse_classes(
    raw: str | None,
    yolo_names: dict[int, str] | None = None,
) -> list[int] | None:
    """Parse --classes: COCO numeric ids, model class names (e.g. potted plant), or all/none."""
    if raw is None:
        return VEHICLE_COCO_CLASSES
    if raw.strip().lower() in {"", "all", "none"}:
        return None
    name_to_id: dict[str, int] = {}
    if yolo_names:
        for cid, name in yolo_names.items():
            name_to_id[str(name).strip().lower()] = int(cid)
    ids: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            ids.append(int(token))
            continue
        key = token.lower()
        if key in name_to_id:
            ids.append(name_to_id[key])
            continue
        raise ValueError(
            f"unknown class '{token}'. Use a numeric class_id, a YOLO class name "
            f"(quote names that contain spaces), or --classes all; example model classes: "
            f"{', '.join(list(yolo_names.values())[:8]) if yolo_names else '…'}"
        )
    return ids


def load_yolo_model(model_path: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return YOLO(model_path)


def load_yolo_models(model_path: str, camera_names: list[str]) -> dict[str, Any]:
    """Load YOLO weights from disk once and deepcopy for the remaining cameras.

    Each camera needs its own model instance: Ultralytics stores per-camera
    tracker state (track IDs, Kalman filters, …) inside the predictor.
    deepcopy avoids N disk reads and parses while keeping tracker namespaces
    isolated.

    On GPU memory: PyTorch CUDA tensors do not share device memory after
    deepcopy; each instance still occupies VRAM. This mainly speeds startup.
    With >4 streams on one GPU, prefer a light model (yolo11n / yolo12n) to
    control VRAM.
    """
    if not camera_names:
        return {}
    base = load_yolo_model(model_path)
    models: dict[str, Any] = {camera_names[0]: base}
    for name in camera_names[1:]:
        models[name] = copy.deepcopy(base)
    return models


def image_point_to_world(
    homography: np.ndarray,
    image_xy: tuple[float, float],
    scale: float,
) -> tuple[float, float]:
    pt = np.asarray([[[image_xy[0], image_xy[1]]]], dtype=np.float32)
    bev = cv.perspectiveTransform(pt, homography.astype(np.float64)).reshape(2)
    return float(bev[0] / scale), float(bev[1] / scale)


def world_point_to_image(
    homography: np.ndarray,
    world_xy: tuple[float, float],
    scale: float,
) -> tuple[float, float]:
    pt = np.asarray([[[world_xy[0] * scale, world_xy[1] * scale]]], dtype=np.float32)
    img = cv.perspectiveTransform(pt, np.linalg.inv(homography.astype(np.float64))).reshape(2)
    return float(img[0]), float(img[1])


RISK_SCALAR_MODES = (
    "metric_jacobian",
    "pixel_orthogonal",
    "pixel_vertical",
    "grazing_angle",
)


@dataclass(frozen=True)
class RiskScalar:
    """fab1 risk scalar: the single geometric quantity driving forward-anchor gating α and self-supervised confidence κ=1-α.

    ``alpha``/``kappa`` keep the same meaning under every ``risk_scalar_mode``
    (κ=1-α), so scalar definitions can be ablated without changing downstream
    MVC / gating logic.
    """

    alpha: float
    kappa: float
    sensitivity_m: float                 # metric local sensitivity s (NaN outside metric_jacobian)
    meters_per_px_u: float                # local u-direction meters/pixel (metric box width)
    meters_per_px_v: float                # local v-direction meters/pixel (metric box height)
    horizon_signed_distance_px: float     # diagnostic: ground-side signed/clamped pixel distance
    valid_ground_side: bool               # False = sample not on the ground side; geometry invalid
    valid_projection: bool                # False = projection denominator degenerate (|ω|≈0)


def homography_metric_jacobian(
    homography: np.ndarray,
    u: float,
    v: float,
    scale_px_per_meter: float,
) -> np.ndarray:
    """Analytic Jacobian J=d(X_m,Y_m)/d(u_px,v_px) of image(u,v) → BEV metric coords, shape (2,2).

    ``homography`` is image → BEV pixels; divide by ``scale_px_per_meter`` (px/m)
    to get meters. Returns inf when the projection denominator |ω|≈0 (image
    approaching the homography vanishing line); callers must check ``np.isfinite``.
    """
    H = homography.astype(np.float64)
    nx = H[0, 0] * u + H[0, 1] * v + H[0, 2]
    ny = H[1, 0] * u + H[1, 1] * v + H[1, 2]
    nw = H[2, 0] * u + H[2, 1] * v + H[2, 2]
    if abs(nw) < 1e-9:
        return np.full((2, 2), np.inf, dtype=np.float64)
    inv_nw2 = 1.0 / (nw * nw)
    dXdu = (H[0, 0] * nw - nx * H[2, 0]) * inv_nw2
    dXdv = (H[0, 1] * nw - nx * H[2, 1]) * inv_nw2
    dYdu = (H[1, 0] * nw - ny * H[2, 0]) * inv_nw2
    dYdv = (H[1, 1] * nw - ny * H[2, 1]) * inv_nw2
    scale = max(float(scale_px_per_meter), 1e-9)
    return np.array([[dXdu, dXdv], [dYdu, dYdv]], dtype=np.float64) / scale


def _horizon_ground_clamped_distance(
    homography: np.ndarray,
    cx: float,
    cy: float,
    frame_wh: tuple[int, int],
) -> tuple[float, bool]:
    """Return (d_ground, valid_ground_side): ground-side signed orthogonal pixel distance; clamp non-ground to 0.

    Ground side is oriented automatically from the sign of a probe at the
    bottom-edge midpoint (independent of camera pitch).
    """
    h20, h21, h22 = (float(homography[2, 0]), float(homography[2, 1]), float(homography[2, 2]))
    norm = math.hypot(h20, h21)
    if norm < 1e-9:
        # Degenerate H: no stable vanishing line; conservatively treat as valid ground side.
        return max(float(cy), 0.0), True
    signed = (h20 * cx + h21 * cy + h22) / norm
    fw, fh = frame_wh
    ground_probe_x = max(float(fw), 1.0) * 0.5
    ground_probe_y = max(float(fh) - 1.0, 0.0)
    ground_probe = (h20 * ground_probe_x + h21 * ground_probe_y + h22) / norm
    ground_sign = 1.0 if ground_probe >= 0.0 else -1.0
    oriented = ground_sign * signed
    return max(oriented, 0.0), oriented > 0.0


def metric_sensitivity_from_homography(
    homography: np.ndarray,
    cx: float,
    cy: float,
    scale_px_per_meter: float,
    frame_wh: tuple[int, int],
    *,
    s0_m: float,
    sensitivity_clip_m: float | None = None,
) -> RiskScalar:
    """fab1 default risk scalar: s = ||J_px @ diag(W_f,H_f)||_F, α=s/(s+s0), κ=s0/(s+s0).

    The risk sample is fixed at the box center ``(cx, cy)`` (does not slide
    with the anchor). Off the ground side or with a degenerate projection,
    clamp α=1, κ=0 and set ``valid_ground_side``/``valid_projection=False``;
    callers should then exclude the observation from MVC teacher selection
    (geometry-fallback inference may still run).

    The paper and Appendix A use s = ||J_px diag(W_f, H_f)||_F
    (per-axis normalized). This is that formula:
    J_norm = J_px diag(W_f, H_f).
    """
    fw, fh = frame_wh
    d_ground, valid_ground_side = _horizon_ground_clamped_distance(homography, cx, cy, frame_wh)
    J = homography_metric_jacobian(homography, cx, cy, scale_px_per_meter)
    valid_projection = bool(np.all(np.isfinite(J)))
    if not valid_projection or not valid_ground_side:
        return RiskScalar(
            alpha=1.0,
            kappa=0.0,
            sensitivity_m=float("inf"),
            meters_per_px_u=float(np.hypot(J[0, 0], J[1, 0])) if valid_projection else float("inf"),
            meters_per_px_v=float(np.hypot(J[0, 1], J[1, 1])) if valid_projection else float("inf"),
            horizon_signed_distance_px=float(d_ground),
            valid_ground_side=valid_ground_side,
            valid_projection=valid_projection,
        )
    J_norm = J @ np.diag([float(fw), float(fh)])
    sensitivity = float(np.linalg.norm(J_norm, ord="fro"))
    if sensitivity_clip_m is not None:
        sensitivity = min(sensitivity, float(sensitivity_clip_m))
    s0 = max(float(s0_m), 1e-6)
    alpha = sensitivity / (sensitivity + s0)
    kappa = s0 / (sensitivity + s0)
    return RiskScalar(
        alpha=float(alpha),
        kappa=float(kappa),
        sensitivity_m=sensitivity,
        meters_per_px_u=float(np.hypot(J[0, 0], J[1, 0])),
        meters_per_px_v=float(np.hypot(J[0, 1], J[1, 1])),
        horizon_signed_distance_px=float(d_ground),
        valid_ground_side=True,
        valid_projection=True,
    )


def pixel_distance_risk_scalar(
    homography: np.ndarray,
    cx: float,
    cy: float,
    d0_frac: float,
    frame_wh: tuple[int, int],
    *,
    mode: str = "pixel_orthogonal",
) -> RiskScalar:
    """Pixel-distance ablation baseline: α=d0/(d+d0), d is ground-side signed (orthogonal or vertical) pixel distance to the horizon.

    Keeps the cla1 meaning (``d0_frac`` is a fraction of frame height, converted
    internally to absolute pixels). Used only for the VI-E four-way risk-scalar
    ablation; not metrically comparable across cameras, so the main experiment
    leaves it off by default.
    """
    h20, h21, h22 = (float(homography[2, 0]), float(homography[2, 1]), float(homography[2, 2]))
    fw, fh = frame_wh
    d0 = max(d0_frac * fh, 1e-3)
    norm = math.hypot(h20, h21)
    if norm < 1e-9 or mode != "pixel_vertical" or abs(h21) < 1e-9:
        d_ground, valid_ground_side = _horizon_ground_clamped_distance(homography, cx, cy, frame_wh)
    else:
        # pixel_vertical: distance to the horizon along the vertical axis (older, simpler baseline).
        v_horizon = -(h20 * cx + h22) / h21
        raw = cy - v_horizon
        ground_probe_x = max(float(fw), 1.0) * 0.5
        ground_probe_y = max(float(fh) - 1.0, 0.0)
        v_horizon_probe = -(h20 * ground_probe_x + h22) / h21
        ground_probe_val = ground_probe_y - v_horizon_probe
        ground_sign = 1.0 if ground_probe_val >= 0.0 else -1.0
        oriented = ground_sign * raw
        d_ground, valid_ground_side = max(oriented, 0.0), oriented > 0.0
    alpha = float(d0 / (d_ground + d0))
    kappa = float(1.0 - alpha)
    return RiskScalar(
        alpha=alpha,
        kappa=kappa,
        sensitivity_m=float("nan"),
        meters_per_px_u=float("nan"),
        meters_per_px_v=float("nan"),
        horizon_signed_distance_px=float(d_ground),
        valid_ground_side=valid_ground_side,
        valid_projection=True,
    )


def compute_risk_scalar(
    xyxy: np.ndarray,
    homography: np.ndarray,
    frame_wh: tuple[int, int],
    *,
    risk_scalar_mode: str = "metric_jacobian",
    scale: float | None = None,
    s0_m: float = 60.0,
    grazing_d0: float = 50.0 / 1080.0,
    sensitivity_clip_m: float | None = None,
) -> RiskScalar:
    """Unified risk-scalar entry: dispatch ``risk_scalar_mode`` to metric Jacobian or pixel-distance implementations.

    The risk sample is fixed at the box center, matching fab1 paper IV-B
    (does not slide with the anchor).
    """
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    if risk_scalar_mode == "metric_jacobian":
        if scale is None:
            raise ValueError("risk_scalar_mode='metric_jacobian' requires scale (px/m)")
        return metric_sensitivity_from_homography(
            homography, cx, cy, scale, frame_wh,
            s0_m=s0_m, sensitivity_clip_m=sensitivity_clip_m,
        )
    if risk_scalar_mode in ("pixel_orthogonal", "pixel_vertical"):
        return pixel_distance_risk_scalar(
            homography, cx, cy, grazing_d0, frame_wh, mode=risk_scalar_mode,
        )
    if risk_scalar_mode == "grazing_angle":
        raise NotImplementedError(
            "risk_scalar_mode='grazing_angle' (focal-length self-estimated grazing angle) is not implemented; ablation placeholder only"
        )
    raise ValueError(
        f"unknown risk_scalar_mode: {risk_scalar_mode!r}; choose from {RISK_SCALAR_MODES}"
    )


def _clamp_anchor_aspect_ratio(
    rho: float,
    aspect_ratio_min: float = DEFAULT_ANCHOR_ASPECT_RATIO_MIN,
    aspect_ratio_max: float = DEFAULT_ANCHOR_ASPECT_RATIO_MAX,
) -> float:
    """Clamp ``rho=h/w`` before it is used as a shape prior."""
    lo = float(aspect_ratio_min)
    hi = float(aspect_ratio_max)
    if lo <= 0 or hi <= 0 or lo > hi:
        raise ValueError("aspect_ratio_min/max must satisfy 0 < min <= max")
    return float(np.clip(float(rho), lo, hi))


def clamped_width_height_ratio_from_bbox(
    xyxy: np.ndarray,
    *,
    aspect_ratio_min: float = DEFAULT_ANCHOR_ASPECT_RATIO_MIN,
    aspect_ratio_max: float = DEFAULT_ANCHOR_ASPECT_RATIO_MAX,
) -> float:
    """Return the ``w/h`` feature after clamping ``h/w``, suppressing extreme box ratios from occlusion/false detections."""
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    rho = _clamp_anchor_aspect_ratio(
        h / w,
        aspect_ratio_min=aspect_ratio_min,
        aspect_ratio_max=aspect_ratio_max,
    )
    return float(1.0 / max(rho, 1e-6))


def slimness_alpha_from_bbox(
    xyxy: np.ndarray,
    *,
    aspect_ratio_min: float = DEFAULT_ANCHOR_ASPECT_RATIO_MIN,
    aspect_ratio_max: float = DEFAULT_ANCHOR_ASPECT_RATIO_MAX,
) -> float:
    """Shape offset between bottom_center and center from clamped ``rho=h/w``."""
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    rho = _clamp_anchor_aspect_ratio(
        h / w,
        aspect_ratio_min=aspect_ratio_min,
        aspect_ratio_max=aspect_ratio_max,
    )
    slimness_sq = rho ** 2
    return float(slimness_sq / (slimness_sq + 1.0))


def gated_anchor_beta(
    xyxy: np.ndarray,
    risk: RiskScalar,
    *,
    use_slimness: bool,
    shape_eta: float = DEFAULT_ANCHOR_SHAPE_ETA,
    aspect_ratio_min: float = DEFAULT_ANCHOR_ASPECT_RATIO_MIN,
    aspect_ratio_max: float = DEFAULT_ANCHOR_ASPECT_RATIO_MAX,
) -> tuple[float, float]:
    """Multiplicative Gating offsetted: return (anchor_beta, gate).

    ``risk_alpha`` (=``risk.alpha``) and ``anchor_beta`` (this function's
    return value, used for image-vertical interpolation) are strictly
    distinct: κ=1-risk_alpha is for confidence/gating and must not be
    inverted from anchor_beta. gate = 4·risk_alpha·(1-risk_alpha): when
    risk_alpha≈0 or 1 the risk scalar dominates; in the mid band
    ``shape_eta`` injects the clamped slimness shape term so detection-box
    ratio jitter does not dominate the anchor.
    """
    if not 0.0 <= float(shape_eta) <= 1.0:
        raise ValueError("shape_eta must be in [0, 1]")
    risk_alpha = risk.alpha
    gate = 4.0 * risk_alpha * (1.0 - risk_alpha)
    if not use_slimness:
        return risk_alpha, gate
    shape_alpha = slimness_alpha_from_bbox(
        xyxy,
        aspect_ratio_min=aspect_ratio_min,
        aspect_ratio_max=aspect_ratio_max,
    )
    return float(risk_alpha + float(shape_eta) * gate * (shape_alpha - risk_alpha)), gate


def local_box_extent_meters(
    xyxy: np.ndarray,
    homography: np.ndarray,
    scale_px_per_meter: float,
    *,
    anchor_xy: tuple[float, float] | None = None,
    extent_clip_m: float | None = None,
) -> tuple[float, float]:
    """Metric box extent: ``w_m=w_px·‖J e_u‖``, ``h_m=h_px·‖J e_v‖``, evaluated at the anchor (fab1 IV-C).

    ``extent_clip_m`` clips only this box feature and does not affect
    ``risk_alpha/kappa`` (those are evaluated at box center, decoupled from
    this function).
    """
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    w_px = max(x2 - x1, 1e-6)
    h_px = max(y2 - y1, 1e-6)
    if anchor_xy is not None:
        eu, ev = float(anchor_xy[0]), float(anchor_xy[1])
    else:
        eu, ev = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    J = homography_metric_jacobian(homography, eu, ev, scale_px_per_meter)
    if not np.all(np.isfinite(J)):
        meters_per_px_u = meters_per_px_v = float("inf")
    else:
        meters_per_px_u = float(np.hypot(J[0, 0], J[1, 0]))
        meters_per_px_v = float(np.hypot(J[0, 1], J[1, 1]))
    w_m = w_px * meters_per_px_u
    h_m = h_px * meters_per_px_v
    if extent_clip_m is not None:
        w_m = min(w_m, float(extent_clip_m))
        h_m = min(h_m, float(extent_clip_m))
    return float(w_m), float(h_m)


def box_anchor(
    xyxy: np.ndarray,
    mode: str,
    *,
    homography: np.ndarray | None = None,
    frame_wh: tuple[int, int] | None = None,
    scale: float | None = None,
    risk_scalar_mode: str = "metric_jacobian",
    s0_m: float = 60.0,
    grazing_d0: float = 50.0 / 1080.0,
    sensitivity_clip_m: float | None = None,
    shape_eta: float = DEFAULT_ANCHOR_SHAPE_ETA,
    aspect_ratio_min: float = DEFAULT_ANCHOR_ASPECT_RATIO_MIN,
    aspect_ratio_max: float = DEFAULT_ANCHOR_ASPECT_RATIO_MAX,
    risk: RiskScalar | None = None,
) -> tuple[float, float]:
    """Pick an anchor on the vertical segment from box bottom_center to center, then project to BEV/world.

    ``risk``: optional precomputed ``RiskScalar`` at box center; reuse it
    instead of recomputing the Jacobian / pixel distance (used inside
    ``compute_geometry_base``). May be omitted when calling this function
    independently.
    """
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    if mode == "center":
        return cx, cy
    if mode == "bottom_center":
        return cx, y2
    if mode == "slimness_squared_offsetted":
        alpha = slimness_alpha_from_bbox(
            xyxy,
            aspect_ratio_min=aspect_ratio_min,
            aspect_ratio_max=aspect_ratio_max,
        )
        return cx, cy + alpha * (y2 - cy)
    if mode in ("grazing_angle_offsetted", "multiplicative_gating_offsetted"):
        if homography is None:
            raise ValueError(f"mode {mode} requires homography")
        if frame_wh is None:
            raise ValueError(f"mode {mode} requires frame_wh")
        r = risk if risk is not None else compute_risk_scalar(
            xyxy, homography, frame_wh,
            risk_scalar_mode=risk_scalar_mode, scale=scale, s0_m=s0_m,
            grazing_d0=grazing_d0, sensitivity_clip_m=sensitivity_clip_m,
        )
        if mode == "grazing_angle_offsetted":
            return cx, cy + r.alpha * (y2 - cy)
        beta, _gate = gated_anchor_beta(
            xyxy,
            r,
            use_slimness=True,
            shape_eta=shape_eta,
            aspect_ratio_min=aspect_ratio_min,
            aspect_ratio_max=aspect_ratio_max,
        )
        return cx, cy + beta * (y2 - cy)
    raise ValueError(f"unknown anchor mode: {mode}")


def is_bbox_clipped(
    xyxy: np.ndarray,
    frame_wh: tuple[int, int],
    margin_px: int = 3,
    margin_ratio: float = 0.01,
) -> bool:
    """Whether the detection box is flush with the frame edge (ground-contact may be off-screen; world coords untrustworthy).

    On side-mounted cameras the box top (y1 ≤ margin) is the most dangerous:
    the contact point is above the horizon and the perspective singularity
    can make bottom_center → world error several meters. Bottom/left/right
    truncation is usually smaller but still worth filtering.
    """
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    w, h = frame_wh
    mx = max(float(margin_px), float(w) * margin_ratio)
    my = max(float(margin_px), float(h) * margin_ratio)
    return (y1 <= my              # top truncation (most dangerous)
            or x1 <= mx           # left truncation
            or x2 >= w - mx       # right truncation
            or y2 >= h - my)      # bottom truncation


def unique_times_and_points(samples: list[TrackSample]) -> tuple[np.ndarray, np.ndarray]:
    """Keep only clipped=False world coordinates; drop edge-truncated frames."""
    times: list[float] = []
    points: list[tuple[float, float]] = []
    last_t: float | None = None
    for sample in sorted(samples, key=lambda s: s.timestamp_sec):
        if sample.clipped:
            continue  # contact point off-screen; world coords untrustworthy; skip matching
        t = float(sample.timestamp_sec)
        if last_t is not None and abs(t - last_t) < 1e-6:
            points[-1] = sample.world_xy
            continue
        times.append(t)
        points.append(sample.world_xy)
        last_t = t
    return np.asarray(times, dtype=np.float64), np.asarray(points, dtype=np.float64)


def interpolate_points(samples: list[TrackSample], times: np.ndarray) -> np.ndarray | None:
    source_t, source_p = unique_times_and_points(samples)
    if len(source_t) < 2:
        return None
    xs = np.interp(times, source_t, source_p[:, 0])
    ys = np.interp(times, source_t, source_p[:, 1])
    return np.stack([xs, ys], axis=1)


def resample_by_index(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) == count:
        return points.astype(np.float64)
    if len(points) == 1:
        return np.repeat(points.astype(np.float64), count, axis=0)
    idx = np.linspace(0.0, float(len(points) - 1), count)
    base = np.arange(len(points), dtype=np.float64)
    xs = np.interp(idx, base, points[:, 0])
    ys = np.interp(idx, base, points[:, 1])
    return np.stack([xs, ys], axis=1)


def normalize_shape(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    path_len = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    scale = max(path_len, float(np.linalg.norm(np.ptp(points, axis=0))), 1e-6)
    return centered / scale


def direction_similarity(points_a: np.ndarray, points_b: np.ndarray) -> float:
    va = points_a[-1] - points_a[0]
    vb = points_b[-1] - points_b[0]
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-6 or nb < 1e-6:
        return 1.0
    cos = float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))
    return (cos + 1.0) * 0.5


def trajectory_similarity(
    track_a: LocalTrack,
    track_b: LocalTrack,
    cfg: MatchConfig,
) -> MatchScores | None:
    samples_a = list(track_a.samples)
    samples_b = list(track_b.samples)
    if len(samples_a) < cfg.min_samples or len(samples_b) < cfg.min_samples:
        return None

    last_a = samples_a[-1]
    last_b = samples_b[-1]
    time_gap = abs(last_a.timestamp_sec - last_b.timestamp_sec)
    if time_gap > cfg.max_time_gap_sec:
        return None

    start = max(samples_a[0].timestamp_sec, samples_b[0].timestamp_sec)
    end = min(samples_a[-1].timestamp_sec, samples_b[-1].timestamp_sec)
    use_time_overlap = end > start and (end - start) >= 0.15

    if use_time_overlap:
        times = np.linspace(start, end, cfg.sample_count)
        points_a = interpolate_points(samples_a, times)
        points_b = interpolate_points(samples_b, times)
        if points_a is None or points_b is None:
            return None
    else:
        pts_a = np.asarray([s.world_xy for s in samples_a[-cfg.sample_count:]], dtype=np.float64)
        pts_b = np.asarray([s.world_xy for s in samples_b[-cfg.sample_count:]], dtype=np.float64)
        points_a = resample_by_index(pts_a, cfg.sample_count)
        points_b = resample_by_index(pts_b, cfg.sample_count)

    distances = np.linalg.norm(points_a - points_b, axis=1)
    mean_distance = float(np.mean(distances))
    latest_distance = math.dist(last_a.world_xy, last_b.world_xy)
    # Both the historical mean and latest point must be close. Using min(...)
    # lets a brief endpoint encounter merge tracks whose overall P_geo distance
    # is too large, which then pollutes MVC residual training pairs.
    gated_distance = max(mean_distance, latest_distance)
    if gated_distance > cfg.max_distance_m:
        return None

    shape_rms = float(
        np.sqrt(
            np.mean(
                np.sum((normalize_shape(points_a) - normalize_shape(points_b)) ** 2, axis=1)
            )
        )
    )
    time_score = math.exp(-time_gap / max(cfg.time_sigma, 1e-6))
    distance_score = math.exp(-gated_distance / max(cfg.distance_sigma, 1e-6))
    shape_score = math.exp(-shape_rms / max(cfg.shape_sigma, 1e-6))
    shape_score *= direction_similarity(points_a, points_b)

    total_weight = cfg.time_weight + cfg.distance_weight + cfg.shape_weight
    score = (
        cfg.time_weight * time_score
        + cfg.distance_weight * distance_score
        + cfg.shape_weight * shape_score
    ) / max(total_weight, 1e-6)
    return MatchScores(
        score=float(score),
        time_score=float(time_score),
        distance_score=float(distance_score),
        shape_score=float(shape_score),
        mean_distance_m=float(mean_distance),
        time_gap_sec=float(time_gap),
    )


@dataclass(frozen=True)
class _TrackMotionState:
    position: np.ndarray
    velocity: np.ndarray | None
    timestamp_sec: float
    position_sigma_m: float
    velocity_sigma_mps: float


def _fit_track_motion_state(
    samples: list[TrackSample],
    *,
    at_head: bool,
    cfg: MatchConfig,
) -> _TrackMotionState | None:
    times, points = unique_times_and_points(samples)
    if len(times) == 0:
        return None
    k = min(8, len(times))
    times = times[:k] if at_head else times[-k:]
    points = points[:k] if at_head else points[-k:]
    endpoint = 0 if at_head else -1
    timestamp = float(times[endpoint])
    if len(times) < 2 or float(times[-1] - times[0]) < 1e-3:
        return _TrackMotionState(
            position=points[endpoint],
            velocity=None,
            timestamp_sec=timestamp,
            position_sigma_m=cfg.stitch_uncertainty_floor_m,
            velocity_sigma_mps=0.0,
        )

    dt = times - timestamp
    design = np.column_stack((dt, np.ones_like(dt)))
    coeff = np.linalg.lstsq(design, points, rcond=None)[0]
    fitted = design @ coeff
    residual = float(np.sqrt(np.mean(np.sum((points - fitted) ** 2, axis=1))))
    position_sigma = max(cfg.stitch_uncertainty_floor_m, residual)
    centered = times - float(np.mean(times))
    time_spread = float(np.sqrt(np.sum(centered**2)))
    velocity_sigma = position_sigma / max(time_spread, 1e-3)
    return _TrackMotionState(
        position=np.asarray(coeff[1], dtype=np.float64),
        velocity=np.asarray(coeff[0], dtype=np.float64),
        timestamp_sec=timestamp,
        position_sigma_m=position_sigma,
        velocity_sigma_mps=velocity_sigma,
    )


def _track_tail_state(
    track: LocalTrack,
    cfg: MatchConfig,
) -> _TrackMotionState | None:
    return _fit_track_motion_state(
        list(track.samples),
        at_head=False,
        cfg=cfg,
    )


def _track_head_state(
    track: LocalTrack,
    cfg: MatchConfig,
) -> _TrackMotionState | None:
    """Head samples are kept separately so a long track does not lose start state from the bounded deque."""
    src = track.head_samples if track.head_samples else list(track.samples)
    return _fit_track_motion_state(src, at_head=True, cfg=cfg)


def _linear_sum_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    """Rectangular Hungarian solver in pure NumPy/Python (rows must not exceed columns).

    The linking matrix adds a dummy column per old tracklet, so rows<=cols
    holds naturally; this avoids pulling a large SciPy runtime onto edge
    deployments.
    """
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("assignment cost must be a 2-D matrix")
    n, m = matrix.shape
    if n == 0:
        return []
    if n > m:
        raise ValueError("Hungarian solver requires rows <= columns")

    # 1-based implementation of the shortest augmenting path algorithm.
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(m + 1, dtype=np.float64)
    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf, dtype=np.float64)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = matrix[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment: list[tuple[int, int]] = []
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment.append((int(p[j] - 1), j - 1))
    return assignment


def _assignment_pair_is_confident(
    row: int,
    col: int,
    scores_by_pair: dict[tuple[int, int], MatchScores],
    cfg: MatchConfig,
) -> bool:
    """Require a Hungarian real edge to be mutual-best and to beat the row/column runner-up by a margin."""
    chosen = scores_by_pair.get((row, col))
    if chosen is None:
        return False
    row_alternatives = [
        scores.score
        for (i, j), scores in scores_by_pair.items()
        if i == row and j != col
    ]
    col_alternatives = [
        scores.score
        for (i, j), scores in scores_by_pair.items()
        if j == col and i != row
    ]
    alternatives = row_alternatives + col_alternatives
    if cfg.stitch_require_mutual_best and any(
        score > chosen.score + 1e-12 for score in alternatives
    ):
        return False
    if alternatives and chosen.score - max(alternatives) < cfg.stitch_min_score_margin:
        return False
    return True


def stitch_compatibility(
    old_track: LocalTrack,
    new_track: LocalTrack,
    cfg: MatchConfig,
) -> MatchScores | None:
    """Score for stitching consecutive same-camera tracklets.

    Unlike cross-camera trajectory_similarity, the two tracks abut in time
    (almost no overlap). We compare "the old track's tail, extrapolated by
    velocity for gap seconds, vs the new track's start", not world distance
    at a shared timestamp. Evidence is geometry and kinematics only, with
    no appearance model, so it stays scene-agnostic.
    """
    if old_track.sample_count < cfg.stitch_min_samples:
        return None
    if new_track.sample_count < cfg.stitch_min_samples:
        return None

    head = _track_head_state(new_track, cfg)
    tail = _track_tail_state(old_track, cfg)
    if head is None or tail is None:
        return None

    gap = head.timestamp_sec - tail.timestamp_sec
    overlap_tolerance = _track_overlap_tolerance_sec(old_track, new_track, cfg)
    if gap < -overlap_tolerance or gap > cfg.stitch_max_gap_sec:
        return None

    # Motion extrapolation; if tail velocity cannot be estimated, fall back to
    # a stationary assumption (reconnect only within a small tolerance).
    prediction_horizon = max(gap, 0.0)
    if tail.velocity is not None:
        predicted = tail.position + tail.velocity * prediction_horizon
    else:
        predicted = tail.position
    error = float(np.linalg.norm(predicted - head.position))
    prediction_sigma = math.sqrt(
        tail.position_sigma_m**2
        + head.position_sigma_m**2
        + (tail.velocity_sigma_mps * prediction_horizon) ** 2
        + (0.5 * cfg.stitch_process_noise_mps2 * prediction_horizon**2) ** 2
    )
    minimum_combined_sigma = math.sqrt(2.0) * cfg.stitch_uncertainty_floor_m
    if (
        prediction_sigma / max(minimum_combined_sigma, 1e-6)
        > cfg.stitch_max_uncertainty_ratio
    ):
        # High uncertainty must not enlarge the accept radius; abstain and leave
        # the pair to cross-camera evidence or later offline refinement.
        return None
    normalized_error = error / max(prediction_sigma, 1e-6)

    # Keep the metric gate when legacy config sets base/rate; default is dimensionless prediction error.
    legacy_tolerance = (
        cfg.stitch_base_tolerance_m
        + cfg.stitch_tolerance_rate_mps * prediction_horizon
    )
    if legacy_tolerance > 0:
        if error > legacy_tolerance:
            return None
        position_ratio = error / max(legacy_tolerance, 1e-6)
    else:
        if normalized_error > cfg.stitch_max_normalized_error:
            return None
        position_ratio = normalized_error / max(
            cfg.stitch_max_normalized_error, 1e-6
        )

    if not math.isfinite(position_ratio):
        return None

    # Direction and speed consistency: when both are moving, headings must not clearly conflict.
    direction_cos = 1.0
    speed_score = 1.0
    if tail.velocity is not None and head.velocity is not None:
        speed_old = float(np.linalg.norm(tail.velocity))
        speed_new = float(np.linalg.norm(head.velocity))
        speed_score = math.exp(
            -abs(speed_old - speed_new) / max(cfg.stitch_speed_sigma_mps, 1e-6)
        )
        if speed_old >= cfg.stitch_min_speed_mps and speed_new >= cfg.stitch_min_speed_mps:
            direction_cos = float(
                np.dot(tail.velocity, head.velocity) / max(speed_old * speed_new, 1e-9)
            )
            if direction_cos < cfg.stitch_direction_min_cos:
                return None

    # Normalize error by track-fit residual, speed uncertainty, and process
    # noise; aside from the legacy-compat path, the score does not depend on
    # a per-scene fixed metric tolerance.
    position_score = math.exp(-0.5 * position_ratio**2)
    time_score = math.exp(-max(gap, 0.0) / max(cfg.stitch_max_gap_sec, 1e-6))
    direction_score = (direction_cos + 1.0) * 0.5
    total_score = (
        0.55 * position_score
        + 0.15 * time_score
        + 0.20 * direction_score
        + 0.10 * speed_score
    )
    return MatchScores(
        score=float(total_score),
        time_score=float(time_score),
        distance_score=float(position_score),
        shape_score=float(0.67 * direction_score + 0.33 * speed_score),
        mean_distance_m=error,
        time_gap_sec=float(gap),
    )


def _calib_ref_image(cam: CameraRuntime) -> np.ndarray:
    """Return the reference image used for canvas bounds (main-stream resolution).

    Canvas bounds must match bev_stitch.py, so always use the main-stream
    (calibration-resolution) image size, not the infer sub-stream frame.
    Only the shape matters; a zero array is enough.
    """
    if cam.calib_wh is not None:
        w, h = cam.calib_wh
        return np.empty((h, w, 3), dtype=np.uint8)
    # No multi-stream info: use first_frame (main stream) directly
    if cam.first_frame is None:
        raise RuntimeError(f"camera '{cam.name}': no first frame; cannot compute BEV canvas bounds")
    return cam.first_frame


# ── Offline recompute merge (--remerge mode) ────────────────────────────────


def _remerge_build_cameras(
    cfg: dict[str, Any],
    db_path: str,
    scene_id: str,
    batch_id: str,
) -> list[CameraRuntime]:
    """Build CameraRuntime objects for build_canvas_geometry without opening any video source.

    Resolution sources (so feature F matches training / online tracking):
    1. ``trajectory_camera_config`` infer/calib (written by a prior tracking
       or import run);
    2. if the DB has no calib, probe the YAML ``image`` still;
    3. if still missing, treat infer as calib (do not scale H from a fictional
       1920×1080 down to the infer resolution).
    """
    conn = init_trajectory_db(db_path, batch_id)
    cameras: list[CameraRuntime] = []
    try:
        for cam_cfg in cfg["cameras"]:
            name = cam_cfg["name"]
            cal = load_calibration(conn, name, scene_id)
            if cal is None:
                raise RuntimeError(
                    f"no calibration for camera '{name}' (scene '{scene_id}') in the database; "
                    "run pipeline/multi_camera_bev_stitch.py --save-cal first."
                )
            homography, scale = cal

            row = conn.execute(
                "SELECT infer_w, infer_h, calib_w, calib_h FROM trajectory_camera_config "
                "WHERE scene_id=? AND batch_id=? AND camera_name=?",
                (scene_id, batch_id, name),
            ).fetchone()
            db_infer = db_calib = None
            if row:
                iw, ih, cw, ch = row
                if iw and ih and int(iw) > 0 and int(ih) > 0:
                    db_infer = (int(iw), int(ih))
                if cw and ch and int(cw) > 0 and int(ch) > 0:
                    db_calib = (int(cw), int(ch))

            png_wh = None
            calib_img_path = cam_cfg.get("image")
            if calib_img_path:
                img_path = resolve_project_path(str(calib_img_path))
                if img_path.is_file():
                    probe = cv.imread(str(img_path))
                    if probe is not None:
                        png_wh = (int(probe.shape[1]), int(probe.shape[0]))
                    else:
                        print(
                            f"  [warn] {name}: cannot read calibration image '{calib_img_path}'"
                        )
                elif db_calib is None:
                    print(
                        f"  [note] {name}: calibration image '{calib_img_path}' not found; "
                        "using resolution from DB trajectory_camera_config (does not change stored P_geo)"
                    )

            if db_calib is not None:
                calib_wh = db_calib
                if png_wh is not None and png_wh != db_calib:
                    print(
                        f"  [note] {name}: calibration image {png_wh[0]}×{png_wh[1]} differs from "
                        f"DB calib {db_calib[0]}×{db_calib[1]}; using DB"
                    )
            elif png_wh is not None:
                calib_wh = png_wh
            elif db_infer is not None:
                calib_wh = db_infer
            else:
                calib_wh = (1920, 1080)
                print(
                    f"  [warn] {name}: no DB resolution and no calibration image; falling back to "
                    f"{calib_wh[0]}×{calib_wh[1]}"
                )

            infer_wh = db_infer or png_wh or calib_wh
            h_effective = adapt_homography_to_resolution(homography, calib_wh, infer_wh)
            if infer_wh != calib_wh:
                print(
                    f"  [remerge] {name}: infer resolution {infer_wh[0]}×{infer_wh[1]}"
                    f" (calib {calib_wh[0]}×{calib_wh[1]}); H rescaled"
                )

            cameras.append(
                CameraRuntime(
                    name=name,
                    cfg=cam_cfg,
                    effective_cfg=copy.deepcopy(cam_cfg),
                    source="",
                    cap=cv.VideoCapture(),
                    first_frame=None,
                    homography=h_effective,
                    calib_homography=homography,
                    scale=scale,
                    calib_wh=calib_wh,
                    infer_wh=infer_wh,
                )
            )
    finally:
        conn.close()
    return cameras


@dataclass(frozen=True)
class _RemergeObs:
    obs_id: int
    camera_name: str
    local_track_id: int
    frame_index: int
    timestamp_sec: float
    image_xy: tuple[float, float]
    world_xy: tuple[float, float]
    bbox_xywh: tuple[float, float, float, float]


def _require_observation_columns(
    conn: sqlite3.Connection,
    columns: list[str],
    *,
    context: str,
) -> None:
    existing = {
        row[1]
        for row in conn.execute("PRAGMA table_info(trajectory_observations)").fetchall()
    }
    missing = [col for col in columns if col not in existing]
    if missing:
        raise RuntimeError(
            f"{context} requires trajectory_observations to contain the current coordinate columns: {', '.join(columns)}; "
            f"missing: {', '.join(missing)}. Regenerate data with the current schema; "
            "no coordinate fallback or mixing for older databases."
        )


def _remerge_load_observations(
    conn: sqlite3.Connection,
    scene_id: str,
    batch_id: str,
    *,
    world_coords: str,
    force_p_geo: bool = False,
) -> list[_RemergeObs]:
    """Load existing observations from the DB for offline merge recompute.

    world_coords controls which coordinates are read for merge:
      'p_geo'  → always world_x/y (pure-homography P_geo);
      'final'  → only final_world_x/y (post g2/MLP); rows with missing values are skipped.

    force_p_geo=True (set by run_remerge when --g2-checkpoint is active):
      force P_geo as the MLP inference input, ignoring world_coords;
      MLP outputs are written back to final_world_x/y; merge distance still
      uses P_geo by default, and final_world only when --remerge-g2-for-merge.
    """
    required_columns = [
        "image_x",
        "image_y",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
    ]
    if force_p_geo or world_coords == "p_geo":
        wx_expr, wy_expr = "world_x", "world_y"
        coord_filter = ""
        required_columns.extend(["world_x", "world_y"])
    elif world_coords == "final":
        wx_expr, wy_expr = "final_world_x", "final_world_y"
        coord_filter = "AND final_world_x IS NOT NULL AND final_world_y IS NOT NULL"
        required_columns.extend(["final_world_x", "final_world_y"])
    else:
        raise ValueError(f"unknown remerge world_coords: {world_coords}")

    _require_observation_columns(conn, required_columns, context="remerge")

    rows = conn.execute(
        f"""
        SELECT id, camera_name, local_track_id, frame_index, timestamp_sec,
               image_x, image_y, {wx_expr}, {wy_expr},
               bbox_x, bbox_y, bbox_w, bbox_h
         FROM trajectory_observations
         WHERE scene_id=? AND batch_id=?
           {coord_filter}
         ORDER BY timestamp_sec, frame_index, camera_name
        """,
        (scene_id, batch_id),
    ).fetchall()
    out: list[_RemergeObs] = []
    for row in rows:
        out.append(
            _RemergeObs(
                obs_id=int(row[0]),
                camera_name=str(row[1]),
                local_track_id=int(row[2]),
                frame_index=int(row[3]),
                timestamp_sec=float(row[4]),
                image_xy=(float(row[5]), float(row[6])),
                world_xy=(float(row[7]), float(row[8])),
                bbox_xywh=(float(row[9]), float(row[10]), float(row[11]), float(row[12])),
            )
        )
    return out


def _remerge_cluster_batches(
    observations: list[_RemergeObs],
    *,
    sync_gap_sec: float,
) -> list[list[_RemergeObs]]:
    """Cluster observations into co-timed rounds (simulating runtime global frames)."""
    if not observations:
        return []
    batches: list[list[_RemergeObs]] = []
    current: list[_RemergeObs] = [observations[0]]
    anchor_t = observations[0].timestamp_sec
    for obs in observations[1:]:
        if obs.timestamp_sec - anchor_t <= sync_gap_sec:
            current.append(obs)
        else:
            batches.append(current)
            current = [obs]
            anchor_t = obs.timestamp_sec
    batches.append(current)
    return batches


def _remerge_apply_updates(
    conn: sqlite3.Connection,
    scene_id: str,
    batch_id: str,
    track_final_gid: dict[tuple[str, int], int],
    g2_final_world: list[tuple[float, float, int]] | None = None,
) -> int:
    """Write recomputed global_id (required) and final_world_x/y (optional) back to the DB.

    g2_final_world format: [(final_x, final_y, obs_id), ...]; written by primary key ``id``.

    Uses a TEMP table + ``UPDATE ... FROM`` instead of per-row ``UPDATE`` on hundreds of thousands of rows.
    """
    if g2_final_world is not None:
        _require_observation_columns(
            conn,
            ["final_world_x", "final_world_y"],
            context="remerge g2 writeback",
        )

    updated = 0
    # Keep temp tables in memory so an ~800k-row INSERT does not spill to disk.
    conn.execute("PRAGMA temp_store=MEMORY")

    conn.execute("DROP TABLE IF EXISTS _remerge_gid")
    conn.execute(
        """
        CREATE TEMP TABLE _remerge_gid (
            camera_name TEXT NOT NULL,
            local_track_id INTEGER NOT NULL,
            global_id INTEGER NOT NULL
        )
        """
    )
    if track_final_gid:
        print(f"  [db] bulk global_id tracks={len(track_final_gid)} …")
        conn.executemany(
            "INSERT INTO _remerge_gid(camera_name, local_track_id, global_id) VALUES (?,?,?)",
            [(camera_name, local_track_id, global_id)
             for (camera_name, local_track_id), global_id in track_final_gid.items()],
        )
        conn.execute(
            "CREATE INDEX _remerge_gid_idx ON _remerge_gid(camera_name, local_track_id)"
        )
        cur = conn.execute(
            """
            UPDATE trajectory_observations AS o
               SET global_id = t.global_id
              FROM _remerge_gid AS t
             WHERE o.scene_id = ?
               AND o.batch_id = ?
               AND o.camera_name = t.camera_name
               AND o.local_track_id = t.local_track_id
            """,
            (scene_id, batch_id),
        )
        updated = int(cur.rowcount)
    conn.execute("DROP TABLE IF EXISTS _remerge_gid")

    if g2_final_world:
        print(f"  [db] bulk final_world rows={len(g2_final_world)} …")
        conn.execute("DROP TABLE IF EXISTS _remerge_g2")
        conn.execute(
            """
            CREATE TEMP TABLE _remerge_g2 (
                obs_id INTEGER PRIMARY KEY,
                final_world_x REAL NOT NULL,
                final_world_y REAL NOT NULL
            ) WITHOUT ROWID
            """
        )
        conn.executemany(
            "INSERT INTO _remerge_g2(obs_id, final_world_x, final_world_y) VALUES (?,?,?)",
            [(obs_id, fx, fy) for fx, fy, obs_id in g2_final_world],
        )
        conn.execute(
            """
            UPDATE trajectory_observations AS o
               SET final_world_x = t.final_world_x,
                   final_world_y = t.final_world_y
              FROM _remerge_g2 AS t
             WHERE o.id = t.obs_id
            """
        )
        conn.execute("DROP TABLE IF EXISTS _remerge_g2")

    conn.commit()
    return updated


def _remerge_summarize(
    conn: sqlite3.Connection,
    scene_id: str,
    batch_id: str,
) -> tuple[int, int, int]:
    row = conn.execute(
        """
        SELECT COUNT(*),
               COUNT(DISTINCT global_id),
               COUNT(DISTINCT camera_name || ':' || local_track_id)
         FROM trajectory_observations
         WHERE scene_id=? AND batch_id=?
        """,
        (scene_id, batch_id),
    ).fetchone()
    return int(row[0]), int(row[1]), int(row[2])


def run_remerge(args: argparse.Namespace) -> None:
    """--remerge mode: recompute cross-camera merge from existing DB observations without re-running YOLO."""
    # ── Flags that do not apply ────────────────────────────────────────────
    _REMERGE_IGNORED = [
        ("--source / --sub-source", "inference input; remerge does not read video"),
        ("--model",                 "YOLO model; remerge does not run inference"),
        ("--classes / --conf / --iou / --imgsz / --device",
                                    "YOLO inference parameters"),
        ("--clahe / --clahe-clip",  "image preprocess; remerge does not read frames"),
        ("--clip-margin / --clip-margin-ratio / --min-valid-streak",
                                    "runtime detection filters; remerge uses existing DB observations"),
        ("--record-every-n-frames / --db-batch-size / --db-flush-every-frames",
                                    "write cadence; remerge writes once in bulk"),
        ("--stats-every-frames",    "runtime stats; remerge reports once at the end"),
        ("--test / --no-view / --view",
                                    "debug video record/display; remerge has no video output"),
        ("--save-camera-dir / --save-bev-video / --save-bev-image",
                                    "video/image output paths; remerge has no video output"),
        ("--bev-background / --display-history-sec / --bev-history-samples",
                                    "BEV render parameters; remerge does not render video"),
        ("--rtsp-transport",        "RTSP transport; remerge does not connect cameras"),
        ("--anchor / --risk-scalar / --anchor-s0-m / --pixel-grazing-d0",
                                    "anchor computation; remerge uses DB world coordinates directly"),
        ("--g2-online-learn and --g2-learn-* / --g2-online-checkpoint-out",
                                    "online self-training; remerge only does batch inference"),
    ]
    print("  [remerge] the following flags have no effect in --remerge mode:")
    for param, reason in _REMERGE_IGNORED:
        print(f"    {param}  →  {reason}")
    g2_ckpt_path = getattr(args, "g2_checkpoint", None)
    g2_for_merge: bool = getattr(args, "remerge_g2_for_merge", False)
    if g2_ckpt_path:
        merge_note = "final_world used for merge distance" if g2_for_merge else "merge still uses DB P_geo; only final_world_x/y are written"
        print(f"    --g2-checkpoint is active: batch-recompute final_world_x/y ({merge_note})")
    print()

    config_path = resolve_project_path(args.config)
    db_path = resolve_project_path(args.db)
    dry_run: bool = getattr(args, "remerge_dry_run", False)
    world_coords: str = getattr(args, "remerge_world_coords", "p_geo")
    sync_gap_sec: float = getattr(args, "remerge_sync_gap_sec", 0.05)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Merge flags reuse the same --merge-* CLI as a live run; YAML trajectory_merge also applies
    merge_section = _merge_config_section(cfg)
    match_cfg = build_match_config(args, merge_section)
    print(
        f"  [cfg] score>={match_cfg.score_threshold:.2f}  "
        f"dist<={match_cfg.max_distance_m:.2f}m  "
        f"sigma(dist/time/shape)=({match_cfg.distance_sigma:.2f}/"
        f"{match_cfg.time_sigma:.2f}/{match_cfg.shape_sigma:.2f})  "
        f"weights(dist/time/shape)=({match_cfg.distance_weight:.2f}/"
        f"{match_cfg.time_weight:.2f}/{match_cfg.shape_weight:.2f})"
    )
    if match_cfg.stitch_enabled:
        print(f"  [stitch] same-camera stitch ON  {_linking_config_summary(match_cfg)}")
    else:
        print("  [stitch] same-camera stitch OFF (default/config)")

    cameras = _remerge_build_cameras(cfg, str(db_path), args.scene, args.batch)
    geometry = build_canvas_geometry(cameras, cfg)
    print(
        f"  [BEV] canvas={geometry.size[0]}x{geometry.size[1]}  "
        f"scale={geometry.scale:.2f}px/m  "
        f"overlap_pairs={sorted(geometry.overlap_pairs)}"
    )

    # ── Optional: load G2 residual MLP ────────────────────────────────────
    g2_controller: Any | None = None
    if g2_ckpt_path:
        try:
            from pipeline.geometry_guided_residual_mlp import (
                MVCResidualController,
                load_camera_translation_corrections,
            )
        except ImportError as exc:
            raise RuntimeError("g2 requires PyTorch: pip install torch") from exc
        ckpt = resolve_project_path(g2_ckpt_path)
        if not ckpt.is_file():
            print(f"  [warn] g2 checkpoint not found: {ckpt}; skipping final_world_x/y update")
        else:
            runtime_corrections = load_camera_translation_corrections(
                getattr(args, "g2_camera_translation_json", None),
                acceptance=getattr(args, "g2_camera_translation_acceptance", "recommended"),
            )
            g2_controller = MVCResidualController.from_checkpoint(
                ckpt,
                camera_translation_corrections=runtime_corrections,
            )
            _configure_g2_residual_invoke(g2_controller, args)
            print(f"  [g2] loaded residual MLP: {ckpt}")
            if runtime_corrections:
                print(
                    "  [g2] runtime fixed camera translations loaded: "
                    f"{runtime_corrections.camera_count()} camera(s), "
                    f"{runtime_corrections.scene_count()} scene map(s)"
                )
            elif g2_controller.training_camera_translation_corrections:
                print(
                    "  [g2][note] checkpoint recorded a training-time camera translation, "
                    "but remerge will not apply it automatically; to write T_cam, pass "
                    "--g2-camera-translation-json."
                )

    # Camera calibration lookup: name → (H_infer, scale, infer_wh)
    # cam.homography was already scaled to infer-resolution coords (infer_wh)
    # by _remerge_build_cameras; wh must share that frame or grazing_alpha/F
    # will disagree with tracking-time features.
    cam_cals: dict[str, tuple[Any, float, tuple[int, int]]] = {
        cam.name: (cam.homography, cam.scale, cam.infer_wh)
        for cam in cameras
        if cam.infer_wh is not None
    }

    conn = init_trajectory_db(str(db_path), args.batch)
    stage_run_id = start_batch_stage(
        conn,
        [args.scene],
        args.batch,
        "remerge",
        {
            "config": str(config_path),
            "dry_run": dry_run,
            "world_coords": world_coords,
            "g2_checkpoint": str(g2_ckpt_path) if g2_ckpt_path else None,
            "g2_for_merge": g2_for_merge,
        },
    )
    try:
        obs_before, gid_before, local_before = _remerge_summarize(
            conn, args.scene, args.batch
        )
        if obs_before == 0:
            raise SystemExit(
                f"scene={args.scene!r} has no trajectory_observations in the DB; "
                "run multi_camera_trajectory_fusion.py first to collect data."
            )

        # When g2 is active, MLP input must be P_geo
        effective_world_coords = "p_geo" if g2_controller is not None else world_coords
        observations = _remerge_load_observations(
            conn, args.scene, args.batch,
            world_coords=effective_world_coords,
            force_p_geo=(g2_controller is not None),
        )
        if effective_world_coords == "final":
            skipped = obs_before - len(observations)
            if skipped > 0:
                print(
                    f"  [remerge][note] final mode skipped {skipped} observations without final_world_x/y; "
                    "no fallback to DB P_geo."
                )
            if not observations:
                raise SystemExit(
                    "remerge --remerge-world-coords final found no final_world_x/y; "
                    "write them first with --g2-checkpoint, or use --remerge-world-coords p_geo."
                )
        batches = _remerge_cluster_batches(observations, sync_gap_sec=sync_gap_sec)
        coord_label = "g2→final_world" if g2_controller is not None else effective_world_coords
        print(
            f"  [data] observations={len(observations)}  "
            f"local_tracks={local_before}  global_ids_before={gid_before}  "
            f"sync_batches={len(batches)}  world_coords={coord_label}"
        )

        if not dry_run:
            conn.execute(
                "DELETE FROM trajectory_merges WHERE scene_id=? AND batch_id=?",
                (args.scene, args.batch),
            )
            conn.execute(
                "DELETE FROM trajectory_global_tracks WHERE scene_id=? AND batch_id=?",
                (args.scene, args.batch),
            )
            conn.commit()

        # ── G2 batch inference: final_world_xy per observation ────────────
        # obs_final_world stores (final_x, final_y, obs_id), written back by PK
        obs_final_world: list[tuple[float, float, int]] = []
        # obs_world_for_merge: same length as observations; coords used for merge distance
        # key: (camera_name, local_track_id, frame_index) -> merge coords
        # (final when g2 is on and explicitly requested, else p_geo)
        merge_world_lut: dict[tuple[str, int, int], tuple[float, float]] = {}
        if g2_controller is not None:
            import numpy as _np
            print(f"  [g2] batch-inferring {len(observations)} observations …")

            # Diagnostics: calib-H reprojection vs DB P_geo, and delta distribution
            geo_diffs: list[float] = []   # |world_geo_from_H - DB_p_geo|
            delta_norms: list[float] = [] # |delta_m|

            for obs in observations:
                H, scale, wh = cam_cals.get(obs.camera_name, (None, None, None))
                key = (obs.camera_name, obs.local_track_id, obs.frame_index)
                if H is None:
                    merge_world_lut[key] = obs.world_xy
                    continue
                bx, by, bw, bh = obs.bbox_xywh
                xyxy = _np.array([bx, by, bx + bw, by + bh], dtype=_np.float64)
                pred = g2_controller.predict_residual(
                    xyxy,
                    H,
                    scale,
                    wh,
                    camera_name=obs.camera_name,
                    scene_id=args.scene,
                    world_geo_override=obs.world_xy,
                )

                # Use DB P_geo (obs.world_xy) as the world-coordinate base and
                # add only the MLP residual delta. Do not use the world_geo that
                # predict_residual reprojects through calib H (that drifts badly
                # when infer resolution differs from calibration resolution).
                dx, dy = float(pred.delta_m[0]), float(pred.delta_m[1])
                tx, ty = g2_controller.camera_translation(obs.camera_name, args.scene)
                fx = obs.world_xy[0] + tx + dx
                fy = obs.world_xy[1] + ty + dy

                geo_diff = float(_np.hypot(
                    pred.world_geo[0] - obs.world_xy[0],
                    pred.world_geo[1] - obs.world_xy[1],
                ))
                geo_diffs.append(geo_diff)
                delta_norms.append(float(_np.hypot(dx, dy)))

                obs_final_world.append((fx, fy, obs.obs_id))
                # --remerge-g2-for-merge chooses final_world vs DB P_geo for merge distance
                # Default is P_geo so a weak MLP cannot break pairs that would otherwise merge
                merge_world_lut[key] = (fx, fy) if g2_for_merge else obs.world_xy

            n = len(geo_diffs)
            if n > 0:
                mean_diff = float(_np.mean(geo_diffs))
                max_diff  = float(_np.max(geo_diffs))
                mean_delta = float(_np.mean(delta_norms))
                max_delta  = float(_np.max(delta_norms))
                if max_diff > 0.5:
                    print(
                        f"  [g2 diag] ⚠ large calib-H reprojection vs DB P_geo: "
                        f"mean={mean_diff:.2f}m  max={max_diff:.2f}m\n"
                        f"    likely cause: DB has no trajectory_camera_config yet (older data); "
                        "remerge substituted calibration-image resolution for infer resolution, so F may be biased.\n"
                        f"    automatically using DB P_geo as the base + MLP delta."
                    )
                else:
                    print(
                        f"  [g2 diag] calib-H reprojection vs DB P_geo is OK: "
                        f"mean={mean_diff:.3f}m  max={max_diff:.3f}m"
                    )
                # Detect MLP tanh saturation: mean ≈ max ≈ max_residual_m*√2 means F is likely wrong
                max_res = getattr(g2_controller.train_cfg, "max_residual_m", 1.5)
                saturate_threshold = max_res * 1.35   # √2 ≈ 1.414, 5% slack
                if mean_delta > saturate_threshold and (max_delta - mean_delta) < 0.05:
                    print(
                        f"  [g2 diag] ⚠ MLP delta looks tanh-saturated: "
                        f"mean|Δ|={mean_delta:.3f}m ≈ max|Δ|={max_delta:.3f}m ≈ √2·r_max;\n"
                        f"    feature F is likely wrong (resolution/H mismatch); "
                        "this final_world_x/y update is unreliable. Run fusion to write\n"
                        "    trajectory_camera_config, then --remerge again."
                    )
                else:
                    print(
                        f"  [g2 delta] mean|Δ|={mean_delta:.3f}m  max|Δ|={max_delta:.3f}m"
                    )
            print(f"  [g2] inference done; will update {len(obs_final_world)} final_world_x/y rows")

        if dry_run:
            # TrajectoryFusion's online path immediately records global tracks / merge events;
            # a SAVEPOINT keeps the algorithm path identical and atomically rolls back the trial writes after reporting.
            conn.execute("SAVEPOINT remerge_dry_run")
            dry_run_savepoint_active = True

        fusion = TrajectoryFusion(
            conn, args.scene, args.batch, geometry, match_cfg,
            max_history=args.history_samples,
            auto_commit_merge_events=not dry_run,
        )
        fusion.next_global_id = 1

        comparisons_total = 0
        merge_event_count = 0
        remerge_sink = None
        if getattr(args, "remerge_profile", None):
            remerge_sink = _load_orin_stage_profile().RemergeProfileSink(args.remerge_profile)
            print(f"  [remerge-profile] {args.remerge_profile}")
        quiet = remerge_sink is not None or bool(getattr(args, "remerge_quiet", False))
        for sync_idx, batch in enumerate(batches):
            latest_ts = max(o.timestamp_sec for o in batch)
            t_add = time.perf_counter()
            for obs in batch:
                key = (obs.camera_name, obs.local_track_id, obs.frame_index)
                merge_xy = merge_world_lut.get(key, obs.world_xy)
                sample = TrackSample(
                    timestamp_sec=obs.timestamp_sec,
                    frame_index=obs.frame_index,
                    image_xy=obs.image_xy,
                    world_xy=merge_xy,
                    bbox_xywh=obs.bbox_xywh,
                    confidence=None,
                    class_id=None,
                    class_name=None,
                    clipped=False,
                )
                fusion.add_sample(obs.camera_name, obs.local_track_id, sample)
            add_ms = (time.perf_counter() - t_add) * 1000.0

            merge_ms = 0.0
            comparisons = 0
            events: list = []
            if sync_idx % args.merge_every_frames == 0:
                t_merge = time.perf_counter()
                comparisons, events = fusion.merge_active(latest_ts)
                merge_ms = (time.perf_counter() - t_merge) * 1000.0
                comparisons_total += comparisons
                merge_event_count += len(events)
                if not quiet:
                    for event in events:
                        event_label = (
                            "same-camera link"
                            if event.merge_kind == "same_camera_link"
                            else "cross-camera merge"
                        )
                        print(
                            f"  [{event_label}] G{event.merged_global_id} -> "
                            f"G{event.kept_global_id}  "
                            f"{event.track_a.camera_name}:L{event.track_a.local_track_id} "
                            f"↔ {event.track_b.camera_name}:L{event.track_b.local_track_id}  "
                            f"score={event.scores.score:.3f}  "
                            f"dist={event.scores.mean_distance_m:.2f}m"
                        )
            if remerge_sink is not None:
                remerge_sink.add_batch(
                    sync_idx=sync_idx,
                    n_obs=len(batch),
                    n_cameras=len({o.camera_name for o in batch}),
                    timestamp_sec=latest_ts,
                    add_ms=add_ms,
                    merge_ms=merge_ms,
                    n_events=len(events),
                    n_comparisons=comparisons,
                )
        if remerge_sink is not None:
            remerge_sink.close()
            print(f"  [remerge-profile] wrote {remerge_sink.path} ({len(remerge_sink.rows)} rows)")

        track_final_gid = {
            key: fusion.uf.find(track.global_id)
            for key, track in fusion.local_tracks.items()
        }
        unique_gids_after = len(set(track_final_gid.values()))

        print(
            f"\n  [result] comparisons={comparisons_total}  merges={merge_event_count}  "
            f"unique_global_ids_after={unique_gids_after}"
        )

        if dry_run:
            conn.execute("ROLLBACK TO remerge_dry_run")
            conn.execute("RELEASE remerge_dry_run")
            dry_run_savepoint_active = False
            print("  [dry-run] trajectory tables were not rewritten (batch-stage audit rows only).")
            return

        fusion.flush_rewrites()

        updated_rows = _remerge_apply_updates(
            conn, args.scene, args.batch, track_final_gid,
            g2_final_world=obs_final_world if obs_final_world else None,
        )
        _, gid_after, _ = _remerge_summarize(conn, args.scene, args.batch)
        merge_rows = conn.execute(
            "SELECT COUNT(*) FROM trajectory_merges WHERE scene_id=? AND batch_id=?",
            (args.scene, args.batch),
        ).fetchone()[0]
        g2_msg = f"  final_world_rows={len(obs_final_world)}" if obs_final_world else ""
        print(
            f"  [db] updated_rows={updated_rows}  "
            f"global_ids {gid_before}->{gid_after}  "
            f"trajectory_merges={merge_rows}{g2_msg}"
        )
    finally:
        if locals().get("dry_run_savepoint_active", False):
            conn.execute("ROLLBACK TO remerge_dry_run")
            conn.execute("RELEASE remerge_dry_run")
        finish_batch_stage(
            conn,
            stage_run_id,
            "failed" if sys.exc_info()[0] is not None else "completed",
            {
                "observations": locals().get("obs_before"),
                "merge_events": locals().get("merge_event_count"),
                "updated_rows": locals().get("updated_rows", 0),
            },
        )
        conn.close()


def build_canvas_geometry(
    cameras: list[CameraRuntime],
    config: dict[str, Any],
) -> CanvasGeometry:
    def safe_roi_points(cam: CameraRuntime, image: np.ndarray) -> np.ndarray:
        if "image_roi" in cam.cfg:
            return camera_roi_points(image, cam.cfg)

        height, width = image.shape[:2]
        h = cam.calib_homography
        xs = np.linspace(0.0, float(width - 1), 17)
        bottom_sign = np.sign(h[2, 0] * (width * 0.5) + h[2, 1] * (height - 1) + h[2, 2])
        if bottom_sign == 0:
            bottom_sign = 1.0
        valid_top: list[float] = []
        horizon_margin = max(8.0, height * 0.02)
        for x in xs:
            if abs(h[2, 1]) < 1e-12:
                y = 0.0
            else:
                horizon_y = -(h[2, 0] * x + h[2, 2]) / h[2, 1]
                y = horizon_y + bottom_sign * np.sign(h[2, 1]) * horizon_margin
            valid_top.append(float(np.clip(y, 0.0, height - 1.0)))
        top = np.column_stack((xs, np.asarray(valid_top, dtype=np.float64)))
        bottom = np.asarray([[width - 1.0, height - 1.0], [0.0, height - 1.0]])
        return np.vstack((top, bottom)).astype(np.float32)

    all_corners = []
    for cam in cameras:
        # Use main-stream size + original cam.cfg (main-stream image_roi) + calib_homography (H_main)
        # so canvas bounds line up with the background produced by bev_stitch.py
        ref_img = _calib_ref_image(cam)
        roi = safe_roi_points(cam, ref_img).reshape(-1, 1, 2)
        all_corners.append(cv.perspectiveTransform(roi, cam.calib_homography).reshape(-1, 2))

    corners = np.vstack(all_corners)
    min_xy = np.floor(corners.min(axis=0)).astype(int)
    max_xy = np.ceil(corners.max(axis=0)).astype(int)
    margin = int(config.get("canvas_margin_px", 50))
    min_xy -= margin
    max_xy += margin
    width = int(max_xy[0] - min_xy[0] + 1)
    height = int(max_xy[1] - min_xy[1] + 1)
    translate = np.asarray(
        [[1.0, 0.0, -min_xy[0]], [0.0, 1.0, -min_xy[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    coverage_masks: dict[str, np.ndarray] = {}
    for cam in cameras:
        # Coverage masks also use main-stream data so overlap detection matches the background BEV
        ref_img = _calib_ref_image(cam)
        mask = np.zeros(ref_img.shape[:2], dtype=np.uint8)
        roi = np.round(safe_roi_points(cam, ref_img)).astype(np.int32)
        cv.fillPoly(mask, [roi], 255)
        warped = cv.warpPerspective(
            mask,
            translate @ cam.calib_homography,
            (width, height),
            flags=cv.INTER_NEAREST,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=0,
        )
        coverage_masks[cam.name] = warped

    overlap_pairs: set[tuple[str, str]] = set()
    names = [c.name for c in cameras]
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            if np.any((coverage_masks[name_a] > 0) & (coverage_masks[name_b] > 0)):
                overlap_pairs.add(tuple(sorted((name_a, name_b))))

    return CanvasGeometry(
        camera_names=names,
        scale=cameras[0].scale,
        min_xy=min_xy,
        translate=translate,
        size=(width, height),
        coverage_masks=coverage_masks,
        overlap_pairs=overlap_pairs,
    )


def adapt_homography_to_resolution(
    H: np.ndarray,
    calib_wh: tuple[int, int],
    infer_wh: tuple[int, int],
) -> np.ndarray:
    """Rescale a homography estimated at main-stream resolution so it can be used at sub-stream resolution.

    Derivation:
        BEV = H_main @ p_main
        p_main = S^{-1} @ p_sub, where S = diag(sx, sy, 1), sx = infer_w/calib_w
        therefore H_sub = H_main @ S^{-1}

    Parameters
    ----------
    H          3x3 homography from calibration (calib_wh coordinate frame)
    calib_wh   calibration still (main-stream snapshot) (width, height)
    infer_wh   infer stream (sub-stream) (width, height)

    Returns
    -------
    H_sub : 3x3 ndarray, homography for sub-stream pixel coordinates
    """
    if calib_wh == infer_wh:
        return H.copy()
    sx = infer_wh[0] / calib_wh[0]
    sy = infer_wh[1] / calib_wh[1]
    # S_inv: sub-stream pixels → main-stream pixels
    S_inv = np.array(
        [[1.0 / sx, 0.0, 0.0],
         [0.0, 1.0 / sy, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return H.astype(np.float64) @ S_inv


def _scale_cam_cfg_roi(
    cam_cfg: dict[str, Any],
    calib_wh: tuple[int, int],
    infer_wh: tuple[int, int],
) -> dict[str, Any]:
    """Copy camera config and scale image_roi from main-stream to infer-stream resolution.

    image_roi is a hand-drawn polygon [[x,y], ...] on the main-stream still;
    sub-stream coordinates must be scaled down or the ROI overflows the frame
    and the mask is wrong.
    """
    import copy as _copy
    effective = _copy.deepcopy(cam_cfg)
    if calib_wh == infer_wh or "image_roi" not in cam_cfg:
        return effective
    sx = infer_wh[0] / calib_wh[0]
    sy = infer_wh[1] / calib_wh[1]
    scaled_roi = [[int(x * sx + 0.5), int(y * sy + 0.5)] for x, y in cam_cfg["image_roi"]]
    effective["image_roi"] = scaled_roi
    return effective


def _is_gst_pipeline(source: str | int) -> bool:
    if not isinstance(source, str):
        return False
    s = source.strip().lower()
    return (
        "nvv4l2decoder" in s
        or s.startswith("filesrc ")
        or "! appsink" in s
    )


def _jetson_gst_bgr_pipeline(video_path: str) -> str:
    """Jetson nvv4l2decoder → BGR appsink. Location must be quoted.

    drop=false + a short blocking queue: nvv4l2decoder with sync=false would
    otherwise drain the file during TensorRT/tracker load, then EOS after a
    few leftover frames (IV-F r0 Town05 died at ~180 / 1800).
    """
    loc = Path(video_path).expanduser().resolve().as_posix()
    return (
        f'filesrc location="{loc}" ! qtdemux ! h264parse ! nvv4l2decoder '
        f"! nvvidconv ! video/x-raw,format=BGRx "
        f"! queue max-size-buffers=8 max-size-time=0 max-size-bytes=0 leaky=no "
        f"! videoconvert ! video/x-raw,format=BGR "
        f"! appsink drop=false max-buffers=8 sync=false"
    )


def _open_capture(source: str | int, rtsp_transport: str = "tcp") -> cv.VideoCapture:
    """Open a video source; for RTSP enable TCP transport and shrink the buffer."""
    if isinstance(source, str) and source.lower().startswith("rtsp"):
        # OpenCV/FFmpeg reads this env var when VideoCapture opens; set it first.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{rtsp_transport}"
        cap = cv.VideoCapture(
            source,
            cv.CAP_FFMPEG,
            [
                cv.CAP_PROP_OPEN_TIMEOUT_MSEC, 8_000,    # connect timeout 8s
                cv.CAP_PROP_READ_TIMEOUT_MSEC, 5_000,    # read timeout 5s
            ],
        )
    else:
        gst_src: str | None = None
        if _is_gst_pipeline(source):
            gst_src = str(source)
        elif (
            isinstance(source, str)
            and os.environ.get("HSG_ORIN_GST") == "1"
            and _is_video_file(source)
        ):
            gst_src = _jetson_gst_bgr_pipeline(source)
        if gst_src is not None:
            if not hasattr(cv, "CAP_GSTREAMER"):
                raise RuntimeError(
                    "HSG_ORIN_GST=1 but this OpenCV has no CAP_GSTREAMER. "
                    "Use JetPack/apt cv2, not pip install opencv-python; "
                    "or export HSG_ORIN_GST=0 for CPU decode."
                )
            cap = cv.VideoCapture(gst_src, cv.CAP_GSTREAMER)
            # Do not cap.set(BUFFERSIZE): OpenCV 4.8 GST backend restarts the
            # pipeline, and the second qtdemux hits "Internal data stream error".
            print(f"  [gst] {gst_src}", flush=True)
            return cap
        cap = cv.VideoCapture(source)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)   # smallest buffer; less backlog and latency
    return cap


def open_cameras(
    cfg: dict[str, Any],
    db_path: str,
    scene_id: str,
    source_overrides: dict[str, str | int],
    rtsp_transport: str = "tcp",
    require_geometry_version: str | None = None,
) -> list[CameraRuntime]:
    """Initialize per-camera calibration and inference sources.

    ``require_geometry_version``: if not None, each camera's DB calibration
    ``geometry_version`` must match this string exactly or raise RuntimeError.
    When a g2 checkpoint is enabled, pass ``GEOMETRY_VERSION`` (exported by
    ``multi_camera_bev_stitch``) so a cla1/earlier homography cannot be used
    silently with fab1 weights — a different Jacobian convention would
    corrupt alpha/kappa and drift MLP feature meaning.
    Pure geometry visualization can leave the default None (no version gate).
    """
    conn = init_calibration_db(db_path)
    cameras: list[CameraRuntime] = []
    for cam_cfg in cfg["cameras"]:
        name = cam_cfg["name"]
        infer_source, mode_label = resolve_tracking_sources(name, cam_cfg, source_overrides)
        if (
            cam_cfg.get("video")
            and (cam_cfg.get("stream") or cam_cfg.get("sub_stream"))
            and name not in source_overrides
            and f"sub:{name}" not in source_overrides
        ):
            print(
                f"  [note] {name}: video is set; offline test will use {cam_cfg['video']!r}, "
                "ignoring YAML stream/sub_stream (for live streams comment out video or use --source)."
            )
        cal = load_calibration(conn, name, scene_id,
                               require_geometry_version=require_geometry_version)
        if cal is None:
            raise RuntimeError(
                f"no calibration data for camera '{name}' (scene '{scene_id}') in the database; "
                "run pipeline/multi_camera_bev_stitch.py --save-cal first."
            )
        homography, scale = cal

        # Calibration resolution always comes from image (main-stream still), independent of infer source
        calib_wh: tuple[int, int] | None = None
        calib_img_path = cam_cfg.get("image")
        if calib_img_path and os.path.isfile(calib_img_path):
            probe = cv.imread(calib_img_path)
            if probe is not None:
                calib_wh = (probe.shape[1], probe.shape[0])

        cap = _open_capture(infer_source, rtsp_transport)
        if not cap.isOpened():
            raise RuntimeError(f"camera '{name}': cannot open inference source {infer_source!r}")

        ok, first = cap.read()
        if not ok or first is None:
            raise RuntimeError(f"camera '{name}': cannot read first frame from {infer_source!r}")

        infer_wh: tuple[int, int] = (first.shape[1], first.shape[0])
        needs_scale = (
            calib_wh is not None
            and (infer_wh[0] != calib_wh[0] or infer_wh[1] != calib_wh[1])
        )

        if needs_scale:
            h_effective = adapt_homography_to_resolution(homography, calib_wh, infer_wh)
            effective_cfg = _scale_cam_cfg_roi(cam_cfg, calib_wh, infer_wh)
            res_note = (
                f"{mode_label} {infer_wh[0]}×{infer_wh[1]}, "
                f"calib image {calib_wh[0]}×{calib_wh[1]} (H/ROI rescaled)"
            )
        elif calib_wh is None:
            print(
                f"  [warn] {name}: calibration image '{calib_img_path}' not found; "
                "cannot rescale by resolution; using the original homography (may be biased)."
            )
            h_effective = homography
            effective_cfg = cam_cfg
            calib_wh = infer_wh
            res_note = f"{mode_label} {infer_wh[0]}×{infer_wh[1]} (no calib image; resolution rescale skipped)"
        else:
            h_effective = homography
            effective_cfg = cam_cfg
            res_note = f"{mode_label} {infer_wh[0]}×{infer_wh[1]} (same resolution as calib image)"

        fps = float(cap.get(cv.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps <= 1e-3:
            fps = 25.0
        cameras.append(
            CameraRuntime(
                name=name,
                cfg=cam_cfg,
                effective_cfg=effective_cfg,
                source=infer_source,
                cap=cap,
                first_frame=first,
                homography=h_effective,
                calib_homography=homography,  # always keep original H_main for canvas bounds
                scale=scale,
                fps=fps,
                rtsp_transport=rtsp_transport,
                calib_wh=calib_wh,
                infer_wh=infer_wh,
            )
        )
        print(f"  [input] {name}: {res_note}, fps≈{fps:.2f}, scale={scale:.1f}px/m")
    conn.close()
    return cameras


def _try_reconnect(cam: CameraRuntime, max_retries: int = 3) -> bool:
    """Try to reopen the video source; return True on success.

    For RTSP: recover after a brief network glitch or camera reboot.
    For files: reopen would restart from the beginning; usually not called.
    """
    for attempt in range(1, max_retries + 1):
        print(f"  [reconnect] {cam.name} attempt {attempt}/{max_retries} …")
        try:
            cam.cap.release()
            time.sleep(min(attempt * 1.5, 5.0))   # backoff
            cam.cap = _open_capture(cam.source, cam.rtsp_transport)
            if cam.cap.isOpened():
                ok, frame = cam.cap.read()
                if ok and frame is not None:
                    cam.last_good_frame = frame
                    cam.consecutive_failures = 0
                    print(f"  [reconnect] {cam.name} reconnected.")
                    return True
        except Exception as exc:  # noqa: BLE001
            print(f"  [reconnect] {cam.name} attempt {attempt} failed: {exc}")
    return False


def _is_live_source(source: str | int) -> bool:
    """Live RTSP/HTTP/USB sources: use wall-clock timestamps (camera and host must be NTP-synced)."""
    if isinstance(source, int):
        return True
    s = source.lower()
    return s.startswith("rtsp") or s.startswith("http://") or s.startswith("https://")


def _frame_timestamp_sec(cam: CameraRuntime) -> float:
    """Timestamp for the current frame (seconds).

    Live streams: time.time() (Unix seconds, aligned across cameras).
    Video files: prefer CAP_PROP_POS_MSEC, else frame_index/fps relative time.
    """
    if _is_live_source(cam.source):
        return time.time()
    pos_msec = float(cam.cap.get(cv.CAP_PROP_POS_MSEC) or 0.0)
    if pos_msec > 1e-3:
        return pos_msec / 1000.0
    return cam.frame_index / max(cam.fps, 1e-6)


def _sync_file_frame_index(cam: CameraRuntime) -> None:
    """Refresh ``cam.frame_index`` to the container frame just retrieved.

    ``CAP_PROP_POS_FRAMES`` is stuck at 0 on some containers/decoders; then
    fall back to incrementing so the ``frame_index/fps`` path in
    ``_frame_timestamp_sec`` stays monotonic. Otherwise the catch-up loop
    never advances the timestamp and spins to the grab cap (looks like
    fast-forward).
    """
    pos = float(cam.cap.get(cv.CAP_PROP_POS_FRAMES) or 0.0)
    if math.isfinite(pos) and pos > 0:
        cam.frame_index = max(0, int(round(pos)) - 1)
    else:
        cam.frame_index += 1


def offline_sync_fps(fps_values: list[float]) -> float:
    """Offline multi-camera shared timeline step: use the smallest valid FPS so higher-rate cameras catch up multiple frames per beat."""
    valid = [float(f) for f in fps_values if math.isfinite(f) and f > 1e-3]
    if not valid:
        return 25.0
    return float(min(valid))


def offline_fps_mismatch(fps_values: list[float], *, rel_tol: float = 0.05) -> bool:
    """Treat container FPS relative difference above ``rel_tol`` as needing timeline catch-up (e.g. 25 vs 50)."""
    valid = [float(f) for f in fps_values if math.isfinite(f) and f > 1e-3]
    if len(valid) < 2:
        return False
    lo = min(valid)
    hi = max(valid)
    return (hi - lo) / lo > rel_tol


def _max_catchup_frames(cam_fps: float, sync_fps: float) -> int:
    """Max catch-up frames per beat: fps ratio plus container timestamp jitter slack."""
    ratio = max(float(cam_fps), 1e-3) / max(float(sync_fps), 1e-3)
    return max(2, int(math.ceil(ratio)) * 2 + 2)


def read_frame_for_timeline(
    cam: CameraRuntime,
    target_t: float,
    *,
    sync_fps: float,
    max_reconnect_retries: int = 3,
) -> tuple[bool, np.ndarray | None, float, bool]:
    """Offline files: catch up to shared timeline ``target_t`` and deliver the nearest frame.

    A higher-rate source (e.g. 50 FPS) on a slower sync beat (e.g. 25 FPS)
    skips intermediate frames so cross-camera ``timestamp_sec`` stays aligned
    and merge active windows / track time overlap remain valid. Live streams
    / async-reader paths do not use this.
    """
    if cam.async_reader_thread is not None and cam.async_reader_thread.is_alive():
        return _snapshot_latest_frame(cam)
    if _is_live_source(cam.source):
        return read_next_frame(cam, max_reconnect_retries=max_reconnect_retries)

    half = 0.5 / max(float(cam.fps), 1e-3)
    max_grabs = _max_catchup_frames(cam.fps, sync_fps)

    if cam.first_frame is not None:
        frame = cam.first_frame
        cam.first_frame = None
        cam.last_good_frame = frame
        # open_cameras already consumed frame 0; start the timeline at 0 to avoid occasional empty POS_MSEC.
        ts = 0.0
        cam.frame_index = 0
        cam.last_timestamp_sec = ts
        if ts + half >= float(target_t):
            return True, frame, ts, True

    # Skipped intermediate frames are grab-only (no retrieve): skip YUV→BGR and Mat alloc.
    # A high-rate source (50→25) saves one full-frame color convert per beat.
    grabs = 0
    ts = cam.last_timestamp_sec
    while grabs < max_grabs:
        if not cam.cap.grab():
            # Offline-file grab failure means EOF; leftover frames behind the shared timeline need not be delivered.
            return False, None, 0.0, False
        grabs += 1
        _sync_file_frame_index(cam)
        ts = _frame_timestamp_sec(cam)
        if ts + half >= float(target_t):
            break

    ok, frame = cam.cap.retrieve()
    if not ok or frame is None:
        return False, None, 0.0, False
    cam.consecutive_failures = 0
    cam.last_good_frame = frame
    cam.last_timestamp_sec = ts
    return True, frame, ts, True


def _snapshot_latest_frame(cam: CameraRuntime) -> tuple[bool, np.ndarray | None, float, bool]:
    """Take the latest frame from the async reader thread (decoupled from YOLO cost)."""
    with cam.read_lock:
        if cam.latest_frame is None:
            if cam.last_good_frame is not None:
                return False, cam.last_good_frame.copy(), time.time(), False
            return False, None, 0.0, False
        frame = cam.latest_frame.copy()
        ts = cam.latest_frame_time
        seq = cam.latest_frame_seq
    fresh = seq != cam.last_consumed_seq
    cam.last_consumed_seq = seq
    cam.last_good_frame = frame
    return True, frame, ts, fresh


def _camera_reader_loop(cam: CameraRuntime, stop: threading.Event) -> None:
    """Background RTSP reader; the main loop only consumes the latest frame so a slow YOLO does not backlog/skip/pad."""
    while not stop.is_set():
        ok, frame = cam.cap.read()
        if ok and frame is not None:
            with cam.read_lock:
                cam.latest_frame = frame
                cam.latest_frame_time = time.time()
                cam.latest_frame_seq += 1
            cam.consecutive_failures = 0
            cam.last_good_frame = frame
            continue

        cam.consecutive_failures += 1
        if _is_live_source(cam.source) and cam.consecutive_failures % 15 == 0:
            _try_reconnect(cam, max_retries=1)
        stop.wait(0.02)


def start_async_readers(cameras: list[CameraRuntime]) -> None:
    for cam in cameras:
        if cam.first_frame is not None:
            cam.latest_frame = cam.first_frame.copy()
            cam.latest_frame_time = time.time()
            cam.latest_frame_seq = 1
            cam.last_good_frame = cam.first_frame
            cam.first_frame = None
        stop = threading.Event()
        cam.async_reader_stop = stop
        cam.async_reader_thread = threading.Thread(
            target=_camera_reader_loop,
            args=(cam, stop),
            name=f"reader-{cam.name}",
            daemon=True,
        )
        cam.async_reader_thread.start()
    print("  [record] async reader threads started (debug video decoupled from YOLO; timelines aligned by sync#)")


def stop_async_readers(cameras: list[CameraRuntime]) -> None:
    for cam in cameras:
        if cam.async_reader_stop is not None:
            cam.async_reader_stop.set()
        if cam.async_reader_thread is not None:
            cam.async_reader_thread.join(timeout=3.0)
        cam.async_reader_stop = None
        cam.async_reader_thread = None


def read_next_frame(
    cam: CameraRuntime,
    max_reconnect_retries: int = 3,
    max_consecutive_failures: int = 10,
) -> tuple[bool, np.ndarray | None, float, bool]:
    """Read the next frame; for RTSP, auto-reconnect after consecutive failures.

    Returns (ok, frame, timestamp_sec, fresh):
      · ok=True, fresh=True   new frame (including latest from the async reader)
      · ok=False, fresh=False temporary pad (repeat last frame; debug video marks STALE)
      · ok=False, frame=None  hard failure (EOF or reconnect exhausted)
    """
    if cam.async_reader_thread is not None and cam.async_reader_thread.is_alive():
        return _snapshot_latest_frame(cam)

    if cam.first_frame is not None:
        frame = cam.first_frame
        cam.first_frame = None
        cam.last_good_frame = frame
        ts = _frame_timestamp_sec(cam) if _is_live_source(cam.source) else 0.0
        cam.last_timestamp_sec = float(ts)
        return True, frame, ts, True

    ok, frame = cam.cap.read()
    if ok and frame is not None:
        cam.consecutive_failures = 0
        cam.last_good_frame = frame
        ts = _frame_timestamp_sec(cam)
        cam.last_timestamp_sec = float(ts)
        return True, frame, ts, True

    # Read failed
    cam.consecutive_failures += 1
    is_live = _is_live_source(cam.source)

    if is_live and cam.consecutive_failures <= max_consecutive_failures:
        # Temporary RTSP failure: pad with the last frame so the main loop does not exit
        print(
            f"  [warn] {cam.name} frame read failed ({cam.consecutive_failures} consecutive), "
            f"padding with last frame …"
        )
        if cam.last_good_frame is not None:
            return False, cam.last_good_frame, time.time(), False

    if is_live and cam.consecutive_failures <= max_consecutive_failures + max_reconnect_retries * 3:
        # After consecutive failures pass the threshold, try reconnect
        if _try_reconnect(cam, max_retries=max_reconnect_retries):
            ok2, frame2 = cam.cap.read()
            if ok2 and frame2 is not None:
                cam.consecutive_failures = 0
                cam.last_good_frame = frame2
                return True, frame2, _frame_timestamp_sec(cam), True

    # Hard failure (non-RTSP EOF, or RTSP reconnect exhausted)
    return False, None, 0.0, False


def parse_ultralytics_tracks(
    result: Any,
    model: Any,
) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "id", None) is None:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    ids = boxes.id.detach().cpu().numpy().astype(int)
    confs = (
        boxes.conf.detach().cpu().numpy()
        if getattr(boxes, "conf", None) is not None
        else np.full(len(ids), np.nan, dtype=np.float32)
    )
    clss = (
        boxes.cls.detach().cpu().numpy().astype(int)
        if getattr(boxes, "cls", None) is not None
        else np.full(len(ids), -1, dtype=np.int32)
    )
    names = getattr(model, "names", {}) or {}
    parsed: list[dict[str, Any]] = []
    for box, track_id, conf, cls_id in zip(xyxy, ids, confs, clss):
        class_name = names.get(int(cls_id), str(int(cls_id))) if int(cls_id) >= 0 else None
        parsed.append(
            {
                "xyxy": box.astype(np.float32),
                "track_id": int(track_id),
                "confidence": None if not np.isfinite(conf) else float(conf),
                "class_id": None if int(cls_id) < 0 else int(cls_id),
                "class_name": class_name,
            }
        )
    return parsed


def color_for_id(track_id: int) -> tuple[int, int, int]:
    # BGR; deterministic hue-ring map by ID so colors are evenly spread.
    hue = (track_id * 47) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_camera_tracks(
    frame: np.ndarray,
    camera_name: str,
    detections: list[tuple[dict[str, Any], TrackSample, int]],
    fusion: TrajectoryFusion,
    *,
    rejected_detections: list[tuple[dict[str, Any], str]] | None = None,
    debug: bool = False,
    frame_index: int = 0,
    infer_wh: tuple[int, int] | None = None,
    image_roi: list[list[int]] | None = None,
    global_sync_index: int = 0,
    session_t0: float | None = None,
    timestamp_sec: float = 0.0,
    frame_fresh: bool = True,
) -> np.ndarray:
    out = frame.copy()
    if debug and image_roi and len(image_roi) >= 3:
        roi_pts = np.asarray(image_roi, dtype=np.int32).reshape(-1, 1, 2)
        cv.polylines(out, [roi_pts], True, (255, 200, 0), 1, cv.LINE_AA)

    for det, reason in rejected_detections or []:
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        color = (0, 140, 255)
        cv.rectangle(out, (x1, y1), (x2, y2), color, 2)
        draw_label(out, f"L{det['track_id']} {reason}", (x1, max(y1 - 6, 16)), color)

    for det, sample, global_id in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        local_id = det["track_id"]
        root = fusion.uf.find(global_id)
        color = color_for_id(root)
        if debug and sample.clipped:
            color = (0, 140, 255)  # orange: bbox on the edge; world coords untrustworthy
        cv.rectangle(out, (x1, y1), (x2, y2), color, 2)
        key = (camera_name, local_id)
        track = fusion.local_tracks.get(key)
        if track is not None and len(track.samples) >= 2:
            pts = np.asarray([s.image_xy for s in track.samples], dtype=np.int32).reshape(-1, 1, 2)
            cv.polylines(out, [pts], False, color, 2)
        ix, iy = sample.image_xy
        wx, wy = sample.world_xy
        if debug:
            ax, ay = int(round(ix)), int(round(iy))
            cv.drawMarker(out, (ax, ay), color, cv.MARKER_CROSS, 12, 2)
            cv.circle(out, (ax, ay), 4, color, -1, cv.LINE_AA)
            cls = sample.class_name or "?"
            conf = sample.confidence
            conf_s = f"{conf:.2f}" if conf is not None else "?"
            clip_s = " CLIP" if sample.clipped else ""
            label = (
                f"{camera_name} L{local_id}/G{root} {cls} c={conf_s}{clip_s} "
                f"t={sample.timestamp_sec:.1f}s anchor=({ix:.0f},{iy:.0f}) "
                f"world=({wx:.2f},{wy:.2f})m"
            )
        else:
            label = (
                f"{camera_name} L{local_id}/G{root} "
                f"t={sample.timestamp_sec:.1f}s img=({ix:.0f},{iy:.0f}) "
                f"w=({wx:.2f},{wy:.2f})m"
            )
        draw_label(out, label, (x1, max(18, y1 - 8)), color)

    if debug:
        res = f"{infer_wh[0]}x{infer_wh[1]}" if infer_wh else f"{frame.shape[1]}x{frame.shape[0]}"
        rel = f"{timestamp_sec - session_t0:.2f}s" if session_t0 is not None else "?"
        stale = "" if frame_fresh else " STALE"
        banner = (
            f"[TEST] {camera_name} sync#{global_sync_index} t={rel}{stale} "
            f"{res} local_frame={frame_index}"
        )
        draw_label(out, banner, (8, 22), (200, 255, 200))
        if not frame_fresh:
            draw_label(out, "STALE (repeat frame)", (8, 44), (0, 140, 255))
    return out


def draw_bev_overlay(
    background: np.ndarray,
    fusion: TrajectoryFusion,
    geometry: CanvasGeometry,
    now_sec: float,
    history_sec: float,
    max_points: int = 0,
) -> np.ndarray:
    out = background.copy()
    groups = fusion.grouped_recent_samples(now_sec, history_sec, max_points=max_points)
    for global_id, samples in groups.items():
        if not samples:
            continue
        color = color_for_id(global_id)
        pts = np.asarray(
            [geometry.world_to_canvas(s.world_xy) for s in samples],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        if len(pts) >= 2:
            cv.polylines(out, [pts], False, color, 3)
        x, y = pts[-1, 0]
        cv.circle(out, (int(x), int(y)), 5, color, -1)
        last = samples[-1]
        label = f"G{global_id} t={last.timestamp_sec:.1f}s ({last.world_xy[0]:.2f},{last.world_xy[1]:.2f})m"
        draw_label(out, label, (int(x) + 8, int(y) - 8), color)
    return out


def draw_bev_coverage_outlines(
    background: np.ndarray,
    geometry: CanvasGeometry,
) -> None:
    for name, mask in geometry.coverage_masks.items():
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        color = color_for_id(abs(hash(name)) % 10000)
        cv.drawContours(background, contours, -1, color, 2, cv.LINE_AA)
        contour = max(contours, key=cv.contourArea)
        moments = cv.moments(contour)
        if moments["m00"] <= 1e-6:
            continue
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        short_name = name.replace("s110_camera_basler_", "").replace("_8mm", "")
        draw_label(background, short_name, (cx + 8, cy), color)


def load_bev_background(
    path: str | None,
    geometry: CanvasGeometry,
) -> np.ndarray:
    width, height = geometry.size
    if path:
        image = cv.imread(path, cv.IMREAD_COLOR)
        if image is not None and image.shape[1] == width and image.shape[0] == height:
            return image
        if image is not None:
            print(
                f"  [warn] BEV background size {image.shape[1]}x{image.shape[0]} "
                f"does not match canvas {width}x{height}; using a blank background."
            )
    background = np.full((height, width, 3), 24, dtype=np.uint8)
    for name, mask in geometry.coverage_masks.items():
        color = np.asarray(color_for_id(abs(hash(name)) % 10000), dtype=np.uint8)
        tinted = np.zeros_like(background)
        tinted[:] = color
        alpha = (mask > 0).astype(np.float32) * 0.12
        background = (background * (1.0 - alpha[..., None]) + tinted * alpha[..., None]).astype(np.uint8)
    draw_bev_coverage_outlines(background, geometry)
    return background


def estimate_observation_bytes(row: ObservationRow) -> int:
    text_bytes = len(row.scene_id.encode("utf-8"))
    text_bytes += len(row.batch_id.encode("utf-8"))
    text_bytes += len(row.camera_name.encode("utf-8"))
    text_bytes += len((row.class_name or "").encode("utf-8"))
    # Coarse SQLite row + page/index overhead; kept simple and stable.
    # Extra final_world_x/y columns (nullable REAL, 8 bytes each).
    return text_bytes + 15 * 8 + 4 * 4 + 80


def flush_observations(
    conn: sqlite3.Connection,
    rows: list[ObservationRow],
    fusion: TrajectoryFusion,
) -> tuple[int, int]:
    if not rows:
        return 0, 0
    now = utc_now()
    payload = []
    byte_count = 0
    for row in rows:
        global_id = fusion.uf.find(row.global_id)
        byte_count += estimate_observation_bytes(row)
        fwx = row.final_world_xy[0] if row.final_world_xy is not None else None
        fwy = row.final_world_xy[1] if row.final_world_xy is not None else None
        payload.append(
            (
                row.scene_id,
                row.batch_id,
                row.camera_name,
                row.local_track_id,
                global_id,
                row.frame_index,
                row.timestamp_sec,
                row.image_xy[0],
                row.image_xy[1],
                row.world_xy[0],
                row.world_xy[1],
                fwx,
                fwy,
                row.bbox_xywh[0],
                row.bbox_xywh[1],
                row.bbox_xywh[2],
                row.bbox_xywh[3],
                row.confidence,
                row.class_id,
                row.class_name,
                now,
            )
        )
    conn.executemany(
        """
        INSERT INTO trajectory_observations
            (scene_id, batch_id, camera_name, local_track_id, global_id, frame_index,
             timestamp_sec, image_x, image_y, world_x, world_y,
             final_world_x, final_world_y,
             bbox_x, bbox_y, bbox_w, bbox_h, confidence, class_id, class_name,
             recorded_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        payload,
    )
    conn.commit()
    rows.clear()
    return len(payload), byte_count


def _resume_cameras_from_db(
    conn: sqlite3.Connection,
    scene_id: str,
    batch_id: str,
    cameras: list["CameraRuntime"],
) -> float:
    """Resume support: seek each offline video to the last interrupted **time**.

    **Resume-point selection** (keeps scene time aligned even with mixed FPS):

    1. Prefer ``MAX(timestamp_sec)`` from ``trajectory_runtime_stats`` — shared
       timeline progress;
    2. If stats have no timestamp, estimate ``MAX(frame_index) / sync_fps``;
    3. Else take the minimum of each camera's ``MAX(timestamp_sec)`` in
       ``trajectory_observations``.

    Each camera seeks to ``resume_t * cam.fps`` local frames (mixed 25/50 FPS
    no longer wrongly shares one frame_index). Returns ``resume_t`` for the
    main-loop timeline; ``0.0`` if there is no data.

    Global IDs continue from the DB via ``TrajectoryFusion._load_next_global_id()``
    at construction. YOLO tracker state starts fresh (``local_track_id`` from 1),
    but because resume-segment ``timestamp_sec`` does not overlap existing
    observations, ``mine_mvc_pairs`` strict bracketing (P3) rejects pairs that
    cross the time gap, so the two segments stay semantically independent.
    """
    offline = [c for c in cameras if not _is_live_source(c.source)]
    sync_fps = offline_sync_fps([c.fps for c in offline]) if offline else 25.0

    row_stats = conn.execute(
        """
        SELECT MAX(timestamp_sec), MAX(frame_index)
          FROM trajectory_runtime_stats
         WHERE scene_id = ? AND batch_id = ?
        """,
        (scene_id, batch_id),
    ).fetchone()
    stats_ts = float(row_stats[0]) if row_stats and row_stats[0] is not None else None
    stats_frame = int(row_stats[1]) if row_stats and row_stats[1] is not None else None

    obs_max_ts: dict[str, float] = {}
    for cam in offline:
        r = conn.execute(
            """
            SELECT MAX(timestamp_sec)
              FROM trajectory_observations
             WHERE scene_id = ? AND batch_id = ? AND camera_name = ?
            """,
            (scene_id, batch_id, cam.name),
        ).fetchone()
        if r and r[0] is not None:
            obs_max_ts[cam.name] = float(r[0])

    if stats_ts is not None and math.isfinite(stats_ts):
        resume_t = max(0.0, stats_ts)
        source = f"trajectory_runtime_stats timestamp={stats_ts:.3f}s"
    elif stats_frame is not None:
        resume_t = max(0.0, float(stats_frame) / max(sync_fps, 1e-6))
        source = (
            f"trajectory_runtime_stats frame={stats_frame} "
            f"→ t≈{resume_t:.3f}s @ sync_fps={sync_fps:.3f}"
        )
    elif obs_max_ts:
        resume_t = max(0.0, min(obs_max_ts.values()))
        source = (
            f"trajectory_observations MIN(MAX timestamp)={resume_t:.3f}s "
            f"({', '.join(f'{k}:{v:.2f}s' for k, v in sorted(obs_max_ts.items()))})"
        )
    else:
        print("  [resume] no existing data in DB; all cameras start from the beginning")
        return 0.0

    # Nudge forward by half a sync step to reduce duplicate writes at the boundary
    resume_t = resume_t + 0.5 / max(sync_fps, 1e-6)
    print(f"  [resume] unified resume point: t={resume_t:.3f}s (source: {source})")

    for cam in offline:
        local_frame = int(round(resume_t * max(cam.fps, 1e-6)))
        ok = cam.cap.set(cv.CAP_PROP_POS_FRAMES, local_frame)
        cam.first_frame = None  # drop open_cameras prefetch; read from the resume point
        if not ok:
            # Some decoders ignore POS_FRAMES; try millisecond seek
            ok = bool(cam.cap.set(cv.CAP_PROP_POS_MSEC, resume_t * 1000.0))
        if not ok:
            print(f"  [resume] {cam.name}: seek to t={resume_t:.3f}s failed; starting from the beginning")
            continue
        cam.frame_index = local_frame
        cam.last_timestamp_sec = resume_t
        actual_ts = float(cam.cap.get(cv.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        print(
            f"  [resume] {cam.name}: local_frame≈{local_frame} "
            f"(fps={cam.fps:.2f}), container time ≈ {actual_ts:.2f}s"
        )
    return resume_t


def flush_camera_configs(
    conn: sqlite3.Connection,
    scene_id: str,
    batch_id: str,
    cameras: list["CameraRuntime"],
) -> None:
    """Write each camera's infer resolution into trajectory_camera_config.

    Called once right after tracking starts so geometry_guided_residual_mlp
    training and --remerge can read the correct infer resolution from the DB,
    keeping clip-boundary filtering and H scaling consistent.
    """
    now = utc_now()
    for cam in cameras:
        if cam.infer_wh is None:
            continue
        iw, ih = cam.infer_wh
        cw = cam.calib_wh[0] if cam.calib_wh else None
        ch = cam.calib_wh[1] if cam.calib_wh else None
        conn.execute(
            """
            INSERT INTO trajectory_camera_config
                (scene_id, batch_id, camera_name, infer_w, infer_h, calib_w, calib_h, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scene_id, batch_id, camera_name) DO UPDATE
                SET infer_w=excluded.infer_w,
                    infer_h=excluded.infer_h,
                    calib_w=excluded.calib_w,
                    calib_h=excluded.calib_h,
                    updated_at=excluded.updated_at
            """,
            (scene_id, batch_id, cam.name, iw, ih, cw, ch, now),
        )
    conn.commit()


def save_runtime_stats(
    conn: sqlite3.Connection,
    scene_id: str,
    batch_id: str,
    frame_index: int,
    timestamp_sec: float,
    active_local_tracks: int,
    active_global_tracks: int,
    comparisons: int,
    history_samples: int,
    rows_written: int,
    rows_per_sec: float,
    bytes_per_sec: float,
) -> None:
    """Insert one runtime-stats row; the caller commits."""
    complexity = f"O(P*T), P={comparisons}, T<={history_samples}"
    conn.execute(
        """
        INSERT INTO trajectory_runtime_stats
            (scene_id, batch_id, frame_index, timestamp_sec, active_local_tracks,
             active_global_tracks, merge_candidate_comparisons, merge_complexity,
             db_rows_written, storage_rows_per_sec, storage_bytes_per_sec, recorded_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            scene_id,
            batch_id,
            frame_index,
            timestamp_sec,
            active_local_tracks,
            active_global_tracks,
            comparisons,
            complexity,
            rows_written,
            rows_per_sec,
            bytes_per_sec,
            utc_now(),
        ),
    )


def apply_test_mode(args: argparse.Namespace) -> Path | None:
    """Enable test mode: auto-set debug video/screenshot output dirs (do not override paths the user already set)."""
    if not args.test:
        return None
    if args.test_dir:
        out_dir = Path(args.test_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = _PROJECT_ROOT / "outputs" / "debug_tracking" / f"{args.scene}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.save_camera_dir:
        args.save_camera_dir = str(out_dir)
    if not args.save_bev_video:
        args.save_bev_video = str(out_dir / "bev_global_tracks.mp4")
    if not args.save_bev_image:
        args.save_bev_image = str(out_dir / "bev_last_frame.png")
    return out_dir


def write_test_manifest(
    path: Path,
    payload: dict[str, Any],
    *,
    merge: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if merge and path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing.update(payload)
        payload = existing
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_test_manifest(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    cameras: list[CameraRuntime],
    test_dir: Path,
) -> dict[str, Any]:
    cam_entries = []
    for cam in cameras:
        roi = cam.effective_cfg.get("image_roi")
        cam_entries.append(
            {
                "name": cam.name,
                "source": str(cam.source),
                "infer_wh": list(cam.infer_wh) if cam.infer_wh else None,
                "calib_wh": list(cam.calib_wh) if cam.calib_wh else None,
                "fps": cam.fps,
                "has_sub_stream": cam.calib_wh is not None and cam.infer_wh is not None
                and cam.calib_wh != cam.infer_wh,
                "image_roi_points": len(roi) if roi else 0,
                "video": f"{cam.name}_substream_tracks.mp4",
            }
        )
    return {
        "mode": "test",
        "started_at": utc_now(),
        "scene": args.scene,
        "config": str(Path(args.config).resolve()),
        "db": str(Path(args.db).resolve()),
        "output_dir": str(test_dir.resolve()),
        "model": args.model,
        "tracker": args.tracker,
        "classes": args.classes,
        "anchor": args.anchor,
        "risk_scalar": args.risk_scalar,
        "anchor_s0_m": args.anchor_s0_m,
        "anchor_shape_eta": args.anchor_shape_eta,
        "anchor_aspect_ratio_min": args.anchor_aspect_ratio_min,
        "anchor_aspect_ratio_max": args.anchor_aspect_ratio_max,
        "pixel_grazing_d0": args.pixel_grazing_d0,
        "metric_sensitivity_clip_m": args.metric_sensitivity_clip_m,
        "g2_checkpoint": getattr(args, "g2_checkpoint", None),
        "g2_online_learn": bool(getattr(args, "g2_online_learn", False)),
        "clip_margin": args.clip_margin,
        "clip_margin_ratio": args.clip_margin_ratio,
        "min_valid_streak": args.min_valid_streak,
        "clahe": bool(getattr(args, "clahe", False)),
        "clahe_clip": getattr(args, "clahe_clip", None),
        "cameras": cam_entries,
        "outputs": {
            "camera_videos": [e["video"] for e in cam_entries],
            "bev_video": Path(args.save_bev_video).name if args.save_bev_video else None,
            "bev_image": Path(args.save_bev_image).name if args.save_bev_image else None,
        },
        "notes": (
            "camera_videos overlay local tracks on the raw sub-stream (or main-stream) frames used by YOLO; "
            "orange EDGE/WAIT boxes show the bbox only and do not create anchors, world coords, or DB observations."
        ),
    }


def _merge_config_section(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("trajectory_merge", config.get("merge", {}))
    return section if isinstance(section, dict) else {}


def _resolve_config_value(
    cli_value: Any,
    default_value: Any,
    section: dict[str, Any],
    key: str,
    cast: Any,
) -> Any:
    if key in section and cli_value == default_value:
        return cast(section[key])
    return cast(cli_value)


def _load_tracker_buffer_frames(tracker: str | None, fallback: int = 30) -> int:
    """Read Ultralytics tracker YAML track_buffer; use a safe default if parsing fails."""
    if not tracker:
        return fallback
    raw = Path(str(tracker))
    candidates = [resolve_project_path(raw)]
    try:
        import ultralytics

        candidates.append(
            Path(ultralytics.__file__).resolve().parent
            / "cfg"
            / "trackers"
            / raw.name
        )
    except ImportError:
        pass
    for path in candidates:
        if not path.is_file():
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
            value = int(payload.get("track_buffer", fallback))
            if value > 0:
                return value
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            continue
    return fallback


def build_match_config(args: argparse.Namespace, merge_section: dict[str, Any]) -> MatchConfig:
    """Build MatchConfig from CLI flags + YAML trajectory_merge (explicit CLI values win)."""
    d = DEFAULT_MATCH_CONFIG

    def rv(cli_value: Any, key: str, cast: Any) -> Any:
        return _resolve_config_value(cli_value, getattr(d, key), merge_section, key, cast)

    if args.stitch_tracker_buffer_frames > 0:
        tracker_buffer_frames = int(args.stitch_tracker_buffer_frames)
    elif "tracker_buffer_frames" in merge_section:
        tracker_buffer_frames = int(merge_section["tracker_buffer_frames"])
    else:
        tracker_buffer_frames = _load_tracker_buffer_frames(
            getattr(args, "tracker", None),
            d.tracker_buffer_frames,
        )

    return MatchConfig(
        score_threshold=rv(args.merge_score_threshold, "score_threshold", float),
        max_time_gap_sec=_resolve_config_value(
            args.merge_max_time_gap, d.max_time_gap_sec,
            merge_section, "max_time_gap_sec", float,
        ),
        max_distance_m=_resolve_config_value(
            args.merge_max_distance, d.max_distance_m,
            merge_section, "max_distance_m", float,
        ),
        min_samples=_resolve_config_value(
            args.merge_min_samples, d.min_samples, merge_section, "min_samples", int,
        ),
        sample_count=_resolve_config_value(
            args.merge_sample_count, d.sample_count, merge_section, "sample_count", int,
        ),
        active_timeout_sec=args.active_timeout,
        time_sigma=_resolve_config_value(
            args.merge_time_sigma, d.time_sigma, merge_section, "time_sigma", float,
        ),
        distance_sigma=_resolve_config_value(
            args.merge_distance_sigma, d.distance_sigma,
            merge_section, "distance_sigma", float,
        ),
        shape_sigma=_resolve_config_value(
            args.merge_shape_sigma, d.shape_sigma, merge_section, "shape_sigma", float,
        ),
        time_weight=_resolve_config_value(
            args.merge_time_weight, d.time_weight, merge_section, "time_weight", float,
        ),
        distance_weight=_resolve_config_value(
            args.merge_distance_weight, d.distance_weight,
            merge_section, "distance_weight", float,
        ),
        shape_weight=_resolve_config_value(
            args.merge_shape_weight, d.shape_weight, merge_section, "shape_weight", float,
        ),
        stitch_enabled=(
            bool(merge_section.get("stitch_enabled", d.stitch_enabled))
            if args.stitch_enabled is None
            else bool(args.stitch_enabled)
        ),
        stitch_finalize_sec=max(
            rv(args.stitch_finalize_sec, "stitch_finalize_sec", float),
            0.0,
        ),
        tracker_buffer_frames=tracker_buffer_frames,
        stitch_finalize_margin_sec=rv(
            args.stitch_finalize_margin,
            "stitch_finalize_margin_sec",
            float,
        ),
        stitch_default_sample_period_sec=rv(
            args.stitch_default_sample_period,
            "stitch_default_sample_period_sec",
            float,
        ),
        stitch_max_gap_sec=rv(args.stitch_max_gap, "stitch_max_gap_sec", float),
        stitch_max_overlap_sec=rv(args.stitch_max_overlap, "stitch_max_overlap_sec", float),
        stitch_base_tolerance_m=rv(args.stitch_base_tolerance, "stitch_base_tolerance_m", float),
        stitch_tolerance_rate_mps=rv(
            args.stitch_tolerance_rate, "stitch_tolerance_rate_mps", float
        ),
        stitch_min_samples=rv(args.stitch_min_samples, "stitch_min_samples", int),
        stitch_direction_min_cos=rv(
            args.stitch_direction_min_cos, "stitch_direction_min_cos", float
        ),
        stitch_min_speed_mps=rv(args.stitch_min_speed, "stitch_min_speed_mps", float),
        stitch_speed_sigma_mps=rv(
            args.stitch_speed_sigma, "stitch_speed_sigma_mps", float
        ),
        stitch_score_threshold=rv(
            args.stitch_score_threshold, "stitch_score_threshold", float
        ),
        stitch_uncertainty_floor_m=rv(
            args.stitch_uncertainty_floor,
            "stitch_uncertainty_floor_m",
            float,
        ),
        stitch_process_noise_mps2=rv(
            args.stitch_process_noise,
            "stitch_process_noise_mps2",
            float,
        ),
        stitch_max_normalized_error=rv(
            args.stitch_max_normalized_error,
            "stitch_max_normalized_error",
            float,
        ),
        stitch_max_uncertainty_ratio=rv(
            args.stitch_max_uncertainty_ratio,
            "stitch_max_uncertainty_ratio",
            float,
        ),
        stitch_require_mutual_best=(
            bool(merge_section.get("stitch_require_mutual_best", d.stitch_require_mutual_best))
            if args.stitch_require_mutual_best is None
            else bool(args.stitch_require_mutual_best)
        ),
        stitch_min_score_margin=rv(
            args.stitch_min_score_margin,
            "stitch_min_score_margin",
            float,
        ),
        guard_sync_tolerance_sec=rv(
            args.guard_sync_tolerance, "guard_sync_tolerance_sec", float
        ),
        guard_sync_period_factor=rv(
            args.guard_sync_period_factor, "guard_sync_period_factor", float
        ),
        guard_max_separation_m=rv(
            args.guard_max_separation, "guard_max_separation_m", float
        ),
        guard_position_sigma_floor_m=rv(
            args.guard_position_sigma_floor,
            "guard_position_sigma_floor_m",
            float,
        ),
        guard_position_sigma_cap_m=rv(
            args.guard_position_sigma_cap,
            "guard_position_sigma_cap_m",
            float,
        ),
        guard_max_normalized_separation=rv(
            args.guard_max_normalized_separation,
            "guard_max_normalized_separation",
            float,
        ),
        guard_min_conflicting_samples=rv(
            args.guard_min_conflicting_samples,
            "guard_min_conflicting_samples",
            int,
        ),
    )


# OpenCV mp4v / MPEG-4 Part 2 per-dimension cap is about 4096px; extra-wide BEV canvases must be scaled to an encodable size.
_MAX_OPENCV_VIDEO_DIM = 4096


def _fit_opencv_video_size(
    frame_size: tuple[int, int], max_dim: int = _MAX_OPENCV_VIDEO_DIM
) -> tuple[tuple[int, int], tuple[int, int]]:
    src_w, src_h = frame_size
    if src_w <= max_dim and src_h <= max_dim:
        return frame_size, frame_size
    scale = min(max_dim / src_w, max_dim / src_h)
    dst_w = max(2, int(round(src_w * scale)))
    dst_h = max(2, int(round(src_h * scale)))
    if dst_w % 2:
        dst_w -= 1
    if dst_h % 2:
        dst_h -= 1
    return (dst_w, dst_h), frame_size


class VideoRecorder:
    """Wrap cv.VideoWriter; shrink source frames that exceed the encoder size cap."""

    def __init__(self, writer: cv.VideoWriter, output_size: tuple[int, int]) -> None:
        self._writer = writer
        self._output_size = output_size

    def write(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        out_w, out_h = self._output_size
        if (w, h) != (out_w, out_h):
            frame = cv.resize(frame, (out_w, out_h), interpolation=cv.INTER_AREA)
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()


def make_video_writer(path: str, fps: float, frame_size: tuple[int, int]) -> VideoRecorder:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_size, src_size = _fit_opencv_video_size(frame_size)
    if video_size != src_size:
        print(
            f"  [record] {output_path.name}: canvas {src_size[0]}×{src_size[1]} "
            f"exceeds encoder cap; output scaled to {video_size[0]}×{video_size[1]}"
        )
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(str(output_path), fourcc, fps, video_size)
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {path}")
    return VideoRecorder(writer, video_size)


def rewrite_video_playback_fps(path: str, fps: float) -> bool:
    """Reset MP4 playback FPS from actual wall-clock duration so multi-camera debug videos have matching length."""
    src = Path(path)
    if not src.is_file():
        return False
    fps = max(1.0, min(60.0, float(fps)))
    cap = cv.VideoCapture(str(src))
    if not cap.isOpened():
        return False
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    tmp = src.with_suffix(".tmp.mp4")
    writer = make_video_writer(str(tmp), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    tmp.replace(src)
    return True


# Concurrent inference threads on one GPU still serialize kernels at the CUDA
# scheduler; lock per device so Ultralytics predictor tensor state does not race.
# CPU inference (device=cpu) has no such limit and can run fully in parallel.
_DEVICE_INFERENCE_LOCKS: dict[str, threading.Lock] = defaultdict(threading.Lock)


def poll_view_quit(enabled: bool, delay_ms: int = 30) -> bool:
    """Handle OpenCV window events; return True on q or Esc.

    Inference may block for seconds; waitKey must still be called regularly
    or on Windows the window will not refresh and keys will not work.
    """
    if not enabled:
        return False
    key = cv.waitKey(max(delay_ms, 1)) & 0xFF
    return key in (ord("q"), 27)


class ShutdownController:
    """Unified safe shutdown for windowed and headless modes (finish cleanup, then exit)."""

    def __init__(self) -> None:
        self.requested = threading.Event()
        self._sigint_count = 0
        self._stdin_stop = threading.Event()
        self._stdin_thread: threading.Thread | None = None

    def request(self, reason: str) -> None:
        if not self.requested.is_set():
            print(f"  [exit] {reason}")
        self.requested.set()

    def poll(self) -> bool:
        return self.requested.is_set()

    def install_signal_handlers(self) -> None:
        def handler(signum: int, frame: Any) -> None:
            del signum, frame
            self._sigint_count += 1
            if self._sigint_count >= 2:
                print("  [exit] interrupted again; forcing exit.")
                raise SystemExit(1)
            self.request("stop signal received (Ctrl+C); shutting down safely …")

        signal.signal(signal.SIGINT, handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, handler)

    def start_stdin_listener(self) -> None:
        """Headless mode: type q / quit / exit and Enter in this terminal to exit."""

        def loop() -> None:
            while not self._stdin_stop.is_set():
                try:
                    line = sys.stdin.readline()
                except Exception:
                    break
                if self._stdin_stop.is_set():
                    break
                if line.strip().lower() in ("q", "quit", "exit"):
                    self.request("terminal q; shutting down safely …")
                    break

        self._stdin_thread = threading.Thread(
            target=loop, name="stdin-quit", daemon=True
        )
        self._stdin_thread.start()

    def stop_stdin_listener(self) -> None:
        self._stdin_stop.set()


def should_quit(view: bool, shutdown: ShutdownController) -> bool:
    if shutdown.poll():
        return True
    return poll_view_quit(view)


def _run_camera_inference(
    cam: CameraRuntime,
    backend: Any,
    args: argparse.Namespace,
    classes: list[int] | None,
    max_reconnect_retries: int = 3,
    *,
    target_timestamp_sec: float | None = None,
    sync_fps: float | None = None,
) -> tuple[CameraRuntime, np.ndarray | None, float, list[dict[str, Any]], bool]:
    """Read the next frame on one camera and run the detect/track backend.

    Called from ThreadPoolExecutor: each camera has its own backend (tracker
    state is isolated). Frame I/O and CPU pre/post can run across cameras;
    GPU inference on the same device is serialized by
    _DEVICE_INFERENCE_LOCKS.

    ``target_timestamp_sec`` / ``sync_fps``: passed for offline multi-camera
    timeline alignment; a higher-rate source catches up to the target within
    one beat. Live streams ignore these and still use ``read_next_frame``.

    Read outcomes:
      · (True, frame, t)        normal frame, continue inference
      · (False, last_frame, t)  temporary RTSP pad; still infer (tracker stays chained)
      · (False, None, 0)        hard failure; return None so the main loop drops this camera
    """
    if (
        target_timestamp_sec is not None
        and sync_fps is not None
        and not _is_live_source(cam.source)
    ):
        ok, frame, timestamp_sec, frame_fresh = read_frame_for_timeline(
            cam,
            float(target_timestamp_sec),
            sync_fps=float(sync_fps),
            max_reconnect_retries=max_reconnect_retries,
        )
    else:
        ok, frame, timestamp_sec, frame_fresh = read_next_frame(
            cam, max_reconnect_retries=max_reconnect_retries
        )
    if frame is None:
        # Hard failure (ok=False and frame=None)
        return cam, None, 0.0, [], False

    # ok=False but frame is a pad: still infer so the tracker does not break
    device_key = str(args.device or "default")
    lock = _DEVICE_INFERENCE_LOCKS[device_key]

    infer_frame = frame
    if getattr(args, "clahe", False):
        infer_frame = apply_clahe_bgr(frame, args.clahe_clip)

    with lock:
        t_track = time.perf_counter()
        detections = backend.track_frame(
            infer_frame,
            conf=args.conf,
            iou=args.iou,
            classes=classes,
            imgsz=args.imgsz,
            device=args.device,
            tracker=args.tracker,
        )
        cam.last_track_ms = (time.perf_counter() - t_track) * 1000.0
    return cam, frame, timestamp_sec, detections, frame_fresh


def _configure_g2_residual_invoke(controller: Any, args: argparse.Namespace) -> None:
    acceptor = getattr(args, "g2_acceptor_json", None)
    force_all = bool(getattr(args, "g2_force_all_residual", False))
    if force_all and acceptor:
        raise ValueError(
            "pass only one of --g2-acceptor-json / --g2-force-all-residual"
        )
    if acceptor:
        acceptor = str(resolve_project_path(acceptor))
    controller.configure_residual_invoke(
        acceptor_json=acceptor,
        force_all=force_all,
    )
    allow = controller.residual_cameras
    cams = "ALL" if allow is None else (",".join(sorted(allow)) or "(none)")
    print(
        f"  [g2] residual invoke={controller.residual_invoke_source}  "
        f"dP cameras={cams}"
    )


def _create_g2_controller(
    args: argparse.Namespace,
    overlap_pairs: set[tuple[str, str]],
) -> Any | None:
    """Load the g6 MVC residual controller on demand (None without --g2-checkpoint)."""
    if not getattr(args, "g2_checkpoint", None):
        return None
    try:
        from pipeline.geometry_guided_residual_mlp import (
            MVCResidualController,
            OnlineLearnConfig,
            TrainConfig,
            load_camera_translation_corrections,
        )
    except ImportError as exc:
        raise RuntimeError(
            "g2 requires PyTorch: pip install torch"
        ) from exc

    ckpt = Path(args.g2_checkpoint)
    runtime_corrections = load_camera_translation_corrections(
        getattr(args, "g2_camera_translation_json", None),
        acceptance=getattr(args, "g2_camera_translation_acceptance", "recommended"),
    )
    online_cfg = OnlineLearnConfig(
        enabled=bool(args.g2_online_learn),
        every_frames=int(args.g2_learn_every_frames),
        epochs=int(args.g2_learn_epochs),
        min_pairs=int(args.g2_learn_min_pairs),
        lr_scale=float(args.g2_learn_lr_scale),
        max_time_gap_sec=float(args.g2_learn_max_time_gap),
        max_pairs_per_learn=int(args.g2_learn_max_pairs),
    )
    if not ckpt.is_file():
        if not online_cfg.enabled:
            raise FileNotFoundError(
                f"g2 checkpoint not found: {ckpt}. "
                "Inference-only mode needs a trained .pt; to start self-training from zero residual, add --g2-online-learn."
            )
        print(
            f"  [g2] checkpoint not found; starting online self-training from a zero-residual MLP: {ckpt.resolve()}"
        )
    # Geometry/feature knobs apply only when the checkpoint is missing (online
    # self-training from zero residual). Once a checkpoint exists,
    # MVCResidualController forces geometry from checkpoint meta so P_geo/F
    # meaning stays aligned with the trained weights (see that class __init__).
    train_cfg = TrainConfig(
        loss_mode=str(args.g2_loss_mode),
        min_teacher_confidence=float(args.g2_min_teacher_confidence),
        min_teacher_confidence_gap=float(args.g2_min_teacher_confidence_gap),
        teacher_confidence_power=float(args.g2_teacher_confidence_power),
        student_weight_floor=float(args.g2_student_weight_floor),
        risk_scalar_mode=str(args.g2_risk_scalar),
        s0_m=float(args.g2_anchor_s0_m),
        feature_l0_m=float(args.g2_feature_l0_m),
        anchor_shape_eta=float(args.g2_anchor_shape_eta),
        anchor_aspect_ratio_min=float(args.g2_anchor_aspect_ratio_min),
        anchor_aspect_ratio_max=float(args.g2_anchor_aspect_ratio_max),
        pixel_grazing_d0=float(args.g2_pixel_grazing_d0),
        sensitivity_clip_m=args.g2_metric_sensitivity_clip_m,
        extent_clip_m=args.g2_extent_clip_m,
        motion_compensation=str(args.g2_motion_compensation),
        time_offsets_sec=(
            json.loads(args.g2_time_offsets_json) if args.g2_time_offsets_json else {}
        ),
    )
    online_checkpoint_out: Path | None = None
    if online_cfg.enabled:
        online_checkpoint_out = (
            Path(args.g2_online_checkpoint_out)
            if args.g2_online_checkpoint_out
            else ckpt.with_name(f"{ckpt.stem}.online{ckpt.suffix}")
        )
    controller = MVCResidualController(
        train_cfg=train_cfg,
        checkpoint_path=ckpt,
        save_checkpoint_path=online_checkpoint_out,
        online_cfg=online_cfg,
        overlap_pairs=overlap_pairs,
        camera_translation_corrections=runtime_corrections,
    )
    _configure_g2_residual_invoke(controller, args)
    mode = "infer+online-learn" if online_cfg.enabled else "infer-only"
    print(
        f"  [g2] MVC residual localization enabled ({mode}), checkpoint={ckpt.resolve()}  "
        f"loss={controller.train_cfg.loss_mode}"
    )
    if getattr(controller, "camera_translation_corrections", None):
        print(
            "  [g2] runtime fixed camera translations loaded: "
            f"{controller.camera_translation_corrections.camera_count()} camera(s), "
            f"{controller.camera_translation_corrections.scene_count()} scene map(s)"
        )
    elif getattr(controller, "training_camera_translation_corrections", None):
        print(
            "  [g2][note] checkpoint recorded a training-time camera translation, "
            "but runtime T_cam is off by default; to include that low-order "
            "correction in final_world_x/y, pass --g2-camera-translation-json."
        )
    if online_cfg.enabled:
        print(
            f"  [g2] online learn every {online_cfg.every_frames} global frames, "
            f"min {online_cfg.min_pairs} pairs, max {online_cfg.max_pairs_per_learn} pairs, "
            f"epochs={online_cfg.epochs}, "
            f"|κA-κB|>={train_cfg.min_teacher_confidence_gap:.3f}, "
            f"out={online_checkpoint_out.resolve() if online_checkpoint_out else '-'}"
        )
    return controller


def _normalize_tracking_args(args: argparse.Namespace) -> None:
    """Resolve a CLI relative path against the project root."""
    args.config = str(resolve_project_path(args.config))
    args.db = str(resolve_project_path(args.db))
    args.model = str(resolve_project_path(args.model))
    for attr in (
        "bev_background",
        "save_camera_dir",
        "save_bev_video",
        "save_bev_image",
        "g2_checkpoint",
        "g2_camera_translation_json",
        "g2_acceptor_json",
        "g2_online_checkpoint_out",
        "test_dir",
    ):
        value = getattr(args, attr, None)
        if value:
            setattr(args, attr, str(resolve_project_path(value)))


def run_tracking(args: argparse.Namespace) -> None:
    _normalize_tracking_args(args)
    test_dir = apply_test_mode(args)
    if test_dir is not None:
        print(f"  [TEST] debug output dir: {test_dir.resolve()}")
    if args.view or args.test or args.save_camera_dir:
        warn_if_no_pil("track overlay")

    use_async_reader = False
    record_videos = False
    session_t0 = time.time()
    bev_writer_path: str | None = None

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    source_overrides = parse_sources(args.source)
    # --sub-source A=rtsp://... is stored as key "sub:A"; open_cameras recognizes it
    for item in args.sub_source or []:
        if "=" not in item:
            raise ValueError("--sub-source format must be NAME=URL, e.g. A=rtsp://192.168.1.1/sub")
        name, raw = item.split("=", 1)
        source_overrides[f"sub:{name.strip()}"] = parse_source(raw)
    # With g2 on, calibration version must match training so a cla1/earlier
    # homography cannot be used silently with a fab1 checkpoint
    # (geometry_version gate; see multi_camera_bev_stitch.py).
    _g2_active = bool(getattr(args, "g2_checkpoint", None))
    cameras = open_cameras(
        cfg, args.db, args.scene, source_overrides,
        rtsp_transport=args.rtsp_transport,
        require_geometry_version=GEOMETRY_VERSION if _g2_active else None,
    )
    if not cameras:
        raise RuntimeError("no cameras in the config")
    if args.clahe:
        print(
            f"  [CLAHE] enabled (clip={args.clahe_clip}, LAB luminance); "
            "applied to YOLO only; BEV/debug video still use the original frame."
        )

    test_manifest_path: Path | None = None
    if test_dir is not None:
        test_manifest_path = test_dir / "run_manifest.json"
        write_test_manifest(test_manifest_path, build_test_manifest(args, cfg, cameras, test_dir))

    geometry = build_canvas_geometry(cameras, cfg)
    print(
        f"  [BEV] canvas={geometry.size[0]}x{geometry.size[1]}, "
        f"scale={geometry.scale:.2f}px/m, overlap_pairs={sorted(geometry.overlap_pairs)}"
    )

    background_path = args.bev_background or cfg.get("output")
    bev_background = load_bev_background(background_path, geometry)

    conn = init_trajectory_db(args.db, args.batch)
    existing_batch_rows = int(
        conn.execute(
            "SELECT COUNT(*) FROM trajectory_observations "
            "WHERE scene_id=? AND batch_id=?",
            (args.scene, args.batch),
        ).fetchone()[0]
    )
    if existing_batch_rows and not getattr(args, "resume", False):
        conn.close()
        raise SystemExit(
            f"scene={args.scene!r} batch={args.batch!r} already has {existing_batch_rows} observations."
            "To avoid duplicate writes, use a new --batch; add --resume explicitly to continue from an interrupted point."
        )
    stage_run_id = start_batch_stage(
        conn,
        [args.scene],
        args.batch,
        "tracking",
        {
            "config": args.config,
            "model": args.model,
            "detect_backend": getattr(args, "detect_backend", "ultralytics"),
            "resume": bool(getattr(args, "resume", False)),
            "g2_checkpoint": getattr(args, "g2_checkpoint", None),
            "g2_camera_translation_json": getattr(
                args, "g2_camera_translation_json", None
            ),
        },
    )
    # Persist canvas geometry in the DB so later replay can convert world coords back to pixels
    conn.execute(
        "INSERT OR REPLACE INTO scene_canvas "
        "(scene_id, scale, min_x, min_y, canvas_w, canvas_h, saved_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (
            args.scene,
            float(geometry.scale),
            float(geometry.min_xy[0]),
            float(geometry.min_xy[1]),
            geometry.size[0],
            geometry.size[1],
            utc_now(),
        ),
    )
    save_scene_overlap_pairs(conn, args.scene, geometry.overlap_pairs, replace=True)
    conn.commit()
    flush_camera_configs(conn, args.scene, args.batch, cameras)
    merge_section = _merge_config_section(cfg)
    match_cfg = build_match_config(args, merge_section)
    print(
        "  [track] merge_cfg "
        f"score>={match_cfg.score_threshold:.2f}, dist<={match_cfg.max_distance_m:.2f}m, "
        f"sigma(d/t/shape)=({match_cfg.distance_sigma:.2f}/"
        f"{match_cfg.time_sigma:.2f}/{match_cfg.shape_sigma:.2f})"
    )
    if match_cfg.stitch_enabled:
        print(f"  [track] same-camera stitch ON  {_linking_config_summary(match_cfg)}")
    else:
        print("  [track] same-camera stitch OFF (default/config)")
    fusion = TrajectoryFusion(
        conn=conn,
        scene_id=args.scene,
        batch_id=args.batch,
        geometry=geometry,
        match_cfg=match_cfg,
        max_history=args.history_samples,
    )
    stage_sink = None
    if getattr(args, "stage_profile", None):
        stage_sink = _load_orin_stage_profile().StageProfileSink(args.stage_profile)
        print(f"  [stage-profile] {args.stage_profile}")
    g2_controller = _create_g2_controller(args, geometry.overlap_pairs)
    if g2_controller is not None:
        from pipeline.geometry_guided_residual_mlp import RuntimeObservation
    else:
        RuntimeObservation = None  # type: ignore[misc, assignment]

    backend_name = str(getattr(args, "detect_backend", "ultralytics") or "ultralytics")
    print(
        f"  [model] backend={backend_name} loading {args.model}, "
        f"preparing independent trackers for {len(cameras)} cameras …"
    )
    models = load_track_backends(args, [cam.name for cam in cameras])
    yolo_names = backend_class_names(models)
    first_backend = next(iter(models.values()))
    # Synthehicle MMDet YOLOX is a single-class head: COCO id filtering is meaningless; disable it.
    if (
        getattr(first_backend, "backend_name", "") == "mmdet_yolox"
        and getattr(getattr(first_backend, "detector", None), "num_classes", 0) == 1
    ):
        classes = None
        print(
            "  [class] mmdet_yolox single-class model: ignoring --classes COCO filter, "
            f"using {yolo_names}"
        )
    else:
        classes = parse_classes(args.classes, yolo_names)
        if classes is not None:
            print(f"  [class] filtering class_id={classes}")
        else:
            print("  [class] no filter (track all classes)")
    pending_rows: list[ObservationRow] = []
    valid_detection_streaks: dict[tuple[str, int], int] = {}
    rows_written_window = 0
    bytes_written_window = 0
    stats_window_start = time.perf_counter()
    latest_timestamp = 0.0
    last_bev_frame = bev_background.copy()
    current_comparisons = 0

    # ── Resume: seek each video to the last interrupted shared time ──
    timeline_origin_sec = 0.0
    if getattr(args, "resume", False):
        timeline_origin_sec = _resume_cameras_from_db(
            conn, args.scene, args.batch, cameras
        )

    record_videos = bool(args.save_camera_dir or args.save_bev_video)
    use_async_reader = record_videos and any(_is_live_source(c.source) for c in cameras)
    session_t0 = time.time()
    if use_async_reader:
        start_async_readers(cameras)

    # Offline cameras catch up on a shared timeline (mixed 25/50 FPS); live streams stay wall-clock.
    live_record_fps_placeholder = 10.0
    has_live_sources = any(_is_live_source(cam.source) for cam in cameras)
    offline_cameras = [cam for cam in cameras if not _is_live_source(cam.source)]
    offline_fps_values = [cam.fps for cam in offline_cameras]
    # Mixed live streams use a different time base (wall-clock vs relative seconds); drop shared timeline and step frame-by-frame.
    use_offline_timeline = bool(offline_cameras) and not has_live_sources
    sync_fps = offline_sync_fps(offline_fps_values) if use_offline_timeline else None
    if use_offline_timeline and sync_fps is not None:
        mismatch = offline_fps_mismatch(offline_fps_values)
        fps_note = ", ".join(f"{c.name}={c.fps:.2f}" for c in offline_cameras)
        print(
            f"  [timeline] offline shared beat sync_fps={sync_fps:.3f} "
            f"({'FPS mismatch, catch up by time' if mismatch else 'FPS close, still advance on the timeline'}); "
            f"source FPS: {fps_note}"
        )
    bev_record_fps = (
        live_record_fps_placeholder
        if has_live_sources
        else float(sync_fps or (np.median(offline_fps_values or [25.0])))
    )

    bev_writer: VideoRecorder | None = None
    bev_writer_path: str | None = None
    if args.save_bev_video:
        bev_writer_path = args.save_bev_video
        bev_writer = make_video_writer(
            bev_writer_path, bev_record_fps, geometry.size
        )

    if args.save_camera_dir:
        camera_dir = Path(args.save_camera_dir)
        camera_dir.mkdir(parents=True, exist_ok=True)
        video_suffix = "_substream_tracks.mp4" if args.test else "_tracks.mp4"
        for cam in cameras:
            if cam.first_frame is not None:
                h, w = cam.first_frame.shape[:2]
            elif cam.latest_frame is not None:
                h, w = cam.latest_frame.shape[:2]
            elif cam.infer_wh is not None:
                w, h = cam.infer_wh
            else:
                raise RuntimeError(f"camera '{cam.name}': no usable frame; cannot determine debug-video size")
            out_path = str(camera_dir / f"{cam.name}{video_suffix}")
            cam.writer_path = out_path
            if _is_live_source(cam.source):
                output_fps = live_record_fps_placeholder
            elif use_offline_timeline and sync_fps is not None:
                # Debug video writes one frame per shared beat; do not use the high container FPS or playback will be fast.
                output_fps = float(sync_fps)
            else:
                output_fps = cam.fps
            cam.writer = make_video_writer(out_path, output_fps, (w, h))
        if args.test:
            print(
                f"  [TEST] recording sub-stream track videos → {camera_dir}/*{video_suffix}"
            )

    print(
        "  [track] merge complexity in the stats window is O(P*T), "
        "P=overlap-gated candidate track pairs, T=recent samples per track."
    )
    shutdown = ShutdownController()
    shutdown.install_signal_handlers()
    if args.view:
        print(
            f"  [track] merge every {args.merge_every_frames} frames, "
            f"inference threads={len(cameras)}."
        )
        print("  [ctrl] with the preview focused, press q or Esc to exit; Ctrl+C also shuts down safely.")
    else:
        shutdown.start_stdin_listener()
        print(
            f"  [track] merge every {args.merge_every_frames} frames, "
            f"inference threads={len(cameras)} (no preview window)."
        )
        print(
            "  [ctrl] type q and Enter in this terminal to exit safely; "
            "Ctrl+C also works (twice to force exit)."
        )

    # Cameras that have hard-failed (EOF or RTSP reconnect exhausted) are no longer inferred
    dead_cameras: set[str] = set()

    global_frame = 0
    executor = ThreadPoolExecutor(max_workers=len(cameras))
    try:
        while True:
            if shutdown.poll():
                break
            if args.max_frames is not None and global_frame >= args.max_frames:
                break

            # Exit when every stream has hard-ended
            live_cameras = [c for c in cameras if c.name not in dead_cameras]
            if not live_cameras:
                print("  [exit] all cameras ended or reconnect failed; exiting.")
                break

            # ── Stage 1: parallel inference ────────────────────────────
            # Each camera reads and runs YOLO in its own thread; shared state
            # such as fusion.add_sample is updated serially on the main thread
            # in stage 2 to avoid races. Poll keys while waiting so a long
            # inference block does not freeze the window.
            target_t: float | None = None
            if use_offline_timeline and sync_fps is not None:
                target_t = float(timeline_origin_sec) + float(global_frame) / float(sync_fps)
            futures = [
                executor.submit(
                    _run_camera_inference,
                    cam, models[cam.name], args, classes,
                    args.max_reconnect_retries,
                    target_timestamp_sec=target_t,
                    sync_fps=sync_fps,
                )
                for cam in live_cameras
            ]
            pending = set(futures)
            cam_results: list[tuple] = []
            while pending:
                if should_quit(args.view, shutdown):
                    break
                done, pending = wait(pending, timeout=0.03)
                for future in done:
                    cam_results.append(future.result())
            if shutdown.poll():
                continue

            cam_results.sort(key=lambda item: item[0].name)

            # ── Stage 2: serial fusion update ──────────────────────────
            any_frame = False
            for cam, frame, timestamp_sec, raw_detections, frame_fresh in cam_results:
                if frame is None:
                    # Hard failure (EOF or RTSP reconnect exhausted)
                    dead_cameras.add(cam.name)
                    print(f"  [exit] {cam.name} hard-failed; dropping from inference (other cameras continue).")
                    continue
                any_frame = True
                latest_timestamp = max(latest_timestamp, timestamp_sec)

                frame_wh = (frame.shape[1], frame.shape[0])
                detections_for_draw: list[tuple[dict[str, Any], TrackSample, int]] = []
                rejected_for_draw: list[tuple[dict[str, Any], str]] = []
                current_track_ids = {int(det["track_id"]) for det in raw_detections}
                geo_ms = srl_ms = add_ms = 0.0
                n_used = 0
                if frame_fresh:
                    for key in list(valid_detection_streaks):
                        if key[0] == cam.name and key[1] not in current_track_ids:
                            del valid_detection_streaks[key]
                for det in raw_detections:
                    x1, y1, x2, y2 = [float(v) for v in det["xyxy"]]
                    track_key = (cam.name, det["track_id"])
                    if not frame_fresh:
                        rejected_for_draw.append((det, "STALE"))
                        continue
                    clipped = is_bbox_clipped(
                        det["xyxy"],
                        frame_wh,
                        margin_px=args.clip_margin,
                        margin_ratio=args.clip_margin_ratio,
                    )
                    if clipped:
                        valid_detection_streaks[track_key] = 0
                        rejected_for_draw.append((det, "EDGE"))
                        continue
                    valid_detection_streaks[track_key] = valid_detection_streaks.get(track_key, 0) + 1
                    if valid_detection_streaks[track_key] < args.min_valid_streak:
                        rejected_for_draw.append(
                            (det, f"WAIT {valid_detection_streaks[track_key]}/{args.min_valid_streak}")
                        )
                        continue
                    # P_geo: always from the homography; clean DB store and MVC training input.
                    t_geo = time.perf_counter()
                    p_geo_anchor_xy = box_anchor(
                        det["xyxy"],
                        args.anchor,
                        homography=cam.homography,
                        frame_wh=frame_wh,
                        scale=cam.scale,
                        risk_scalar_mode=args.risk_scalar,
                        s0_m=args.anchor_s0_m,
                        grazing_d0=args.pixel_grazing_d0,
                        sensitivity_clip_m=args.metric_sensitivity_clip_m,
                        shape_eta=args.anchor_shape_eta,
                        aspect_ratio_min=args.anchor_aspect_ratio_min,
                        aspect_ratio_max=args.anchor_aspect_ratio_max,
                    )
                    p_geo_xy = image_point_to_world(cam.homography, p_geo_anchor_xy, cam.scale)
                    geo_ms += (time.perf_counter() - t_geo) * 1000.0
                    n_used += 1

                    # g2 residual: used in-memory for matching, BEV display, and online learn.
                    # DB world_x/y always store P_geo; g2 goes to final_world_x/y
                    # (remerge can also batch-recompute with --g2-checkpoint).
                    final_world_xy: tuple[float, float] | None = None
                    display_anchor_xy = p_geo_anchor_xy
                    if g2_controller is not None:
                        t_srl = time.perf_counter()
                        pred = g2_controller.predict_residual(
                            det["xyxy"],
                            cam.homography,
                            cam.scale,
                            frame_wh,
                            camera_name=cam.name,
                            scene_id=args.scene,
                            world_geo_override=p_geo_xy,
                        )
                        srl_ms += (time.perf_counter() - t_srl) * 1000.0
                        dx, dy = float(pred.delta_m[0]), float(pred.delta_m[1])
                        tx, ty = g2_controller.camera_translation(cam.name, args.scene)
                        final_world_xy = (p_geo_xy[0] + tx + dx, p_geo_xy[1] + ty + dy)
                        g2_anchor = world_point_to_image(cam.homography, final_world_xy, cam.scale)
                        display_anchor_xy = (
                            g2_anchor if np.all(np.isfinite(g2_anchor)) else pred.image_anchor_geo
                        )

                    sample = TrackSample(
                        timestamp_sec=timestamp_sec,
                        frame_index=cam.frame_index,
                        image_xy=display_anchor_xy,
                        world_xy=final_world_xy if final_world_xy is not None else p_geo_xy,
                        bbox_xywh=(x1, y1, x2 - x1, y2 - y1),
                        confidence=det["confidence"],
                        class_id=det["class_id"],
                        class_name=det["class_name"],
                        clipped=clipped,
                    )
                    t_add = time.perf_counter()
                    global_id = fusion.add_sample(cam.name, det["track_id"], sample)
                    add_ms += (time.perf_counter() - t_add) * 1000.0
                    if g2_controller is not None and RuntimeObservation is not None:
                        g2_controller.record_runtime_observation(
                            RuntimeObservation(
                                camera_name=cam.name,
                                global_id=global_id,
                                local_track_id=det["track_id"],
                                timestamp_sec=timestamp_sec,
                                xyxy=det["xyxy"],
                                homography=cam.homography,
                                scale=cam.scale,
                                frame_wh=frame_wh,
                                clipped=clipped,
                                scene_id=args.scene,
                            )
                        )
                    detections_for_draw.append((det, sample, global_id))
                    if cam.frame_index % args.record_every_n_frames == 0:
                        pending_rows.append(
                            ObservationRow(
                                scene_id=args.scene,
                                batch_id=args.batch,
                                camera_name=cam.name,
                                local_track_id=det["track_id"],
                                global_id=global_id,
                                frame_index=cam.frame_index,
                                timestamp_sec=timestamp_sec,
                                image_xy=p_geo_anchor_xy,
                                world_xy=p_geo_xy,
                                final_world_xy=final_world_xy,
                                bbox_xywh=sample.bbox_xywh,
                                confidence=sample.confidence,
                                class_id=sample.class_id,
                                class_name=sample.class_name,
                            )
                        )

                # Overlay frames are consumed only by the preview and debug video;
                # in batch (--no-view and no recording) full-frame copy + per-box
                # drawing dominates, so render lazily if a consumer exists.
                if args.view or cam.writer is not None:
                    roi = cam.effective_cfg.get("image_roi")
                    drawn = draw_camera_tracks(
                        frame,
                        cam.name,
                        detections_for_draw,
                        fusion,
                        rejected_detections=rejected_for_draw,
                        debug=args.test,
                        frame_index=cam.frame_index,
                        infer_wh=cam.infer_wh,
                        image_roi=roi if isinstance(roi, list) else None,
                        global_sync_index=global_frame,
                        session_t0=session_t0,
                        timestamp_sec=timestamp_sec,
                        frame_fresh=frame_fresh,
                    )
                    if cam.writer is not None:
                        cam.writer.write(drawn)
                        if not frame_fresh:
                            cam.stale_write_count += 1
                    if args.view:
                        cv.imshow(f"{cam.name} tracks", drawn)
                # On the offline timeline, frame_index was already aligned to the
                # container during catch-up; equal-FPS per-frame mode still increments per beat.
                if stage_sink is not None:
                    stage_sink.add_frame(
                        camera=cam.name,
                        frame_index=cam.frame_index,
                        timestamp_sec=timestamp_sec,
                        n_det=len(raw_detections),
                        n_used=n_used,
                        track_ms=float(cam.last_track_ms),
                        geo_ms=geo_ms,
                        srl_ms=srl_ms,
                        add_ms=add_ms,
                    )
                if not (use_offline_timeline and not _is_live_source(cam.source)):
                    cam.frame_index += 1

            if should_quit(args.view, shutdown):
                continue

            if not any_frame:
                # No live camera produced a frame this round (all hard-failed)
                break

            # Offline timeline: use the target time as merge now so one camera's
            # POS_MSEC jitter cannot inflate latest_timestamp and drop others via active_timeout.
            if target_t is not None:
                latest_timestamp = max(latest_timestamp, float(target_t))

            # ── Stage 3: cross-camera merge (every N frames) ───────────
            if global_frame % args.merge_every_frames == 0:
                t_merge = time.perf_counter()
                current_comparisons, merge_events = fusion.merge_active(latest_timestamp)
                if stage_sink is not None:
                    stage_sink.add_merge(
                        frame_index=global_frame,
                        timestamp_sec=latest_timestamp,
                        merge_ms=(time.perf_counter() - t_merge) * 1000.0,
                        n_events=len(merge_events),
                        n_comparisons=current_comparisons,
                    )
                for event in merge_events:
                    event_label = (
                        "same-camera link"
                        if event.merge_kind == "same_camera_link"
                        else "cross-camera merge"
                    )
                    print(
                        f"  [{event_label}] G{event.merged_global_id} -> "
                        f"G{event.kept_global_id} "
                        f"{event.track_a.camera_name}:L{event.track_a.local_track_id} "
                        f"↔ {event.track_b.camera_name}:L{event.track_b.local_track_id} "
                        f"score={event.scores.score:.3f} dist={event.scores.mean_distance_m:.2f}m"
                    )

            # ── Stage 4: flush observations ────────────────────────────
            should_flush = (
                len(pending_rows) >= args.db_batch_size
                or global_frame % args.db_flush_every_frames == 0
            )
            if should_flush:
                written, written_bytes = flush_observations(conn, pending_rows, fusion)
                rows_written_window += written
                bytes_written_window += written_bytes

            # ── Stage 5: BEV track render ──────────────────────────────
            # Same: without preview or BEV recording, do not redraw full track
            # history every frame; --save-bev-image renders once at exit.
            if args.view or bev_writer is not None:
                last_bev_frame = draw_bev_overlay(
                    bev_background,
                    fusion,
                    geometry,
                    latest_timestamp,
                    args.display_history_sec,
                    max_points=args.bev_history_samples,
                )
                if bev_writer is not None:
                    bev_writer.write(last_bev_frame)
                if args.view:
                    cv.imshow("global BEV trajectories", last_bev_frame)
            if should_quit(args.view, shutdown):
                continue

            # ── Stage 6: stats + bulk DB commit ────────────────────────
            # Combine flush_rewrites and save_runtime_stats into one commit,
            # instead of commit-per-merge plus commit-per-stats.
            if g2_controller is not None:
                g2_controller.maybe_incremental_learn()

            if global_frame % args.stats_every_frames == 0:
                fusion.flush_rewrites()
                elapsed = max(time.perf_counter() - stats_window_start, 1e-6)
                rows_per_sec = rows_written_window / elapsed
                bytes_per_sec = bytes_written_window / elapsed
                active_local = len(fusion.active_tracks(latest_timestamp))
                active_global = fusion.active_global_count(latest_timestamp)
                save_runtime_stats(
                    conn,
                    args.scene,
                    args.batch,
                    global_frame,
                    latest_timestamp,
                    active_local,
                    active_global,
                    current_comparisons,
                    args.history_samples,
                    rows_written_window,
                    rows_per_sec,
                    bytes_per_sec,
                )
                conn.commit()  # one commit: deferred global_id rewrite + stats INSERT
                print(
                    f"  [stats] frame={global_frame} active={active_local}/{active_global} "
                    f"comparisons={current_comparisons} "
                    f"storage={rows_per_sec:.1f} rows/s, {bytes_per_sec / 1024:.1f} KiB/s"
                )
                rows_written_window = 0
                bytes_written_window = 0
                stats_window_start = time.perf_counter()

            global_frame += 1

    except KeyboardInterrupt:
        stage_status = "interrupted"
        shutdown.request("KeyboardInterrupt; shutting down safely …")
    finally:
        if stage_sink is not None:
            stage_sink.close()
            print(f"  [stage-profile] wrote {stage_sink.path} ({len(stage_sink.rows)} rows)")
        shutdown.stop_stdin_listener()
        executor.shutdown(wait=False)
        if use_async_reader:
            stop_async_readers(cameras)
        wall_duration = max(time.time() - session_t0, 1e-3)
        live_playback_fps = global_frame / wall_duration if global_frame > 0 else 25.0
        live_playback_fps = max(1.0, min(60.0, live_playback_fps))

        for cam in cameras:
            cam.cap.release()
            if cam.writer is not None:
                cam.writer.release()
                cam.writer = None
        if bev_writer is not None:
            bev_writer.release()
            bev_writer = None

        if record_videos and global_frame > 0 and has_live_sources:
            print(
                f"  [record] live streams aligned to wall-clock playback fps: {live_playback_fps:.2f} fps "
                f"({global_frame} sync frames / {wall_duration:.1f}s)"
            )
            for cam in cameras:
                if (
                    _is_live_source(cam.source)
                    and cam.writer_path
                    and rewrite_video_playback_fps(cam.writer_path, live_playback_fps)
                ):
                    print(f"  [record] {cam.name} corrected: {cam.writer_path}")
            if bev_writer_path and rewrite_video_playback_fps(
                bev_writer_path, live_playback_fps
            ):
                print(f"  [record] BEV corrected: {bev_writer_path}")
        elif record_videos and global_frame > 0:
            print(
                f"  [record] offline files keep the source timeline, BEV={bev_record_fps:.3f} fps; "
                "not rewritten from YOLO inference wall time."
            )

        if test_manifest_path is not None:
            write_test_manifest(
                test_manifest_path,
                {
                    "finished_at": utc_now(),
                    "global_frames": global_frame,
                    "wall_duration_sec": round(wall_duration, 3),
                    "playback_fps": round(
                        live_playback_fps if has_live_sources else bev_record_fps,
                        3,
                    ),
                    "playback_fps_mode": "wall_clock_live" if has_live_sources else "source_timeline",
                    "per_camera_output_fps": {
                        cam.name: round(
                            live_playback_fps if _is_live_source(cam.source) else cam.fps,
                            3,
                        )
                        for cam in cameras
                    },
                    "async_reader": use_async_reader,
                    "per_camera_frames": {cam.name: cam.frame_index for cam in cameras},
                    "per_camera_stale_writes": {
                        cam.name: cam.stale_write_count for cam in cameras
                    },
                },
                merge=True,
            )
            print(f"  [TEST] run manifest updated: {test_manifest_path}")
        written, _ = flush_observations(conn, pending_rows, fusion)
        if written:
            print(f"  [DB] wrote {written} trajectory observations.")
        rewrite_count = fusion.flush_rewrites()
        if rewrite_count:
            conn.commit()
        if g2_controller is not None:
            if g2_controller.flush_learn():
                print("  [g2] finishing a last incremental self-train before exit.")
            g2_controller.save_checkpoint()
        if args.save_bev_image:
            out_path = Path(args.save_bev_image)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            imwrite(
                out_path,
                draw_bev_overlay(
                    bev_background,
                    fusion,
                    geometry,
                    latest_timestamp,
                    args.display_history_sec,
                    max_points=args.bev_history_samples,
                ),
            )
            print(f"saved: {out_path}")
        finish_batch_stage(
            conn,
            stage_run_id,
            (
                "failed"
                if sys.exc_info()[0] is not None
                else locals().get("stage_status", "completed")
            ),
            {
                "global_frames": global_frame,
                "wall_duration_sec": round(wall_duration, 3),
            },
        )
        conn.close()
        if args.view:
            cv.destroyAllWindows()
        print("  [exit] shutdown complete.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-camera vehicle trajectory drawing, world mapping, and cross-camera track merge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Offline: set cameras[].video to an MP4 in YAML (comment out stream/sub_stream), or:
  py pipeline/multi_camera_trajectory_fusion.py --config configs/my_scene.yaml --db cals.db --scene parking --batch run_001 \\
      --source A=outputs/recorded/run/A.mp4 --model nn_models/yolo11n.pt --bev-background outputs/bev_mosaic.png

  # For shade/low-contrast scenes try CLAHE (YOLO input only):
  py pipeline/multi_camera_trajectory_fusion.py --config configs/my_scene.yaml --db cals.db --scene parking --batch run_001 \\
      --clahe --clahe-clip 2.0 --model nn_models/yolo11n.pt --bev-background outputs/bev_mosaic.png

  # Test mode: auto-record per-camera sub-stream YOLO track videos + BEV video under outputs/debug_tracking/
  py pipeline/multi_camera_trajectory_fusion.py --config configs/my_scene.yaml --db cals.db --scene parking --batch run_001 \\
      --model nn_models/yolo11n.pt --device 0 --bev-background outputs/bev_mosaic.png --test --no-view

Trajectory tables
-----------------
  trajectory_observations   per-batch local track points, world coords, time, and global IDs
  trajectory_merges         cross-camera fusion / same-camera linking events, type, and scores
  trajectory_runtime_stats  candidate-comparison complexity O(P*T) and live write rate
  trajectory_batches       scene/batch identity and stage-audit metadata
        """,
    )
    parser.add_argument("--config", required=True, help="YAML config file path")
    parser.add_argument("--db", default="cals.db", help="SQLite calibration/trajectory database path")
    parser.add_argument("--scene", default="parking", help="scene ID (default parking)")
    parser.add_argument(
        "--batch",
        required=True,
        help="trajectory batch ID; tracking/train/remerge/replay must use the same value",
    )
    parser.add_argument(
        "--source",
        action="append",
        metavar="NAME=PATH_OR_INDEX",
        help="override YAML camera.video/stream/source; repeatable",
    )
    parser.add_argument(
        "--detect-backend",
        dest="detect_backend",
        choices=["ultralytics", "mmdet_yolox"],
        default="ultralytics",
        help=(
            "Detect/track backend: ultralytics=YOLO.track()+built-in BoT-SORT (default); "
            "mmdet_yolox=pure PyTorch MMDet YOLOX .pth + standalone BoT-SORT"
        ),
    )
    parser.add_argument(
        "--model",
        default="nn_models/yolo11n.pt",
        help="weights path: ultralytics .pt; mmdet_yolox MMDet YOLOX .pth",
    )
    parser.add_argument(
        "--tracker",
        default="botsort.yaml",
        help="BoT-SORT hyperparameter YAML (both backends; ultralytics also uses it as tracker name)",
    )
    parser.add_argument(
        "--mmdet-class-name",
        dest="mmdet_class_name",
        default="vehicle",
        help="mmdet_yolox single-class head class_name (default vehicle)",
    )
    parser.add_argument(
        "--classes",
        default="2,3,5,7",
        help='class filter: COCO numeric id or class name (quote names with spaces); all=no filter',
    )
    parser.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--device", default=None, help="inference device, e.g. 0/cpu")
    parser.add_argument("--imgsz", type=int, default=None, help="detection input size (YOLO / YOLOX)")
    parser.add_argument(
        "--clahe",
        action="store_true",
        help="apply CLAHE (LAB luminance) to the infer frame before model.track; overlays/saves still use the original frame",
    )
    parser.add_argument(
        "--clahe-clip",
        type=float,
        default=2.0,
        dest="clahe_clip",
        metavar="LIMIT",
        help="CLAHE clipLimit (default 2.0; too high adds noisy false detections; 1.5-3.0 is typical)",
    )
    parser.add_argument(
        "--anchor",
        choices=[
            "multiplicative_gating_offsetted",
            "grazing_angle_offsetted",
            "slimness_squared_offsetted",
            "bottom_center",
            "center",
        ],
        default="multiplicative_gating_offsetted",
        help=(
            "Target anchor: multiplicative_gating_offsetted (default, risk scalar + multiplicative slimness gate), "
            "grazing_angle_offsetted (risk scalar only), slimness_squared_offsetted (height/width^2 only), "
            "bottom_center (box bottom center) or center (rectangle center)"
        ),
    )
    parser.add_argument(
        "--risk-scalar",
        dest="risk_scalar",
        choices=RISK_SCALAR_MODES,
        default="metric_jacobian",
        help=(
            "fab1 risk-scalar definition (default metric_jacobian: homography Jacobian metric local sensitivity s=||J_H||_F, "
            "α=s/(s+s0), comparable across cameras/resolutions); pixel_orthogonal/pixel_vertical are cla1 "
            "pixel-distance ablation baselines; grazing_angle is an unimplemented ablation placeholder"
        ),
    )
    parser.add_argument(
        "--anchor-s0-m",
        type=float,
        default=60.0,
        dest="anchor_s0_m",
        metavar="METERS",
        help="metric sensitivity saturation s0 (meters); only for --risk-scalar=metric_jacobian (default 60)",
    )
    parser.add_argument(
        "--anchor-shape-eta",
        type=float,
        default=DEFAULT_ANCHOR_SHAPE_ETA,
        dest="anchor_shape_eta",
        metavar="ETA",
        help="multiplicative-gate shape-prior mix η (default 0.75)",
    )
    parser.add_argument(
        "--anchor-aspect-ratio-min",
        type=float,
        default=DEFAULT_ANCHOR_ASPECT_RATIO_MIN,
        dest="anchor_aspect_ratio_min",
        metavar="RHO_MIN",
        help="shape-prior rho=h/w lower bound (default 0.5)",
    )
    parser.add_argument(
        "--anchor-aspect-ratio-max",
        type=float,
        default=DEFAULT_ANCHOR_ASPECT_RATIO_MAX,
        dest="anchor_aspect_ratio_max",
        metavar="RHO_MAX",
        help="shape-prior rho=h/w upper bound (default 3.0)",
    )
    parser.add_argument(
        "--pixel-grazing-d0",
        type=float,
        default=50.0 / 1080.0,
        dest="pixel_grazing_d0",
        metavar="FRAC",
        help=(
            "soft saturation scale for the pixel-distance risk-scalar ablation (frame-height fraction): d0_px = FRAC × frame_h, "
            "α=d0/(d+d0); only for --risk-scalar=pixel_orthogonal/pixel_vertical"
            " (default 50/1080≈0.046)"
        ),
    )
    parser.add_argument(
        "--metric-sensitivity-clip-m",
        type=float,
        default=None,
        dest="metric_sensitivity_clip_m",
        metavar="METERS",
        help=(
            "clip metric sensitivity s upper bound (meters); changes α/κ; no clip by default. "
            "Must match training / the g2 checkpoint (stored in checkpoint meta)"
        ),
    )
    parser.add_argument("--history-samples", type=int, default=180, help="samples kept per local track")
    parser.add_argument("--display-history-sec", type=float, default=20.0, help="BEV display window in seconds")
    parser.add_argument(
        "--bev-history-samples",
        type=int,
        default=60,
        help="max points drawn per global BEV track (0=unlimited); lower to cut per-frame sort+polyline cost",
    )
    parser.add_argument("--active-timeout", type=float, default=2.0, help="seconds without a sighting before dropping from merge")
    parser.add_argument(
        "--merge-score-threshold",
        type=float,
        default=DEFAULT_MATCH_CONFIG.score_threshold,
        help="cross-camera merge total-score threshold",
    )
    parser.add_argument(
        "--merge-max-time-gap",
        type=float,
        default=DEFAULT_MATCH_CONFIG.max_time_gap_sec,
        help="max time gap allowed for a merge",
    )
    parser.add_argument(
        "--merge-max-distance",
        type=float,
        default=DEFAULT_MATCH_CONFIG.max_distance_m,
        help="max world distance allowed for a merge (meters)",
    )
    parser.add_argument(
        "--merge-min-samples",
        type=int,
        default=DEFAULT_MATCH_CONFIG.min_samples,
        help="min track samples required to merge",
    )
    parser.add_argument(
        "--merge-sample-count",
        type=int,
        default=DEFAULT_MATCH_CONFIG.sample_count,
        help="shape-match resample count",
    )
    parser.add_argument(
        "--merge-time-sigma",
        type=float,
        default=DEFAULT_MATCH_CONFIG.time_sigma,
        help="exponential decay scale for merge time score (seconds)",
    )
    parser.add_argument(
        "--merge-distance-sigma",
        type=float,
        default=DEFAULT_MATCH_CONFIG.distance_sigma,
        help="exponential decay scale for merge distance score (meters)",
    )
    parser.add_argument(
        "--merge-shape-sigma",
        type=float,
        default=DEFAULT_MATCH_CONFIG.shape_sigma,
        help="exponential decay scale for merge shape score",
    )
    parser.add_argument(
        "--merge-time-weight",
        type=float,
        default=DEFAULT_MATCH_CONFIG.time_weight,
        help="time weight in the merge total score",
    )
    parser.add_argument(
        "--merge-distance-weight",
        type=float,
        default=DEFAULT_MATCH_CONFIG.distance_weight,
        help="distance weight in the merge total score",
    )
    parser.add_argument(
        "--merge-shape-weight",
        type=float,
        default=DEFAULT_MATCH_CONFIG.shape_weight,
        help="shape weight in the merge total score",
    )
    parser.add_argument(
        "--merge-every-frames",
        type=int,
        default=5,
        help="run cross-camera merge every N global frames (default 5; lower reduces per-frame cost; ~0.2s at 25fps)",
    )
    # ── Same-camera tracklet stitching (temporal reconnect) ──────────────
    parser.add_argument(
        "--stitch",
        dest="stitch_enabled",
        action="store_true",
        default=None,
        help="enable sliding-window Hungarian same-camera tracklet linking (off by default)",
    )
    parser.add_argument(
        "--no-stitch",
        dest="stitch_enabled",
        action="store_false",
        help="disable same-camera tracklet linking (overrides YAML trajectory_merge.stitch_enabled)",
    )
    parser.add_argument(
        "--stitch-finalize-sec",
        type=float,
        default=DEFAULT_MATCH_CONFIG.stitch_finalize_sec,
        help=(
            "local-ID finalize period (seconds); 0=auto from tracker track_buffer × measured sample period"
        ),
    )
    parser.add_argument(
        "--stitch-tracker-buffer-frames",
        type=int,
        default=0,
        help="tracker lost-track hold frames; 0=read track_buffer from --tracker YAML",
    )
    parser.add_argument(
        "--stitch-finalize-margin",
        type=float,
        default=DEFAULT_MATCH_CONFIG.stitch_finalize_margin_sec,
        help="extra finalize margin beyond track_buffer×sample period (seconds)",
    )
    parser.add_argument(
        "--stitch-default-sample-period",
        type=float,
        default=DEFAULT_MATCH_CONFIG.stitch_default_sample_period_sec,
        help="fallback sample period when a track has too few samples (seconds)",
    )
    parser.add_argument(
        "--stitch-max-gap",
        type=float,
        default=DEFAULT_MATCH_CONFIG.stitch_max_gap_sec,
        help="max same-camera stitch time gap (seconds)",
    )
    parser.add_argument(
        "--stitch-max-overlap",
        type=float,
        default=DEFAULT_MATCH_CONFIG.stitch_max_overlap_sec,
        help="allowed slight overlap for same-camera linking; 0=auto from both measured sample periods",
    )
    parser.add_argument(
        "--stitch-base-tolerance",
        type=float,
        default=DEFAULT_MATCH_CONFIG.stitch_base_tolerance_m,
        help="legacy metric base tolerance; 0=use prediction-uncertainty normalized gating",
    )
    parser.add_argument(
        "--stitch-tolerance-rate",
        type=float,
        default=DEFAULT_MATCH_CONFIG.stitch_tolerance_rate_mps,
        help="legacy metric tolerance growth rate; both 0 disables the legacy gate",
    )
    parser.add_argument(
        "--stitch-min-samples",
        type=int,
        default=DEFAULT_MATCH_CONFIG.stitch_min_samples,
        help="min samples required on both sides of a stitch",
    )
    parser.add_argument(
        "--stitch-direction-min-cos",
        type=float,
        default=DEFAULT_MATCH_CONFIG.stitch_direction_min_cos,
        help="min heading cosine when both tracks are moving (-1~1)",
    )
    parser.add_argument(
        "--stitch-min-speed",
        type=float,
        default=DEFAULT_MATCH_CONFIG.stitch_min_speed_mps,
        help="below this speed (m/s) treat as stationary and skip heading check",
    )
    parser.add_argument(
        "--stitch-speed-sigma",
        type=float,
        default=DEFAULT_MATCH_CONFIG.stitch_speed_sigma_mps,
        help="exponential decay scale for same-camera linking speed-difference score (m/s)",
    )
    parser.add_argument(
        "--stitch-score-threshold",
        type=float,
        default=DEFAULT_MATCH_CONFIG.stitch_score_threshold,
        help="min accepted score for a Hungarian same-camera linking real edge",
    )
    parser.add_argument(
        "--stitch-uncertainty-floor",
        type=float,
        default=DEFAULT_MATCH_CONFIG.stitch_uncertainty_floor_m,
        help="track position uncertainty floor (meters)",
    )
    parser.add_argument(
        "--stitch-process-noise",
        type=float,
        default=DEFAULT_MATCH_CONFIG.stitch_process_noise_mps2,
        help="motion-extrapolation process-noise scale (m/s^2)",
    )
    parser.add_argument(
        "--stitch-max-normalized-error",
        type=float,
        default=DEFAULT_MATCH_CONFIG.stitch_max_normalized_error,
        help="max dimensionless prediction error (Mahalanobis-like gate)",
    )
    parser.add_argument(
        "--stitch-max-uncertainty-ratio",
        type=float,
        default=DEFAULT_MATCH_CONFIG.stitch_max_uncertainty_ratio,
        help="max predicted-std / two-end uncertainty-floor ratio; abstain above this",
    )
    parser.add_argument(
        "--stitch-require-mutual-best",
        dest="stitch_require_mutual_best",
        action="store_true",
        default=None,
        help="require the Hungarian link to be best on both the old-track row and new-track column (on by default)",
    )
    parser.add_argument(
        "--no-stitch-mutual-best",
        dest="stitch_require_mutual_best",
        action="store_false",
        help="disable mutual-best constraint for same-camera linking",
    )
    parser.add_argument(
        "--stitch-min-score-margin",
        type=float,
        default=DEFAULT_MATCH_CONFIG.stitch_min_score_margin,
        help="min score margin of the chosen edge over the row/column runner-up",
    )
    parser.add_argument(
        "--guard-sync-tolerance",
        "--guard-overlap-tolerance",
        dest="guard_sync_tolerance",
        type=float,
        default=DEFAULT_MATCH_CONFIG.guard_sync_tolerance_sec,
        help="max time gap for co-timed samples; 0=derive from measured sample period (legacy alias kept)",
    )
    parser.add_argument(
        "--guard-sync-period-factor",
        type=float,
        default=DEFAULT_MATCH_CONFIG.guard_sync_period_factor,
        help="auto sync tolerance as a multiple of the larger sample period",
    )
    parser.add_argument(
        "--guard-max-separation",
        type=float,
        default=DEFAULT_MATCH_CONFIG.guard_max_separation_m,
        help="legacy fixed spatial-conflict distance (meters); 0=use normalized spatial-conflict gate",
    )
    parser.add_argument(
        "--guard-position-sigma-floor",
        type=float,
        default=DEFAULT_MATCH_CONFIG.guard_position_sigma_floor_m,
        help="same-camera spatial-guard position uncertainty floor (meters)",
    )
    parser.add_argument(
        "--guard-position-sigma-cap",
        type=float,
        default=DEFAULT_MATCH_CONFIG.guard_position_sigma_cap_m,
        help="same-camera spatial-guard position uncertainty cap (meters); keeps high-maneuver residual from disabling the guard",
    )
    parser.add_argument(
        "--guard-max-normalized-separation",
        type=float,
        default=DEFAULT_MATCH_CONFIG.guard_max_normalized_separation,
        help="max dimensionless separation of same-camera co-timed positions",
    )
    parser.add_argument(
        "--guard-min-conflicting-samples",
        type=int,
        default=DEFAULT_MATCH_CONFIG.guard_min_conflicting_samples,
        help="min co-timed same-camera spatial-conflict sample pairs required to reject a union",
    )
    parser.add_argument("--record-every-n-frames", type=int, default=1, help="observation write interval in frames")
    parser.add_argument(
        "--clip-margin",
        type=int,
        default=3,
        help=(
            "min pixels for bbox edge filtering; actual margin is max of this and --clip-margin-ratio"
        ),
    )
    parser.add_argument(
        "--clip-margin-ratio",
        type=float,
        default=0.01,
        help="bbox edge filter as a fraction of frame width/height (default 0.01, i.e. 1%%)",
    )
    parser.add_argument(
        "--min-valid-streak",
        type=int,
        default=3,
        help="consecutive valid frames after the target fully leaves the frame edge before writing world tracks (default 3)",
    )
    parser.add_argument("--db-batch-size", type=int, default=256, help="observation bulk-write batch size")
    parser.add_argument("--db-flush-every-frames", type=int, default=10, help="flush the DB at most every N global frames")
    parser.add_argument("--stats-every-frames", type=int, default=30, help="runtime-stats write/print interval")
    parser.add_argument("--bev-background", default=None, help="stitched BEV background image; default is YAML output")
    parser.add_argument("--save-bev-video", default=None, help="save BEV video with global tracks")
    parser.add_argument("--save-bev-image", default=None, help="save the last BEV track frame on exit")
    parser.add_argument("--save-camera-dir", default=None, help="directory for per-camera boxed track videos")
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Test/debug mode: write sub-stream YOLO track videos, BEV global-track video, and last BEV frame "
            "under outputs/debug_tracking/<scene>_<timestamp>/ (override with --test-dir); "
            "overlays anchors, CLIP marks, and image_roi"
        ),
    )
    parser.add_argument(
        "--test-dir",
        default=None,
        help="--test output directory; default outputs/debug_tracking/<scene>_<timestamp>",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="max global-frame rounds to process")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume mode: read MAX(frame_index) per camera from the DB and seek the video to the last "
            "interrupted position; global_id continues from the DB max. For recovering offline "
            "video files after an unexpected stop; not for live RTSP."
        ),
    )
    parser.add_argument(
        "--no-view",
        dest="view",
        action="store_false",
        help="do not pop an OpenCV live window; type q+Enter in this terminal or Ctrl+C to exit safely",
    )
    parser.add_argument(
        "--rtsp-transport",
        default="tcp",
        choices=["tcp", "udp"],
        help="RTSP transport (default tcp; tcp is more stable, udp has lower latency)",
    )
    parser.add_argument(
        "--max-reconnect-retries",
        type=int,
        default=5,
        help="max RTSP reconnects after a drop (default 5; 0=no reconnect, drop that camera on failure)",
    )
    parser.add_argument(
        "--sub-source",
        nargs="*",
        default=None,
        metavar="NAME=URL",
        help=(
            "YOLO inference sub-stream for a camera, format NAME=URL (repeatable). "
            "Example: --sub-source A=rtsp://192.168.1.64/sub B=rtsp://192.168.1.65/sub. "
            "The calibration homography is rescaled by main/sub-stream resolution; no YAML edit needed. "
            "You can also set sub_stream: rtsp://... on the YAML cameras entry."
        ),
    )
    g2 = parser.add_argument_group(
        "g2 geometry-guided residual learning (g6 MVC self-supervision; needs PyTorch)",
    )
    g2.add_argument(
        "--g2-checkpoint",
        default=None,
        metavar="PATH",
        help=(
            "Enable g2: P_final = P_geo + ΔP, ΔP=tanh(MLP(F))*r_max. "
            "Inference-only needs a trained .pt; with --g2-online-learn you can start from zero residual"
        ),
    )
    g2.add_argument(
        "--g2-camera-translation-json",
        default=None,
        metavar="PATH",
        help=(
            "Explicit runtime fixed per-camera translation correction JSON. "
            "Training correction in the checkpoint is metadata only and is not auto-applied to final_world_x/y."
        ),
    )
    g2.add_argument(
        "--g2-acceptor-json",
        default=None,
        metavar="PATH",
        help=(
            "Val-fitted residual invoke JSON (residual_cameras). Cameras not on "
            "the list receive dP=0; T_c is not gated. Without this flag, dP is "
            "restricted to cameras on the checkpoint training overlap graph. "
            "Mutually exclusive with --g2-force-all-residual."
        ),
    )
    g2.add_argument(
        "--g2-force-all-residual",
        action="store_true",
        help=(
            "Apply the shared residual head to every camera (negative-transfer "
            "diagnostic). Mutually exclusive with --g2-acceptor-json."
        ),
    )
    g2.add_argument(
        "--g2-camera-translation-acceptance",
        dest="g2_camera_translation_acceptance",
        choices=["recommended", "caution", "any"],
        default="recommended",
        help=(
            "Filter scenes in --g2-camera-translation-json by diagnostics_by_scene.decision.status "
            "(default 'recommended', apply low-order correction only when status=recommended). "
            "'caution' keeps recommended+caution; 'any' keeps all (no filter). "
            "Under recommended/caution, scenes with missing or unknown status are skipped; "
            "legacy JSON must pass 'any' explicitly."
        ),
    )
    g2.add_argument(
        "--g2-online-learn",
        action="store_true",
        help="buffer MVC pairs in overlap at runtime and incrementally fine-tune the MLP",
    )
    g2.add_argument(
        "--g2-loss-mode",
        choices=("asymmetric_teacher", "symmetric_weighted"),
        default="asymmetric_teacher",
        help="online-learn MVC loss: default stop-gradient asymmetric teacher; symmetric_weighted for ablation",
    )
    g2.add_argument(
        "--g2-min-teacher-confidence",
        type=float,
        default=0.20,
        help="at least one view must reach this geometric confidence ω to be used for online learn (default 0.20)",
    )
    g2.add_argument(
        "--g2-min-teacher-confidence-gap",
        type=float,
        default=0.05,
        help=(
            "min confidence gap |κA-κB| for an online MVC pair; "
            "skip the pair if below; skip the whole update if too few qualifying pairs (default 0.05)"
        ),
    )
    g2.add_argument(
        "--g2-online-checkpoint-out",
        default=None,
        metavar="PATH",
        help=(
            "online-learn extra save path; default appends .online to the input checkpoint, "
            "without overwriting the offline baseline checkpoint"
        ),
    )
    g2.add_argument(
        "--g2-teacher-confidence-power",
        type=float,
        default=1.0,
        help="teacher confidence power; >1 favors high-confidence geometric views (default 1.0)",
    )
    g2.add_argument(
        "--g2-student-weight-floor",
        type=float,
        default=0.05,
        help="min student weight kept on a high-confidence view so the teacher cannot fully freeze (default 0.05)",
    )
    g2.add_argument(
        "--g2-risk-scalar",
        dest="g2_risk_scalar",
        choices=RISK_SCALAR_MODES,
        default="metric_jacobian",
        help=(
            "risk-scalar definition when online-learning from zero residual (default metric_jacobian); "
            "an existing checkpoint forces its training-time definition and this flag is ignored"
        ),
    )
    g2.add_argument(
        "--g2-anchor-s0-m",
        dest="g2_anchor_s0_m",
        type=float,
        default=60.0,
        metavar="METERS",
        help="metric sensitivity saturation s0 when online-learning from zero residual (default 60)",
    )
    g2.add_argument(
        "--g2-feature-l0-m",
        dest="g2_feature_l0_m",
        type=float,
        default=10.0,
        metavar="METERS",
        help="metric box-feature normalization L0 when online-learning from zero residual (default 10)",
    )
    g2.add_argument(
        "--g2-anchor-shape-eta",
        dest="g2_anchor_shape_eta",
        type=float,
        default=DEFAULT_ANCHOR_SHAPE_ETA,
        metavar="ETA",
        help="shape-prior mix η when online-learning from zero residual (default 0.75)",
    )
    g2.add_argument(
        "--g2-anchor-aspect-ratio-min",
        dest="g2_anchor_aspect_ratio_min",
        type=float,
        default=DEFAULT_ANCHOR_ASPECT_RATIO_MIN,
        metavar="RHO_MIN",
        help="rho=h/w lower bound when online-learning from zero residual (default 0.5)",
    )
    g2.add_argument(
        "--g2-anchor-aspect-ratio-max",
        dest="g2_anchor_aspect_ratio_max",
        type=float,
        default=DEFAULT_ANCHOR_ASPECT_RATIO_MAX,
        metavar="RHO_MAX",
        help="rho=h/w upper bound when online-learning from zero residual (default 3.0)",
    )
    g2.add_argument(
        "--g2-pixel-grazing-d0",
        dest="g2_pixel_grazing_d0",
        type=float,
        default=50.0 / 1080.0,
        metavar="FRAC",
        help="soft saturation scale when --g2-risk-scalar is a pixel-distance ablation (frame-height fraction, default 50/1080)",
    )
    g2.add_argument(
        "--g2-metric-sensitivity-clip-m",
        dest="g2_metric_sensitivity_clip_m",
        type=float,
        default=None,
        metavar="METERS",
        help="clip metric sensitivity s when online-learning from zero residual (no clip by default)",
    )
    g2.add_argument(
        "--g2-extent-clip-m",
        dest="g2_extent_clip_m",
        type=float,
        default=None,
        metavar="METERS",
        help="clip metric box width/height features when online-learning from zero residual (no clip by default)",
    )
    g2.add_argument(
        "--g2-motion-compensation",
        dest="g2_motion_compensation",
        choices=("off", "nearest", "linear_interpolate"),
        default="linear_interpolate",
        help="online MVC pair construction: default interpolate to a common t*; nearest/off for ablation (ix)",
    )
    g2.add_argument(
        "--g2-time-offsets-json",
        dest="g2_time_offsets_json",
        default=None,
        metavar="JSON",
        help='fixed camera-pair time offset for online learn (seconds), JSON dict, e.g. \'{"c003:c004": 0.032}\'',
    )
    g2.add_argument(
        "--g2-learn-every-frames",
        type=int,
        default=300,
        help="try an online incremental learn every N global frames (default 300)",
    )
    g2.add_argument(
        "--g2-learn-epochs",
        type=int,
        default=3,
        help="epochs per online incremental learn (default 3)",
    )
    g2.add_argument(
        "--g2-learn-min-pairs",
        type=int,
        default=64,
        help="min MVC pairs to trigger online learn (default 64)",
    )
    g2.add_argument(
        "--g2-learn-lr-scale",
        type=float,
        default=0.3,
        help="online-learn lr = offline lr × this scale (default 0.3)",
    )
    g2.add_argument(
        "--g2-learn-max-time-gap",
        type=float,
        default=0.35,
        help="max time gap for online MVC pairing (seconds, default 0.35)",
    )
    g2.add_argument(
        "--g2-learn-max-pairs",
        type=int,
        default=2048,
        help="max high-value MVC pairs per online learn; caps edge compute (default 2048)",
    )
    parser.set_defaults(view=True)

    # ── remerge mode (optional) ──────────────────────────────────────────
    remerge_group = parser.add_argument_group(
        "remerge offline merge recompute (--remerge)",
        "Recompute global_id from existing trajectory_observations without re-running YOLO/video.\n"
        "Only --config / --db / --scene / --batch / --merge-* / --stitch* / --guard-* / "
        "--tracker / --active-timeout / --merge-every-frames / --history-samples apply; "
        "other flags (model, video source, BEV render, …) have no effect and are reported.",
    )
    remerge_group.add_argument(
        "--remerge",
        action="store_true",
        help="enable offline merge-recompute mode",
    )
    remerge_group.add_argument(
        "--remerge-dry-run",
        action="store_true",
        dest="remerge_dry_run",
        help="print merge events only; do not write the DB (use with --remerge)",
    )
    remerge_group.add_argument(
        "--remerge-world-coords",
        choices=("p_geo", "final"),
        default="p_geo",
        dest="remerge_world_coords",
        help=(
            "Which world coordinates to use for merge distance: "
            "p_geo=world_x/y only (pure P_geo, default), "
            "final=final_world_x/y only (skip missing rows; no P_geo fallback)"
        ),
    )
    remerge_group.add_argument(
        "--remerge-sync-gap-sec",
        type=float,
        default=0.05,
        dest="remerge_sync_gap_sec",
        metavar="SEC",
        help="time window for clustering DB observations into co-timed rounds (seconds, default 0.05)",
    )
    remerge_group.add_argument(
        "--remerge-profile",
        default="",
        dest="remerge_profile",
        help="Write per-sync-batch remerge timings to this CSV (no YOLO).",
    )
    remerge_group.add_argument(
        "--remerge-quiet",
        action="store_true",
        dest="remerge_quiet",
        help="Do not print each merge event (used with --remerge-profile).",
    )
    parser.add_argument(
        "--stage-profile",
        default="",
        dest="stage_profile",
        help="Write per-frame YOLO+BoT-SORT / P_geo / SRL timings to this CSV.",
    )
    remerge_group.add_argument(
        "--remerge-g2-for-merge",
        action="store_true",
        dest="remerge_g2_for_merge",
        help=(
            "With remerge + g2, whether g2-corrected final_world coordinates are used for cross-camera merge distance. "
            "Off by default (write final_world_x/y to the DB; merge still uses DB P_geo), "
            "so merge is independent of MLP quality; enable after the MLP has converged for a more precise merge."
        ),
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.record_every_n_frames < 1:
        parser.error("--record-every-n-frames must be >= 1")
    if args.clip_margin < 0:
        parser.error("--clip-margin must be >= 0")
    if not 0 <= args.clip_margin_ratio < 0.5:
        parser.error("--clip-margin-ratio must be in [0, 0.5)")
    if args.min_valid_streak < 1:
        parser.error("--min-valid-streak must be >= 1")
    if args.db_flush_every_frames < 1:
        parser.error("--db-flush-every-frames must be >= 1")
    if args.stats_every_frames < 1:
        parser.error("--stats-every-frames must be >= 1")
    if args.merge_every_frames < 1:
        parser.error("--merge-every-frames must be >= 1")
    if args.merge_score_threshold < 0:
        parser.error("--merge-score-threshold must be >= 0")
    if args.merge_max_time_gap <= 0:
        parser.error("--merge-max-time-gap must be > 0")
    if args.merge_max_distance <= 0:
        parser.error("--merge-max-distance must be > 0")
    if args.merge_min_samples < 1:
        parser.error("--merge-min-samples must be >= 1")
    if args.merge_sample_count < 2:
        parser.error("--merge-sample-count must be >= 2")
    if args.merge_time_sigma <= 0:
        parser.error("--merge-time-sigma must be > 0")
    if args.merge_distance_sigma <= 0:
        parser.error("--merge-distance-sigma must be > 0")
    if args.merge_shape_sigma <= 0:
        parser.error("--merge-shape-sigma must be > 0")
    if args.merge_time_weight < 0 or args.merge_distance_weight < 0 or args.merge_shape_weight < 0:
        parser.error("--merge-*-weight must be >= 0")
    if args.merge_time_weight + args.merge_distance_weight + args.merge_shape_weight <= 0:
        parser.error("at least one --merge-*-weight must be > 0")
    if args.stitch_max_gap <= 0:
        parser.error("--stitch-max-gap must be > 0")
    if args.stitch_finalize_sec < 0:
        parser.error("--stitch-finalize-sec must be >= 0")
    if args.stitch_tracker_buffer_frames < 0:
        parser.error("--stitch-tracker-buffer-frames must be >= 0")
    if args.stitch_finalize_margin < 0:
        parser.error("--stitch-finalize-margin must be >= 0")
    if args.stitch_default_sample_period <= 0:
        parser.error("--stitch-default-sample-period must be > 0")
    if args.stitch_max_overlap < 0:
        parser.error("--stitch-max-overlap must be >= 0")
    if args.stitch_base_tolerance < 0:
        parser.error("--stitch-base-tolerance must be >= 0")
    if args.stitch_tolerance_rate < 0:
        parser.error("--stitch-tolerance-rate must be >= 0")
    if args.stitch_min_samples < 2:
        parser.error("--stitch-min-samples must be >= 2")
    if not -1.0 <= args.stitch_direction_min_cos <= 1.0:
        parser.error("--stitch-direction-min-cos must be in [-1, 1]")
    if args.stitch_min_speed < 0:
        parser.error("--stitch-min-speed must be >= 0")
    if args.stitch_speed_sigma <= 0:
        parser.error("--stitch-speed-sigma must be > 0")
    if not 0.0 <= args.stitch_score_threshold <= 1.0:
        parser.error("--stitch-score-threshold must be in [0, 1]")
    if args.stitch_uncertainty_floor <= 0:
        parser.error("--stitch-uncertainty-floor must be > 0")
    if args.stitch_process_noise < 0:
        parser.error("--stitch-process-noise must be >= 0")
    if args.stitch_max_normalized_error <= 0:
        parser.error("--stitch-max-normalized-error must be > 0")
    if args.stitch_max_uncertainty_ratio <= 0:
        parser.error("--stitch-max-uncertainty-ratio must be > 0")
    if args.stitch_min_score_margin < 0:
        parser.error("--stitch-min-score-margin must be >= 0")
    if args.guard_sync_tolerance < 0:
        parser.error("--guard-sync-tolerance must be >= 0")
    if args.guard_sync_period_factor <= 0:
        parser.error("--guard-sync-period-factor must be > 0")
    if args.guard_max_separation < 0:
        parser.error("--guard-max-separation must be >= 0")
    if args.guard_position_sigma_floor <= 0:
        parser.error("--guard-position-sigma-floor must be > 0")
    if args.guard_position_sigma_cap < args.guard_position_sigma_floor:
        parser.error("--guard-position-sigma-cap must be >= --guard-position-sigma-floor")
    if args.guard_max_normalized_separation <= 0:
        parser.error("--guard-max-normalized-separation must be > 0")
    if args.guard_min_conflicting_samples <= 0:
        parser.error("--guard-min-conflicting-samples must be > 0")
    if args.bev_history_samples < 0:
        parser.error("--bev-history-samples must be >= 0")
    if args.clahe_clip <= 0:
        parser.error("--clahe-clip must be > 0")
    if args.pixel_grazing_d0 <= 0 or args.pixel_grazing_d0 > 1.0:
        parser.error("--pixel-grazing-d0 must be in (0, 1] (frame-height fraction)")
    if args.risk_scalar not in RISK_SCALAR_MODES:
        parser.error(f"--risk-scalar must be one of {RISK_SCALAR_MODES}")
    if args.anchor_s0_m <= 0:
        parser.error("--anchor-s0-m must be > 0")
    if args.anchor_shape_eta < 0 or args.anchor_shape_eta > 1:
        parser.error("--anchor-shape-eta must be in [0, 1]")
    if args.anchor_aspect_ratio_min <= 0 or args.anchor_aspect_ratio_max <= 0:
        parser.error("--anchor-aspect-ratio-min/max must be > 0")
    if args.anchor_aspect_ratio_min > args.anchor_aspect_ratio_max:
        parser.error("--anchor-aspect-ratio-min must be <= --anchor-aspect-ratio-max")
    if args.g2_learn_every_frames < 1:
        parser.error("--g2-learn-every-frames must be >= 1")
    if args.g2_learn_epochs < 1:
        parser.error("--g2-learn-epochs must be >= 1")
    if args.g2_learn_min_pairs < 2:
        parser.error("--g2-learn-min-pairs must be >= 2")
    if args.g2_learn_lr_scale <= 0:
        parser.error("--g2-learn-lr-scale must be > 0")
    if args.g2_learn_max_time_gap <= 0:
        parser.error("--g2-learn-max-time-gap must be > 0")
    if args.g2_learn_max_pairs < 1:
        parser.error("--g2-learn-max-pairs must be >= 1")
    if args.g2_min_teacher_confidence < 0 or args.g2_min_teacher_confidence > 1:
        parser.error("--g2-min-teacher-confidence must be in [0, 1]")
    if (
        args.g2_min_teacher_confidence_gap < 0
        or args.g2_min_teacher_confidence_gap > 1
    ):
        parser.error("--g2-min-teacher-confidence-gap must be in [0, 1]")
    if args.g2_teacher_confidence_power <= 0:
        parser.error("--g2-teacher-confidence-power must be > 0")
    if args.g2_student_weight_floor < 0:
        parser.error("--g2-student-weight-floor must be >= 0")
    if args.g2_anchor_shape_eta < 0 or args.g2_anchor_shape_eta > 1:
        parser.error("--g2-anchor-shape-eta must be in [0, 1]")
    if args.g2_anchor_aspect_ratio_min <= 0 or args.g2_anchor_aspect_ratio_max <= 0:
        parser.error("--g2-anchor-aspect-ratio-min/max must be > 0")
    if args.g2_anchor_aspect_ratio_min > args.g2_anchor_aspect_ratio_max:
        parser.error("--g2-anchor-aspect-ratio-min must be <= --g2-anchor-aspect-ratio-max")
    if args.remerge:
        run_remerge(args)
        return
    run_tracking(args)


if __name__ == "__main__":
    main()
