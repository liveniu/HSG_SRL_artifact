#!/usr/bin/env python3
"""Copy paper-relevant sources from the development repo into this artifact.

The development tree is never modified. This script is optional after the
artifact has been populated; a standalone GitHub upload already contains the
vendored copies.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ARTIFACT = Path(__file__).resolve().parents[1]
UPSTREAM = ARTIFACT.parents[1]  # development repo when nested as supplementaries/artifact

SOURCE_FILES = [
    "pipeline/multi_camera_trajectory_fusion.py",
    "pipeline/geometry_guided_residual_mlp.py",
    "pipeline/residual_invoke.py",
    "pipeline/fit_residual_acceptor.py",
    "pipeline/g2_low_order_correction.py",
    "pipeline/multi_camera_bev_stitch.py",
    "evaluation/__init__.py",
    "evaluation/fab2_common.py",
    "evaluation/fab2_manifest.py",
    "evaluation/train_fab2_site_frozen.py",
    "evaluation/fit_fab2_low_order.py",
    "evaluation/eval_fab2_localization.py",
    "evaluation/eval_fab2_tracking.py",
    "evaluation/bev_hota.py",
    "evaluation/eval_fab2_bev_hota.py",
    "evaluation/eval_fab2_core_ablation.py",
    "evaluation/eval_fab2_kappa_strata.py",
    "evaluation/eval_fab2_mining_quality.py",
    "evaluation/eval_fab2_premise_audit.py",
    "evaluation/eval_fab2_calibration.py",
    "evaluation/eval_fab2_det_gt_match.py",
    "evaluation/eval_fab2_gt_quality_audit.py",
    "evaluation/eval_fab2_hyperparams.py",
    "evaluation/run_fab2_perturbation_recovery.py",
    "evaluation/profile_fab2_edge.py",
    "utils/project_paths.py",
    "utils/scene_geometry_db.py",
    "utils/trajectory_batches.py",
    "utils/g2_preflight_diagnostic.py",
    "utils/image_io.py",
    "utils/image_text.py",
    "detection/__init__.py",
    "detection/detect_track_backends.py",
    "detection/mmdet_yolox_infer.py",
    "config/botsort_static.yaml",
]

IMPORTERS = [
    "experiments/LUMPI/import_lumpi_mvc.py",
    "experiments/synthehicle/import_synthehicle_mvc.py",
]

SITE_CONFIGS = [
    "config/lumpi_M3.yaml",
    "config/lumpi_M6.yaml",
    "config/synth_Town04_O_day.yaml",
    "config/synth_Town05_O_day.yaml",
]

TC_SITES = ("lumpi_M3", "lumpi_M6", "synth_Town04_O_day", "synth_Town05_O_day")

FIGURE_SVGS = (
    "fig1_overall_hsg_srl_architecture_fab3.svg",
    "fig2_homography_sensitivity_guidance.svg",
    "fig5_geometric_localization.svg",
    "fig6_translation_gate_recovery.svg",
    "figd1_low_order_calibration_screening.svg",
)

NOTICE = (
    "# Vendored HSG-SRL artifact copy. Isolated from the development repo;\n"
    "# edit this file only. See src/SOURCE_MAP.csv.\n"
)

ABS_PATH_RE = re.compile(r"[A-Za-z]:\\\\[^\"'\\s]+|/home/[^\"'\\s]+|/Users/[^\"'\\s]+")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]")

SITE_YAML_HEADER = """# Paper-site configuration for the HSG-SRL artifact.
# Licensed videos and first-frame images are not redistributed.
# Replace the data/external placeholders after you download the dataset.
#
# Import: python scripts/import_paper_sites.py --site <site>
# Train/eval: scripts/train_paper_sites.py and scripts/eval_paper_tables.py
# Fusion reads cameras, overlap, and merge gates. It does not read
# residual_train (importer-era leftover; paper train uses --max-pair-dist-m 2).
# Camera names must match importer camera_name values.
# Homographies come from import (--h-source model).
"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _insert_notice(text: str) -> str:
    if "Vendored HSG-SRL artifact copy" in text:
        return text
    if text.startswith("#!"):
        first, rest = text.split("\n", 1)
        return first + "\n" + NOTICE + rest
    return NOTICE + text


def _patch_project_paths(text: str) -> str:
    return text.replace(
        "PROJECT_ROOT = Path(__file__).resolve().parents[1]",
        "# Artifact isolation: this file lives at src/utils/project_paths.py.\n"
        "PROJECT_ROOT = Path(__file__).resolve().parents[2]",
        1,
    )


def _patch_fab2_common(text: str) -> str:
    return text.replace(
        'FAB2_DEFAULT_RESULTS = _ROOT / "research" / "results" / "fab2"',
        "_ARTIFACT_ROOT = Path(__file__).resolve().parents[2]\n"
        "FAB2_DEFAULT_RESULTS = _ARTIFACT_ROOT / \"results\"",
        1,
    )


def _patch_acceptor_usage(text: str) -> str:
    return (
        text.replace("experiments_alpha/step2/yolo_lumpi_M6.db", "data/work/lumpi_M6.db")
        .replace(
            "experiments_alpha/step4/checkpoints/lumpi_M6_seed0.pt",
            "results/runs/lumpi_M6/training/frozen_checkpoints/lumpi_M6_seed0.pt",
        )
        .replace(
            "experiments_alpha/step4/manifests/lumpi_M6/manifest/dataset_split_manifest.csv",
            "splits/lumpi_M6/manifest/dataset_split_manifest.csv",
        )
        .replace(
            "experiments_alpha/step4/tc/lumpi_M6_Tc.json",
            "results/runs/lumpi_M6/training/lumpi_M6_preflight_Tc.json",
        )
        .replace(
            "experiments_alpha/step4/acceptor/lumpi_M6_seed0.json",
            "results/runs/lumpi_M6/acceptor/lumpi_M6_seed0.json",
        )
    )


def _patch_hota_docs(text: str) -> str:
    text = text.replace(
        "This is the scorer used by ``experiments_alpha/step4/tools/eval_site.py``.",
        "This is the recorded Table IV scorer (Hungarian TP at a BEV gate).",
    )
    text = text.replace(
        "This is the paper identity metric (same formulas as\n"
        "``experiments_alpha/step4/tools/eval_site.py``). It is *not*\n"
        "``eval_fab2_tracking.py`` (that file's ``hota_proxy``).",
        "This is the paper Table IV identity metric. It is *not*\n"
        "``eval_fab2_tracking.py`` (that file's ``hota_proxy``).",
    )
    return text


def _patch_recovery(text: str) -> str:
    """Keep Table IX scale/rotation from looking under the closed-repo data tree."""
    old = (
        '        _ROOT / "train-test_data" / "synthehicle_core" / "calibration"\n'
        '        / "overlapping" / town / "camera_info"'
    )
    if old in text:
        text = text.replace(old, '        _synthehicle_dataset_root(synthehicle_root) / "calibration"\n        / "overlapping" / town / "camera_info"')
    if "import os" not in text:
        text = text.replace("import argparse\n", "import argparse\nimport os\n", 1)
    if "def _synthehicle_dataset_root" not in text:
        needle = "ROT_DEG_PER_METER_DOSE = 0.4\n"
        helper = '''ROT_DEG_PER_METER_DOSE = 0.4


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

'''
        text = text.replace(needle, helper, 1)
    if "synthehicle-root" not in text:
        text = text.replace(
            'parser.add_argument("--dose-type", choices=("affine", "scale", "rotation"), default="affine")\n',
            'parser.add_argument("--dose-type", choices=("affine", "scale", "rotation"), default="affine")\n'
            '    parser.add_argument(\n'
            '        "--synthehicle-root",\n'
            '        default="",\n'
            '        help=(\n'
            '            "Synthehicle core root (calibration/overlapping/<Town>/camera_info). "\n'
            '            "Needed for --dose-type scale|rotation unless HSGSRL_SYNTHEHICLE_ROOT is set. "\n'
            '            "Table IX affine doses do not read this directory."\n'
            '        ),\n'
            '    )\n',
            1,
        )
    sig_old = (
        "def camera_ground_points_from_calib(\n"
        "    conn: sqlite3.Connection, scene: str, cameras: list[str]\n"
        ") -> dict[str, np.ndarray]:"
    )
    sig_new = (
        "def camera_ground_points_from_calib(\n"
        "    conn: sqlite3.Connection, scene: str, cameras: list[str],\n"
        "    *, synthehicle_root: str | Path | None = None,\n"
        ") -> dict[str, np.ndarray]:"
    )
    text = text.replace(sig_old, sig_new, 1)
    text = text.replace(
        "cam_grounds = camera_ground_points_from_calib(conn, scene, cameras)",
        "cam_grounds = camera_ground_points_from_calib(\n"
        "            conn, scene, cameras, synthehicle_root=args.synthehicle_root or None,\n"
        "        )",
    )
    return text


def _patch_importer(text: str, dataset: str) -> str:
    old_root = (
        "_ROOT = Path(__file__).resolve().parents[2]\n"
        f"_{dataset}_DIR = Path(__file__).resolve().parent\n"
        "if str(_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(_ROOT))\n"
    )
    new_root = (
        "_ARTIFACT_ROOT = Path(__file__).resolve().parents[2]\n"
        "_ROOT = _ARTIFACT_ROOT\n"
        "_SRC = _ARTIFACT_ROOT / \"src\"\n"
        f"_{dataset}_DIR = Path(__file__).resolve().parent\n"
        "if str(_SRC) not in sys.path:\n"
        "    sys.path.insert(0, str(_SRC))\n"
    )
    text = text.replace(old_root, new_root, 1)
    if dataset == "LUMPI":
        text = text.replace(
            '_ROOT / "train-test_data" / "LUMPI"',
            '_ROOT / "data" / "external" / "LUMPI"',
        )
    else:
        text = text.replace(
            '_ROOT / "train-test_data" / "synthehicle_core"',
            '_ROOT / "data" / "external" / "synthehicle_core"',
        )
    return text


def _copy_text(src: Path, dst: Path, patcher=None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8")
    if patcher is not None:
        text = patcher(text)
    if dst.suffix == ".py":
        text = _insert_notice(text)
    dst.write_text(text, encoding="utf-8", newline="\n")


def _sanitize_yaml(text: str) -> str:
    text = re.sub(
        r"(?m)^(    image: ).+$",
        r"\1data/external/media/<download-first-frame.png>",
        text,
    )
    text = re.sub(
        r"(?m)^(    video: ).+$",
        r"\1data/external/<licensed-dataset>/.../video.mp4",
        text,
    )
    text = re.sub(
        r"(?m)^(  db: ).+$",
        r"\1data/work/<site>.db",
        text,
    )
    text = re.sub(
        r"(?m)^(  data_root: ).+$",
        r"\1data/external/<licensed-dataset>",
        text,
    )
    text = re.sub(
        r"(?m)^(\s+checkpoint: ).+$",
        r"\1results/checkpoints/<site>_residual.pt",
        text,
    )
    kept: list[str] = []
    for line in text.splitlines(True):
        stripped = line.lstrip()
        if stripped.startswith("#") and CJK_RE.search(line):
            continue
        if "select_image_roi.py" in line:
            continue
        if re.match(r"^#\s*[=─━\-]+\s*$", line):
            continue
        if stripped.rstrip("\r\n") == "#":
            continue
        kept.append(line)
    body_lines = "".join(kept).splitlines()
    i = 0
    while i < len(body_lines) and (
        not body_lines[i].strip() or body_lines[i].lstrip().startswith("#")
    ):
        i += 1
    body = "\n".join(body_lines[i:]).strip() + "\n"
    notes = []
    if "lumpi_experiment:" in body:
        notes.append("# LUMPI merge gates are slightly looser than Synthehicle.")
    notes.append("# video/image paths are placeholders; image_roi is the fusion ground mask.")
    return SITE_YAML_HEADER + "\n" + "\n".join(notes) + "\n\n" + body


def _sanitize_json_paths(blob: dict) -> dict:
    def scrub(value):
        if isinstance(value, str):
            if ABS_PATH_RE.search(value) or "experiments_alpha" in value or "experiments_gamma" in value:
                name = Path(value.replace("\\", "/")).name
                return f"<local-input>/{name}"
            return value
        if isinstance(value, list):
            return [scrub(v) for v in value]
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items()}
        return value

    return scrub(blob)


def _extract_split_windows(upstream: Path, out_csv: Path) -> None:
    rows = []
    for site in TC_SITES:
        src = (
            upstream
            / "experiments_alpha"
            / "step4"
            / "manifests"
            / site
            / "manifest"
            / "dataset_split_manifest.csv"
        )
        if not src.is_file():
            continue
        stats: dict[str, dict[str, float]] = {}
        with src.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                split = (row.get("split") or "").strip()
                if split not in {"train", "val", "test"}:
                    continue
                try:
                    t = float(row["timestamp"])
                except (KeyError, TypeError, ValueError):
                    continue
                rec = stats.setdefault(
                    split,
                    {"n": 0, "t_min": t, "t_max": t},
                )
                rec["n"] += 1
                rec["t_min"] = min(rec["t_min"], t)
                rec["t_max"] = max(rec["t_max"], t)
        for split, rec in sorted(stats.items()):
            rows.append(
                {
                    "site": site,
                    "split": split,
                    "n_rows": int(rec["n"]),
                    "t_min_sec": f"{rec['t_min']:.6f}",
                    "t_max_sec": f"{rec['t_max']:.6f}",
                    "protocol": "contiguous_time_60_10_30",
                }
            )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["site", "split", "n_rows", "t_min_sec", "t_max_sec", "protocol"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _copy_topology(upstream: Path) -> None:
    dest_dir = ARTIFACT / "manifests" / "topology"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for site in TC_SITES:
        src = (
            upstream
            / "experiments_alpha"
            / "step4"
            / "manifests"
            / site
            / "manifest"
            / "homography_topology_manifest.csv"
        )
        if src.is_file():
            shutil.copy2(src, dest_dir / f"{site}_homography_topology.csv")


def _copy_tc(upstream: Path) -> None:
    dest = ARTIFACT / "manifests" / "preflight"
    dest.mkdir(parents=True, exist_ok=True)
    for site in TC_SITES:
        src = upstream / "experiments_alpha" / "step4" / "tc" / f"{site}_Tc.json"
        if not src.is_file():
            continue
        blob = json.loads(src.read_text(encoding="utf-8"))
        blob = _sanitize_json_paths(blob)
        (dest / f"{site}_preflight_Tc.json").write_text(
            json.dumps(blob, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )


def _copy_acceptor(upstream: Path) -> None:
    """Sanitized copies of the label-free residual-invoke acceptor JSONs.

    These record which cameras receive the frozen residual (Appendix C selector,
    defaults in Appendix E:
    N_min=32 validation pairs, mean-pair-distance tolerance 2%, then a
    deployed-mask D_S recheck). Val-only,
    no test GT and no GT identities, so they are safe to publish like the
    T_c preflight artifacts.
    """
    dest = ARTIFACT / "manifests" / "acceptor"
    dest.mkdir(parents=True, exist_ok=True)
    src_dir = upstream / "experiments_alpha" / "step4" / "acceptor"
    if not src_dir.is_dir():
        return
    for src in sorted(src_dir.glob("*_seed*.json")):
        blob = json.loads(src.read_text(encoding="utf-8"))
        blob = _sanitize_json_paths(blob)
        (dest / src.name).write_text(
            json.dumps(blob, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )


def _copy_result_summaries(upstream: Path) -> None:
    dest = ARTIFACT / "results" / "recorded_runs"
    dest.mkdir(parents=True, exist_ok=True)
    src_root = upstream / "experiments_alpha" / "step4" / "outputs"
    names = (
        "summary.md",
        "identity_metrics.csv",
        "subgraph_summary.md",
        "temporal_extrapolation.md",
        "subgraph_identity.csv",
    )
    for site in TC_SITES:
        site_src = src_root / site
        if not site_src.is_dir():
            continue
        site_dest = dest / site
        site_dest.mkdir(parents=True, exist_ok=True)
        for name in names:
            src = site_src / name
            if src.is_file():
                shutil.copy2(src, site_dest / name)


def main() -> int:
    if not (UPSTREAM / "pipeline" / "geometry_guided_residual_mlp.py").is_file():
        print(
            "Development repo not found next to this artifact; "
            "vendored sources are left unchanged.",
            file=sys.stderr,
        )
        return 0

    rows = []
    src_root = ARTIFACT / "src"
    src_root.mkdir(parents=True, exist_ok=True)

    patchers = {
        "utils/project_paths.py": _patch_project_paths,
        "evaluation/fab2_common.py": _patch_fab2_common,
        "pipeline/fit_residual_acceptor.py": _patch_acceptor_usage,
        "evaluation/bev_hota.py": _patch_hota_docs,
        "evaluation/eval_fab2_bev_hota.py": _patch_hota_docs,
        "evaluation/run_fab2_perturbation_recovery.py": _patch_recovery,
    }
    for rel in SOURCE_FILES:
        src = UPSTREAM / rel
        if not src.is_file():
            print(f"skip missing {rel}", file=sys.stderr)
            continue
        if rel.startswith("config/"):
            dst = ARTIFACT / "configs" / src.name
        else:
            dst = src_root / rel
        _copy_text(src, dst, patchers.get(rel))
        rows.append(
            {
                "original": rel.replace("\\", "/"),
                "artifact": str(dst.relative_to(ARTIFACT)).replace("\\", "/"),
                "sha256": _sha256(dst),
                "bytes": dst.stat().st_size,
            }
        )

    for rel in IMPORTERS:
        src = UPSTREAM / rel
        if not src.is_file():
            continue
        dst = ARTIFACT / rel
        dataset = "LUMPI" if "LUMPI" in rel else "SYNTH"
        _copy_text(src, dst, lambda text, d=dataset: _patch_importer(text, d))
        rows.append(
            {
                "original": rel.replace("\\", "/"),
                "artifact": str(dst.relative_to(ARTIFACT)).replace("\\", "/"),
                "sha256": _sha256(dst),
                "bytes": dst.stat().st_size,
            }
        )

    cfg_dest = ARTIFACT / "configs"
    cfg_dest.mkdir(parents=True, exist_ok=True)
    for rel in SITE_CONFIGS:
        src = UPSTREAM / rel
        if not src.is_file():
            continue
        text = _sanitize_yaml(src.read_text(encoding="utf-8"))
        dst = cfg_dest / src.name
        dst.write_text(text, encoding="utf-8", newline="\n")
        rows.append(
            {
                "original": rel.replace("\\", "/"),
                "artifact": str(dst.relative_to(ARTIFACT)).replace("\\", "/"),
                "sha256": _sha256(dst),
                "bytes": dst.stat().st_size,
            }
        )

    fig_src = UPSTREAM / "research" / "pics" / "fab2"
    fig_dest = ARTIFACT / "figures" / "sources"
    fig_dest.mkdir(parents=True, exist_ok=True)
    for name in FIGURE_SVGS:
        src = fig_src / name
        if src.is_file():
            shutil.copy2(src, fig_dest / name)
    # Keep a sibling copy for the closed-source IEEE zip layout, if nested.
    sibling = ARTIFACT.parents[0] / "figures" / "sources"
    if sibling != fig_dest:
        sibling.mkdir(parents=True, exist_ok=True)
        for name in FIGURE_SVGS:
            src = fig_dest / name
            if src.is_file():
                shutil.copy2(src, sibling / name)

    _extract_split_windows(UPSTREAM, ARTIFACT / "splits" / "split_windows.csv")
    _copy_topology(UPSTREAM)
    _copy_tc(UPSTREAM)
    _copy_acceptor(UPSTREAM)
    _copy_result_summaries(UPSTREAM)

    map_path = src_root / "SOURCE_MAP.csv"
    with map_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["original", "artifact", "sha256", "bytes"])
        writer.writeheader()
        writer.writerows(rows)
    hits = _cjk_hits(ARTIFACT)
    if hits:
        print(
            f"warning: {len(hits)} published files still contain CJK after vendoring; "
            "the GitHub snapshot must stay English (python scripts/check_english.py).",
            file=sys.stderr,
        )
        for rel, n in hits[:20]:
            print(f"  {rel}: {n} line(s)", file=sys.stderr)
    print(f"vendored {len(rows)} files into {ARTIFACT}")
    return 0


def _cjk_hits(root: Path) -> list[tuple[str, int]]:
    skip_dirs = {".git", "__pycache__", ".venv", ".idea", ".vscode", "external", "work"}
    skip_suffix = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".pt", ".pth", ".db", ".pyc"}
    hits: list[tuple[str, int]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() in skip_suffix:
            continue
        if any(part in skip_dirs for part in path.relative_to(root).parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        n = sum(1 for line in text.splitlines() if CJK_RE.search(line))
        if n:
            hits.append((path.relative_to(root).as_posix(), n))
    return hits


if __name__ == "__main__":
    raise SystemExit(main())
