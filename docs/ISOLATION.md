# Isolation from the development repository

This artifact is a **copy-and-wrap** of the paper-relevant programs. The
development repository is not imported at runtime and is not modified by
these files.

## Standalone GitHub root

Treat this folder as the repository root. Smoke, import, train, and eval
wrappers resolve paths from here (`scripts/bootstrap.py`: `ARTIFACT_ROOT`,
`SRC_ROOT = artifact/src`). They do not need `experiments_alpha`,
`train-test_data`, or the closed development tree.

Licensed LUMPI / Synthehicle media stay under `data/external/` after you
download them (`docs/DATASETS.md`). Work databases go under `data/work/`.

## What was copied

See `src/SOURCE_MAP.csv` after `scripts/vendor_sources.py` runs. The map
lists original relative paths, artifact destinations, and sha256 hashes.
English-normalized copies can differ from a later vendor refresh.

Included:

- HSG / fusion / residual / preflight modules
- FAB2 evaluation tools used by Tables III--IX and Table X
  (`eval_fab2_bev_hota.py` is the Table IV scorer; `eval_fab2_tracking.py`
  is a diagnostic `hota_proxy` only)
- LUMPI and Synthehicle importers
- Paper-site YAML cameras, overlap, and merge gates

Not copied:

- `evaluation/eval_fab2_linker_control.py` (not a principal-table route;
  same-camera tracklet linking is closed in the main manuscript)
- `experiments_*` exploration trees
- work databases, YOLO caches, and frozen `*.pt` weights
- full GT manifests
- `tools/print_ChArUco.py` and `pipeline/select_image_roi.py`
- YOLO26n / BEVHeight weights and the Fig. 2(b) LUMPI overlay generator

## Patches applied only to copies

1. `src/utils/project_paths.py` — `PROJECT_ROOT` is the artifact root
   (`parents[2]`), not `src/`.
2. `src/evaluation/fab2_common.py` — default results directory is
   `artifact/results`.
3. Importers — `sys.path` includes `src/`; default data roots are
   `data/external/LUMPI` and `data/external/synthehicle_core`.
4. Site YAMLs — licensed video/image paths replaced by placeholders;
   Chinese comments dropped.
5. Table IX scale/rotation doses read Synthehicle `camera_info` from
   `HSGSRL_SYNTHEHICLE_ROOT` / `--synthehicle-root` /
   `data/external/synthehicle_core` (not `train-test_data`). Affine
   doses (the Table IX principal class) do not need that directory.

No other behavioral rewrites are applied. Paper-explicit hyperparameters
are injected by `scripts/`, not by changing argparse defaults in the copies.

Wrappers such as `scripts/train_paper_sites.py` set the subprocess cwd to
`src/` so vendored packages import. Relative YAML `output:` / `debug_dir`
paths still resolve through `PROJECT_ROOT` (the artifact root), not cwd.

## English-only snapshot

The published tree must not contain CJK characters. Gate:

```text
python scripts/check_english.py
```

Refreshing copies from the closed development tree can reintroduce
Chinese comments. Do not publish a failed English check. Re-apply the
English pass (or skip `vendor_sources.py` after publication).

## How to refresh copies

From a checkout that still nests this folder as
`supplementaries/artifact` inside the development repo:

```text
python scripts/vendor_sources.py
python scripts/check_english.py
```

After the artifact is published as its own GitHub repository, that script
becomes a no-op unless the development tree is placed next to it again.
