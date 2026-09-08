# HSG-SRL reproducibility artifact

**Paper:** Homography-Sensitivity-Guided BEV Localization with Bounded Residual
Refinement for Measurement-Free Roadside Multi-Camera Vehicle Tracking

**Status:** anonymized supplementary / reproducibility package for peer review.
The camera-ready GitHub URL and DOI will be added after acceptance.

**Contact:** use the IEEE submission system for this manuscript.

This directory is self-contained and is intended to be uploaded later as an
independent GitHub repository. It does **not** modify the development tree
from which the sources were extracted. All published files are English.
Run `python scripts/check_english.py` before upload.

## 1. What this package contains

| Path | Role |
| --- | --- |
| `src/pipeline/` | HSG anchoring, residual MLP, translation preflight, BEV fusion |
| `src/evaluation/` | Split manifests, frozen training wrapper, localization/tracking eval |
| `src/utils/`, `src/detection/` | Shared geometry DB helpers and optional 2-D backends |
| `experiments/` | Isolated LUMPI / Synthehicle importers |
| `configs/` | Paper-site overlap, cameras, principal-run flags |
| `scripts/` | Smoke test, import, train, eval, checksums |
| `splits/` | Time-split protocol and recorded windows (no GT coordinates) |
| `manifests/` | Checkpoint index, sanitized preflight `T_c`, residual acceptor JSON (Appendix C selector), file checksums |
| `results/tables/` | Paper Tables II--X and Appendix B.1 |
| `figures/` | Manuscript SVG sources (Fig. 2(b) is a license placeholder) |
| `data/init/` | Synthetic smoke SQLite (created by `scripts/init_smoke_data.py`) |
| `docs/` | Dataset licenses, command map, isolation notes |

Approximate size after the synthetic init package is built: a few tens of
megabytes. Licensed videos and work databases are **not** included and can
add tens of gigabytes after download.

## 2. Platform

- OS: Windows 10/11, Linux, or macOS
- Python 3.10--3.12
- Optional GPU: CUDA-capable GPU for YOLO detection and residual training
- CPU is enough for the synthetic smoke test and HSG geometry
- Optional edge profile (Table X): NVIDIA Jetson Orin Nano Super 8GB

## 3. Install

```text
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements-minimal.txt
```

Install `requirements.txt` only when you will run detection or the full
four-site reproduction.

## 4. Smoke test (no licensed data)

```text
python scripts/init_smoke_data.py
python scripts/run_smoke_test.py
```

Expected output: `SMOKE PASS` and `results/smoke/smoke_report.json`.
The smoke package has two synthetic cameras, unlabeled MVC pairs, a 60/10/30
time split, and a 7-D HSG feature. It does not contain LUMPI or Synthehicle
content. If the shipped init DB already exists, `init_smoke_data.py` is a
no-op; use `--force` only to rebuild that tiny synthetic package.

The smoke test trains a 3-epoch seed-0 checkpoint and then fits the
label-free residual acceptor (Appendix C selector, `pipeline/fit_residual_acceptor.py`)
on the *val* split of that same fixture, writing
`results/smoke/smoke_acceptor_seed0.json`. The reported per-camera status
(`recommended` / `thin_val` / `not_recommended`) can vary with the tiny
fixture; any of the three is a valid selector outcome, not a smoke-test
failure -- only a non-zero exit code is.

Typical runtime: a few minutes on CPU.

## 5. Four-site reproduction route

1. Accept the LUMPI and Synthehicle licenses and download the original data
   (see `docs/DATASETS.md`).
2. Import origin databases:

```text
set HSGSRL_LUMPI_ROOT=...
set HSGSRL_SYNTHEHICLE_ROOT=...
python scripts/import_paper_sites.py --site lumpi_M6
```

3. Run detection and local tracking once per camera, then remerge with
   `src/pipeline/multi_camera_trajectory_fusion.py` and
   `configs/<site>.yaml`. Same-camera tracklet linking stays **off**.
4. Build a frozen split / GT index with
   `src/evaluation/fab2_manifest.py --split-mode time --train-frac 0.60 --val-frac 0.10`.
5. Train the frozen residual (paper-explicit flags). Per seed this also
   fits the label-free residual acceptor (Appendix C selector) on the *val* split
   and writes a remerged, evaluable DB copy with `final_world_x/y` = P_final
   (checkpoint + accepted `T_c` + that seed's acceptor JSON):

```text
python scripts/train_paper_sites.py --site lumpi_M6 --db data/work/yolo_lumpi_M6.db --split-manifest splits/lumpi_M6/manifest/gt_index.csv
```

   This writes, per seed: `results/runs/lumpi_M6/training/frozen_checkpoints/lumpi_M6_seed{N}.pt`,
   `results/runs/lumpi_M6/acceptor/lumpi_M6_seed{N}.json`, and
   `results/runs/lumpi_M6/eval/lumpi_M6_seed{N}.db`. Pass `--skip-acceptor` /
   `--skip-remerge` to stop earlier (e.g. checkpoint-only reproduction).
6. Print or run the core evaluation commands. The HSG-Geo row reads
   `world_x/y` (P_geo) straight from the detection DB; the HSG-SRL-Frozen
   ("ours") row reads `final_world_x/y` (P_final) from the seed's remerged
   eval DB from step 5. Table IV additionally needs the origin GT sqlite and
   the det-gt match report:

```text
python scripts/eval_paper_tables.py --site lumpi_M6 --db data/work/yolo_lumpi_M6.db --seed 0 --gt-index splits/lumpi_M6/manifest/gt_index.csv --origin-db data/work/origin_lumpi_M6.db --match-report results/runs/lumpi_M6/det_gt_match_report.json
```

Add `--execute` only after the licensed databases exist. Without `--execute`
the script prints the command mapping. This wrapper covers Tables III, IV,
V/VII, and IX (HSG-Geo and HSG-SRL-Frozen rows only; Table IV needs
`--origin-db` and `--match-report`. Baseline-IPM needs the
`anchor_mode=bottom_center` recomputation path in `docs/CODE_TO_PAPER.md`,
not a stored DB column). Table VI, Table VIII, and Table X use the
specialized entries listed in `results/tables/TABLE_TO_SCRIPT.csv`.

Recorded principal-run summaries (no raw frames) are under
`results/recorded_runs/`.

## 6. Paper-explicit flags versus library defaults

The vendored argparse defaults still say `--max-residual-m 1.5`,
`--max-pair-dist-m 3.0`, and `--lambda-reg 0.05`. **Do not use those defaults
to reproduce the paper.** The recorded principal runs pass:

- `--max-residual-m 1.0` (Table II)
- `--max-pair-dist-m 2`
- `--lambda-reg 0.10`
- `--risk-scalar metric_jacobian`
- `--anchor-mode multiplicative_gating_offsetted`
- `--loss-mode asymmetric_teacher`
- seeds `0,1,2`

The residual acceptor (Appendix C selector, defaults in Appendix E;
`pipeline/fit_residual_acceptor.py`) has its own two constants, orthogonal
to the flags above: `N_min = 32` val MVC
pairs (`--min-pairs`, thin-val rule) and a 2% mean-pair-distance worsening
tolerance (`WorsenTol = 1.02`, not currently exposed as a flag). A camera on
the checkpoint's training overlap graph with fewer than `N_min` val pairs is
`status=thin_val` and gets \(\Delta P = 0\); a camera with enough pairs whose
residual field worsens *mean* val pair distance by more than 2% is
`status=not_recommended` and also gets \(\Delta P = 0\). After the per-camera
tests, the selector recomputes pooled disagreement \(D_S\) with \(\Delta P\)
only on the selected cameras and drops cameras if that combination worsens
the mean; on the four paper sites this recheck does not change
`residual_cameras`. p90/median are recorded as diagnostics only, not kill
switches.

See `configs/paper_defaults.yaml`, `docs/CODE_DEFAULTS_VS_PAPER.md`, and
`docs/RECORDED_RUN_FLAGS.md`.

## 7. Outputs and paper map

| Paper item | Artifact file |
| --- | --- |
| Tables II--X, B.1 | `results/tables/` |
| Command log | `results/runs/<site>/train_fab2_site_frozen_commands.csv` |
| Checkpoint index | `manifests/frozen_checkpoint_manifest.csv` |
| Translation gate | `manifests/preflight/<site>_preflight_Tc.json` |
| Residual acceptor (Appendix C selector) | `manifests/acceptor/<site>_seed{N}.json` |
| Per-seed evaluable DB (P_final) | `results/runs/<site>/eval/<site>_seed{N}.db` |
| File checksums | `manifests/ARTIFACT_MANIFEST.csv` |

GT is used only for evaluation, teacher-premise audit, and external
reference. It is never used for MVC mining, low-order preflight, residual
training, or model selection.

## 8. Checkpoints

Frozen `*.pt` weights are **not** redistributed. After training, fill
`manifests/frozen_checkpoint_manifest.csv` with sha256, `feature_version`,
`risk_scalar_mode`, `anchor_mode`, and pair counts. New checkpoints start as
pure geometric fallback (`tanh` residual near zero).

A checkpoint alone does not decide which cameras run with \(\Delta P\)
active at inference/remerge time. That is a separate, label-free decision
made by the residual acceptor JSON (Appendix C selector; see Section 6),
which *is*
safe to redistribute (val-split MVC statistics only, no test GT, no GT
identities). Pass it to `pipeline/multi_camera_trajectory_fusion.py
--remerge` as `--g2-acceptor-json`; without it, fusion still zeros
\(\Delta P\) on cameras absent from the checkpoint's training overlap
graph, but does not apply the val-quality check.

## 9. Expected runtime and failures

| Step | Typical time |
| --- | --- |
| Smoke test | minutes |
| Import one site | tens of minutes |
| Detection + remerge | hours, GPU recommended |
| Residual train, 3 seeds | tens of minutes to a few hours per site |
| Table IX recovery | hours per site |

Common failures:

- licensed data missing → set `HSGSRL_LUMPI_ROOT` / `HSGSRL_SYNTHEHICLE_ROOT`
- path mismatch → run scripts from this artifact root; do not point at the
  development repo
- GPU unavailable → detection/train need `requirements.txt`; smoke can use CPU
- GT identity leakage guard → training stops if `global_id` matches GT; use
  pseudo identities from geometry-only remerge
- preflight `not_recommended` → deploy `T_c = 0`; do not force a translation

## 10. Isolation and later GitHub upload

Sources under `src/` and `experiments/` are copies. Isolation patches are
limited to path roots (`src/utils/project_paths.py`, default result directory,
importer `data/external/` roots). See `docs/ISOLATION.md` and
`src/SOURCE_MAP.csv`. Run `python scripts/check_english.py` before upload.

To publish this folder as its own repository, `git init` here (or copy
this folder) and push that root. The parent development tree is closed
source and must not be uploaded. Do not include `data/external/`, work
databases, licensed frames, `.idea/`, or `*.pt`.
