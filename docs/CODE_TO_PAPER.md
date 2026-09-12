# Code-to-paper map

Paper numbering (fab3): **Algorithm 1** is HSG anchor construction
(Section III-C). **Algorithm D.1** is the translation gate (Appendix D).
There is no Algorithm 2 in the manuscript; the offline train/freeze
pipeline below is the prose procedure in Sections III-D--III-F.

## Offline train / freeze pipeline (Sections III-D--III-F)

| Pseudocode | Artifact entry |
| --- | --- |
| BuildHSGObservations | `pipeline/geometry_guided_residual_mlp.py` `compute_geometry_base`, `GeometryDataView.iter_observations_from_db` |
| MineMVCPairs / AlignAndFilter | `GeometryDataView.mine_mvc_pairs`, `interpolate_anchor_feature` |
| FixedTranslationPreflight | `evaluation/fit_fab2_low_order.py`, `pipeline/g2_low_order_correction.py` |
| TrainResidualMLP | `ResidualMLPModel.train_offline` via `scripts/train_paper_sites.py` (not `evaluation/train_fab2_site_frozen.py` alone: that wrapper must forward Appendix E `--lambda-mean-delta 1.0 --lambda-pair-center 0.5`) |
| FitResidualAcceptor (Appendix C selector; defaults in Appendix E) | `pipeline/fit_residual_acceptor.py` via `scripts/train_paper_sites.py` (per-seed, after the checkpoint) |
| RunBEVFusion | `pipeline/multi_camera_trajectory_fusion.py` `run_remerge` |

## Algorithm 1 — HSG anchor (`BuildHSGAnchor`)

| Symbol | Function |
| --- | --- |
| `s = ||J_px diag(Wf,Hf)||_F` | `homography_metric_jacobian`, `metric_sensitivity_from_homography`, `compute_risk_scalar` |
| `alpha`, `kappa=1-alpha` | `RiskScalar` |
| `beta` / anchor | `gated_anchor_beta`, `box_anchor` |
| `F` 7-D | `build_feature_vector` / `observation_to_features` |

Contracts that must stay true:

- Risk is sampled at the box center, not the sliding anchor.
- Default `risk_scalar_mode="metric_jacobian"`.
- Non-ground or degenerate projections set `alpha=1`, `kappa=0` and cannot be
  MVC teachers.
- Clipped boxes are dropped (`is_bbox_clipped`).
- `F` has no camera id, height, pitch, or intrinsics.

## Algorithm D.1 — translation gate

`pipeline/g2_low_order_correction.py` (`fit_translations`,
`_assess_correction`) and `evaluation/fit_fab2_low_order.py` (coarse-to-fine).
Diagnostic-only helper: `utils/g2_preflight_diagnostic.py`.

`diagnostics_by_scene[site].decision.status` is the paper gate used by
training. The top-level JSON `decision` may be more conservative when an
affine pair warning is present. Affine dominance is a diagnostic flag;
it does not itself set fit status. `recommended` is a usable candidate;
`caution` needs explicit review; `not_recommended` must not be deployed
(`T_c=0`). For the M6 results in Table VIII, fitting and gate statistics
use a subset capped at 5000 training pairs after filtering (Appendix D).

Training-time `--camera-translation-json` and runtime
`--g2-camera-translation-json` are independent.

## Appendix C — label-free residual invocation (residual selector)

`\Delta P` (the frozen residual head's output) is applied per camera, not
globally. Coverage and the val-quality check are two separate gates.
Numerical defaults ($N_{\min}=32$, $\epsilon_q=0.02$) are Appendix E.

| Pseudocode / rule | Artifact entry |
| --- | --- |
| Coverage: camera must be on the checkpoint's *training* overlap graph | `pipeline/residual_invoke.py` `overlap_from_meta`, `cameras_on_overlap` |
| Thin-val rule: `n_val_pairs < N_min=32` -> status=`thin_val` (on graph) or `insufficient` (not on graph), \(\Delta P=0\) | `pipeline/fit_residual_acceptor.py` `decide_camera` |
| Val quality: mean val MVC pair distance must not worsen vs. T_c-only by more than 2% (`WorsenTol=1.02`) | `pipeline/fit_residual_acceptor.py` `worsens_mean`, `decide_camera` |
| Deployed-mask recheck: after per-camera tests, recompute pooled \(D_S\) with \(\Delta P\) only on selected cameras; drop cameras if the combination worsens the mean | `pipeline/fit_residual_acceptor.py` `distances_under_mask`, `close_deployed_mask` |
| p90 / median pair distance | diagnostics only, never a kill switch (`tail_notes`) |
| Load an acceptor JSON at runtime | `pipeline/residual_invoke.py` `load_residual_acceptor_json`; `geometry_guided_residual_mlp.py` `configure_residual_invoke`, `residual_delta_allowed` |
| Runtime CLI | `pipeline/multi_camera_trajectory_fusion.py --g2-acceptor-json <path>` (mutually exclusive with `--g2-force-all-residual`, which is a diagnostic-only override) |

Contracts that must stay true:

- The acceptor is fit on the *val* split only; it must not read test keys,
  GT coordinates, or GT identities (`fit_residual_acceptor.py` raises if the
  mined `global_id` looks like a leaked GT identity).
- Coverage uses the overlap graph recorded in the checkpoint (the graph the
  MLP was actually trained on), not the yaml fusion triangle or the DB's
  full `scene_overlap_pairs`.
- Without an acceptor JSON, fusion still zeros \(\Delta P\) on cameras that
  are not on the checkpoint's training overlap graph, but does not apply
  the val mean-pair-distance check.
- `--g2-force-all-residual` is a diagnostic override (e.g. for the ablation
  in Table VI) and must not be used for the principal per-site tables.

## Appendix F — frozen homography perturbation

The roll/pitch stress diagnostics in Appendix F are produced by
`evaluation/eval_fab2_calibration.py` (`frozen_homography_perturbation.csv`).
They are not a principal table; Section IV-E keeps them separate from
Table IX. Constant-dose recovery (Table IX) remains
`evaluation/run_fab2_perturbation_recovery.py`.

## Unified BEV fusion

`TrajectoryFusion.merge_active`, `_union_allowed`, `_overlap_gate`,
`run_remerge`. Same-camera stitching stays off for principal tables.
`run_remerge` writes `final_world_x/y` (P_final) when `--g2-checkpoint` is
given; `--g2-acceptor-json` gates \(\Delta P\) per camera as above.

## Evaluation

| Paper | Tool |
| --- | --- |
| Table III | `eval_fab2_localization.py` (`--world-coords p_geo` for HSG-Geo, `--world-coords final` against the remerged eval DB for HSG-SRL-Frozen; see `scripts/eval_paper_tables.py`) |
| Table IV | `eval_fab2_bev_hota.py` (Hungarian TP at the BEV gate, LocA = mean \(1-d/\mathrm{gate}\) on TPs, HOTA \(=\sqrt{\mathrm{DetA}\cdot\mathrm{AssA}}\), BEV-HOTA = mean over 0.5/1/1.5/2 m). `eval_fab2_tracking.py` is a coarser `hota_proxy` and must not be treated as Table IV. |
| Tables V, VII | `eval_fab2_core_ablation.py` |
| Table VI / kappa slices | `eval_fab2_kappa_strata.py` |
| Table VI residual scope / acceptor status | `pipeline/fit_residual_acceptor.py` output JSON (`manifests/acceptor/<site>_seed{N}.json`); see Appendix C above |
| Tables VIII--IX | preflight JSON + `run_fab2_perturbation_recovery.py` |
| Table X | `profile_fab2_edge.py` |
| Appendix B.1 | `eval_fab2_premise_audit.py` |
| Appendix F | `eval_fab2_calibration.py` (`frozen_homography_perturbation.csv`) |

Table IV needs the origin GT sqlite (`--gt-db`), a split manifest with
test keys, and `det_gt_match_report.json` (`--match-report`) for camera
time maps. Score `world` on the geometric baseline remerge
(`<site>_baseline.db`) and `final_world` on each seed eval DB. A common
GT offset is applied only when `reference_verdict.txt` starts with
`REFERENCE OFFSET` (M6 recorded as `NO COMMON OFFSET` and is not shifted).
M6 uses `--max-t 360`. `--cameras C05,C06` is the Table IV footnote.

Baseline-IPM localization is not a stored DB column: recompute with
`anchor_mode=bottom_center` (HSG feature construction with that anchor).
