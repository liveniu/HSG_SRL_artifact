# Manifests

| File | Role |
| --- | --- |
| `ARTIFACT_MANIFEST.csv` | sha256 of published files (built by `scripts/build_manifest.py`) |
| `frozen_checkpoint_manifest.csv` | Index of the 12 principal-run checkpoints; weights are not shipped |
| `preflight/<site>_preflight_Tc.json` | Sanitized recorded translation-gate artifacts. Table VIII uses `diagnostics_by_scene[<site>].decision.status` (M6/M3/Town04 `recommended`; Town05 `not_recommended`, \(T_c=0\)). The top-level JSON `decision.status` may be `caution` when an affine pair warning is present; that diagnostic does not itself withhold \(T_c\). Pair-mean statistics on M6 use the Appendix D 5000-pair cap. |
| `acceptor/<site>_seed{N}.json` | Sanitized residual acceptor JSON (Appendix C selector): per-camera `apply`/`status`, val-only MVC statistics; no test GT, no GT identities |
| `topology/<site>_homography_topology.csv` | Homography source / scale / overlap edges. The recorded `geometry_version` string `fab1_metric_jacobian_v1` is a pipeline identifier, not a manuscript table. |
| `do_not_redistribute.txt` | License and size exclusions |

After you train locally, overwrite the checkpoint index with sha256,
`feature_version`, pair counts, and `max_residual_m = 1.0`.
