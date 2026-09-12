# Manifests

| File | Role |
| --- | --- |
| `ARTIFACT_MANIFEST.csv` | sha256 of published files (built by `scripts/build_manifest.py`) |
| `frozen_checkpoint_manifest.csv` | Index of the 12 principal-run checkpoints; weights are not shipped |
| `preflight/<site>_preflight_Tc.json` | Sanitized recorded translation-gate artifacts |
| `acceptor/<site>_seed{N}.json` | Sanitized residual acceptor JSON (Appendix C selector): per-camera `apply`/`status`, val-only MVC statistics; no test GT, no GT identities |
| `topology/<site>_homography_topology.csv` | Homography source / scale / overlap edges |
| `do_not_redistribute.txt` | License and size exclusions |

After you train locally, overwrite the checkpoint index with sha256,
`feature_version`, pair counts, and `max_residual_m = 1.0`.
