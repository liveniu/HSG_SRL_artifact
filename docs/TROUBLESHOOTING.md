# Troubleshooting

**Missing licensed data.** The smoke test must pass without LUMPI or
Synthehicle. Four-site commands fail with exit code 2 if
`HSGSRL_LUMPI_ROOT` / `HSGSRL_SYNTHEHICLE_ROOT` are unset and
`data/external/` is empty.

**Import path errors.** Run wrappers from the artifact root. Vendored
modules live under `src/`. `scripts/bootstrap.py` puts `src/` on
`sys.path`. Do not add the development-repo root.

**GT identity leakage.** Training stops if mined pairs use official GT
identities. Remerge on `P_geo` first so `global_id` is a pseudo identity.
`--allow-gt-identity-mining` is a supervised smoke hatch only.

**Split manifest required.** Unsplit training is legacy diagnostics and is
not a paper principal-run path.

**Preflight rejected.** Town05 clean-frame exit is no-op. A
`not_recommended` status must deploy `T_c=0`. Affine dominance is a routing
signal toward homography re-estimation, not a license to apply an affine
model.

**Residual acceptor: `thin_val` / `insufficient` / `not_recommended`.**
These are valid outcomes of the Appendix C selector, not failures.
`thin_val` means the camera is on the checkpoint's training overlap graph
but has fewer than `N_min=32` validation MVC pairs; `insufficient` means
the same but the camera is also off that graph; `not_recommended` means
enough val pairs but the residual worsens mean val pair distance by more
than 2%. All three set \(\Delta P=0\) for that camera. Only a non-zero
`fit_residual_acceptor.py` exit code (e.g. the GT-identity-leakage guard,
or the "no overlap graph" `SystemExit`) is an actual failure. Do not pass
`--g2-force-all-residual` to work around a `thin_val`/`not_recommended`
status on the principal per-site tables; it is a diagnostic-only override
(used for the Table VI forced-residual check).

**Torch missing.** HSG geometry can be imported without a successful train.
Install `requirements-minimal.txt` for the smoke residual step.

**Orin / Table X.** `profile_fab2_edge.py` is the paper tool. Live
`--stage-profile` in fusion looks for optional instrumentation that is not
required for Tables III--IX.

**Windows paths.** Recorded preflight JSON files in `manifests/preflight/`
have local paths replaced by `<local-input>/...`.

**Table IX scale/rotation doses.** Affine (the paper principal class) does
not need Synthehicle `camera_info`. `--dose-type scale` or `rotation`
needs `HSGSRL_SYNTHEHICLE_ROOT` or `--synthehicle-root` pointing at the
dataset root (`calibration/overlapping/<Town>/camera_info`).
