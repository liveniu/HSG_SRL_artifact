# Code defaults versus paper principal-run flags

The vendored copies keep the development-repo argparse defaults so the
extracted programs stay recognizable. The paper tables were produced by
**explicit flags**, not those defaults.

| Flag | Vendored argparse default | Paper / recorded principal run |
| --- | --- | --- |
| `--max-residual-m` | 1.5 | **1.0** (Table II) |
| `--max-pair-dist-m` | 3.0 | **2** |
| `--lambda-reg` | 0.05 | **0.10** (Table II and recorded commands) |
| `--lambda-mean-delta` | **0.0** (disabled) | **1.0** (Appendix E \(R_{\mathrm{mean}}\)) |
| `--lambda-pair-center` | **0.0** (disabled) | **0.5** (Appendix E \(R_{\mathrm{pair}}\)) |
| `--lambda-smooth` | 0.02 | **0.02** (Table II \(\lambda_s\)) |
| `--min-teacher-confidence` | 0.20 | 0.20 |
| `--min-teacher-confidence-gap` | 0.05 | 0.05 |
| `--risk-scalar` | metric_jacobian | metric_jacobian |
| `--anchor-mode` | multiplicative_gating_offsetted | multiplicative_gating_offsetted |
| `--loss-mode` | asymmetric_teacher | asymmetric_teacher |
| `--motion-compensation` | linear_interpolate | linear_interpolate |
| seeds | 0,1,2 | 0,1,2 |

The residual acceptor (`pipeline/fit_residual_acceptor.py`, Appendix C
selector) has no train-time discrepancy like the table above: its argparse
default `--min-pairs 32` already matches the paper's $N_{\min}=32$
(Appendix E), and its worsen-tolerance constant `WorsenTol=1.02` (2%,
$\epsilon_q=0.02$) is not exposed as a flag at all, so there is no argparse
default to diverge from. The deployed-mask recheck (`close_deployed_mask`)
uses the same tolerance and is not a separate hyperparameter.

`scripts/train_paper_sites.py` and `scripts/run_smoke_test.py` pass the
recorded principal-run values, including `--lambda-mean-delta 1.0` and
`--lambda-pair-center 0.5`. The frozen-training wrapper
`evaluation/train_fab2_site_frozen.py` also forwards those two paper
weights (its own argparse defaults are 1.0 / 0.5). The underlying trainer
in `pipeline/geometry_guided_residual_mlp.py` still defaults both to 0
(off); a naked `train` without flags is not a paper run.

The command CSV is the reproduction source of truth when a library argparse
default differs from the paper-explicit flag.

Stage-dependent pair windows (do not conflate):

- Residual MLP train: `--max-pair-dist-m 2` (recorded logs and checkpoints).
- T_c preflight: fit wrapper default **3.0** m (stored in `*_Tc.json`).
- Importer-era YAML comments with `80` were never used by paper train.
- See `docs/RECORDED_RUN_FLAGS.md` for the alpha/beta/gamma evidence.
