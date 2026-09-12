# Recorded-run flags

This note records what the paper tables actually used. It is not a
library-default listing. Reproduce those values with
`scripts/train_paper_sites.py` and `configs/paper_defaults.yaml`.

## Residual MLP (principal method rows)

Every recorded principal train command passed these flags explicitly:

| Flag | Recorded value | Library argparse default |
| --- | --- | --- |
| `--lambda-reg` | **0.10** | 0.05 |
| `--lambda-mean-delta` | **1.0** | 0.0 (off) |
| `--lambda-pair-center` | **0.5** | 0.0 (off) |
| `--max-pair-dist-m` | **2** | 3.0 |
| `--max-residual-m` | **1.0** | 1.5 |
| `--min-mvc-pairs` | **64** (flag omitted; argparse default) | 64 |
| `--lambda-smooth` | 0.02 | 0.02 |

No recorded principal train used `--lambda-reg 0.05` or `--max-pair-dist-m 80`.

Evidence (closed development logs, not shipped with this artifact):

- The paper-site train driver hard-codes
  `--lambda-reg 0.10 --max-pair-dist-m 2 --max-residual-m 1.0`.
  It does not pass `--min-mvc-pairs`, so the trainer default 64 applies.
- Command line 1 of every recorded `train_*_seed*.log` on the four paper
  sites (plus a Town03 seed-0 attempt) matches those flags.
- Checkpoint `meta.train_config` on the four paper-site seed-0 weights:
  `lambda_reg=0.1`, `max_pair_dist_m=2.0`, `max_residual_m=1.0`.
- Ablation and recovery trains reuse the same three residual flags.

Table II and Appendix E therefore list `lambda_reg=0.10` because that is
the recorded principal-run value, not the unused argparse default 0.05.

The importer-era site YAML field `max_pair_dist_m: 80` was never
passed by the paper train drivers. It is leftover example text from
early LUMPI/Synthehicle importers. Pipeline fusion does not read
`residual_train`.

## T_c preflight (not residual training)

`evaluation/fit_fab2_low_order.py` was invoked without
`--max-pair-dist-m`. The recorded preflight JSON therefore stores
`"max_pair_dist_m": 3.0` (the fit-wrapper default).

This 3.0 m window is the coarse-to-fine translation-fit pair gate.
It is not the residual-MLP mining gate (2 m) and not the old 80 m
importer comment.

Perturbation-recovery first-round / wide-window T_c fits additionally
use `--tc-wide-window-m 10` on round 0, then fall back to 3.0 m on later
rounds. That is a Table IX procedure, not the principal Table III/IV
train recipe.

## Residual acceptor (Appendix C selector)

The paper-site train driver calls `pipeline/fit_residual_acceptor.py`
once per seed, right after that seed's checkpoint, with `--db`,
`--scene`, `--batch`, `--checkpoint`, `--split-manifest`, `--tc-json`,
`--seed`, `--out`. It never passes `--min-pairs`, so every recorded
acceptor JSON used the argparse default 32, matching $N_{\min}=32$ in
Appendix E. `scripts/train_paper_sites.py` in this artifact mirrors that
call (see `--min-val-pairs`, default 32). The acceptor JSON also stores
`deployed_mask` (\(D_S\) under the actual on/off set). On the four paper
sites the recheck does not change `residual_cameras`.

## What to reproduce

Use `scripts/train_paper_sites.py` or the flags in
`configs/paper_defaults.yaml`. Do not copy argparse defaults, and do
not copy leftover `max_pair_dist_m: 80` from `residual_train` in the
site YAML files.
