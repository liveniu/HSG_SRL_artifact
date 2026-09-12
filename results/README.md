# Results

`tables/` holds the manuscript numbers for Tables II--X and Appendix B.1,
plus a compact Appendix F median archive (`appendix_f_frozen_homography.csv`).
The closed-repo raw dump `frozen_homography_perturbation.csv` is not shipped.
`TABLE_TO_SCRIPT.csv` maps each item to the evaluation entry that regenerates
it.

`recorded_runs/` stores the principal Frozen markdown and identity CSVs
from the four paper sites (Tables III--IV, plus the M6 C05--C06 footnote
and Table VI later-C06 temporal row). See `recorded_runs/README.md`.
Those files contain aggregate metrics, not raw frames or GT coordinates.
Tables V, VII, VIII--X, Table VI spatial slices, and BEVHeight live only in
`tables/`.

`smoke/` is local output from `scripts/run_smoke_test.py` and is gitignored.
Regenerate it after checkout; do not commit `smoke_report.json` or the
smoke acceptor JSON.
