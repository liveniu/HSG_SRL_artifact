# Results

`tables/` holds the manuscript numbers for Tables II--X and Appendix B.1.
`TABLE_TO_SCRIPT.csv` maps each item to the evaluation entry that regenerates
it.

`recorded_runs/` stores the principal-run markdown and identity CSVs from
the four paper sites. Those files contain aggregate metrics, not raw
frames or GT coordinates.

`smoke/` is local output from `scripts/run_smoke_test.py` and is gitignored.
Regenerate it after checkout; do not commit `smoke_report.json` or the
smoke acceptor JSON.
