# Later independent GitHub upload

This folder is the repository root for the public artifact. The parent
development tree is **not** published.

Suggested steps after acceptance:

1. Create an empty GitHub repository (name still open).
2. Push **this folder** as the repository root (`git init` here, or copy
   the folder). Do not upload the parent development repo.
3. Confirm `.gitignore` excludes `.idea/`, `data/external/`, `data/work/`,
   `results/runs/`, `results/smoke/`, `*.pt`, and `*.db` except
   `data/init/smoke_pair.db`.
4. Keep `data/init/smoke_pair.db`, `manifests/`, `results/tables/`,
   `results/recorded_runs/`, and `figures/sources/`.
5. Run `python scripts/check_english.py` before upload. The published
   tree must not contain CJK characters.
6. Replace the anonymous contact in `README.md` and `CITATION.cff` with
   authors, URL, and Zenodo DOI.
7. Do not add LUMPI or Synthehicle media. Fig. 2(b) stays a placeholder.

Blind-review uploads must keep authors, usernames, and absolute host paths
out of JSON and logs. The vendored preflight files already use
`<local-input>/...`.
