# Public GitHub root

This folder is the repository root for
https://github.com/liveniu/HSG_SRL_artifact. The parent development tree
is **not** published. The companion manuscript is a named IEEE submission
(not double-blind).

Keep the published tree as follows:

1. Push **this folder** as the repository root. Do not upload the parent
   development repo.
2. Confirm `.gitignore` excludes `.idea/`, `data/external/`, `data/work/`,
   `results/runs/`, `results/smoke/`, `*.pt`, and `*.db` except
   `data/init/smoke_pair.db`.
3. Keep `data/init/smoke_pair.db`, `manifests/`, `results/tables/`,
   `results/recorded_runs/`, and `figures/sources/`.
4. Run `python scripts/check_english.py` before upload. The published
   tree must not contain CJK characters.
5. Do not add LUMPI or Synthehicle media. Fig. 2(b) stays a placeholder.

Absolute host paths stay out of JSON and logs. The vendored preflight
files already use `<local-input>/...`.
