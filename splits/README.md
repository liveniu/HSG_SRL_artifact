# Dataset splits

The paper uses a contiguous time cut of **60% train / 10% val / 30% test**.
Adjacent frames of one trajectory stay on one side of the cut.

This folder ships:

- `split_protocol.json` — the protocol, independent of licensed labels
- `split_windows.csv` — per-site time ranges and row counts extracted from the
  recorded manifests, **without** GT coordinates or identities
- `smoke_split_manifest.csv` — synthetic smoke-test split (created by
  `scripts/init_smoke_data.py`)

Do **not** expect a full `dataset_split_manifest.csv` or `gt_index.csv` here.
Those files contain licensed or derived annotations. After you import a site:

```text
python src/evaluation/fab2_manifest.py ^
  --db data/work/<site>_gt.db ^
  --batch paper_yolo_001 ^
  --scene <site> ^
  --split-mode time ^
  --train-frac 0.60 ^
  --val-frac 0.10 ^
  --out-dir splits/<site>
```

Pass the resulting `gt_index.csv` to training and evaluation. Training uses
only the `split` and `source_detection_id` fields; GT columns stay in the
evaluation interface.
