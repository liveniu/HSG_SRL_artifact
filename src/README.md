# Vendored source

Python packages copied from the development repository:

```text
pipeline/     HSG, residual MLP, translation preflight, BEV fusion
evaluation/   splits, training wrapper, localization and Table IV BEV-HOTA
utils/        path root, geometry DB, batch schema
detection/    optional YOLO / BoT-SORT backends
```

`SOURCE_MAP.csv` is written by `scripts/vendor_sources.py`. Isolation
patches are listed in `docs/ISOLATION.md`. The published copies are
English; refreshing from the closed tree can reintroduce Chinese comments.

Same-camera linker-control evaluation is intentionally absent. The vendored
fusion module still retains the off-by-default CLI switches inherited from
the development repository; principal-table reproduction keeps them off.
