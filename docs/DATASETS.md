# External datasets

This artifact does **not** redistribute LUMPI or Synthehicle frames, videos,
LiDAR, or official annotations. Download them under the original licenses and
point the importers at your local copies.

## LUMPI

- Paper: S. Busch, C. Koetsier, J. Axmann, and C. Brenner, "LUMPI: The Leibniz
  University Multi-Perspective Intersection Dataset," IEEE IV, 2022.
  doi: [10.1109/IV51971.2022.9827157](https://doi.org/10.1109/IV51971.2022.9827157)
- Official dataset page and DOI:
  [data.uni-hannover.de/en/dataset/lumpi](https://data.uni-hannover.de/en/dataset/lumpi),
  [10.25835/z54qcu1b](https://doi.org/10.25835/z54qcu1b).
- License on the official page: Creative Commons Attribution-NonCommercial
  3.0.
- Download only from the official host, for example:

```text
https://data.uni-hannover.de/vault/ikg/busch/LUMPI/
```

- Paper sites: Measurement 6 (cameras C05, C06, C07) and Measurement 3
  (cameras C08, C09, C10).
- Expected root layout after download:

```text
<LUMPI>/
  meta.json
  Label/Measurement3/Label.csv
  Label/Measurement6/Label.csv
  Measurement3/cam/{8,9,10}/video.mp4
  Measurement6/cam/{5,6,7}/video.mp4
```

```text
set HSGSRL_LUMPI_ROOT=<LUMPI>
python scripts/import_paper_sites.py --site lumpi_M6
python scripts/import_paper_sites.py --site lumpi_M3
```

The importer uses `--h-source model` (calibrated ground-plane homography).
Do not use a data-fitted homography for the principal tables.

## Synthehicle Core

- Paper: F. Herzog et al., "Synthehicle: Multi-vehicle multi-camera tracking
  in virtual cities," WACVW, 2023.
- Official download page:
  [github.com/fubel/synthehicle/wiki/Download](https://github.com/fubel/synthehicle/wiki/Download).
- License on the official download page: Creative Commons
  Attribution-NonCommercial-ShareAlike 4.0.
- Download the core package from the official links listed on that page. The
  current direct core-data URL is:

```text
https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=18e2eac4&path=Datenbanken%5CSynthehicle%5Csynthehicle_core.tar.gz
```

- Paper sites: overlapping daytime Town04 and Town05 (`Town04-O-day`,
  `Town05-O-day`).
- Expected root:

```text
<synthehicle_core>/
  calibration/overlapping/Town04/camera_info/
  calibration/overlapping/Town05/camera_info/
  train/Town04-O-day/C01/...
  train/Town05-O-day/C01/...
```

```text
set HSGSRL_SYNTHEHICLE_ROOT=<synthehicle_core>
python scripts/import_paper_sites.py --site synth_Town04_O_day
python scripts/import_paper_sites.py --site synth_Town05_O_day
```

The importer uses `--h-source model` and `--bbox-source gt_tight`. Amodal
`out_bbox` boxes are not the paper 2-D input.

Table IX affine doses do not read `camera_info`. `--dose-type scale` or
`rotation` needs the same `HSGSRL_SYNTHEHICLE_ROOT` (or
`--synthehicle-root`) so `calibration/overlapping/<Town>/camera_info`
is visible.

## Protocol notes

- Ground-truth identities and 3-D coordinates are evaluation-only.
- MVC mining uses pseudo-global identities from geometry-only fusion.
- Clipped boxes are dropped from world-coordinate use and MVC mining.
- Same-camera tracklet linking stays off for the principal tables.

## Files that must not be committed to the public artifact repo

See `manifests/do_not_redistribute.txt`. After download, keep licensed data
under `data/external/` (gitignored) or an external path via the environment
variables above.
