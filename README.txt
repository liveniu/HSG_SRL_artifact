HSG-SRL reproducibility artifact
================================

Paper: Homography-Sensitivity-Guided BEV Localization with Bounded Residual
Refinement for Measurement-Free Roadside Multi-Camera Vehicle Tracking

Public repository: https://github.com/liveniu/HSG_SRL_artifact
Contact: Jin Niu, Zhejiang University (LICENSE / CITATION.cff).

Quick start
-----------
python -m venv .venv
pip install -r requirements-minimal.txt
python scripts/init_smoke_data.py
python scripts/run_smoke_test.py

Four-site reproduction requires licensed LUMPI and Synthehicle downloads.
See README.md and docs/DATASETS.md.

IEEE README sections
--------------------
Description: core HSG-SRL source, paper configs, split protocol, command
wrappers, sanitized translation-gate JSON, and paper table archives.
Size: tens of MB without licensed data; licensed videos are much larger.
Platform: Python 3.10-3.12; optional CUDA GPU; optional Jetson Orin for Table X.
Environment: see requirements.txt and requirements-minimal.txt.
Major components: src/pipeline, src/evaluation, scripts, configs, results/tables.
Setup: create a venv and install requirements; download datasets separately.
Run: scripts/run_smoke_test.py; scripts/train_paper_sites.py; scripts/eval_paper_tables.py.
Output: results/smoke/smoke_report.json and results/runs/<site>/.
Contact: Jin Niu, Zhejiang University.
