# Deep Learning Based Defect Detection and Rule-Based Correction Framework for FDM 3D Printing

**Course:** CSCI 6379 · **Team plan:** [plan/FDM_Team_Work_Plan_v2.pdf](plan/FDM_Team_Work_Plan_v2.pdf) (two-week sprint, Oct 23 – Nov 6, 2026)

| Person | Owns | RQ |
|---|---|---|
| **Hena** (CS) | Data pipeline, model training, simulation, RQ3 agreement evaluation; report: Data, Experimental Setup, RQ3 results | RQ1/RQ2 training, RQ3 evaluation |
| **Harris** (CS) | Review of the model training and results; report: Methods, RQ1/RQ2 results (starting from `reports/MODEL_RESULTS.md`) | RQ1, RQ2 write-up |
| **Agya** (CS) | Class audit, defect definitions, correction rule table; report: Background, Related Work, rule discussion | RQ3 rules |

## Folders

```
├── README.md                 start here
├── run_hena_pipeline.py      runs the whole data + RQ3 pipeline
├── plan/                     team work plan PDF, project proposal
├── papers/                   reference papers
├── dataset/                  Roboflow export zip (input, never edited)
├── data/
│   ├── raw/                  six classes extracted from the zip
│   ├── processed/            train/ val/ test/ folders, one subfolder per class
│   ├── manifests/            data_split_manifest.csv, class weights, class mapping
│   └── rules/                rule_table.json and class_audit.csv (Agya fills both)
├── src/                      code
├── figures/                  report figures
├── reports/                  report sections and RQ3 results
├── docs/                     18 scenarios, independent reference table
└── archive/                  outdated files (synthetic placeholder data, old plan). Do not use.
```

## After cloning

The repository holds the code, the split images (`data/processed/train`, `val`, `test`; 4,187 images), split manifests, rules, reports and figures. The Roboflow zip and model weights are not included (too large for GitHub).

- **To read the report:** `reports/FINAL_REPORT.md` (Word version: `reports/FINAL_REPORT.docx`); figures are in `figures/`.
- **To review the class audit or rule table:** edit `data/rules/class_audit.csv` or `data/rules/rule_table.json`, commit and push, then re-run `python3 run_hena_pipeline.py`. The `image_path` column of the audit sheet points to the images in `data/processed/train/`.
- **To train or evaluate the models:** the split images are already in `data/processed/`, so only the environment is needed:
  ```bash
  python3.11 -m venv .venv
  .venv/bin/pip install torch==2.5.1 torchvision==0.20.1 scikit-learn==1.5.2 matplotlib==3.9.2 pillow==11.0.0 numpy==1.26.4
  .venv/bin/python -m src.train --model all
  ```
- **To rebuild the split from scratch:** download the Roboflow export (folder format, v1) from https://universe.roboflow.com/project-jkfnh/3d-printing-defect-classification into `dataset/`, then run `.venv/bin/python -m src.ingest_roboflow`, `.venv/bin/python -m src.scene_groups` and `python3 run_hena_pipeline.py`.

The RQ3 reference recommendations are in `src/independent_reference.py` and `docs/INDEPENDENT_REFERENCE_TABLE.md`. They were kept out of the repository until the rule table was finished (Nov 18).

## Status (Nov 18)

All steps have been run: data, both models (12 training runs), the rule table and the RQ3 evaluation. All report sections are drafted in `reports/FINAL_REPORT.md`; each owner reviews their sections.

- **Harrish:** review Methods and RQ1/RQ2 results (Sections 6–7) against `src/train.py`, `reports/model_metrics_*.json` and the figures.
- **Agya:** review Background, Related Work and the rule table (Sections 2, 3, 8; `data/rules/rule_table.json`). The rule table was drafted by Hena, so a manufacturing review matters for RQ3 (see Section 5.2 of the report).

## Models (RQ1, RQ2)

Uses the project environment in `.venv` (Python 3.11, PyTorch 2.5.1, Apple GPU via MPS):

```bash
.venv/bin/python -m src.scene_groups           # once: group near-duplicate frames into scenes (read by the split)
.venv/bin/python -m src.train --model all        # MobileNetV2 and ResNet-50, identical settings
.venv/bin/python -m src.recommend <image.jpg>    # image -> defect -> recommendation
```

Training writes `checkpoints/`, `reports/predictions_*.csv` (used by RQ3), `reports/model_metrics_*.json` and the figures. If the class audit changes the split, retrain before re-running RQ3.

## Run

```bash
python3 run_hena_pipeline.py
```

It first checks any files received from Agya or Harrish and stops with a list of problems to send back if one is malformed (`python3 -m src.check_inputs` runs only this check). It then extracts the six classes from the zip, applies Agya's audit, keeps at most 30 images per near-duplicate scene, makes the 70/15/15 split (whole scenes per split, seed 42), draws the EDA figures, runs the RQ3 evaluation when the rule table and predictions exist, and regenerates [reports/DATA_AND_EXPERIMENTAL_SETUP.md](reports/DATA_AND_EXPERIMENTAL_SETUP.md).

After the class audit changes the split, rebuild the packages for Harrish and Agya with `python3 -m src.make_send_packages` (writes `to_send/`).

The final report outline, with an owner and due date for each section, is in [reports/FINAL_REPORT_OUTLINE.md](reports/FINAL_REPORT_OUTLINE.md).

## Current data

6,132 images extracted; 4,187 kept after removing near-duplicate video frames (max 30 per scene):
stringing 1,011 · warping 1,078 · layer shifting 108 · under-extrusion 1,421 · over-extrusion 360 · bed adhesion 209.
Split: 2,930 train / 631 val / 626 test. Details in `reports/DATA_AND_EXPERIMENTAL_SETUP.md` (Section 4.4).
