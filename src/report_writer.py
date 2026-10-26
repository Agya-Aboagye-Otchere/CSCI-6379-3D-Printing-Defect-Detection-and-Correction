"""
report_writer.py
================
Writes reports/DATA_AND_EXPERIMENTAL_SETUP.md (report Sections 4-5 and RQ3 results)
from the pipeline's own outputs, so every number matches the data actually used.
Authored by Hena Parween (Part 1: Data and Simulation).
"""

import json
from pathlib import Path

from src.data_loader import TARGET_CLASSES, DISPLAY_NAMES
from src.class_audit import load_audit_corrections, SAMPLES_PER_CLASS
from src.ingest_roboflow import ROBOFLOW_TO_TARGET
from src.simulation_scenarios import PREDEFINED_SCENARIOS

MANIFESTS = Path("data/manifests")


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _rq3_results(report: dict) -> str:
    if report.get("status") != "OK":
        return ("*Pending.* The rule table (`data/rules/rule_table.json`, Agya) has not been filled in yet. "
                "Re-run `python3 run_hena_pipeline.py` once it is.")
    s = report["scenario_evaluation"]
    text = [
        "**Scenario evaluation** (rule table vs. reference, assuming a correct and confident classification):",
        "",
        _table(["Metric", "Value"], [
            ["Primary action agreement (parameter and direction)", f"{s['primary_agreement_rate_pct']}%"],
            ["Primary parameter agreement", f"{s['primary_parameter_agreement_rate_pct']}%"],
            ["Any-action overlap", f"{s['any_action_overlap_rate_pct']}%"],
            ["Cohen's kappa (primary action)", s["cohens_kappa_primary"]],
        ]),
        "",
        f"{s['note']}",
        "",
    ]
    sims = report.get("end_to_end_simulation") or {}
    if not sims:
        text.append("**End-to-end simulation:** *pending* the test-set prediction files from Harrish "
                    "(`reports/predictions_<model>.csv`).")
    else:
        text += ["**End-to-end simulation** (test split, model prediction -> rule -> compared with the reference "
                 "for the image's true class):", "",
                 _table(["Model", "Test images", "End-to-end agreement", "Agreement when recommending", "Abstained"],
                        [[m, v["n_test_images"], f"{v['end_to_end_agreement_rate_pct']}%",
                          f"{v['agreement_when_recommending_pct']}%", f"{v['abstention_rate_pct']}%"]
                         for m, v in sorted(sims.items())]),
                 "", "See `figures/rq3_end_to_end_outcomes.png` and `figures/rq3_per_class_agreement.png`."]
    return "\n".join(text)


def _scenes_text(scenes, dist) -> str:
    if not scenes:
        return ("*Not computed:* run `.venv/bin/python -m src.scene_groups`. Without it, near-duplicate video frames "
                "can fall into different splits and inflate test results.")
    cls = scenes["classes"]
    rows = [[DISPLAY_NAMES[c], cls[c]["images"], cls[c]["scenes"], f"{cls[c]['largest_scene']} ({100 * cls[c]['largest_scene_share']:.0f}%)"]
            for c in TARGET_CLASSES]
    resid = scenes.get("residual_similarity_after_split", {}).get("test")
    resid_text = (f" After the split, the median test image's most similar training image has cosine similarity "
                  f"{resid['median_max_cosine_to_train']}, and {100 * resid['share_at_or_above_0.9']:.1f}% of test images have a training "
                  "image at 0.9 or above (these are matches across classes, which scene grouping does not cover)."
                  if resid else "")
    return f"""Many images are frames of the same video or re-uploads of one photo with rotation, noise or colour changes, under
different file names. Grouping by file name alone leaves them in different splits: with that split, 18% of test images
had a training image with a near-identical perceptual hash, and the median test image had a training neighbour at cosine
similarity 0.92 (ImageNet ResNet-50 features). Models trained on that split scored 94-96% test accuracy, which overstated
how well they generalise to new prints (those runs are kept in `archive/leaky_split_models/`).

We therefore group images into *scenes* (`src/scene_groups.py`): each image is embedded with an ImageNet-pretrained
ResNet-50 (not the fine-tuned classifier, so grouping follows content rather than class), and within each class two images
are linked when their cosine similarity is at least {scenes['similarity_threshold']} or they share a Roboflow source photo. The
threshold was chosen by inspecting image pairs: almost every pair above it showed the same printed object. Linked images form
one scene.

{_table(["Class", "Images", "Scenes", "Largest scene"], rows)}

Because frames of one scene are not independent samples, at most {dist.get('max_images_per_scene', '?')} images per scene are kept
(chosen at random with the fixed seed); {dist.get('near_duplicate_frames_dropped', 0)} near-duplicate frames are removed. This matters most
for layer shifting, whose {cls['layer_shifting']['images']} images come from only {cls['layer_shifting']['scenes']} scenes: its validation and test
sets contain few scenes, so its per-class results are less certain than the other classes'.{resid_text}"""


def write_data_report(output_path: str = "reports/DATA_AND_EXPERIMENTAL_SETUP.md"):
    ingest = json.loads((MANIFESTS / "ingest_summary.json").read_text(encoding="utf-8"))
    dist = json.loads((MANIFESTS / "class_distribution.json").read_text(encoding="utf-8"))
    rq3_path = Path("reports/AGREEMENT_RATE_EVALUATION.json")
    rq3 = json.loads(rq3_path.read_text(encoding="utf-8")) if rq3_path.exists() else {}
    corrections = load_audit_corrections()
    scenes_path = Path("data/scene_groups_summary.json")
    scenes = json.loads(scenes_path.read_text(encoding="utf-8")) if scenes_path.exists() else None
    sp = dist["splits"]
    totals = {c: sp["train"][c] + sp["val"][c] + sp["test"][c] for c in TARGET_CLASSES}
    n_total = sum(totals.values())
    rf_names = {v: k for k, v in ROBOFLOW_TO_TARGET.items()}
    largest, smallest = max(totals, key=totals.get), min(totals, key=totals.get)

    if corrections:
        n_drop = sum(v == "drop" for v in corrections.values())
        audit_text = (f"Agya (manufacturing) reviewed {SAMPLES_PER_CLASS} random training images per class. "
                      f"{n_drop} image(s) were removed and {len(corrections) - n_drop} relabelled before the final split.")
    else:
        audit_text = (f"*Pending:* Agya (manufacturing) is reviewing {SAMPLES_PER_CLASS} random training images per class "
                      "(`data/rules/class_audit.csv`); corrections are applied automatically on the next pipeline run.")

    md = f"""# Data and Experimental Setup

*Generated by `src/report_writer.py` from the pipeline outputs. Do not edit by hand; re-run `python3 run_hena_pipeline.py`.*

## 4. Data

### 4.1 Source
We use the *3D Printing Defect Classification* dataset from Roboflow Universe
(https://universe.roboflow.com/project-jkfnh/3d-printing-defect-classification, CC BY 4.0), version 1,
exported on July 17, 2025. The export contains 22,048 images with image-level class labels in folder format.
Roboflow auto-oriented every image and resized it to 640 x 640 pixels (stretch); no augmentation was applied in the export.

### 4.2 Class selection
Following the proposal, we keep six defect classes and exclude all other categories, including "No Defect".

{_table(["Class", "Roboflow folder", "Images extracted"], [[DISPLAY_NAMES[c], rf_names[c], ingest["images_per_class"][c]] for c in TARGET_CLASSES] + [["**Total**", "", f"**{ingest['images_extracted']}**"]])}

### 4.3 Cleaning
- Every image was opened and verified; corrupt files are dropped. MD5 hashes remove exact duplicates.
- Roboflow renames files to `<original>_jpg.rf.<hash>.jpg`. Grouping by the original name shows that the
  {ingest['images_extracted']} extracted images come from {ingest['unique_source_photos']} distinct source photos.
- {ingest['source_photos_dropped_conflicting_labels']} source photo(s) carried two different defect labels and were removed.
- {audit_text}

### 4.4 Near-duplicate scenes
{_scenes_text(scenes, dist)}

### 4.5 Train / validation / test split
Roboflow's own split is not used: {ingest['source_photos_in_multiple_roboflow_splits']} source photos appear in more than one
of its train/valid/test folders. We make a stratified 70/15/15 split per class with seed {dist['random_seed']}, assigning whole
scenes to one split (largest scenes first, each to the split furthest below its target size), so no scene appears in two splits.
Both models (RQ1, RQ2) use exactly this split, recorded in `data/manifests/data_split_manifest.csv`.

{_table(["Class", "Train", "Val", "Test", "Total"], [[DISPLAY_NAMES[c], sp["train"][c], sp["val"][c], sp["test"][c], totals[c]] for c in TARGET_CLASSES] + [["**Total**", sp["total_train"], sp["total_val"], sp["total_test"], n_total]])}

![Class distribution](../figures/class_distribution.png)
![Split distribution](../figures/split_distribution.png)

### 4.6 Class imbalance
The classes are imbalanced: {DISPLAY_NAMES[largest]} has {totals[largest]} images and {DISPLAY_NAMES[smallest]} has
{totals[smallest]} ({totals[largest] / totals[smallest]:.1f}x fewer). The training loss uses inverse-frequency class weights
w_c = N / (K * N_c), computed on the training split:

{_table(["Class", "Weight"], [[DISPLAY_NAMES[c], dist["class_weights_inverse_frequency"][c]] for c in TARGET_CLASSES])}

Because of the imbalance, macro-averaged precision, recall and F1 (RQ1) are reported alongside accuracy.

### 4.7 Preprocessing and augmentation
All images are resized to 224 x 224 and normalized with ImageNet channel statistics
(mean 0.485/0.456/0.406, std 0.229/0.224/0.225), matching the ImageNet-pretrained MobileNet and ResNet backbones.
Training images only are augmented with random horizontal and vertical flips, rotation of up to +/-15 degrees,
and brightness, contrast and saturation jitter, following the hue/saturation augmentation of De La Rosa et al. (2024).
Validation and test images are only resized and normalized.

![Sample images](../figures/sample_defect_grid.png)

## 5. Experimental Setup

### 5.1 Classification (RQ1, RQ2)
MobileNet and ResNet are fine-tuned on the split above with identical settings (optimizer, learning rate, epochs,
early stopping, augmentation, class weights). Training details and results are in the Methods and Results sections.

### 5.2 Correction framework evaluation (RQ3)
**Independence.** The rule table (`data/rules/rule_table.json`) is written by Agya (manufacturing). The reference
recommendations (`docs/INDEPENDENT_REFERENCE_TABLE.md`) were compiled separately by Hena from a different set of
sources (Prusa, Simplify3D, All3DP, Gibson et al.), and Agya does not see them before submitting the rule table.

**Scenario evaluation.** {len(PREDEFINED_SCENARIOS)} predefined scenarios (`docs/PREDEFINED_SCENARIOS.md`) cover the six defects at three
severity levels on PLA, PETG and ABS. Each is passed to the rule engine as a correct, confident classification, and its
primary action is compared with the reference. Metrics: primary action agreement (same parameter and direction),
primary parameter agreement, any-action overlap between the two action lists, and Cohen's kappa on the primary action.

**End-to-end simulation.** Instead of physical printer control, each test image is replayed through the full system:
the classifier's predicted class and confidence go to the rule engine, which abstains when confidence is below 0.65
(the threshold of De La Rosa et al., 2024). Each recommendation is scored against the reference for the image's *true*
class, so classification errors count as wrong recommendations. We report, per model: end-to-end agreement (share of all
test images that received the reference action), agreement when the system recommends, and the abstention rate.

### 5.3 RQ3 results
{_rq3_results(rq3)}
"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(md, encoding="utf-8")
    print(f"[+] Report sections written to: {output_path}")
