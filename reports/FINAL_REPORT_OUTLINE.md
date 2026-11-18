# Final Report Outline

**Title:** Deep Learning Based Defect Detection and Rule-Based Correction Framework for FDM 3D Printing
**Team:** Hena (data and simulation), Harrish (models), Agya (manufacturing)
**Merge:** Hena. All sections were drafted on Nov 18, 2026 in `reports/FINAL_REPORT.md` (Word version: `reports/FINAL_REPORT.docx`); each owner reviews their sections.

| # | Section | Owner | Due | Status |
|---|---|---|---|---|
| 1 | Introduction and problem | Agya | Nov 5 | Drafted (Nov 18) in `FINAL_REPORT.md` / `FINAL_REPORT.docx`; owner reviews |
| 2 | Background: the six defects and their physical causes | Agya | Nov 3 | Drafted (Nov 18) in `FINAL_REPORT.md` / `FINAL_REPORT.docx`; owner reviews |
| 3 | Related work | Agya | Nov 3 | Drafted (Nov 18) in `FINAL_REPORT.md` / `FINAL_REPORT.docx`; owner reviews |
| 4 | Data | Hena | Done | Drafted (Nov 18) in `FINAL_REPORT.md` / `FINAL_REPORT.docx`; owner reviews |
| 5 | Experimental setup | Hena | Done | Drafted (Nov 18) in `FINAL_REPORT.md` / `FINAL_REPORT.docx`; owner reviews |
| 6 | Methods: MobileNet and ResNet | Harrish | Nov 4 | Drafted (Nov 18) in `FINAL_REPORT.md` / `FINAL_REPORT.docx`; owner reviews |
| 7 | Results: RQ1 and RQ2 | Harrish | Nov 4 | Drafted (Nov 18) in `FINAL_REPORT.md` / `FINAL_REPORT.docx`; owner reviews |
| 8 | Correction framework: the rule table | Agya | Nov 4 | Drafted (Nov 18) in `FINAL_REPORT.md` / `FINAL_REPORT.docx`; owner reviews |
| 9 | Results: RQ3 | Hena | Nov 4 | Drafted (Nov 18) in `FINAL_REPORT.md` / `FINAL_REPORT.docx`; owner reviews |
| 10 | Discussion: error interpretation and limitations | Agya + Harrish + Hena | Nov 5 | Drafted (Nov 18) in `FINAL_REPORT.md` / `FINAL_REPORT.docx`; owner reviews |
| 11 | Conclusion | Hena | Nov 6 | Drafted (Nov 18) in `FINAL_REPORT.md` / `FINAL_REPORT.docx`; owner reviews |
| | References | Everyone | Nov 6 | Drafted (Nov 18) in `FINAL_REPORT.md` / `FINAL_REPORT.docx`; owner reviews |

## What goes in each section

**1. Introduction (Agya).** FDM and why defects matter (dimensional accuracy, strength, wasted material). Why monitoring is hard without a skilled operator. The three research questions as written in the proposal. One paragraph on the contribution: a classifier plus a rule-based recommendation system, evaluated without physical printer control.

**2. Background (Agya).** For each of the six defects: what it looks like, its physical root cause, and which printing parameters drive it. One short paragraph each.

**3. Related work (Agya).** The five proposal papers: Jin et al. 2019, Brion, Shen and Pattinson 2022 (warp), Brion and Pattinson 2022 (multi-head), Lu et al. 2023, De La Rosa et al. 2024. For each: method, result, and how this project differs. The proposal's Related Work text can be reused.

**4-5. Data and experimental setup (Hena).** Copy from `reports/DATA_AND_EXPERIMENTAL_SETUP.md` after the final pipeline run, so the numbers match.

**6. Methods (Harrish, from `reports/MODEL_RESULTS.md`).** Models were trained by Hena with `src/train.py`; Harrish reviews the code and writes this up. Both architectures, the replaced classifier head, the two training stages (frozen backbone, then fine-tuning), the identical settings used for both models (optimizer, learning rate, batch size, epochs, early stopping, class-weighted loss), and the hardware.

**7. Results RQ1 and RQ2 (Harrish).** A table with accuracy, macro precision, recall and F1, and mean inference time per image for each model. Per-class recall. Confusion matrices and training curves. A direct answer to RQ1 and RQ2.

**8. Correction framework (Agya).** The rule table with a citation for each rule, and the reasoning behind the primary action for each defect. The 0.65 confidence threshold (De La Rosa et al. 2024).

**9. Results RQ3 (Hena).** Scenario agreement and end-to-end simulation for both models, from section 5.3 of the generated file, plus `figures/rq3_end_to_end_outcomes.png` and `figures/rq3_per_class_agreement.png`. Discuss which defects disagree with the reference and why.

**10. Discussion.** Agya: why the models confuse certain classes, in physical terms (for example, warping versus bed adhesion). Harrish: MobileNet versus ResNet trade-off between accuracy and speed. Hena: limitations: no physical printer loop, Roboflow images from mixed sources, class imbalance, rules that ignore severity.

**11. Conclusion (Hena).** One paragraph answering each RQ, and future work.

## Figures checklist

| Figure | File | Owner |
|---|---|---|
| Sample image per defect | `figures/sample_defect_grid.png` | Hena (ready) |
| Class distribution | `figures/class_distribution.png` | Hena (ready) |
| Split distribution | `figures/split_distribution.png` | Hena (ready) |
| Training curves, both models | `figures/training_curves_*.png` | Hena (ready after training) |
| Confusion matrices, both models | `figures/confusion_matrix_*.png` | Hena (ready after training) |
| RQ3 outcomes | `figures/rq3_end_to_end_outcomes.png` | Hena (automatic) |
| RQ3 per-class agreement | `figures/rq3_per_class_agreement.png` | Hena (automatic) |
