"""
rq3_figures.py
==============
Report figures for RQ3, drawn from reports/AGREEMENT_RATE_EVALUATION.json.
Authored by Hena Parween (Part 1: Data and Simulation).

1. figures/rq3_end_to_end_outcomes.png: share of test images where the system gave the
   reference action, a different action, or abstained (confidence < threshold), per model.
2. figures/rq3_per_class_agreement.png: end-to-end agreement per true defect class, per model.
Nothing is drawn until the rule table and model predictions exist.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import TARGET_CLASSES, DISPLAY_NAMES

OUTCOMES = [("correct_primary_action", "Reference action", "#2E7D5B"),
            ("wrong_primary_action", "Different action", "#C0504D"),
            ("abstained", "Abstained (low confidence)", "#A6A6A6")]


def generate_rq3_figures(report_path: str = "reports/AGREEMENT_RATE_EVALUATION.json", output_dir: str = "figures"):
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    sims = report.get("end_to_end_simulation") or {}
    if report.get("status") != "OK" or not sims:
        print("[=] RQ3 figures skipped: waiting for the rule table and model predictions.")
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = sorted(sims)

    fig, ax = plt.subplots(figsize=(7, 1.2 + 0.7 * len(models)))
    for i, m in enumerate(models):
        left = 0.0
        for key, label, color in OUTCOMES:
            share = 100 * sims[m]["outcomes"].get(key, 0) / sims[m]["n_test_images"]
            ax.barh(i, share, left=left, color=color, label=label if i == 0 else None)
            if share >= 6:
                ax.text(left + share / 2, i, f"{share:.0f}%", ha="center", va="center", color="white", fontsize=9)
            left += share
    ax.set_yticks(range(len(models)), models)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Test images (%)")
    ax.set_title("End-to-end recommendation outcome on the test split")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "rq3_end_to_end_outcomes.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(models)
    for j, m in enumerate(models):
        per_class = sims[m]["per_class_outcomes"]
        rates = [100 * per_class.get(c, {}).get("correct_primary_action", 0) / max(1, sum(per_class.get(c, {}).values()))
                 for c in TARGET_CLASSES]
        ax.bar([i + (j - (len(models) - 1) / 2) * width for i in range(len(TARGET_CLASSES))], rates, width, label=m)
    ax.set_xticks(range(len(TARGET_CLASSES)), [DISPLAY_NAMES[c].split(" /")[0] for c in TARGET_CLASSES],
                  rotation=20, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("End-to-end agreement (%)")
    ax.set_title("Agreement with the reference action, by true defect class")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "rq3_per_class_agreement.png", dpi=200)
    plt.close(fig)
    print(f"[+] RQ3 figures written to {out}/rq3_end_to_end_outcomes.png and rq3_per_class_agreement.png")
