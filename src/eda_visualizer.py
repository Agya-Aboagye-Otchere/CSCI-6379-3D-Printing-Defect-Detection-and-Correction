"""
eda_visualizer.py
=================
Exploratory Data Analysis (EDA) and Visualization Generator.
Authored by Hena Parween (Part 1: Data and Simulation).

Generates:
1. figures/class_distribution.png: Class breakdown across the 6 defect categories.
2. figures/split_distribution.png: Stratified train/val/test split representation.
3. figures/sample_defect_grid.png: 2x3 annotated grid of defect types for the report.
"""

import os
import json
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure headless matplotlib backend and writable config dir
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import TARGET_CLASSES, DISPLAY_NAMES


def generate_eda_figures(
    manifest_dir: str = "data/manifests",
    processed_dir: str = "data/processed",
    output_dir: str = "figures"
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    dist_file = Path(manifest_dir) / "class_distribution.json"
    if not dist_file.exists():
        print(f"[!] {dist_file} not found. Please run data_loader first.")
        return
        
    with open(dist_file, "r") as f:
        dist_data = json.load(f)
        
    splits = dist_data["splits"]
    train_counts = splits["train"]
    val_counts = splits["val"]
    test_counts = splits["test"]
    
    classes = TARGET_CLASSES
    readable_labels = [DISPLAY_NAMES[c] for c in classes]
    total_counts = [train_counts[c] + val_counts[c] + test_counts[c] for c in classes]
    
    # -------------------------------------------------------------
    # 1. Class Distribution Bar Chart
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 5.5))
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
    bars = plt.bar(readable_labels, total_counts, color=colors, edgecolor="black", alpha=0.85, width=0.55)
    
    plt.title("FDM 3D Printing Defect Classification Dataset: Class Distribution", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Defect Category", fontsize=11, fontweight="bold")
    plt.ylabel("Number of Samples", fontsize=11, fontweight="bold")
    plt.xticks(rotation=20, ha="right", fontsize=9.5)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.ylim(0, max(total_counts) * 1.18)
    
    # Annotate bar values
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 2, f"{int(h)}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    plt.tight_layout()
    class_dist_fig = out_path / "class_distribution.png"
    plt.savefig(class_dist_fig, dpi=300)
    plt.close()
    print(f"[+] Saved class distribution figure: {class_dist_fig}")
    
    # -------------------------------------------------------------
    # 2. Stratified Split Proportions (Train/Val/Test)
    # -------------------------------------------------------------
    plt.figure(figsize=(11, 5.5))
    x = np.arange(len(classes))
    width = 0.25
    
    tr_vals = [train_counts[c] for c in classes]
    va_vals = [val_counts[c] for c in classes]
    te_vals = [test_counts[c] for c in classes]
    
    b1 = plt.bar(x - width, tr_vals, width, label="Train (70%)", color="#1f77b4", edgecolor="black")
    b2 = plt.bar(x, va_vals, width, label="Validation (15%)", color="#ff7f0e", edgecolor="black")
    b3 = plt.bar(x + width, te_vals, width, label="Test (15%)", color="#2ca02c", edgecolor="black")
    
    plt.title("Stratified Dataset Split Distribution across FDM Defect Classes", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Defect Category", fontsize=11, fontweight="bold")
    plt.ylabel("Sample Count", fontsize=11, fontweight="bold")
    plt.xticks(x, [c.replace("_", " ").title() for c in classes], rotation=15, ha="right", fontsize=9.5)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    split_dist_fig = out_path / "split_distribution.png"
    plt.savefig(split_dist_fig, dpi=300)
    plt.close()
    print(f"[+] Saved split distribution figure: {split_dist_fig}")
    
    # -------------------------------------------------------------
    # 3. 2x3 Grid of Sample Defect Images
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, cls in enumerate(classes):
        ax = axes[idx]
        cls_folder = Path(processed_dir) / "train" / cls
        sample_imgs = list(cls_folder.glob("*.jpg")) + list(cls_folder.glob("*.png"))
        
        if sample_imgs:
            sample_img_path = sample_imgs[0]
            with Image.open(sample_img_path) as img:
                ax.imshow(img)
        else:
            # Placeholder if no sample image
            ax.text(0.5, 0.5, f"No image for\n{cls}", ha="center", va="center")
            
        ax.set_title(f"{idx+1}. {DISPLAY_NAMES[cls]}", fontsize=11, fontweight="bold", pad=6)
        ax.axis("off")
        
    plt.suptitle("Exemplar Visual Signatures of Selected FDM 3D Printing Defects", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    grid_fig = out_path / "sample_defect_grid.png"
    plt.savefig(grid_fig, dpi=300)
    plt.close()
    print(f"[+] Saved sample defect grid figure: {grid_fig}")


if __name__ == "__main__":
    generate_eda_figures()
