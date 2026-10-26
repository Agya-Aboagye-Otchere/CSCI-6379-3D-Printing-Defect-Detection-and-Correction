"""
make_send_packages.py
=====================
Rebuilds the zip files Hena sends to teammates, in to_send/.
Authored by Hena Parween (Part 1: Data and Simulation).

Re-run after the class audit changes the split, so Harrish trains on the current data.
Agya's package never contains the independent reference (RQ3 independence).
"""

import zipfile
from pathlib import Path

OUT_DIR = Path("to_send")
PLAN = Path("plan/FDM_Team_Work_Plan_v2.pdf")
PACKAGES = {
    # Harrish reviews the training and writes RQ1/RQ2 up: code, results, figures, checkpoints and the data they came from
    "Harrish_files.zip": [PLAN, Path("README.md"), Path("src"), Path("reports/MODEL_RESULTS.md"),
                          *sorted(Path("reports").glob("model_metrics_*.json")),
                          *sorted(Path("reports").glob("predictions_*.csv")),
                          *sorted(Path("figures").glob("training_curves_*.png")),
                          *sorted(Path("figures").glob("confusion_matrix_*.png")),
                          Path("checkpoints"), Path("data/processed"), Path("data/manifests")],
    "Agya_files.zip": [PLAN, Path("data/rules/rule_table.json"), Path("data/rules/class_audit.csv"),
                       Path("data/processed/train")],
}
NEVER_TO_AGYA = ("independent_reference", "INDEPENDENT_REFERENCE")


def _files(paths):
    for p in paths:
        if p.is_dir():
            yield from (f for f in sorted(p.rglob("*")) if f.is_file() and f.name != ".DS_Store" and "__pycache__" not in f.parts)
        elif p.exists():
            yield p


def make_send_packages():
    OUT_DIR.mkdir(exist_ok=True)
    for name, paths in PACKAGES.items():
        files = list(_files(paths))
        if name.startswith("Agya") and any(k in str(f) for f in files for k in NEVER_TO_AGYA):
            raise RuntimeError("Agya's package would include the independent reference; aborting.")
        with zipfile.ZipFile(OUT_DIR / name, "w", zipfile.ZIP_STORED) as zf:  # JPGs are already compressed
            for f in files:
                zf.write(f)
        size_mb = (OUT_DIR / name).stat().st_size / 1e6
        print(f"[+] {OUT_DIR / name}: {len(files)} files, {size_mb:.0f} MB")


if __name__ == "__main__":
    make_send_packages()
