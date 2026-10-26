"""
class_audit.py
==============
Label audit hand-off between Agya (manufacturing) and Hena (data).
Authored by Hena Parween (Part 1: Data and Simulation).

1. make_audit_sheet() lists 50 random training images per class in
   data/rules/class_audit.csv for Agya to review.
2. Agya fills `issue` and `suggested_class` for any mislabelled image:
   suggested_class = one of the six class names (relabel) or "drop" (remove).
   Rows left blank mean the label is correct.
3. load_audit_corrections() is read by data_loader.py on the next pipeline run,
   so corrections are applied by image content (MD5), not by file path.
"""

import csv
import random
from pathlib import Path
from typing import Dict

from src.data_loader import TARGET_CLASSES

AUDIT_PATH = Path("data/rules/class_audit.csv")
SAMPLES_PER_CLASS = 50
AUDIT_SEED = 7


def make_audit_sheet(manifest_csv: Path = Path("data/manifests/data_split_manifest.csv"),
                     audit_path: Path = AUDIT_PATH) -> None:
    """Creates the audit sheet once; never overwrites a sheet Agya may have started."""
    if audit_path.exists():
        print(f"[=] Audit sheet already exists, leaving it untouched: {audit_path}")
        return
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "train"]
    rng = random.Random(AUDIT_SEED)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "labelled_class", "md5", "issue", "suggested_class"])
        for cls in TARGET_CLASSES:
            cls_rows = [r for r in rows if r["class_name"] == cls]
            for r in rng.sample(cls_rows, min(SAMPLES_PER_CLASS, len(cls_rows))):
                w.writerow([r["image_path"], cls, r["md5"], "", ""])
    print(f"[+] Audit sheet for Agya written to: {audit_path}")


def load_audit_corrections(audit_path: Path = AUDIT_PATH) -> Dict[str, str]:
    """md5 -> new class name or "drop", for every row Agya filled in."""
    if not audit_path.exists():
        return {}
    corrections = {}
    with open(audit_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            target = row["suggested_class"].strip().lower().replace(" ", "_").replace("-", "_")
            if not target:
                continue
            if target != "drop" and target not in TARGET_CLASSES:
                raise ValueError(f"Unknown suggested_class '{row['suggested_class']}' for {row['image_path']}")
            corrections[row["md5"]] = target
    return corrections
