"""
ingest_roboflow.py
==================
Extracts the 6 target defect classes from the Roboflow export zip into data/raw/.
Authored by Hena Parween (Part 1: Data and Simulation).

The Roboflow export (v1, folder format) ships its own train/valid/test folders.
We ignore those splits and pool all images, because 104 source images appear in
more than one Roboflow split; data_loader.py re-splits them by source image.

Images whose source appears under two different target classes are dropped,
since their label is ambiguous.
"""

import json
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

DEFAULT_ZIP = Path("dataset/3D Printing Defect Classification.v1i.folder.zip")

# Roboflow folder name -> project class name
ROBOFLOW_TO_TARGET = {
    "Stringing": "stringing",
    "Warping": "warping",
    "Layer Shifting": "layer_shifting",
    "Under Extrusion": "under_extrusion",
    "over extrusion": "over_extrusion",
    "bed adhesion": "bed_adhesion",
}

# Roboflow renames files to "<original>_<ext>.rf.<hash>.<ext>"
RF_NAME = re.compile(r"^(?P<source>.+)_(?:jpe?g|png|webp|bmp)\.rf\.[0-9a-f]+\.\w+$", re.IGNORECASE)


def source_id(filename: str) -> str:
    """Original (pre-Roboflow) image name, used to group copies of the same photo."""
    m = RF_NAME.match(filename)
    return m.group("source") if m else Path(filename).stem


def ingest(zip_path: Path = DEFAULT_ZIP, raw_dir: Path = Path("data/raw"),
           summary_path: Path = Path("data/manifests/ingest_summary.json")) -> dict:
    if not zip_path.exists():
        raise FileNotFoundError(f"Roboflow export not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        members = []
        for name in zf.namelist():
            parts = name.split("/")
            if len(parts) != 3 or not parts[2] or parts[1] not in ROBOFLOW_TO_TARGET:
                continue
            members.append((name, parts[0], ROBOFLOW_TO_TARGET[parts[1]], parts[2]))

        classes_per_source = defaultdict(set)
        splits_per_source = defaultdict(set)
        for _, rf_split, cls, fname in members:
            classes_per_source[source_id(fname)].add(cls)
            splits_per_source[source_id(fname)].add(rf_split)
        conflicting = {s for s, c in classes_per_source.items() if len(c) > 1}
        leaking = {s for s, sp in splits_per_source.items() if len(sp) > 1 and s not in conflicting}

        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        counts = defaultdict(int)
        for name, rf_split, cls, fname in members:
            if source_id(fname) in conflicting:
                continue
            dest = raw_dir / cls / f"{rf_split}__{fname}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
            counts[cls] += 1

    summary = {
        "zip_file": zip_path.name,
        "images_extracted": sum(counts.values()),
        "images_per_class": dict(counts),
        "unique_source_photos": len(classes_per_source) - len(conflicting),
        "source_photos_dropped_conflicting_labels": len(conflicting),
        "source_photos_in_multiple_roboflow_splits": len(leaking),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[+] Extracted {sum(counts.values())} images from {zip_path.name} into {raw_dir}/")
    print(f"    - {len(leaking)} source photos appear in more than one Roboflow split (re-split by photo later)")
    print(f"    - Dropped {len(conflicting)} source image(s) labelled with two different classes")
    for cls in ROBOFLOW_TO_TARGET.values():
        print(f"    - {cls}: {counts[cls]}")
    return dict(counts)


if __name__ == "__main__":
    ingest()
