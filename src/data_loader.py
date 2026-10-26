"""
data_loader.py
==============
Data Pipeline Module for FDM 3D Printing Defect Classification.
Authored by Hena Parween (Part 1: Data and Simulation).

This module handles:
1. Dataset ingestion & filtering for the 6 target FDM defect classes:
   - Stringing
   - Warping
   - Layer Shifting
   - Under-Extrusion
   - Over-Extrusion
   - Bed Adhesion
   (Excluding 'No Defect' and 'Unlabeled' per project specification).
2. Data cleaning: corrupted image check, deduplication via MD5 hashing.
3. Stratified 70/15/15 train/val/test splitting with fixed seed (42).
4. Generation of master split manifest CSV, class weights, and directory hierarchies.
5. Automated sample image generation for offline testing & reproducible verification.
"""

import csv
import os
import sys
import glob
import json
import hashlib
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image

from src.ingest_roboflow import source_id

# Set consistent random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TARGET_CLASSES = [
    "stringing",
    "warping",
    "layer_shifting",
    "under_extrusion",
    "over_extrusion",
    "bed_adhesion"
]

CLASS_TO_ID = {cls_name: idx for idx, cls_name in enumerate(TARGET_CLASSES)}
ID_TO_CLASS = {idx: cls_name for cls_name, idx in CLASS_TO_ID.items()}

# Human-readable display names
DISPLAY_NAMES = {
    "stringing": "Stringing / Oozing",
    "warping": "Warping / Corner Lifting",
    "layer_shifting": "Layer Shifting / Step Misalignment",
    "under_extrusion": "Under-Extrusion",
    "over_extrusion": "Over-Extrusion",
    "bed_adhesion": "Bed Adhesion Failure"
}


def calculate_md5(file_path: Path) -> str:
    """Computes MD5 hash of an image file for deduplication."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_image(file_path: Path) -> Tuple[bool, Optional[Tuple[int, int]]]:
    """
    Verifies that the file is an uncorrupted image and returns dimensions (width, height).
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        with Image.open(file_path) as img:
            return True, img.size
    except Exception:
        return False, None


# Frames of one scene (mostly video) are near-duplicates, not independent samples; keep at most this many per scene
MAX_IMAGES_PER_SCENE = 30


def load_scene_groups(path: Path = Path("data/scene_groups.csv")) -> Dict[str, str]:
    """md5 -> scene id from src/scene_groups.py, or {} if it has not been run."""
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row["md5"]: row["scene_id"] for row in csv.DictReader(f)}


class FDMDatasetPipeline:
    """
    Manages end-to-end ingestion, cleaning, stratified splitting, and manifest creation.
    """
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed", manifest_dir: str = "data/manifests"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.manifest_dir = Path(manifest_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    def scan_and_clean_raw(self) -> Dict[str, List[Dict]]:
        """
        Scans raw directories, filters for the 6 target classes,
        removes corrupt images, and detects duplicates using MD5 hashes.
        """
        cleaned_records_by_class = {cls: [] for cls in TARGET_CLASSES}
        seen_hashes = set()
        corrupted_count = 0
        duplicate_count = 0
        from src.class_audit import load_audit_corrections  # local import: class_audit imports this module
        audit_corrections = load_audit_corrections()
        scene_of = load_scene_groups()
        if not scene_of:
            print("[!] data/scene_groups.csv not found: grouping by Roboflow source photo only, which leaves near-duplicate "
                  "video frames across splits. Run `.venv/bin/python -m src.scene_groups` first.")
        dropped_by_audit = 0
        relabelled_by_audit = 0
        
        # Check if raw directory has images
        raw_images = list(self.raw_dir.glob("**/*.[jJ][pP][gG]")) + list(self.raw_dir.glob("**/*.[pP][nN][gG]"))
        if len(raw_images) == 0:
            raise FileNotFoundError(
                f"No images found in {self.raw_dir}. Run `python3 -m src.ingest_roboflow` first."
            )

        print(f"[*] Scanning {len(raw_images)} candidate images...")

        for img_path in raw_images:
            # The class is the parent folder only; original filenames sometimes name other defects
            parent_name = img_path.parent.name.lower().replace("-", "_").replace(" ", "_")
            matched_cls = parent_name if parent_name in CLASS_TO_ID else None
            
            # Skip if not one of the 6 targeted classes (e.g. 'no_defect', 'unlabeled')
            if matched_cls is None:
                continue
                
            is_valid, dims = verify_image(img_path)
            if not is_valid:
                corrupted_count += 1
                continue
                
            md5 = calculate_md5(img_path)
            if md5 in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(md5)

            correction = audit_corrections.get(md5)
            if correction == "drop":
                dropped_by_audit += 1
                continue
            if correction:
                matched_cls = correction
                relabelled_by_audit += 1
            
            cleaned_records_by_class[matched_cls].append({
                "source_path": str(img_path),
                "filename": img_path.name,
                "class_name": matched_cls,
                "class_id": CLASS_TO_ID[matched_cls],
                # Near-duplicate scene (src/scene_groups.py) -> same group, same split
                "group_id": scene_of.get(md5) or source_id(img_path.name.split("__", 1)[-1]),
                "md5": md5,
                "width": dims[0],
                "height": dims[1]
            })

        print(f"[+] Cleaning complete:")
        print(f"    - Unique valid images: {len(seen_hashes)}")
        print(f"    - Corrupt images dropped: {corrupted_count}")
        print(f"    - Duplicates dropped: {duplicate_count}")
        print(f"    - Class audit: {dropped_by_audit} dropped, {relabelled_by_audit} relabelled")
        for cls in TARGET_CLASSES:
            print(f"    - Class '{cls}': {len(cleaned_records_by_class[cls])} images")
            
        return cleaned_records_by_class

    def create_stratified_split(
        self,
        cleaned_records: Dict[str, List[Dict]],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> List[Dict]:
        """
        Splits data with stratified sampling: 70% train, 15% validation, 15% test.
        Uses fixed seed (42) so MobileNet and ResNet use identical splits.
        """
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"
        
        all_split_records = []
        split_stats = {"train": {}, "val": {}, "test": {}}
        self.capped_frames = 0
        
        for cls in TARGET_CLASSES:
            # Split whole groups (source images), not files, so no photo leaks across splits
            groups: Dict[str, List[Dict]] = {}
            for rec in cleaned_records[cls]:
                groups.setdefault(rec["group_id"], []).append(rec)
            group_ids = sorted(groups)
            rng = random.Random(SEED + CLASS_TO_ID[cls])
            for gid in group_ids:
                if len(groups[gid]) > MAX_IMAGES_PER_SCENE:
                    frames = sorted(groups[gid], key=lambda r: r["md5"])
                    self.capped_frames += len(frames) - MAX_IMAGES_PER_SCENE
                    groups[gid] = rng.sample(frames, MAX_IMAGES_PER_SCENE)
            rng.shuffle(group_ids)
            # Largest scenes first (shuffled order breaks ties), each to the split furthest below its
            # target size; filling splits in sequence would hand a scene of hundreds of frames to val or test
            group_ids.sort(key=lambda g: len(groups[g]), reverse=True)

            total_n = sum(len(g) for g in groups.values())
            targets = {"train": total_n * train_ratio, "val": total_n * val_ratio, "test": total_n * test_ratio}
            assigned = {"train": [], "val": [], "test": []}
            for gid in group_ids:
                split_name = max(targets, key=lambda s: targets[s] - len(assigned[s]))
                assigned[split_name].extend(groups[gid])
            train_items, val_items, test_items = assigned["train"], assigned["val"], assigned["test"]
            
            for item in train_items:
                item["split"] = "train"
                all_split_records.append(item)
            for item in val_items:
                item["split"] = "val"
                all_split_records.append(item)
            for item in test_items:
                item["split"] = "test"
                all_split_records.append(item)
                
            split_stats["train"][cls] = len(train_items)
            split_stats["val"][cls] = len(val_items)
            split_stats["test"][cls] = len(test_items)
            
        print(f"\n[+] Dropped {self.capped_frames} near-duplicate frames (max {MAX_IMAGES_PER_SCENE} images per scene)")
        print("\n[+] Stratified Split Summary (70/15/15):")
        print(f"{'Class Name':<20} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}")
        print("-" * 60)
        for cls in TARGET_CLASSES:
            tr = split_stats["train"][cls]
            va = split_stats["val"][cls]
            te = split_stats["test"][cls]
            tot = tr + va + te
            print(f"{cls:<20} | {tr:<8} | {va:<8} | {te:<8} | {tot:<8}")
        print("-" * 60)
        
        return all_split_records

    def materialize_dataset(self, split_records: List[Dict]):
        """
        Copies images into data/processed/{train,val,test}/{class_name}/
        and outputs data_split_manifest.csv, class_mapping.json, and class_distribution.json.
        """
        # Rebuild from scratch so relabelled or dropped images do not linger in old folders
        if self.processed_dir.exists():
            shutil.rmtree(self.processed_dir)
        for split in ["train", "val", "test"]:
            for cls in TARGET_CLASSES:
                (self.processed_dir / split / cls).mkdir(parents=True, exist_ok=True)
                
        manifest_csv_path = self.manifest_dir / "data_split_manifest.csv"
        class_mapping_path = self.manifest_dir / "class_mapping.json"
        class_dist_path = self.manifest_dir / "class_distribution.json"
        
        # Write CSV manifest
        with open(manifest_csv_path, "w", encoding="utf-8") as f_csv:
            f_csv.write("image_path,class_name,class_id,split,group_id,md5,width,height\n")
            
            for rec in split_records:
                src_path = Path(rec["source_path"])
                target_rel_path = Path(rec["split"]) / rec["class_name"] / f"{rec['md5'][:8]}_{src_path.name}"
                dest_path = self.processed_dir / target_rel_path
                
                # Copy or write
                if not dest_path.exists():
                    shutil.copy2(src_path, dest_path)
                    
                rec["processed_path"] = str(dest_path)
                f_csv.write(f"{dest_path},{rec['class_name']},{rec['class_id']},{rec['split']},{rec['group_id']},{rec['md5']},{rec['width']},{rec['height']}\n")
                
        print(f"[+] Master split manifest written to: {manifest_csv_path}")
        
        # Save class mapping
        with open(class_mapping_path, "w", encoding="utf-8") as f_map:
            json.dump({
                "class_to_id": CLASS_TO_ID,
                "id_to_class": ID_TO_CLASS,
                "display_names": DISPLAY_NAMES
            }, f_map, indent=2)
        print(f"[+] Class mapping written to: {class_mapping_path}")
        
        # Compute class weights for training set (inverse frequency)
        # w_c = N / (K * N_c)
        train_counts = {cls: 0 for cls in TARGET_CLASSES}
        val_counts = {cls: 0 for cls in TARGET_CLASSES}
        test_counts = {cls: 0 for cls in TARGET_CLASSES}
        
        for rec in split_records:
            if rec["split"] == "train":
                train_counts[rec["class_name"]] += 1
            elif rec["split"] == "val":
                val_counts[rec["class_name"]] += 1
            elif rec["split"] == "test":
                test_counts[rec["class_name"]] += 1
                
        n_train_total = sum(train_counts.values())
        k = len(TARGET_CLASSES)
        class_weights = {}
        for cls, count in train_counts.items():
            class_weights[cls] = round(n_train_total / (k * count), 4) if count > 0 else 1.0
            
        distribution_data = {
            "splits": {
                "train": train_counts,
                "val": val_counts,
                "test": test_counts,
                "total_train": n_train_total,
                "total_val": sum(val_counts.values()),
                "total_test": sum(test_counts.values())
            },
            "class_weights_inverse_frequency": class_weights,
            "random_seed": SEED,
            "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "max_images_per_scene": MAX_IMAGES_PER_SCENE,
            "near_duplicate_frames_dropped": getattr(self, "capped_frames", 0),
            "scene_grouping": "data/scene_groups.csv" if load_scene_groups() else "Roboflow source photo only"
        }
        
        with open(class_dist_path, "w", encoding="utf-8") as f_dist:
            json.dump(distribution_data, f_dist, indent=2)
        print(f"[+] Class distribution & weights written to: {class_dist_path}")


def run_pipeline():
    pipeline = FDMDatasetPipeline()
    cleaned = pipeline.scan_and_clean_raw()
    split_records = pipeline.create_stratified_split(cleaned)
    pipeline.materialize_dataset(split_records)


if __name__ == "__main__":
    run_pipeline()
