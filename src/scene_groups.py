"""
scene_groups.py
===============
Groups near-duplicate images (frames of the same video, re-uploads with rotation,
noise or colour shifts) into "scenes", so that a whole scene goes to one split.
Authored by Hena Parween (Part 1: Data and Simulation).

Why: grouping by Roboflow source filename is not enough. With the filename split,
18% of test images had a train image with a near-identical perceptual hash, and the
median test image had a train neighbour at cosine similarity 0.92; layer shifting
(899 images) turned out to come from only ~33-70 distinct scenes.

How: every image is embedded with an ImageNet-pretrained ResNet-50 (not the fine-tuned
classifier, so grouping follows scene content, not class). Within each class, images are
linked if their cosine similarity is >= SIMILARITY_THRESHOLD or they share a Roboflow
source photo, and linked images form one scene (connected components). The threshold
was chosen by inspecting test/train pairs: almost every pair at >= 0.75 showed the same
printed object or scene.

Writes data/scene_groups.csv (md5, class_name, scene_id), read by data_loader.py, and
data/scene_groups_summary.json for the report. Needs PyTorch:
  .venv/bin/python -m src.scene_groups            # compute scenes
  .venv/bin/python -m src.scene_groups --check    # residual test->train similarity of the current split
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models
from torchvision.transforms import v2

from src.data_loader import TARGET_CLASSES
from src.ingest_roboflow import source_id

RAW_DIR = Path("data/raw")
OUT_CSV = Path("data/scene_groups.csv")
OUT_SUMMARY = Path("data/scene_groups_summary.json")
SIMILARITY_THRESHOLD = 0.75


def _device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def embed(paths, batch_size=64):
    device = _device()
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    normalize = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    out = []
    for start in range(0, len(paths), batch_size):
        xs = []
        for p in paths[start:start + batch_size]:
            with Image.open(p) as img:
                xs.append(torch.from_numpy(np.array(img.convert("RGB").resize((224, 224)))).permute(2, 0, 1).float() / 255)
        out.append(torch.nn.functional.normalize(model(normalize(torch.stack(xs)).to(device)), dim=1).cpu())
    return torch.cat(out).numpy()


def _components(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    return [find(i) for i in range(n)]


def compute_scene_groups():
    rows, summary = [], {"similarity_threshold": SIMILARITY_THRESHOLD, "classes": {}}
    for cls in TARGET_CLASSES:
        paths = sorted(p for p in (RAW_DIR / cls).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        emb = embed(paths)
        sim = emb @ emb.T
        edges = list(zip(*np.where(np.triu(sim >= SIMILARITY_THRESHOLD, 1))))
        by_source = {}
        for i, p in enumerate(paths):
            by_source.setdefault(source_id(p.name.split("__", 1)[-1]), []).append(i)
        edges += [(members[0], j) for members in by_source.values() for j in members[1:]]
        roots = _components(len(paths), edges)
        scene_index = {r: k for k, r in enumerate(dict.fromkeys(roots))}
        sizes = np.bincount([scene_index[r] for r in roots])
        for p, r in zip(paths, roots):
            rows.append((hashlib.md5(p.read_bytes()).hexdigest(), cls, f"{cls}_scene{scene_index[r]:04d}"))
        summary["classes"][cls] = {"images": len(paths), "scenes": len(sizes), "largest_scene": int(sizes.max()),
                                   "largest_scene_share": round(float(sizes.max() / len(paths)), 3)}
        print(f"    - {cls}: {len(paths)} images in {len(sizes)} scenes (largest {sizes.max()})")
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["md5", "class_name", "scene_id"])
        w.writerows(rows)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[+] Scene groups written to {OUT_CSV}")


def check_split(manifest=Path("data/manifests/data_split_manifest.csv")):
    """Highest cosine similarity of each val/test image to any train image, for the current split."""
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    emb = embed([r["image_path"] for r in rows])
    split = np.array([r["split"] for r in rows])
    result = {}
    for name in ("val", "test"):
        best = (emb[split == name] @ emb[split == "train"].T).max(1)
        result[name] = {"median_max_cosine_to_train": round(float(np.median(best)), 3),
                        f"share_at_or_above_{SIMILARITY_THRESHOLD}": round(float((best >= SIMILARITY_THRESHOLD).mean()), 3),
                        "share_at_or_above_0.9": round(float((best >= 0.9).mean()), 3)}
    print(json.dumps(result, indent=2))
    summary = json.loads(OUT_SUMMARY.read_text()) if OUT_SUMMARY.exists() else {}
    summary["residual_similarity_after_split"] = result
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    check_split() if args.check else compute_scene_groups()
