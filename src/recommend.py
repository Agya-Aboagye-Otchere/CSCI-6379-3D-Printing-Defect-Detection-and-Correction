"""
recommend.py
============
End-to-end inference: image -> defect class + confidence -> corrective recommendation.

Uses a trained checkpoint from src/train.py and the rule table in data/rules/rule_table.json
through the same rule_engine that the RQ3 evaluation uses.

  .venv/bin/python -m src.recommend path/to/image.jpg [more images] [--model resnet]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.agreement_evaluator import load_rule_table, rule_engine
from src.train import IMG_SIZE, NORMALIZE, build_model, pick_device

MANIFESTS = Path("data/manifests")


def load_classifier(name: str, device):
    mapping = json.loads((MANIFESTS / "class_mapping.json").read_text())
    classes = [c for c, _ in sorted(mapping["class_to_id"].items(), key=lambda kv: kv[1])]
    model, _ = build_model(name, len(classes))
    model.load_state_dict(torch.load(Path("checkpoints") / f"{name}_best.pth", map_location=device, weights_only=True))
    return model.to(device).eval(), classes, mapping["display_names"]


@torch.no_grad()
def classify(model, classes, image_path: str, device):
    with Image.open(image_path) as img:
        arr = np.array(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    x = NORMALIZE(torch.from_numpy(arr).permute(2, 0, 1).float().div(255)).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1)[0].cpu()
    conf, idx = probs.max(0)
    return classes[idx], float(conf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--model", choices=["mobilenet", "resnet"], default="resnet")
    args = parser.parse_args()
    device = pick_device()
    model, classes, display = load_classifier(args.model, device)
    table = load_rule_table()
    for path in args.images:
        defect, conf = classify(model, classes, path, device)
        print(f"\n{path}\n  Predicted defect: {display[defect]} (confidence {conf:.2f})")
        if table is None:
            print("  Recommendation: unavailable until the rule table is filled in")
            continue
        rec = rule_engine(defect, conf, table)
        if rec["status"] == "ABSTAIN":
            print(f"  Recommendation: none, confidence below {table.get('confidence_threshold', 0.65)}; inspect manually")
            continue
        rule = table["rules"][defect]
        for label, action in [("Primary", rule["primary_action"])] + [("Also", a) for a in rule.get("secondary_actions", [])]:
            if action.get("parameter"):
                print(f"  {label}: {action['direction']} {action['parameter']} {action.get('delta', '')}".rstrip())


if __name__ == "__main__":
    main()
