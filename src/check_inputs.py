"""
check_inputs.py
===============
Validates the files Hena receives from teammates before they are used:
- data/rules/class_audit.csv        (Agya)
- data/rules/rule_table.json        (Agya)
- reports/predictions_<model>.csv   (Harrish)
Authored by Hena Parween (Part 1: Data and Simulation).

Run on its own (python3 -m src.check_inputs) to get a list of problems to send back,
or via run_hena_pipeline.py, which stops before RQ3 if a received file is malformed.
"""

import csv
import json
from pathlib import Path
from typing import List

from src.data_loader import TARGET_CLASSES

AUDIT_PATH = Path("data/rules/class_audit.csv")
RULE_TABLE_PATH = Path("data/rules/rule_table.json")
MANIFEST_PATH = Path("data/manifests/data_split_manifest.csv")
DIRECTIONS = {"INCREASE", "DECREASE", "ENABLE", "CHECK"}
PREDICTION_COLUMNS = ["image_path", "true_class", "pred_class", "confidence"]


def check_audit(path: Path = AUDIT_PATH) -> List[str]:
    if not path.exists():
        return []
    problems = []
    with open(path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            target = row.get("suggested_class", "").strip().lower().replace(" ", "_").replace("-", "_")
            if target and target != "drop" and target not in TARGET_CLASSES:
                problems.append(f"class_audit.csv line {i}: suggested_class '{row['suggested_class']}' "
                                f"must be one of {', '.join(TARGET_CLASSES)} or 'drop'")
            if target == row.get("labelled_class"):
                problems.append(f"class_audit.csv line {i}: suggested_class equals the current label; leave it blank instead")
    return problems


def check_rule_table(path: Path = RULE_TABLE_PATH) -> List[str]:
    if not path.exists():
        return [f"{path} is missing"]
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"rule_table.json is not valid JSON (line {e.lineno}): {e.msg}"]
    rules = table.get("rules", {})
    if not any(r.get("primary_action", {}).get("parameter") for r in rules.values()):
        return []  # not started yet; agreement_evaluator reports it as pending
    vocab = set(table.get("parameter_vocabulary", []))
    problems = []
    for defect in TARGET_CLASSES:
        rule = rules.get(defect)
        if rule is None:
            problems.append(f"rule_table.json: no entry for '{defect}'")
            continue
        actions = [("primary_action", rule.get("primary_action", {}))]
        actions += [(f"secondary_actions[{j}]", a) for j, a in enumerate(rule.get("secondary_actions", []))]
        for name, a in actions:
            param, direction = a.get("parameter", "").strip(), a.get("direction", "").strip().upper()
            if not param:
                if name == "primary_action":
                    problems.append(f"rule_table.json {defect}: primary_action.parameter is empty")
                continue
            if param not in vocab:
                problems.append(f"rule_table.json {defect} {name}: parameter '{param}' is not in parameter_vocabulary "
                                "(use a listed name, or add it to the vocabulary and tell Hena)")
            if direction not in DIRECTIONS:
                problems.append(f"rule_table.json {defect} {name}: direction '{a.get('direction', '')}' "
                                f"must be one of {', '.join(sorted(DIRECTIONS))}")
        if not rule.get("citations"):
            problems.append(f"rule_table.json {defect}: add at least one citation")
    threshold = table.get("confidence_threshold")
    if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        problems.append(f"rule_table.json: confidence_threshold must be a number between 0 and 1, got {threshold!r}")
    return problems


def check_predictions(predictions_dir: Path = Path("reports"), manifest: Path = MANIFEST_PATH) -> List[str]:
    files = sorted(predictions_dir.glob("predictions_*.csv"))
    if not files:
        return []
    with open(manifest, newline="", encoding="utf-8") as f:
        test = {r["image_path"]: r["class_name"] for r in csv.DictReader(f) if r["split"] == "test"}
    problems = []
    for pf in files:
        with open(pf, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing_cols = [c for c in PREDICTION_COLUMNS if c not in (reader.fieldnames or [])]
            if missing_cols:
                problems.append(f"{pf.name}: missing column(s) {', '.join(missing_cols)}")
                continue
            rows = list(reader)
        seen = set()
        for i, row in enumerate(rows, start=2):
            path = row["image_path"]
            if path not in test:
                problems.append(f"{pf.name} line {i}: '{path}' is not a test-split image in data_split_manifest.csv")
            elif row["true_class"] != test[path]:
                problems.append(f"{pf.name} line {i}: true_class '{row['true_class']}' but the manifest says '{test[path]}'")
            if row["pred_class"] not in TARGET_CLASSES:
                problems.append(f"{pf.name} line {i}: pred_class '{row['pred_class']}' is not a class name")
            try:
                if not 0.0 <= float(row["confidence"]) <= 1.0:
                    raise ValueError
            except ValueError:
                problems.append(f"{pf.name} line {i}: confidence '{row['confidence']}' must be a probability between 0 and 1")
            if path in seen:
                problems.append(f"{pf.name} line {i}: '{path}' appears twice")
            seen.add(path)
        if len(test) - len(seen & test.keys()):
            problems.append(f"{pf.name}: {len(test) - len(seen & test.keys())} of {len(test)} test images have no prediction "
                            "(was it made with an older split? re-run on the current manifest)")
    if len(problems) > 20:
        problems = problems[:20] + [f"... and {len(problems) - 20} more"]
    return problems


def check_all() -> List[str]:
    problems = check_audit() + check_rule_table() + check_predictions()
    if problems:
        print(f"[!] {len(problems)} problem(s) in files received from teammates:")
        for p in problems:
            print(f"    - {p}")
    else:
        print("[+] Received files look valid (or have not arrived yet).")
    return problems


if __name__ == "__main__":
    check_all()
