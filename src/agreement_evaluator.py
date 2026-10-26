"""
agreement_evaluator.py
======================
Recommendation Agreement Rate & Simulation Framework for RQ3.
Authored by Hena Parween (Part 1: Data and Simulation).

RQ3: "How accurately can a rule-based correction framework associate the six classified
FDM defects with appropriate corrective printing actions?"

Independence: the rule table is written by the manufacturing teammate and read from
data/rules/rule_table.json. Hena's reference (src/independent_reference.py) is compiled
from separate sources. This module only compares the two; it contains no rules itself.

Two evaluations:
1. Scenario evaluation - rule table vs. reference on the 18 predefined scenarios,
   assuming a correct, confident classification (isolates rule quality).
2. End-to-end simulation - replays the CS teammate's test-set predictions
   (image -> predicted class + confidence -> rule -> recommendation) and scores each
   recommendation against the reference for the image's TRUE class. This measures what
   a user would actually be told, including misclassifications and low-confidence abstentions.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from src.simulation_scenarios import PREDEFINED_SCENARIOS
from src.independent_reference import INDEPENDENT_REFERENCE_BENCHMARK, get_reference_for_defect

RULE_TABLE_PATH = Path("data/rules/rule_table.json")
DEFAULT_CONFIDENCE_THRESHOLD = 0.65  # De La Rosa et al. (2024)
ACTION_KEYS = ("primary_action", "secondary_action", "tertiary_action")


def load_rule_table(path: Path = RULE_TABLE_PATH) -> Optional[Dict[str, Any]]:
    """Returns the manufacturing teammate's rule table, or None if it is missing or unfilled."""
    if not path.exists():
        print(f"[!] Rule table not found at {path}.")
        return None
    with open(path, encoding="utf-8") as f:
        table = json.load(f)
    missing = [
        defect for defect in INDEPENDENT_REFERENCE_BENCHMARK
        if not (table.get("rules", {}).get(defect, {}).get("primary_action", {}).get("parameter"))
    ]
    if missing:
        print(f"[!] Rule table is not filled in yet for: {', '.join(missing)}")
        return None
    return table


def rule_engine(defect_class: str, confidence: float, table: Dict[str, Any]) -> Dict[str, Any]:
    """Maps a predicted defect + confidence to the rule table's recommendation."""
    threshold = table.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
    if confidence < threshold:
        return {"status": "ABSTAIN", "primary": None, "actions": []}
    rule = table["rules"][defect_class]
    actions = [(a["parameter"], a["direction"]) for a in [rule["primary_action"], *rule.get("secondary_actions", [])]
               if a.get("parameter")]
    return {"status": "RECOMMEND", "primary": actions[0], "actions": actions}


def reference_actions(defect_class: str) -> List[Tuple[str, str]]:
    ref = get_reference_for_defect(defect_class)
    return [(ref[k]["parameter"], ref[k]["direction"]) for k in ACTION_KEYS if k in ref]


def cohens_kappa(labels_a: List[str], labels_b: List[str]) -> float:
    """Cohen's kappa between two raters, with chance agreement from their own label frequencies."""
    n = len(labels_a)
    po = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    freq_a, freq_b = Counter(labels_a), Counter(labels_b)
    pe = sum(freq_a[k] * freq_b[k] for k in freq_a) / (n * n)
    return 1.0 if pe == 1.0 else (po - pe) / (1.0 - pe)


def evaluate_scenarios(table: Dict[str, Any], scenarios: List[Dict[str, Any]] = PREDEFINED_SCENARIOS) -> Dict[str, Any]:
    logs = []
    for sc in scenarios:
        defect = sc["ground_truth_defect"]
        rec = rule_engine(defect, 1.0, table)  # rule quality only: assume a confident, correct classification
        ref = reference_actions(defect)
        logs.append({
            "scenario_id": sc["scenario_id"],
            "defect_class": defect,
            "severity": sc["severity"],
            "rule_primary": "/".join(rec["primary"]),
            "reference_primary": "/".join(ref[0]),
            "primary_parameter_match": rec["primary"][0] == ref[0][0],
            "primary_full_match": rec["primary"] == ref[0],
            "any_action_overlap": bool(set(rec["actions"]) & set(ref)),
        })
    n = len(logs)
    return {
        "n_scenarios": n,
        "n_distinct_defects": len({l["defect_class"] for l in logs}),
        "primary_agreement_rate_pct": round(100 * sum(l["primary_full_match"] for l in logs) / n, 2),
        "primary_parameter_agreement_rate_pct": round(100 * sum(l["primary_parameter_match"] for l in logs) / n, 2),
        "any_action_overlap_rate_pct": round(100 * sum(l["any_action_overlap"] for l in logs) / n, 2),
        "cohens_kappa_primary": round(cohens_kappa([l["rule_primary"] for l in logs],
                                                   [l["reference_primary"] for l in logs]), 4),
        "note": ("The rule engine maps by defect class only, so the 18 scenarios reduce to "
                 f"{len({l['defect_class'] for l in logs})} distinct comparisons; severity does not change the outcome."),
        "scenario_logs": logs,
    }


def simulate_on_predictions(table: Dict[str, Any], predictions_csv: Path) -> Dict[str, Any]:
    """
    predictions_csv columns (written by the CS teammate's inference script on the test split):
    image_path, true_class, pred_class, confidence
    """
    with open(predictions_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    per_class = defaultdict(Counter)
    outcomes = Counter()
    for row in rows:
        true_cls, pred_cls, conf = row["true_class"], row["pred_class"], float(row["confidence"])
        rec = rule_engine(pred_cls, conf, table)
        if rec["status"] == "ABSTAIN":
            outcome = "abstained"
        elif rec["primary"] == reference_actions(true_cls)[0]:
            outcome = "correct_primary_action"
        else:
            outcome = "wrong_primary_action"
        outcomes[outcome] += 1
        per_class[true_cls][outcome] += 1
    n = len(rows)
    acted = n - outcomes["abstained"]
    return {
        "predictions_file": str(predictions_csv),
        "n_test_images": n,
        "end_to_end_agreement_rate_pct": round(100 * outcomes["correct_primary_action"] / n, 2),
        "agreement_when_recommending_pct": round(100 * outcomes["correct_primary_action"] / acted, 2) if acted else None,
        "abstention_rate_pct": round(100 * outcomes["abstained"] / n, 2),
        "outcomes": dict(outcomes),
        "per_class_outcomes": {k: dict(v) for k, v in per_class.items()},
    }


def evaluate_recommendation_agreement(
    predictions_dir: str = "reports",
    output_report_path: str = "reports/AGREEMENT_RATE_EVALUATION.json",
) -> Dict[str, Any]:
    table = load_rule_table()
    if table is None:
        report = {"status": "PENDING_RULE_TABLE",
                  "message": f"Fill in {RULE_TABLE_PATH} (Agya) and re-run."}
    else:
        report = {"status": "OK", "scenario_evaluation": evaluate_scenarios(table), "end_to_end_simulation": {}}
        prediction_files = sorted(Path(predictions_dir).glob("predictions_*.csv"))
        if not prediction_files:
            print(f"[!] No {predictions_dir}/predictions_<model>.csv yet (Harrish); skipping end-to-end simulation.")
        for pf in prediction_files:
            model_name = pf.stem.removeprefix("predictions_")
            report["end_to_end_simulation"][model_name] = simulate_on_predictions(table, pf)

    Path(output_report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[+] RQ3 report written to: {output_report_path} (status: {report['status']})")
    if report["status"] == "OK":
        s = report["scenario_evaluation"]
        print(f"    - Scenario primary-action agreement: {s['primary_agreement_rate_pct']}% (kappa {s['cohens_kappa_primary']})")
        for model_name, sim in report["end_to_end_simulation"].items():
            print(f"    - {model_name}: end-to-end agreement {sim['end_to_end_agreement_rate_pct']}%, "
                  f"abstained {sim['abstention_rate_pct']}%")
    return report


if __name__ == "__main__":
    evaluate_recommendation_agreement()
