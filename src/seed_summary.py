"""
seed_summary.py
===============
Summarises the training runs in runs/<config>_seed<seed>/ (written by src/train.py --out).

1. Picks the final training config: the one with the higher mean best-validation macro-F1,
   averaged over both models and all seeds. The test split plays no part in this choice.
2. Promotes the final config's seed-42 run to the main outputs (reports/, figures/, checkpoints/),
   which RQ3 and the report read.
3. Writes reports/seed_runs_summary.json: mean and standard deviation over seeds of each
   test metric, per config and model, for the RQ2 comparison.

  python3 -m src.seed_summary
"""

import json
import re
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

RUNS = Path("runs")
PRIMARY_SEED = 42
METRICS = ["best_val_macro_f1", "test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1"]


def collect():
    runs = defaultdict(dict)  # (config, model) -> {seed: metrics}
    for run_dir in sorted(RUNS.glob("*_seed*")):
        m = re.fullmatch(r"(.+)_seed(\d+)", run_dir.name)
        if not m:
            continue
        for path in (run_dir / "reports").glob("model_metrics_*.json"):
            metrics = json.loads(path.read_text(encoding="utf-8"))
            runs[(m.group(1), metrics["model"])][int(m.group(2))] = metrics
    return runs


def _mean_std(values):
    return {"mean": round(statistics.mean(values), 4),
            "std": round(statistics.stdev(values), 4) if len(values) > 1 else None, "n": len(values)}


def summarise():
    runs = collect()
    if not runs:
        print("[=] No runs in runs/ yet.")
        return None
    summary = {"configs": {}, "selection_rule": "highest mean best_val_macro_f1 over both models and all seeds"}
    for (config, model), by_seed in sorted(runs.items()):
        entry = {"seeds": sorted(by_seed)}
        for key in METRICS:
            entry[key] = _mean_std([by_seed[s][key] for s in sorted(by_seed)])
        cpu = [by_seed[s]["inference_ms_per_image"]["cpu"] for s in sorted(by_seed)]
        entry["inference_ms_per_image_cpu"] = _mean_std(cpu)
        entry["per_class_recall"] = {c: _mean_std([by_seed[s]["per_class"][c]["recall"] for s in sorted(by_seed)])
                                     for c in next(iter(by_seed.values()))["per_class"]}
        summary["configs"].setdefault(config, {})[model] = entry

    val_by_config = {config: statistics.mean(v["best_val_macro_f1"]["mean"] for v in models.values())
                     for config, models in summary["configs"].items()}
    final = max(val_by_config, key=val_by_config.get)
    summary["mean_val_macro_f1_by_config"] = {k: round(v, 4) for k, v in val_by_config.items()}
    summary["final_config"] = final
    Path("reports/seed_runs_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[+] Mean validation macro-F1 by config: {summary['mean_val_macro_f1_by_config']} -> final: {final}")

    source = RUNS / f"{final}_seed{PRIMARY_SEED}"
    if source.exists():
        for sub in ("reports", "figures", "checkpoints"):
            for f in (source / sub).iterdir():
                shutil.copy2(f, Path(sub) / f.name)
        print(f"[+] Promoted {source} to reports/, figures/ and checkpoints/")
    return summary


if __name__ == "__main__":
    summarise()
