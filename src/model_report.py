"""
model_report.py
===============
Writes reports/MODEL_RESULTS.md (report Sections 6-7: Methods and RQ1/RQ2 results)
from reports/model_metrics_<model>.json, written by src/train.py.
"""

import json
from pathlib import Path

from src.data_loader import DISPLAY_NAMES

MODELS = [("mobilenet", "MobileNetV2"), ("resnet", "ResNet-50")]


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _pct(x):
    return f"{100 * x:.1f}%"


def write_model_report(output_path: str = "reports/MODEL_RESULTS.md"):
    metrics = {}
    for key, _ in MODELS:
        path = Path("reports") / f"model_metrics_{key}.json"
        if path.exists():
            metrics[key] = json.loads(path.read_text(encoding="utf-8"))
    if not metrics:
        print("[=] Model results skipped: no reports/model_metrics_<model>.json yet (run src/train.py).")
        return
    label = dict(MODELS)
    cfg = next(iter(metrics.values()))["config"]
    any_m = next(iter(metrics.values()))
    classes = any_m["confusion_matrix"]["labels"]
    devices = [d for d in any_m["inference_ms_per_image"] if d != "cpu"] + ["cpu"]

    overall = [[f"**{label[k]}**", f"{m['parameters'] / 1e6:.1f} M", _pct(m["test_accuracy"]),
                _pct(m["test_macro_precision"]), _pct(m["test_macro_recall"]), _pct(m["test_macro_f1"])]
               + [f"{m['inference_ms_per_image'][d]:.1f}" for d in devices]
               for k, m in metrics.items()]
    per_class = [[DISPLAY_NAMES[c], m0["per_class"][c]["support"]]
                 + [_pct(metrics[k]["per_class"][c]["recall"]) for k in metrics]
                 for c in classes for m0 in [any_m]]
    training = [[label[k], m["epochs_run"], m["best_epoch"], _pct(m["best_val_macro_f1"]), m["train_minutes"]]
                for k, m in metrics.items()]

    seeds_path = Path("reports/seed_runs_summary.json")
    seeds = json.loads(seeds_path.read_text(encoding="utf-8")) if seeds_path.exists() else None
    seed_section = "*Pending:* run the extra seeds and `python3 -m src.seed_summary`."
    if seeds:
        final = seeds["final_config"]

        def ms(e):
            return f"{100 * e['mean']:.1f} ± {100 * e['std']:.1f}" if e["std"] is not None else f"{100 * e['mean']:.1f}"
        seed_rows = [[f"{cfg}{' (final)' if cfg == final else ''}", label[k], ", ".join(map(str, e["seeds"])),
                      ms(e["best_val_macro_f1"]), ms(e["test_accuracy"]), ms(e["test_macro_f1"])]
                     for cfg, models_ in seeds["configs"].items() for k, e in models_.items()]
        seed_section = (f"Each configuration was trained with seeds {', '.join(map(str, seeds['configs'][final]['mobilenet']['seeds']))}. "
                        f"The final configuration (**{final}**) was chosen by mean validation macro-F1 "
                        f"({', '.join(f'{k}: {100 * v:.1f}%' for k, v in seeds['mean_val_macro_f1_by_config'].items())}); "
                        "the test split was not used for the choice. Values are mean ± standard deviation over seeds, in percent. "
                        "Sections 7.1-7.3 show the seed-42 run of the final configuration.\n\n"
                        + _table(["Config", "Model", "Seeds", "Val macro-F1", "Test accuracy", "Test macro-F1"], seed_rows))

    if seeds and len(seeds["configs"][seeds["final_config"]]) == 2:
        fin = seeds["configs"][seeds["final_config"]]
        mob_f1, res_f1 = fin["mobilenet"]["test_macro_f1"], fin["resnet"]["test_macro_f1"]
        mob_cpu, res_cpu = fin["mobilenet"]["inference_ms_per_image_cpu"]["mean"], fin["resnet"]["inference_ms_per_image_cpu"]["mean"]
        gap = 100 * abs(mob_f1["mean"] - res_f1["mean"])
        spread = 100 * max(mob_f1["std"] or 0, res_f1["std"] or 0)
        leader = "MobileNetV2" if mob_f1["mean"] > res_f1["mean"] else "ResNet-50"
        verdict = ("larger than the seed-to-seed spread, so the difference is consistent"
                   if gap > 2 * spread else "within about two standard deviations of the seed-to-seed spread, so the two "
                   "models should be treated as comparable in accuracy")
        rq2 = (f"Over {fin['mobilenet']['test_macro_f1']['n']} seeds, {leader} has the higher mean test macro-F1 "
               f"(MobileNetV2 {100 * mob_f1['mean']:.1f} ± {100 * (mob_f1['std'] or 0):.1f}%, ResNet-50 {100 * res_f1['mean']:.1f} ± "
               f"{100 * (res_f1['std'] or 0):.1f}%). The gap of {gap:.1f} points is {verdict}. MobileNetV2 is "
               f"{res_cpu / mob_cpu:.1f}x faster per image on the CPU ({mob_cpu:.1f} vs. {res_cpu:.1f} ms) with "
               f"{metrics['resnet']['parameters'] / metrics['mobilenet']['parameters']:.0f}x fewer parameters.")
    elif len(metrics) == 2:
        mob, res = metrics["mobilenet"], metrics["resnet"]
        better = "ResNet-50" if res["test_macro_f1"] > mob["test_macro_f1"] else "MobileNetV2"
        speed = res["inference_ms_per_image"]["cpu"] / mob["inference_ms_per_image"]["cpu"]
        rq2 = (f"{better} has the higher test macro-F1 ({_pct(max(res['test_macro_f1'], mob['test_macro_f1']))} vs. "
               f"{_pct(min(res['test_macro_f1'], mob['test_macro_f1']))}, a difference of "
               f"{100 * abs(res['test_macro_f1'] - mob['test_macro_f1']):.1f} points). On the CPU, MobileNetV2 is "
               f"{speed:.1f}x faster per image ({mob['inference_ms_per_image']['cpu']:.1f} vs. "
               f"{res['inference_ms_per_image']['cpu']:.1f} ms) with {res['parameters'] / mob['parameters']:.0f}x fewer parameters. "
               "Both models were trained and tested on the same split with the same settings, from a single seed, "
               "so small differences should not be over-interpreted.")
    else:
        rq2 = "*Pending:* only one model has been trained."

    md = f"""# Methods and Results (RQ1, RQ2)

*Generated by `src/model_report.py` from `reports/model_metrics_<model>.json`. Re-run `src/train.py` to update.*

## 6. Methods

Two ImageNet-pretrained convolutional networks from torchvision {any_m['torch_version'].split('+')[0]} are fine-tuned:
**MobileNetV2** (IMAGENET1K_V1 weights), a lightweight network built from inverted residual blocks with depthwise
separable convolutions, and **ResNet-50** (IMAGENET1K_V2 weights), a 50-layer residual network. In both, the final
ImageNet classifier is replaced by a new fully connected layer with six outputs, one per defect.

Both models use the same data split, preprocessing and augmentation (Section 4), and identical training settings:

{_table(["Setting", "Value"], [
    ["Loss", cfg["loss"]],
    ["Optimizer", f"{cfg['optimizer']}, weight decay {cfg['weight_decay']}"],
    ["Batch size", cfg["batch_size"]],
    ["Stage 1: frozen backbone, head only", f"{cfg['stage1_head_epochs']} epochs, learning rate {cfg['stage1_lr']}"],
    ["Stage 2: full fine-tuning", f"up to {cfg['stage2_max_epochs']} epochs, learning rate {cfg['stage2_lr']}"],
    ["Learning-rate schedule", cfg["lr_scheduler"]],
    ["Early stopping", f"no improvement in validation macro-F1 for {cfg['early_stopping_patience']} epochs"],
    ["Model selection", "checkpoint with the highest validation macro-F1"],
    ["Configuration", cfg.get("name", "baseline")],
    ["Label smoothing", cfg.get("label_smoothing", 0.0)],
    ["Random resized crop (scale)", cfg.get("random_resized_crop_scale") or "none"],
    ["Seed (Sections 7.1-7.3)", cfg["seed"]],
    ["Hardware", f"Apple M4 Pro, PyTorch {any_m['torch_version']} on {any_m['device'].upper()}"],
])}

The test split is used once, for the selected checkpoint. Inference time is the mean over 100 test images,
one image at a time (batch size 1) after 10 warm-up images, measured on the {any_m['device'].upper()} GPU and on the CPU.

## 7. Results

### 7.1 Overall test performance (RQ1, RQ2)

{_table(["Model", "Parameters", "Accuracy", "Macro precision", "Macro recall", "Macro F1"]
        + [f"ms/image ({d.upper()})" for d in devices], overall)}

### 7.2 Per-class recall

{_table(["Defect", "Test images"] + [label[k] for k in metrics], per_class)}

### 7.3 Training

{_table(["Model", "Epochs run", "Best epoch", "Best validation macro-F1", "Training time (min)"], training)}

{chr(10).join(f"![{label[k]} training curves](../figures/training_curves_{k}.png)" for k in metrics)}

{chr(10).join(f"![{label[k]} confusion matrix](../figures/confusion_matrix_{k}.png)" for k in metrics)}

### 7.4 Variation across seeds and configurations

{seed_section}

### 7.5 Answers

**RQ1.** {"; ".join(f"{label[k]} reaches {_pct(m['test_accuracy'])} test accuracy and {_pct(m['test_macro_f1'])} macro-F1 on the {m['test_images']} held-out test images" for k, m in metrics.items())}.

**RQ2.** {rq2}
"""
    Path(output_path).write_text(md, encoding="utf-8")
    print(f"[+] Model results written to: {output_path}")


if __name__ == "__main__":
    write_model_report()
