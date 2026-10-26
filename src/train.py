"""
train.py
========
Transfer learning for RQ1 and RQ2: MobileNetV2 vs. ResNet-50 on the six-defect split.

Both models use identical settings (split, augmentation, class weights, optimizer,
learning rates, epochs, early stopping, seed). Training has two stages:
  1. backbone frozen, only the new 6-class head is trained;
  2. the whole network is fine-tuned at a lower learning rate.
The checkpoint with the best validation macro-F1 is evaluated once on the test split.

Outputs per model:
  checkpoints/<model>_best.pth
  reports/predictions_<model>.csv     (image_path,true_class,pred_class,confidence; read by RQ3)
  reports/model_metrics_<model>.json
  figures/confusion_matrix_<model>.png, figures/training_curves_<model>.png

Run with the project environment:
  .venv/bin/python -m src.train --model mobilenet
  .venv/bin/python -m src.train --model resnet
  .venv/bin/python -m src.train --model all
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torchvision import models
from torchvision.transforms import v2

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MANIFESTS = Path("data/manifests")
SEED = 42
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Shared by both models (RQ2 requires identical configurations). "baseline" was run first and over-fitted
# (training loss near 0, validation loss rising); "regularized" was added and the final config is the one with
# the higher mean validation macro-F1. The test split is not used for this choice.
CONFIGS = {
    "baseline": {
        "batch_size": 32,
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "stage1_head_epochs": 5,
        "stage1_lr": 1e-3,
        "stage2_max_epochs": 30,
        "stage2_lr": 1e-4,
        "lr_scheduler": "ReduceLROnPlateau(factor=0.1, patience=2) on validation macro-F1",
        "early_stopping_patience": 6,
        "label_smoothing": 0.0,
        "random_resized_crop_scale": None,
        "loss": "cross-entropy with inverse-frequency class weights",
    },
}
CONFIGS["regularized"] = {**CONFIGS["baseline"], "weight_decay": 0.05, "stage2_lr": 3e-5, "label_smoothing": 0.1,
                          "random_resized_crop_scale": [0.6, 1.0]}
CONFIG = {**CONFIGS["baseline"], "name": "baseline", "seed": SEED}  # replaced in main()
OUT = Path(".")  # output root: reports/, figures/, checkpoints/ go under it


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_split():
    """Reads the manifest and decodes every image once into a 224x224 uint8 tensor (about 0.9 GB)."""
    classes = [c for c, _ in sorted(json.loads((MANIFESTS / "class_mapping.json").read_text())["class_to_id"].items(),
                                    key=lambda kv: kv[1])]
    with open(MANIFESTS / "data_split_manifest.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    split = {}
    for name in ("train", "val", "test"):
        part = [r for r in rows if r["split"] == name]
        images = torch.empty((len(part), 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)
        for i, r in enumerate(part):
            with Image.open(r["image_path"]) as img:
                arr = np.array(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
            images[i] = torch.from_numpy(arr).permute(2, 0, 1)
        labels = torch.tensor([int(r["class_id"]) for r in part])
        split[name] = {"images": images, "labels": labels, "paths": [r["image_path"] for r in part]}
        print(f"[*] Loaded {name}: {len(part)} images")
    weights_by_name = json.loads((MANIFESTS / "class_distribution.json").read_text())["class_weights_inverse_frequency"]
    class_weights = torch.tensor([weights_by_name[c] for c in classes], dtype=torch.float32)
    return split, classes, class_weights


def build_train_aug(cfg):
    """Flips, +/-15 degree rotation and colour jitter as in src/preprocessing.py; plus random crops if configured."""
    steps = []
    if cfg.get("random_resized_crop_scale"):
        steps.append(v2.RandomResizedCrop(IMG_SIZE, scale=tuple(cfg["random_resized_crop_scale"]), antialias=True))
    steps += [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(), v2.RandomRotation(15),
              v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.2)]
    return v2.Compose(steps)


TRAIN_AUG = build_train_aug(CONFIG)
NORMALIZE = v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)


def batches(data, batch_size, shuffle, augment, generator=None):
    n = len(data["labels"])
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        imgs = data["images"][idx]
        if augment:
            imgs = torch.stack([TRAIN_AUG(img) for img in imgs])
        yield NORMALIZE(imgs.float() / 255.0), data["labels"][idx], idx


def build_model(name: str, num_classes: int):
    """ImageNet-pretrained backbone with a new classifier head. Returns (model, head_parameters)."""
    if name == "mobilenet":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        head = model.classifier
    elif name == "resnet":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        head = model.fc
    else:
        raise ValueError(f"Unknown model '{name}'")
    return model, list(head.parameters())


@torch.no_grad()
def predict(model, data, device, batch_size=64):
    model.eval()
    probs = []
    for x, _, _ in batches(data, batch_size, shuffle=False, augment=False):
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
    return torch.cat(probs)


def macro_f1(labels, preds, num_classes):
    """Averaged over all classes, so a class the model never predicts counts as 0, as on the test set."""
    return precision_recall_fscore_support(labels, preds, labels=range(num_classes), average="macro", zero_division=0)[2]


def run_epoch(model, data, device, criterion, optimizer, generator):
    model.train()
    total, n = 0.0, 0
    for x, y, _ in batches(data, CONFIG["batch_size"], shuffle=True, augment=True, generator=generator):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


def train_model(name, split, classes, class_weights, device):
    set_seed(CONFIG["seed"])
    generator = torch.Generator().manual_seed(CONFIG["seed"])
    model, head_params = build_model(name, len(classes))
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=CONFIG["label_smoothing"])
    history = []
    best = {"f1": -1.0, "epoch": None}
    ckpt = OUT / "checkpoints" / f"{name}_best.pth"
    ckpt.parent.mkdir(exist_ok=True)

    def validate(epoch, stage, train_loss):
        probs = predict(model, split["val"], device)
        val_loss = nn.functional.nll_loss(torch.log(probs + 1e-12), split["val"]["labels"],
                                          weight=class_weights).item()
        preds = probs.argmax(1).numpy()
        labels = split["val"]["labels"].numpy()
        rec = {"epoch": epoch, "stage": stage, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4),
               "val_accuracy": round(accuracy_score(labels, preds), 4), "val_macro_f1": round(macro_f1(labels, preds, len(classes)), 4)}
        history.append(rec)
        improved = rec["val_macro_f1"] > best["f1"]
        if improved:
            best.update(f1=rec["val_macro_f1"], epoch=epoch)
            torch.save(model.state_dict(), ckpt)
        print(f"    [{name}] epoch {epoch:2d} ({stage}) train_loss {rec['train_loss']:.3f} val_loss {rec['val_loss']:.3f} "
              f"val_acc {rec['val_accuracy']:.3f} val_macroF1 {rec['val_macro_f1']:.3f}{'  *' if improved else ''}")
        return rec["val_macro_f1"]

    start = time.time()
    # Stage 1: train only the new head
    for p in model.parameters():
        p.requires_grad = False
    for p in head_params:
        p.requires_grad = True
    optimizer = torch.optim.AdamW(head_params, lr=CONFIG["stage1_lr"], weight_decay=CONFIG["weight_decay"])
    epoch = 0
    for _ in range(CONFIG["stage1_head_epochs"]):
        epoch += 1
        validate(epoch, "head", run_epoch(model, split["train"], device, criterion, optimizer, generator))

    # Stage 2: fine-tune everything
    for p in model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["stage2_lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=2)
    since_best = 0
    for _ in range(CONFIG["stage2_max_epochs"]):
        epoch += 1
        before = best["f1"]
        f1 = validate(epoch, "finetune", run_epoch(model, split["train"], device, criterion, optimizer, generator))
        scheduler.step(f1)
        since_best = 0 if best["f1"] > before else since_best + 1
        if since_best >= CONFIG["early_stopping_patience"]:
            print(f"    [{name}] early stopping at epoch {epoch}")
            break
    train_minutes = (time.time() - start) / 60
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return model, history, best, train_minutes


@torch.no_grad()
def time_inference(model, data, device, n_images=100, warmup=10):
    """Mean milliseconds per single image (batch size 1), after warm-up."""
    model.eval()
    model.to(device)
    xs = [x for x, _, _ in batches({k: v[:n_images + warmup] if k != "paths" else v for k, v in data.items()},
                                   1, shuffle=False, augment=False)]
    sync = torch.mps.synchronize if device.type == "mps" else (torch.cuda.synchronize if device.type == "cuda" else (lambda: None))
    for x in xs[:warmup]:
        model(x.to(device))
    sync()
    start = time.perf_counter()
    for x in xs[warmup:]:
        model(x.to(device))
    sync()
    return 1000 * (time.perf_counter() - start) / (len(xs) - warmup)


def evaluate_and_save(name, model, history, best, train_minutes, split, classes, device):
    probs = predict(model, split["test"], device)
    conf, preds = probs.max(1)
    labels = split["test"]["labels"].numpy()
    preds = preds.numpy()

    with open(OUT / "reports" / f"predictions_{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "true_class", "pred_class", "confidence"])
        for path, t, p, c in zip(split["test"]["paths"], labels, preds, conf.numpy()):
            w.writerow([path, classes[t], classes[p], f"{c:.4f}"])

    p, r, f1, support = precision_recall_fscore_support(labels, preds, labels=range(len(classes)), zero_division=0)
    cm = confusion_matrix(labels, preds, labels=range(len(classes)))
    n_params = sum(t.numel() for t in model.parameters())
    metrics = {
        "model": name,
        "architecture": "MobileNetV2 (ImageNet)" if name == "mobilenet" else "ResNet-50 (ImageNet)",
        "parameters": n_params,
        "best_epoch": best["epoch"],
        "best_val_macro_f1": best["f1"],
        "epochs_run": len(history),
        "train_minutes": round(train_minutes, 1),
        "test_images": int(len(labels)),
        "test_accuracy": round(accuracy_score(labels, preds), 4),
        "test_macro_precision": round(float(p.mean()), 4),
        "test_macro_recall": round(float(r.mean()), 4),
        "test_macro_f1": round(float(f1.mean()), 4),
        "per_class": {c: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                          "f1": round(float(f1[i]), 4), "support": int(support[i])} for i, c in enumerate(classes)},
        "confusion_matrix": {"labels": classes, "matrix": cm.tolist()},
        "inference_ms_per_image": {
            device.type: round(time_inference(model, split["test"], device), 2),
            "cpu": round(time_inference(model, split["test"], torch.device("cpu")), 2),
        },
        "device": device.type,
        "torch_version": torch.__version__,
        "config": CONFIG,
        "history": history,
    }
    (OUT / "reports" / f"model_metrics_{name}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_figures(name, history, cm, classes)
    print(f"[+] {name}: test accuracy {metrics['test_accuracy']:.4f}, macro-F1 {metrics['test_macro_f1']:.4f}, "
          f"{metrics['inference_ms_per_image']} ms/image")
    return metrics


def plot_figures(name, history, cm, classes):
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="validation")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Weighted cross-entropy"); ax1.set_title("Loss")
    ax2.plot(epochs, [h["val_accuracy"] for h in history], label="val accuracy")
    ax2.plot(epochs, [h["val_macro_f1"] for h in history], label="val macro-F1")
    ax2.set_xlabel("Epoch"); ax2.set_ylim(0, 1); ax2.set_title("Validation")
    stage2 = next((h["epoch"] for h in history if h["stage"] == "finetune"), None)
    for ax in (ax1, ax2):
        if stage2:
            ax.axvline(stage2 - 0.5, color="grey", linestyle=":", linewidth=1)
        ax.legend(frameon=False)
    fig.suptitle(f"{name}: head training, then fine-tuning (dotted line)")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"training_curves_{name}.png", dpi=200)
    plt.close(fig)

    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    short = [c.replace("_", " ") for c in classes]
    ax.set_xticks(range(len(classes)), short, rotation=35, ha="right")
    ax.set_yticks(range(len(classes)), short)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{name}: test confusion matrix (count, row share)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"confusion_matrix_{name}.png", dpi=200)
    plt.close(fig)


def main():
    global CONFIG, OUT, TRAIN_AUG
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mobilenet", "resnet", "all"], default="all")
    parser.add_argument("--config", choices=sorted(CONFIGS), default="baseline")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=".", help="output root for reports/, figures/ and checkpoints/")
    args = parser.parse_args()
    CONFIG = {**CONFIGS[args.config], "name": args.config, "seed": args.seed}
    TRAIN_AUG = build_train_aug(CONFIG)
    OUT = Path(args.out)
    for sub in ("reports", "figures", "checkpoints"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"[*] Device: {device}, config: {args.config}, seed: {args.seed}, output: {OUT}")
    split, classes, class_weights = load_split()
    for name in (["mobilenet", "resnet"] if args.model == "all" else [args.model]):
        print(f"\n>>> Training {name}")
        model, history, best, minutes = train_model(name, split, classes, class_weights, device)
        evaluate_and_save(name, model, history, best, minutes, split, classes, device)


if __name__ == "__main__":
    main()
