#!/usr/bin/env python3
"""
VGG16 transfer-learning classifier for Pneumonia vs Normal CT scans (PyTorch).

Expected folder layout:

    pneumoniaDetection/
    |-- train_vgg16_pneumonia.py
    `-- dataset/
        |-- Normal/      *.jpg / *.png
        `-- Pneumonia/   *.jpg / *.png

Usage
-----
    pip install torch torchvision scikit-learn pillow matplotlib numpy
    python train_vgg16_pneumonia.py --data_dir dataset
    python train_vgg16_pneumonia.py --predict path/to/scan.png     # after training

What this script does (and why)
-------------------------------
1. Scans the dataset, drops corrupt files and EXACT duplicates (MD5).
2. Groups near-duplicate images (difference-hash) and splits train/val/test BY GROUP,
   so near-identical slices never leak across splits. This is the #1 reason CT
   pneumonia models report fake 99% accuracy. If you have patient IDs, grouping by
   patient is better still (see GROUPING NOTE below).
3. Builds a DataLoader pipeline with augmentation (train only) and VGG16 ImageNet
   normalisation baked in, so the saved model takes standard PIL-range images.
4. Trains in two phases:
      Phase 1: frozen VGG16 base, train the new head (lr 1e-3)
      Phase 2: unfreeze the last conv block(s), fine-tune with a tiny lr (1e-5)
5. Picks the decision threshold on the VALIDATION set (Youden's J), then reports
   test-set metrics at 0.5 and at the tuned threshold: accuracy, sensitivity,
   specificity, precision, F1, ROC-AUC, confusion matrix.
6. Saves curves, ROC, confusion matrix and Grad-CAM heatmaps so you can check the
   model looks at the lungs and not at borders/text/scanner artefacts.

GROUPING NOTE: if the original dataset has patient IDs in file names, replace
`group_key()` below to return the patient ID. Research use only; not a medical device.
"""

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tv_models, transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Train VGG16 for pneumonia CT classification (PyTorch)")
    p.add_argument("--data_dir", default="dataset", help="folder containing the 2 class folders")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs_head", type=int, default=8, help="phase 1 (frozen base) max epochs")
    p.add_argument("--epochs_ft", type=int, default=30, help="phase 2 (fine-tune) max epochs")
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_ft", type=float, default=1e-5)
    p.add_argument("--unfreeze", choices=["block5", "block4"], default="block5",
                   help="how much of VGG16 to fine-tune in phase 2")
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--positive_class", default="Pneumonia")
    p.add_argument("--hash_size", type=int, default=8, help="dHash size for near-duplicate grouping")
    p.add_argument("--no_group_split", action="store_true",
                   help="disable near-duplicate grouping (NOT recommended)")
    p.add_argument("--weights", default="imagenet", help="'imagenet' or 'none'")
    p.add_argument("--mixed_precision", action="store_true", help="use float16 on modern GPUs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (0 = main process, recommended on Windows)")
    p.add_argument("--predict", default=None, help="path to a single image; runs inference and exits")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_device(mixed_precision):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("No GPU found — training on CPU (will be slow)")
    if mixed_precision and device.type == "cuda":
        print("Mixed precision (float16) enabled")
    elif mixed_precision:
        print("Mixed precision requested but no GPU — disabled")
    return device


# --------------------------------------------------------------------------- #
# Data scanning, de-duplication, leakage-safe splitting
# --------------------------------------------------------------------------- #
def dhash(path, size=8):
    """Difference hash: similar-looking images give the same hash string."""
    resample = getattr(Image, "Resampling", Image).LANCZOS
    with Image.open(path) as im:
        im = im.convert("L").resize((size + 1, size), resample)
    a = np.asarray(im, dtype=np.int16)
    return "".join("1" if b else "0" for b in (a[:, 1:] > a[:, :-1]).ravel())


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def group_key(record):
    """Split-group id. Replace with a patient ID parser if your filenames contain one."""
    return record["dhash"]


def scan_dataset(data_dir, positive_class, hash_size):
    data_dir = Path(data_dir)
    class_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    if len(class_dirs) != 2:
        raise SystemExit(f"Expected exactly 2 class folders in {data_dir}, found {[d.name for d in class_dirs]}")
    names = [d.name for d in class_dirs]
    if positive_class not in names:
        raise SystemExit(f"--positive_class '{positive_class}' not in {names}")
    negative = next(n for n in names if n != positive_class)
    class_names = [negative, positive_class]  # label 0 = negative, 1 = positive
    print(f"Classes: 0 -> {negative}, 1 -> {positive_class}")

    records, corrupt = [], 0
    for label, cname in enumerate(class_names):
        files = sorted(p for p in (data_dir / cname).rglob("*") if p.suffix.lower() in IMG_EXTS)
        print(f"  {cname}: {len(files)} files found")
        for p in files:
            try:
                with Image.open(p) as im:
                    im.verify()
                records.append({"path": str(p), "label": label, "md5": md5(p), "dhash": dhash(p, hash_size)})
            except Exception as e:
                corrupt += 1
                print(f"  [skip corrupt] {p}: {e}")
    print(f"Corrupt/unreadable skipped: {corrupt}")
    return records, class_names


def drop_exact_duplicates(records):
    by_md5 = defaultdict(list)
    for r in records:
        by_md5[r["md5"]].append(r)
    kept, dupes, conflicts = [], 0, 0
    conflict_details = []
    for items in by_md5.values():
        if len({r["label"] for r in items}) > 1:
            conflicts += len(items)
            conflict_details.extend(r["path"] for r in items)
            continue
        kept.append(items[0])
        dupes += len(items) - 1
    print(f"Exact duplicates removed: {dupes} | label-conflicting files removed: {conflicts}")
    if conflict_details:
        print(f"  WARNING: {len(conflict_details)} files had identical bytes but different labels:")
        for p in conflict_details[:20]:
            print(f"    {p}")
        if len(conflict_details) > 20:
            print(f"    ... and {len(conflict_details) - 20} more")
    return kept


def grouped_split(records, val_frac, test_frac, seed, use_groups=True):
    y = np.array([r["label"] for r in records])
    groups = np.array([group_key(r) if use_groups else str(i) for i, r in enumerate(records)])

    sizes = Counter(groups)
    biggest = max(sizes.values())
    print(f"Near-duplicate groups: {len(sizes)} | largest group: {biggest} images")
    if biggest > 0.02 * len(records):
        print("  WARNING: very large group; consider a larger --hash_size (e.g. 12) for finer grouping.")

    idx = np.arange(len(records))
    n_test = max(2, int(round(1 / test_frac)))
    sgkf = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=seed)
    trainval, test = next(sgkf.split(idx, y, groups))

    n_val = max(2, int(round((1 - test_frac) / val_frac)))
    sgkf2 = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=seed)
    tr_rel, va_rel = next(sgkf2.split(trainval, y[trainval], groups[trainval]))
    train, val = trainval[tr_rel], trainval[va_rel]

    assert not (set(groups[train]) & set(groups[test])), "group leakage train/test"
    assert not (set(groups[train]) & set(groups[val])), "group leakage train/val"
    assert not (set(groups[val]) & set(groups[test])), "group leakage val/test"

    def pack(ix):
        return [records[i]["path"] for i in ix], y[ix].astype(np.float32)

    for name, ix in [("train", train), ("val", val), ("test", test)]:
        c = Counter(y[ix].tolist())
        print(f"  {name:5s}: {len(ix):5d} images | class counts {dict(sorted(c.items()))}")
    return pack(train), pack(val), pack(test)


def save_split_csv(path, split):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows(zip(split[0], split[1].astype(int)))


# --------------------------------------------------------------------------- #
# Dataset & DataLoader
# --------------------------------------------------------------------------- #
# ImageNet normalisation (used by torchvision VGG16)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PneumoniaDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label


def build_transforms(img_size, training):
    if training:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.0),  # lungs are symmetric but keep orientation
            transforms.ColorJitter(contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def make_loader(paths, labels, img_size, batch_size, training, seed, num_workers=0):
    transform = build_transforms(img_size, training)
    ds = PneumoniaDataset(paths, labels, transform)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=training,
        num_workers=num_workers, pin_memory=True, generator=g,
        persistent_workers=num_workers > 0,
    )


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class VGG16Pneumonia(nn.Module):
    """VGG16 with a custom binary classification head."""

    # Map block names to VGG16 feature layer index ranges
    BLOCK_RANGES = {
        "block1": (0, 4), "block2": (4, 9), "block3": (9, 16),
        "block4": (16, 23), "block5": (23, 30),
    }

    def __init__(self, pretrained=True):
        super().__init__()
        weights = tv_models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = tv_models.vgg16(weights=weights)
        self.features = vgg.features  # all conv blocks
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )
        # L2 regularisation is applied via optimizer weight_decay instead of inline
        self._freeze_base()

    def _freeze_base(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_blocks(self, mode="block5"):
        blocks = ["block5"] if mode == "block5" else ["block4", "block5"]
        # First freeze everything
        self._freeze_base()
        # Then selectively unfreeze
        n_unfrozen = 0
        for block_name in blocks:
            start, end = self.BLOCK_RANGES[block_name]
            for layer in self.features[start:end]:
                for p in layer.parameters():
                    p.requires_grad = True
                    n_unfrozen += 1
        print(f"Fine-tuning {n_unfrozen} parameters in {', '.join(blocks)}")

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x  # raw logits; sigmoid applied in loss / inference


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        probs = torch.sigmoid(logits).detach()
        preds = (probs >= 0.5).float()
        running_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    try:
        epoch_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        epoch_auc = 0.0
    return epoch_loss, epoch_acc, epoch_auc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        running_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    try:
        epoch_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        epoch_auc = 0.0
    return epoch_loss, epoch_acc, epoch_auc


def train_phase(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, epochs, patience, ckpt_path, scaler=None):
    """Train with early stopping on val AUC. Returns history dict and best val AUC."""
    history = {"loss": [], "acc": [], "auc": [], "val_loss": [], "val_acc": [], "val_auc": []}
    best_auc = -1.0
    wait = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_auc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        va_loss, va_acc, va_auc = evaluate(model, val_loader, criterion, device)

        history["loss"].append(tr_loss)
        history["acc"].append(tr_acc)
        history["auc"].append(tr_auc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_auc"].append(va_auc)

        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"loss {tr_loss:.4f}  acc {tr_acc:.4f}  auc {tr_auc:.4f} | "
              f"val_loss {va_loss:.4f}  val_acc {va_acc:.4f}  val_auc {va_auc:.4f}")

        # Checkpoint best
        if va_auc > best_auc:
            best_auc = va_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), ckpt_path)
            print(f"    -> saved checkpoint (val_auc={va_auc:.4f})")
            wait = 0
        else:
            wait += 1

        # ReduceLROnPlateau
        if scheduler is not None:
            scheduler.step(va_loss)

        # Early stopping
        if wait >= patience:
            print(f"    Early stopping at epoch {epoch} (patience={patience})")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best weights (val_auc={best_auc:.4f})")
    return history


# --------------------------------------------------------------------------- #
# Evaluation + plots
# --------------------------------------------------------------------------- #
def get_all_probs(model, loader, device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_probs)


def metrics_at(y_true, probs, thr):
    pred = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    return {
        "threshold": float(thr), "accuracy": float((tp + tn) / len(y_true)),
        "sensitivity_recall": float(sens), "specificity": float(spec),
        "precision": float(prec), "f1": float(f1),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def youden_threshold(y_true, probs):
    fpr, tpr, thr = roc_curve(y_true, probs)
    best = np.argmax(tpr - fpr)
    return float(thr[best])


def plot_history(histories, out):
    merged = defaultdict(list)
    for h in histories:
        for k, v in h.items():
            merged[k].extend(v)
    split_at = len(histories[0]["loss"])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for a, (k, title) in zip(ax, [("loss", "Loss"), ("acc", "Accuracy"), ("auc", "AUC")]):
        a.plot(merged[k], label="train")
        a.plot(merged["val_" + k], label="val")
        a.axvline(split_at - 1, color="gray", ls="--", label="fine-tune starts")
        a.set_title(title)
        a.set_xlabel("epoch")
        a.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_confusion(cm, class_names, title, out):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], class_names)
    ax.set_yticks([0, 1], class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_roc(y_true, probs, out):
    fpr, tpr, _ = roc_curve(y_true, probs)
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, probs):.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC (test)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def load_image_tensor(path, img_size):
    """Load an image as a normalised tensor (1, 3, H, W) ready for the model."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


def load_image_array(path, img_size):
    """Load image as a uint8 numpy array for display."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize((img_size, img_size))
    return np.asarray(im, dtype=np.uint8)


def grad_cam_grid(model, paths, labels, class_names, img_size, out, device,
                  n=8, threshold=0.5, seed=42):
    """Grad-CAM on VGG16's last conv layer. Check that the model uses lung tissue."""
    model.eval()

    # Hook into the last conv layer (block5_conv3 equivalent = features[28])
    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations["value"] = out

    def bwd_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    last_conv = model.features[28]  # Conv2d in block5
    fwd_handle = last_conv.register_forward_hook(fwd_hook)
    bwd_handle = last_conv.register_full_backward_hook(bwd_hook)

    rng = np.random.RandomState(seed)
    pick = []
    for c in (0, 1):
        ix = np.where(labels == c)[0]
        pick += list(rng.choice(ix, size=min(n // 2, len(ix)), replace=False))

    cols = 4
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")

    for ax, i in zip(axes, pick):
        img_tensor = load_image_tensor(paths[i], img_size).to(device)
        img_tensor.requires_grad_(True)

        logits = model(img_tensor)
        score = torch.sigmoid(logits).squeeze()
        model.zero_grad()
        score.backward()

        act = activations["value"].detach()
        grad = gradients["value"].detach()
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((act * weights).sum(dim=1, keepdim=True))[0, 0].cpu().numpy()
        cam = cam / (cam.max() + 1e-8)
        cam = np.asarray(
            Image.fromarray((cam * 255).astype("uint8")).resize((img_size, img_size))
        ) / 255.0

        prob = score.item()
        arr = load_image_array(paths[i], img_size)
        ax.imshow(arr)
        ax.imshow(cam, cmap="jet", alpha=0.4)
        pred = class_names[int(prob >= threshold)]
        ax.set_title(f"true: {class_names[int(labels[i])]}\npred: {pred} ({prob:.2f})", fontsize=9)
        ax.axis("off")

    fwd_handle.remove()
    bwd_handle.remove()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Inference helper
# --------------------------------------------------------------------------- #
def predict_single(out_dir, image_path, img_size):
    out_dir = Path(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = json.loads((out_dir / "run_meta.json").read_text())
    saved_size = meta.get("args", {}).get("img_size", img_size)
    if saved_size != img_size:
        print(f"[info] Overriding --img_size {img_size} with training value {saved_size}")
    img_size = saved_size

    model = VGG16Pneumonia(pretrained=False)
    model.load_state_dict(torch.load(out_dir / "vgg16_pneumonia_final.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    img_tensor = load_image_tensor(image_path, img_size).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(img_tensor)).item()

    thr = meta["tuned_threshold"]
    label = meta["class_names"][int(prob >= thr)]
    print(f"P({meta['class_names'][1]}) = {prob:.4f}  | threshold = {thr:.3f}  ->  {label}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = setup_device(args.mixed_precision)

    if args.predict:
        predict_single(out_dir, args.predict, args.img_size)
        return

    # ---- data ----
    records, class_names = scan_dataset(args.data_dir, args.positive_class, args.hash_size)
    records = drop_exact_duplicates(records)
    train, val, test = grouped_split(records, args.val_frac, args.test_frac, args.seed,
                                     use_groups=not args.no_group_split)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        save_split_csv(out_dir / f"split_{name}.csv", split)

    train_loader = make_loader(*train, args.img_size, args.batch_size, True, args.seed, args.num_workers)
    val_loader = make_loader(*val, args.img_size, args.batch_size, False, args.seed, args.num_workers)
    test_loader = make_loader(*test, args.img_size, args.batch_size, False, args.seed, args.num_workers)

    # Class weights for imbalanced data
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=train[1].astype(int))
    pos_weight = torch.tensor([cw[1] / cw[0]], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Class weights: {{0: {cw[0]:.4f}, 1: {cw[1]:.4f}}} | pos_weight: {pos_weight.item():.4f}")

    # ---- model ----
    pretrained = str(args.weights).lower() != "none"
    model = VGG16Pneumonia(pretrained=pretrained).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable:     {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Mixed precision scaler
    scaler = torch.GradScaler("cuda") if (args.mixed_precision and device.type == "cuda") else None

    # ---- Phase 1: train head only (VGG16 frozen) ----
    print("\n=== Phase 1: training classifier head (VGG16 frozen) ===")
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer1 = optim.Adam(head_params, lr=args.lr_head, weight_decay=1e-4)
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer1, mode="min", factor=0.3, patience=2, min_lr=1e-7)

    h1 = train_phase(model, train_loader, val_loader, criterion, optimizer1, scheduler1,
                     device, args.epochs_head, patience=4,
                     ckpt_path=out_dir / "phase1_best.pth", scaler=scaler)

    # ---- Phase 2: fine-tune top VGG16 block(s) ----
    # Reload best Phase 1 weights
    phase1_ckpt = out_dir / "phase1_best.pth"
    if phase1_ckpt.exists():
        model.load_state_dict(torch.load(phase1_ckpt, map_location=device, weights_only=True))
        print("Loaded best Phase 1 weights from checkpoint")

    print("\n=== Phase 2: fine-tuning top VGG16 block(s) ===")
    model.unfreeze_blocks(args.unfreeze)

    # Different LR for base vs head
    base_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and n.startswith("features")]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("features")]
    optimizer2 = optim.Adam([
        {"params": base_params, "lr": args.lr_ft},
        {"params": head_params, "lr": args.lr_ft * 10},  # head gets 10x the fine-tune lr
    ], weight_decay=1e-4)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2, mode="min", factor=0.3, patience=3, min_lr=1e-7)

    h2 = train_phase(model, train_loader, val_loader, criterion, optimizer2, scheduler2,
                     device, args.epochs_ft, patience=6,
                     ckpt_path=out_dir / "phase2_best.pth", scaler=scaler)

    # Save final model
    torch.save(model.state_dict(), out_dir / "vgg16_pneumonia_final.pth")
    plot_history([h1, h2], out_dir / "training_curves.png")

    # ---- evaluation ----
    print("\n=== Evaluation ===")
    val_probs = get_all_probs(model, val_loader, device)
    test_probs = get_all_probs(model, test_loader, device)
    y_val, y_test = val[1].astype(int), test[1].astype(int)

    tuned_thr = youden_threshold(y_val, val_probs)
    res = {
        "class_names": class_names,
        "tuned_threshold": tuned_thr,
        "test_roc_auc": float(roc_auc_score(y_test, test_probs)),
        "test_at_0.5": metrics_at(y_test, test_probs, 0.5),
        "test_at_tuned": metrics_at(y_test, test_probs, tuned_thr),
        "n_train": len(train[0]), "n_val": len(val[0]), "n_test": len(test[0]),
        "args": vars(args),
    }
    print(f"Test ROC-AUC: {res['test_roc_auc']:.4f}")
    for key in ("test_at_0.5", "test_at_tuned"):
        m = res[key]
        print(f"[{key}] thr={m['threshold']:.3f} acc={m['accuracy']:.4f} sens={m['sensitivity_recall']:.4f} "
              f"spec={m['specificity']:.4f} prec={m['precision']:.4f} f1={m['f1']:.4f}")
    print("\nClassification report (threshold 0.5):")
    print(classification_report(y_test, (test_probs >= 0.5).astype(int),
                                target_names=class_names, digits=4, zero_division=0))

    for tag, thr in (("0.5", 0.5), ("tuned", tuned_thr)):
        cm = confusion_matrix(y_test, (test_probs >= thr).astype(int), labels=[0, 1])
        plot_confusion(cm, class_names, f"Test confusion (thr={thr:.2f})", out_dir / f"confusion_{tag}.png")
    plot_roc(y_test, test_probs, out_dir / "roc_test.png")
    (out_dir / "run_meta.json").write_text(json.dumps(res, indent=2))

    try:
        grad_cam_grid(model, test[0], test[1].astype(int), class_names, args.img_size,
                      out_dir / "gradcam_samples.png", device, threshold=tuned_thr, seed=args.seed)
    except Exception as e:
        print(f"[warn] Grad-CAM skipped: {e}")

    print(f"\nDone. Model + reports saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
