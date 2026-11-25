#!/usr/bin/env python3
"""
DenseNet121 transfer-learning classifier for Pneumonia vs Normal (PyTorch).

Folder layout (same as your VGG16 project):

    pneumoniaDetection/
    |-- train_densenet121_pneumonia.py
    `-- dataset/
        |-- Normal/
        `-- Pneumonia/

Usage
-----
    pip install torch torchvision scikit-learn pillow matplotlib numpy

    # Train with a fresh leakage-safe split
    python train_densenet121_pneumonia.py --data_dir dataset

    # Train on EXACTLY the same split as your VGG16 run (fair comparison)
    python train_densenet121_pneumonia.py --split_dir outputs

    # Predict one image after training
    python train_densenet121_pneumonia.py --predict path/to/scan.png

Differences from the VGG16 script
---------------------------------
- DenseNet121 backbone (~7M params) with GAP + small head.
- BatchNorm layers stay in eval mode while their weights are frozen
  (otherwise small batches corrupt the pretrained statistics).
- Near-duplicate grouping uses Hamming distance + union-find, not exact hash match.
- Grad-CAM works with DenseNet (no in-place ReLU / backward-hook problems).
- LR scheduler and early stopping both use validation AUC.
- --split_dir lets you reuse split_train/val/test.csv from another run.

Research use only; not a medical device.
"""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Train DenseNet121 for pneumonia classification")
    p.add_argument("--data_dir", default="dataset")
    p.add_argument("--output_dir", default="outputs_densenet121")
    p.add_argument("--split_dir", default=None,
                   help="folder with split_train/val/test.csv to reuse (e.g. 'outputs' from the VGG16 run)")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs_head", type=int, default=8)
    p.add_argument("--epochs_ft", type=int, default=30)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_ft", type=float, default=1e-5)
    p.add_argument("--unfreeze", choices=["block4", "block3_4"], default="block4",
                   help="phase 2: fine-tune denseblock4 (+norm5), or denseblock3+4 (+transition3, norm5)")
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--positive_class", default="Pneumonia")
    p.add_argument("--hash_size", type=int, default=8, help="dHash size (8 -> 64-bit hash)")
    p.add_argument("--hamming", type=int, default=3,
                   help="images with dHash distance <= this are grouped together")
    p.add_argument("--no_group_split", action="store_true")
    p.add_argument("--weights", default="imagenet", help="'imagenet' or 'none'")
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--predict", default=None)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_device(mixed_precision, quiet=False):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if not quiet:
            props = torch.cuda.get_device_properties(0)
            print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {props.total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        if not quiet:
            print("No GPU found - training on CPU (will be slow)")
    if not quiet and mixed_precision:
        print("Mixed precision enabled" if device.type == "cuda" else "Mixed precision requested but no GPU - disabled")
    return device


# --------------------------------------------------------------------------- #
# Data scanning, de-duplication, leakage-safe splitting
# --------------------------------------------------------------------------- #
def dhash_int(path, size=8):
    """Difference hash as a Python int (size*size bits)."""
    resample = getattr(Image, "Resampling", Image).LANCZOS
    with Image.open(path) as im:
        im = im.convert("L").resize((size + 1, size), resample)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_class_names(data_dir, positive_class):
    class_dirs = sorted(d for d in Path(data_dir).iterdir() if d.is_dir())
    names = [d.name for d in class_dirs]
    if len(names) != 2:
        raise SystemExit(f"Expected exactly 2 class folders in {data_dir}, found {names}")
    if positive_class not in names:
        raise SystemExit(f"--positive_class '{positive_class}' not in {names}")
    negative = next(n for n in names if n != positive_class)
    return [negative, positive_class]  # label 0 = negative, 1 = positive


def scan_dataset(data_dir, class_names, hash_size):
    print(f"Classes: 0 -> {class_names[0]}, 1 -> {class_names[1]}")
    records, corrupt = [], 0
    for label, cname in enumerate(class_names):
        files = sorted(p for p in (Path(data_dir) / cname).rglob("*") if p.suffix.lower() in IMG_EXTS)
        print(f"  {cname}: {len(files)} files found")
        for p in files:
            try:
                with Image.open(p) as im:
                    im.verify()
                records.append({"path": str(p), "label": label, "md5": md5(p),
                                "dhash": dhash_int(p, hash_size)})
            except Exception as e:
                corrupt += 1
                print(f"  [skip corrupt] {p}: {e}")
    print(f"Corrupt/unreadable skipped: {corrupt}")
    return records


def drop_exact_duplicates(records):
    by_md5 = defaultdict(list)
    for r in records:
        by_md5[r["md5"]].append(r)
    kept, dupes, conflicts, conflict_paths = [], 0, 0, []
    for items in by_md5.values():
        if len({r["label"] for r in items}) > 1:
            conflicts += len(items)
            conflict_paths.extend(r["path"] for r in items)
            continue
        kept.append(items[0])
        dupes += len(items) - 1
    print(f"Exact duplicates removed: {dupes} | label-conflicting files removed: {conflicts}")
    if conflict_paths:
        print(f"  WARNING: {len(conflict_paths)} files had identical bytes but different labels (first 20):")
        for p in conflict_paths[:20]:
            print(f"    {p}")
    return kept


def popcount64(x):
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def cluster_by_hamming(hashes, max_dist):
    """Union-find: images whose dHash differ by <= max_dist bits share a group id."""
    n = len(hashes)
    h = np.array(hashes, dtype=np.uint64)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n - 1):
        d = popcount64(h[i + 1:] ^ h[i])
        for j in np.nonzero(d <= max_dist)[0]:
            ra, rb = find(i), find(i + 1 + int(j))
            if ra != rb:
                parent[rb] = ra
    return np.array([find(i) for i in range(n)])


def grouped_split(records, val_frac, test_frac, seed, hamming, use_groups=True):
    y = np.array([r["label"] for r in records])
    if use_groups:
        groups = cluster_by_hamming([r["dhash"] for r in records], hamming)
    else:
        groups = np.arange(len(records))

    sizes = Counter(groups.tolist())
    biggest = max(sizes.values())
    print(f"Near-duplicate groups: {len(sizes)} | largest group: {biggest} images")
    if biggest > 0.05 * len(records):
        print("  WARNING: very large group (chaining). Lower --hamming (e.g. 2) for finer grouping.")

    idx = np.arange(len(records))
    n_test = max(2, int(round(1 / test_frac)))
    trainval, test = next(StratifiedGroupKFold(n_test, shuffle=True, random_state=seed).split(idx, y, groups))
    n_val = max(2, int(round((1 - test_frac) / val_frac)))
    tr_rel, va_rel = next(StratifiedGroupKFold(n_val, shuffle=True, random_state=seed)
                          .split(trainval, y[trainval], groups[trainval]))
    train, val = trainval[tr_rel], trainval[va_rel]

    assert not (set(groups[train]) & set(groups[test])), "group leakage train/test"
    assert not (set(groups[train]) & set(groups[val])), "group leakage train/val"
    assert not (set(groups[val]) & set(groups[test])), "group leakage val/test"

    def pack(ix):
        return [records[i]["path"] for i in ix], y[ix].astype(np.float32)

    for name, ix in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name:5s}: {len(ix):5d} images | class counts {dict(sorted(Counter(y[ix].tolist()).items()))}")
    return pack(train), pack(val), pack(test)


def save_split_csv(path, split):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows(zip(split[0], split[1].astype(int)))


def load_split_csv(path):
    paths, labels = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            paths.append(row["path"])
            labels.append(int(row["label"]))
    return paths, np.array(labels, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Dataset & DataLoader
# --------------------------------------------------------------------------- #
class PneumoniaDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)


def build_transforms(img_size, training):
    norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if training:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(contrast=0.1),
            *norm,
        ])
    return transforms.Compose([transforms.Resize((img_size, img_size)), *norm])


def make_loader(paths, labels, img_size, batch_size, training, seed, num_workers=0):
    ds = PneumoniaDataset(paths, labels, build_transforms(img_size, training))
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=training, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), generator=g,
                      persistent_workers=num_workers > 0)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class DenseNet121Pneumonia(nn.Module):
    """DenseNet121 backbone + GAP + small binary head (outputs raw logits)."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = tv_models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = tv_models.densenet121(weights=weights).features  # ends with norm5 (1024 ch)
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )
        self.freeze_base()

    def freeze_base(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze(self, mode="block4"):
        self.freeze_base()
        prefixes = ["denseblock4", "norm5"]
        if mode == "block3_4":
            prefixes += ["denseblock3", "transition3"]
        n = 0
        for name, p in self.features.named_parameters():
            if any(name.startswith(pf) for pf in prefixes):
                p.requires_grad = True
                n += p.numel()
        print(f"Fine-tuning {n:,} backbone parameters in: {', '.join(prefixes)}")

    def train(self, mode=True):
        """Keep BatchNorm layers with frozen weights in eval mode (preserves ImageNet statistics)."""
        super().train(mode)
        if mode:
            for m in self.features.modules():
                if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
                    m.eval()
        return self

    def forward_features(self, x):
        return self.features(x)

    def forward_head(self, feats):
        z = F.relu(feats)                       # non in-place
        z = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return self.head(z)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    probs_all, labels_all = [], []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            if scaler is not None and training:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images).squeeze(1)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)
                if training:
                    loss.backward()
                    optimizer.step()
            probs = torch.sigmoid(logits.detach().float())
            total_loss += loss.item() * images.size(0)
            correct += ((probs >= 0.5).float() == labels).sum().item()
            total += images.size(0)
            probs_all.extend(probs.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    try:
        auc = roc_auc_score(labels_all, probs_all)
    except ValueError:
        auc = 0.0
    return total_loss / total, correct / total, auc


def train_phase(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, epochs, patience, ckpt_path, scaler=None):
    hist = {"loss": [], "acc": [], "auc": [], "val_loss": [], "val_acc": [], "val_auc": []}
    best_auc, wait = -1.0, 0
    for epoch in range(1, epochs + 1):
        tl, ta, tauc = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        vl, va, vauc = run_epoch(model, val_loader, criterion, device)
        for k, v in zip(hist, (tl, ta, tauc, vl, va, vauc)):
            hist[k].append(v)
        print(f"  Epoch {epoch:3d}/{epochs} | loss {tl:.4f} acc {ta:.4f} auc {tauc:.4f} | "
              f"val_loss {vl:.4f} val_acc {va:.4f} val_auc {vauc:.4f}")
        if vauc > best_auc:
            best_auc, wait = vauc, 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"    -> saved checkpoint (val_auc={vauc:.4f})")
        else:
            wait += 1
        if scheduler is not None:
            scheduler.step(vauc)
        if wait >= patience:
            print(f"    Early stopping at epoch {epoch} (patience={patience})")
            break
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    print(f"  Restored best weights (val_auc={best_auc:.4f})")
    return hist


# --------------------------------------------------------------------------- #
# Evaluation + plots
# --------------------------------------------------------------------------- #
@torch.no_grad()
def get_all_probs(model, loader, device):
    model.eval()
    out = []
    for images, _ in loader:
        out.extend(torch.sigmoid(model(images.to(device)).squeeze(1)).cpu().numpy())
    return np.array(out)


def metrics_at(y_true, probs, thr):
    pred = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    return {"threshold": float(thr), "accuracy": float((tp + tn) / len(y_true)),
            "sensitivity_recall": float(sens), "specificity": float(spec),
            "precision": float(prec), "f1": float(f1),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def youden_threshold(y_true, probs):
    fpr, tpr, thr = roc_curve(y_true, probs)
    j = tpr - fpr
    j[~np.isfinite(thr)] = -1.0
    return float(thr[int(np.argmax(j))])


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
    tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    with Image.open(path) as im:
        return tf(im.convert("RGB")).unsqueeze(0)


def load_image_array(path, img_size):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((img_size, img_size)), dtype=np.uint8)


def grad_cam_grid(model, paths, labels, class_names, img_size, out, device,
                  n=8, threshold=0.5, seed=42):
    """Grad-CAM on the final DenseNet feature map (norm5 output)."""
    model.eval()
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
        x = load_image_tensor(paths[i], img_size).to(device).requires_grad_(True)
        with torch.enable_grad():
            feats = model.forward_features(x)
            feats.retain_grad()
            logit = model.forward_head(feats).squeeze()
            model.zero_grad()
            logit.backward()
        weights = feats.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feats.detach()).sum(dim=1))[0].cpu().numpy()
        cam = cam / (cam.max() + 1e-8)
        cam = np.asarray(Image.fromarray((cam * 255).astype("uint8")).resize((img_size, img_size))) / 255.0

        prob = torch.sigmoid(logit).item()
        ax.imshow(load_image_array(paths[i], img_size))
        ax.imshow(cam, cmap="jet", alpha=0.4)
        ax.set_title(f"true: {class_names[int(labels[i])]}\n"
                     f"pred: {class_names[int(prob >= threshold)]} ({prob:.2f})", fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def predict_single(out_dir, image_path, img_size):
    out_dir = Path(out_dir)
    device = setup_device(False, quiet=True)
    meta = json.loads((out_dir / "run_meta.json").read_text())
    img_size = meta.get("args", {}).get("img_size", img_size)

    model = DenseNet121Pneumonia(pretrained=False)
    model.load_state_dict(torch.load(out_dir / "densenet121_pneumonia_final.pth",
                                     map_location=device, weights_only=True))
    model.to(device).eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(load_image_tensor(image_path, img_size).to(device))).item()
    thr = meta["tuned_threshold"]
    label = meta["class_names"][int(prob >= thr)]
    print(f"P({meta['class_names'][1]}) = {prob:.4f} | threshold = {thr:.3f} -> {label}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.predict:
        predict_single(out_dir, args.predict, args.img_size)
        return

    set_seed(args.seed)
    device = setup_device(args.mixed_precision)
    class_names = get_class_names(args.data_dir, args.positive_class)

    # ---- data ----
    if args.split_dir:
        sd = Path(args.split_dir)
        train, val, test = (load_split_csv(sd / f"split_{n}.csv") for n in ("train", "val", "test"))
        print(f"Reusing splits from {sd}: train={len(train[0])} val={len(val[0])} test={len(test[0])}")
    else:
        records = scan_dataset(args.data_dir, class_names, args.hash_size)
        records = drop_exact_duplicates(records)
        train, val, test = grouped_split(records, args.val_frac, args.test_frac, args.seed,
                                         args.hamming, use_groups=not args.no_group_split)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        save_split_csv(out_dir / f"split_{name}.csv", split)

    train_loader = make_loader(*train, args.img_size, args.batch_size, True, args.seed, args.num_workers)
    val_loader = make_loader(*val, args.img_size, args.batch_size, False, args.seed, args.num_workers)
    test_loader = make_loader(*test, args.img_size, args.batch_size, False, args.seed, args.num_workers)

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=train[1].astype(int))
    pos_weight = torch.tensor([cw[1] / cw[0]], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Class weights: 0={cw[0]:.4f}, 1={cw[1]:.4f} | pos_weight={pos_weight.item():.4f}")

    # ---- model ----
    model = DenseNet121Pneumonia(pretrained=str(args.weights).lower() != "none").to(device)
    print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,} | "
          f"trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    scaler = None
    if args.mixed_precision and device.type == "cuda":
        scaler = (torch.amp.GradScaler("cuda") if hasattr(torch.amp, "GradScaler")
                  else torch.cuda.amp.GradScaler())

    # ---- Phase 1: head only ----
    print("\n=== Phase 1: training classifier head (backbone frozen) ===")
    opt1 = optim.Adam([p for p in model.parameters() if p.requires_grad],
                      lr=args.lr_head, weight_decay=1e-4)
    sch1 = optim.lr_scheduler.ReduceLROnPlateau(opt1, mode="max", factor=0.3, patience=2, min_lr=1e-7)
    h1 = train_phase(model, train_loader, val_loader, criterion, opt1, sch1, device,
                     args.epochs_head, patience=4, ckpt_path=out_dir / "phase1_best.pth", scaler=scaler)

    # ---- Phase 2: fine-tune top blocks ----
    print("\n=== Phase 2: fine-tuning top DenseNet blocks ===")
    model.unfreeze(args.unfreeze)
    base = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("features")]
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("features")]
    opt2 = optim.Adam([{"params": base, "lr": args.lr_ft},
                       {"params": head, "lr": args.lr_ft * 10}], weight_decay=1e-4)
    sch2 = optim.lr_scheduler.ReduceLROnPlateau(opt2, mode="max", factor=0.3, patience=3, min_lr=1e-7)
    h2 = train_phase(model, train_loader, val_loader, criterion, opt2, sch2, device,
                     args.epochs_ft, patience=6, ckpt_path=out_dir / "phase2_best.pth", scaler=scaler)

    torch.save(model.state_dict(), out_dir / "densenet121_pneumonia_final.pth")
    plot_history([h1, h2], out_dir / "training_curves.png")

    # ---- evaluation ----
    print("\n=== Evaluation ===")
    val_probs = get_all_probs(model, val_loader, device)
    test_probs = get_all_probs(model, test_loader, device)
    y_val, y_test = val[1].astype(int), test[1].astype(int)

    tuned_thr = youden_threshold(y_val, val_probs)
    res = {
        "model": "densenet121",
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
