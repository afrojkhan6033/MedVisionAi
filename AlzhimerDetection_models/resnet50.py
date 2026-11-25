"""
ResNet50 trainer for the Alzheimer MRI Preprocessed Dataset (4 classes).

Single self-contained file:
  scan folders -> dedupe -> stratified 70/15/15 split (cached to splits.csv)
  -> 2-phase transfer learning -> early stopping on val macro-F1
  -> final report on the held-out test set.

Usage:
  pip install torch torchvision scikit-learn pandas pillow matplotlib tqdm
  python resnet50.py --data_root "D:/datasets/Alzheimer (Preprocessed Data)"

The split file is written to <out_dir>/../splits.csv and reused on later runs,
so vgg16.py can train on the exact same split for a fair comparison.
"""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

# ordinal order: healthy -> most severe
CLASSES = ["Non_Demented", "Very_Mild_Demented", "Mild_Demented", "Moderate_Demented"]
NC = len(CLASSES)
LABELS = list(range(NC))
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ----------------------------------------------------------------------------- utils
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ----------------------------------------------------------------------------- data
def dhash(img, size=16):
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten()).tobytes().hex()


def build_splits(root, seed, csv_path, dedupe=True):
    csv_path = Path(csv_path)
    if csv_path.exists():
        print(f"Reusing existing split: {csv_path}")
        return pd.read_csv(csv_path)

    root = Path(root)
    rows = []
    for label, cls in enumerate(CLASSES):
        dirs = [d for d in root.rglob(cls) if d.is_dir()]
        if root.name == cls:
            dirs.append(root)
        if not dirs:
            raise SystemExit(f"Class folder '{cls}' not found under {root}")
        for d in dirs:
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() in EXTS:
                    rows.append({"path": str(p.resolve()), "label": label, "cls": cls})
    df = pd.DataFrame(rows)
    print(f"Found {len(df)} images\n{df.cls.value_counts().to_string()}")

    if dedupe:
        md5s, dhs = [], []
        for p in tqdm(df.path, desc="hashing"):
            with Image.open(p) as im:
                md5s.append(hashlib.md5(np.asarray(im.convert("L")).tobytes()).hexdigest())
                dhs.append(dhash(im))
        df["md5"], df["dh"] = md5s, dhs
        n0 = len(df)
        conflict = df.groupby("dh").label.nunique()
        df = df[~df.dh.isin(conflict[conflict > 1].index)]      # same image, different labels -> drop
        n_conf = n0 - len(df)
        df = df.drop_duplicates("md5").drop_duplicates("dh").drop(columns=["md5", "dh"])
        print(f"Removed {n_conf} label-conflicting + {n0 - n_conf - len(df)} duplicate images -> {len(df)} left")
    df = df.reset_index(drop=True)

    tv, te = train_test_split(df.index, test_size=0.15, stratify=df.label, random_state=seed)
    tr, va = train_test_split(tv, test_size=0.15 / 0.85, stratify=df.label[tv], random_state=seed)
    df["split"] = ""
    df.loc[tr, "split"], df.loc[va, "split"], df.loc[te, "split"] = "train", "val", "test"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(pd.crosstab(df.cls, df.split).to_string(), f"\nSaved {csv_path}")
    return df


class AddNoise:
    def __init__(self, std=0.01, p=0.3):
        self.std, self.p = std, p

    def __call__(self, t):
        return (t + torch.randn_like(t) * self.std).clamp(0, 1) if random.random() < self.p else t


def make_transform(size, train, hflip=False):
    resize = T.Resize((size, size), interpolation=InterpolationMode.BICUBIC, antialias=True)
    norm = T.Normalize(MEAN, STD)
    if not train:
        return T.Compose([resize, T.ToTensor(), norm])
    aug = [T.RandomAffine(10, translate=(0.05, 0.05), scale=(0.9, 1.1), interpolation=InterpolationMode.BILINEAR),
           T.ColorJitter(brightness=0.2, contrast=0.2)]
    if hflip:
        aug.append(T.RandomHorizontalFlip())
    return T.Compose(aug + [resize, T.ToTensor(), AddNoise(), norm])


class MRIDataset(Dataset):
    """Images are tiny (128x128 gray): cache them in RAM, replicate to 3 channels on the fly."""

    def __init__(self, df, transform):
        self.labels = df.label.astype(int).tolist()
        self.transform = transform
        self.imgs = []
        for p in tqdm(df.path, desc="loading", leave=False):
            with Image.open(p) as im:
                self.imgs.append(im.convert("L").copy())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.transform(self.imgs[i].convert("RGB")), self.labels[i]


# ----------------------------------------------------------------------------- model
def build_model(pretrained=True, drop=0.3):
    w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = models.resnet50(weights=w)
    m.fc = nn.Sequential(nn.Dropout(drop), nn.Linear(m.fc.in_features, NC))
    return m


def param_groups(model):
    head = list(model.fc.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return backbone, head


def set_backbone_trainable(model, flag):
    backbone, head = param_groups(model)
    for p in backbone:
        p.requires_grad = flag
    for p in head:
        p.requires_grad = True


def warmup_cosine(total, warmup):
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
    return f


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        with torch.autocast(device.type, enabled=amp):
            out = model(x.to(device, non_blocking=True))
        ps.append(out.float().softmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def ece_score(y, probs, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges, ece = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


def compute_metrics(y, probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum(1, keepdims=True)
    pred = probs.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    try:
        auc = float(roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=LABELS))
    except ValueError:
        auc = float("nan")
    diff = np.abs(pred - y)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f.mean()),
        "auc_ovr": auc,
        "ece": ece_score(y, probs),
        "ordinal_mae": float(diff.mean()),
        "errors_2plus_stages": float((diff >= 2).mean()),
        "per_class": {CLASSES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                   "f1": float(f[i]), "support": int(s[i])} for i in LABELS},
        "confusion_matrix": confusion_matrix(y, pred, labels=LABELS).tolist(),
    }


def bootstrap_ci(y, probs, n_boot=1000, seed=0):
    """Stratified bootstrap -> 95% CI for macro-F1, balanced acc and per-class recall."""
    rng, pred = np.random.default_rng(seed), probs.argmax(1)
    cls_idx = [np.where(y == c)[0] for c in LABELS]
    keys = ["macro_f1", "balanced_acc"] + [f"recall_{c}" for c in CLASSES]
    res = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(ix, len(ix)) for ix in cls_idx if len(ix)])
        yt, yp = y[idx], pred[idx]
        res["macro_f1"].append(f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0))
        res["balanced_acc"].append(balanced_accuracy_score(yt, yp))
        for c, v in zip(CLASSES, recall_score(yt, yp, labels=LABELS, average=None, zero_division=0)):
            res[f"recall_{c}"].append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in res.items()}


def plot_confusion(cm, path, title):
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, fmt, sub in [(axes[0], cm, "d", "counts"), (axes[1], norm, ".2f", "row-normalised")]:
        ax.imshow(data, cmap="Blues")
        names = [c.replace("_", "\n") for c in CLASSES]
        ax.set_xticks(LABELS, names, fontsize=8)
        ax.set_yticks(LABELS, names, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title} ({sub})")
        for i in LABELS:
            for j in LABELS:
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > data.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def plot_history(hist, path):
    h = pd.DataFrame(hist)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h.epoch, h.train_loss)
    ax[0].set_title("train loss")
    ax[1].plot(h.epoch, h.val_macro_f1, label="val macro-F1")
    ax[1].plot(h.epoch, h.val_bal_acc, label="val balanced acc")
    ax[1].legend()
    ax[1].set_title("validation")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def report(name, y, probs, out_dir, n_boot):
    m = compute_metrics(y, probs.copy())
    m["ci95"] = bootstrap_ci(y, probs, n_boot)
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(m, indent=2))
    plot_confusion(m["confusion_matrix"], out_dir / f"{name}_confusion_matrix.png", name)
    ci = m["ci95"]
    print(f"\n===== {name.upper()} ({len(y)} images) =====")
    print(f"accuracy {m['accuracy']:.4f} | balanced acc {m['balanced_acc']:.4f} "
          f"(CI {ci['balanced_acc'][0]:.3f}-{ci['balanced_acc'][1]:.3f})")
    print(f"macro F1 {m['macro_f1']:.4f} (CI {ci['macro_f1'][0]:.3f}-{ci['macro_f1'][1]:.3f}) "
          f"| AUC {m['auc_ovr']:.4f} | ECE {m['ece']:.4f}")
    print(f"ordinal MAE {m['ordinal_mae']:.4f} | errors >= 2 stages apart: {m['errors_2plus_stages']:.4f}")
    for c, d in m["per_class"].items():
        lo, hi = ci[f"recall_{c}"]
        print(f"  {c:<20} P {d['precision']:.3f}  R {d['recall']:.3f} (CI {lo:.2f}-{hi:.2f})  "
              f"F1 {d['f1']:.3f}  n={d['support']}")
    return m


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="folder containing the 4 class folders")
    ap.add_argument("--out_dir", default="runs/resnet50")
    ap.add_argument("--splits", default=None, help="default: <out_dir>/../splits.csv (shared across models)")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--head_epochs", type=int, default=3, help="phase 1: backbone frozen")
    ap.add_argument("--warmup_epochs", type=float, default=1)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_backbone", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--imbalance", choices=["none", "sampler", "weights", "both"], default="both")
    ap.add_argument("--imbalance_power", type=float, default=0.5)
    ap.add_argument("--hflip", action="store_true", help="horizontal-flip augmentation (off by default)")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--no_test", action="store_true", help="skip the final test-set evaluation")
    ap.add_argument("--workers", type=int, default=4, help="use 0 if DataLoader workers fail on Windows")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"device={device} amp={amp}")

    # ---- data
    splits = args.splits or str(out.parent / "splits.csv")
    df = build_splits(args.data_root, args.seed, splits)
    tr_df, va_df, te_df = (df[df.split == s] for s in ("train", "val", "test"))
    tr_ds = MRIDataset(tr_df, make_transform(args.img_size, True, args.hflip))
    va_ds = MRIDataset(va_df, make_transform(args.img_size, False))

    counts = np.bincount(tr_ds.labels, minlength=NC).astype(float)
    w = (counts.sum() / np.maximum(counts, 1)) ** args.imbalance_power
    sampler = loss_w = None
    if args.imbalance in ("sampler", "both"):
        sampler = WeightedRandomSampler(torch.tensor(w[tr_ds.labels], dtype=torch.double), len(tr_ds), True)
    if args.imbalance in ("weights", "both"):
        loss_w = torch.tensor(w / w.mean(), dtype=torch.float, device=device)
    print(f"train class counts {counts.astype(int).tolist()} | imbalance={args.imbalance} "
          f"| loss weights {None if loss_w is None else loss_w.cpu().numpy().round(2).tolist()}")

    kw = dict(num_workers=args.workers, pin_memory=amp, persistent_workers=args.workers > 0)
    tr_loader = DataLoader(tr_ds, args.batch_size, sampler=sampler, shuffle=sampler is None, drop_last=True, **kw)
    va_loader = DataLoader(va_ds, 64, shuffle=False, **kw)

    # ---- model / loss
    model = build_model(not args.no_pretrained, args.drop).to(device)
    crit = nn.CrossEntropyLoss(weight=loss_w, label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    def start_phase(phase):
        if phase == 1:      # head only, constant LR
            set_backbone_trainable(model, False)
            return AdamW(model.fc.parameters(), lr=args.lr_head, weight_decay=args.weight_decay), None
        set_backbone_trainable(model, True)     # full fine-tune, discriminative LRs, warmup + cosine
        backbone, head = param_groups(model)
        opt = AdamW([{"params": backbone, "lr": args.lr_backbone},
                     {"params": head, "lr": args.lr_head}], weight_decay=args.weight_decay)
        total = max(1, (args.epochs - args.head_epochs) * len(tr_loader))
        return opt, LambdaLR(opt, warmup_cosine(total, int(args.warmup_epochs * len(tr_loader))))

    # ---- training loop
    best_f1, bad, phase, hist = -1.0, 0, 0, []
    opt = sched = None
    for epoch in range(1, args.epochs + 1):
        want = 1 if epoch <= args.head_epochs else 2
        if want != phase:
            phase, bad = want, 0
            opt, sched = start_phase(phase)
            print(f"\n--- phase {phase}: {'head only (backbone frozen)' if phase == 1 else 'full fine-tune'} ---")

        model.train()
        if phase == 1:      # keep ImageNet BatchNorm statistics while the backbone is frozen
            for mod in model.modules():
                if isinstance(mod, nn.BatchNorm2d):
                    mod.eval()

        t0, run_loss, n = time.time(), 0.0, 0
        for x, y in tqdm(tr_loader, desc=f"epoch {epoch}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                logits = model(x)
            loss = crit(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if sched:
                sched.step()
            run_loss += loss.item() * len(y)
            n += len(y)

        yv, pv = predict(model, va_loader, device, amp)
        m = compute_metrics(yv, pv)
        hist.append({"epoch": epoch, "phase": phase, "train_loss": run_loss / n, "val_acc": m["accuracy"],
                     "val_bal_acc": m["balanced_acc"], "val_macro_f1": m["macro_f1"]})
        rec = " ".join(f"{d['recall']:.2f}" for d in m["per_class"].values())
        print(f"epoch {epoch:02d} | loss {run_loss / n:.4f} | val acc {m['accuracy']:.4f} "
              f"bal {m['balanced_acc']:.4f} F1 {m['macro_f1']:.4f} | recall/class [{rec}] | {time.time() - t0:.0f}s")

        if m["macro_f1"] > best_f1 + 1e-4:
            best_f1, bad = m["macro_f1"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_macro_f1": best_f1,
                        "args": vars(args)}, out / "best.pt")
            print(f"   saved best (val macro-F1 {best_f1:.4f})")
        elif phase == 2:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping: no val macro-F1 improvement for {args.patience} epochs")
                break

    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    plot_history(hist, out / "training_curves.png")

    # ---- final evaluation with the best checkpoint
    ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    print(f"\nLoaded best checkpoint from epoch {ck['epoch']}")
    yv, pv = predict(model, va_loader, device, amp)
    report("val", yv, pv, out, args.n_boot)

    if not args.no_test:    # touch the test set once, at the very end; do not tune on it
        te_ds = MRIDataset(te_df, make_transform(args.img_size, False))
        te_loader = DataLoader(te_ds, 64, shuffle=False, **kw)
        yt, pt = predict(model, te_loader, device, amp)
        report("test", yt, pt, out, args.n_boot)
        np.save(out / "test_probs.npy", pt)
        pd.DataFrame({"path": te_df.path.values, "true": yt, "pred": pt.argmax(1)}).to_csv(
            out / "test_predictions.csv", index=False)
    print(f"\nAll outputs saved in {out.resolve()}")


if __name__ == "__main__":
    main()
