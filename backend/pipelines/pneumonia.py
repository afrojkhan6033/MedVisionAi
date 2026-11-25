"""Pneumonia binary adapter — PyTorch DenseNet-121 + VGG-16."""
from __future__ import annotations
from typing import Any, Optional

import numpy as np
from PIL import Image

from .base import BasePipeline

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PneumoniaPipeline(BasePipeline):
    def __init__(self, manifest: dict) -> None:
        super().__init__(manifest)
        self._models: dict[str, Any] = {}
        self._torch = None
        self._cfg = manifest.get("input", {})
        self._positive = manifest.get("positive_class", "Pneumonia")

    def load(self) -> None:
        try:
            import torch, torch.nn as nn, torch.nn.functional as F  # noqa: E401
            import torchvision.models as tvm
            self._torch = torch
        except ImportError:
            print("[pneumonia] PyTorch not installed"); return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = str(device)

        for mdl in self.manifest.get("models", []):
            if not mdl.get("enabled", True):
                continue
            wpath = self.artifact_dir / mdl["weights"]
            if not wpath.exists():
                print(f"[pneumonia] Weights not found: {wpath}"); continue

            mid = mdl["id"]
            try:
                if mid == "densenet121":
                    backbone = tvm.densenet121(weights=None).features
                    m = _DenseNetBinary(backbone)
                elif mid == "vgg16":
                    features = tvm.vgg16(weights=None).features
                    m = _VGGBinary(features)
                else:
                    continue
                m.load_state_dict(torch.load(wpath, map_location=device, weights_only=True))
                m.to(device).eval()
                self._models[mid] = m
                print(f"[pneumonia] Loaded {mid}")
            except Exception as exc:
                print(f"[pneumonia] Failed to load {mid}: {exc}")
        self.loaded = bool(self._models)

    def preprocess(self, img: Image.Image) -> Any:
        from torchvision import transforms as T
        size = (self._cfg.get("height", 224), self._cfg.get("width", 224))
        tfm = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return tfm(img.convert("RGB")).unsqueeze(0)

    def predict(self, inputs: Any) -> dict:
        torch = self._torch
        results: dict[str, float] = {}
        if not self._models:
            return {"per_model": results}
        device = next(next(iter(self._models.values())).parameters()).device
        x = inputs.to(device)
        with torch.no_grad():
            for mid, m in self._models.items():
                logit = m(x).squeeze()
                prob = torch.sigmoid(logit).item()
                results[mid] = prob
        return {"per_model": results}

    def postprocess(self, raw: dict) -> dict:
        per_model = raw.get("per_model", {})
        if not per_model:
            return {"prediction": "Unavailable", "confidence": 0.0, "is_abnormal": False,
                    "probabilities": {c: 0.0 for c in self.classes}, "model_results": []}

        ew = self.manifest.get("ensemble", {}).get("weights", [])
        model_results, weighted_prob = [], 0.0
        for i, (mid, prob) in enumerate(per_model.items()):
            pred = self._positive if prob >= 0.5 else self.classes[0]
            w = ew[i] if i < len(ew) else 1.0
            weighted_prob += prob * w
            dname = mid
            for mdl in self.manifest.get("models", []):
                if mdl["id"] == mid: dname = mdl["display_name"]
            model_results.append({"name": dname, "prediction": pred, "confidence": float(max(prob, 1 - prob))})
        weighted_prob /= sum(ew[:len(per_model)]) if ew else len(per_model)

        is_positive = weighted_prob >= 0.5
        pred_label = self._positive if is_positive else self.classes[0]
        conf = weighted_prob if is_positive else 1 - weighted_prob
        return {
            "prediction": pred_label, "confidence": float(conf), "is_abnormal": is_positive, "threshold": 0.5,
            "probabilities": {self.classes[0]: float(1 - weighted_prob), self._positive: float(weighted_prob)},
            "model_results": model_results,
        }

    def explain(self, img: Image.Image, inputs: Any, target_class: int) -> Optional[str]:
        torch = self._torch
        if torch is None or "densenet121" not in self._models:
            return None
        model = self._models["densenet121"]
        try:
            device = next(model.parameters()).device
            x = inputs.to(device).requires_grad_(True)
            with torch.enable_grad():
                feats = model.features(x)
                feats.retain_grad()
                import torch.nn.functional as F
                z = F.relu(feats)
                z = F.adaptive_avg_pool2d(z, 1).flatten(1)
                logit = model.head(z).squeeze()
                model.zero_grad()
                logit.backward()
            w = feats.grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((w * feats.detach()).sum(dim=1))[0].cpu().numpy()
            cam = cam / (cam.max() + 1e-8)
            return self._cam_overlay(img, cam)
        except Exception:
            return None


# ── lightweight model wrappers (match training scripts) ──────────
class _DenseNetBinary:
    """Mirrors DenseNet121Pneumonia from train_densenet121_pneumonia.py."""
    def __init__(self, features):
        import torch.nn as nn, torch.nn.functional as F
        self._nn, self._F = nn, F

        class _M(nn.Module):
            def __init__(self2):
                super().__init__()
                self2.features = features
                self2.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(1024, 256), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(256, 1))
            def forward(self2, x):
                f = self2.features(x)
                z = F.relu(f)
                z = F.adaptive_avg_pool2d(z, 1).flatten(1)
                return self2.head(z)
        self._model = _M()

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def __getattr__(self, name):
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        return getattr(self._model, name)


class _VGGBinary:
    """Mirrors VGG16Pneumonia from train_vgg16_pneumonia.py."""
    def __init__(self, features):
        import torch.nn as nn

        class _M(nn.Module):
            def __init__(self2):
                super().__init__()
                self2.features = features
                self2.gap = nn.AdaptiveAvgPool2d(1)
                self2.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(256, 1))
            def forward(self2, x):
                x = self2.features(x)
                x = self2.gap(x).view(x.size(0), -1)
                return self2.head(x)
        self._model = _M()

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def __getattr__(self, name):
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        return getattr(self._model, name)
