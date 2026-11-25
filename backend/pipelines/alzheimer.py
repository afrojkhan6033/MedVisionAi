"""Alzheimer's 4-class adapter — PyTorch ResNet-50 + VGG-16 ensemble."""
from __future__ import annotations
from typing import Any, Optional
from pathlib import Path

import numpy as np
from PIL import Image

from .base import BasePipeline

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AlzheimerPipeline(BasePipeline):
    def __init__(self, manifest: dict) -> None:
        super().__init__(manifest)
        self._models: dict[str, Any] = {}
        self._temps: dict[str, float] = {}
        self._torch = None
        self._cfg = manifest.get("input", {})

    # ── load ─────────────────────────────────────────────────────
    def load(self) -> None:
        try:
            import torch, torchvision.models as tvm  # noqa: E401
            self._torch = torch
        except ImportError:
            print("[alzheimer] PyTorch not installed — adapter disabled")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = str(device)
        nc = len(self.classes)

        for mdl in self.manifest.get("models", []):
            if not mdl.get("enabled", True):
                continue
            wpath = self.artifact_dir / mdl["weights"]
            if not wpath.exists():
                print(f"[alzheimer] Weights not found: {wpath}")
                continue

            mid = mdl["id"]
            try:
                if mid == "resnet50":
                    import torch.nn as nn
                    m = tvm.resnet50(weights=None)
                    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, nc))
                elif mid == "vgg16":
                    import torch.nn as nn
                    m = tvm.vgg16(weights=None)
                    m.classifier[6] = nn.Linear(m.classifier[6].in_features, nc)
                else:
                    continue

                ck = torch.load(wpath, map_location=device, weights_only=False)
                state = ck.get("model", ck) if isinstance(ck, dict) else ck
                m.load_state_dict(state)
                for mod in m.modules():
                    if isinstance(mod, nn.ReLU):
                        mod.inplace = False
                m.to(device).eval()
                self._models[mid] = m
                print(f"[alzheimer] Loaded {mid} from {wpath}")
            except Exception as exc:
                print(f"[alzheimer] Failed to load {mid}: {exc}")

        cal_path = self.artifact_dir / "calibration.json"
        if cal_path.exists():
            try:
                import json
                cal = json.loads(cal_path.read_text(encoding="utf-8"))
                self._temps["resnet50"] = float(cal.get("resnet50_temperature", 1.0))
                self._temps["vgg16"] = float(cal.get("vgg16_temperature", 1.0))
            except Exception:
                pass

        self.loaded = bool(self._models)

    # ── preprocess ───────────────────────────────────────────────
    def preprocess(self, img: Image.Image) -> Any:
        torch = self._torch
        from torchvision import transforms as T

        size = (self._cfg.get("height", 224), self._cfg.get("width", 224))
        tfm = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        rgb = img.convert("L").convert("RGB")
        return tfm(rgb).unsqueeze(0)

    # ── predict ──────────────────────────────────────────────────
    def predict(self, inputs: Any) -> dict:
        torch = self._torch
        results: dict[str, np.ndarray] = {}
        if not self._models:
            return {"per_model": results}

        device = next(iter(self._models.values())).fc[1].weight.device if "resnet50" in self._models else \
                 next(iter(self._models.values())).classifier[6].weight.device
        x = inputs.to(device)

        with torch.no_grad():
            for mid, m in self._models.items():
                temp = self._temps.get(mid, 1.0)
                logits = m(x) / temp
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                results[mid] = probs
        return {"per_model": results}

    # ── postprocess ──────────────────────────────────────────────
    def postprocess(self, raw: dict) -> dict:
        per_model = raw.get("per_model", {})
        if not per_model:
            return {
                "prediction": "Unavailable",
                "confidence": 0.0,
                "is_abnormal": False,
                "probabilities": {c: 0.0 for c in self.classes},
                "model_results": [],
            }

        ensemble_w = self.manifest.get("ensemble", {}).get("weights", [])
        model_results = []
        weighted_probs = np.zeros(len(self.classes))

        for i, (mid, probs) in enumerate(per_model.items()):
            pred_idx = int(np.argmax(probs))
            w = ensemble_w[i] if i < len(ensemble_w) else 1.0
            weighted_probs += probs * w
            dname = mid
            for mdl in self.manifest.get("models", []):
                if mdl["id"] == mid:
                    dname = mdl["display_name"]
            model_results.append({
                "name": dname,
                "prediction": self.classes[pred_idx],
                "confidence": float(probs[pred_idx]),
            })

        if ensemble_w:
            weighted_probs /= sum(ensemble_w[: len(per_model)])
        else:
            weighted_probs /= len(per_model)

        pred_idx = int(np.argmax(weighted_probs))
        return {
            "prediction": self.classes[pred_idx],
            "predicted_idx": pred_idx,
            "confidence": float(weighted_probs[pred_idx]),
            "is_abnormal": pred_idx != 0,  # 0 = Non_Demented
            "probabilities": {c: float(weighted_probs[j]) for j, c in enumerate(self.classes)},
            "model_results": model_results,
        }

    # ── explain (Grad-CAM) ───────────────────────────────────────
    def explain(self, img: Image.Image, inputs: Any, target_class: int) -> Optional[str]:
        torch = self._torch
        if torch is None or "resnet50" not in self._models:
            return None

        model = self._models["resnet50"]
        target_layer = model.layer4[-1]
        activations, gradients = {}, {}

        def fwd(m, i, o):
            activations["v"] = o.detach()
        def bwd(m, gi, go):
            gradients["v"] = go[0].detach()

        fh = target_layer.register_forward_hook(fwd)
        bh = target_layer.register_full_backward_hook(bwd)

        try:
            device = next(model.parameters()).device
            x = inputs.to(device).requires_grad_(True)
            out = model(x)
            model.zero_grad()
            out[0, target_class].backward()

            w = gradients["v"].mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((w * activations["v"]).sum(dim=1))[0].cpu().numpy()
            cam = cam / (cam.max() + 1e-8)
            return self._cam_overlay(img, cam)
        except Exception:
            return None
        finally:
            fh.remove()
            bh.remove()
