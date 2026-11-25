"""Brain tumor binary adapter — PyTorch native port of the 5-layer CNN architecture."""
from __future__ import annotations
from typing import Any, Optional
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .base import BasePipeline


class _BrainTumorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = nn.ZeroPad2d(2)
        self.conv = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=0, bias=True)
        self.bn = nn.BatchNorm2d(32, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU()
        self.pool0 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.fc = nn.Linear(6272, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool0(x)
        x = self.pool1(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, 6272)
        x = self.fc(x)
        return self.sig(x)


class BrainTumorPipeline(BasePipeline):
    def __init__(self, manifest: dict) -> None:
        super().__init__(manifest)
        self._model: Optional[_BrainTumorCNN] = None
        self._positive = manifest.get("positive_class", "Tumor")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = str(self.device)

    def load(self) -> None:
        wpath: Optional[Path] = None
        for mdl in self.manifest.get("models", []):
            if mdl.get("enabled", True):
                p = self.artifact_dir / mdl["weights"]
                if p.exists():
                    wpath = p
                    break
        if wpath is None:
            print("[brain_tumor] No weight file found")
            return

        try:
            m = _BrainTumorCNN()
            with h5py.File(str(wpath), "r") as f:
                cw = f["model_weights"]["conv0"]["conv0"]["kernel:0"][:]
                cb = f["model_weights"]["conv0"]["conv0"]["bias:0"][:]
                m.conv.weight.data = torch.from_numpy(cw.transpose(3, 2, 0, 1))
                m.conv.bias.data = torch.from_numpy(cb)

                gamma = f["model_weights"]["bn0"]["bn0"]["gamma:0"][:]
                beta = f["model_weights"]["bn0"]["bn0"]["beta:0"][:]
                mean = f["model_weights"]["bn0"]["bn0"]["moving_mean:0"][:]
                var = f["model_weights"]["bn0"]["bn0"]["moving_variance:0"][:]
                m.bn.weight.data = torch.from_numpy(gamma)
                m.bn.bias.data = torch.from_numpy(beta)
                m.bn.running_mean.data = torch.from_numpy(mean)
                m.bn.running_var.data = torch.from_numpy(var)

                fc_w = f["model_weights"]["fc"]["fc"]["kernel:0"][:]
                fc_b = f["model_weights"]["fc"]["fc"]["bias:0"][:]
                m.fc.weight.data = torch.from_numpy(fc_w.T)
                m.fc.bias.data = torch.from_numpy(fc_b)

            m.to(self.device).eval()
            self._model = m
            self.loaded = True
            print(f"[brain_tumor] Loaded model successfully onto {self.device}")
        except Exception as exc:
            print(f"[brain_tumor] Load failed: {exc}")

    # ── Pure Pillow/NumPy contour & crop (no cv2 required) ───────
    @staticmethod
    def _find_brain_crop(img_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
        gray = np.mean(img_rgb, axis=2)
        mask = gray > 45
        if not np.any(mask):
            return img_rgb, None
        ys, xs = np.where(mask)
        y_min, y_max = int(np.min(ys)), int(np.max(ys))
        x_min, x_max = int(np.min(xs)), int(np.max(xs))
        cropped = img_rgb[y_min : y_max + 1, x_min : x_max + 1]
        return cropped, (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)

    def preprocess(self, img: Image.Image) -> Any:
        arr = np.array(img.convert("RGB"))
        cropped, bbox = self._find_brain_crop(arr)
        target_w = self.manifest["input"]["width"]
        target_h = self.manifest["input"]["height"]
        cropped_pil = Image.fromarray(cropped).resize((target_w, target_h), Image.Resampling.BILINEAR)
        arr_norm = np.asarray(cropped_pil, dtype=np.float32) / 255.0
        # (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(arr_norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return {"tensor": tensor, "bbox": bbox, "original": arr}

    def predict(self, inputs: Any) -> dict:
        if self._model is None:
            return {"prob": 0.5}
        with torch.no_grad():
            prob = float(self._model(inputs["tensor"]).item())
        return {"prob": prob, "bbox": inputs.get("bbox"), "original": inputs.get("original")}

    def postprocess(self, raw: dict) -> dict:
        prob = raw.get("prob", 0.5)
        has_tumor = prob > 0.5
        pred_label = self._positive if has_tumor else self.classes[0]
        conf = prob if has_tumor else 1.0 - prob
        dname = "Custom CNN"
        for mdl in self.manifest.get("models", []):
            if mdl.get("enabled", True):
                dname = mdl.get("display_name", dname)
        return {
            "prediction": pred_label,
            "predicted_idx": 1 if has_tumor else 0,
            "confidence": float(conf),
            "is_abnormal": has_tumor,
            "threshold": 0.5,
            "probabilities": {self.classes[0]: float(1.0 - prob), self.classes[1]: float(prob)},
            "model_results": [{"name": dname, "prediction": pred_label, "confidence": float(conf)}],
            "_bbox": raw.get("bbox"),
            "_original": raw.get("original"),
        }

    def explain(self, img: Image.Image, inputs: Any, target_class: int) -> Optional[str]:
        bbox = inputs.get("bbox")
        if bbox is None:
            return None
        try:
            x, y, w, h = bbox
            arr = np.array(img.convert("RGB"))
            overlay = arr.copy()
            # Draw green bounding box (thickness 3)
            overlay[max(0, y - 2) : min(arr.shape[0], y + 2), x : x + w] = [16, 185, 129]
            overlay[max(0, y + h - 2) : min(arr.shape[0], y + h + 2), x : x + w] = [16, 185, 129]
            overlay[y : y + h, max(0, x - 2) : min(arr.shape[1], x + 2)] = [16, 185, 129]
            overlay[y : y + h, max(0, x + w - 2) : min(arr.shape[1], x + w + 2)] = [16, 185, 129]
            return self._pil_to_base64(Image.fromarray(overlay))
        except Exception:
            return None
