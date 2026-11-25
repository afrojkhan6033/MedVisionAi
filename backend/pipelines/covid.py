"""COVID-19 adapter — Keras 3 (PyTorch backend) ResNet-50 + VGG-16 + Xception ensemble."""
from __future__ import annotations
import os

# Configure Keras to run natively via PyTorch backend (bypasses Windows DLL issue)
os.environ["KERAS_BACKEND"] = "torch"

from typing import Any
import numpy as np
import torch
from PIL import Image
import keras

from .base import BasePipeline


class CovidPipeline(BasePipeline):
    def __init__(self, manifest: dict) -> None:
        super().__init__(manifest)
        self._models: dict[str, Any] = {}
        self._model_sizes: dict[str, tuple[int, int]] = {}
        self._positive = manifest.get("positive_class", "COVID-19")
        self._cfg = manifest.get("input", {})
        self.device_name = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        default_size = (self._cfg.get("width", 64), self._cfg.get("height", 64))

        for mdl in self.manifest.get("models", []):
            if not mdl.get("enabled", True):
                continue
            wpath = self.artifact_dir / mdl["weights"]
            if not wpath.exists():
                print(f"[covid] Weights not found: {wpath}")
                continue
            try:
                m = keras.models.load_model(str(wpath))
                self._models[mdl["id"]] = m
                sz = mdl.get("input_size", default_size)
                self._model_sizes[mdl["id"]] = (int(sz[0]), int(sz[1]))
                print(f"[covid] Loaded {mdl['id']} (size={self._model_sizes[mdl['id']]})")
            except Exception as exc:
                print(f"[covid] Failed to load {mdl['id']}: {exc}")

        self.loaded = bool(self._models)

    def preprocess(self, img: Image.Image) -> Any:
        return {"rgb": img.convert("RGB")}

    def predict(self, inputs: Any) -> dict:
        rgb: Image.Image = inputs["rgb"]
        results: dict[str, np.ndarray] = {}
        for mid, model in self._models.items():
            size = self._model_sizes.get(mid, (64, 64))
            resized = rgb.resize(size, Image.Resampling.LANCZOS)
            arr = np.asarray(resized, dtype=np.float32) / 255.0
            tensor = np.expand_dims(arr, axis=0)
            preds = model.predict(tensor, verbose=0)
            if hasattr(preds, "numpy"):
                preds = preds.numpy()
            results[mid] = np.asarray(preds)[0]
        return {"per_model": results}

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

        votes: dict[str, int] = {c: 0 for c in self.classes}
        model_results = []
        avg_probs = np.zeros(len(self.classes))

        for mid, probs in per_model.items():
            pred_idx = int(np.argmax(probs))
            pred_label = self.classes[pred_idx]
            votes[pred_label] += 1
            avg_probs += probs
            dname = mid
            for mdl in self.manifest.get("models", []):
                if mdl["id"] == mid:
                    dname = mdl["display_name"]
            model_results.append({
                "name": dname,
                "prediction": pred_label,
                "confidence": float(probs[pred_idx]),
            })

        avg_probs /= len(per_model)
        final_pred = max(votes, key=lambda k: votes[k])
        final_idx = self.classes.index(final_pred)

        return {
            "prediction": final_pred,
            "predicted_idx": final_idx,
            "confidence": float(avg_probs[final_idx]),
            "is_abnormal": final_pred == self._positive,
            "probabilities": {c: float(avg_probs[i]) for i, c in enumerate(self.classes)},
            "model_results": model_results,
        }
