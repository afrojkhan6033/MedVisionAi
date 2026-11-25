"""Abstract base for every disease-detection pipeline adapter."""
from __future__ import annotations
import base64
import io
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from ..schemas import (
    ConsensusInfo,
    ExplainabilityResult,
    ModelResult,
    PredictionResponse,
)


class BasePipeline(ABC):
    """
    Every adapter implements four hooks:
        load        → load weights into memory
        preprocess  → PIL Image → framework-ready tensor
        predict     → tensor → raw output dict
        postprocess → raw → (prediction, confidence, probabilities, consensus)
        explain     → (image, inputs) → optional base64 overlay
    The concrete ``run()`` method chains them and returns a canonical
    PredictionResponse.
    """

    def __init__(self, manifest: dict) -> None:
        self.manifest = manifest
        self.task_id: str = manifest["task_id"]
        self.display_name: str = manifest["display_name"]
        self.classes: list[str] = manifest["classes"]
        self.artifact_dir = Path(manifest["_dir"])
        self.loaded = False
        self.device_name = "cpu"
        self.available = self._check_weights()

    # ── helpers ──────────────────────────────────────────────────
    def _check_weights(self) -> bool:
        for mdl in self.manifest.get("models", []):
            if mdl.get("enabled", True):
                if (self.artifact_dir / mdl["weights"]).exists():
                    return True
        return False

    @staticmethod
    def _pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/{fmt.lower()};base64,{b64}"

    def _cam_overlay(self, img: Image.Image, cam: np.ndarray) -> str:
        """Blend a normalized [0,1] 2D CAM heatmap onto *img* as a Jet overlay."""
        cam_uint8 = np.uint8(np.clip(cam, 0.0, 1.0) * 255)
        cam_resized = np.asarray(
            Image.fromarray(cam_uint8).resize((img.width, img.height), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        try:
            import matplotlib
            cmap = matplotlib.colormaps["jet"]
            heatmap = np.uint8(cmap(cam_resized)[:, :, :3] * 255)
        except Exception:
            heatmap = np.stack([np.uint8(cam_resized * 255), np.zeros_like(cam_uint8), np.uint8((1 - cam_resized) * 255)], axis=-1)
        orig = np.asarray(img.convert("RGB"), dtype=np.float32)
        blended = np.uint8(np.clip(orig * 0.55 + heatmap * 0.45, 0, 255))
        return self._pil_to_base64(Image.fromarray(blended))

    # ── abstract hooks ───────────────────────────────────────────
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def preprocess(self, img: Image.Image) -> Any: ...

    @abstractmethod
    def predict(self, inputs: Any) -> dict: ...

    @abstractmethod
    def postprocess(self, raw: dict) -> dict: ...

    def explain(self, img: Image.Image, inputs: Any, target_class: int) -> Optional[str]:
        """Return a base64 overlay or None.  Override in subclasses."""
        return None

    # ── canonical run ────────────────────────────────────────────
    def run(self, img: Image.Image) -> PredictionResponse:
        t0 = time.perf_counter()
        warnings: list[str] = []

        if not self.available:
            warnings.append("Model weights not found — predictions may be unavailable.")

        inputs = self.preprocess(img)
        raw = self.predict(inputs)
        post = self.postprocess(raw)

        # explainability
        xai: Optional[ExplainabilityResult] = None
        try:
            overlay = self.explain(img, inputs, post.get("predicted_idx", 0))
            if overlay:
                xtype = self.manifest.get("explainability", "gradcam")
                xai = ExplainabilityResult(type=xtype, overlay_base64=overlay)
        except Exception as exc:
            warnings.append(f"Explainability skipped: {exc}")

        elapsed = (time.perf_counter() - t0) * 1000

        # build consensus
        model_results = [
            ModelResult(name=m["name"], prediction=m["prediction"], confidence=m["confidence"])
            for m in post.get("model_results", [])
        ]
        strategy = self.manifest.get("ensemble", {}).get("strategy", "single")
        agreed = len(set(m.prediction for m in model_results)) <= 1 if model_results else True
        consensus = ConsensusInfo(agreed=agreed, strategy=strategy, models=model_results)

        return PredictionResponse(
            request_id=uuid.uuid4().hex[:12],
            task_id=self.task_id,
            display_name=self.display_name,
            prediction=post["prediction"],
            confidence=round(post["confidence"], 4),
            is_abnormal=post.get("is_abnormal", False),
            threshold_used=post.get("threshold"),
            probabilities={k: round(v, 4) for k, v in post["probabilities"].items()},
            consensus=consensus,
            explainability=xai,
            warnings=warnings,
            inference_ms=round(elapsed, 1),
            device=self.device_name,
        )
