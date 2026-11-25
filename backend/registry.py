"""Thread-safe LRU Model Registry — loads / caches / evicts pipeline adapters."""
from __future__ import annotations
import gc
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from .config import ARTIFACTS_DIR, MAX_LOADED_PIPELINES
from .pipelines.base import BasePipeline


class ModelRegistry:
    """Discovers manifest files at startup and lazily loads adapters on demand."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, BasePipeline] = OrderedDict()
        self._manifests: dict[str, dict] = {}
        self._scan_manifests()

    # ── discovery ────────────────────────────────────────────────
    def _scan_manifests(self) -> None:
        if not ARTIFACTS_DIR.exists():
            print("[registry] artifacts/ directory not found — no models available")
            return
        for mf in sorted(ARTIFACTS_DIR.glob("*/manifest.json")):
            try:
                data = json.loads(mf.read_text(encoding="utf-8"))
                data["_dir"] = str(mf.parent)
                self._manifests[data["task_id"]] = data
                print(f"[registry] Found manifest: {data['task_id']} ({data['display_name']})")
            except Exception as exc:
                print(f"[registry] Skipping {mf}: {exc}")

    # ── public queries ───────────────────────────────────────────
    @property
    def task_ids(self) -> list[str]:
        return list(self._manifests.keys())

    def get_manifest(self, task_id: str) -> Optional[dict]:
        return self._manifests.get(task_id)

    def all_manifests(self) -> dict[str, dict]:
        return dict(self._manifests)

    def is_available(self, task_id: str) -> bool:
        """True when at least one enabled model has its weight file on disk."""
        m = self._manifests.get(task_id)
        if not m:
            return False
        d = Path(m["_dir"])
        return any(
            (d / mdl["weights"]).exists()
            for mdl in m.get("models", [])
            if mdl.get("enabled", True)
        )

    def model_available(self, task_id: str, model_id: str) -> bool:
        m = self._manifests.get(task_id)
        if not m:
            return False
        d = Path(m["_dir"])
        for mdl in m.get("models", []):
            if mdl["id"] == model_id:
                return (d / mdl["weights"]).exists()
        return False

    # ── acquire / cache ──────────────────────────────────────────
    def acquire(self, task_id: str) -> BasePipeline:
        with self._lock:
            if task_id in self._cache:
                self._cache.move_to_end(task_id)
                return self._cache[task_id]

            while len(self._cache) >= MAX_LOADED_PIPELINES:
                evict_id, evict_pipe = self._cache.popitem(last=False)
                print(f"[registry] Evicting '{evict_id}'")
                del evict_pipe
                gc.collect()

            manifest = self._manifests.get(task_id)
            if not manifest:
                raise ValueError(f"Unknown task: '{task_id}'")

            pipeline = self._create_pipeline(task_id, manifest)
            pipeline.load()
            self._cache[task_id] = pipeline
            print(f"[registry] Loaded '{task_id}' ({len(self._cache)}/{MAX_LOADED_PIPELINES} slots)")
            return pipeline

    @property
    def loaded_ids(self) -> list[str]:
        return list(self._cache.keys())

    # ── factory ──────────────────────────────────────────────────
    @staticmethod
    def _create_pipeline(task_id: str, manifest: dict) -> BasePipeline:
        from .pipelines.alzheimer import AlzheimerPipeline
        from .pipelines.pneumonia import PneumoniaPipeline
        from .pipelines.brain_tumor import BrainTumorPipeline
        from .pipelines.covid import CovidPipeline

        factories: dict[str, type[BasePipeline]] = {
            "alzheimer": AlzheimerPipeline,
            "pneumonia": PneumoniaPipeline,
            "brain_tumor": BrainTumorPipeline,
            "covid": CovidPipeline,
        }
        cls = factories.get(task_id)
        if cls is None:
            raise ValueError(f"No adapter for task '{task_id}'")
        return cls(manifest)
