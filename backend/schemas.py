"""Canonical response schemas — Pydantic v2."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class ModelResult(BaseModel):
    name: str
    prediction: str
    confidence: float = Field(ge=0, le=1)


class ConsensusInfo(BaseModel):
    agreed: bool
    strategy: str = "single"
    models: list[ModelResult] = []


class ExplainabilityResult(BaseModel):
    type: str = "gradcam"
    overlay_base64: Optional[str] = None


class PredictionResponse(BaseModel):
    request_id: str
    task_id: str
    display_name: str
    prediction: str
    confidence: float = Field(ge=0, le=1)
    is_abnormal: bool
    threshold_used: Optional[float] = None
    probabilities: dict[str, float]
    consensus: ConsensusInfo
    explainability: Optional[ExplainabilityResult] = None
    warnings: list[str] = []
    inference_ms: float
    device: str = "cpu"


class ModelInfo(BaseModel):
    id: str
    display_name: str
    enabled: bool
    available: bool = False


class CatalogTask(BaseModel):
    task_id: str
    display_name: str
    description: str = ""
    modality: str
    classes: list[str]
    models: list[ModelInfo]
    available: bool
    explainability: str = "none"


class ModalityInfo(BaseModel):
    id: str
    display_name: str
    icon: str


class CatalogResponse(BaseModel):
    tasks: list[CatalogTask]
    modalities: list[ModalityInfo] = [
        ModalityInfo(id="brain_mri", display_name="Brain MRI", icon="brain"),
        ModalityInfo(id="chest_scan", display_name="Chest Scan", icon="stethoscope"),
    ]


class HealthResponse(BaseModel):
    status: str = "ok"
    device: str = "cpu"
    loaded_pipelines: list[str] = []
    gpu_name: Optional[str] = None
