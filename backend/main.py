"""MedvisionAi — FastAPI backend entry-point."""
from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from .config import CORS_ORIGINS
from .input_guard import validate_and_decode
from .registry import ModelRegistry
from .schemas import (
    CatalogResponse,
    CatalogTask,
    HealthResponse,
    ModelInfo,
    PredictionResponse,
)

# ── Lifespan ─────────────────────────────────────────────────────
registry: ModelRegistry | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global registry
    print("🩺  MedvisionAi — starting up …")
    registry = ModelRegistry()
    yield
    print("🩺  MedvisionAi — shutting down …")
    registry = None


# ── App ──────────────────────────────────────────────────────────
app = FastAPI(
    title="MedvisionAi",
    version="1.0.0",
    description="Unified medical-imaging AI inference API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_registry() -> ModelRegistry:
    if registry is None:
        raise HTTPException(503, "Server is starting up — try again in a moment.")
    return registry


# ── Routes ───────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"service": "MedvisionAi", "version": "1.0.0", "docs": "/docs"}


@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    reg = _get_registry()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HealthResponse(
        status="ok",
        device=device,
        loaded_pipelines=reg.loaded_ids,
        gpu_name=gpu_name,
    )


@app.get("/api/v1/catalog", response_model=CatalogResponse)
async def catalog():
    reg = _get_registry()
    tasks: list[CatalogTask] = []
    for tid, m in reg.all_manifests().items():
        models = [
            ModelInfo(
                id=mdl["id"],
                display_name=mdl["display_name"],
                enabled=mdl.get("enabled", True),
                available=reg.model_available(tid, mdl["id"]),
            )
            for mdl in m.get("models", [])
        ]
        tasks.append(
            CatalogTask(
                task_id=tid,
                display_name=m["display_name"],
                description=m.get("description", ""),
                modality=m.get("modality", "unknown"),
                classes=m.get("classes", []),
                models=models,
                available=reg.is_available(tid),
                explainability=m.get("explainability", "none"),
            )
        )
    return CatalogResponse(tasks=tasks)


@app.post("/api/v1/predict/{task_id}", response_model=PredictionResponse)
async def predict_single(task_id: str, file: UploadFile = File(...)):
    reg = _get_registry()

    manifest = reg.get_manifest(task_id)
    if manifest is None:
        raise HTTPException(404, f"Unknown task: '{task_id}'")
    if not reg.is_available(task_id):
        raise HTTPException(
            503,
            f"Model weights for '{task_id}' are not available. "
            "Train the model and place weights in the artifacts/ directory.",
        )

    img = await validate_and_decode(file)
    pipeline = reg.acquire(task_id)
    result: PredictionResponse = await asyncio.to_thread(pipeline.run, img)
    return result


@app.post("/api/v1/predict/panel/{modality}")
async def predict_panel(modality: str, file: UploadFile = File(...)):
    """Run every available task that matches *modality* on a single upload."""
    reg = _get_registry()

    matching = [
        tid
        for tid, m in reg.all_manifests().items()
        if m.get("modality") == modality and reg.is_available(tid)
    ]
    if not matching:
        raise HTTPException(
            404, f"No available models for modality '{modality}'."
        )

    img = await validate_and_decode(file)

    async def _run(tid: str) -> PredictionResponse:
        pipe = reg.acquire(tid)
        return await asyncio.to_thread(pipe.run, img)

    results = await asyncio.gather(*[_run(t) for t in matching], return_exceptions=True)

    out: list[dict] = []
    for tid, r in zip(matching, results):
        if isinstance(r, Exception):
            out.append({"task_id": tid, "error": str(r)})
        else:
            out.append(r.model_dump())
    return out


# ── Global error handler ─────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    from fastapi.responses import JSONResponse

    code = getattr(exc, "status_code", 500)
    detail = getattr(exc, "detail", str(exc))
    return JSONResponse(
        status_code=code,
        content={
            "error": True,
            "detail": detail,
            "request_id": uuid.uuid4().hex[:12],
        },
    )


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
