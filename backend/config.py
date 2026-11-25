"""MedvisionAi — global backend configuration."""
from __future__ import annotations
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MAX_LOADED_PIPELINES: int = 4
MAX_FILE_SIZE_MB: int = 10
CORS_ORIGINS: list[str] = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
