"""Validates uploaded images before they reach inference — zero disk writes."""
from __future__ import annotations
import io
from PIL import Image
from fastapi import UploadFile, HTTPException
from .config import MAX_FILE_SIZE_MB

_MAGIC = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG":      "image/png",
    b"BM":           "image/bmp",
    b"RIFF":         "image/webp",
}


async def validate_and_decode(file: UploadFile) -> Image.Image:
    """Read upload bytes, validate magic & size, return PIL Image in RGB."""
    data = await file.read()

    if len(data) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_FILE_SIZE_MB} MB limit.")
    if len(data) < 16:
        raise HTTPException(400, "File too small to be a valid image.")

    recognised = any(data[: len(m)] == m for m in _MAGIC)
    if not recognised:
        raise HTTPException(400, "Unrecognised format. Upload JPEG, PNG, BMP or WebP.")

    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception:
        raise HTTPException(400, "Failed to decode image — file may be corrupt.")
