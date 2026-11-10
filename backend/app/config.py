"""Runtime configuration, driven entirely by environment variables.

Every heavy AI engine is optional. With the default ``auto`` settings the
service picks the best engine that is actually installed and silently falls
back to fast classical-CV implementations, so the API runs out of the box.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Where per-job assets (PNG layers, PSD, manifest) are written + served from.
    output_dir: Path = field(
        default_factory=lambda: Path(_env("DECOMPOSER_OUTPUT_DIR", str(BACKEND_ROOT / "outputs")))
    )
    # Engine selection: easyocr | tesseract | paddle | heuristic
    ocr_engine: str = field(default_factory=lambda: _env("DECOMPOSER_OCR_ENGINE", "easyocr").lower())
    ocr_lang: str = field(default_factory=lambda: _env("DECOMPOSER_OCR_LANG", "hi,en"))
    # auto | sam2 | rembg | classical
    segmentation_engine: str = field(
        default_factory=lambda: _env("DECOMPOSER_SEGMENTATION_ENGINE", "auto").lower()
    )
    sam2_checkpoint: Optional[str] = field(default_factory=lambda: os.getenv("SAM2_CHECKPOINT"))
    sam2_config: str = field(
        default_factory=lambda: _env("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_s.yaml")
    )
    # auto | lama | sd | opencv   ("sd" is never chosen automatically - it is very heavy)
    inpaint_engine: str = field(default_factory=lambda: _env("DECOMPOSER_INPAINT_ENGINE", "auto").lower())
    sd_inpaint_model: str = field(
        default_factory=lambda: _env("SD_INPAINT_MODEL", "stabilityai/stable-diffusion-2-inpainting")
    )
    max_image_side: int = field(default_factory=lambda: _env_int("DECOMPOSER_MAX_IMAGE_SIDE", 8192))
    max_upload_mb: int = field(default_factory=lambda: _env_int("DECOMPOSER_MAX_UPLOAD_MB", 64))
    # If set, asset URLs are built from this instead of the incoming request URL.
    public_base_url: Optional[str] = field(default_factory=lambda: os.getenv("DECOMPOSER_PUBLIC_BASE_URL"))
    job_ttl_hours: int = field(default_factory=lambda: _env_int("DECOMPOSER_JOB_TTL_HOURS", 24))


def get_settings() -> Settings:
    return Settings()

