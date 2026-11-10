"""Background inpainting: backfill pixels hidden behind removed text/objects.

Engine resolution order for ``DECOMPOSER_INPAINT_ENGINE=auto``:
    LaMa (``simple-lama-inpainting``)  ->  OpenCV multi-scale Telea

Stable Diffusion inpainting (``diffusers``) is available with
``DECOMPOSER_INPAINT_ENGINE=sd`` but is never selected automatically.

Every engine output is composited back with a feathered mask so pixels outside
the hole are guaranteed to be untouched.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2
import numpy as np

from .common import dilate, logger, module_available, torch_device

MAX_FULLRES_HOLE_PX = 120_000  # above this, OpenCV inpaints on a pyramid level
LAMA_ONNX_PATH = Path(__file__).resolve().parent.parent / "models" / "lama_fp32.onnx"
LAMA_DOWNLOAD_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"


def _ensure_lama_model() -> bool:
    if LAMA_ONNX_PATH.exists():
        return True
    try:
        import urllib.request

        logger.info("LaMa model not found locally. Downloading from HuggingFace (%s)...", LAMA_DOWNLOAD_URL)
        LAMA_ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LAMA_DOWNLOAD_URL, str(LAMA_ONNX_PATH))
        logger.info("LaMa model downloaded successfully: %s", LAMA_ONNX_PATH)
        return True
    except Exception as exc:
        logger.warning("Auto-download of LaMa model failed (%s); will fall back to alternatives.", exc)
        return False


def _lama_available() -> bool:
    """Check if LaMa ONNX model or simple-lama is ready to run."""
    if (LAMA_ONNX_PATH.exists() or _ensure_lama_model()) and module_available("onnxruntime"):
        return True
    return module_available("simple_lama_inpainting")


@dataclass
class InpaintResult:
    image: np.ndarray  # uint8 RGB
    engine: str
    warnings: List[str] = field(default_factory=list)


class InpaintService:
    ENGINES = ("lama", "sd", "opencv")

    def __init__(self, engine: str = "auto", sd_model: str = "") -> None:
        self.preference = engine
        self.sd_model = sd_model
        self._lock = threading.Lock()
        self._backend = None
        self._failed: set = set()

    def available_engines(self) -> List[str]:
        out = []
        if _lama_available():
            out.append("lama")
        if module_available("diffusers") and module_available("torch"):
            out.append("sd")
        out.append("opencv")
        return out

    def _load(self, warnings: List[str]):
        with self._lock:
            if self._backend is not None:
                return self._backend
            if self.preference in self.ENGINES:
                order = [self.preference] + [e for e in ("lama", "opencv") if e != self.preference]
            else:
                order = ["lama", "opencv"]  # "sd" only on explicit request
            avail = self.available_engines()
            for name in [e for e in order if e in avail and e not in self._failed]:
                try:
                    if name == "lama":
                        if LAMA_ONNX_PATH.exists() and module_available("onnxruntime"):
                            import onnxruntime as ort

                            opts = ort.SessionOptions()
                            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                            opts.intra_op_num_threads = 4
                            model = ort.InferenceSession(str(LAMA_ONNX_PATH), opts, providers=["CPUExecutionProvider"])
                        else:
                            from simple_lama_inpainting import SimpleLama

                            model = SimpleLama()
                    elif name == "sd":
                        import torch
                        from diffusers import StableDiffusionInpaintPipeline

                        device = torch_device()
                        model = StableDiffusionInpaintPipeline.from_pretrained(
                            self.sd_model, torch_dtype=torch.float16 if device == "cuda" else torch.float32
                        ).to(device)
                    else:
                        model = None
                    self._backend = (name, model)
                    logger.info("Inpainting engine ready: %s", name)
                    return self._backend
                except Exception as exc:  # pragma: no cover
                    self._failed.add(name)
                    msg = f"Inpainting engine '{name}' failed to load ({exc!s}); falling back."
                    logger.warning(msg)
                    warnings.append(msg)
            self._backend = ("opencv", None)
            return self._backend

    def inpaint(self, rgb: np.ndarray, mask: np.ndarray) -> InpaintResult:
        warnings: List[str] = []
        mask = mask.astype(bool)
        if not mask.any():
            return InpaintResult(rgb.copy(), "none", warnings)
        name, model = self._load(warnings)
        try:
            if name == "lama":
                filled = _run_lama(model, rgb, mask)
            elif name == "sd":
                filled = _run_sd(model, rgb, mask)
            else:
                filled = _run_opencv(rgb, mask)
        except Exception as exc:  # pragma: no cover
            msg = f"Inpainting engine '{name}' failed at inference ({exc!s}); using OpenCV."
            logger.warning(msg)
            warnings.append(msg)
            name, filled = "opencv", _run_opencv(rgb, mask)
        return InpaintResult(_composite(rgb, filled, mask), name, warnings)


def _composite(original: np.ndarray, filled: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend ``filled`` into ``original`` only inside a feathered version of ``mask``."""
    grown = dilate(mask, 2).astype(np.float32)
    m = cv2.GaussianBlur(grown, (0, 0), 1.5)
    m[mask] = 1.0
    m = m[..., None]
    out = original.astype(np.float32) * (1.0 - m) + filled.astype(np.float32) * m
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _run_opencv(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    m8 = mask.astype(np.uint8) * 255
    hole = int(mask.sum())
    h, w = mask.shape
    if hole <= MAX_FULLRES_HOLE_PX:
        out = cv2.inpaint(bgr, m8, 5, cv2.INPAINT_TELEA)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    # Large holes: solve structure at low resolution, then refine the seams.
    scale = max(0.1, (MAX_FULLRES_HOLE_PX / hole) ** 0.5)
    sw, sh = max(8, int(w * scale)), max(8, int(h * scale))
    small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    small_m = (cv2.resize(m8, (sw, sh), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8) * 255
    small_m = cv2.dilate(small_m, np.ones((3, 3), np.uint8))
    coarse = cv2.inpaint(small, small_m, 5, cv2.INPAINT_TELEA)
    up = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    merged = bgr.copy()
    merged[mask] = up[mask]
    # Smooth the hole interior slightly to hide upsampling artefacts.
    blurred = cv2.GaussianBlur(merged, (0, 0), 1.0 / scale * 0.5)
    merged[mask] = blurred[mask]
    return cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)


def _run_lama(model, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if hasattr(model, "run"):  # ONNX Runtime InferenceSession
        h, w = rgb.shape[:2]
        side = 512
        img_512 = cv2.resize(rgb, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        img_tensor = np.transpose(img_512, (2, 0, 1))[np.newaxis, ...]
        mask_512 = (cv2.resize(mask.astype(np.uint8) * 255, (side, side), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
        mask_tensor = mask_512[np.newaxis, np.newaxis, ...]

        pred = model.run(None, {"image": img_tensor, "mask": mask_tensor})[0]
        out_512 = np.transpose(pred[0], (1, 2, 0))
        if out_512.max() <= 1.0:
            out_512 = out_512 * 255.0
        out_512 = np.clip(out_512, 0, 255).astype(np.uint8)

        out_full = cv2.resize(out_512, (w, h), interpolation=cv2.INTER_CUBIC)
        filled = rgb.copy()
        filled[mask] = out_full[mask]
        return filled

    # Fallback to simple-lama
    from PIL import Image

    out = model(Image.fromarray(rgb), Image.fromarray(mask.astype(np.uint8) * 255))
    out = np.asarray(out.convert("RGB"))
    if out.shape[:2] != rgb.shape[:2]:  # LaMa pads to a multiple of 8
        out = out[: rgb.shape[0], : rgb.shape[1]]
    return out


def _run_sd(pipe, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:  # pragma: no cover
    from PIL import Image

    h, w = rgb.shape[:2]
    side = 512
    img = Image.fromarray(rgb).resize((side, side), Image.LANCZOS)
    msk = Image.fromarray(dilate(mask, 4).astype(np.uint8) * 255).resize((side, side), Image.NEAREST)
    result = pipe(prompt="clean background, seamless, high quality", image=img, mask_image=msk).images[0]
    return np.asarray(result.convert("RGB").resize((w, h), Image.LANCZOS))

