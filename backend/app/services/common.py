"""Small helpers shared by the service modules."""
from __future__ import annotations

import importlib.util
import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("decomposer")


def module_available(name: str) -> bool:
    """True if ``name`` is importable, without actually importing it (cheap)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def torch_device() -> str:
    try:
        import torch  # noqa: WPS433 - optional heavy dependency

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover - torch not installed
        pass
    return "cpu"


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return (x0, y0, x1, y1) exclusive bbox of non-zero pixels, or None."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def rgb_to_hex(rgb) -> str:
    r, g, b = (int(np.clip(round(float(c)), 0, 255)) for c in rgb[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


def resize_to_max(img: np.ndarray, max_side: int, interpolation=cv2.INTER_AREA) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale >= 1.0:
        return img, 1.0
    out = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=interpolation)
    return out, scale

