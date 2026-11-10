"""Shared fixtures: synthetic test artwork, in-process client and live-server client."""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

WIDTH, HEIGHT = 640, 420
CIRCLE = {"cx": 450, "cy": 250, "r": 95, "color": (220, 40, 40)}
TEXT = {"xy": (40, 40), "string": "HELLO WORLD", "size": 48, "color": (20, 30, 60)}


def _gradient(y: int) -> tuple:
    t = y / (HEIGHT - 1)
    top, bottom = np.array([120, 180, 235]), np.array([235, 245, 255])
    return tuple(int(v) for v in top * (1 - t) + bottom * t)


def make_artwork() -> Dict:
    """Gradient sky + red disc (object) + dark headline (text). Returns image + ground truth."""
    img = Image.new("RGB", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(img)
    for y in range(HEIGHT):
        draw.line([(0, y), (WIDTH, y)], fill=_gradient(y))
    c = CIRCLE
    disc_box = (c["cx"] - c["r"], c["cy"] - c["r"], c["cx"] + c["r"], c["cy"] + c["r"])
    draw.ellipse(disc_box, fill=c["color"])
    try:
        font = ImageFont.truetype("arial.ttf", TEXT["size"])
    except OSError:
        font = ImageFont.load_default(size=TEXT["size"])
    draw.text(TEXT["xy"], TEXT["string"], fill=TEXT["color"], font=font)
    text_box = draw.textbbox(TEXT["xy"], TEXT["string"], font=font)
    return {"image": img, "disc_box": disc_box, "text_box": text_box, "bg_at": _gradient}


@pytest.fixture(scope="session")
def artwork() -> Dict:
    return make_artwork()


@pytest.fixture(scope="session")
def png_bytes(artwork) -> bytes:
    buf = io.BytesIO()
    artwork["image"].save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient

    os.environ["DECOMPOSER_OUTPUT_DIR"] = str(tmp_path_factory.mktemp("outputs"))
    from app.main import create_app

    with TestClient(create_app(), base_url="http://testserver") as c:
        yield c


@pytest.fixture(scope="session")
def live_url() -> str:
    """Base URL of a running uvicorn server; tests using it are skipped if it is down."""
    import requests

    url = os.getenv("DECOMPOSER_TEST_URL", "http://127.0.0.1:8000").rstrip("/")
    try:
        requests.get(f"{url}/api/v1/health", timeout=3).raise_for_status()
    except Exception as exc:  # pragma: no cover - depends on environment
        pytest.skip(f"Live server not reachable at {url}: {exc}")
    return url

