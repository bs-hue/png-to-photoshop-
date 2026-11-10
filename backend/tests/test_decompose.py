"""Contract + behaviour tests for ``POST /api/v1/decompose-png``.

Two groups:
  * in-process tests via FastAPI's TestClient (always run);
  * live tests against the uvicorn dev server on http://127.0.0.1:8000
    (marked ``live``; skipped automatically when the server is not running).
"""
from __future__ import annotations

import base64
import io
import json
from typing import Dict, List

import numpy as np
import pytest
from PIL import Image

from app.schemas import DecomposeResponse, LayerTreeNode, LayerType

RAW_HEADERS = {"Content-Type": "image/png"}


# ---------------------------------------------------------------- helpers
def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union else 0.0


def box(layer: Dict):
    b = layer["bounds"]
    return b["left"], b["top"], b["left"] + b["width"], b["top"] + b["height"]


def tree_leaf_ids(node: LayerTreeNode) -> List[str]:
    out = [node.layer_id] if node.layer_id else []
    for child in node.children:
        out += tree_leaf_ids(child)
    return out


def hex_to_rgb(h: str):
    return tuple(int(h[i : i + 2], 16) for i in (1, 3, 5))


@pytest.fixture(scope="module")
def full_result(client, png_bytes) -> Dict:
    r = client.post("/api/v1/decompose-png", content=png_bytes, headers=RAW_HEADERS)
    assert r.status_code == 200, r.text
    return r.json()


# ----------------------------------------------------------- availability
def test_root_and_health(client):
    root = client.get("/")
    # Root now serves the web UI (HTML) or falls back to JSON
    assert root.status_code == 200
    ct = root.headers.get("content-type", "")
    if "text/html" in ct:
        assert b"PNG" in root.content or b"decompos" in root.content.lower()
    else:
        assert root.json()["service"] == "png-decomposer"
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    for stage in ("ocr", "segmentation", "inpainting", "psd"):
        assert body["engines_available"][stage], stage
    assert "heuristic" in body["engines_available"]["ocr"]
    assert "classical" in body["engines_available"]["segmentation"]
    assert "opencv" in body["engines_available"]["inpainting"]


def test_openapi_documents_endpoint(client):
    spec = client.get("/openapi.json").json()
    op = spec["paths"]["/api/v1/decompose-png"]["post"]
    assert {"multipart/form-data", "application/json", "image/png"} <= set(op["requestBody"]["content"])
    assert "DecomposeResponse" in json.dumps(op["responses"]["200"])


# ------------------------------------------------------------- contract
def test_response_schema_and_layer_structure(full_result, artwork):
    resp = DecomposeResponse.model_validate(full_result)  # strict schema validation
    assert resp.status == "completed"
    assert (resp.width, resp.height) == artwork["image"].size
    assert [l.z_index for l in resp.layers] == list(range(len(resp.layers)))
    assert len({l.id for l in resp.layers}) == len(resp.layers)

    types = [l.type for l in resp.layers]
    assert types[0] == LayerType.ORIGINAL and resp.layers[0].visible is False
    assert types[1] == LayerType.BACKGROUND
    assert LayerType.OBJECT in types and LayerType.TEXT in types
    # bottom -> top: background < objects < text
    first_text = types.index(LayerType.TEXT)
    assert all(t != LayerType.OBJECT for t in types[first_text:])

    # Tree references every layer exactly once, in the same order.
    assert resp.layer_tree.type == LayerType.DOCUMENT
    assert tree_leaf_ids(resp.layer_tree) == [l.id for l in resp.layers]
    groups = {c.id: c for c in resp.layer_tree.children if c.type == LayerType.GROUP}
    assert {"group_foreground", "group_text"} <= set(groups)
    for layer in resp.layers:
        if layer.type in (LayerType.OBJECT, LayerType.TEXT):
            assert layer.parent_id in groups
        b = layer.bounds
        assert b.left >= 0 and b.top >= 0 and b.right <= resp.width and b.bottom <= resp.height
        assert b.width > 0 and b.height > 0
        assert layer.asset_url and layer.asset_url.startswith("http://testserver/assets/")

    for layer in resp.layers:
        if layer.type == LayerType.TEXT:
            assert layer.text is not None
            assert layer.text.color_hex.startswith("#") and len(layer.text.color_hex) == 7
            assert layer.text.font_size_px > 0
            assert layer.text.alignment.value in ("left", "center", "right")
    assert set(resp.timings_ms) >= {"decode", "ocr", "segmentation", "inpainting", "psd", "total"}
    assert resp.engines.psd in ("psd-tools", "ag-psd") and resp.psd_url


def test_assets_are_downloadable_and_match_bounds(client, full_result):
    for layer in full_result["layers"]:
        r = client.get(layer["asset_url"])
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        img = Image.open(io.BytesIO(r.content))
        assert img.mode == "RGBA"
        assert img.size == (layer["bounds"]["width"], layer["bounds"]["height"])
        if layer["type"] in ("object", "text"):
            alpha = np.asarray(img)[..., 3]
            assert alpha.min() == 0 and alpha.max() > 200, "cut-outs must be transparent PNGs"
            # Tight crop: every border row/column touches some visible pixel.
            assert alpha[0].any() and alpha[-1].any() and alpha[:, 0].any() and alpha[:, -1].any()


def test_psd_is_valid_and_layered(client, full_result):
    from psd_tools import PSDImage

    r = client.get(full_result["psd_url"])
    assert r.status_code == 200
    psd = PSDImage.open(io.BytesIO(r.content))
    assert psd.size == (full_result["width"], full_result["height"])
    pixel_layers = [l for l in psd.descendants() if l.kind in ("pixel", "type")]
    groups = [l.name for l in psd.descendants() if l.is_group()]
    assert len(pixel_layers) == len(full_result["layers"])
    assert {"Foreground Elements", "Text Group"} <= set(groups)
    names = {l.name for l in pixel_layers}
    assert {l["name"] for l in full_result["layers"]} == names


def test_job_manifest_endpoint(client, full_result):
    r = client.get(f"/api/v1/jobs/{full_result['job_id']}")
    assert r.status_code == 200
    assert r.json()["layers"] == [dict(l, asset_base64=None) for l in full_result["layers"]]
    assert client.get("/api/v1/jobs/deadbeefdeadbeef").status_code == 404
    assert client.get("/api/v1/jobs/..%2F..%2Fetc").status_code == 404


# ------------------------------------------------------------ behaviour
def test_object_isolated_where_the_disc_is(full_result, artwork):
    objects = [l for l in full_result["layers"] if l["type"] == "object"]
    best = max(iou(box(o), artwork["disc_box"]) for o in objects)
    assert best > 0.8, f"disc not isolated (best IoU {best:.2f})"


def test_text_localised_with_style(full_result, artwork):
    texts = [l for l in full_result["layers"] if l["type"] == "text"]
    best = max(texts, key=lambda t: iou(box(t), artwork["text_box"]))
    assert iou(box(best), artwork["text_box"]) > 0.6
    color = hex_to_rgb(best["text"]["color_hex"])
    assert max(abs(a - b) for a, b in zip(color, (20, 30, 60))) < 40, best["text"]["color_hex"]
    assert 0.5 * 48 <= best["text"]["font_size_px"] <= 1.6 * 48
    assert best["text"]["alignment"] == "left"


def test_background_is_inpainted(client, full_result, artwork):
    bg_layer = next(l for l in full_result["layers"] if l["type"] == "background")
    assert full_result["engines"]["inpainting"] in ("opencv", "lama", "sd")
    bg = np.asarray(Image.open(io.BytesIO(client.get(bg_layer["asset_url"]).content)).convert("RGB")).astype(int)
    x0, y0, x1, y1 = artwork["disc_box"]
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    expected = np.array(artwork["bg_at"](cy))
    assert np.abs(bg[cy, cx] - expected).max() < 45, "red disc must be removed from the background"
    tx0, ty0, tx1, ty1 = artwork["text_box"]
    patch = bg[ty0:ty1, tx0:tx1]
    assert patch.mean(axis=2).min() > 90, "dark text must be removed from the background"


def test_options_disable_every_stage(client, png_bytes):
    r = client.post(
        "/api/v1/decompose-png",
        params={"extract_text": "false", "isolate_objects": "false", "inpaint_background": "false",
                "extract_design": "false", "build_psd": "false", "include_original": "false"},
        content=png_bytes,
        headers=RAW_HEADERS,
    )
    assert r.status_code == 200, r.text
    body = DecomposeResponse.model_validate(r.json())
    assert [l.type for l in body.layers] == [LayerType.BACKGROUND]
    assert body.engines.model_dump() == {"ocr": "disabled", "segmentation": "disabled",
                                         "inpainting": "disabled", "design": "disabled", "psd": "disabled"}
    assert body.psd_url is None
    bg = client.get(body.layers[0].asset_url).content
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(bg)).convert("RGB")),
                          np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB")))


# ------------------------------------------------------- input encodings
def test_multipart_upload(client, png_bytes):
    r = client.post(
        "/api/v1/decompose-png",
        files={"file": ("art.png", png_bytes, "image/png")},
        data={"options": json.dumps({"extract_text": False, "build_psd": False})},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["options"]["extract_text"] is False
    assert not any(l["type"] == "text" for l in body["layers"])


def test_json_base64_with_inline_assets(client, png_bytes):
    payload = {"image_base64": "data:image/png;base64," + base64.b64encode(png_bytes).decode(),
               "options": {"include_base64": True, "build_psd": False}}
    r = client.post("/api/v1/decompose-png", json=payload)
    assert r.status_code == 200, r.text
    for layer in r.json()["layers"]:
        raw = base64.b64decode(layer["asset_base64"])
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"
        assert raw == client.get(layer["asset_url"]).content


def test_rgba_input_is_flattened(client):
    img = Image.new("RGBA", (64, 48), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    r = client.post("/api/v1/decompose-png", content=buf.getvalue(), headers=RAW_HEADERS)
    assert r.status_code == 200
    assert (r.json()["width"], r.json()["height"]) == (64, 48)


# ------------------------------------------------------------- errors
@pytest.mark.parametrize(
    "kwargs, status",
    [
        ({"content": b"hello", "headers": {"Content-Type": "text/plain"}}, 415),
        ({"content": b"not an image at all", "headers": RAW_HEADERS}, 400),
        ({"content": b"", "headers": RAW_HEADERS}, 400),
        ({"json": {"image_base64": "%%%not-base64%%%"}}, 400),
        ({"json": {"options": {}}}, 422),
        ({"json": {"image_base64": "aGk=", "options": {"max_objects": -1}}}, 422),
        ({"files": {"wrong": ("a.png", b"x", "image/png")}}, 422),
    ],
)
def test_error_contracts(client, kwargs, status):
    r = client.post("/api/v1/decompose-png", **kwargs)
    assert r.status_code == status, r.text
    assert "detail" in r.json()


def test_invalid_query_option(client, png_bytes):
    r = client.post("/api/v1/decompose-png", params={"max_objects": "lots"}, content=png_bytes, headers=RAW_HEADERS)
    assert r.status_code == 422


def test_design_spec_extraction(full_result):
    assert "design" in full_result
    design = full_result["design"]
    assert design is not None
    assert len(design["palette"]) >= 2
    for swatch in design["palette"]:
        assert swatch["hex"].startswith("#") and len(swatch["hex"]) == 7
        assert len(swatch["rgb"]) == 3
        assert 0.0 <= swatch["coverage"] <= 1.0
        assert swatch["role"] in ("dominant", "accent", "background", "neutral")
    assert "design_tokens" in design
    assert "canvas_width_px" in design["design_tokens"]
    assert full_result["engines"]["design"] == "opencv-classical"


# ------------------------------------------------- live uvicorn server
@pytest.mark.live
def test_live_server_health(live_url):
    import requests

    r = requests.get(f"{live_url}/api/v1/health", timeout=10)
    assert r.status_code == 200 and r.json()["status"] == "ok"


@pytest.mark.live
def test_live_server_decompose_roundtrip(live_url, png_bytes):
    import requests

    r = requests.post(f"{live_url}/api/v1/decompose-png", data=png_bytes, headers=RAW_HEADERS, timeout=300)
    assert r.status_code == 200, r.text
    body = DecomposeResponse.model_validate(r.json())
    assert body.layers and body.psd_url and body.psd_url.startswith(live_url)
    for layer in body.layers:
        asset = requests.get(layer.asset_url, timeout=30)
        assert asset.status_code == 200 and asset.content[:4] == b"\x89PNG"
    assert requests.get(body.psd_url, timeout=30).content[:4] == b"8BPS"

