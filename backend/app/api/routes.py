"""HTTP routes for the decomposition service (mounted under ``/api/v1``)."""
from __future__ import annotations

import base64
import binascii
import json
import re
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from .. import __version__
from ..schemas import DecomposeJSONRequest, DecomposeOptions, DecomposeResponse, HealthResponse
from ..services.pipeline import ImageDecodeError, ImageTooLargeError

router = APIRouter(prefix="/api/v1", tags=["decomposition"])

_OPTION_FIELDS = set(DecomposeOptions.model_fields)
_DATA_URL = re.compile(r"^data:[\w/+.-]+;base64,", re.IGNORECASE)

_DECOMPOSE_OPENAPI: Dict[str, Any] = {
    "requestBody": {
        "required": True,
        "description": (
            "Three equivalent encodings are accepted:\n"
            "* `multipart/form-data` - `file` part + optional `options` JSON part (or individual option fields).\n"
            "* `application/json` - `{\"image_base64\": \"...\", \"options\": {...}}`.\n"
            "* raw `image/png` (or `image/jpeg`, `application/octet-stream`) body - options as query params.\n"
            "  This is what the Photoshop UXP plugin uses."
        ),
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                        "file": {"type": "string", "format": "binary"},
                        "options": {"type": "string", "description": "JSON-encoded DecomposeOptions"},
                    },
                }
            },
            "application/json": {"schema": {"$ref": "#/components/schemas/DecomposeJSONRequest"}},
            "image/png": {"schema": {"type": "string", "format": "binary"}},
        },
    }
}


def _parse_bool_options(source: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in source.items():
        if k in _OPTION_FIELDS:
            if isinstance(v, str):
                v_low = v.strip().lower()
                if v_low in ("true", "1", "yes"):
                    out[k] = True
                elif v_low in ("false", "0", "no"):
                    out[k] = False
                else:
                    out[k] = v
            else:
                out[k] = v
    return out


def _options_from(source: Dict[str, Any]) -> DecomposeOptions:
    try:
        return DecomposeOptions(**_parse_bool_options(source))
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc


def _decode_b64(value: str) -> bytes:
    value = _DATA_URL.sub("", value.strip())
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="image_base64 is not valid base64.") from exc


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    pipeline = request.app.state.pipeline
    s = pipeline.settings
    return HealthResponse(
        version=__version__,
        engines_available=pipeline.engines_available(),
        engines_configured={"ocr": s.ocr_engine, "segmentation": s.segmentation_engine, "inpainting": s.inpaint_engine},
    )


@router.post(
    "/decompose-png",
    response_model=DecomposeResponse,
    openapi_extra=_DECOMPOSE_OPENAPI,
    responses={400: {"description": "Bad image"}, 413: {"description": "Too large"}, 415: {"description": "Bad type"}},
)
async def decompose_png(request: Request) -> DecomposeResponse:
    pipeline = request.app.state.pipeline
    max_bytes = pipeline.settings.max_upload_mb * 1024 * 1024
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Payload exceeds {pipeline.settings.max_upload_mb} MB.")

    ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
    query = dict(request.query_params)

    if ctype == "multipart/form-data":
        form = await request.form()
        upload = form.get("file") or form.get("image")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(status_code=422, detail="Multipart body must contain a 'file' part.")
        data = await upload.read()
        opts: Dict[str, Any] = dict(query)
        if form.get("options"):
            try:
                opts.update(json.loads(str(form.get("options"))))
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=422, detail="'options' part must be valid JSON.") from exc
        opts.update({k: v for k, v in form.items() if k in _OPTION_FIELDS})
        options = _options_from(opts)
    elif ctype == "application/json":
        try:
            payload = DecomposeJSONRequest.model_validate_json(await request.body())
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc
        data = _decode_b64(payload.image_base64)
        options = payload.options
    elif ctype.startswith("image/") or ctype == "application/octet-stream":
        data = await request.body()
        options = _options_from(query)
    else:
        raise HTTPException(
            status_code=415,
            detail="Use multipart/form-data, application/json (base64) or a raw image/* body.",
        )

    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Payload exceeds {pipeline.settings.max_upload_mb} MB.")

    try:
        return await run_in_threadpool(pipeline.run, data, options, str(request.base_url))
    except ImageDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImageTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request) -> Dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{16}", job_id):
        raise HTTPException(status_code=404, detail="Unknown job.")
    path = request.app.state.pipeline.settings.output_dir / job_id / "manifest.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Unknown job.")
    return json.loads(path.read_text(encoding="utf-8"))

