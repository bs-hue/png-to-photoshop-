"""FastAPI application entry point.

Run with:
    uvicorn app.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api.routes import router
from .config import get_settings
from .services.pipeline import DecompositionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app() -> FastAPI:
    settings = get_settings()
    pipeline = DecompositionPipeline(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        removed = pipeline.cleanup_old_jobs()
        if removed:
            logging.getLogger("decomposer").info("Removed %d expired job folders", removed)
        yield

    app = FastAPI(
        title="PNG -> Photoshop Layers Decomposer",
        version=__version__,
        description="Decomposes flat raster images into text, object and background layers.",
        lifespan=lifespan,
    )
    app.state.pipeline = pipeline
    # UXP does not enforce CORS, but browsers / tooling hitting the API do.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(router)
    app.mount("/assets", StaticFiles(directory=str(settings.output_dir)), name="assets")
    # Serve the minimal monochrome dashboard we added
    dashboard_dir = Path(__file__).parent / "dashboard"
    app.mount("/dashboard", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")

    # Serve the web upload UI
    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/ui", StaticFiles(directory=str(static_dir)), name="ui-static")

    @app.get("/", include_in_schema=False)
    async def root():
        # Prefer the upload UI if present
        upload_file = Path(__file__).parent / "dashboard" / "upload.html"
        if upload_file.is_file():
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url="/dashboard/upload.html")
        # Fallback to existing dashboard
        dashboard_file = Path(__file__).parent / "dashboard" / "dashboard_bw.html"
        if dashboard_file.is_file():
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url="/dashboard/dashboard_bw.html")
        # Fallback to static UI or service info
        ui_index = Path(__file__).parent / "static" / "index.html"
        if ui_index.is_file():
            return FileResponse(str(ui_index), media_type="text/html")
        return {"service": "png-decomposer", "version": __version__, "docs": "/docs", "health": "/api/v1/health"}
    return app


app = create_app()

