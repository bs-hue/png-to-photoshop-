"""End-to-end orchestration: PNG bytes -> layer assets + manifest + PSD."""
from __future__ import annotations

import base64
import io
import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ..config import Settings
from ..schemas import (
    Bounds,
    DecomposeOptions,
    DecomposeResponse,
    DesignSpec,
    ColorSwatchSchema,
    EngineReport,
    GradientLayerSchema,
    GradientStopSchema,
    LayerInfo,
    LayerTreeNode,
    LayerType,
    LayoutSectionSchema,
    ShapeLayerSchema,
    TextAlignment,
    TextInfo,
)
from .common import dilate, logger
from .design import analyse_design, detect_text_containers
from .inpaint import InpaintService
from .ocr import OCRService
from .psd_builder import PSD_ENGINE, PsdGroupNode, PsdPixelNode, build_psd, build_psd_with_agpsd
from .segmentation import SegmentationService

Image.MAX_IMAGE_PIXELS = 16384 * 16384


class ImageDecodeError(ValueError):
    """Raised when the payload is not a decodable raster image."""


class ImageTooLargeError(ValueError):
    """Raised when the image exceeds the configured maximum side length."""


class DecompositionPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ocr = OCRService(settings.ocr_engine, settings.ocr_lang)
        self.segmentation = SegmentationService(
            settings.segmentation_engine, settings.sam2_checkpoint, settings.sam2_config
        )
        self.inpainter = InpaintService(settings.inpaint_engine, settings.sd_inpaint_model)
        self._run_lock = threading.Lock()  # AI models are not re-entrant
        settings.output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- helpers
    def engines_available(self) -> Dict[str, List[str]]:
        return {
            "ocr": self.ocr.available_engines(),
            "segmentation": self.segmentation.available_engines(),
            "inpainting": self.inpainter.available_engines(),
            "design": ["opencv-classical"],
            "psd": [PSD_ENGINE],
        }

    def cleanup_old_jobs(self) -> int:
        cutoff = time.time() - self.settings.job_ttl_hours * 3600
        removed = 0
        for job_dir in self.settings.output_dir.glob("*"):
            if job_dir.is_dir() and job_dir.stat().st_mtime < cutoff:
                shutil.rmtree(job_dir, ignore_errors=True)
                removed += 1
        return removed

    def decode(self, data: bytes) -> np.ndarray:
        if not data:
            raise ImageDecodeError("Empty image payload.")
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ImageDecodeError(f"Payload is not a decodable image: {exc}") from exc
        img = ImageOps.exif_transpose(img)
        if max(img.size) > self.settings.max_image_side:
            raise ImageTooLargeError(
                f"Image is {img.size[0]}x{img.size[1]}; max side is {self.settings.max_image_side}px."
            )
        if img.mode in ("RGBA", "LA", "P", "PA") or "transparency" in img.info:
            rgba = img.convert("RGBA")
            canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            canvas.alpha_composite(rgba)
            img = canvas
        return np.asarray(img.convert("RGB"), dtype=np.uint8).copy()

    # --------------------------------------------------------------------- run
    def run(self, data: bytes, options: DecomposeOptions, base_url: str) -> DecomposeResponse:
        with self._run_lock:
            return self._run(data, options, base_url)

    def _run(self, data: bytes, options: DecomposeOptions, base_url: str) -> DecomposeResponse:
        t_start = time.perf_counter()
        timings: Dict[str, float] = {}
        warnings: List[str] = []

        def tick(stage: str, t0: float) -> float:
            now = time.perf_counter()
            timings[stage] = round((now - t0) * 1000.0, 1)
            return now

        t = time.perf_counter()
        rgb = self.decode(data)
        h, w = rgb.shape[:2]
        job_id = uuid.uuid4().hex[:16]
        job_dir = self.settings.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        base = (self.settings.public_base_url or base_url).rstrip("/")
        t = tick("decode", t)

        # 1. OCR -----------------------------------------------------------
        regions = []
        text_mask = np.zeros((h, w), bool)
        ocr_engine = "disabled"
        if options.extract_text:
            ocr_res = self.ocr.detect(rgb)
            regions, text_mask, ocr_engine = ocr_res.regions, ocr_res.text_mask, ocr_res.engine
            warnings += ocr_res.warnings
            if ocr_engine == "heuristic" and regions:
                warnings.append(
                    "Heuristic OCR localised text but cannot read it; install PaddleOCR/EasyOCR "
                    "for editable strings. Text regions are delivered as raster layers."
                )
        t = tick("ocr", t)

        # 2. Segmentation --------------------------------------------------
        objects = []
        seg_engine = "disabled"
        if options.isolate_objects:
            seg = self.segmentation.segment(
                rgb, exclude=text_mask, max_objects=options.max_objects, min_area_ratio=options.min_object_area_ratio
            )
            objects, seg_engine = seg.objects, seg.engine
            warnings += seg.warnings
        t = tick("segmentation", t)

        # 2b. CTA Buttons / Badges / Containers ----------------------------
        cta_shapes = []
        cta_mask = np.zeros((h, w), bool)
        if options.extract_design or options.extract_text:
            try:
                cta_shapes = detect_text_containers(rgb, regions)
                for c_shp in cta_shapes:
                    if c_shp.mask is not None:
                        cta_mask |= c_shp.mask
            except Exception as exc:
                logger.warning("CTA container detection failed: %s", exc)

        # 3. Inpainting ----------------------------------------------------
        inpaint_engine = "disabled"
        background = rgb
        if options.inpaint_background:
            hole = text_mask.copy() | cta_mask
            for obj in objects:
                # Inpaint every extracted object so the background plate is 100% clean
                hole |= dilate(obj.mask, 4)
            res = self.inpainter.inpaint(rgb, hole)
            background, inpaint_engine = res.image, res.engine
            warnings += res.warnings
        t = tick("inpainting", t)

        # 3b. Design extraction --------------------------------------------
        design_spec = None
        design_engine = "disabled"
        _shapes = list(cta_shapes)
        _gradients = []
        _sections = []
        if options.extract_design:
            try:
                obj_mask = np.zeros((h, w), bool)
                for obj in objects:
                    obj_mask |= obj.mask
                da = analyse_design(rgb, text_mask, obj_mask)
                warnings += da.warnings
                design_engine = "opencv-classical"
                _gradients = da.gradients
                _sections = da.sections
                design_spec = DesignSpec(
                    palette=[
                        ColorSwatchSchema(hex=s.hex, rgb=list(s.rgb), coverage=s.coverage, role=s.role)
                        for s in da.palette
                    ],
                    design_tokens=da.design_tokens,
                    section_count=len(da.sections),
                )
            except Exception as exc:
                logger.exception("Design analysis failed")
                warnings.append(f"Design analysis failed: {exc!s}")
                design_engine = "failed"
        t = tick("design", t)

        # 4. Assets + manifest --------------------------------------------
        layers: List[LayerInfo] = []
        psd_nodes: List = []
        root = LayerTreeNode(id="document", name="Document", type=LayerType.DOCUMENT)

        def save_asset(name: str, img: Image.Image) -> Dict[str, Optional[str]]:
            path = job_dir / f"{name}.png"
            img.save(path, format="PNG", optimize=False, compress_level=4)
            rel = f"{job_id}/{path.name}"
            b64 = base64.b64encode(path.read_bytes()).decode("ascii") if options.include_base64 else None
            return {"asset_path": rel, "asset_url": f"{base}/assets/{rel}", "asset_base64": b64}

        def add_layer(info: LayerInfo, parent: LayerTreeNode) -> None:
            info.z_index = len(layers)
            layers.append(info)
            parent.children.append(LayerTreeNode(id=info.id, name=info.name, type=info.type, layer_id=info.id))

        full = Bounds(left=0, top=0, width=w, height=h)

        # ------------------------------------------------------------------
        # STRICT 4-TIER PHOTOSHOP HIERARCHY (ag-psd / PSD bottom-to-top order):
        # Tier 4 (Index 0): Background Group
        # Tier 3 (Index 1): Foreground Elements
        # Tier 2 (Index 2): CTAs & Badges Group
        # Tier 1 (Index 3): Text Group
        # ------------------------------------------------------------------

        # TIER 4: Background Group (Bottom)
        group_bg = LayerTreeNode(id="group_background", name="Background Group", type=LayerType.GROUP)
        root.children.append(group_bg)
        psd_group_bg = PsdGroupNode("Background Group")
        psd_nodes.append(psd_group_bg)

        # Optional original reference at the very bottom
        if options.include_original:
            orig_img = Image.fromarray(rgb).convert("RGBA")
            add_layer(
                LayerInfo(
                    id="original",
                    name="Original (reference)",
                    type=LayerType.ORIGINAL,
                    parent_id=group_bg.id,
                    bounds=full,
                    z_index=0,
                    visible=False,
                    **save_asset("original", orig_img),
                ),
                group_bg,
            )
            psd_group_bg.children.append(PsdPixelNode("Original (reference)", orig_img, 0, 0, visible=False))

        bg_img = Image.fromarray(background).convert("RGBA")
        bg_name = "Background (inpainted)" if inpaint_engine not in ("disabled", "none") else "Background"
        add_layer(
            LayerInfo(
                id="background",
                name=bg_name,
                type=LayerType.BACKGROUND,
                parent_id=group_bg.id,
                bounds=full,
                z_index=0,
                **save_asset("background", bg_img),
            ),
            group_bg,
        )
        psd_group_bg.children.append(PsdPixelNode(bg_name, bg_img, 0, 0))

        # TIER 3: Foreground Elements (Isolated subjects, people, laptops, props)
        group_fg = LayerTreeNode(id="group_foreground", name="Foreground Elements", type=LayerType.GROUP)
        root.children.append(group_fg)
        psd_group_fg = PsdGroupNode("Foreground Elements")
        psd_nodes.append(psd_group_fg)

        if objects:
            for i, obj in enumerate(objects, start=1):
                x0, y0, x1, y1 = obj.box
                rgba = np.dstack([rgb[y0:y1, x0:x1], obj.alpha[y0:y1, x0:x1]])
                img = Image.fromarray(rgba, "RGBA")
                lbl = getattr(obj, "label", "object")
                name = lbl.title() if lbl and lbl != "object" else f"Object {i}"
                add_layer(
                    LayerInfo(
                        id=f"object_{i}",
                        name=name,
                        type=LayerType.OBJECT,
                        parent_id=group_fg.id,
                        bounds=Bounds(left=x0, top=y0, width=x1 - x0, height=y1 - y0),
                        z_index=0,
                        score=round(obj.score, 4),
                        **save_asset(f"object_{i}", img),
                    ),
                    group_fg,
                )
                psd_group_fg.children.append(PsdPixelNode(name, img, x0, y0))

        # TIER 2: CTAs & Badges Group (Native vector shape layers + gradient fills + sub-text)
        group_cta = LayerTreeNode(id="group_cta_badges", name="CTAs & Badges Group", type=LayerType.GROUP)
        root.children.append(group_cta)
        psd_group_cta = PsdGroupNode("CTAs & Badges Group")
        psd_nodes.append(psd_group_cta)

        # Detect text regions associated with CTA shapes/containers
        shape_text_map: Dict[int, List[int]] = {i: [] for i in range(len(_shapes))}
        assigned_cta_text_indices: set[int] = set()

        for s_idx, shp in enumerate(_shapes):
            sx0, sy0, sx1, sy1 = shp.box
            for r_idx, reg in enumerate(regions):
                if r_idx in assigned_cta_text_indices:
                    continue
                rx0, ry0, rx1, ry1 = reg.box
                ix0, iy0 = max(sx0, rx0), max(sy0, ry0)
                ix1, iy1 = min(sx1, rx1), min(sy1, ry1)
                if ix1 > ix0 and iy1 > iy0:
                    inter_area = (ix1 - ix0) * (iy1 - iy0)
                    reg_area = max(1, (rx1 - rx0) * (ry1 - ry0))
                    if (inter_area / reg_area) >= 0.50:
                        shape_text_map[s_idx].append(r_idx)
                        assigned_cta_text_indices.add(r_idx)

        def make_text_layer(i: int, reg, parent_node: LayerTreeNode, psd_parent_group: PsdGroupNode, prefix: str = "text") -> None:
            x0, y0, x1, y1 = reg.box
            color = np.empty((y1 - y0, x1 - x0, 3), np.uint8)
            color[:] = reg.color_rgb
            img = Image.fromarray(np.dstack([color, reg.alpha]), "RGBA")
            label = reg.content if reg.content else f"Text region {i}"
            clean_label = re.sub(r"^T:\s*", "", label, flags=re.IGNORECASE).strip() or f"Text region {i}"
            name = clean_label[:120]
            layer_id = f"{prefix}_{i}"
            add_layer(
                LayerInfo(
                    id=layer_id,
                    name=name,
                    type=LayerType.TEXT,
                    parent_id=parent_node.id,
                    bounds=Bounds(left=x0, top=y0, width=x1 - x0, height=y1 - y0),
                    z_index=0,
                    score=round(reg.confidence, 4),
                    text=TextInfo(
                        content=clean_label,
                        font_size_px=reg.font_size_px,
                        color_hex=reg.color_hex,
                        alignment=TextAlignment(reg.alignment),
                        confidence=round(reg.confidence, 4),
                        editable=reg.editable,
                        font_family=reg.font_family,
                        polygon=reg.polygon,
                        leading_pt=getattr(reg, "leading_pt", None),
                        tracking=getattr(reg, "tracking", 0),
                        angle_deg=getattr(reg, "angle_deg", 0.0),
                    ),
                    **save_asset(layer_id, img),
                ),
                parent_node,
            )
            psd_parent_group.children.append(PsdPixelNode(name, img, x0, y0))

        if _shapes:
            for s_idx, shp in enumerate(_shapes, start=1):
                x0, y0, x1, y1 = shp.box
                bw, bh = max(1, x1 - x0), max(1, y1 - y0)
                shape_crop = rgb[y0:y1, x0:x1]
                if getattr(shp, "mask", None) is not None:
                    alpha_chan = (shp.mask[y0:y1, x0:x1].astype(np.uint8) * 255)
                else:
                    alpha_chan = np.full((bh, bw), int(round(shp.opacity * 255)), dtype=np.uint8)
                rgba = np.dstack([shape_crop, alpha_chan])
                img = Image.fromarray(rgba, "RGBA")
                name = shp.name if getattr(shp, "name", None) and shp.name != "Shape" else f"Shape {s_idx} ({shp.kind})"
                shape_layer_id = f"shape_{s_idx}"
                add_layer(
                    LayerInfo(
                        id=shape_layer_id,
                        name=name,
                        type=LayerType.SHAPE,
                        parent_id=group_cta.id,
                        bounds=Bounds(left=x0, top=y0, width=bw, height=bh),
                        z_index=0,
                        shape=ShapeLayerSchema(
                            kind=shp.kind,
                            fill_hex=shp.fill_hex,
                            stroke_hex=shp.stroke_hex,
                            stroke_width=shp.stroke_width,
                            vertices=shp.vertices,
                            corner_radius=getattr(shp, "corner_radius", 0.0),
                            gradient=getattr(shp, "gradient", None),
                        ),
                        **save_asset(shape_layer_id, img),
                    ),
                    group_cta,
                )
                psd_group_cta.children.append(PsdPixelNode(name, img, x0, y0))

                # Place matched CTA sub-text directly above the button shape
                for r_idx in shape_text_map.get(s_idx - 1, []):
                    make_text_layer(r_idx + 1, regions[r_idx], group_cta, psd_group_cta, prefix="cta_text")

        # TIER 1: Text Group (Top: editable Photoshop type layers with Google Fonts, explicit leading/tracking)
        group_text = LayerTreeNode(id="group_text", name="Text Group", type=LayerType.GROUP)
        root.children.append(group_text)
        psd_group_text = PsdGroupNode("Text Group")
        psd_nodes.append(psd_group_text)

        for i, reg in enumerate(regions, start=1):
            if (i - 1) in assigned_cta_text_indices:
                continue
            make_text_layer(i, reg, group_text, psd_group_text, prefix="text")
        t = tick("assets", t)

        # 5. PSD -----------------------------------------------------------
        psd_url = None
        psd_engine = "disabled"
        if options.build_psd:
            try:
                # First write standard PSD via psd-tools as baseline
                build_psd(w, h, psd_nodes, job_dir / "decomposed.psd")
                psd_url = f"{base}/assets/{job_id}/decomposed.psd"
                psd_engine = PSD_ENGINE
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("PSD build failed")
                warnings.append(f"PSD compilation failed: {exc!s}")
                psd_engine = "failed"
        t = tick("psd", t)
        timings["total"] = round((time.perf_counter() - t_start) * 1000.0, 1)

        response = DecomposeResponse(
            job_id=job_id,
            width=w,
            height=h,
            options=options,
            layers=layers,
            layer_tree=root,
            design=design_spec,
            psd_url=psd_url,
            manifest_url=f"{base}/api/v1/jobs/{job_id}",
            engines=EngineReport(
                ocr=ocr_engine,
                segmentation=seg_engine,
                inpainting=inpaint_engine,
                design=design_engine,
                psd=psd_engine,
            ),
            timings_ms=timings,
            warnings=warnings,
        )
        manifest = response.model_dump(mode="json")
        for layer in manifest["layers"]:  # keep the on-disk manifest small
            layer["asset_base64"] = None
        (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # Upgrade PSD with native editable Type layers via ag-psd if possible
        if options.build_psd:
            if build_psd_with_agpsd(job_dir):
                response.engines.psd = "ag-psd"
                response.psd_url = f"{base}/assets/{job_id}/decomposed.psd"
                logger.info("PSD upgraded with native Photoshop Type layers via ag-psd")

        return response

