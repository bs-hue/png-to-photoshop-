"""Pydantic contracts shared by the API, the pipeline and the test-suite."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class LayerType(str, Enum):
    DOCUMENT = "document"
    GROUP = "group"
    ORIGINAL = "original"
    BACKGROUND = "background"
    OBJECT = "object"
    TEXT = "text"
    SHAPE = "shape"
    GRADIENT = "gradient"
    SECTION = "section"


class TextAlignment(str, Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class Bounds(BaseModel):
    """Pixel rectangle in document space (origin = top-left of the canvas)."""

    left: int
    top: int
    width: int = Field(ge=0)
    height: int = Field(ge=0)

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


class DecomposeOptions(BaseModel):
    extract_text: bool = Field(True, description="Run OCR and emit editable text layers.")
    isolate_objects: bool = Field(True, description="Segment foreground subjects into transparent cut-outs.")
    inpaint_background: bool = Field(True, description="Backfill the background behind removed text/objects.")
    extract_design: bool = Field(True, description="Extract shapes, color palette, gradients, layout sections.")
    build_psd: bool = Field(True, description="Compile a layered .psd file next to the PNG assets.")
    include_base64: bool = Field(False, description="Inline every layer PNG as base64 in the response.")
    include_original: bool = Field(True, description="Add a hidden copy of the source image as bottom layer.")
    max_objects: int = Field(8, ge=0, le=64)
    min_object_area_ratio: float = Field(0.002, ge=0.0, le=1.0)


class DecomposeJSONRequest(BaseModel):
    image_base64: str = Field(..., description="PNG/JPEG bytes as base64 (data-URL prefix allowed).")
    options: DecomposeOptions = Field(default_factory=DecomposeOptions)


class TextInfo(BaseModel):
    content: str = Field(..., description="Recognised string. Empty when the engine can only localise text.")
    font_size_px: float = Field(..., gt=0)
    color_hex: str = Field(..., pattern=r"^#[0-9A-F]{6}$")
    alignment: TextAlignment
    confidence: float = Field(..., ge=0.0, le=1.0)
    editable: bool = Field(..., description="True when content is known and a native text layer can be built.")
    font_family: str = Field("ArialMT", description="Suggested PostScript font name.")
    polygon: List[List[int]] = Field(default_factory=list, description="Detected quad, document space.")
    leading_pt: Optional[float] = Field(None, description="Typographic line leading (baseline-to-baseline distance).")
    tracking: int = Field(0, description="Photoshop tracking / letter spacing.")
    angle_deg: float = Field(0.0, description="Baseline rotation angle in degrees.")


class LayerInfo(BaseModel):
    id: str
    name: str
    type: LayerType
    parent_id: Optional[str] = None
    bounds: Bounds
    z_index: int = Field(..., ge=0, description="0 = bottom-most layer.")
    visible: bool = True
    opacity: int = Field(100, ge=0, le=100)
    blend_mode: str = "normal"
    asset_url: Optional[str] = None
    asset_path: Optional[str] = Field(None, description="Path relative to the /assets mount.")
    asset_base64: Optional[str] = None
    asset_mime: Optional[str] = "image/png"
    score: Optional[float] = Field(None, ge=0.0, le=1.0)
    text: Optional[TextInfo] = None
    shape: Optional[ShapeLayerSchema] = None
    gradient: Optional[GradientLayerSchema] = None
    section: Optional[LayoutSectionSchema] = None


class LayerTreeNode(BaseModel):
    id: str
    name: str
    type: LayerType
    layer_id: Optional[str] = Field(None, description="Reference into `layers` for leaf nodes.")
    children: List["LayerTreeNode"] = Field(
        default_factory=list, description="Ordered bottom -> top, like the layers list."
    )


LayerTreeNode.model_rebuild()


class EngineReport(BaseModel):
    ocr: str
    segmentation: str
    inpainting: str
    design: str
    psd: str


class ColorSwatchSchema(BaseModel):
    hex: str
    rgb: List[int]
    coverage: float
    role: str


class ShapeLayerSchema(BaseModel):
    kind: str
    fill_hex: str
    stroke_hex: Optional[str] = None
    stroke_width: int = 1
    vertices: List[List[int]] = Field(default_factory=list)
    corner_radius: float = 0.0
    gradient: Optional[Dict[str, Any]] = None


class GradientStopSchema(BaseModel):
    pos: float
    hex: str


class GradientLayerSchema(BaseModel):
    kind: str           # "linear" | "radial"
    angle_deg: float
    stops: List[GradientStopSchema]


class LayoutSectionSchema(BaseModel):
    label: str
    bg_hex: str
    confidence: float


class DesignSpec(BaseModel):
    palette: List[ColorSwatchSchema] = Field(default_factory=list)
    design_tokens: Dict[str, Any] = Field(default_factory=dict)
    section_count: int = 0


class DecomposeResponse(BaseModel):
    job_id: str
    status: Literal["completed"] = "completed"
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    options: DecomposeOptions
    layers: List[LayerInfo] = Field(..., description="Flat list ordered bottom -> top.")
    layer_tree: LayerTreeNode
    design: Optional[DesignSpec] = None
    psd_url: Optional[str] = None
    manifest_url: Optional[str] = None
    engines: EngineReport
    timings_ms: Dict[str, float]
    warnings: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    engines_available: Dict[str, List[str]]
    engines_configured: Dict[str, str]

