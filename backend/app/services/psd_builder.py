"""PSD compiler: turns the decomposed layer tree into a layered ``.psd`` file.

Uses ``psd-tools`` (>= 1.10 creation API). psd-tools cannot author native
type layers, so text is written as a correctly positioned raster layer whose
name carries the recognised string; the Photoshop UXP plugin re-creates
those as live, editable text layers from the JSON manifest.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from PIL import Image

PSD_ENGINE = "ag-psd"


def build_psd_with_agpsd(job_dir: Path) -> bool:
    """Run build_psd.js to compile native Photoshop type layers using ag-psd."""
    script_path = Path(__file__).resolve().parents[2] / "build_psd.js"
    if not script_path.exists():
        return False
    try:
        proc = subprocess.run(
            ["node", str(script_path), str(job_dir)],
            capture_output=True,
            text=True,
            timeout=45,
        )
        return proc.returncode == 0
    except Exception:
        return False


@dataclass
class PsdPixelNode:
    name: str
    image: Image.Image  # RGBA
    left: int
    top: int
    visible: bool = True
    opacity: int = 100  # percent


@dataclass
class PsdGroupNode:
    name: str
    children: List["PsdNode"] = field(default_factory=list)  # bottom -> top
    visible: bool = True


PsdNode = Union[PsdPixelNode, PsdGroupNode]


def _safe_mac_name(name: str) -> str:
    """Ensure layer name is encodable by psd-tools legacy mac_roman writer without crashing."""
    try:
        name[:250].encode("mac_roman")
        return name[:250]
    except Exception:
        safe = "".join(c if ord(c) < 128 else "_" for c in name)
        return (safe.strip("_") or "Layer")[:250]


def build_psd(width: int, height: int, nodes: List[PsdNode], out_path: Path) -> Optional[Path]:
    """Write ``nodes`` (ordered bottom -> top) to ``out_path``. Returns the path."""
    from psd_tools import PSDImage
    from psd_tools.api.layers import Group, PixelLayer

    psd = PSDImage.new("RGB", (width, height), color=(255, 255, 255))

    def emit(node: PsdNode, parent) -> None:
        if isinstance(node, PsdGroupNode):
            group = Group.new(parent, _safe_mac_name(node.name), open_folder=True)
            group.visible = node.visible
            for child in node.children:
                emit(child, group)
            return
        img = node.image if node.image.mode == "RGBA" else node.image.convert("RGBA")
        layer = PixelLayer.frompil(img, parent, _safe_mac_name(node.name), top=int(node.top), left=int(node.left))
        layer.visible = node.visible
        layer.opacity = int(round(max(0, min(100, node.opacity)) * 2.55))

    for node in nodes:
        emit(node, psd)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    psd.save(str(out_path))
    return out_path

