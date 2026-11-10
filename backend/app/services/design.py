"""Design analysis module: extracts everything beyond text and objects.

Extracts:
  - Dominant color palette (K-means clustering)
  - Geometric shapes: rectangles, circles, lines, polygons (OpenCV contours)
  - Gradient regions: linear / radial fills detected via pixel sampling
  - Layout sections: horizontal bands → Header / Hero / Content / Footer / CTA
  - Design spec: typography summary, spacing grid, design tokens
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage

from .common import logger, mask_bbox, rgb_to_hex

# ─────────────────────────────────────────── data classes ──────────────────
@dataclass
class ColorSwatch:
    hex: str
    rgb: Tuple[int, int, int]
    coverage: float          # fraction of image pixels
    role: str                # "dominant" | "accent" | "background" | "neutral"
    name: str = ""


@dataclass
class ShapeRegion:
    kind: str                # "rect" | "rounded_rect" | "circle" | "ellipse" | "line" | "polygon"
    box: Tuple[int, int, int, int]   # x0, y0, x1, y1 (exclusive)
    fill_hex: str
    stroke_hex: Optional[str]
    stroke_width: int
    opacity: float           # 0-1 estimated from alpha / uniformity
    is_filled: bool
    vertices: List[List[int]] = field(default_factory=list)  # document space
    name: str = "Shape"
    mask: Optional[np.ndarray] = None
    corner_radius: float = 0.0
    gradient: Optional[Dict] = None


@dataclass
class GradientRegion:
    kind: str                # "linear" | "radial"
    box: Tuple[int, int, int, int]
    stops: List[Dict]        # [{"pos": 0.0, "hex": "#RRGGBB"}, ...]
    angle_deg: float         # for linear; 0 for radial


@dataclass
class LayoutSection:
    label: str               # "header" | "hero" | "content" | "cta" | "footer" | "sidebar"
    box: Tuple[int, int, int, int]
    bg_hex: str
    confidence: float


@dataclass
class DesignAnalysis:
    palette: List[ColorSwatch]
    shapes: List[ShapeRegion]
    gradients: List[GradientRegion]
    sections: List[LayoutSection]
    design_tokens: Dict      # spacing, radius, shadow hints
    warnings: List[str] = field(default_factory=list)


# ─────────────────────────────────────────── main entry ────────────────────
def analyse_design(
    rgb: np.ndarray,
    text_mask: np.ndarray,
    object_mask: np.ndarray,
    n_palette: int = 8,
    max_shapes: int = 24,
) -> DesignAnalysis:
    H, W = rgb.shape[:2]
    warnings: List[str] = []

    # work on a scaled-down copy for performance
    scale = min(1.0, 1200.0 / max(H, W))
    small = cv2.resize(rgb, (max(1, int(W * scale)), max(1, int(H * scale))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]

    # masks at working resolution
    tm_s = cv2.resize(text_mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
    om_s = cv2.resize(object_mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
    content_mask = tm_s | om_s

    palette = _extract_palette(small, content_mask, n_palette)
    shapes = _extract_shapes(small, content_mask, scale, max_shapes)
    gradients = _extract_gradients(small, content_mask, scale)
    sections = _extract_sections(small, content_mask, scale)
    tokens = _derive_tokens(shapes, palette, W, H)

    return DesignAnalysis(
        palette=palette,
        shapes=shapes,
        gradients=gradients,
        sections=sections,
        design_tokens=tokens,
        warnings=warnings,
    )


# ─────────────────────────────────────────── palette ───────────────────────
def _extract_palette(rgb: np.ndarray, exclude: np.ndarray, n: int) -> List[ColorSwatch]:
    flat = rgb[~exclude].reshape(-1, 3).astype(np.float32)
    if len(flat) < n * 4:
        return []

    # subsample for speed
    step = max(1, len(flat) // 8000)
    sample = flat[::step]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    _, labels, centers = cv2.kmeans(sample, min(n, len(sample)), None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    centers = np.clip(np.round(centers), 0, 255).astype(np.uint8)

    total = len(labels)
    counts = np.bincount(labels.ravel(), minlength=len(centers))
    order = np.argsort(counts)[::-1]

    swatches = []
    for rank, idx in enumerate(order):
        rgb_t = tuple(int(v) for v in centers[idx])
        cov = float(counts[idx]) / total
        if cov < 0.005:
            break
        brightness = int(centers[idx]) if centers[idx].ndim == 0 else int(np.mean(centers[idx]))
        saturation = _saturation(centers[idx])
        if rank == 0:
            role = "dominant"
        elif saturation > 0.45:
            role = "accent"
        elif brightness > 200 or brightness < 40:
            role = "background"
        else:
            role = "neutral"
        swatches.append(ColorSwatch(hex=rgb_to_hex(rgb_t), rgb=rgb_t, coverage=round(cov, 4), role=role))
    return swatches


def _saturation(rgb_arr) -> float:
    r, g, b = [float(x) / 255.0 for x in rgb_arr[:3]]
    mx, mn = max(r, g, b), min(r, g, b)
    return (mx - mn) / mx if mx > 0 else 0.0


# ─────────────────────────────────────────── buttons & containers ──────────
def detect_text_containers(rgb: np.ndarray, text_regions: list) -> List[ShapeRegion]:
    """Detect button pills, badges, stickers, and cards enclosing text elements."""
    H, W = rgb.shape[:2]
    text_boxes = []

    for reg in text_regions:
        if hasattr(reg, "box"):
            tx0, ty0, tx1, ty1 = reg.box
            content = getattr(reg, "content", "Button")
        elif isinstance(reg, (list, tuple)) and len(reg) >= 4:
            tx0, ty0, tx1, ty1 = reg[:4]
            content = reg[4] if len(reg) > 4 else "Button"
        else:
            continue
        tw, th = max(1, tx1 - tx0), max(1, ty1 - ty0)
        text_boxes.append((tx0, ty0, tw, th, content))

    test_boxes = []
    # 1. Individual text lines
    for b in text_boxes:
        tx, ty, tw, th, content = b
        if 20 < tw < 0.85 * W and 12 < th < 0.35 * H:
            test_boxes.append((tx, ty, tx + tw, ty + th, [content]))

    # 2. Clusters of nearby lines (e.g. multi-line buttons, circular stickers)
    n = len(text_boxes)
    for i in range(n):
        b1 = text_boxes[i]
        for j in range(i + 1, n):
            b2 = text_boxes[j]
            dx = max(0, max(b1[0], b2[0]) - min(b1[0] + b1[2], b2[0] + b2[2]))
            dy = max(0, max(b1[1], b2[1]) - min(b1[1] + b1[3], b2[1] + b2[3]))
            if dx <= 25 and dy <= 25:
                ux0 = min(b1[0], b2[0])
                uy0 = min(b1[1], b2[1])
                ux1 = max(b1[0] + b1[2], b2[0] + b2[2])
                uy1 = max(b1[1] + b1[3], b2[1] + b2[3])
                if (ux1 - ux0) < 0.80 * W and (uy1 - uy0) < 0.35 * H:
                    test_boxes.append((ux0, uy0, ux1, uy1, [b1[4], b2[4]]))

    candidates = []

    # 0. Direct full-canvas button, badge, pill, and sticker detection
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    thresholds = []
    _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholds.extend([th1, ~th1])
    _, th_s = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholds.extend([th_s, ~th_s])
    # Neon / glowing stroke threshold (for buttons with glowing neon borders on dark backgrounds)
    neon_th = ((hsv[:, :, 2] > 165) & (hsv[:, :, 1] > 35)).astype(np.uint8) * 255
    thresholds.append(neon_th)
    for ci in (1, 2):
        if lab[:, :, ci].std() > 5.0:
            _, th_lab = cv2.threshold(lab[:, :, ci], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresholds.extend([th_lab, ~th_lab])

    for th in thresholds:
        opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 0.002 * H * W or area > 0.20 * H * W:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            # Only skip if it spans the full width AND height (i.e. canvas background)
            if (bx <= 2 and (bx + bw) >= W - 2 and by <= 2 and (by + bh) >= H - 2) or (by <= 2 and (by + bh) >= H - 2):
                continue
            # A true button container must be larger than a single headline glyph
            if bw < 85 or bh < 30:
                continue
            aspect = bw / float(bh)
            if aspect < 0.65 or aspect > 16.0:
                continue
            hull = cv2.convexHull(cnt)
            solidity = area / max(1.0, cv2.contourArea(hull))
            if solidity < 0.80:
                continue
            fill_rect = area / float(bw * bh)
            if fill_rect < 0.52:
                continue

            c_mask = np.zeros((H, W), np.uint8)
            cv2.drawContours(c_mask, [cnt], -1, 255, -1)
            dil = cv2.dilate(c_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
            ring = (dil > 0) & (c_mask == 0)
            if ring.sum() < 30:
                continue
            int_color = np.median(rgb[c_mask > 0], axis=0)
            ext_color = np.median(rgb[ring], axis=0)
            
            # Contrast with exterior: check fill contrast or stroke/border contrast
            ys, xs = np.nonzero(c_mask > 0)
            stroke_mask = c_mask & ~cv2.erode(c_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            stroke_color = np.median(rgb[stroke_mask > 0], axis=0) if (stroke_mask > 0).any() else int_color
            contrast = max(float(np.linalg.norm(int_color - ext_color)), float(np.linalg.norm(stroke_color - ext_color)))
            if contrast < 26.0:
                continue

            inner_edges = grad_mag[c_mask > 0]
            if len(inner_edges) < 20 or inner_edges.std() < 10.0:
                continue

            matching_labels = []
            for tx, ty, tw, th, content in text_boxes:
                tcx = tx + tw // 2
                tcy = ty + th // 2
                if bx - 10 <= tcx <= bx + bw + 10 and by - 10 <= tcy <= by + bh + 10:
                    matching_labels.append(content)

            # Must contain recognized text inside the button container
            if not matching_labels:
                continue

            lbl = " ".join(matching_labels).strip() or "CTA Button"
            clean_lbl = "Button / Badge: " + lbl if lbl != "CTA Button" else "CTA Button"
            candidates.append({
                "box": (bx, by, bx + bw, by + bh),
                "mask": c_mask > 0,
                "label": clean_lbl,
                "area": area,
            })

    for x0_b, y0_b, x1_b, y1_b, labels in test_boxes:
        bw = x1_b - x0_b
        bh = y1_b - y0_b

        px = max(24, int(bw * 0.32))
        py = max(18, int(bh * 0.42))
        sx0 = max(0, x0_b - px)
        sy0 = max(0, y0_b - py)
        sx1 = min(W, x1_b + px)
        sy1 = min(H, y1_b + py)
        crop = rgb[sy0:sy1, sx0:sx1]
        ch, cw = crop.shape[:2]
        if ch < 20 or cw < 30:
            continue

        border_mask = np.zeros((ch, cw), bool)
        border_mask[:3, :] = border_mask[-3:, :] = True
        border_mask[:, :3] = border_mask[:, -3:] = True
        border_px = crop[border_mask]
        mean_border = np.median(border_px, axis=0)

        bins = []
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        _, th_g = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bins.append(th_g)
        bins.append(~th_g)

        # Outlined button stroke edges (e.g. glowing/stroke cards with transparent or dark fill)
        edges = cv2.Canny(gray, 20, 85)
        edge_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        filled_edges = ndimage.binary_fill_holes(edge_closed).astype(np.uint8) * 255
        bins.append(filled_edges)

        lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
        for ci in (1, 2):
            c_chan = lab[:, :, ci]
            if c_chan.std() > 6.0:
                _, th_c = cv2.threshold(c_chan, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                bins.append(th_c)
                bins.append(~th_c)

        tc_x = (x0_b + bw // 2) - sx0
        tc_y = (y0_b + bh // 2) - sy0

        best_cnt = None
        best_score = -1.0

        for bin_img in bins:
            if bin_img[border_mask].mean() > 100:
                continue
            cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area < bw * bh * 0.65 or area > cw * ch * 0.88:
                    continue
                if cv2.pointPolygonTest(c, (float(tc_x), float(tc_y)), False) >= 0:
                    cbx, cby, cbw, cbh = cv2.boundingRect(c)
                    # Only skip if it fills the entire crop vertically and horizontally
                    if (cby <= 1 and (cby + cbh) >= ch - 1) or (cbw >= cw - 2 and cbh >= ch - 2):
                        continue
                    if cbw >= bw * 0.70 and cbh >= bh * 0.70:
                        fill = area / float(cbw * cbh)
                        if fill > 0.52:
                            c_mask = np.zeros((ch, cw), np.uint8)
                            cv2.drawContours(c_mask, [c], -1, 255, -1)
                            interior_px = crop[c_mask > 0]
                            if len(interior_px) == 0:
                                continue
                            mean_int = np.median(interior_px, axis=0)
                            # Contrast with background: test interior fill contrast OR stroke border contrast
                            stroke_zone = cv2.dilate(c_mask, np.ones((3, 3), np.uint8)) & ~cv2.erode(c_mask, np.ones((3, 3), np.uint8))
                            stroke_px = crop[stroke_zone > 0]
                            stroke_contrast = float(np.linalg.norm(np.median(stroke_px, axis=0) - mean_border)) if len(stroke_px) > 0 else 0.0
                            contrast = max(float(np.linalg.norm(mean_int - mean_border)), stroke_contrast)
                            if contrast < 14.0:
                                continue
                            score = contrast * fill * (area / float(bw * bh))
                            if score > best_score:
                                best_score = score
                                best_cnt = (c, sx0, sy0, sx1, sy1, " ".join(labels))

        if best_cnt is not None:
            c, sx0, sy0, sx1, sy1, lbl = best_cnt
            cbx, cby, cbw, cbh = cv2.boundingRect(c)
            abs_box = (sx0 + cbx, sy0 + cby, sx0 + cbx + cbw, sy0 + cby + cbh)
            full_mask = np.zeros((H, W), np.uint8)
            cv2.drawContours(full_mask[sy0:sy1, sx0:sx1], [c], -1, 255, -1)
            candidates.append({"box": abs_box, "mask": full_mask > 0, "label": lbl, "area": (full_mask > 0).sum()})

    # Also check for circular badges (HoughCircles) around text
    for x0_b, y0_b, x1_b, y1_b, labels in test_boxes:
        bw = x1_b - x0_b
        bh = y1_b - y0_b
        if 0.6 < bw / float(bh) < 1.7 and 60 < bw < 350:
            cx = (x0_b + x1_b) // 2
            cy = (y0_b + y1_b) // 2
            rad = int(max(bw, bh) * 0.70)
            if rad > 30 and cx - rad >= 0 and cy - rad >= 0 and cx + rad < W and cy + rad < H:
                c_crop = rgb[cy - rad:cy + rad, cx - rad:cx + rad]
                c_gray = cv2.cvtColor(c_crop, cv2.COLOR_BGR2GRAY)
                c_blur = cv2.GaussianBlur(c_gray, (9, 9), 2)
                circles = cv2.HoughCircles(c_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80, param1=100, param2=30, minRadius=int(rad * 0.6), maxRadius=rad)
                if circles is not None:
                    hc = circles[0][0]
                    hr = int(hc[2])
                    abs_cx = (cx - rad) + int(hc[0])
                    abs_cy = (cy - rad) + int(hc[1])
                    circle_mask = np.zeros((H, W), np.uint8)
                    cv2.circle(circle_mask, (abs_cx, abs_cy), hr, 255, -1)
                    candidates.append({
                        "box": (abs_cx - hr, abs_cy - hr, abs_cx + hr, abs_cy + hr),
                        "mask": circle_mask > 0,
                        "label": "Badge: " + " ".join(labels),
                        "area": int(np.pi * hr * hr)
                    })

    # Filter valid buttons & badges by geometry
    filtered_candidates = []
    for cand in candidates:
        bw = cand["box"][2] - cand["box"][0]
        bh = cand["box"][3] - cand["box"][1]
        asp = bw / float(bh)
        is_pill_or_rect = (1.30 <= asp <= 16.0 and bh < 0.28 * H)
        is_round_badge = (0.75 <= asp <= 1.35 and bw < 0.35 * W and cand["area"] < 0.08 * H * W)
        if not (is_pill_or_rect or is_round_badge):
            continue
        filtered_candidates.append(cand)

    # Deduplicate overlapping
    filtered_candidates.sort(key=lambda x: x["area"], reverse=True)
    merged = []
    for cand in filtered_candidates:
        x0, y0, x1, y1 = cand["box"]
        dup = False
        for m in merged:
            mx0, my0, mx1, my1 = m["box"]
            ix0, iy0 = max(x0, mx0), max(y0, my0)
            ix1, iy1 = min(x1, mx1), min(y1, my1)
            if ix1 > ix0 and iy1 > iy0:
                inter = (ix1 - ix0) * (iy1 - iy0)
                a1 = (x1 - x0) * (y1 - y0)
                a2 = (mx1 - mx0) * (my1 - my0)
                if inter > 0.45 * min(a1, a2):
                    dup = True
                    break
        if not dup:
            merged.append(cand)

    results = []
    for item in merged:
        box = item["box"]
        mask = item["mask"]
        text_label = item["label"]
        x0, y0, x1, y1 = box
        crop = rgb[y0:y1, x0:x1]
        c_mask = mask[y0:y1, x0:x1]
        geo = analyze_container_geometry(crop, c_mask)

        clean_lbl = re.sub(r"^(Button\s*/\s*Badge:\s*|Badge:\s*)+", "", text_label or "", flags=re.IGNORECASE).strip()
        clean_name = f"Button / Badge: {clean_lbl}" if clean_lbl else "CTA Button"
        results.append(
            ShapeRegion(
                kind=geo["kind"],
                box=box,
                fill_hex=geo["fill_hex"],
                stroke_hex=geo["stroke_hex"],
                stroke_width=geo["stroke_width"],
                opacity=1.0,
                is_filled=True,
                vertices=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                name=clean_name,
                mask=mask,
                corner_radius=geo["corner_radius"],
                gradient=geo["gradient"],
            )
        )

    return results


def analyze_container_geometry(crop: np.ndarray, mask: np.ndarray) -> dict:
    """Analyze a button / container crop and its mask to extract geometry and styling (Phase 3)."""
    h, w = mask.shape
    bw, bh = max(1, w), max(1, h)

    # 1. Corner radius estimation
    diag_len = min(bw, bh) // 2
    deltas = []
    for cx, cy in [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]:
        dx_dir = 1 if cx == 0 else -1
        dy_dir = 1 if cy == 0 else -1
        d = 0
        for step in range(diag_len):
            px = cx + step * dx_dir
            py = cy + step * dy_dir
            if 0 <= px < w and 0 <= py < h and mask[py, px]:
                d = step
                break
        deltas.append(d)

    avg_delta = float(np.median(deltas))
    corner_r = avg_delta * 2.414  # R = delta / (sqrt(2) - 1)

    if corner_r < 3.0:
        kind = "rect"
        corner_r = 0.0
    elif corner_r >= (bh * 0.38):
        kind = "pill"
        corner_r = round(bh / 2.0, 1)
    else:
        kind = "rounded_rect"
        corner_r = round(corner_r, 1)

    # 2. Color & Gradient extraction
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    inner_mask = cv2.erode(mask.astype(np.uint8), kernel) > 0
    if inner_mask.sum() < 20:
        inner_mask = mask > 0

    interior_rgb = crop[inner_mask]
    med_color = tuple(int(v) for v in np.median(interior_rgb, axis=0)) if len(interior_rgb) > 0 else (200, 200, 200)
    std_color = float(np.mean(np.std(interior_rgb, axis=0))) if len(interior_rgb) > 0 else 0.0

    gradient_info = None
    if std_color > 14.0 and bw >= 30 and bh >= 16:
        w20 = max(2, int(bw * 0.20))
        h20 = max(2, int(bh * 0.20))

        m_left = inner_mask[:, :w20]
        m_right = inner_mask[:, -w20:]
        m_top = inner_mask[:h20, :]
        m_bottom = inner_mask[-h20:, :]

        c_left = np.median(crop[:, :w20][m_left], axis=0) if m_left.any() else np.array(med_color)
        c_right = np.median(crop[:, -w20:][m_right], axis=0) if m_right.any() else np.array(med_color)
        c_top = np.median(crop[:h20, :][m_top], axis=0) if m_top.any() else np.array(med_color)
        c_bottom = np.median(crop[-h20:, :][m_bottom], axis=0) if m_bottom.any() else np.array(med_color)

        diff_h = float(np.linalg.norm(c_right - c_left))
        diff_v = float(np.linalg.norm(c_bottom - c_top))

        if diff_h > 18.0 and diff_h >= diff_v:
            gradient_info = {
                "type": "linear",
                "angle_deg": 0.0,
                "stops": [
                    {"pos": 0.0, "hex": rgb_to_hex(tuple(int(v) for v in c_left))},
                    {"pos": 1.0, "hex": rgb_to_hex(tuple(int(v) for v in c_right))},
                ],
            }
        elif diff_v > 18.0:
            gradient_info = {
                "type": "linear",
                "angle_deg": 90.0,
                "stops": [
                    {"pos": 0.0, "hex": rgb_to_hex(tuple(int(v) for v in c_top))},
                    {"pos": 1.0, "hex": rgb_to_hex(tuple(int(v) for v in c_bottom))},
                ],
            }

    # 3. Stroke / border extraction
    border_mask = (mask > 0) & (~inner_mask)
    stroke_hex = None
    stroke_width = 0
    if border_mask.any() and border_mask.sum() > 20:
        border_med = np.median(crop[border_mask], axis=0)
        if np.linalg.norm(border_med - np.array(med_color)) > 26.0:
            stroke_hex = rgb_to_hex(tuple(int(v) for v in border_med))
            stroke_width = max(1, int(round(avg_delta * 0.25)))

    return {
        "kind": kind,
        "corner_radius": corner_r,
        "fill_hex": rgb_to_hex(med_color),
        "gradient": gradient_info,
        "stroke_hex": stroke_hex,
        "stroke_width": stroke_width,
    }


# ─────────────────────────────────────────── shapes ────────────────────────
def _extract_shapes(rgb: np.ndarray, exclude: np.ndarray, scale: float, max_shapes: int) -> List[ShapeRegion]:
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Multi-scale Canny
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 90)

    # Dilate slightly to close small gaps
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    shapes: List[ShapeRegion] = []
    min_area = (h * w) * 0.0015          # skip tiny noise
    max_area = (h * w) * 0.96            # skip full-frame rectangle

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # skip if entirely inside text/object regions
        mask_check = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask_check, [cnt], -1, 1, -1)
        if exclude[mask_check > 0].mean() > 0.82:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        x1, y1 = x + bw, y + bh

        # Classify shape type
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        n_vertices = len(approx)
        aspect = bw / max(1, bh)

        fill_rgb = _median_color(rgb, mask_check > 0)
        stroke_rgb = _edge_color(rgb, cnt)

        is_uniform = _region_uniformity(rgb, mask_check > 0) > 0.65
        if not is_uniform:
            continue  # skip photos / complex textures

        if n_vertices == 4 and 0.7 < aspect < 1.4:
            kind = "rect" if _is_sharp_rect(approx) else "rounded_rect"
        elif n_vertices <= 6 and 0.85 < aspect < 1.15:
            kind = "circle" if _circularity(cnt) > 0.75 else "ellipse"
        elif n_vertices == 2 or (bh < 0.05 * h and bw > 0.15 * w):
            kind = "line"
        else:
            kind = "polygon"

        # Scale back to original resolution
        sx = 1.0 / scale
        shapes.append(ShapeRegion(
            kind=kind,
            box=(int(x * sx), int(y * sx), int(x1 * sx), int(y1 * sx)),
            fill_hex=rgb_to_hex(fill_rgb),
            stroke_hex=rgb_to_hex(stroke_rgb) if stroke_rgb is not None else None,
            stroke_width=max(1, int(2 / scale)),
            opacity=round(float(is_uniform), 2),
            is_filled=is_uniform,
            vertices=[[int(p[0][0] * sx), int(p[0][1] * sx)] for p in approx],
        ))

        if len(shapes) >= max_shapes:
            break

    return shapes


def _median_color(rgb, mask):
    px = rgb[mask]
    if len(px) == 0:
        return (128, 128, 128)
    med = np.median(px, axis=0)
    return tuple(int(v) for v in med)


def _edge_color(rgb, cnt):
    h, w = rgb.shape[:2]
    edge_mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(edge_mask, [cnt], -1, 1, 2)
    px = rgb[edge_mask > 0]
    if len(px) == 0:
        return None
    return tuple(int(v) for v in np.median(px, axis=0))


def _region_uniformity(rgb, mask):
    px = rgb[mask].astype(np.float32)
    if len(px) < 10:
        return 0.0
    std = np.std(px, axis=0).mean()
    return float(np.exp(-std / 30.0))


def _is_sharp_rect(approx) -> bool:
    if len(approx) != 4:
        return False
    pts = approx.reshape(4, 2)
    for i in range(4):
        a = pts[i]
        b = pts[(i + 1) % 4]
        c = pts[(i + 2) % 4]
        ab = b - a; bc = c - b
        cos_a = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc) + 1e-6)
        if abs(cos_a) > 0.25:  # not close to 90°
            return False
    return True


def _circularity(cnt) -> float:
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    return 4 * math.pi * area / (peri * peri + 1e-6)


# ─────────────────────────────────────────── gradients ─────────────────────
def _extract_gradients(rgb: np.ndarray, exclude: np.ndarray, scale: float) -> List[GradientRegion]:
    h, w = rgb.shape[:2]
    gradients: List[GradientRegion] = []
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Divide canvas into a 3×3 grid and check each zone for gradient-like smoothness
    rows, cols = 3, 3
    rh, cw = h // rows, w // cols
    visited = np.zeros((h, w), bool)

    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * rh, (r + 1) * rh
            x0, x1 = c * cw, (c + 1) * cw
            zone_rgb = rgb[y0:y1, x0:x1]
            zone_excl = exclude[y0:y1, x0:x1]
            zone_vis = visited[y0:y1, x0:x1]
            zone_free = ~zone_excl & ~zone_vis
            if zone_free.mean() < 0.5:
                continue
            px = zone_rgb[zone_free].astype(np.float32)
            if len(px) < 200:
                continue

            std_spatial = _spatial_gradient_score(zone_rgb, zone_free)
            if std_spatial < 15.0:
                continue  # uniform, not a gradient

            # Sample along H and V axes to determine direction
            h_profile = zone_rgb[:, w // (2 * cols), :].astype(np.float32)  # vertical strip
            v_profile = zone_rgb[h // (2 * rows), :, :].astype(np.float32)  # horizontal strip
            h_var = np.std(h_profile, axis=0).mean()
            v_var = np.std(v_profile, axis=0).mean()

            if h_var < 8 and v_var < 8:
                continue

            kind = "linear"
            angle = 90.0 if h_var > v_var else 0.0

            # Sample gradient stops (5 evenly spaced)
            zh, zw = zone_rgb.shape[:2]
            if h_var > v_var:
                stops_px = [zone_rgb[min(zh - 1, int(i * (zh - 1) / 4)), min(zw - 1, (x0 + x1) // 2 - x0)] for i in range(5)]
            else:
                stops_px = [zone_rgb[min(zh - 1, (y0 + y1) // 2 - y0), min(zw - 1, int(i * (zw - 1) / 4))] for i in range(5)]
            stops = [{"pos": round(i / 4, 2), "hex": rgb_to_hex(p)} for i, p in enumerate(stops_px)]

            sx = 1.0 / scale
            gradients.append(GradientRegion(
                kind=kind,
                box=(int(x0 * sx), int(y0 * sx), int(x1 * sx), int(y1 * sx)),
                stops=stops,
                angle_deg=angle,
            ))
            visited[y0:y1, x0:x1] = True

    return gradients


def _spatial_gradient_score(rgb, mask) -> float:
    px = rgb.astype(np.float32)
    gx = cv2.Sobel(px, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(px, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2).mean(axis=2)
    return float(mag[mask].mean()) if mask.any() else 0.0


# ─────────────────────────────────────────── layout sections ───────────────
_SECTION_LABELS = ["header", "hero", "content", "cta", "content", "cta", "footer"]


def _extract_sections(rgb: np.ndarray, exclude: np.ndarray, scale: float) -> List[LayoutSection]:
    H, W = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Look for horizontal bands of uniform color — these are typical section dividers
    row_var = []
    for y in range(H):
        row = gray[y, ~exclude[y] if exclude[y].any() else slice(None)]
        row_var.append(float(np.std(row)) if len(row) > 10 else 0.0)
    row_var = np.array(row_var, dtype=np.float32)
    smoothed = cv2.GaussianBlur(row_var.reshape(-1, 1), (1, max(3, H // 30 * 2 + 1)), 0).ravel()

    # Find low-variance bands = section dividers
    threshold = np.percentile(smoothed, 35)
    low = smoothed < threshold
    sections: List[LayoutSection] = []
    in_section = False
    seg_start = 0
    seg_list = []

    for y in range(H):
        if not in_section and not low[y]:
            seg_start = y
            in_section = True
        elif in_section and low[y]:
            if y - seg_start > H * 0.04:
                seg_list.append((seg_start, y))
            in_section = False
    if in_section and H - seg_start > H * 0.04:
        seg_list.append((seg_start, H))

    label_map = ["header", "hero", "content", "cta", "content", "footer"]
    for i, (y0, y1) in enumerate(seg_list[:6]):
        label = label_map[min(i, len(label_map) - 1)]
        bg_rgb = _median_color(rgb[y0:y1], ~exclude[y0:y1])
        sx = 1.0 / scale
        sections.append(LayoutSection(
            label=label,
            box=(0, int(y0 * sx), int(W * sx), int(y1 * sx)),
            bg_hex=rgb_to_hex(bg_rgb),
            confidence=round(1.0 - float(smoothed[y0:y1].mean()) / max(1.0, row_var.max()), 3),
        ))

    return sections


# ─────────────────────────────────────────── design tokens ─────────────────
def _derive_tokens(shapes: List[ShapeRegion], palette: List[ColorSwatch], W: int, H: int) -> Dict:
    tokens: Dict = {}

    # Spacing: infer base unit from inter-shape gaps
    gaps = []
    sorted_s = sorted(shapes, key=lambda s: s.box[0])
    for i in range(len(sorted_s) - 1):
        gap = sorted_s[i + 1].box[0] - sorted_s[i].box[2]
        if 2 < gap < W * 0.15:
            gaps.append(gap)
    if gaps:
        base_unit = int(np.median(gaps))
        tokens["spacing_base_px"] = base_unit
        tokens["spacing_scale"] = [base_unit * m for m in [0.5, 1, 1.5, 2, 3, 4]]

    # Border radius hints
    rounded = [s for s in shapes if s.kind == "rounded_rect"]
    if rounded:
        tokens["border_radius_hint_px"] = "detected — likely 4–12 px based on shape curvature"

    # Color roles
    tokens["color_tokens"] = {s.role: s.hex for s in palette}

    # Canvas
    tokens["canvas_width_px"] = W
    tokens["canvas_height_px"] = H
    tokens["aspect_ratio"] = round(W / H, 3)

    return tokens

