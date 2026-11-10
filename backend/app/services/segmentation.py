"""Object segmentation: isolate foreground subjects as soft alpha masks.

Engine resolution order for ``DECOMPOSER_SEGMENTATION_ENGINE=auto``:
    SAM 2 (needs ``sam2`` + ``SAM2_CHECKPOINT``)  ->  rembg (U^2-Net)  ->  classical

The classical engine models the background as a smooth 2nd-order colour
surface fitted to the canvas border, thresholds the residual to seed a
GrabCut refinement, then splits the foreground into connected objects.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .common import dilate, logger, mask_bbox, module_available, resize_to_max, torch_device

WORK_SIDE = 640  # working resolution for the classical engine


@dataclass
class ObjectRegion:
    mask: np.ndarray  # bool HxW (hard mask, used for inpainting)
    alpha: np.ndarray  # uint8 HxW (soft matte, used for the cut-out)
    box: Tuple[int, int, int, int]  # (x0, y0, x1, y1) of alpha > 0, exclusive
    score: float
    label: str = "object"


@dataclass
class SegmentationResult:
    objects: List[ObjectRegion]
    engine: str
    warnings: List[str] = field(default_factory=list)


class SegmentationService:
    ENGINES = ("yunet", "sam2", "rembg", "classical")

    def __init__(self, engine: str = "auto", sam2_checkpoint: Optional[str] = None, sam2_config: str = "") -> None:
        self.preference = engine
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        self.yunet_model = Path(__file__).resolve().parent.parent / "models" / "yunet.onnx"
        self._lock = threading.Lock()
        self._backend = None
        self._failed: set = set()

    def available_engines(self) -> List[str]:
        out = []
        if self.yunet_model.exists() and hasattr(cv2, "FaceDetectorYN"):
            out.append("yunet")
        if module_available("sam2") and self.sam2_checkpoint:
            out.append("sam2")
        if module_available("rembg"):
            out.append("rembg")
        out.append("classical")
        return out

    def _load(self, warnings: List[str]):
        with self._lock:
            if self._backend is not None:
                return self._backend
            order = list(self.ENGINES)
            if self.preference in order:
                order.remove(self.preference)
                order.insert(0, self.preference)
            avail = self.available_engines()
            for name in [e for e in order if e in avail and e not in self._failed]:
                try:
                    if name == "yunet":
                        self._backend = (name, str(self.yunet_model))
                    elif name == "sam2":
                        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
                        from sam2.build_sam import build_sam2

                        sam = build_sam2(self.sam2_config, self.sam2_checkpoint, device=torch_device())
                        model = SAM2AutomaticMaskGenerator(sam, points_per_side=24, pred_iou_thresh=0.8)
                        self._backend = (name, model)
                    elif name == "rembg":
                        from rembg import new_session

                        model = new_session("u2net")
                        self._backend = (name, model)
                    else:
                        self._backend = (name, None)
                    logger.info("Segmentation engine ready: %s", name)
                    return self._backend
                except Exception as exc:  # pragma: no cover - optional deps
                    self._failed.add(name)
                    msg = f"Segmentation engine '{name}' failed to load ({exc!s}); falling back."
                    logger.warning(msg)
                    warnings.append(msg)
            self._backend = ("classical", None)
            return self._backend

    def segment(
        self,
        rgb: np.ndarray,
        exclude: Optional[np.ndarray] = None,
        max_objects: int = 8,
        min_area_ratio: float = 0.002,
    ) -> SegmentationResult:
        warnings: List[str] = []
        h, w = rgb.shape[:2]
        exclude = np.zeros((h, w), bool) if exclude is None else exclude.astype(bool)
        if max_objects <= 0:
            return SegmentationResult([], "disabled", warnings)
        name, model = self._load(warnings)
        masks_with_labels: List[Tuple[np.ndarray, str]] = []
        try:
            if name == "yunet":
                # 1. Detect people cleanly with YuNet + GrabCut
                people = _yunet_masks(str(self.yunet_model), rgb, exclude, min_area_ratio)
                combined_exclude = exclude.copy()
                for p_soft, _ in people:
                    combined_exclude |= (p_soft >= 0.5)

                # 2. Extract remaining props/decor with classical background subtraction
                props = _split_components(_classical_mask(rgb, combined_exclude), min_area_ratio)
                masks_with_labels = list(people)
                for i, pm in enumerate(props, start=1):
                    masks_with_labels.append((pm, f"Object {i}"))
            elif name == "sam2":
                masks_with_labels = [(m, "object") for m in _sam2_masks(model, rgb, min_area_ratio)]
            elif name == "rembg":
                masks_with_labels = [(m, "object") for m in _split_components(_rembg_mask(model, rgb), min_area_ratio)]
            else:
                masks_with_labels = [(m, "object") for m in _split_components(_classical_mask(rgb, exclude), min_area_ratio)]
        except Exception as exc:  # pragma: no cover
            msg = f"Segmentation engine '{name}' failed at inference ({exc!s}); using classical."
            logger.warning(msg)
            warnings.append(msg)
            name = "classical"
            masks_with_labels = [(m, "object") for m in _split_components(_classical_mask(rgb, exclude), min_area_ratio)]

        objects: List[ObjectRegion] = []
        min_px = max(16, int(max(0.005, min_area_ratio) * h * w))
        
        # Collect hard masks for detected people to prevent fragments overlapping them
        person_hard_masks = []
        for soft, label in masks_with_labels:
            if "person" in label.lower() and (soft >= 0.5).sum() >= min_px:
                person_hard_masks.append(soft >= 0.5)

        for soft, label in masks_with_labels:
            hard_pre = soft >= 0.5
            if hard_pre.any():
                overlap_text = (hard_pre & exclude).sum() / float(hard_pre.sum())
                if overlap_text > 0.25:
                    continue

            soft = soft.copy()
            is_person = "person" in label.lower()
            soft[exclude] = 0.0  # text belongs to the text layers, never baked into subject cutouts
            hard = soft >= 0.5
            area_px = int(hard.sum())
            if area_px < min_px:
                continue

            alpha = (np.clip(soft, 0, 1) * 255).astype(np.uint8)
            box = mask_bbox(alpha > 0)
            if box is None:
                continue

            ratio = area_px / float(h * w)
            # Giant non-person blobs (> 18% of image or wide background panels) are background, not foreground objects
            if not is_person:
                bw_box = box[2] - box[0]
                bh_box = box[3] - box[1]
                if ratio > 0.18:
                    continue
                if bw_box > 0.60 * w:
                    continue
                if bh_box > 0.60 * h:
                    continue
                if bw_box > 0.75 * w and box[3] >= h - 5:
                    continue

            # Check overlap with any person mask to eliminate sub-body fragments (ties, collars, pockets)
            if not is_person and person_hard_masks:
                overlap_person = max((hard & pm).sum() / float(area_px) for pm in person_hard_masks)
                if overlap_person > 0.25:
                    continue

            score = float(soft[hard].mean()) if hard.any() else 0.0
            objects.append(ObjectRegion(mask=hard, alpha=alpha, box=box, score=min(1.0, score), label=label))

        objects.sort(key=lambda o: int(o.mask.sum()), reverse=True)
        objects = objects[:max_objects]
        return SegmentationResult(objects, name, warnings)
        # Paint order: larger objects first (bottom), so small ones stay visible.
        return SegmentationResult(objects, name, warnings)


# ======================================================================= engines
def _feather(hard: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    soft = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), sigma)
    soft[hard] = np.maximum(soft[hard], 0.5)
    return soft


def _split_components(mask: np.ndarray, min_area_ratio: float) -> List[np.ndarray]:
    """Split a (soft or hard) full-image foreground mask into per-object soft masks."""
    h, w = mask.shape
    soft = mask.astype(np.float32)
    if soft.max() > 1.0:
        soft /= 255.0
    hard = (soft >= 0.5).astype(np.uint8)

    # Detach full-bleed bottom ground / earth horizon from floating or seated foreground subjects
    bot_start = int(0.82 * h)
    bot_region = hard[bot_start:, :]
    n_bot, labels_bot, stats_bot, _ = cv2.connectedComponentsWithStats(bot_region, connectivity=8)
    ground_mask = np.zeros_like(hard)
    for i in range(1, n_bot):
        bw = stats_bot[i, cv2.CC_STAT_WIDTH]
        if bw > 0.65 * w:
            ground_mask[bot_start:, :][labels_bot == i] = 1

    isolated_hard = hard & ~ground_mask

    n, labels, stats, _ = cv2.connectedComponentsWithStats(isolated_hard, connectivity=8)
    min_px = max(16, int(min_area_ratio * h * w))
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_px:
            continue
        comp = labels == i
        bx = stats[i, cv2.CC_STAT_LEFT]
        by = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        # Filter out thin line-art filigree and background swirls (e.g. mandala line art)
        rect_fill = area / float(max(1, bw * bh))
        cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = cnts[0]
            hull = cv2.convexHull(c)
            solidity = area / max(1.0, cv2.contourArea(hull))
            perim = cv2.arcLength(c, True)
            compactness = (perim * perim) / max(1.0, area)
            if (rect_fill < 0.28 or solidity < 0.50) and compactness > 45.0:
                continue

        zone = dilate(comp, 3)  # keep the soft fringe that belongs to this component
        out.append(np.where(zone, soft, 0.0).astype(np.float32))
    return out


def _detect_silhouette(rgb: np.ndarray, exclude: np.ndarray) -> List[Tuple[np.ndarray, str]]:
    H, W = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cx0, cx1 = int(0.15 * W), int(0.85 * W)
    cy0, cy1 = int(0.12 * H), int(0.90 * H)
    sub = gray[cy0:cy1, cx0:cx1]

    for thr in [35, 45, 55, 65]:
        dark = (sub < thr).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if 0.04 * H * W < area < 0.45 * H * W:
                bw = stats[i, cv2.CC_STAT_WIDTH]
                bh = stats[i, cv2.CC_STAT_HEIGHT]
                if bw > 0.22 * W and bh > 0.30 * H:
                    comp = (labels == i)
                    dil = cv2.dilate(comp.astype(np.uint8), np.ones((9, 9), np.uint8))
                    mask = np.zeros((H, W), np.float32)
                    mask[cy0:cy1, cx0:cx1] = dil.astype(np.float32)
                    soft = _feather(mask > 0.5, 1.5)
                    return [(soft, "Silhouette 1")]
    return []


def _yunet_masks(
    model_path: str, rgb: np.ndarray, exclude: np.ndarray, min_area_ratio: float
) -> List[Tuple[np.ndarray, str]]:
    H, W = rgb.shape[:2]
    p = Path(model_path)
    if not p.exists() or not hasattr(cv2, "FaceDetectorYN"):
        return _detect_silhouette(rgb, exclude)
    try:
        det = cv2.FaceDetectorYN.create(str(p), "", (W, H), score_threshold=0.45)
        faces = det.detect(rgb)[1]
    except Exception as exc:
        logger.warning("YuNet face detection error: %s", exc)
        return _detect_silhouette(rgb, exclude)

    if faces is None or len(faces) == 0:
        return _detect_silhouette(rgb, exclude)

    masks_with_labels: List[Tuple[np.ndarray, str]] = []
    person_idx = 1
    for f in faces:
        fx, fy, fw, fh = [int(v) for v in f[:4]]
        # Expand around face to enclose full human body (including wide arms in lotus / seated pose)
        left = max(0, fx - int(fw * 3.2))
        top = max(0, fy - int(fh * 0.45))
        right = min(W, fx + fw + int(fw * 3.5))
        bottom = min(H, fy + fh + int(fh * 4.0))
        # Extend to bottom of canvas for standing / seated human figures so torso/arms/lap are not cut off
        if bottom > 0.70 * H or (fy + fh * 3.0) > 0.65 * H:
            bottom = H
        bw = right - left
        bh = bottom - top
        if bw < 80 or bh < 120:
            continue

        crop = rgb[top:bottom, left:right]
        ch, cw = crop.shape[:2]
        scale = min(1.0, 480.0 / max(ch, cw))
        small = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))))
        sh, sw = small.shape[:2]

        gc = np.full((sh, sw), cv2.GC_BGD, np.uint8)
        rx = max(1, int(0.02 * sw))
        ry = max(1, int(0.02 * sh))
        rw = min(sw - rx - 1, int(0.96 * sw))
        # If figure extends to bottom of canvas, include the entire bottom row in foreground calculation
        rh = sh - ry if bottom == H else min(sh - ry - 1, int(0.94 * sh))
        gc[ry:ry + rh, rx:rx + rw] = cv2.GC_PR_FGD

        fx_s = int((fx - left) * scale)
        fy_s = int((fy - top) * scale)
        fw_s = int(fw * scale)
        fh_s = int(fh * scale)
        if fw_s > 0 and fh_s > 0:
            gc[max(0, fy_s):min(sh, fy_s + fh_s), max(0, fx_s):min(sw, fx_s + fw_s)] = cv2.GC_FGD
            # Torso seed directly beneath the face down to bottom (covering wide lotus pose)
            torso_top = min(sh - 1, fy_s + fh_s)
            gc[torso_top:sh, max(0, fx_s - int(fw_s * 1.8)):min(sw, fx_s + fw_s + int(fw_s * 1.8))] = cv2.GC_PR_FGD

        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(small, gc, (rx, ry, rw, rh), bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
            cand = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
        except Exception:
            cand = np.ones((sh, sw), bool)

        cand_full = cv2.resize(cand.astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST) > 0
        soft_crop = _feather(cand_full, 1.2)
        full_mask = np.zeros((H, W), np.float32)
        full_mask[top:bottom, left:right] = soft_crop

        masks_with_labels.append((full_mask, f"Person {person_idx}"))
        person_idx += 1

    return masks_with_labels


def _sam2_masks(generator, rgb: np.ndarray, min_area_ratio: float) -> List[np.ndarray]:  # pragma: no cover
    h, w = rgb.shape[:2]
    total = float(h * w)
    proposals = []
    for m in generator.generate(rgb):
        seg = m["segmentation"].astype(bool)
        area = seg.sum() / total
        if area < min_area_ratio or area > 0.6:
            continue
        border = np.concatenate([seg[0], seg[-1], seg[:, 0], seg[:, -1]]).mean()
        if border > 0.3:  # mostly background-like regions touch the frame
            continue
        proposals.append((float(m.get("predicted_iou", 0.9)) * area, seg))
    proposals.sort(key=lambda p: p[0], reverse=True)
    chosen: List[np.ndarray] = []
    taken = np.zeros((h, w), bool)
    for _, seg in proposals:
        if (seg & taken).sum() > 0.5 * seg.sum():  # mostly inside an already chosen object
            continue
        chosen.append(seg)
        taken |= seg
    return [_feather(s) for s in chosen]


def _rembg_mask(session, rgb: np.ndarray) -> np.ndarray:  # pragma: no cover
    from PIL import Image
    from rembg import remove

    out = remove(Image.fromarray(rgb), session=session, only_mask=True)
    return np.asarray(out.convert("L"), dtype=np.float32) / 255.0


def _poly_features(xs: np.ndarray, ys: np.ndarray, w: int, h: int) -> np.ndarray:
    x = xs.astype(np.float32) / max(1, w - 1) - 0.5
    y = ys.astype(np.float32) / max(1, h - 1) - 0.5
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)


def _classical_mask(rgb: np.ndarray, exclude: np.ndarray) -> np.ndarray:
    """Return a soft (0..1 float) full-resolution foreground mask."""
    H, W = rgb.shape[:2]
    small, scale = resize_to_max(rgb, WORK_SIDE)
    h, w = small.shape[:2]
    excl = cv2.resize(exclude.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    smallf = small.astype(np.float32)

    b = max(2, int(0.03 * min(h, w)))
    border = np.zeros((h, w), bool)
    border[:b, :] = border[-b:, :] = True
    border[:, :b] = border[:, -b:] = True
    sample = border & ~excl
    ys, xs = np.nonzero(sample)
    if len(xs) < 20:
        return np.zeros((H, W), np.float32)

    # Fit a smooth colour surface to the border => background model.
    coef, *_ = np.linalg.lstsq(_poly_features(xs, ys, w, h), smallf[ys, xs], rcond=None)
    gy, gx = np.mgrid[0:h, 0:w]
    bg = (_poly_features(gx.ravel(), gy.ravel(), w, h) @ coef).reshape(h, w, 3)
    resid = np.linalg.norm(smallf - bg, axis=2)

    border_res = resid[sample]
    thr = min(45.0, max(22.0, float(np.percentile(border_res, 75)) * 1.4))
    fg = (resid > thr) & ~excl
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    if fg.sum() < 0.001 * h * w or fg.mean() > 0.95:
        return np.zeros((H, W), np.float32)

    # GrabCut refinement seeded by the residual map.
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc[fg] = cv2.GC_PR_FGD
    gc[(resid > 2.5 * thr) & ~excl] = cv2.GC_FGD
    gc[border & ~fg] = cv2.GC_BGD
    refined = fg
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
        cand = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
        # Accept refinement only if it does not wildly disagree with the seed.
        if 0.3 * fg.sum() <= cand.sum() <= 3.0 * fg.sum() and cand.mean() < 0.95:
            refined = cand
    except cv2.error as exc:  # pragma: no cover
        logger.debug("GrabCut skipped: %s", exc)

    soft = refined.astype(np.float32)
    if scale < 1.0:
        soft = cv2.resize(soft, (W, H), interpolation=cv2.INTER_LINEAR)
    soft = cv2.GaussianBlur(soft, (0, 0), max(0.8, 0.6 / max(scale, 1e-3) * 0.5))
    soft[soft < 0.02] = 0.0
    return soft

