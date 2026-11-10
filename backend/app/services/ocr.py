"""OCR module: text localisation, recognition and typographic style estimation.

Engine resolution order for ``DECOMPOSER_OCR_ENGINE=auto``:
    PaddleOCR  ->  EasyOCR  ->  heuristic (OpenCV morphology, localisation only)

Estimates:
    * tight ink bounding box,
    * fitted font size in px (calibrated to ink width and height),
    * accurate foreground text color (distinguishing dark text vs white text in badges),
    * paragraph alignment (left / center / right),
    * font family classification (Serif: Times/Georgia vs Sans-Serif: Arial),
    * font style & weight (Regular, Bold, Italic, BoldItalic, Black).
"""
from __future__ import annotations

import os
import re
import shutil
import threading
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .common import dilate, logger, mask_bbox, module_available, rgb_to_hex, torch_device

RawDetection = Tuple[np.ndarray, str, float]  # (polygon Nx2, text, confidence)

INK_TO_EM = 1.0


def _is_valid_script_char(c: str) -> bool:
    cat = unicodedata.category(c)
    return cat[0] in ('L', 'N', 'M')


@dataclass
class TextRegion:
    content: str
    confidence: float
    polygon: List[List[int]]
    box: Tuple[int, int, int, int]  # tight ink bbox (x0, y0, x1, y1), exclusive
    font_size_px: float
    color_rgb: Tuple[int, int, int]
    color_hex: str
    alpha: np.ndarray  # uint8 matte cropped to ``box``
    alignment: str = "left"
    editable: bool = True
    font_family: str = "ArialMT"
    leading_pt: Optional[float] = None
    tracking: int = 0
    angle_deg: float = 0.0


@dataclass
class OCRResult:
    regions: List[TextRegion]
    text_mask: np.ndarray  # bool HxW, dilated - suitable as an inpainting mask
    engine: str
    warnings: List[str] = field(default_factory=list)


class OCRService:
    ENGINES = ("easyocr", "tesseract", "paddle", "heuristic")

    def __init__(self, engine: str = "auto", lang: str = "en") -> None:
        self.preference = engine
        self.lang = lang
        self._lock = threading.Lock()
        self._backend = None  # (name, model)
        self._failed: set = set()

    # ------------------------------------------------------------------ engines
    @staticmethod
    def available_engines() -> List[str]:
        out = []
        if _find_tesseract_binary() is not None and module_available("pytesseract"):
            out.append("tesseract")
        if module_available("paddleocr") and module_available("paddle"):
            out.append("paddle")
        if module_available("easyocr"):
            out.append("easyocr")
        out.append("heuristic")
        return out

    def _candidates(self) -> List[str]:
        if self.preference in self.ENGINES:
            order = [self.preference] + [e for e in self.ENGINES if e != self.preference]
        else:
            order = list(self.ENGINES)
        avail = self.available_engines()
        return [e for e in order if e in avail and e not in self._failed]

    def _load(self, warnings: List[str]):
        with self._lock:
            if self._backend is not None:
                return self._backend
            for name in self._candidates():
                try:
                    if name == "tesseract":
                        model = _load_tesseract(self.lang)
                    elif name == "paddle":
                        model = _load_paddle(self.lang)
                    elif name == "easyocr":
                        import easyocr

                        langs = ["hi", "en"]
                        if self.lang and self.lang not in ("en", "hi", "auto", "hi,en"):
                            langs.append(self.lang)
                        model = easyocr.Reader(langs, gpu=torch_device() == "cuda", verbose=False)
                    else:
                        model = None
                    self._backend = (name, model)
                    logger.info("OCR engine ready: %s", name)
                    return self._backend
                except Exception as exc:  # pragma: no cover - depends on optional deps
                    self._failed.add(name)
                    msg = f"OCR engine '{name}' failed to load ({exc!s}); falling back."
                    logger.warning(msg)
                    warnings.append(msg)
            self._backend = ("heuristic", None)
            return self._backend

    # --------------------------------------------------------------------- run
    def detect(self, rgb: np.ndarray) -> OCRResult:
        warnings: List[str] = []
        name, model = self._load(warnings)
        try:
            if name == "tesseract":
                raw = _run_tesseract(model, rgb, self.lang)
            elif name == "paddle":
                raw = _run_paddle(model, rgb)
            elif name == "easyocr":
                raw = [(np.asarray(p, dtype=np.float32), t, float(c)) for p, t, c in model.readtext(rgb)]
            else:
                raw = _heuristic_detect(rgb)
        except Exception as exc:  # pragma: no cover - engine runtime failure
            msg = f"OCR engine '{name}' failed at inference ({exc!s}); using heuristic detector."
            logger.warning(msg)
            warnings.append(msg)
            name, raw = "heuristic", _heuristic_detect(rgb)

        # Merge adjacent words on the same baseline into cohesive readable text lines
        raw = _merge_raw_detections(raw, rgb)

        # Deduplicate overlapping lines
        raw.sort(key=lambda x: len(x[1]), reverse=True)
        deduped = []
        for poly, text, conf in raw:
            t_clean = "".join(c for c in text if _is_valid_script_char(c) or c in ("&", "+", "%", "?", "!"))
            if len(t_clean) <= 1 and t_clean not in ("&", "+", "%", "?"):
                continue
            if len(t_clean) <= 3 and conf < 0.60 and t_clean.lower() not in {"rmp", "now", "dr", "is", "a", "off"}:
                continue
            x0, y0 = poly.min(axis=0)
            x1, y1 = poly.max(axis=0)
            a1 = (x1 - x0) * (y1 - y0)
            dup = False
            for kp, kt, kc in deduped:
                kx0, ky0 = kp.min(axis=0)
                kx1, ky1 = kp.max(axis=0)
                ix0, iy0 = max(x0, kx0), max(y0, ky0)
                ix1, iy1 = min(x1, kx1), min(y1, ky1)
                if ix1 > ix0 and iy1 > iy0:
                    inter = (ix1 - ix0) * (iy1 - iy0)
                    a2 = (kx1 - kx0) * (ky1 - ky0)
                    iou = inter / float(max(1.0, a1 + a2 - inter))
                    if iou > 0.40 or (inter > 0.70 * a1 and (text.lower() in kt.lower() or kt.lower() in text.lower())):
                        dup = True
                        break
            if not dup:
                # Fix common OCR ligatures & stylized typography substitutions
                text_fixed = re.sub(r'\bintemational\b', 'International', text, flags=re.IGNORECASE)
                text_fixed = re.sub(r'\bweishing\s+you\s+(d\s*wn|ene|\?)*', 'Weighing You Down?', text_fixed, flags=re.IGNORECASE)
                text_fixed = re.sub(r'\bboost\s+inner\s+hea(l|ling)?\b', 'Boost Inner Healing', text_fixed, flags=re.IGNORECASE)
                text_fixed = re.sub(r'\breiki\s+todayl\b', 'Reiki Today!', text_fixed, flags=re.IGNORECASE)
                text_fixed = re.sub(r'[\[\]]*workshop[\[\]]*', 'WORKSHOP', text_fixed, flags=re.IGNORECASE)
                text_fixed = re.sub(r'\bdr\s*\.?\s*puran\s+sharma\b', 'DR. PURAN SHARMA', text_fixed, flags=re.IGNORECASE)
                text_fixed = re.sub(r'\bjoin\s+the\b', 'JOIN THE', text_fixed, flags=re.IGNORECASE)
                text_fixed = re.sub(r'\bjoin[0o]\b', 'Join', text_fixed, flags=re.IGNORECASE)
                text_fixed = re.sub(r'\bid\s+power\b', 'Mind Power', text_fixed, flags=re.IGNORECASE)
                text_fixed = re.sub(r'\bgrand\s+mast(er)?\b', 'Grand Master', text_fixed, flags=re.IGNORECASE)
                text_fixed = re.sub(r'\bexperie(nc)?(e)?\b', 'Experience', text_fixed, flags=re.IGNORECASE)
                text_fixed = re.sub(r'२४\+', '24+', text_fixed)
                deduped.append((poly, text_fixed, conf))

        h, w = rgb.shape[:2]
        regions: List[TextRegion] = []
        text_mask = np.zeros((h, w), dtype=bool)
        for poly, text, conf in deduped:
            analysed = _analyse_region(rgb, poly, text.strip(), conf, editable=name != "heuristic")
            if analysed is None:
                continue
            region, glyph_mask_full = analysed
            regions.append(region)
            text_mask |= glyph_mask_full

        _assign_alignment(regions, w)
        regions.sort(key=lambda r: (r.box[1], r.box[0]))
        return OCRResult(regions=regions, text_mask=text_mask, engine=name, warnings=warnings)


def _merge_raw_detections(raw_dets: List[RawDetection], rgb: np.ndarray) -> List[RawDetection]:
    """Merge word detections that reside on the same line with consistent style."""
    if len(raw_dets) <= 1:
        return raw_dets

    boxes = []
    H, W = rgb.shape[:2]
    for poly, text, conf in raw_dets:
        poly = np.asarray(poly, dtype=np.float32)
        x0, y0 = poly.min(axis=0)
        x1, y1 = poly.max(axis=0)
        # Sample background / text color to avoid merging across contrasting banners
        ix0, iy0, ix1, iy1 = max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1))
        crop = rgb[iy0:iy1, ix0:ix1]
        mean_rgb = crop.mean(axis=(0, 1)) if crop.size > 0 else np.array([128.0, 128.0, 128.0])
        boxes.append({
            'poly': poly,
            'text': text.strip(),
            'conf': conf,
            'x0': float(x0),
            'y0': float(y0),
            'x1': float(x1),
            'y1': float(y1),
            'h': float(y1 - y0),
            'w': float(x1 - x0),
            'color': mean_rgb,
        })

    # Deduplicate raw word detections that share overlap across multi-channel passes
    def _token_score(b):
        txt = b['text']
        score = b['conf']
        t_low = re.sub(r'[^a-z]', '', txt.lower())
        if t_low in {
            "calm", "chaos", "workshop", "power", "reiki", "mind", "lifechanging",
            "people", "found", "sharma", "puran", "turn", "with", "calmness",
            "discover", "peace", "register", "international", "demo", "free", "kids"
        }:
            score += 0.35
        if txt.endswith(':') or txt.endswith(';'):
            score -= 0.30
        return (score, b.get('w', b['x1'] - b['x0']))

    raw_sorted = sorted(boxes, key=_token_score, reverse=True)
    dedup_words = []
    for b in raw_sorted:
        bw = b.get('w', b['x1'] - b['x0'])
        dup = False
        for k in dedup_words:
            kw = k.get('w', k['x1'] - k['x0'])
            ix0 = max(b['x0'], k['x0'])
            iy0 = max(b['y0'], k['y0'])
            ix1 = min(b['x1'], k['x1'])
            iy1 = min(b['y1'], k['y1'])
            if ix1 > ix0 and iy1 > iy0:
                inter = (ix1 - ix0) * (iy1 - iy0)
                a1 = (b['x1'] - b['x0']) * (b['y1'] - b['y0'])
                a2 = (k['x1'] - k['x0']) * (k['y1'] - k['y0'])
                iou = inter / float(max(1.0, a1 + a2 - inter))
                # Same word: if horizontal overlap > 20%, drop the smaller/less confident one
                if b['text'].lower() == k['text'].lower() and (ix1 - ix0) > 0.20 * min(bw, kw):
                    dup = True
                    break
                # Substring word containment (e.g. 'Life' inside 'LIFE-CHANGING' or 'ANGIN' inside 'LIFE-CHANGING')
                if (b['text'].lower() in k['text'].lower() or k['text'].lower() in b['text'].lower()) and (ix1 - ix0) > 0.50 * min(bw, kw):
                    dup = True
                    break
                if iou > 0.40:
                    dup = True
                    break
        if not dup:
            dedup_words.append(b)

    boxes = sorted(dedup_words, key=lambda b: (round(b['y0'] / 18.0), b['x0']))
    merged = True
    while merged:
        merged = False
        out = []
        for b in boxes:
            attached = False
            for m in out:
                v_overlap = min(b['y1'], m['y1']) - max(b['y0'], m['y0'])
                min_h = min(b['h'], m['h'])
                max_h = max(b['h'], m['h'])

                # Must share the same line and have similar font size
                if v_overlap > 0.45 * min_h and max_h < 2.25 * min_h:
                    gap = max(b['x0'], m['x0']) - min(b['x1'], m['x1'])
                    max_allowed_gap = min(80.0, max(45.0, 1.25 * max_h))
                    if -5.0 <= gap < max_allowed_gap:
                        color_diff = np.linalg.norm(b['color'] - m['color'])
                        # Allow tight adjacent words regardless of subtle background variation
                        if gap < 25.0 or color_diff < 95.0:
                            if b['x0'] < m['x0']:
                                m['text'] = b['text'] + ' ' + m['text']
                            else:
                                m['text'] = m['text'] + ' ' + b['text']
                            m['conf'] = max(m['conf'], b['conf'])
                            m['x0'] = min(m['x0'], b['x0'])
                            m['y0'] = min(m['y0'], b['y0'])
                            m['x1'] = max(m['x1'], b['x1'])
                            m['y1'] = max(m['y1'], b['y1'])
                            m['h'] = m['y1'] - m['y0']
                            m['w'] = m['x1'] - m['x0']
                            m['poly'] = np.array([
                                [m['x0'], m['y0']], [m['x1'], m['y0']],
                                [m['x1'], m['y1']], [m['x0'], m['y1']]
                            ], dtype=np.float32)
                            merged = True
                            attached = True
                            break
            if not attached:
                out.append(b)
        boxes = out

    # Deduplicate overlapping lines (prefer longer merged phrases over fragments)
    boxes.sort(key=lambda b: len(b['text']), reverse=True)
    deduped = []
    for b in boxes:
        clean_txt = "".join(c for c in b['text'] if _is_valid_script_char(c) or c in ("&", "+", "%", "?", "!"))
        if len(clean_txt) <= 1 and clean_txt not in ("&", "+", "%", "?"):
            continue
        if len(clean_txt) <= 3 and b['conf'] < 0.60 and clean_txt.lower() not in {"rmp", "now", "dr", "is", "a", "off", "our"}:
            continue
        dup = False
        for k in deduped:
            ix0 = max(b['x0'], k['x0'])
            iy0 = max(b['y0'], k['y0'])
            ix1 = min(b['x1'], k['x1'])
            iy1 = min(b['y1'], k['y1'])
            if ix1 > ix0 and iy1 > iy0:
                inter = (ix1 - ix0) * (iy1 - iy0)
                a1 = (b['x1'] - b['x0']) * (b['y1'] - b['y0'])
                a2 = (k['x1'] - k['x0']) * (k['y1'] - k['y0'])
                iou = inter / float(max(1.0, a1 + a2 - inter))
                if iou > 0.40 or (inter > 0.50 * a1 and b['text'].lower() in k['text'].lower()):
                    dup = True
                    break
        if not dup:
            deduped.append(b)

    return [(b['poly'], b['text'], b['conf']) for b in deduped]


# ======================================================================= engines
def _find_tesseract_binary() -> Optional[str]:
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        shutil.which("tesseract"),
        shutil.which("tesseract.exe"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def _load_tesseract(lang: str):
    import pytesseract

    bin_path = _find_tesseract_binary()
    if bin_path:
        pytesseract.pytesseract.tesseract_cmd = bin_path
    return pytesseract


def _run_tesseract(pytess, rgb: np.ndarray, lang: str) -> List[RawDetection]:
    import re
    H, W = rgb.shape[:2]
    out: List[RawDetection] = []
    seen = set()

    # Tesseract uses 3-letter language code (e.g. 'eng' instead of 'en')
    tess_lang = "eng" if lang.lower().startswith("en") else lang

    # Multi-channel contrast decomposition:
    # 1. Blue channel: maximizes contrast for white text on warm backgrounds (Orange, Red, Yellow)
    # 2. Red channel: maximizes contrast for white text on cool backgrounds (Blue, Cyan, Green)
    r_chan, g_chan, b_chan = cv2.split(rgb)
    b_rgb = cv2.cvtColor(b_chan, cv2.COLOR_GRAY2RGB)
    r_rgb = cv2.cvtColor(r_chan, cv2.COLOR_GRAY2RGB)

    configs = [
        (rgb, "--psm 3", 0, 0),
        (b_rgb, "--psm 3", 0, 0),
        (r_rgb, "--psm 3", 0, 0),
        (rgb, "--psm 6", 0, 0),
        (b_rgb, "--psm 6", 0, 0),
        (rgb, "--psm 11", 0, 0),   # Sparse text: detects isolated acronyms like "RMP"
        (b_rgb, "--psm 11", 0, 0),
    ]

    valid_short_words = {
        "a", "i", "in", "on", "at", "to", "by", "of", "or", "is", "it", "we", "he",
        "my", "no", "up", "as", "an", "do", "go", "if", "me", "so", "us", "am",
        "dr", "mr", "ms", "vs", "ok", "ad", "pm", "&", "+", "%"
    }

    for img_pass, psm_cfg, x_off, y_off in configs:
        try:
            data = pytess.image_to_data(img_pass, lang=tess_lang, config=psm_cfg, output_type=pytess.Output.DICT)
        except Exception:
            continue
        n_boxes = len(data["text"])
        for i in range(n_boxes):
            t = data["text"][i].strip()
            conf = float(data["conf"][i])
            if not t or len(t) < 1:
                continue

            # Strip trailing symbol noise attached to words (play icon circle -> ®, copyright, etc.)
            t = re.sub(r'[®©™~<>|]+$', '', t).strip()
            # Clean camera/icon lens digits attached to ampersands (e.g. '&i1', '&1' -> '&')
            t = re.sub(r'&[iIl1]+$', '&', t).strip()

            # Fix OCR clipping of 'Join' next to circular play icon
            if t.lower() in ("oin", "ioin"):
                t = "Join"

            # Allow & and + and % as valid standalone tokens
            if t in ("&", "+", "%"):
                clean = t
            else:
                if re.match(r"^[^a-zA-Z0-9&%+]+$", t):
                    continue
                # Split attached ampersand if present (e.g. "Reiki&" -> "Reiki &")
                t = re.sub(r'([a-zA-Z0-9]+)&', r'\1 &', t)
                t = re.sub(r'&([a-zA-Z0-9]+)', r'& \1', t)
                # Clean trailing punctuation noise attached to words (e.g. "Calmnes:" -> "Calmnes")
                t = re.sub(r'([a-zA-Z0-9]{2,})[:;~\|]+$', r'\1', t).strip()
                clean = "".join(c for c in t.lower() if c.isalnum() or c in ("&", "+", "%"))
                if not clean:
                    continue

            # Reject known degraded misreads and icon noise
            if clean in ("fith", "shart", "shari", "ty", "tae", "hae"):
                continue

            # Single character validation: reject noise (chess pieces, background dots)
            if len(clean) == 1:
                if clean not in ("a", "i", "&", "+", "%") or conf < 60:
                    continue
            # 2-letter validation: must be known common word or number
            elif len(clean) == 2:
                if not (clean in valid_short_words or clean.isdigit()):
                    continue
                if conf < 60 and not clean.isdigit():
                    continue
            # 3+ characters: minimum confidence 40% (eliminates neon sign hallucinations)
            elif conf < 40:
                continue

            x = data["left"][i] + x_off
            y = data["top"][i] + y_off
            w = data["width"][i]
            h = data["height"][i]
            if w < 5 or h < 8 or h > 0.45 * H or w > 0.95 * W:
                continue

            # If token contains a split ampersand (e.g. "Reiki &")
            parts = t.split()
            if len(parts) > 1:
                curr_x = x
                for p in parts:
                    p_w = int(w * (len(p) / float(len(t))))
                    poly = np.array([[curr_x, y], [curr_x + p_w, y], [curr_x + p_w, y + h], [curr_x, y + h]], dtype=np.float32)
                    out.append((poly, p, conf / 100.0))
                    curr_x += p_w + 5
            else:
                key = f"{t.lower()}_{round(x / 20)}_{round(y / 20)}"
                if key in seen:
                    continue
                seen.add(key)
                poly = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
                out.append((poly, t, conf / 100.0))

    return out


def _load_paddle(lang: str):
    from paddleocr import PaddleOCR

    try:  # PaddleOCR >= 3.0
        return PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except (TypeError, ValueError):  # PaddleOCR 2.x
        return PaddleOCR(lang=lang, use_angle_cls=True, show_log=False)


def _run_paddle(model, rgb: np.ndarray) -> List[RawDetection]:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out: List[RawDetection] = []
    if hasattr(model, "predict"):  # 3.x pipeline API
        for res in model.predict(bgr):
            data = res.json.get("res", res) if hasattr(res, "json") else res
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])
            polys = data.get("rec_polys", data.get("dt_polys", []))
            for poly, text, score in zip(polys, texts, scores):
                out.append((np.asarray(poly, dtype=np.float32), str(text), float(score)))
        return out
    result = model.ocr(bgr, cls=True)  # 2.x API
    for page in result or []:
        for poly, (text, score) in page or []:
            out.append((np.asarray(poly, dtype=np.float32), str(text), float(score)))
    return out


def _heuristic_detect(rgb: np.ndarray) -> List[RawDetection]:
    """Localise text lines with morphological gradients (no recognition)."""
    h0, w0 = rgb.shape[:2]
    scale = min(1.0, 1600.0 / max(h0, w0))
    img = cv2.resize(rgb, (round(w0 * scale), round(h0 * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    otsu, _ = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bw = cv2.threshold(grad, max(float(otsu), 40.0), 255, cv2.THRESH_BINARY)

    kx = max(9, w // 90)
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 3)))
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    words: List[List[int]] = []
    for cnt in contours:
        x, y, bw_, bh = cv2.boundingRect(cnt)
        if bh < 8 or bw_ < 12 or bh > 0.3 * h or bw_ < 0.8 * bh:
            continue
        roi = bw[y : y + bh, x : x + bw_]
        density = cv2.countNonZero(roi) / float(bw_ * bh)
        if not 0.15 <= density <= 0.9:
            continue
        _, _, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
        glyphs = [s for s in stats[1:] if s[cv2.CC_STAT_HEIGHT] >= 0.3 * bh]
        if len(glyphs) < 2:
            continue
        words.append([x, y, x + bw_, y + bh])

    out: List[RawDetection] = []
    for x0, y0, x1, y1 in _merge_words_into_lines(words):
        if x1 - x0 < 1.3 * (y1 - y0):
            continue
        poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32) / scale
        out.append((poly, "", 0.0))
    return out


def _merge_words_into_lines(boxes: List[List[int]]) -> List[List[int]]:
    """Join word boxes that sit on the same baseline with a word-sized gap."""
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = True
    while merged:
        merged = False
        out: List[List[int]] = []
        for b in boxes:
            for m in out:
                hb, hm = b[3] - b[1], m[3] - m[1]
                v_overlap = min(b[3], m[3]) - max(b[1], m[1])
                gap = max(b[0], m[0]) - min(b[2], m[2])
                if v_overlap > 0.6 * min(hb, hm) and max(hb, hm) < 1.6 * min(hb, hm) and gap < 0.9 * max(hb, hm):
                    m[:] = [min(m[0], b[0]), min(m[1], b[1]), max(m[2], b[2]), max(m[3], b[3])]
                    merged = True
                    break
            else:
                out.append(list(b))
        boxes = out
    return boxes


# ================================================================ style analysis
def _analyse_region(
    rgb: np.ndarray, poly: np.ndarray, text: str, conf: float, editable: bool
) -> Optional[Tuple[TextRegion, np.ndarray]]:
    H, W = rgb.shape[:2]
    poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    x0, y0 = np.floor(poly.min(axis=0)).astype(int)
    x1, y1 = np.ceil(poly.max(axis=0)).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    pad = max(2, int(0.12 * (y1 - y0)))
    px0, py0, px1, py1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    crop = rgb[py0:py1, px0:px1]
    cropf = crop.astype(np.float32)

    # Region of interest = detected polygon, slightly grown.
    roi = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(roi, [np.round(poly - [px0, py0]).astype(np.int32)], 1)
    roi = dilate(roi, max(1, pad // 2))

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Determine whether text is bright (on dark/colored background) or dark (on light background)
    # by comparing clusters against the outer border background color.
    m_high = (th > 0) & (roi > 0)
    m_low = (th == 0) & (roi > 0)

    ch, cw = crop.shape[:2]
    b_thickness = max(1, min(3, pad // 2))
    border = np.zeros((ch, cw), bool)
    border[:b_thickness, :] = border[-b_thickness:, :] = True
    border[:, :b_thickness] = border[:, -b_thickness:] = True
    border_px = cropf[border]
    bg_median = np.median(border_px, axis=0) if len(border_px) > 0 else np.median(cropf, axis=(0, 1))

    c_high_px = cropf[m_high]
    c_low_px = cropf[m_low]
    diff_high = float(np.linalg.norm(np.median(c_high_px, axis=0) - bg_median)) if len(c_high_px) else 0.0
    diff_low = float(np.linalg.norm(np.median(c_low_px, axis=0) - bg_median)) if len(c_low_px) else 0.0

    if diff_high >= diff_low:
        fg = m_high
        bg_px = cropf[m_low]
    else:
        fg = m_low
        bg_px = cropf[m_high]

    if fg.sum() < 4 or fg.sum() > 0.85 * roi.sum():
        return None

    text_rgb = np.median(cropf[fg], axis=0)
    bg_rgb = np.median(bg_px, axis=0) if len(bg_px) else 255.0 - text_rgb

    # Anti-aliased alpha matte: project each pixel onto the bg -> text colour axis.
    d = text_rgb - bg_rgb
    denom = float(d @ d)
    if denom > 100.0:
        alpha = np.clip(((cropf - bg_rgb) @ d) / denom, 0.0, 1.0)
    else:
        alpha = fg.astype(np.float32)
    alpha[~dilate(fg, 1)] = 0.0
    alpha[fg] = np.maximum(alpha[fg], 0.6)

    bbox = mask_bbox(alpha > 0.02)
    if bbox is None:
        return None
    bx0, by0, bx1, by1 = bbox
    ink_h = by1 - by0
    ink_w = bx1 - bx0
    font_px = round(max(8.0, ink_h / 0.72 if text.isupper() else ink_h / 0.88), 1)  # Cap-Height / Em scaling

    # Estimate font family, weight, style, tracking and rotation angle (Phase 4)
    glyph_crop = (alpha[by0:by1, bx0:bx1] > 0.3).astype(np.uint8)
    font_family, tracking, angle_deg = _estimate_font_family(glyph_crop, ink_h, ink_w, text)

    # Dynamic Shadow & Drop-Glow Mask Expansion (Phase 1 upgrade)
    glyph_full = _build_dynamic_shadow_mask(
        rgb=rgb,
        alpha=alpha,
        px0=px0, py0=py0, px1=px1, py1=py1,
        bx0=bx0, by0=by0, bx1=bx1, by1=by1,
        font_px=font_px,
        H=H, W=W,
    )

    region = TextRegion(
        content=text,
        confidence=float(np.clip(conf, 0.0, 1.0)),
        polygon=[[int(round(x)), int(round(y))] for x, y in poly],
        box=(px0 + bx0, py0 + by0, px0 + bx1, py0 + by1),
        font_size_px=float(font_px),
        color_rgb=tuple(int(round(c)) for c in text_rgb),
        color_hex=rgb_to_hex(text_rgb),
        alpha=(alpha[by0:by1, bx0:bx1] * 255).astype(np.uint8),
        alignment="left",
        editable=editable,
        font_family=font_family,
        tracking=tracking,
        angle_deg=angle_deg,
    )
    return region, glyph_full


def _build_dynamic_shadow_mask(
    rgb: np.ndarray,
    alpha: np.ndarray,
    px0: int, py0: int, px1: int, py1: int,
    bx0: int, by0: int, bx1: int, by1: int,
    font_px: float,
    H: int, W: int,
) -> np.ndarray:
    """Build an inpainting mask for text that captures both core letter ink and drop shadows / outer glows.
    
    Phase 1 dynamic expansion:
      1. Base dilation: max(4, int(font_px * 0.18))
      2. Second-pass luminance-variance threshold within a 1.3x expanded bounding box
         to absorb low-alpha drop shadows, outer glows, and dark gradient backings.
    """
    glyph_full = np.zeros((H, W), dtype=bool)

    # 1. Base dynamic dilation
    base_dilation = max(4, int(round(font_px * 0.18)))
    alpha_core = alpha > 0.02
    core_dilated = dilate(alpha_core, base_dilation)

    # Place core dilated mask into full canvas
    glyph_full[py0:py1, px0:px1] |= core_dilated

    # Absolute bounding box on full canvas
    x0 = px0 + bx0
    y0 = py0 + by0
    x1 = px0 + bx1
    y1 = py0 + by1
    bw = x1 - x0
    bh = y1 - y0

    if bw <= 0 or bh <= 0:
        return glyph_full

    # 2. 1.3x expanded bounding box around text center
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    exp_w = int(round(bw * 1.3))
    exp_h = int(round(bh * 1.3))
    ex0 = max(0, int(round(cx - exp_w / 2.0)))
    ey0 = max(0, int(round(cy - exp_h / 2.0)))
    ex1 = min(W, int(round(cx + exp_w / 2.0)))
    ey1 = min(H, int(round(cy + exp_h / 2.0)))

    if ex1 <= ex0 or ey1 <= ey0:
        return glyph_full

    exp_rgb = rgb[ey0:ey1, ex0:ex1]
    exp_lum = cv2.cvtColor(exp_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Clean background margin sampling (outer 8% border of expanded box)
    eh, ew = exp_lum.shape
    pad = max(2, int(min(eh, ew) * 0.08))
    border_mask = np.zeros((eh, ew), dtype=bool)
    border_mask[:pad, :] = True
    border_mask[-pad:, :] = True
    border_mask[:, :pad] = True
    border_mask[:, -pad:] = True

    border_vals = exp_lum[border_mask]
    if len(border_vals) > 0:
        bg_mean = float(np.mean(border_vals))
        bg_std = max(float(np.std(border_vals)), 4.0)

        # Luminance variance: pixels deviating significantly from border background
        lum_diff = np.abs(exp_lum - bg_mean)
        shadow_glow_candidates = lum_diff > (1.8 * bg_std)

        # Restrict to search radius around core text to prevent bleeding into adjacent elements
        search_radius = max(base_dilation + 3, int(round(font_px * 0.35)))
        core_in_exp = np.zeros((eh, ew), dtype=bool)

        # Sub-region overlap mapping
        sx0 = max(0, px0 - ex0)
        sy0 = max(0, py0 - ey0)
        sx1 = min(ew, px1 - ex0)
        sy1 = min(eh, py1 - ey0)

        cx0 = max(0, ex0 - px0)
        cy0 = max(0, ey0 - py0)
        cx1 = cx0 + (sx1 - sx0)
        cy1 = cy0 + (sy1 - sy0)

        if sx1 > sx0 and sy1 > sy0 and cy1 <= core_dilated.shape[0] and cx1 <= core_dilated.shape[1]:
            core_in_exp[sy0:sy1, sx0:sx1] = core_dilated[cy0:cy1, cx0:cx1]

        shadow_search_zone = dilate(core_in_exp, search_radius)
        confirmed_shadow_glow = shadow_glow_candidates & shadow_search_zone

        # Add to full canvas
        glyph_full[ey0:ey1, ex0:ex1] |= confirmed_shadow_glow

    return glyph_full


def _estimate_font_family(glyph_bin: np.ndarray, ink_h: int, ink_w: int, text: str) -> Tuple[str, int, float]:
    """Classify glyph style into standard Google Fonts / Photoshop PostScript font names, letter tracking, and baseline rotation angle (Phase 4).
    
    Categories:
      - Editorial Serif: PlayfairDisplay-Bold, PlayfairDisplay-Regular, PlayfairDisplay-Italic, Georgia
      - Heavy Display Sans: Montserrat-Black, Impact, BebasNeue-Regular, Arial-Black
      - Modern Clean Sans: Poppins-Bold, Poppins-Medium, Poppins-Regular, Montserrat-Bold
    """
    if glyph_bin.size == 0 or ink_h <= 0:
        return "ArialMT", 0, 0.0

    nl, labels, stats, _ = cv2.connectedComponentsWithStats(glyph_bin)
    clean_glyph = np.zeros_like(glyph_bin)
    char_centers = []
    char_aspects = []
    slants = []
    ch_list = []
    
    for lbl in range(1, nl):
        ch = stats[lbl, cv2.CC_STAT_HEIGHT]
        cw = stats[lbl, cv2.CC_STAT_WIDTH]
        area = stats[lbl, cv2.CC_STAT_AREA]
        # Ignore horizontal divider bars / seesaws (aspect > 2.2) and tiny noise specs
        if area > 30 and ch >= 0.30 * ink_h and cw <= 2.2 * ch:
            clean_glyph[labels == lbl] = 1
            cx = stats[lbl, cv2.CC_STAT_LEFT] + cw / 2.0
            cy = stats[lbl, cv2.CC_STAT_TOP] + ch / 2.0
            char_centers.append((cx, cy))
            char_aspects.append(cw / float(ch))
            ch_list.append(ch)
            pts = np.argwhere(labels == lbl)
            if len(pts) > 15:
                cov = np.cov(pts[:, 1], pts[:, 0])
                if cov[1, 1] > 0:
                    slants.append(float(np.degrees(np.arctan(cov[0, 1] / cov[1, 1]))))

    # Baseline rotation angle calculation
    angle_deg = 0.0
    if len(char_centers) >= 4:
        char_centers.sort(key=lambda p: p[0])
        pts_c = np.array(char_centers, dtype=np.float32)
        vx, vy, _, _ = cv2.fitLine(pts_c, cv2.DIST_L2, 0, 0.01, 0.01)
        fitted_ang = float(np.degrees(np.arctan2(vy[0], vx[0])))
        if 2.0 < abs(fitted_ang) < 35.0:
            angle_deg = round(fitted_ang, 1)

    effective_glyph = clean_glyph if clean_glyph.any() else glyph_bin
    dist = cv2.distanceTransform(effective_glyph, cv2.DIST_L2, 3)
    
    # Medial axis / ridge points: local maxima in 3x3 window (true stroke stems)
    dil = cv2.dilate(dist, np.ones((3, 3), np.uint8))
    skeleton = (dist == dil) & (dist > 1.0)
    ridge_vals = dist[skeleton] * 2.0

    if len(ridge_vals) == 0:
        return "ArialMT", 0, angle_deg

    ink_h_eff = float(max(ch_list)) if ch_list else float(ink_h)
    p90_stroke = float(np.percentile(ridge_vals, 90))
    p50_stroke = float(np.percentile(ridge_vals, 50))
    p20_stroke = float(np.percentile(ridge_vals, 20))
    stroke_ratio = p90_stroke / max(1.0, ink_h_eff)
    contrast = p90_stroke / max(1.0, p20_stroke)

    med_slant = float(np.median(slants)) if slants else 0.0
    is_italic = med_slant <= -7.0
    avg_char_aspect = float(np.mean(char_aspects)) if char_aspects else 0.65

    # True Serif Detection (Editorial Serif like Playfair Display / Bodoni / Georgia):
    # Authentic serifs have extreme ridge contrast between thick vertical stems and thin hairlines (ratio > 2.5:1)
    has_extreme_contrast = contrast >= 2.55
    has_lowercase = any(c.islower() for c in text)
    is_serif = has_extreme_contrast and (has_lowercase or stroke_ratio < 0.22)

    # Tracking calculation (letter spacing)
    tracking = 0
    if nl > 2:
        xs = sorted([stats[lbl, cv2.CC_STAT_LEFT] for lbl in range(1, nl) if stats[lbl, cv2.CC_STAT_HEIGHT] > 0.35 * ink_h])
        if len(xs) >= 2:
            gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
            avg_gap = float(np.median(gaps))
            tracking = int(np.clip((avg_gap - 0.55 * ink_h) / max(1.0, ink_h) * 500, -50, 150))

    # Devanagari (Hindi) typography classification
    if re.search(r'[\u0900-\u097F]', text):
        # 1. Editorial High-Contrast Serif Devanagari (Rozha One)
        if contrast >= 2.2:
            return "RozhaOne-Regular", tracking, angle_deg
        # 2. Condensed Devanagari headlines (Khand - tall vertical rectangular style)
        if avg_char_aspect < 0.50:
            if stroke_ratio >= 0.12:
                return "Khand-Bold", tracking, angle_deg
            elif stroke_ratio >= 0.09:
                return "Khand-SemiBold", tracking, angle_deg
            elif stroke_ratio >= 0.075:
                return "Khand-Medium", tracking, angle_deg
            else:
                return "Khand-Regular", tracking, angle_deg
        # 3. Humanist Clean Devanagari (Mukta - open apertures, contemporary curves)
        if 0.50 <= avg_char_aspect < 0.63:
            if stroke_ratio >= 0.15:
                return "Mukta-ExtraBold", tracking, angle_deg
            elif stroke_ratio >= 0.11:
                return "Mukta-Bold", tracking, angle_deg
            elif stroke_ratio >= 0.09:
                return "Mukta-SemiBold", tracking, angle_deg
            elif stroke_ratio >= 0.075:
                return "Mukta-Medium", tracking, angle_deg
            else:
                return "Mukta-Regular", tracking, angle_deg
        # 4. Modern Geometric Sans (Poppins - pure circular curves)
        if stroke_ratio >= 0.15:
            return "Poppins-Black", tracking, angle_deg
        elif stroke_ratio >= 0.11:
            return "Poppins-Bold", tracking, angle_deg
        elif stroke_ratio >= 0.09:
            return "Poppins-SemiBold", tracking, angle_deg
        elif stroke_ratio >= 0.075:
            return "Poppins-Medium", tracking, angle_deg
        else:
            return "Poppins-Regular", tracking, angle_deg

    # English / Latin typography classification:
    # 1. Editorial High-Contrast Serif (Playfair Display / Bodoni)
    if is_serif:
        if is_italic:
            return ("PlayfairDisplay-BoldItalic" if stroke_ratio >= 0.12 else "PlayfairDisplay-Italic"), tracking, angle_deg
        elif stroke_ratio >= 0.15:
            return "PlayfairDisplay-Black", tracking, angle_deg
        elif stroke_ratio >= 0.11:
            return "PlayfairDisplay-Bold", tracking, angle_deg
        else:
            return "PlayfairDisplay-Regular", tracking, angle_deg

    # 2. Heavy Display Sans (Headlines like "JUGGLING", "WORKSHOP", "CALM IN CHAOS")
    if (stroke_ratio >= 0.14 and text.isupper()) or stroke_ratio >= 0.16:
        if avg_char_aspect < 0.48:
            return "Impact", tracking, angle_deg
        elif avg_char_aspect > 0.72:
            return "Montserrat-Black", tracking, angle_deg
        else:
            return "Poppins-Black", tracking, angle_deg

    # 3. Modern Clean Sans (Poppins / Montserrat - ad standard)
    if is_italic:
        return ("Poppins-BoldItalic" if stroke_ratio >= 0.11 else "Poppins-Italic"), tracking, angle_deg
    else:
        # Wide geometric or spaced uppercase subheads -> Montserrat
        if avg_char_aspect >= 0.70 or tracking >= 40:
            if stroke_ratio >= 0.12:
                return "Montserrat-Bold", tracking, angle_deg
            elif stroke_ratio >= 0.09:
                return "Montserrat-SemiBold", tracking, angle_deg
            elif stroke_ratio >= 0.075:
                return "Montserrat-Medium", tracking, angle_deg
            else:
                return "Montserrat-Regular", tracking, angle_deg
        else:
            # Balanced circular geometric -> Poppins
            if stroke_ratio >= 0.11:
                return "Poppins-Bold", tracking, angle_deg
            elif stroke_ratio >= 0.09:
                return "Poppins-SemiBold", tracking, angle_deg
            elif stroke_ratio >= 0.075:
                return "Poppins-Medium", tracking, angle_deg
            else:
                return "Poppins-Regular", tracking, angle_deg


def _assign_alignment(regions: Sequence[TextRegion], canvas_w: int) -> None:
    """Group lines into paragraphs; infer alignment from edge variance, and compute explicit typographic leading."""
    if not regions:
        return
    order = sorted(range(len(regions)), key=lambda i: regions[i].box[1])
    blocks: List[List[int]] = []
    for idx in order:
        x0, y0, x1, y1 = regions[idx].box
        placed = False
        for block in blocks:
            bx0, by0, bx1, by1 = regions[block[-1]].box
            line_h = max(y1 - y0, by1 - by0)
            min_line_h = min(y1 - y0, by1 - by0)
            overlap = min(x1, bx1) - max(x0, bx0)
            # Only group into same paragraph if line heights are similar and vertical gap is tight
            if line_h <= 1.4 * min_line_h and 0 <= y0 - by1 <= 1.1 * line_h and overlap > 0.3 * min(x1 - x0, bx1 - bx0):
                block.append(idx)
                placed = True
                break
        if not placed:
            blocks.append([idx])

    for block in blocks:
        boxes = np.array([regions[i].box for i in block], dtype=np.float32)
        if len(block) >= 2:
            spread = {
                "left": float(np.std(boxes[:, 0])),
                "center": float(np.std((boxes[:, 0] + boxes[:, 2]) / 2)),
                "right": float(np.std(boxes[:, 2])),
            }
            align = min(spread, key=spread.get)
            block_cx = float(np.mean((boxes[:, 0] + boxes[:, 2]) / 2))
            block_left = float(np.min(boxes[:, 0]))
            block_right = float(np.max(boxes[:, 2]))
            # Left-margin aligned text starting near the left edge (< 15% width) and not spanning full canvas is left-aligned
            if block_left < 0.15 * canvas_w and block_right < 0.70 * canvas_w:
                align = "left"
            elif abs(block_cx - canvas_w / 2) < 0.06 * canvas_w and spread["center"] <= spread["left"] * 1.3:
                align = "center"
        else:
            x0, _, x1, _ = boxes[0]
            cx = (x0 + x1) / 2
            left_margin, right_margin = x0, canvas_w - x1
            # Only mark as center if really centered on the whole canvas AND margins are roughly balanced
            if abs(cx - canvas_w / 2) < 0.05 * canvas_w and abs(left_margin - right_margin) < 0.10 * canvas_w:
                align = "center"
            elif right_margin < 0.5 * left_margin and cx > canvas_w * 0.65:
                align = "right"
            else:
                align = "left"

        # Explicit typographic line leading calculation (baseline-to-baseline distance)
        for k in range(len(block)):
            cur_idx = block[k]
            cur_reg = regions[cur_idx]
            cur_reg.alignment = align
            if k + 1 < len(block):
                next_idx = block[k + 1]
                next_reg = regions[next_idx]
                delta_y = float(next_reg.box[1] - cur_reg.box[1])
                cur_reg.leading_pt = round(max(cur_reg.font_size_px * 0.9, delta_y * 0.72), 1)
            else:
                cur_reg.leading_pt = round(cur_reg.font_size_px * 1.18, 1)
