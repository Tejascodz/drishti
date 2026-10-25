"""Plate crops: padding, perspective rectification, deskew and two-line stitching."""

from __future__ import annotations

import cv2
import numpy as np

from .types import BBox


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def crop_box(img: np.ndarray, bbox: BBox, fx: float = 0.0, fy: float = 0.0) -> np.ndarray:
    H, W = img.shape[:2]
    b = bbox.expand(fx, fy).clip(W, H)
    x1, y1, x2, y2 = int(b.x1), int(b.y1), int(np.ceil(b.x2)), int(np.ceil(b.y2))
    return img[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]


def warp_quad(img: np.ndarray, quad: np.ndarray, fx: float = 0.0, fy: float = 0.0, min_h: int = 48) -> np.ndarray:
    """Perspective-rectify the quadrilateral (TL, TR, BR, BL), padded by (fx, fy) of its size."""
    q = order_quad(quad)
    wt = np.linalg.norm(q[1] - q[0])
    wb = np.linalg.norm(q[2] - q[3])
    hl = np.linalg.norm(q[3] - q[0])
    hr = np.linalg.norm(q[2] - q[1])
    w, h = max(wt, wb), max(hl, hr)
    if w < 4 or h < 4:
        raise ValueError("degenerate quad")
    scale = max(1.0, min_h / h)
    ow, oh = w * (1 + 2 * fx) * scale, h * (1 + 2 * fy) * scale
    dst = np.array(
        [[fx * w * scale, fy * h * scale], [(1 + fx) * w * scale, fy * h * scale], [(1 + fx) * w * scale, (1 + fy) * h * scale], [fx * w * scale, (1 + fy) * h * scale]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, M, (int(round(ow)), int(round(oh))), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def corners_from_keypoints(kpts: np.ndarray | None, bbox: BBox, min_conf: float = 0.5) -> np.ndarray | None:
    """Validate 4 detector keypoints; returns ordered quad or None."""
    if kpts is None or len(kpts) < 4:
        return None
    k = np.asarray(kpts[:4], dtype=np.float32)
    if k.shape[1] >= 3 and np.any(k[:, 2] < min_conf):
        return None
    quad = order_quad(k[:, :2])
    area = cv2.contourArea(quad)
    if area < 0.25 * bbox.area or area > 2.0 * bbox.area or not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
        return None
    return quad


def _text_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask of character pixels, choosing polarity so text is the minority class."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = 255 - th
    h, w = gray.shape
    core = (slice(int(0.15 * h), int(0.85 * h)), slice(int(0.1 * w), int(0.9 * w)))
    frac_dark = dark[core].mean() / 255.0
    return dark if frac_dark < 0.5 else th


def _char_components(mask: np.ndarray, min_h: float, max_h: float):
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if min_h <= h <= max_h and 0.08 * h <= w <= 1.2 * h and a >= 0.15 * w * h:
            out.append((cents[i][0], cents[i][1], h))
    return out


def estimate_skew(crop: np.ndarray) -> float:
    """Text-baseline angle in degrees (positive = clockwise), 0 when undetermined."""
    if crop.size == 0 or min(crop.shape[:2]) < 10:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h = gray.shape[0]
    comps = _char_components(_text_mask(gray), 0.25 * h, 0.9 * h)
    if len(comps) < 4:
        return 0.0
    pts = np.array([[c[0], c[1]] for c in comps], dtype=np.float32)
    vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return float(np.degrees(np.arctan2(vy, vx)))


def deskew(crop: np.ndarray, min_deg: float = 3.0, max_deg: float = 30.0) -> np.ndarray | None:
    ang = estimate_skew(crop)
    if not (min_deg <= abs(ang) <= max_deg):
        return None
    h, w = crop.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _plate_region(gray: np.ndarray):
    """Locate the plate body inside a padded crop.

    Returns (glyph mask, (x, y, w, h), body mask) or None. Both polarities are tried
    (dark-on-light HSRP, yellow-on-black rental); the body is the largest compact
    component that does not wrap around the crop."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, light = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = gray.shape
    best = None
    for body in (light, 255 - light):
        closed = cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=4)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < 0.2 * h * w or bw < 0.35 * w or bh < 0.3 * h:
                continue
            touches = int(x <= 1) + int(y <= 1) + int(x + bw >= w - 1) + int(y + bh >= h - 1)
            comp = cv2.morphologyEx((labels == i).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
            rect = comp[y : y + bh, x : x + bw].mean()
            score = area * rect * (0.25 if touches >= 3 else 1.0)
            if best is None or score > best[0]:
                best = (score, 255 - body, (int(x), int(y), int(bw), int(bh)), comp)
    if best is None:
        return None
    # glyphs: re-threshold inside the body only (globally, faint glyphs merge with the plate)
    x, y, bw, bh = best[2]
    roi = blur[y : y + bh, x : x + bw]
    _, local = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    glyph_local = 255 - local if local.mean() > 127 else local  # glyphs are the minority
    glyphs = np.zeros_like(gray)
    glyphs[y : y + bh, x : x + bw] = glyph_local
    return glyphs, best[2], best[3]


def _body_angle(body_mask: np.ndarray) -> float:
    """Angle (degrees, image coords) of the plate body's long edge, in [-45, 45]."""
    pts = cv2.findNonZero(body_mask)
    if pts is None or len(pts) < 10:
        return 0.0
    box = cv2.boxPoints(cv2.minAreaRect(pts))
    edges = [box[(i + 1) % 4] - box[i] for i in range(4)]
    e = max(edges, key=lambda v: float(np.hypot(*v)))
    ang = float(np.degrees(np.arctan2(e[1], e[0])))
    while ang > 45:
        ang -= 180 if ang > 135 else 90
    while ang < -45:
        ang += 180 if ang < -135 else 90
    return ang


def _rotate(img: np.ndarray, ang: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def two_line_layout(crop: np.ndarray) -> tuple[int, int, int, int, int] | None:
    """(x1, y1, x2, y2, split_row) of a level two-line plate inside `crop`, or None.

    Uses the row profile of glyph pixels inside the plate body, so it holds up at
    low resolution where neighbouring characters merge into one blob."""
    if crop.size == 0 or crop.shape[0] < 12:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    scale = 96.0 / gray.shape[0]
    g = cv2.resize(gray, (max(16, int(gray.shape[1] * scale)), 96), interpolation=cv2.INTER_CUBIC)
    found = _plate_region(g)
    if found is None:
        return None
    text, (x, y, bw, bh), _ = found
    if bw / max(bh, 1) > 3.0:  # a wide body is a single-line plate
        return None
    mx, my = int(0.06 * bw), int(0.06 * bh)
    body = text[y + my : y + bh - my, x + mx : x + bw - mx]
    if body.shape[0] < 16:
        return None
    prof = body.mean(axis=1) / 255.0
    prof = np.convolve(prof, np.ones(5) / 5, mode="same")
    n = len(prof)
    lo, hi = int(0.3 * n), int(0.7 * n)
    v = lo + int(np.argmin(prof[lo:hi]))
    top_peak, bot_peak = prof[:v].max(initial=0), prof[v:].max(initial=0)
    if min(top_peak, bot_peak) < 0.12 or prof[v] > 0.5 * min(top_peak, bot_peak):
        return None
    inv = 1.0 / scale
    return (int(x * inv), int(y * inv), int(np.ceil((x + bw) * inv)), int(np.ceil((y + bh) * inv)), int(round((y + my + v) * inv)))


def level_plate(crop: np.ndarray, min_deg: float = 2.0, max_deg: float = 25.0) -> np.ndarray:
    """Rotate a crop so the plate body is horizontal (no-op if already level / unknown)."""
    if crop.size == 0 or crop.shape[0] < 12:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    scale = 96.0 / gray.shape[0]
    g = cv2.resize(gray, (max(16, int(gray.shape[1] * scale)), 96), interpolation=cv2.INTER_CUBIC)
    found = _plate_region(g)
    if found is None:
        return crop
    ang = _body_angle(found[2])
    return _rotate(crop, ang) if min_deg <= abs(ang) <= max_deg else crop


def two_line_views(crop: np.ndarray, out_h: int = 48) -> list[np.ndarray]:
    """Stitched single-line renderings of a possibly two-line plate (0 or 1 images)."""
    leveled = level_plate(crop)
    layout = two_line_layout(leveled)
    return [] if layout is None else [stitch_two_lines(leveled, layout, out_h)]


def two_line_split(crop: np.ndarray) -> int | None:
    lay = two_line_layout(crop)
    return None if lay is None else lay[4]


def stitch_two_lines(crop: np.ndarray, layout: tuple[int, int, int, int, int] | int, out_h: int = 48) -> np.ndarray:
    """Place the lower line to the right of the upper line: 'TN12AD / 6629' -> 'TN12AD 6629'.

    `layout` is the tuple from two_line_layout (preferred) or just a split row."""
    if isinstance(layout, (int, np.integer)):
        x1, y1, x2, y2, split = 0, 0, crop.shape[1], crop.shape[0], int(layout)
    else:
        x1, y1, x2, y2, split = layout
    body = crop[y1:y2, x1:x2]
    split = int(np.clip(split - y1, 2, body.shape[0] - 2))
    top, bottom = body[:split], body[split:]

    def _norm(part: np.ndarray) -> np.ndarray:
        h, w = part.shape[:2]
        return cv2.resize(part, (max(1, int(round(w * out_h / max(h, 1)))), out_h), interpolation=cv2.INTER_CUBIC)

    t, b = _norm(top), _norm(bottom)
    edge = np.concatenate([body[:, 0].reshape(-1, body.shape[-1]), body[:, -1].reshape(-1, body.shape[-1])])
    fill = np.median(edge, axis=0).astype(crop.dtype)
    gap = np.empty((out_h, max(2, out_h // 5), crop.shape[-1]), dtype=crop.dtype)
    gap[:] = fill
    return np.concatenate([t, gap, b], axis=1)


def estimate_quad(crop: np.ndarray) -> np.ndarray | None:
    """Find the plate's own border inside a padded crop (classical, used when the
    detector has no corner head). Returns quad in crop coords or None."""
    if crop.size == 0 or min(crop.shape[:2]) < 16:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    g = cv2.bilateralFilter(gray, 5, 40, 40)
    edges = cv2.Canny(g, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.35 * w * h or area > 0.97 * w * h:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        q = order_quad(approx.reshape(4, 2))
        qw = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))
        qh = max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))
        if qh <= 0 or not (1.2 <= qw / qh <= 7.0):
            continue
        if area > best_area:
            best, best_area = q, area
    return best
