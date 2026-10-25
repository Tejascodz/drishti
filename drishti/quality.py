"""Crop quality: decides which frames to trust and which crops to fuse."""

from __future__ import annotations

import cv2
import numpy as np


def plate_quality(crop: np.ndarray, plate_height_px: float) -> float:
    """Score in [0, 1] combining resolution, sharpness, contrast and exposure."""
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    g = cv2.resize(gray, (max(8, int(gray.shape[1] * 48 / max(gray.shape[0], 1))), 48), interpolation=cv2.INTER_AREA)
    sharp = float(cv2.Laplacian(g, cv2.CV_32F).var())
    s = 1.0 - np.exp(-sharp / 250.0)
    contrast = min(1.0, float(g.std()) / 55.0)
    clipped = float(((g > 250) | (g < 5)).mean())
    exposure = max(0.0, 1.0 - 2.5 * clipped)
    size = min(1.0, plate_height_px / 36.0) ** 0.6
    return float(np.clip(size * (0.55 * s + 0.45 * contrast) * exposure, 0.0, 1.0))


_COLORS = {
    "white": (0, 0, 92), "silver": (0, 0, 70), "grey": (0, 0, 45), "black": (0, 0, 12),
    "red": (0, 70, 60), "orange": (22, 75, 75), "yellow": (50, 70, 75), "green": (120, 55, 50),
    "blue": (220, 60, 50), "brown": (25, 55, 35),
}


def dominant_color(vehicle_crop: np.ndarray) -> str | None:
    """Coarse body colour from the lower-middle of a vehicle box (avoids windows/road)."""
    if vehicle_crop is None or vehicle_crop.size == 0 or min(vehicle_crop.shape[:2]) < 12:
        return None
    h, w = vehicle_crop.shape[:2]
    roi = vehicle_crop[int(0.45 * h) : int(0.85 * h), int(0.15 * w) : int(0.85 * w)]
    hsv = cv2.cvtColor(cv2.resize(roi, (48, 32)), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    data = hsv.copy()
    k = 3
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k)
    hh, ss, vv = centers[int(np.argmax(counts))]
    hue, sat, val = hh * 2.0, ss / 2.55, vv / 2.55  # degrees, %, %
    if sat < 18:
        if val > 68:
            return "white"
        if val > 48:
            return "silver"
        if val > 22:
            return "grey"
        return "black"
    best, bd = None, 1e9
    for name, (ch, cs, cv) in _COLORS.items():
        if cs == 0:
            continue
        dh = min(abs(hue - ch), 360 - abs(hue - ch)) / 180.0
        d = 3 * dh + abs(sat - cs) / 100 + abs(val - cv) / 100
        if d < bd:
            best, bd = name, d
    return best
