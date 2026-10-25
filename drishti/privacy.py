"""Privacy helpers: keyed hashing of plate text and redaction of plates in media."""

from __future__ import annotations

import hashlib
import hmac
import os

import cv2
import numpy as np

from .types import BBox


def plate_digest(text: str, key_env: str = "DRISHTI_HMAC_KEY") -> str:
    """HMAC-SHA256 of the normalised plate. Equal plates hash equal, so lookups and
    watchlists still work, but the stored value cannot be reversed without the key."""
    key = os.environ.get(key_env)
    if not key:
        raise RuntimeError(f"privacy.hash_plates is on but {key_env} is not set")
    norm = "".join(ch for ch in text.upper() if ch.isalnum())
    return hmac.new(key.encode(), norm.encode(), hashlib.sha256).hexdigest()


def blur_region(img: np.ndarray, bbox: BBox, pad: float = 0.15) -> None:
    H, W = img.shape[:2]
    b = bbox.expand(pad, pad).clip(W, H)
    x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = img[y1:y2, x1:x2]
    k = max(3, (min(roi.shape[:2]) // 3) | 1)
    small = cv2.resize(roi, (max(1, roi.shape[1] // 8), max(1, roi.shape[0] // 8)), interpolation=cv2.INTER_LINEAR)
    img[y1:y2, x1:x2] = cv2.GaussianBlur(cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST), (k, k), 0)
