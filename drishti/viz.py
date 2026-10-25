"""Overlay drawing for images and videos."""

from __future__ import annotations

import cv2
import numpy as np

from .types import BBox

_GREEN = (60, 200, 60)
_AMBER = (0, 170, 255)
_RED = (40, 40, 230)
_GREY = (160, 160, 160)


def _label(img: np.ndarray, text: str, x: int, y: int, color) -> None:
    scale = max(0.5, min(1.2, img.shape[1] / 1600))
    th = max(1, int(round(scale * 2)))
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    y = max(tht + base + 2, y)
    x = max(0, min(x, img.shape[1] - tw - 6))  # keep the label inside the frame
    cv2.rectangle(img, (x, y - tht - base - 4), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - base - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th, cv2.LINE_AA)


def draw_plate(img: np.ndarray, bbox: BBox, text: str | None, conf: float, corners=None, track_id: int | None = None,
               low: bool = False, alert: bool = False, vehicle: BBox | None = None) -> None:
    color = _RED if alert else (_AMBER if low else _GREEN)
    if vehicle is not None:
        cv2.rectangle(img, (int(vehicle.x1), int(vehicle.y1)), (int(vehicle.x2), int(vehicle.y2)), _GREY, 1)
    if corners is not None:
        cv2.polylines(img, [np.asarray(corners, np.int32).reshape(-1, 1, 2)], True, color, 2)
    else:
        cv2.rectangle(img, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), color, 2)
    tag = (f"#{track_id} " if track_id is not None else "") + (text or "?") + f" {conf:.2f}"
    _label(img, tag, int(bbox.x1), int(bbox.y1) - 4, color)
