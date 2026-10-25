"""Lane zone and trigger line for fixed cameras (toll plazas, gates, parking barriers).

* roi:  polygon; only plates whose centre lies inside are tracked and read, so the
        next lane, the queue behind and roadside clutter are ignored.
* trigger_line:  when a tracked plate's centre crosses this segment, the vehicle's
        event is emitted at once with the best evidence so far. This gives exactly
        one decision per vehicle at the barrier instead of waiting for the track to end.

Coordinates may be fractions of the frame (all values <= 1) or pixels.
"""

from __future__ import annotations

import cv2
import numpy as np


def _parse_points(spec) -> np.ndarray | None:
    """[[x, y], ...] or 'x,y x,y ...' -> (N, 2) float array."""
    if spec is None or spec == "" or spec == []:
        return None
    if isinstance(spec, str):
        pts = [[float(v) for v in tok.split(",")] for tok in spec.replace(";", " ").split()]
    else:
        pts = [[float(v) for v in p] for p in spec]
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected a list of x,y points, got {spec!r}")
    return arr


def _segments_cross(p1, p2, q1, q2) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


class Zones:
    def __init__(self, roi=None, trigger_line=None, trigger_direction: str = "any"):
        self._roi = _parse_points(roi)
        self._line = _parse_points(trigger_line)
        if self._roi is not None and len(self._roi) < 3:
            raise ValueError("roi needs at least 3 points")
        if self._line is not None and len(self._line) != 2:
            raise ValueError("trigger_line needs exactly 2 points")
        self.direction = str(trigger_direction or "any").lower()
        self._cache: dict[tuple[int, int], tuple] = {}

    @classmethod
    def from_config(cls, zcfg) -> "Zones":
        zcfg = zcfg or {}
        return cls(zcfg.get("roi"), zcfg.get("trigger_line"), zcfg.get("trigger_direction", "any"))

    @property
    def active(self) -> bool:
        return self._roi is not None or self._line is not None

    @property
    def has_line(self) -> bool:
        return self._line is not None

    def _px(self, pts: np.ndarray | None, W: int, H: int) -> np.ndarray | None:
        if pts is None:
            return None
        if np.all(np.abs(pts) <= 1.0):
            return pts * np.array([W, H], dtype=np.float64)
        return pts

    def pixels(self, W: int, H: int):
        key = (W, H)
        if key not in self._cache:
            self._cache[key] = (self._px(self._roi, W, H), self._px(self._line, W, H))
        return self._cache[key]

    def inside(self, x: float, y: float, W: int, H: int) -> bool:
        roi, _ = self.pixels(W, H)
        if roi is None:
            return True
        return cv2.pointPolygonTest(roi.astype(np.float32).reshape(-1, 1, 2), (float(x), float(y)), False) >= 0

    def crossed(self, prev: tuple[float, float], cur: tuple[float, float], W: int, H: int) -> bool:
        _, line = self.pixels(W, H)
        if line is None or not _segments_cross(prev, cur, line[0], line[1]):
            return False
        if self.direction == "any":
            return True
        dx, dy = cur[0] - prev[0], cur[1] - prev[1]
        if abs(dy) >= abs(dx):
            moved = "down" if dy > 0 else "up"
        else:
            moved = "right" if dx > 0 else "left"
        return moved == self.direction

    def draw(self, img: np.ndarray) -> None:
        H, W = img.shape[:2]
        roi, line = self.pixels(W, H)
        if roi is not None:
            cv2.polylines(img, [roi.astype(np.int32).reshape(-1, 1, 2)], True, (255, 200, 0), 2, cv2.LINE_AA)
        if line is not None:
            a, b = line.astype(int)
            cv2.line(img, tuple(a), tuple(b), (0, 215, 255), 3, cv2.LINE_AA)


def grid_frame(frame: np.ndarray, zones: Zones | None = None, step: float = 0.1) -> np.ndarray:
    """Frame with a labelled 0..1 grid (and current zones) for choosing coordinates."""
    out = frame.copy()
    H, W = out.shape[:2]
    scale = max(0.4, min(1.0, W / 1600))
    for k in range(1, int(round(1 / step))):
        f = k * step
        x, y = int(f * W), int(f * H)
        cv2.line(out, (x, 0), (x, H), (255, 255, 255), 1)
        cv2.line(out, (0, y), (W, y), (255, 255, 255), 1)
        cv2.putText(out, f"{f:.1f}", (x + 3, int(18 * scale) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 255, 255), 2)
        cv2.putText(out, f"{f:.1f}", (4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 255, 255), 2)
    if zones is not None:
        zones.draw(out)
    return out
