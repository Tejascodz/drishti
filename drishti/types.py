"""Core data structures shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def area(self) -> float:
        return self.w * self.h

    def as_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    def to_list(self) -> list[int]:
        return [int(round(v)) for v in (self.x1, self.y1, self.x2, self.y2)]

    def clip(self, width: int, height: int) -> "BBox":
        return BBox(
            float(np.clip(self.x1, 0, width - 1)),
            float(np.clip(self.y1, 0, height - 1)),
            float(np.clip(self.x2, 0, width)),
            float(np.clip(self.y2, 0, height)),
        )

    def expand(self, fx: float, fy: float) -> "BBox":
        dx, dy = self.w * fx, self.h * fy
        return BBox(self.x1 - dx, self.y1 - dy, self.x2 + dx, self.y2 + dy)

    def iou(self, other: "BBox") -> float:
        ix = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        iy = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def containment(self, outer: "BBox") -> float:
        """Fraction of this box that lies inside `outer`."""
        ix = max(0.0, min(self.x2, outer.x2) - max(self.x1, outer.x1))
        iy = max(0.0, min(self.y2, outer.y2) - max(self.y1, outer.y1))
        return (ix * iy) / self.area if self.area > 0 else 0.0

    @staticmethod
    def from_array(a) -> "BBox":
        return BBox(float(a[0]), float(a[1]), float(a[2]), float(a[3]))


@dataclass
class Detection:
    bbox: BBox
    score: float
    label: str = "plate"
    keypoints: np.ndarray | None = None  # (K, 3): x, y, visibility/conf in image coords


@dataclass
class Candidate:
    text: str
    score: float  # log-probability (higher is better)
    fmt: str


@dataclass
class PlateRead:
    text: str | None
    confidence: float
    fmt: str
    char_conf: list[float] = field(default_factory=list)
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    raw_text: str = ""
    format_penalty: float = 0.0
    low_confidence: bool = False
    abstained: bool = False
    n_views: int = 1
    agreement: float = 1.0
    verified_by: str | None = None
    string_prob: float = 0.0  # grammar-aware probability of the string before agreement/calibration
    level: str | None = None  # Good / Average / Poor (IHMCL-style confidence band)
    rescued: bool = False  # read only after enhanced retries (zoom / contrast / brightness / grey)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "format": self.fmt,
            "char_confidence": [round(float(c), 4) for c in self.char_conf],
            "alternatives": [{"text": t, "confidence": round(float(c), 4)} for t, c in self.alternatives],
            "raw_text": self.raw_text,
            "format_penalty": round(float(self.format_penalty), 3),
            "low_confidence": self.low_confidence,
            "abstained": self.abstained,
            "views": self.n_views,
            "agreement": round(float(self.agreement), 3),
            "verified_by": self.verified_by,
            "confidence_level": self.level,
            "rescued": self.rescued,
        }


@dataclass
class VehicleInfo:
    bbox: BBox
    label: str
    score: float
    color: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox.to_list(),
            "type": self.label,
            "score": round(float(self.score), 3),
            "color": self.color,
        }


@dataclass
class PlateResult:
    bbox: BBox
    det_score: float
    read: PlateRead
    quality: float
    corners: np.ndarray | None = None
    two_line: bool = False
    vehicle: VehicleInfo | None = None
    crop: np.ndarray | None = None  # rectified crop (not serialised)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox.to_list(),
            "detection_score": round(float(self.det_score), 4),
            "corners": None if self.corners is None else np.round(self.corners, 1).tolist(),
            "two_line": self.two_line,
            "quality": round(float(self.quality), 3),
            "plate": self.read.to_dict(),
            "vehicle": None if self.vehicle is None else self.vehicle.to_dict(),
        }


@dataclass
class PlateEvent:
    event_id: str
    track_id: int
    camera: str
    text: str | None
    confidence: float
    fmt: str
    first_ts: float
    last_ts: float
    first_frame: int
    last_frame: int
    n_frames: int
    n_reads: int
    bbox: BBox
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    agreement: float = 1.0
    vehicle_type: str | None = None
    vehicle_color: str | None = None
    direction: str | None = None
    provisional: bool = False
    low_confidence: bool = False
    watchlist_hit: dict | None = None
    crop: np.ndarray | None = None
    crop_path: str | None = None
    verified_by: str | None = None
    trigger: str | None = None  # "line" (crossed the trigger line) | "track_end" | "early"
    corrects: str | None = None  # event_id this event supersedes (later evidence changed the read)
    level: str | None = None  # Good / Average / Poor (IHMCL-style confidence band)
    plate_width_px: int | None = None  # widest the plate was seen; one-line plates want >= 100 px
    duplicate_of: str | None = None  # same vehicle already recorded by an overlapping camera

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "track_id": self.track_id,
            "camera": self.camera,
            "plate": self.text,
            "confidence": round(float(self.confidence), 4),
            "format": self.fmt,
            "first_ts": round(self.first_ts, 3),
            "last_ts": round(self.last_ts, 3),
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "frames": self.n_frames,
            "reads": self.n_reads,
            "bbox": self.bbox.to_list(),
            "alternatives": [{"text": t, "confidence": round(float(c), 4)} for t, c in self.alternatives],
            "agreement": round(float(self.agreement), 3),
            "vehicle_type": self.vehicle_type,
            "vehicle_color": self.vehicle_color,
            "direction": self.direction,
            "provisional": self.provisional,
            "low_confidence": self.low_confidence,
            "watchlist_hit": self.watchlist_hit,
            "crop_path": self.crop_path,
            "verified_by": self.verified_by,
            "trigger": self.trigger,
            "corrects": self.corrects,
            "confidence_level": self.level,
            "plate_width_px": self.plate_width_px,
            "duplicate_of": self.duplicate_of,
        }
