"""Camera health: catches the field problems that silently lower read rates.

A tolling camera can keep streaming while its images stop being readable: a dusty or
mud-sprayed window, a lens knocked out of focus, a loose bracket that moved the view, an
infrared illuminator that failed at night, glare, a frozen encoder, a sensor growing bad
pixels. Each check runs on a small copy of the frame about once a second and raises an
alert when a condition persists, then clears it when it recovers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

MESSAGES = {
    "too_dark": "image too dark: infrared illuminator or exposure fault?",
    "overexposed": "large overexposed area: glare, sun or exposure fault",
    "soft_image": "image much softer than usual: dirty or wet window, fog, or focus drift",
    "view_blocked": "most of the view is flat: lens covered or blocked",
    "frozen_feed": "frames are not changing: camera or encoder frozen",
    "camera_moved": "view shifted from its reference: bracket loose or camera knocked",
    "sensor_defects": "many defective pixels learned: sensor ageing or damage",
}


@dataclass
class _Check:
    count: int = 0
    active: bool = False


@dataclass
class CameraHealth:
    camera: str = "cam0"
    check_every_s: float = 1.0
    dark_below: float = 12.0
    overexposed_above: float = 0.25
    soft_ratio: float = 0.30
    frozen_after_s: float = 10.0
    shift_frac: float = 0.02
    persist_checks: int = 3
    baseline_checks: int = 30
    width: int = 320
    max_defects_frac: float = 0.001
    _last: float | None = None
    _prev: np.ndarray | None = None
    _ref: np.ndarray | None = None
    _sharp_hist: list = field(default_factory=list)
    _frozen_since: float | None = None
    _checks: dict = field(default_factory=dict)
    last_metrics: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, camera: str, cfg) -> "CameraHealth":
        keys = ("check_every_s", "dark_below", "overexposed_above", "soft_ratio", "frozen_after_s", "shift_frac",
                "persist_checks")
        return cls(camera=camera, **{k: cfg[k] for k in keys if cfg and k in cfg})

    def update(self, frame: np.ndarray, ts: float | None = None, bad_pixels: int = 0) -> list[dict]:
        ts = time.time() if ts is None else float(ts)
        if self._last is not None and ts - self._last < self.check_every_s:
            return []
        self._last = ts
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        h = max(8, int(gray.shape[0] * self.width / max(gray.shape[1], 1)))
        small = cv2.resize(gray, (self.width, h), interpolation=cv2.INTER_AREA)
        m: dict[str, float] = {}
        m["brightness"] = float(small.mean())
        m["overexposed"] = float((small > 250).mean())
        m["sharpness"] = float(cv2.Laplacian(small, cv2.CV_32F).var())
        cells = [c for row in np.array_split(small, 4, 0) for c in np.array_split(row, 4, 1)]
        m["flat_cells"] = float(np.mean([c.std() < 3.0 for c in cells]))
        dark = m["brightness"] < self.dark_below

        frozen = self._prev is not None and self._prev.shape == small.shape and not np.any(cv2.absdiff(small, self._prev))
        if frozen:
            self._frozen_since = self._frozen_since if self._frozen_since is not None else ts
        else:
            self._frozen_since = None
        self._prev = small

        if not dark and m["overexposed"] < self.overexposed_above:
            if len(self._sharp_hist) < self.baseline_checks:
                self._sharp_hist.append(m["sharpness"])
                if len(self._sharp_hist) == self.baseline_checks:
                    self._ref = small.astype(np.float32)
        base = float(np.median(self._sharp_hist)) if len(self._sharp_hist) >= self.baseline_checks else None

        shift = 0.0
        if self._ref is not None and not dark and self._ref.shape == small.shape:
            win = cv2.createHanningWindow(small.shape[::-1], cv2.CV_32F)
            (dx, dy), resp = cv2.phaseCorrelate(self._ref, small.astype(np.float32), win)
            if resp > 0.05:
                shift = float(np.hypot(dx, dy)) / self.width
        m["shift"] = shift
        self.last_metrics = m

        conds = {
            "too_dark": dark,
            "overexposed": m["overexposed"] >= self.overexposed_above,
            "soft_image": base is not None and not dark and m["sharpness"] < self.soft_ratio * base,
            "view_blocked": not dark and m["flat_cells"] >= 0.5,
            "frozen_feed": self._frozen_since is not None and ts - self._frozen_since >= self.frozen_after_s,
            "camera_moved": shift >= self.shift_frac,
            "sensor_defects": bad_pixels > self.max_defects_frac * gray.size,
        }
        alerts = []
        for name, bad in conds.items():
            c = self._checks.setdefault(name, _Check())
            c.count = c.count + 1 if bad else 0
            need = 1 if name in ("frozen_feed", "sensor_defects") else self.persist_checks
            if bad and not c.active and c.count >= need:
                c.active = True
                alerts.append(self._alert(name, "raised", ts, m))
            elif not bad and c.active:
                c.active = False
                alerts.append(self._alert(name, "cleared", ts, m))
        return alerts

    def _alert(self, name: str, state: str, ts: float, m: dict) -> dict:
        return {"camera": self.camera, "check": name, "state": state, "ts": round(ts, 3),
                "message": MESSAGES[name] if state == "raised" else f"recovered: {name}",
                "metrics": {k: round(v, 4) for k, v in m.items()}}

    def status(self) -> dict:
        return {"camera": self.camera, "active": sorted(k for k, c in self._checks.items() if c.active),
                "metrics": {k: round(v, 4) for k, v in self.last_metrics.items()}}
