"""Image clean-up in front of detection and OCR, as a camera ISP would do it.

* Bad pixel correction (BPC, also called defective pixel correction). Sensors develop hot
  (always bright), dead (always dark) and stuck pixels; demosaicing smears each one into a
  small coloured cluster. On a 30 px tall plate a single such dot can turn 0 into 8 or C
  into G. Indian MLFF camera specifications (IHMCL) list BPC as a required feature; this
  module does it in software for cameras that lack it or whose sensors age.
    - dynamic: a pixel brighter (darker) than every neighbour by more than a threshold is
      an outlier and is replaced by the neighbourhood median. A 5x5 ring catches 2x2
      demosaic clusters too. Character strokes are never isolated, so they are kept.
    - static, learned in video: pixels flagged in most frames (hot/dead) or that do not
      change while their neighbourhood does (stuck) form a per-camera defect map, which
      is corrected in every frame and can be saved and reloaded.
* Edge enhancement: an unsharp-masked copy of soft plate crops, added as an extra OCR
  view. The fusion step keeps it only when it agrees with the other views, so it can
  help a blurred plate without hurting a sharp one.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("drishti.imaging")

_RING3 = np.ones((3, 3), np.uint8)
_RING3[1, 1] = 0
_RING5 = np.ones((5, 5), np.uint8)
_RING5[1:4, 1:4] = 0


def outlier_mask(img: np.ndarray, threshold: float = 40.0, cluster: bool = True) -> np.ndarray:
    """Boolean (H, W) mask of isolated hot or dead pixels (any channel)."""
    masks = []
    for ring in ([_RING3, _RING5] if cluster else [_RING3]):
        nb_max = cv2.dilate(img, ring)
        nb_min = cv2.erode(img, ring)
        a = img.astype(np.int16)
        hot = a > nb_max.astype(np.int16) + threshold
        dead = a < nb_min.astype(np.int16) - threshold
        m = hot | dead
        masks.append(m.any(axis=2) if m.ndim == 3 else m)
    out = masks[0]
    for m in masks[1:]:
        out = out | m
    return out


def correct_pixels(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace masked pixels by the median of their 5x5 neighbourhood (robust to a few bad neighbours)."""
    if not mask.any():
        return img
    out = img.copy()
    med = cv2.medianBlur(img, 5)
    out[mask] = med[mask]
    return out


class BadPixelCorrector:
    def __init__(self, threshold: float = 40.0, cluster: bool = True, learn: bool = True,
                 persist: float = 0.6, min_frames: int = 100, map_path: str | Path | None = None):
        self.threshold, self.cluster, self.learn = float(threshold), bool(cluster), bool(learn)
        self.persist, self.min_frames = float(persist), int(min_frames)
        self.map_path = Path(map_path).expanduser() if map_path else None
        self.static: np.ndarray | None = None
        self._hits: np.ndarray | None = None
        self._act: np.ndarray | None = None
        self._prev: np.ndarray | None = None
        self.frames = 0
        if self.map_path is not None and self.map_path.exists():
            try:
                self.static = np.load(self.map_path)
                log.info("loaded %d known bad pixels from %s", int(self.static.sum()), self.map_path)
            except Exception as exc:  # a corrupt map must not stop the camera
                log.warning("could not load bad pixel map %s: %s", self.map_path, exc)

    @property
    def n_static(self) -> int:
        return 0 if self.static is None else int(self.static.sum())

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        dyn = outlier_mask(frame, self.threshold, self.cluster)
        if self.static is not None and self.static.shape != dyn.shape:
            self.static = None  # resolution changed: relearn
        if self.learn:
            self._accumulate(frame, dyn)
        mask = dyn if self.static is None else (dyn | self.static)
        return correct_pixels(frame, mask)

    def _accumulate(self, frame: np.ndarray, dyn: np.ndarray) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self._hits is None or self._hits.shape != dyn.shape:
            self._hits = np.zeros(dyn.shape, np.uint16)
            self._act = np.zeros(dyn.shape, np.float32)
            self._prev, self.frames = None, 0
        self._hits += dyn.astype(np.uint16)
        if self._prev is not None:
            self._act += cv2.absdiff(gray, self._prev).astype(np.float32)
        self._prev = gray
        self.frames += 1
        if self.frames >= self.min_frames and self.frames % self.min_frames == 0:
            self._update_static()

    def _update_static(self) -> None:
        n = self.frames
        hot_dead = self._hits >= self.persist * n
        # stuck: no temporal change while the 5x5 neighbourhood changes by > 2 levels per frame on average
        local = cv2.blur(self._act, (5, 5))
        stuck = (self._act < 0.1 * local) & (local > 2.0 * (n - 1))
        new = hot_dead | stuck
        before = self.n_static
        self.static = new if self.static is None else (self.static | new)
        if self.n_static != before:
            log.info("bad pixel map: %d pixels after %d frames", self.n_static, n)
            if self.map_path is not None:
                try:
                    self.map_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(self.map_path, self.static)
                except OSError as exc:
                    log.warning("could not save bad pixel map: %s", exc)
        self._hits[:] = 0
        self._act[:] = 0
        self.frames = 0


def edge_enhance(crop: np.ndarray, amount: float = 0.8, sigma_frac: float = 0.03) -> np.ndarray:
    """Unsharp mask scaled to the crop height (so a 30 px and a 120 px plate get the same treatment)."""
    sigma = max(0.6, sigma_frac * crop.shape[0])
    blur = cv2.GaussianBlur(crop, (0, 0), sigma)
    return cv2.addWeighted(crop, 1.0 + amount, blur, -amount, 0)


def sharpness(crop: np.ndarray) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    g = cv2.resize(gray, (max(8, int(gray.shape[1] * 48 / max(gray.shape[0], 1))), 48), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


# ------------------------------------------------------------------ rescue views
# Hard plates (small, dark, low contrast, faded) get a second look through several
# cleaned-up copies. Each copy is read separately and the fusion step keeps only
# what the copies agree on, so one enhancement inventing a stroke cannot win alone.

def zoom(crop: np.ndarray, factor: float = 3.0) -> np.ndarray:
    """Bicubic upscaling plus a light unsharp mask: more pixels per stroke before the OCR resize."""
    big = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    return edge_enhance(big, amount=0.6)


def local_contrast(crop: np.ndarray) -> np.ndarray:
    """CLAHE on lightness only: lifts faded or shadowed characters without shifting colours."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def auto_gamma(crop: np.ndarray, target: float = 0.5) -> np.ndarray:
    """Brighten dark crops / darken washed-out ones so mid-grey lands near `target`."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean = float(np.clip(gray.mean() / 255.0, 0.02, 0.98))
    gamma = float(np.clip(np.log(target) / np.log(mean), 0.3, 3.0))
    lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(crop, lut)


def best_channel_gray(crop: np.ndarray) -> np.ndarray:
    """Grey image from the colour channel with the most contrast, stretched to full range.
    Yellow and green plates lose contrast in plain grey; their best channel keeps it."""
    chans = cv2.split(crop)
    ch = max(chans, key=lambda c: float(c.std()))
    ch = cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(ch, cv2.COLOR_GRAY2BGR)


RESCUE_VIEWS = {
    "zoom": zoom,
    "contrast": local_contrast,
    "gamma": lambda c: local_contrast(auto_gamma(c)),
    "gray": best_channel_gray,
    "zoom-contrast": lambda c: zoom(local_contrast(auto_gamma(c))),
}
