"""Fixed-slot plate OCR (fast-plate-ocr compatible ONNX + plate config YAML).

Returns the full (S, V) probability matrix per crop, which is what makes
format-constrained decoding and logit-level multi-frame fusion possible.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import yaml

from .. import hub
from ..imgutils import softmax
from ..runtime import has_dynamic_batch, make_session

log = logging.getLogger(__name__)

_INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos4": cv2.INTER_LANCZOS4,
}


class SlotOCR:
    def __init__(self, model: str, config: str | None = None, runtime_cfg: dict | None = None, cache_dir: str | None = None):
        path = hub.resolve(model, "model", cache_dir)
        cfg_path = Path(config) if config else (hub.resolve(model, "config", cache_dir) if model in hub.REGISTRY else None)
        if cfg_path is None:
            guess = path.with_name(path.stem + "_plate_config.yaml")
            if not guess.exists():
                raise FileNotFoundError("OCR plate config YAML is required for a local OCR model (models.ocr_config)")
            cfg_path = guess
        self.cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        self.alphabet: str = self.cfg["alphabet"]
        self.pad_char: str = self.cfg["pad_char"]
        self.slots: int = int(self.cfg["max_plate_slots"])
        self.h: int = int(self.cfg["img_height"])
        self.w: int = int(self.cfg["img_width"])
        self.keep_aspect: bool = bool(self.cfg.get("keep_aspect_ratio", False))
        self.interp = _INTERP.get(str(self.cfg.get("interpolation", "linear")), cv2.INTER_LINEAR)
        self.rgb = str(self.cfg.get("image_color_mode", "rgb")).lower() == "rgb"
        self.pad_color = tuple(self.cfg.get("padding_color", (114, 114, 114)))
        self.regions: list[str] = list(self.cfg.get("plate_regions") or [])

        # CoreML recompiles the graph whenever the batch size changes and can stall for minutes on
        # this small model; CPU reads a crop in a few ms. Other accelerators are kept.
        self.session = make_session(path, runtime_cfg, allow_coreml=False)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_float = "float" in inp.type
        self.dynamic_batch = has_dynamic_batch(self.session)
        names = [o.name for o in self.session.get_outputs()]
        self.plate_out = "plate" if "plate" in names else names[0]
        self.region_out = "region" if "region" in names else None
        log.info("ocr %s slots=%d alphabet=%d", path.name, self.slots, len(self.alphabet))

    def preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        img = crop_bgr
        if self.keep_aspect:
            h, w = img.shape[:2]
            r = min(self.h / h, self.w / w)
            nh, nw = max(1, int(round(h * r))), max(1, int(round(w * r)))
            img = cv2.resize(img, (nw, nh), interpolation=self.interp)
            top = (self.h - nh) // 2
            left = (self.w - nw) // 2
            img = cv2.copyMakeBorder(img, top, self.h - nh - top, left, self.w - nw - left, cv2.BORDER_CONSTANT, value=self.pad_color)
        else:
            img = cv2.resize(img, (self.w, self.h), interpolation=self.interp)
        if self.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None]
        return img

    def __call__(self, crops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray | None]:
        """crops (BGR) -> probs (N, S, V) and optional region probs (N, R)."""
        if not crops:
            return np.zeros((0, self.slots, len(self.alphabet)), np.float32), None
        batch = np.stack([self.preprocess(c) for c in crops])
        batch = batch.astype(np.float32) if self.input_float else batch.astype(np.uint8)
        outs = [self.plate_out] + ([self.region_out] if self.region_out else [])
        if self.dynamic_batch:
            res = self.session.run(outs, {self.input_name: batch})
            plates = res[0]
            regions = res[1] if self.region_out else None
        else:
            parts = [self.session.run(outs, {self.input_name: batch[i : i + 1]}) for i in range(len(batch))]
            plates = np.concatenate([p[0] for p in parts])
            regions = np.concatenate([p[1] for p in parts]) if self.region_out else None
        plates = plates.reshape(len(crops), self.slots, len(self.alphabet)).astype(np.float64)
        sums = plates.sum(axis=-1)
        if not np.allclose(sums, 1.0, atol=1e-2) or plates.min() < 0:
            plates = softmax(plates, axis=-1)
        return plates, regions

    def region_name(self, region_probs: np.ndarray | None) -> tuple[str | None, float]:
        if region_probs is None or not self.regions:
            return None, 0.0
        i = int(np.argmax(region_probs))
        return (self.regions[i] if i < len(self.regions) else None), float(region_probs[i])
