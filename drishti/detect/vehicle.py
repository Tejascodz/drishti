"""Vehicle detector: RF-DETR (default, Apache-2.0) or any YOLO-family COCO export."""

from __future__ import annotations

import cv2
import numpy as np

from .. import hub
from ..imgutils import sigmoid
from ..runtime import input_hw, make_session
from ..types import BBox, Detection
from .plate import YoloDetector

# COCO ids: RF-DETR uses the original 91-way 1-indexed ids, Ultralytics the 80-way 0-indexed ones.
_RFDETR_VEHICLES = {2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}
_YOLO_VEHICLES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class VehicleDetector:
    def __init__(self, model: str, runtime_cfg: dict | None = None, conf: float = 0.4, cache_dir: str | None = None):
        self.fmt = hub.spec_format(model)
        self.conf = conf
        self._yolo = None
        if self.fmt == "rfdetr":
            path = hub.resolve(model, "model", cache_dir)
            # CoreML is often slower or incompatible with DETR graphs; other accelerators are kept.
            self.session = make_session(path, runtime_cfg, allow_coreml=False)
            self.input_name = self.session.get_inputs()[0].name
            self.hw = input_hw(self.session, (384, 384))
        else:
            self._yolo = YoloDetector(model, runtime_cfg, conf=conf, num_classes=80, class_names=_YOLO_VEHICLES,
                                      tiling="off", cache_dir=cache_dir, default_label="vehicle")

    def __call__(self, img_bgr: np.ndarray) -> list[Detection]:
        if self._yolo is not None:
            return _dedupe(self._yolo(img_bgr))
        H, W = img_bgr.shape[:2]
        h, w = self.hw
        x = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_LINEAR)
        x = ((x.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None].astype(np.float32)
        boxes, logits = self.session.run(None, {self.input_name: x})[:2]
        boxes, logits = boxes[0], logits[0]
        ids = np.array([i for i in sorted(_RFDETR_VEHICLES) if i < logits.shape[1]])
        probs = sigmoid(logits[:, ids])
        best = probs.argmax(axis=1)
        score = probs[np.arange(len(probs)), best]
        dets = []
        for q in np.where(score >= self.conf)[0]:
            cx, cy, bw, bh = boxes[q]
            bb = BBox((cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H).clip(W, H)
            dets.append(Detection(bbox=bb, score=float(score[q]), label=_RFDETR_VEHICLES[int(ids[best[q]])]))
        return _dedupe(dets)


def _dedupe(dets: list[Detection], thr: float = 0.7) -> list[Detection]:
    dets = sorted(dets, key=lambda d: -d.score)
    kept: list[Detection] = []
    for d in dets:
        if all(d.bbox.iou(k.bbox) < thr for k in kept):
            kept.append(d)
    return kept
