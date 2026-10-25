"""YOLO-family ONNX detector (plates by default, any classes on request).

Supported output layouts (auto-detected from the graph):
  yolov9-e2e  (K, 7)            [batch, x1, y1, x2, y2, class, score]   (open-image-models)
  yolo-e2e    (N, M, 6 + 3K)    [x1, y1, x2, y2, score, class, kpts...] (YOLO26 end-to-end, detect or pose)
  yolo-raw    (N, 4 + nc + 3K, A) center-xywh head, NMS applied here    (YOLO11 / YOLO26 exported with nms=None)

For YOLO26-pose models trained on 4 plate corners (training/detector), the
keypoints are returned so the crop can be perspective-rectified before OCR.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .. import hub
from ..imgutils import letterbox, nms, tiles
from ..runtime import has_dynamic_batch, input_hw, make_session
from ..types import BBox, Detection

log = logging.getLogger(__name__)


class YoloDetector:
    def __init__(
        self,
        model: str,
        runtime_cfg: dict | None = None,
        conf: float = 0.3,
        fmt: str | None = None,
        num_classes: int = 1,
        num_keypoints: int | None = None,
        class_names: dict[int, str] | None = None,
        tiling: str = "auto",
        tile_overlap: float = 0.2,
        cache_dir: str | None = None,
        default_label: str = "plate",
    ):
        path = hub.resolve(model, "model", cache_dir)
        self.session = make_session(path, runtime_cfg)
        self.input_name = self.session.get_inputs()[0].name
        self.hw = input_hw(self.session, (640, 640))
        self.dynamic_batch = has_dynamic_batch(self.session)
        self.conf = conf
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.class_names = class_names
        self.tiling = str(tiling).lower()
        self.tile_overlap = tile_overlap
        self.default_label = default_label
        self.fmt = fmt or hub.spec_format(model) or self._sniff_format()
        log.info("detector %s fmt=%s input=%s", path.name, self.fmt, self.hw)

    def _sniff_format(self) -> str:
        shape = self.session.get_outputs()[0].shape
        if len(shape) == 2 and shape[-1] == 7:
            return "yolov9-e2e"
        if len(shape) == 3:
            last = shape[-1]
            if isinstance(last, int) and 6 <= last < 128 and (last - 6) % 3 == 0:
                return "yolo-e2e"
            return "yolo-raw"
        raise ValueError(f"Unrecognised detector output shape {shape}")

    # ------------------------------------------------------------- inference
    def _prep(self, img_bgr: np.ndarray):
        lb, r, (dw, dh) = letterbox(img_bgr, self.hw)
        x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
        return x, r, dw, dh

    def _run(self, batch: np.ndarray) -> list[np.ndarray]:
        if self.dynamic_batch:
            return self._split(self.session.run(None, {self.input_name: batch})[0], batch.shape[0])
        outs: list[np.ndarray] = []
        for i in range(batch.shape[0]):
            outs.extend(self._split(self.session.run(None, {self.input_name: batch[i : i + 1]})[0], 1))
        return outs

    def _split(self, out: np.ndarray, n: int) -> list[np.ndarray]:
        if self.fmt == "yolov9-e2e":
            return [out[out[:, 0] == b] if out.size else out for b in range(n)]
        return [out[b] for b in range(out.shape[0])]

    def _decode(self, rows: np.ndarray, r: float, dw: float, dh: float, x0: int = 0, y0: int = 0):
        """One image's raw rows -> (boxes xyxy, scores, class ids, keypoints|None) in source coords."""
        kpts = None
        if self.fmt == "yolov9-e2e":
            if rows.size == 0:
                return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int), None
            boxes, scores, cls = rows[:, 1:5].astype(np.float32), rows[:, 6].astype(np.float32), rows[:, 5].astype(int)
        elif self.fmt == "yolo-e2e":
            boxes, scores, cls = rows[:, :4].astype(np.float32), rows[:, 4].astype(np.float32), rows[:, 5].astype(int)
            if rows.shape[1] > 6:
                kpts = rows[:, 6:].reshape(len(rows), -1, 3).astype(np.float32)
        else:  # yolo-raw: (C, A)
            pred = rows.T.astype(np.float32)
            nk = self.num_keypoints
            if nk is None:
                extra = pred.shape[1] - 4 - self.num_classes
                nk = extra // 3 if extra > 0 and extra % 3 == 0 else 0
            cls_scores = pred[:, 4 : 4 + self.num_classes]
            cls = cls_scores.argmax(axis=1)
            scores = cls_scores[np.arange(len(cls)), cls]
            xywh = pred[:, :4]
            boxes = np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], axis=1)
            keep = scores >= self.conf
            boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
            if nk:
                kpts = pred[keep, 4 + self.num_classes : 4 + self.num_classes + 3 * nk].reshape(-1, nk, 3)
            k = nms(boxes, scores, 0.5)
            boxes, scores, cls = boxes[k], scores[k], cls[k]
            kpts = kpts[k] if kpts is not None else None

        keep = scores >= self.conf
        if self.class_names is not None:
            keep &= np.isin(cls, list(self.class_names))
        boxes, scores, cls = boxes[keep].copy(), scores[keep], cls[keep]
        kpts = kpts[keep].copy() if kpts is not None else None
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / r + x0
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / r + y0
        if kpts is not None:
            kpts[..., 0] = (kpts[..., 0] - dw) / r + x0
            kpts[..., 1] = (kpts[..., 1] - dh) / r + y0
        return boxes, scores, cls, kpts

    def _should_tile(self, w: int, h: int) -> bool:
        if self.tiling in ("on", "true"):
            return True
        if self.tiling == "auto":
            return max(w, h) >= 3.5 * max(self.hw)
        return False

    def __call__(self, img_bgr: np.ndarray) -> list[Detection]:
        H, W = img_bgr.shape[:2]
        regions = [(0, 0, W, H)]
        if self._should_tile(W, H):
            regions += tiles(W, H, int(2 * max(self.hw)), self.tile_overlap)
        preps = [self._prep(img_bgr[y1:y2, x1:x2]) for (x1, y1, x2, y2) in regions]
        outs = self._run(np.stack([p[0] for p in preps]))

        parts = [self._decode(rows, r, dw, dh, reg[0], reg[1]) for reg, (_, r, dw, dh), rows in zip(regions, preps, outs)]
        boxes = np.concatenate([p[0] for p in parts])
        scores = np.concatenate([p[1] for p in parts])
        cls = np.concatenate([p[2] for p in parts])
        kpts = np.concatenate([p[3] for p in parts]) if all(p[3] is not None for p in parts) else None
        if len(regions) > 1 and len(boxes):
            keep = nms(boxes, scores, 0.45)
            boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
            kpts = kpts[keep] if kpts is not None else None

        dets = []
        for i in range(len(boxes)):
            bb = BBox.from_array(boxes[i]).clip(W, H)
            if bb.w < 2 or bb.h < 2:
                continue
            label = self.class_names.get(int(cls[i]), self.default_label) if self.class_names else self.default_label
            dets.append(Detection(bbox=bb, score=float(scores[i]), label=label, keypoints=None if kpts is None else kpts[i]))
        dets.sort(key=lambda d: -d.score)
        return dets


class PlateDetector(YoloDetector):
    """Plate detector: one class, optional 4 corner keypoints (TL, TR, BR, BL)."""

    def __init__(self, model: str, runtime_cfg: dict | None = None, conf: float = 0.3, tiling: str = "auto",
                 tile_overlap: float = 0.2, cache_dir: str | None = None, fmt: str | None = None):
        super().__init__(model, runtime_cfg, conf=conf, fmt=fmt, num_classes=1, tiling=tiling,
                         tile_overlap=tile_overlap, cache_dir=cache_dir, default_label="plate")
