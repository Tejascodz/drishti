"""The recognition engine: still images and video streams.

Image path:   detect plates -> several crop views per plate (padding levels,
              corner rectification or deskew, two-line stitch) -> one batched OCR
              pass -> per-view constrained decode -> consistency-gated fusion.
Video path:   the same per frame, plus ByteTrack-style tracking; each track keeps
              its observations and best crops, and on track end the evidence is
              fused across frames (logit-level + multi-frame image restoration)
              into a single event. High-confidence tracks can emit early.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from .config import Config, expand_path, load_config
from .detect import PlateDetector, VehicleDetector
from .formats import FormatGrammar
from .health import CameraHealth
from .tolling import confidence_level, toll_decision
from .fusion import Observation, align_and_stack, combined_confidence, fuse
from .geometry import corners_from_keypoints, crop_box, deskew, level_plate, stitch_two_lines, two_line_views, warp_quad
from .ocr import SlotOCR
from .ocr.vlm import from_config as vlm_from_config
from .privacy import blur_region, plate_digest
from .quality import dominant_color, plate_quality
from .store import EventStore
from .track import ByteTracker, Track
from .types import BBox, Detection, PlateEvent, PlateRead, PlateResult, VehicleInfo
from .viz import draw_plate
from .watchlist import Watchlist
from .zones import Zones

log = logging.getLogger(__name__)


class SuperResolver:
    """Wraps any x2/x4 single-image SR ONNX (NCHW float RGB in [0,1])."""

    def __init__(self, path: str, runtime_cfg: dict):
        from .runtime import make_session

        self.session = make_session(expand_path(path), runtime_cfg)
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, crop: np.ndarray) -> np.ndarray:
        x = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        y = self.session.run(None, {self.input_name: x})[0][0].transpose(1, 2, 0)
        return cv2.cvtColor(np.clip(y * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


@dataclass
class _Read:
    """Everything learned from one detection in one frame."""

    det: Detection
    observations: list[Observation]  # one per crop view
    read: PlateRead
    quality: float
    quad: np.ndarray | None
    two_line: bool
    crop: np.ndarray  # middle-padding view, used for restoration / evidence
    frame_obs: Observation | None = None  # views fused into one observation for this frame


class Drishti:
    def __init__(self, cfg: Config | None = None, config_path: str | None = None, **overrides):
        self.cfg = cfg if cfg is not None else load_config(config_path, overrides or None)
        c = self.cfg
        rt, cache = c.runtime, c.models.cache_dir
        self.plate_det = PlateDetector(c.models.plate_detector, rt, conf=c.detection.plate_conf,
                                       tiling=c.detection.tiling, tile_overlap=c.detection.tile_overlap, cache_dir=cache)
        self.vehicle_det = (VehicleDetector(c.models.vehicle_detector, rt, conf=c.detection.vehicle_conf, cache_dir=cache)
                            if c.models.vehicle_detector else None)
        self.ocr = SlotOCR(c.models.ocr, c.models.ocr_config, rt, cache)
        region = str(c.recognition.region)
        self.mode = "free" if region in ("generic", "free", "none") else str(c.recognition.get("mode", "auto"))
        self.grammar = FormatGrammar.from_pack(None if self.mode == "free" else region, self.ocr.alphabet, self.ocr.pad_char)
        self.sr = SuperResolver(c.models.sr_model, rt) if c.models.get("sr_model") else None
        self.vlm = vlm_from_config(c.vlm, region) if c.vlm.enabled else None
        calib_path = c.recognition.get("calibration")
        self.calib = json.loads(expand_path(calib_path).read_text()) if calib_path else None
        self.views = [tuple(v) for v in c.ocr.views]
        # small plates: wide paddings shrink the characters inside the fixed OCR input and add
        # misleading reads, so only tight crops are used below this width
        self.small_views = [tuple(v) for v in (c.ocr.get("small_views") or c.ocr.views)]
        self.small_plate_px = float(c.ocr.get("small_plate_px", 0) or 0)
        im = c.get("imaging") or {}
        self.imaging = im
        self.bpc = self.make_bpc(learn=False) if str(im.get("bad_pixel", "off")).lower() in ("on", "true") else None

    def make_bpc(self, learn: bool, camera: str | None = None):
        from .imaging import BadPixelCorrector

        im = self.imaging
        map_path = None
        if learn and camera and im.get("bpc_map_dir"):
            safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in camera)
            map_path = expand_path(im.get("bpc_map_dir")) / f"{safe}.npy"
        return BadPixelCorrector(im.get("bpc_threshold", 40), im.get("bpc_cluster", True),
                                 learn and bool(im.get("bpc_learn", True)), map_path=map_path)

    # ----------------------------------------------------------------- reads
    def _decode(self, logp: np.ndarray):
        r = self.cfg.recognition
        return self.grammar.decode(logp, mode=self.mode, fallback_nats=r.auto_fallback_nats, top_k=r.top_k_alternatives)

    def _views(self, frame: np.ndarray, det: Detection):
        oc = self.cfg.ocr
        quad = corners_from_keypoints(det.keypoints, det.bbox) if oc.rectify else None
        views: list[tuple[str, np.ndarray, float]] = []
        pads = self.small_views if det.bbox.w < self.small_plate_px else self.views
        for fx, fy in pads:
            img = None
            if quad is not None:
                try:
                    img = warp_quad(frame, quad, fx, fy)
                except ValueError:
                    img = None
            if img is None:
                img = crop_box(frame, det.bbox, fx, fy)
            if img.size:
                views.append((f"pad{fx:.2f}", img, 1.0))
        if not views:
            return [], quad, False
        if oc.rectify and quad is None:
            mid = views[len(views) // 2][1]
            ds = deskew(mid)
            if ds is not None:
                views.append(("deskew", ds, 1.0))
        two_line = False
        if str(oc.two_line) != "off":
            aspect = det.bbox.w / max(det.bbox.h, 1e-6)
            if aspect < 2.8:
                # candidate single-line renderings of a stacked plate; OCR evidence arbitrates
                mid = views[min(1, len(views) - 1)][1]
                stitched = two_line_views(mid)
                for k, img in enumerate(stitched):
                    views.append((f"stitch{k}", img, 1.0))
                if not stitched and aspect < 2.2:
                    views.append(("stitch-mid", stitch_two_lines(level_plate(mid), mid.shape[0] // 2), 1.0))
        ee = str(self.imaging.get("edge_enhance", "off")).lower()
        if ee in ("on", "auto", "true"):
            from .imaging import edge_enhance, sharpness

            mid = views[min(1, len(views) - 1)][1]
            if ee != "auto" or sharpness(mid) < float(self.imaging.get("edge_enhance_below", 400)):
                views.append(("enhanced", edge_enhance(mid), 1.0))
        if self.sr is not None and det.bbox.h < oc.sr_below_height:
            views = [(n, self.sr(v), w) for n, v, w in views]
        return views, quad, two_line

    def _read_many(self, frame: np.ndarray, dets: list[Detection], frame_weight: bool = False) -> list[_Read | None]:
        """Batched OCR over all views of all detections in one frame."""
        plans = []
        flat: list[np.ndarray] = []
        for det in dets:
            if det.bbox.h < self.cfg.detection.min_plate_height:
                plans.append(None)
                continue
            views, quad, two_line = self._views(frame, det)
            if not views:
                plans.append(None)
                continue
            start = len(flat)
            flat.extend(v for _, v, _ in views)
            plans.append((det, views, quad, two_line, start))
        if not flat:
            return [None] * len(dets)
        probs, _ = self.ocr(flat)
        logps = np.log(probs + 1e-9)

        out: list[_Read | None] = []
        for plan in plans:
            if plan is None:
                out.append(None)
                continue
            det, views, quad, two_line, start = plan
            q = plate_quality(views[0][1], det.bbox.h)
            base_w = det.score * (0.2 + q) if frame_weight else 1.0
            obs = []
            for k, (name, _img, w) in enumerate(views):
                lp = logps[start + k]
                dec = self._decode(lp)
                # views that read cleanly under the grammar count for more
                obs.append(Observation(lp, base_w * w * (0.1 + dec.prob), dec, name, q))
            fr = fuse(obs, self.grammar, self.mode, self.cfg.recognition.auto_fallback_nats, self.cfg.recognition.top_k_alternatives)
            read = self._to_read(fr, n_obs=1, quality=q, n_views=len(views))
            stitched = [o for o in obs if o.source.startswith("stitch")]
            flat_ = [o for o in obs if not o.source.startswith("stitch")]
            is_two = bool(stitched) and max(o.weight for o in stitched) > max((o.weight for o in flat_), default=0.0)
            mid = views[min(1, len(views) - 1)][1]
            evidence = next((v for n, v, _ in views if n.startswith("stitch")), mid) if is_two else mid
            frame_ob = Observation(fr.logp, (det.score * (0.2 + q) if frame_weight else 1.0) * (0.1 + fr.decode.prob),
                                   fr.decode, "frame", q) if fr.logp is not None else None
            out.append(_Read(det, obs, read, q, quad, is_two or two_line, evidence, frame_ob))
        if self.cfg.ocr.get("rescue", False):
            self._rescue(out, frame_weight)
        return out

    def _rescue(self, reads: list, frame_weight: bool) -> None:
        """Second look at unreadable / low-confidence plates through enhanced copies of the crop.

        The copies are fused with the original views by the usual consistency-gated fusion, so an
        enhancement only changes the answer when the copies agree with each other and the original
        evidence. A rescued read is capped below the automatic-charge threshold: it is a suggestion
        for the reviewer, not a decision."""
        from .imaging import RESCUE_VIEWS

        oc, rc = self.cfg.ocr, self.cfg.recognition
        below = float(oc.get("rescue_below", 0.8))
        todo = [i for i, r in enumerate(reads)
                if r is not None and (r.read.text is None or r.read.confidence < below)]
        if not todo:
            return
        flat, spans = [], []
        for i in todo:
            r = reads[i]
            start = len(flat)
            names = []
            for name, fn in RESCUE_VIEWS.items():
                try:
                    img = fn(r.crop)
                except cv2.error:
                    continue
                flat.append(img)
                names.append(name)
                if r.two_line or r.det.bbox.w / max(r.det.bbox.h, 1e-6) < 2.8:
                    for k, st in enumerate(two_line_views(img)):
                        flat.append(st)
                        names.append(f"{name}-stitch{k}")
            spans.append((i, start, names))
        if not flat:
            return
        probs, _ = self.ocr(flat)
        logps = np.log(probs + 1e-9)
        cap = float(oc.get("rescue_max_confidence", 0.94))
        for i, start, names in spans:
            r = reads[i]
            base_w = r.det.score * (0.2 + r.quality) if frame_weight else 1.0
            extra = []
            for k, name in enumerate(names):
                lp = logps[start + k]
                dec = self._decode(lp)
                extra.append(Observation(lp, base_w * 0.7 * (0.1 + dec.prob), dec, f"rescue-{name}", r.quality))
            fr = fuse(r.observations + extra, self.grammar, self.mode, rc.auto_fallback_nats, rc.top_k_alternatives)
            new = self._to_read(fr, n_obs=1, quality=r.quality, n_views=len(r.observations) + len(extra))
            if new.text is None or new.confidence <= r.read.confidence:
                continue
            new.confidence = min(new.confidence, cap)
            new.low_confidence = new.confidence < rc.min_confidence
            new.level = confidence_level(new.confidence, (self.cfg.get("tolling") or {}).get("levels"))
            new.rescued = True
            r.read, r.observations = new, r.observations + extra
            if fr.logp is not None:
                r.frame_obs = Observation(fr.logp, base_w * (0.1 + fr.decode.prob), fr.decode, "frame-rescued", r.quality)

    def _to_read(self, fr, n_obs: int, quality: float, n_views: int) -> PlateRead:
        rc = self.cfg.recognition
        d = fr.decode
        conf = combined_confidence(d.prob, fr.agreement, n_obs, quality, d.format_penalty, self.calib)
        read = PlateRead(
            text=d.text, confidence=conf, fmt=d.fmt, char_conf=d.char_probs, alternatives=d.alternatives,
            raw_text=d.raw_text, format_penalty=d.format_penalty, n_views=n_views, agreement=fr.agreement,
            string_prob=d.prob,
        )
        read.low_confidence = conf < rc.min_confidence
        if conf < rc.abstain_below or not d.text:
            read.abstained = True
            read.text = None
        read.level = confidence_level(conf if read.text else 0.0, (self.cfg.get("tolling") or {}).get("levels"))
        return read

    def _maybe_verify(self, read: PlateRead, crop: np.ndarray) -> PlateRead:
        if self.vlm is None:
            return read
        if read.text is None:
            return self._vlm_read_unreadable(read, crop)
        if read.confidence >= self.cfg.vlm.trigger_below:
            return read
        answer = self.vlm.verify(crop, read.text, read.alternatives, self.grammar)
        if answer:
            if answer != read.text:
                read.raw_text = read.raw_text or read.text
                read.text = answer
            read.verified_by = "vlm"
            read.confidence = max(read.confidence, self.cfg.vlm.trigger_below)
            read.low_confidence = read.confidence < self.cfg.recognition.min_confidence
            read.level = confidence_level(read.confidence, (self.cfg.get("tolling") or {}).get("levels"))
        return read

    def _vlm_read_unreadable(self, read: PlateRead, crop: np.ndarray) -> PlateRead:
        """The OCR found nothing legible: ask the VLM to read the crop on its own. Its answer counts
        only if it is a valid plate under the active grammar, and it is always capped at
        `vlm.unreadable_confidence` (Poor), so it goes to manual review and never auto-charges."""
        vc = self.cfg.vlm
        if not vc.get("read_unreadable", True) or crop is None or crop.size == 0:
            return read
        try:
            ans = self.vlm.read(crop)
        except Exception as exc:  # network / server errors must never break recognition
            log.warning("VLM read skipped: %s", exc)
            return read
        fmt = self.grammar.is_valid(ans.text) if ans.text else None
        if fmt is None:
            return read
        read.text, read.fmt, read.abstained, read.verified_by = ans.text, fmt, False, "vlm-only"
        read.confidence = min(float(vc.get("unreadable_confidence", 0.5)), self.cfg.recognition.min_confidence)
        read.alternatives = [(ans.text, read.confidence)]
        read.low_confidence = True
        read.level = confidence_level(read.confidence, (self.cfg.get("tolling") or {}).get("levels"))
        return read

    def read_padded_crop(self, img: np.ndarray):
        """Read a crop framed like the middle view (e.g. a restored stack) by cutting
        the configured padding views back out of it and fusing them."""
        if img is None or img.size == 0:
            return None
        mfx, mfy = self.views[min(1, len(self.views) - 1)]
        H, W = img.shape[:2]
        bw, bh = W / (1 + 2 * mfx), H / (1 + 2 * mfy)
        x0, y0 = mfx * bw, mfy * bh
        crops = []
        for fx, fy in self.views:
            x1, y1 = max(0, int(round(x0 - fx * bw))), max(0, int(round(y0 - fy * bh)))
            x2, y2 = min(W, int(round(x0 + bw + fx * bw))), min(H, int(round(y0 + bh + fy * bh)))
            if x2 - x1 > 4 and y2 - y1 > 4:
                crops.append(img[y1:y2, x1:x2])
        if not crops:
            return None
        probs, _ = self.ocr(crops)
        obs = []
        for k, pr in enumerate(probs):
            lp = np.log(pr + 1e-9)
            dec = self._decode(lp)
            obs.append(Observation(lp, 0.1 + dec.prob, dec, f"restored{k}", 1.0))
        rc = self.cfg.recognition
        return fuse(obs, self.grammar, self.mode, rc.auto_fallback_nats, rc.top_k_alternatives)

    # -------------------------------------------------------------- vehicles
    @staticmethod
    def _associate(plates: list[BBox], vehicles: list[Detection]) -> list[int | None]:
        out = []
        for pb in plates:
            best, best_area = None, 1e18
            for j, v in enumerate(vehicles):
                if pb.containment(v.bbox) >= 0.7 and v.bbox.area < best_area:
                    best, best_area = j, v.bbox.area
            out.append(best)
        return out

    def _vehicle_info(self, frame: np.ndarray, v: Detection) -> VehicleInfo:
        crop = crop_box(frame, v.bbox)
        return VehicleInfo(v.bbox, v.label, v.score, dominant_color(crop))

    # ----------------------------------------------------------- image API
    def recognize(self, image: np.ndarray | str) -> list[PlateResult]:
        frame = cv2.imread(str(image)) if isinstance(image, (str, Path)) else image
        if frame is None:
            raise ValueError(f"could not read image {image}")
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if self.bpc is not None:
            frame = self.bpc(frame)
        dets = self.plate_det(frame)
        reads = self._read_many(frame, dets)
        vehicles = self.vehicle_det(frame) if (self.vehicle_det is not None and dets) else []
        assoc = self._associate([d.bbox for d in dets], vehicles)
        results = []
        for det, r, vi in zip(dets, reads, assoc):
            if r is None:
                continue
            read = self._maybe_verify(r.read, r.crop)
            results.append(PlateResult(det.bbox, det.score, read, r.quality, r.quad, r.two_line,
                                       self._vehicle_info(frame, vehicles[vi]) if vi is not None else None, r.crop))
        results.sort(key=lambda p: p.bbox.x1)
        return results

    def toll(self, item) -> dict:
        """Transaction decision for a PlateEvent / PlateResult, or the main plate of an image
        (see drishti.tolling)."""
        if isinstance(item, (str, Path, np.ndarray)):
            found = self.recognize(item)
            results = [r for r in found if r.read.text] or found
            if not results:
                return toll_decision(self, PlateEvent("none", -1, "image", None, 0.0, "", 0, 0, 0, 0, 0, 0,
                                                      BBox(0, 0, 0, 0)))
            item = max(results, key=lambda r: (r.read.confidence, r.bbox.w))
        return toll_decision(self, item)

    def annotate(self, frame: np.ndarray, results: list[PlateResult]) -> np.ndarray:
        out = frame.copy()
        for p in results:
            draw_plate(out, p.bbox, p.read.text, p.read.confidence, p.corners, low=p.read.low_confidence,
                       vehicle=p.vehicle.bbox if p.vehicle else None)
        return out

    # ----------------------------------------------------------- video API
    def session(self, camera: str = "cam0", fps: float = 25.0, **kw) -> "VideoSession":
        return VideoSession(self, camera=camera, fps=fps, **kw)

    def process_video(self, source: str | int, camera: str = "cam0", output: str | None = None,
                      on_event: Callable[[PlateEvent], None] | None = None, max_frames: int | None = None,
                      store: EventStore | None = None, watchlist: Watchlist | None = None, realtime_ts: bool = False,
                      on_health: Callable[[dict], None] | None = None) -> list[PlateEvent]:
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
        if not cap.isOpened():
            raise IOError(f"cannot open video source {source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer = None
        sess = self.session(camera, fps, on_event=on_event, store=store, watchlist=watchlist, on_health=on_health)
        every = max(1, int(self.cfg.video.every_n))
        idx = 0
        events: list[PlateEvent] = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if max_frames is not None and idx >= max_frames:
                    break
                ts = time.time() if realtime_ts else idx / fps
                if idx % every == 0:
                    events.extend(e for e in sess.step(frame, idx, ts) if not e.provisional)
                    if output is not None:
                        if writer is None:
                            h, w = frame.shape[:2]
                            writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps / every, (w, h))
                        writer.write(sess.draw(frame))
                idx += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        events.extend(sess.finish())
        return events


@dataclass
class _TrackState:
    observations: list[Observation] = field(default_factory=list)  # one fused observation per read frame
    crops: list[tuple[float, np.ndarray]] = field(default_factory=list)
    best_crop: np.ndarray | None = None
    best_quality: float = -1.0
    n_reads: int = 0
    vehicle_votes: Counter = field(default_factory=Counter)
    color_votes: Counter = field(default_factory=Counter)
    display: PlateRead | None = None
    provisional_sent: bool = False
    event_id: str = ""
    quads: list = field(default_factory=list)
    crossed: bool = False
    trigger_sent: bool = False
    trigger_text: str | None = None


class VideoSession:
    """Stateful per-camera processing. Feed frames with step(); call finish() at the end."""

    def __init__(self, engine: Drishti, camera: str = "cam0", fps: float = 25.0,
                 on_event: Callable[[PlateEvent], None] | None = None, store: EventStore | None = None,
                 watchlist: Watchlist | None = None, on_health: Callable[[dict], None] | None = None):
        self.e = engine
        self.cfg = engine.cfg
        v = self.cfg.video
        self.tracker = ByteTracker(v.track_high_thresh, v.track_low_thresh, v.new_track_thresh, v.match_thresh, v.max_age,
                                   float(v.get("track_buffer", 0.3)), float(v.get("track_buffer_lost", 0.5)))
        self.camera, self.fps = camera, fps
        self.on_event, self.store, self.watchlist = on_event, store, watchlist
        self.states: dict[int, _TrackState] = {}
        self.track_meta: dict[int, Track] = {}
        self.ts_of_frame: dict[int, float] = {}
        self._last_matches: list[tuple[Track, Detection]] = []
        self._vehicles: list[Detection] = []
        self.zones = Zones.from_config(self.cfg.get("zones"))
        self.bpc = engine.make_bpc(learn=True, camera=camera) if engine.bpc is not None else None
        hc = self.cfg.get("health") or {}
        self.health = CameraHealth.from_config(camera, hc) if hc.get("enabled", True) else None
        self.on_health = on_health
        self.health_alerts: list[dict] = []
        crops_dir = self.cfg.storage.crops_dir if (self.cfg.storage.save_crops and store is not None) else None
        self.crops_dir = expand_path(crops_dir) if crops_dir else None
        if self.crops_dir:
            self.crops_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ step
    def step(self, frame: np.ndarray, frame_idx: int, ts: float | None = None) -> list[PlateEvent]:
        ts = frame_idx / self.fps if ts is None else ts
        self.ts_of_frame[frame_idx] = ts
        H, W = frame.shape[:2]
        if self.health is not None:
            for a in self.health.update(frame, ts, self.bpc.n_static if self.bpc is not None else 0):
                log.warning("[%s] camera health %s: %s", self.camera, a["state"], a["message"])
                self.health_alerts.append(a)
                if self.on_health is not None:
                    self.on_health(a)
        if self.bpc is not None:
            frame = self.bpc(frame)
        dets = self.e.plate_det(frame)
        if self.zones.active:
            dets = [d for d in dets if self.zones.inside(d.bbox.cx, d.bbox.cy, W, H)]
        matches = self.tracker.update(dets, frame_idx)
        self._last_matches = [(t, dets[i]) for t, i in matches]
        if self.zones.has_line:
            for t, i in matches:
                st = self.states.setdefault(t.track_id, _TrackState(event_id=self._event_id(t)))
                if not st.crossed and len(t.history) >= 2:
                    (_, a), (_, b) = t.history[-2], t.history[-1]
                    if self.zones.crossed((a.cx, a.cy), (b.cx, b.cy), W, H):
                        st.crossed = True

        if self.e.vehicle_det is not None and frame_idx % max(1, int(self.cfg.video.vehicle_every_n)) == 0 and matches:
            self._vehicles = self.e.vehicle_det(frame)

        to_read: list[tuple[Track, Detection]] = []
        for t, i in matches:
            st = self.states.setdefault(t.track_id, _TrackState(event_id=self._event_id(t)))
            self.track_meta[t.track_id] = t
            confident = st.display is not None and st.display.confidence >= 0.97 and st.n_reads >= 5
            if confident and dets[i].bbox.h <= self._recent_height(t):
                continue  # already sure and the plate is not getting larger: track only, skip OCR
            to_read.append((t, dets[i]))

        reads = self.e._read_many(frame, [d for _, d in to_read], frame_weight=True)
        assoc = self.e._associate([d.bbox for _, d in to_read], self._vehicles)
        events: list[PlateEvent] = []
        for (t, d), r, vi in zip(to_read, reads, assoc):
            if r is None:
                continue
            st = self.states[t.track_id]
            if r.frame_obs is not None:
                st.observations.append(r.frame_obs)
            st.n_reads += 1
            if r.quad is not None:
                st.quads.append(r.quad)
            topk = int(self.cfg.video.top_k_frames)
            st.crops.append((r.quality, r.crop))
            st.crops.sort(key=lambda x: -x[0])
            del st.crops[topk:]
            if r.quality > st.best_quality:
                st.best_quality, st.best_crop = r.quality, r.crop
            if len(st.observations) > 64:  # keep the sharpest frames, not the most self-confident ones
                st.observations.sort(key=lambda o: -o.quality)
                del st.observations[48:]
            if vi is not None:
                v = self._vehicles[vi]
                st.vehicle_votes[v.label] += 1
                if st.n_reads % 3 == 1:
                    col = dominant_color(crop_box(frame, v.bbox))
                    if col:
                        st.color_votes[col] += 1
            st.display = self._fused_read(st, use_restoration=False)
            ev = None if self.zones.has_line else self._maybe_early(t, st)
            if ev is not None:
                events.append(ev)

        if self.zones.has_line:  # vehicles that crossed the line get their decision now
            for t, _ in matches:
                st = self.states.get(t.track_id)
                if st is not None and st.crossed and not st.trigger_sent and st.n_reads > 0:
                    ev = self._trigger(t, st)
                    if ev is not None:
                        events.append(ev)

        for t in self.tracker.pop_removed():
            ev = self._finalize(t)
            if ev is not None:
                events.append(ev)
        return events

    def _recent_height(self, t: Track) -> float:
        hs = [b.h for _, b in t.history[-6:-1]]
        return max(hs) if hs else 0.0

    def finish(self) -> list[PlateEvent]:
        out = []
        for t in self.tracker.flush():
            ev = self._finalize(t)
            if ev is not None:
                out.append(ev)
        return out

    # ---------------------------------------------------------------- fusion
    def _fused_read(self, st: _TrackState, use_restoration: bool) -> PlateRead:
        """Track-level decision.

        Frames are selected by image quality (not OCR confidence: degraded frames
        of the same plate tend to agree on the same wrong string). The best crops
        are also registered and averaged; a restored image integrates the pixel
        evidence of all its frames, so it carries their combined weight."""
        e = self.e
        rc = self.cfg.recognition
        k = int(self.cfg.video.top_k_frames)
        obs = sorted(st.observations, key=lambda o: -o.quality)[: max(k, 1)]
        frame_w = sum(o.weight for o in obs) or 1.0
        if use_restoration and self.cfg.video.mf_restore and len(st.crops) >= 3:
            stacks = []
            for kk in sorted({min(5, len(st.crops)), len(st.crops)}):
                sub = st.crops[:kk]
                img = align_and_stack([c for _, c in sub], [q + 1e-3 for q, _ in sub])
                if img is not None:
                    stacks.append(img)
            for img in stacks:
                fr_r = e.read_padded_crop(img)
                if fr_r is not None:
                    obs.append(Observation(fr_r.logp, frame_w * (0.1 + fr_r.decode.prob), fr_r.decode, "mf-restore", st.best_quality))
        fr = fuse(obs, e.grammar, e.mode, rc.auto_fallback_nats, rc.top_k_alternatives)
        return e._to_read(fr, n_obs=st.n_reads, quality=max(st.best_quality, 0.0), n_views=len(obs))

    def _event_id(self, t: Track) -> str:
        raw = f"{self.camera}|{t.track_id}|{t.start_frame}|{id(self)}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _direction(t: Track) -> str | None:
        if len(t.history) < 3:
            return None
        (_, a), (_, b) = t.history[0], t.history[-1]
        ratio = b.h / max(a.h, 1e-6)
        if ratio > 1.15:
            return "approaching"
        if ratio < 0.87:
            return "receding"
        dx = b.cx - a.cx
        if abs(dx) > 0.5 * a.w:
            return "left_to_right" if dx > 0 else "right_to_left"
        return "stationary"

    def _build_event(self, t: Track, st: _TrackState, read: PlateRead, provisional: bool) -> PlateEvent:
        first_ts = self.ts_of_frame.get(t.start_frame, t.start_frame / self.fps)
        last_ts = self.ts_of_frame.get(t.last_frame, t.last_frame / self.fps)
        return PlateEvent(
            event_id=st.event_id, track_id=t.track_id, camera=self.camera, text=read.text, confidence=read.confidence,
            fmt=read.fmt, first_ts=first_ts, last_ts=last_ts, first_frame=t.start_frame, last_frame=t.last_frame,
            n_frames=len(t.history), n_reads=st.n_reads, bbox=t.bbox, alternatives=read.alternatives,
            agreement=read.agreement, vehicle_type=st.vehicle_votes.most_common(1)[0][0] if st.vehicle_votes else None,
            vehicle_color=st.color_votes.most_common(1)[0][0] if st.color_votes else None, direction=self._direction(t),
            provisional=provisional, low_confidence=read.low_confidence, crop=st.best_crop, verified_by=read.verified_by,
            level=read.level, plate_width_px=int(round(max((b.w for _, b in t.history), default=0))),
        )

    def _maybe_early(self, t: Track, st: _TrackState) -> PlateEvent | None:
        thr = float(self.cfg.video.early_emit_confidence)
        if st.provisional_sent or st.display is None or st.display.text is None:
            return None
        if st.n_reads < int(self.cfg.video.min_hits) or st.display.confidence < thr:
            return None
        st.provisional_sent = True
        ev = self._build_event(t, st, st.display, provisional=True)
        ev.trigger = "early"
        self._publish(ev)
        return ev

    def _trigger(self, t: Track, st: _TrackState) -> PlateEvent | None:
        """The plate crossed the trigger line: decide now with everything seen so far."""
        read = self._fused_read(st, use_restoration=True)
        if st.best_crop is not None:
            read = self.e._maybe_verify(read, st.best_crop)
        st.trigger_sent, st.trigger_text = True, read.text
        ev = self._build_event(t, st, read, provisional=False)
        ev.trigger = "line"
        self._publish(ev)
        return ev

    def _finalize(self, t: Track) -> PlateEvent | None:
        st = self.states.pop(t.track_id, None)
        self.track_meta.pop(t.track_id, None)
        if st is None or st.n_reads == 0:
            return None
        if st.n_reads < int(self.cfg.video.min_hits) and (st.display is None or st.display.confidence < 0.9):
            return None
        zc = self.cfg.get("zones") or {}
        if self.zones.has_line and not st.trigger_sent and not zc.get("emit_uncrossed", True):
            return None
        read = self._fused_read(st, use_restoration=True)
        if st.best_crop is not None:
            read = self.e._maybe_verify(read, st.best_crop)
        if st.trigger_sent:
            if read.text == st.trigger_text:
                return None  # the decision at the line stands
            ev = self._build_event(t, st, read, provisional=False)
            ev.corrects, ev.event_id = st.event_id, st.event_id + "-c"
        else:
            ev = self._build_event(t, st, read, provisional=False)
        ev.trigger = "track_end"
        self._publish(ev)
        return ev

    # -------------------------------------------------------------- outputs
    def _publish(self, ev: PlateEvent) -> None:
        if self.watchlist is not None and ev.text:
            hit = self.watchlist.match(ev.text, ev.alternatives)
            if hit is not None:
                ev.watchlist_hit = hit.to_dict()
                self._webhook(ev)
        if self.store is not None and not ev.provisional:
            pv = self.cfg.privacy
            plate_value = plate_digest(ev.text, pv.hmac_key_env) if (pv.hash_plates and ev.text) else ev.text
            dd = self.cfg.get("dedupe") or {}
            if dd.get("enabled") and plate_value and not ev.corrects:
                ev.duplicate_of = self.store.recent_similar(
                    plate_value, float(dd.get("window_s", 10)), exclude_camera=self.camera,
                    cameras=dd.get("cameras") or None,
                    max_distance=0.0 if pv.hash_plates else float(dd.get("max_distance", 1.0)))
            if self.crops_dir is not None and ev.crop is not None:
                path = self.crops_dir / f"{ev.event_id}.jpg"
                crop = ev.crop.copy()
                if pv.blur_non_matches and not ev.watchlist_hit:
                    blur_region(crop, BBox(0, 0, crop.shape[1], crop.shape[0]), pad=0.0)
                cv2.imwrite(str(path), crop)
                ev.crop_path = str(path)
            self.store.add_event(ev, plate_value)
        if self.on_event is not None:
            self.on_event(ev)

    def _webhook(self, ev: PlateEvent) -> None:
        url = self.cfg.alerts.get("webhook_url")
        if not url:
            return
        body = json.dumps(ev.to_dict()).encode()

        def _send():
            try:
                req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
                urllib.request.urlopen(req, timeout=5).close()
            except Exception as exc:  # alerts must not stop recognition
                log.warning("webhook failed: %s", exc)

        threading.Thread(target=_send, daemon=True).start()

    # ------------------------------------------------------------- drawing
    def draw(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        if self.zones.active:
            self.zones.draw(out)
        blur = self.cfg.privacy.blur_non_matches
        for t, d in self._last_matches:
            st = self.states.get(t.track_id)
            read = st.display if st else None
            text = read.text if read else None
            alert = False
            if self.watchlist is not None and text:
                alert = self.watchlist.match(text, read.alternatives) is not None
            if blur and not alert:
                blur_region(out, d.bbox)
            draw_plate(out, d.bbox, text, read.confidence if read else 0.0, track_id=t.track_id,
                       low=bool(read and read.low_confidence), alert=alert)
        return out


def iter_images(paths: Iterable[str]) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            # skip Drishti's own annotated copies (--save-vis writes <name>_drishti.jpg)
            yield from sorted(x for x in pp.rglob("*") if x.suffix.lower() in exts and not x.stem.endswith("_drishti"))
        elif pp.exists():
            yield pp
