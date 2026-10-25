"""REST API (FastAPI).

    POST /v1/recognize            multipart image -> plates (optionally annotated JPEG, base64)
    POST /v1/recognize/video      multipart video -> fused plate events
    GET  /v1/events               query stored events (?plate=KA05*&camera=&limit=)
    GET  /v1/watchlist            list
    POST /v1/watchlist            {"plate": "...", "reason": "...", "level": "alert"}
    DELETE /v1/watchlist/{plate}
    GET  /healthz                 models and execution providers
    GET  /metrics                 Prometheus text format

Set DRISHTI_API_KEY to require `X-API-Key: <key>` (or `Authorization: Bearer <key>`)
on every /v1 endpoint. Always set it when the server is reachable from a network.
"""

from __future__ import annotations

import base64
import hmac
import os
from contextlib import asynccontextmanager
import tempfile
import threading
import time

import cv2
import numpy as np

try:
    from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
    from fastapi.responses import PlainTextResponse
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install 'drishti-anpr[server]'") from exc

from ..config import load_config
from ..pipeline import Drishti
from ..store import EventStore
from ..watchlist import WatchEntry, Watchlist

MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_VIDEO_BYTES = 512 * 1024 * 1024
API_KEY_ENV = "DRISHTI_API_KEY"


class WatchIn(BaseModel):
    plate: str
    reason: str = ""
    level: str = "alert"


class _Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests = 0
        self.plates = 0
        self.errors = 0
        self.latency_sum = 0.0
        self.buckets = {0.05: 0, 0.1: 0, 0.25: 0, 0.5: 0, 1.0: 0, 2.5: 0, float("inf"): 0}

    def observe(self, seconds: float, plates: int) -> None:
        with self.lock:
            self.requests += 1
            self.plates += plates
            self.latency_sum += seconds
            for b in self.buckets:
                if seconds <= b:
                    self.buckets[b] += 1

    def render(self) -> str:
        with self.lock:
            lines = [
                "# TYPE drishti_requests_total counter", f"drishti_requests_total {self.requests}",
                "# TYPE drishti_plates_total counter", f"drishti_plates_total {self.plates}",
                "# TYPE drishti_errors_total counter", f"drishti_errors_total {self.errors}",
                "# TYPE drishti_recognize_seconds histogram",
            ]
            for b, c in self.buckets.items():
                le = "+Inf" if b == float("inf") else str(b)
                lines.append(f'drishti_recognize_seconds_bucket{{le="{le}"}} {c}')
            lines += [f"drishti_recognize_seconds_sum {self.latency_sum:.6f}", f"drishti_recognize_seconds_count {self.requests}"]
        return "\n".join(lines) + "\n"


def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="not a decodable image")
    return img


def _require_key(api_key: str | None):
    def check(x_api_key: str | None = Header(None), authorization: str | None = Header(None)) -> None:
        if not api_key:
            return
        given = x_api_key or (authorization[7:] if authorization and authorization.lower().startswith("bearer ") else "")
        if not hmac.compare_digest(given.encode(), api_key.encode()):
            raise HTTPException(status_code=401, detail="missing or wrong API key")

    return check


def create_app(config_path: str | None = None, engine: Drishti | None = None, api_key: str | None = None) -> FastAPI:
    cfg = engine.cfg if engine is not None else load_config(config_path)
    auth = [Depends(_require_key(api_key if api_key is not None else os.environ.get(API_KEY_ENV)))]
    state: dict = {"engine": engine}
    metrics = _Metrics()
    store = EventStore(cfg.storage.db_path)
    lock = threading.Lock()

    def get_engine() -> Drishti:
        with lock:
            if state["engine"] is None:
                state["engine"] = Drishti(cfg)
        return state["engine"]

    @asynccontextmanager
    async def lifespan(_app):
        get_engine()  # load models before the first request
        yield
        store.close()

    app = FastAPI(title="Drishti ANPR", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    def healthz():
        e = get_engine()
        return {
            "status": "ok",
            "plate_detector": str(cfg.models.plate_detector),
            "vehicle_detector": cfg.models.vehicle_detector,
            "ocr": str(cfg.models.ocr),
            "region": str(cfg.recognition.region),
            "providers": e.plate_det.session.get_providers(),
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def prom():
        return metrics.render()

    @app.post("/v1/recognize", dependencies=auth)
    async def recognize(file: UploadFile = File(...), annotate: bool = Query(False)):
        data = await file.read()
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="image too large")
        img = _decode_image(data)
        t = time.perf_counter()
        try:
            res = get_engine().recognize(img)
        except Exception as exc:
            metrics.errors += 1
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        dt = time.perf_counter() - t
        metrics.observe(dt, len(res))
        wl = Watchlist(store.watch_list(), cfg.alerts.watchlist_max_distance)
        plates = []
        for r in res:
            d = r.to_dict()
            hit = wl.match(r.read.text, r.read.alternatives)
            d["watchlist_hit"] = hit.to_dict() if hit else None
            plates.append(d)
        out = {"plates": plates, "latency_ms": round(dt * 1000, 1), "image_size": list(img.shape[:2])}
        if annotate:
            ok, buf = cv2.imencode(".jpg", get_engine().annotate(img, res), [cv2.IMWRITE_JPEG_QUALITY, 85])
            out["annotated_jpeg_base64"] = base64.b64encode(buf.tobytes()).decode() if ok else None
        return out

    @app.post("/v1/toll", dependencies=auth)
    async def toll(file: UploadFile = File(...)):
        """Transaction decision (auto above the confidence threshold, else manual_review) for an image."""
        data = await file.read()
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="image too large")
        img = _decode_image(data)
        try:
            return get_engine().toll(img)
        except Exception as exc:
            metrics.errors += 1
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/v1/recognize/video", dependencies=auth)
    async def recognize_video(file: UploadFile = File(...), camera: str = Query("upload"), persist: bool = Query(False)):
        suffix = os.path.splitext(file.filename or "clip.mp4")[1] or ".mp4"
        fd, path = tempfile.mkstemp(suffix=suffix)
        size = 0
        try:
            with os.fdopen(fd, "wb") as fh:
                while chunk := await file.read(1 << 20):
                    size += len(chunk)
                    if size > MAX_VIDEO_BYTES:
                        raise HTTPException(status_code=413, detail="video too large")
                    fh.write(chunk)
            wl = Watchlist(store.watch_list(), cfg.alerts.watchlist_max_distance)
            t = time.perf_counter()
            events = get_engine().process_video(path, camera=camera, store=store if persist else None, watchlist=wl)
            dt = time.perf_counter() - t
            metrics.observe(dt, len(events))
            return {"events": [e.to_dict() for e in events], "seconds": round(dt, 2)}
        finally:
            if os.path.exists(path):
                os.remove(path)

    @app.get("/v1/events", dependencies=auth)
    def events(plate: str | None = None, camera: str | None = None, since: float | None = None, limit: int = 100):
        return {"events": store.query(plate=plate, camera=camera, since=since, limit=min(limit, 1000))}

    @app.get("/v1/watchlist", dependencies=auth)
    def watch_list():
        return {"entries": [{"plate": e.plate, "reason": e.reason, "level": e.level} for e in store.watch_list()]}

    @app.post("/v1/watchlist", dependencies=auth)
    def watch_add(item: WatchIn):
        if item.level not in ("alert", "info", "allow"):
            raise HTTPException(status_code=422, detail="level must be alert | info | allow")
        store.watch_add(WatchEntry(item.plate, item.reason, item.level))
        return {"ok": True}

    @app.delete("/v1/watchlist/{plate}", dependencies=auth)
    def watch_remove(plate: str):
        if not store.watch_remove(plate):
            raise HTTPException(status_code=404, detail="not on watchlist")
        return {"ok": True}

    return app
