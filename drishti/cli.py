"""Command line interface.

    drishti image photo.jpg [more.jpg | folder/] [--save-vis out/] [--json]
    drishti video clip.mp4|rtsp://...|0 [--out events.jsonl] [--save-vis annotated.mp4]
                 [--roi "x,y x,y ..."] [--line "x1,y1 x2,y2"] [--direction down]
    drishti zones clip.mp4 --frame 100 --out zones.jpg
    drishti serve [--host 0.0.0.0 --port 8000]
    drishti eval labels.csv [--report report.json]
    drishti calibrate eval_report.json --out calibration.json
    drishti bench [--image photo.jpg] [--runs 20]
    drishti models pull | list
    drishti watch add KA05MX3021 --reason stolen | list | remove KA05MX3021
    drishti events [--plate KA05*] [--limit 20]
    drishti vlm-check [--vlm-model qwen3-vl:2b] [--image crop.jpg]
    drishti label images/ --out data/ocr_real      (OCR + local VLM auto-labelling)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path


def _engine(args):
    from .config import load_config
    from .pipeline import Drishti

    overrides: dict = {}
    if getattr(args, "region", None):
        overrides.setdefault("recognition", {})["region"] = args.region
    if getattr(args, "no_vehicles", False):
        overrides.setdefault("models", {})["vehicle_detector"] = None
    if getattr(args, "providers", None):
        overrides.setdefault("runtime", {})["providers"] = args.providers
    if getattr(args, "vlm", False):
        overrides.setdefault("vlm", {})["enabled"] = True
    for flag, key in (("roi", "roi"), ("line", "trigger_line"), ("direction", "trigger_direction")):
        if getattr(args, flag, None):
            overrides.setdefault("zones", {})[key] = getattr(args, flag)
    for flag, key in (("vlm_model", "model"), ("vlm_endpoint", "endpoint"), ("vlm_backend", "backend")):
        if getattr(args, flag, None):
            overrides.setdefault("vlm", {})[key] = getattr(args, flag)
    cfg = load_config(args.config, overrides or None)
    return Drishti(cfg)


def _debug_plates(eng, img, stem: str, out: Path) -> None:
    """Every crop the OCR reads for each detected plate, saved as an image, with what it read."""
    import cv2
    import numpy as np

    from .imaging import RESCUE_VIEWS

    out.mkdir(parents=True, exist_ok=True)
    for k, det in enumerate(sorted(eng.plate_det(img), key=lambda d: d.bbox.x1), 1):  # same order as the reads
        b = det.bbox
        print(f"  plate {k}: box {b.w:.0f}x{b.h:.0f} px (aspect {b.w / max(b.h, 1e-6):.1f}:1) at "
              f"({b.x1:.0f},{b.y1:.0f}), detector score {det.score:.2f}")
        views, _, _ = eng._views(img, det)
        if not views:
            continue
        crops = [(name, v) for name, v, _ in views]
        mid = views[min(1, len(views) - 1)][1]
        crops += [(f"retry-{name}", fn(mid)) for name, fn in RESCUE_VIEWS.items()]
        probs, _ = eng.ocr([c for _, c in crops])
        for (name, crop), pr in zip(crops, probs):
            dec = eng._decode(np.log(pr + 1e-9))
            cv2.imwrite(str(out / f"{stem}_plate{k}_{name}.png"), crop)
            print(f"     {name:<22} {crop.shape[1]:>4}x{crop.shape[0]:<4} px  OCR '{dec.raw_text}'  "
                  f"as plate '{dec.text}' ({dec.prob:.2f})")
    print(f"  crops saved in {out}/")


def cmd_image(args) -> int:
    import cv2

    from .pipeline import iter_images

    eng = _engine(args)
    out_dir = Path(args.save_vis) if args.save_vis else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    results_all = []
    for path in iter_images(args.inputs):
        img = cv2.imread(str(path))
        if img is None:
            print(f"skip unreadable {path}", file=sys.stderr)
            continue
        t = time.perf_counter()
        res = eng.recognize(img)
        ms = (time.perf_counter() - t) * 1000
        rec = {"image": str(path), "latency_ms": round(ms, 1), "plates": [r.to_dict() for r in res]}
        results_all.append(rec)
        if not args.json:
            print(f"{path.name}: {len(res) or 'no'} plate{'s' if len(res) != 1 else ''} found  [{ms:.0f} ms]")
            min_w = (eng.cfg.get("tolling") or {}).get("min_plate_width_px") or 100
            for k, r in enumerate(res, 1):
                w = int(round(r.bbox.w))
                if r.read.text:
                    note = "  -> needs review" if (r.read.low_confidence or r.read.level == "Poor") else ""
                    if r.read.rescued:
                        note += "  (read after zoom / contrast / brightness retry)"
                    if r.read.verified_by == "vlm-only":
                        note += "  (read by the vision model alone; the OCR could not read it)"
                    print(f"  {k}. {r.read.text:<12} confidence {r.read.confidence:.2f} ({r.read.level}){note}")
                else:
                    why = f"too small to read, {w} px wide" if w < min_w else "plate found but not readable"
                    guesses = list(dict.fromkeys([g for g in [r.read.raw_text] + [t for t, _ in r.read.alternatives] if g]))
                    guess = f"  best guess: {', '.join(guesses[:3])} (very unsure, check by eye)" if guesses else ""
                    print(f"  {k}. {'unreadable':<12} ({why}){guess}")
        if out_dir:
            cv2.imwrite(str(out_dir / f"{path.stem}_drishti.jpg"), eng.annotate(img, res))
        if args.debug and not args.json:
            _debug_plates(eng, img, path.stem, Path(args.debug))
    if args.json:
        print(json.dumps(results_all if len(results_all) != 1 else results_all[0], indent=2))
    return 0


def cmd_toll(args) -> int:
    eng = _engine(args)
    res = eng.toll(args.image)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        conf = f"confidence {res['confidence']:.2f}, {res['level']}" if res["plate"] else "no readable plate"
        print(f"Camera read : {res['plate'] or 'nothing readable'}  ({conf})")
        if res["action"] == "auto":
            print(f"Decision    : AUTO - charge {res['vrn']}")
        else:
            print(f"Decision    : {res['action'].upper().replace('_', ' ')}")
        for r in res["reasons"]:
            print(f"  - {r}")
    return 0


def cmd_video(args) -> int:
    from .store import EventStore
    from .watchlist import Watchlist

    eng = _engine(args)
    store = EventStore(args.db or eng.cfg.storage.db_path) if (args.db or args.store) else None
    wl = Watchlist(store.watch_list() if store else [], eng.cfg.alerts.watchlist_max_distance)
    out = open(args.out, "a", encoding="utf-8") if args.out else None

    def on_event(ev):
        d = ev.to_dict()
        line = json.dumps(d)
        if out:
            out.write(line + "\n")
            out.flush()
        tag = "PROVISIONAL" if ev.provisional else "EVENT"
        hit = f"  WATCHLIST: {ev.watchlist_hit['reason'] or ev.watchlist_hit['plate']}" if ev.watchlist_hit else ""
        if ev.corrects:
            tag = "CORRECTION"
        elif ev.trigger == "line":
            tag = "LINE"
        if ev.duplicate_of:
            tag = "DUPLICATE"
        print(f"[{tag}] t={ev.last_ts:7.2f}s  {ev.text or '?':<12} conf={ev.confidence:.2f} {ev.level or ''}  "
              f"{ev.vehicle_type or '-'}/{ev.vehicle_color or '-'}  {ev.direction or ''}{hit}", flush=True)

    def on_health(a):
        print(f"[HEALTH {a['state'].upper()}] {a['camera']}: {a['message']}", flush=True)
        if out:
            out.write(json.dumps({"health": a}) + "\n")
            out.flush()

    src = args.source
    live = src.isdigit() or "://" in src
    t = time.perf_counter()
    events = eng.process_video(src, camera=args.camera, output=args.save_vis, on_event=on_event,
                               max_frames=args.max_frames, store=store, watchlist=wl, realtime_ts=live,
                               on_health=on_health)
    dt = time.perf_counter() - t
    print(f"{len(events)} plate events in {dt:.1f}s", file=sys.stderr)
    if out:
        out.close()
    return 0


def cmd_zones(args) -> int:
    import cv2

    from .config import load_config
    from .zones import Zones, grid_frame

    src = args.source
    frame = cv2.imread(src) if Path(src).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp") else None
    if frame is None:
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"cannot read a frame from {src}", file=sys.stderr)
            return 1
    zc = dict(load_config(args.config).zones)
    if args.roi:
        zc["roi"] = args.roi
    if args.line:
        zc["trigger_line"] = args.line
    cv2.imwrite(args.out, grid_frame(frame, Zones.from_config(zc)))
    print(f"wrote {args.out}: read coordinates off the 0-1 grid, then pass --roi / --line or set them in a config")
    return 0


def cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print("pip install 'drishti-anpr[server]' to use the API server", file=sys.stderr)
        return 1
    import os

    from .server.app import API_KEY_ENV, create_app

    if args.host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get(API_KEY_ENV):
        print(f"warning: serving on {args.host} without an API key; anyone on the network can read events "
              f"and edit the watchlist. Set {API_KEY_ENV}.", file=sys.stderr)
    uvicorn.run(create_app(args.config), host=args.host, port=args.port, log_level="info")
    return 0


def cmd_eval(args) -> int:
    from .evaluate import evaluate

    eng = _engine(args)
    report = evaluate(eng, args.labels, iou_free=True)
    s = report["summary"]
    print(json.dumps(s, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"full report -> {args.report}", file=sys.stderr)
    return 0


def cmd_calibrate(args) -> int:
    from .evaluate import fit_calibration

    report = json.loads(Path(args.report).read_text())
    calib = fit_calibration(report["items"])
    Path(args.out).write_text(json.dumps(calib, indent=2))
    print(json.dumps(calib, indent=2))
    print(f"set recognition.calibration: {args.out}", file=sys.stderr)
    return 0


def cmd_bench(args) -> int:
    import cv2
    import numpy as np

    eng = _engine(args)
    img = cv2.imread(args.image) if args.image else None
    if img is None:
        img = np.full((1080, 1920, 3), 90, np.uint8)
        cv2.rectangle(img, (900, 700), (1100, 750), (235, 235, 235), -1)
        cv2.putText(img, "KA05MX3021", (905, 742), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3)
    for _ in range(3):
        eng.recognize(img)
    det_t, full_t = [], []
    for _ in range(args.runs):
        t = time.perf_counter()
        eng.plate_det(img)
        det_t.append(time.perf_counter() - t)
        t = time.perf_counter()
        eng.recognize(img)
        full_t.append(time.perf_counter() - t)
    crop = img[700:750, 900:1100] if args.image is None else img[: img.shape[0] // 8, : img.shape[1] // 4]
    ocr_t = []
    for _ in range(args.runs):
        t = time.perf_counter()
        eng.ocr([crop])
        ocr_t.append(time.perf_counter() - t)
    med = lambda xs: 1000 * float(np.median(xs))  # noqa: E731
    print(json.dumps({
        "providers": eng.plate_det.session.get_providers(),
        "plate_detector_ms": round(med(det_t), 2),
        "ocr_per_crop_ms": round(med(ocr_t), 2),
        "recognize_image_ms": round(med(full_t), 2),
        "image_size": list(img.shape[:2]),
    }, indent=2))
    return 0


def cmd_models(args) -> int:
    from . import hub

    if args.action == "list":
        for name, spec in sorted(hub.REGISTRY.items()):
            print(f"{name:42s} {spec.kind:16s} {spec.license}")
        return 0
    from .config import load_config

    cfg = load_config(args.config)
    for name in [cfg.models.plate_detector, cfg.models.vehicle_detector, cfg.models.ocr, cfg.models.get("sr_model")]:
        if name and name in hub.REGISTRY:
            for role in hub.REGISTRY[name].files:
                if role == "keras" and not args.with_training:
                    continue
                print(hub.fetch(name, role, cfg.models.cache_dir))
    return 0


def _store(args):
    from .config import load_config
    from .store import EventStore

    cfg = load_config(args.config)
    return EventStore(args.db or cfg.storage.db_path)


def _vlm_from_args(args, region=None):
    from .config import load_config
    from .ocr.vlm import from_config

    cfg = load_config(args.config)
    v = dict(cfg.vlm)
    for flag, key in (("vlm_model", "model"), ("vlm_endpoint", "endpoint"), ("vlm_backend", "backend")):
        if getattr(args, flag, None):
            v[key] = getattr(args, flag)
    return from_config(v, region or cfg.recognition.region)


def cmd_vlm_check(args) -> int:
    vlm = _vlm_from_args(args)
    info = vlm.check()
    if info.get("reachable") and info.get("model_present") and args.image:
        import cv2

        crop = cv2.imread(args.image)
        t = time.perf_counter()
        ans = vlm.read(crop)
        info["test_read"] = {"text": ans.text, "readable": ans.readable, "seconds": round(time.perf_counter() - t, 2)}
    print(json.dumps(info, indent=2))
    return 0 if info.get("reachable") and info.get("model_present") else 1


def cmd_label(args) -> int:
    from .label import run_labelling

    args.no_vehicles = True
    eng = _engine(args)
    vlm = None
    if not args.no_vlm:
        vlm = _vlm_from_args(args, eng.cfg.recognition.region)
        info = vlm.check()
        if not (info.get("reachable") and info.get("model_present")):
            print(json.dumps(info, indent=2), file=sys.stderr)
            print("VLM not ready. Fix the above, or run with --no-vlm (OCR-only auto-accept).", file=sys.stderr)
            return 1
    summary = run_labelling(eng, args.inputs, args.out, vlm=vlm, labels_dir=args.labels_dir,
                            trust_ocr=args.trust_ocr, limit=args.limit, crops=args.crops)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_watch(args) -> int:
    from .watchlist import WatchEntry

    st = _store(args)
    if args.action == "add":
        st.watch_add(WatchEntry(args.plate, args.reason or "", args.level))
        print(f"added {args.plate}")
    elif args.action == "remove":
        print("removed" if st.watch_remove(args.plate) else "not found")
    else:
        for e in st.watch_list():
            print(f"{e.plate:12s} {e.level:6s} {e.reason}")
    return 0


def cmd_events(args) -> int:
    st = _store(args)
    for e in st.query(plate=args.plate, limit=args.limit, camera=args.camera):
        print(json.dumps(e))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="drishti", description="Automatic number plate recognition")
    ap.add_argument("--config", action="append", help="YAML config overriding the defaults; repeat to layer several (later ones win), e.g. --config models/drishti.yaml --config configs/toll-booth.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def vlm_opts(p):
        p.add_argument("--vlm-model", help="e.g. qwen3-vl:2b (Ollama) or a model id on an OpenAI-compatible server")
        p.add_argument("--vlm-endpoint", help="default http://localhost:11434 (Ollama)")
        p.add_argument("--vlm-backend", choices=["ollama", "openai"])

    def common(p):
        p.add_argument("--region", help="india | brazil | generic | path/to/pack.yaml")
        p.add_argument("--no-vehicles", action="store_true", help="skip vehicle detection (faster)")
        p.add_argument("--providers", help="auto | cpu | cuda | tensorrt | coreml | openvino")
        p.add_argument("--vlm", action="store_true", help="ask the local VLM (Ollama) about low-confidence reads")
        vlm_opts(p)

    p = sub.add_parser("image", help="recognise plates in images or folders")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--save-vis", help="folder for annotated images")
    p.add_argument("--debug", metavar="DIR", help="save every crop the OCR reads and print what it read from each")
    p.add_argument("--json", action="store_true")
    common(p)
    p.set_defaults(func=cmd_image)

    p = sub.add_parser("toll", help="toll transaction decision for an image (auto above 95%%, else manual review)")
    p.add_argument("image")
    p.add_argument("--json", action="store_true")
    common(p)
    p.set_defaults(func=cmd_toll)

    p = sub.add_parser("video", help="process a file, RTSP/HTTP stream or webcam index")
    p.add_argument("source")
    p.add_argument("--camera", default="cam0")
    p.add_argument("--out", help="append events as JSON lines")
    p.add_argument("--save-vis", help="annotated output video (.mp4)")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--store", action="store_true", help="persist events to the SQLite store")
    p.add_argument("--db", help="SQLite path (implies --store)")
    p.add_argument("--roi", help='lane polygon, fractions of the frame: "0.2,0.3 0.8,0.3 0.95,1 0.05,1"')
    p.add_argument("--line", help='trigger line: "0.1,0.75 0.9,0.75" (event fires when a plate crosses it)')
    p.add_argument("--direction", choices=["any", "down", "up", "left", "right"], help="only count crossings this way")
    common(p)
    p.set_defaults(func=cmd_video)

    p = sub.add_parser("zones", help="save a frame with a 0-1 grid to choose --roi / --line")
    p.add_argument("source", help="video file, stream URL, webcam index or image")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--out", default="zones.jpg")
    p.add_argument("--roi")
    p.add_argument("--line")
    p.set_defaults(func=cmd_zones)

    p = sub.add_parser("serve", help="run the REST API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("eval", help="evaluate on a labelled CSV (image_path,plate_text)")
    p.add_argument("labels")
    p.add_argument("--report")
    common(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("calibrate", help="fit the confidence model from an eval report")
    p.add_argument("report")
    p.add_argument("--out", default="calibration.json")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("bench", help="latency benchmark")
    p.add_argument("--image")
    p.add_argument("--runs", type=int, default=20)
    common(p)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("label", help="auto-label plate text with OCR + a local VLM; review the rest in a page")
    p.add_argument("inputs", nargs="+", help="image files or folders (YOLO labels next to images are used as boxes)")
    p.add_argument("--out", required=True, help="output folder: crops/, labels.jsonl, review.html")
    p.add_argument("--labels-dir", help="folder of YOLO .txt boxes, if not next to the images")
    p.add_argument("--crops", action="store_true", help="images are already cropped plates (whole image = plate)")
    p.add_argument("--no-vlm", action="store_true", help="OCR only: auto-accept only very confident valid reads")
    p.add_argument("--trust-ocr", type=float, default=0.985, help="(--no-vlm) confidence needed to auto-accept")
    p.add_argument("--limit", type=int, help="process at most N new images (resumable)")
    common(p)
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("vlm-check", help="check the local VLM (Ollama) is running and the model is installed")
    p.add_argument("--image", help="optional plate crop to test-read")
    vlm_opts(p)
    p.set_defaults(func=cmd_vlm_check)

    p = sub.add_parser("models", help="download or list models")
    p.add_argument("action", choices=["pull", "list"])
    p.add_argument("--with-training", action="store_true", help="also fetch Keras weights for OCR fine-tuning")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("watch", help="manage the watchlist")
    p.add_argument("action", choices=["add", "remove", "list"])
    p.add_argument("plate", nargs="?")
    p.add_argument("--reason")
    p.add_argument("--level", default="alert", choices=["alert", "info", "allow"])
    p.add_argument("--db")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("events", help="query stored events")
    p.add_argument("--plate", help="exact plate or wildcard, e.g. KA05*")
    p.add_argument("--camera")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--db")
    p.set_defaults(func=cmd_events)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if args.cmd == "watch" and args.action in ("add", "remove") and not args.plate:
        print("plate required", file=sys.stderr)
        return 2
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
