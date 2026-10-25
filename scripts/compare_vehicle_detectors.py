"""Compare vehicle detectors side by side: every photo annotated by each model, with each vehicle's
type and score, the plates read, and the detector's time per photo.

    python scripts/compare_vehicle_detectors.py "photos/" --out vehicles
    python scripts/compare_vehicle_detectors.py photos/ --models rf-detr-nano-384-coco ~/Downloads/rf-detr-large-704-coco.onnx
    open vehicles/index.html
"""

from __future__ import annotations

import argparse
import html
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drishti.config import load_config  # noqa: E402
from drishti.detect.vehicle import VehicleDetector  # noqa: E402
from drishti.pipeline import Drishti, iter_images  # noqa: E402

_COLORS = {"car": (60, 180, 75), "motorcycle": (255, 140, 0), "bus": (200, 60, 200), "truck": (0, 120, 255),
           "bicycle": (230, 200, 0)}

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vehicle detectors</title>
<style>
:root{{--bg:#f6f5f2;--panel:#fff;--ink:#1d1c1a;--muted:#6b6862;--line:#e3e0da}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--panel:#1f1f1d;--ink:#ecebe7;--muted:#a19e97;--line:#34332f}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,sans-serif}}
main{{max-width:1300px;margin:0 auto;padding:16px}}
h1{{font-size:20px}} h2{{font-size:16px;margin:0 0 10px}}
table{{border-collapse:collapse;margin:8px 0 18px}} td,th{{border:1px solid var(--line);padding:4px 10px;text-align:left}}
.photo{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:18px}}
.row{{display:grid;grid-template-columns:repeat({n},1fr);gap:10px}}
@media (max-width:800px){{.row{{grid-template-columns:1fr}}}}
figure{{margin:0}} figcaption{{color:var(--muted);font-size:13px;margin:0 0 4px}}
img{{width:100%;max-height:520px;object-fit:contain;background:#000;border-radius:8px;display:block}}
.meta{{color:var(--muted);font-size:13px;margin-top:4px}}
</style></head><body><main>
<h1>Vehicle detectors side by side</h1>
{summary}
{body}
</main></body></html>
"""


def draw(img: np.ndarray, vehicles, plates) -> np.ndarray:
    out = img.copy()
    t = max(2, round(out.shape[1] / 600))
    for v in vehicles:
        b = v.bbox
        c = _COLORS.get(v.label, (200, 200, 200))
        cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), c, t)
        label = f"{v.label} {v.score:.2f}"
        fs = 0.4 * t
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, t)
        y = max(int(b.y1), th + 4)
        cv2.rectangle(out, (int(b.x1), y - th - 4), (int(b.x1) + tw + 4, y), c, -1)
        cv2.putText(out, label, (int(b.x1) + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), t, cv2.LINE_AA)
    for p in plates:
        b = p.bbox
        cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 0, 255), t)
        if p.read.text:
            cv2.putText(out, p.read.text, (int(b.x1), int(b.y2) + 12 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t,
                        (0, 0, 255), t, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="images or folders")
    ap.add_argument("--models", nargs="+", default=["rf-detr-nano-384-coco", "rf-detr-large-704-coco"],
                    help="hub names or .onnx paths of vehicle detectors")
    ap.add_argument("--config", action="append", default=None)
    ap.add_argument("--providers", default="cpu")
    ap.add_argument("--conf", type=float, default=0.4, help="vehicle score threshold")
    ap.add_argument("--out", default="vehicles")
    a = ap.parse_args()

    rt = {"providers": a.providers}
    plates_eng = Drishti(load_config(a.config, {"runtime": rt, "models": {"vehicle_detector": None}}))
    dets = [VehicleDetector(str(Path(m).expanduser()) if Path(m).expanduser().exists() else m, rt, conf=a.conf)
            for m in a.models]
    names = [Path(m).name.removesuffix(".onnx") for m in a.models]
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)

    times: list[list[float]] = [[] for _ in dets]
    counts = [0 for _ in dets]
    blocks = []
    for n_img, path in enumerate(iter_images(a.inputs), 1):
        img = cv2.imread(str(path))
        if img is None:
            continue
        plates = plates_eng.recognize(img)
        figs = []
        for k, (det, name) in enumerate(zip(dets, names)):
            det(img) if n_img == 1 else None  # first call warms the session up; not timed
            t = time.perf_counter()
            vehicles = det(img)
            times[k].append(time.perf_counter() - t)
            counts[k] += len(vehicles)
            fn = f"{n_img:03d}_{k}.jpg"
            cv2.imwrite(str(out / "img" / fn), draw(img, vehicles, plates))
            kinds = ", ".join(f"{v.label} {v.score:.2f}" for v in sorted(vehicles, key=lambda v: -v.score)) or "none"
            figs.append(f'<figure><figcaption>{html.escape(name)} · {len(vehicles)} vehicles · '
                        f'{1000 * times[k][-1]:.0f} ms</figcaption><img src="img/{fn}" alt="{html.escape(name)}">'
                        f'<div class="meta">{html.escape(kinds)}</div></figure>')
        reads = ", ".join(p.read.text or "unreadable" for p in plates) or "no plate"
        blocks.append(f'<section class="photo"><h2>{html.escape(path.name)}</h2><div class="row">{"".join(figs)}</div>'
                      f'<div class="meta">plates: {html.escape(reads)}</div></section>')
        print(f"{path.name}: plates {reads}", flush=True)

    rows = "".join(f"<tr><td>{html.escape(nm)}</td><td>{c}</td><td>{1000 * np.median(ts):.0f} ms</td></tr>"
                   for nm, c, ts in zip(names, counts, times) if ts)
    summary = (f"<table><tr><th>model</th><th>vehicles found</th><th>median time per photo</th></tr>{rows}</table>"
               if rows else "<p>no images found</p>")
    (out / "index.html").write_text(PAGE.format(n=len(dets), summary=summary, body="\n".join(blocks)), encoding="utf-8")
    for nm, c, ts in zip(names, counts, times):
        if ts:
            print(f"{nm:<28} vehicles {c:>4}   median {1000 * np.median(ts):.0f} ms per photo")
    print(f"open {out / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
