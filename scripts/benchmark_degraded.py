"""Robustness benchmark: labelled photos shrunk until the plate is small, then JPEG-compressed
and blurred the way messaging apps and cheap cameras degrade them.

Each photo is scaled so its labelled plate is --widths pixels wide (measured with the
detector on the original), saved at each --jpeg quality, with and without a light blur.
Reports exact-match rate and mean confidence per plate width, plus latency.

    python scripts/benchmark_degraded.py samples/labels.csv
    python scripts/benchmark_degraded.py samples/labels.csv --config my.yaml --widths 70 55 45
    python scripts/benchmark_degraded.py samples/labels.csv --vlm-model qwen3-vl:2b   # with the local VLM
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drishti.config import load_config  # noqa: E402
from drishti.pipeline import Drishti  # noqa: E402
from drishti.watchlist import normalize_plate  # noqa: E402


def degrade(img: np.ndarray, scale: float, quality: int, blur: float) -> np.ndarray:
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if blur > 0:
        small = cv2.GaussianBlur(small, (0, 0), blur)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", help="labels CSV (image_path,plate_text)")
    ap.add_argument("--config", action="append", default=None)
    ap.add_argument("--widths", type=int, nargs="+", default=[70, 55, 45], help="target plate widths in px")
    ap.add_argument("--jpeg", type=int, nargs="+", default=[60, 30], help="JPEG qualities")
    ap.add_argument("--blur", type=float, nargs="+", default=[0.0, 0.8], help="Gaussian blur sigmas")
    ap.add_argument("--providers", default="cpu")
    ap.add_argument("--vlm-model", help="also check / read plates with this Ollama model")
    ap.add_argument("--verbose", action="store_true", help="print every case")
    a = ap.parse_args()

    over = {"runtime": {"providers": a.providers}, "models": {"vehicle_detector": None}}
    if a.vlm_model:
        over["vlm"] = {"enabled": True, "model": a.vlm_model}
    eng = Drishti(load_config(a.config, over))
    labels = Path(a.labels)
    rows = list(csv.DictReader(open(labels, newline="", encoding="utf-8")))
    missing = [r["image_path"] for r in rows if not (labels.parent / r["image_path"]).exists()]
    if missing:
        print(f"missing images listed in {labels}: {', '.join(missing)}")
        print("for samples/, run: bash scripts/fetch_samples.sh samples")
        return 1

    stats: dict[int, list] = defaultdict(list)
    lat = []
    for row in rows:
        img = cv2.imread(str(labels.parent / row["image_path"]))
        truth = normalize_plate(row["plate_text"].split("|")[0])
        ref = [r for r in eng.recognize(img) if r.read.text == truth]
        if not ref:
            print(f"skip {row['image_path']}: the original photo is not read correctly")
            continue
        width = ref[0].bbox.w
        for w in a.widths:
            for q in a.jpeg:
                for b in a.blur:
                    im = degrade(img, w / width, q, b)
                    t = time.perf_counter()
                    res = eng.recognize(im)
                    lat.append(time.perf_counter() - t)
                    hit = next((r for r in res if r.read.text == truth), None)
                    stats[w].append((hit is not None, hit.read.confidence if hit else 0.0))
                    if a.verbose:
                        got = [r.read.text for r in res]
                        print(f"{row['image_path']:<20} {w:>3}px q{q:<3} blur{b:<4} {'ok ' if hit else 'MISS'} {got}")

    print("\n| plate width | cases | read exactly | mean confidence when right |")
    print("|---|---|---|---|")
    total = []
    for w in a.widths:
        s = stats[w]
        total += s
        right = [c for ok, c in s if ok]
        conf = f"{np.mean(right):.2f}" if right else "-"
        print(f"| {w} px | {len(s)} | {len(right)} ({100 * len(right) / max(len(s), 1):.0f}%) | {conf} |")
    right = [c for ok, c in total if ok]
    print(f"| all | {len(total)} | {len(right)} ({100 * len(right) / max(len(total), 1):.0f}%) | "
          f"{np.mean(right) if right else 0:.2f} |")
    print(f"\nlatency per photo: median {1000 * np.median(lat):.0f} ms, p95 {1000 * np.percentile(lat, 95):.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
