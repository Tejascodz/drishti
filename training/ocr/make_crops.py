"""Build an OCR training set from full images, cropped exactly the way Drishti
crops at inference time (same padding views, same rectification), so training
and deployment see the same framing.

Two input modes:

  boxes   CSV: image_path,x1,y1,x2,y2,plate_text[,c1x,c1y,c2x,c2y,c3x,c3y,c4x,c4y]
          (ground-truth boxes, optional TL/TR/BR/BL corners)
  auto    CSV: image_path,plate_text   (several plates separated by '|')
          runs the Drishti detector + OCR and keeps detections whose read is within
          `--max-edit` of a label — cheap labelling when only plate strings are known.

    python training/ocr/make_crops.py boxes labels.csv data/ocr_real --views 3 --val 0.1
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from drishti.fusion import levenshtein  # noqa: E402
from drishti.geometry import crop_box, order_quad, warp_quad  # noqa: E402
from drishti.types import BBox  # noqa: E402
from drishti.watchlist import normalize_plate  # noqa: E402


def _views(img, bbox: BBox, quad, rng: random.Random, n: int):
    out = []
    for _ in range(n):
        fx, fy = rng.uniform(0.02, 0.18), rng.uniform(0.06, 0.28)  # spans Drishti's default view range
        if quad is not None and rng.random() < 0.7:
            try:
                out.append(warp_quad(img, quad, fx, fy))
                continue
            except ValueError:
                pass
        out.append(crop_box(img, bbox, fx, fy))
    return out


def mode_boxes(csv_path: Path, rng: random.Random, n_views: int):
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    cache: dict[str, np.ndarray] = {}
    for r in rows:
        ip = str((csv_path.parent / r["image_path"]).resolve())
        if ip not in cache:
            cache.clear()
            cache[ip] = cv2.imread(ip)
        img = cache[ip]
        if img is None:
            continue
        b = BBox(float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))
        quad = None
        if r.get("c1x"):
            quad = order_quad([[float(r[f"c{i}x"]), float(r[f"c{i}y"])] for i in range(1, 5)])
        text = normalize_plate(r["plate_text"])
        for v in _views(img, b, quad, rng, n_views):
            yield v, text


def mode_auto(csv_path: Path, rng: random.Random, n_views: int, max_edit: int, config: str | None):
    from drishti.config import load_config
    from drishti.pipeline import Drishti

    eng = Drishti(load_config(config, {"models": {"vehicle_detector": None}}))
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    kept = 0
    for r in rows:
        img = cv2.imread(str(csv_path.parent / r["image_path"]))
        if img is None:
            continue
        labels = [normalize_plate(t) for t in r["plate_text"].split("|") if t.strip()]
        for res in eng.recognize(img):
            cand = res.read.text or res.read.raw_text or ""
            best = min(labels, key=lambda t: levenshtein(t, cand)) if labels else None
            if best is None or levenshtein(best, cand) > max_edit:
                continue
            kept += 1
            for v in _views(img, res.bbox, res.corners, rng, n_views):
                yield v, best
    print(f"auto mode kept {kept} plates", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["boxes", "auto"])
    ap.add_argument("labels")
    ap.add_argument("out")
    ap.add_argument("--views", type=int, default=3, help="randomly padded crops per plate")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--max-edit", type=int, default=2)
    ap.add_argument("--config")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    writers = {}
    files = []
    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        fh = open(out / split / "annotations.csv", "w", newline="", encoding="utf-8")
        files.append(fh)
        writers[split] = csv.writer(fh)
        writers[split].writerow(["image_path", "plate_text"])
    gen = mode_boxes(Path(a.labels), rng, a.views) if a.mode == "boxes" else mode_auto(Path(a.labels), rng, a.views, a.max_edit, a.config)
    # split by plate string so the same vehicle never lands in both train and val
    split_of: dict[str, str] = {}
    n = 0
    for crop, text in gen:
        if crop.size == 0 or not text:
            continue
        split = split_of.setdefault(text, "val" if rng.random() < a.val else "train")
        name = f"{n:07d}.jpg"
        cv2.imwrite(str(out / split / "images" / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        writers[split].writerow([f"images/{name}", text])
        n += 1
    for fh in files:
        fh.close()
    print(f"wrote {n} crops -> {out}")


if __name__ == "__main__":
    main()
