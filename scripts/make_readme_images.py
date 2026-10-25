"""Build the README results image from real photos.

Every photo is read with the given config; photos with at least one read plate are
candidates. The grid shows each picked photo annotated, with the plate crop enlarged
underneath next to the text, confidence and level Drishti reported. Picks prefer
confident reads from different states, so the grid shows variety, not repetition.

Only use photos you may publish (your own, or openly licensed such as the CC0
`kedarsai/indian-license-plates-with-labels` Kaggle set).

    python scripts/make_readme_images.py data/kaggle_in/images --config models/drishti.yaml --out docs/images
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drishti.config import load_config  # noqa: E402
from drishti.pipeline import Drishti, iter_images  # noqa: E402

TILE_W, PHOTO_H, STRIP_H = 640, 360, 96
INK, PAPER = (235, 235, 235), (32, 32, 32)
LEVEL_COLOUR = {"Good": (90, 200, 90), "Average": (40, 170, 240), "Poor": (80, 80, 230)}


def letterbox(img: np.ndarray, w: int, h: int) -> np.ndarray:
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    out = np.zeros((h, w, 3), np.uint8)
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y : y + r.shape[0], x : x + r.shape[1]] = r
    return out


def tile(eng: Drishti, img: np.ndarray, res) -> np.ndarray:
    photo = letterbox(eng.annotate(img, res), TILE_W, PHOTO_H)
    strip = np.full((STRIP_H, TILE_W, 3), PAPER, np.uint8)
    best = max((r for r in res if r.read.text), key=lambda r: r.read.confidence)
    crop = best.crop if best.crop is not None and best.crop.size else img[:1, :1]
    ch = STRIP_H - 16
    cw = min(int(crop.shape[1] * ch / max(crop.shape[0], 1)), TILE_W // 2)
    strip[8 : 8 + ch, 8 : 8 + cw] = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_CUBIC)
    x = 24 + cw
    cv2.putText(strip, best.read.text, (x, 44), cv2.FONT_HERSHEY_DUPLEX, 1.1, INK, 2, cv2.LINE_AA)
    level = best.read.level or ""
    cv2.putText(strip, f"{best.read.confidence:.2f} {level}", (x, 80), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                LEVEL_COLOUR.get(level, INK), 1, cv2.LINE_AA)
    return np.vstack([photo, strip])


def pick(cands: list[tuple[float, str, Path]], n: int) -> list[Path]:
    """Most confident first, one photo per state code until every state is used."""
    cands = sorted(cands, key=lambda c: -c[0])
    chosen, states = [], set()
    for conf, text, path in cands:
        if text[:2] not in states and len(chosen) < n:
            chosen.append(path)
            states.add(text[:2])
    for _, _, path in cands:
        if len(chosen) >= n:
            break
        if path not in chosen:
            chosen.append(path)
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="photos or folders of photos you may publish")
    ap.add_argument("--config", action="append", default=None, help="e.g. models/drishti.yaml")
    ap.add_argument("--out", default="docs/images")
    ap.add_argument("--count", type=int, default=6, help="photos in the grid (even number)")
    ap.add_argument("--scan", type=int, default=300, help="read at most this many photos")
    ap.add_argument("--providers", default="cpu")
    a = ap.parse_args()

    cfg = load_config(a.config, {"runtime": {"providers": a.providers}, "models": {"vehicle_detector": None}})
    eng = Drishti(cfg)
    cands, n_photos, n_found, n_read = [], 0, 0, 0
    for path in iter_images(a.inputs):
        if n_photos >= a.scan:
            break
        img = cv2.imread(str(path))
        if img is None:
            continue
        n_photos += 1
        res = eng.recognize(img)
        n_found += len(res)
        reads = [r for r in res if r.read.text]
        n_read += len(reads)
        if reads:
            top = max(reads, key=lambda r: r.read.confidence)
            cands.append((top.read.confidence, top.read.text, path))
        print(f"\r{n_photos} photos, {n_found} plates found, {n_read} read", end="", flush=True)
    print()
    if len(cands) < 2:
        print("not enough photos with a readable plate")
        return 1

    tiles = []
    for path in pick(cands, a.count - a.count % 2):
        img = cv2.imread(str(path))
        tiles.append(tile(eng, img, eng.recognize(img)))
    if len(tiles) % 2:
        tiles = tiles[:-1]
    grid = np.vstack([np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "results.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {out / 'results.jpg'}  ({len(tiles)} photos picked from {n_photos} read; "
          f"{n_found} plates found, {n_read} read)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
