"""Convert plate datasets to YOLO-pose format with 4 corner keypoints (TL, TR, BR, BL).

A corner head lets Drishti perspective-rectify plates before OCR. Sources:

  ccpd      CCPD-style file names that encode the 4 vertices (290k images; great for
            corner pre-training even though the plates are Chinese)
  labelme   LabelMe JSON: 4-point polygons (corners) or rectangles labelled plate*
  yolo      existing YOLO bbox labels (corners marked not-visible: box loss only)

Output layout (Ultralytics):
  OUT/images/{train,val}/*.jpg   OUT/labels/{train,val}/*.txt
  line: 0 cx cy w h  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4   (normalised)

    python training/detector/convert.py ccpd /data/CCPD2019/ccpd_base out/ccpd --val 0.05
    python training/detector/convert.py labelme /data/indian_labelme out/in --val 0.1
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def order_quad(pts) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def label_line(quad: np.ndarray | None, box: tuple[float, float, float, float], W: int, H: int) -> str:
    x1, y1, x2, y2 = box
    cx, cy, bw, bh = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H
    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
    if quad is None:
        parts += ["0 0 0"] * 4
    else:
        for x, y in quad:
            parts.append(f"{np.clip(x / W, 0, 1):.6f} {np.clip(y / H, 0, 1):.6f} 2")
    return " ".join(parts)


def _emit(out: Path, split: str, img_path: Path, lines: list[str], link: bool) -> None:
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    dst = out / "images" / split / img_path.name
    if not dst.exists():
        if link:
            dst.symlink_to(img_path.resolve())
        else:
            shutil.copy2(img_path, dst)
    (out / "labels" / split / (img_path.stem + ".txt")).write_text("\n".join(lines) + "\n")


def from_ccpd(src: Path):
    for p in sorted(src.rglob("*")):
        if p.suffix.lower() not in IMG_EXT:
            continue
        f = p.stem.split("-")
        if len(f) < 5:
            continue
        try:
            (bx1, by1), (bx2, by2) = [tuple(map(float, xy.split("&"))) for xy in f[2].split("_")]
            v = [tuple(map(float, xy.split("&"))) for xy in f[3].split("_")]  # RB, LB, LT, RT
        except ValueError:
            continue
        quad = np.array([v[2], v[3], v[0], v[1]], dtype=np.float32)
        img = cv2.imread(str(p))
        if img is None:
            continue
        H, W = img.shape[:2]
        yield p, [label_line(quad, (bx1, by1, bx2, by2), W, H)]


def from_labelme(src: Path):
    for jp in sorted(src.rglob("*.json")):
        data = json.loads(jp.read_text())
        img_path = jp.parent / data.get("imagePath", jp.with_suffix(".jpg").name)
        if not img_path.exists():
            continue
        W, H = data.get("imageWidth"), data.get("imageHeight")
        if not W or not H:
            im = cv2.imread(str(img_path))
            H, W = im.shape[:2]
        lines = []
        for sh in data.get("shapes", []):
            if not str(sh.get("label", "")).lower().startswith(("plate", "number", "lp", "license")):
                continue
            pts = np.array(sh["points"], dtype=np.float32)
            if sh.get("shape_type") == "rectangle" or len(pts) == 2:
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                lines.append(label_line(None, (x1, y1, x2, y2), W, H))
            elif len(pts) == 4:
                q = order_quad(pts)
                x1, y1 = q.min(axis=0)
                x2, y2 = q.max(axis=0)
                lines.append(label_line(q, (x1, y1, x2, y2), W, H))
        if lines:
            yield img_path, lines


def from_yolo(src: Path):
    """src has images/ and labels/ (YOLO bbox, any class -> plate)."""
    for lp in sorted((src / "labels").rglob("*.txt")):
        rel = lp.relative_to(src / "labels").with_suffix("")
        img = next((src / "images" / rel.parent / (rel.name + e) for e in IMG_EXT if (src / "images" / rel.parent / (rel.name + e)).exists()), None)
        if img is None:
            continue
        lines = []
        for row in lp.read_text().split("\n"):
            v = row.split()
            if len(v) >= 5:
                cx, cy, w, h = map(float, v[1:5])
                lines.append(" ".join(["0", f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"] + ["0 0 0"] * 4))
        if lines:
            yield img, lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fmt", choices=["ccpd", "labelme", "yolo"])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    gen = {"ccpd": from_ccpd, "labelme": from_labelme, "yolo": from_yolo}[a.fmt](Path(a.src))
    rnd = random.Random(a.seed)
    out = Path(a.out)
    n = 0
    for img, lines in gen:
        _emit(out, "val" if rnd.random() < a.val else "train", img, lines, link=not a.copy)
        n += 1
        if a.limit and n >= a.limit:
            break
    print(f"converted {n} images -> {out}")


if __name__ == "__main__":
    main()
