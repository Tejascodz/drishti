"""Pascal VOC XML (with a plate-text attribute) -> Drishti eval CSV.

Handles the DataCluster Indian plates sample, where each <object> carries the
plate characters in an attribute such as `number_plate_text`, and CVAT-style
<attributes><attribute><name>..</name><value>..</value></attribute></attributes>.

    python scripts/voc_to_csv.py annotations/ labels.csv [--images ../images]
"""

from __future__ import annotations

import argparse
import csv
import os
import xml.etree.ElementTree as ET
from pathlib import Path

TEXT_KEYS = ("number_plate_text", "plate_text", "text", "license_plate_text", "number")


def plate_texts(xml_path: Path) -> tuple[str | None, list[str]]:
    root = ET.parse(xml_path).getroot()
    fname = root.findtext("filename")
    texts = []
    for obj in root.iter("object"):
        val = None
        for k in TEXT_KEYS:
            node = obj.find(k)
            if node is not None and (node.text or "").strip():
                val = node.text
                break
        if val is None:
            for attr in obj.iter("attribute"):
                if (attr.findtext("name") or "").strip().lower() in TEXT_KEYS:
                    val = attr.findtext("value")
                    break
        if val:
            t = "".join(ch for ch in val.upper() if ch.isalnum())
            if t:
                texts.append(t)
    return fname, texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("annotations")
    ap.add_argument("out_csv")
    ap.add_argument("--images", help="image folder (default: ../images next to the annotations)")
    a = ap.parse_args()
    ann = Path(a.annotations)
    img_dir = Path(a.images) if a.images else ann.parent / "images"
    out = Path(a.out_csv)
    rows, no_text = [], 0
    for xp in sorted(ann.glob("*.xml")):
        fname, texts = plate_texts(xp)
        img = img_dir / (fname or xp.with_suffix(".jpg").name)
        if not img.exists():
            cands = list(img_dir.glob(xp.stem + ".*"))
            if not cands:
                continue
            img = cands[0]
        if not texts:
            no_text += 1
            continue
        rows.append((os.path.relpath(img, out.parent), "|".join(texts)))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "plate_text"])
        w.writerows(rows)
    print(f"wrote {len(rows)} images to {out} ({no_text} without plate text skipped)")


if __name__ == "__main__":
    main()
