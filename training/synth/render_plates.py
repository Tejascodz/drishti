"""Synthetic Indian plate generator for OCR training.

Strings are sampled from the same region pack the decoder uses (so every sample
is a valid registration mark), rendered in HSRP-like layouts and colour schemes,
then pushed through a realistic capture chain (degrade.py). Output is a
fast-plate-ocr dataset: images/ + annotations.csv (image_path, plate_text).

    bash training/synth/fetch_fonts.sh
    python training/synth/render_plates.py --out data/synth_in --n 60000 --val 2000

Half of the plates are seen from above (toll-plaza camera geometry) and 15% are
infrared night frames by default; see --overhead-frac and --ir-frac.

Two-line plates (two-wheelers, goods vehicles) are rendered stacked and labelled
top line first, which is how a slot OCR reads a stacked plate.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from degrade import degrade_plate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "drishti" / "formats" / "packs" / "india.yaml"
FONT_DIR = Path(__file__).resolve().parent / "fonts"
FALLBACK_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Narrow Bold.ttf",
    "C:/Windows/Fonts/arialnb.ttf",
]

# Registration volume is very uneven across states; weight the common ones up.
STATE_WEIGHTS = {"MH": 9, "UP": 9, "KA": 8, "TN": 8, "DL": 7, "GJ": 7, "RJ": 6, "HR": 5, "KL": 5, "TG": 5, "AP": 5,
                 "MP": 5, "WB": 5, "PB": 4, "BR": 4, "OD": 3, "CG": 3, "JH": 3, "AS": 2, "UK": 2, "HP": 2, "JK": 2, "TS": 2,
                 "GA": 1, "CH": 1, "PY": 1}

# (pattern name, weight) — mirrors real-world mix, with rarer layouts oversampled a little
LAYOUT_WEIGHTS = {"std2": 0.52, "std1": 0.17, "dl": 0.08, "std0": 0.03, "bh": 0.08, "va": 0.01, "army": 0.01, "diplo": 0.01,
                  "std3": 0.0}

# category -> (background RGB, text RGB, probability)
SCHEMES = {
    "private": ((245, 245, 242), (15, 15, 15), 0.52),
    "commercial": ((250, 205, 25), (15, 15, 15), 0.2),
    "ev_private": ((25, 140, 70), (250, 250, 250), 0.13),
    "ev_commercial": ((25, 140, 70), (250, 205, 25), 0.05),
    "rental": ((15, 15, 15), (250, 205, 25), 0.05),
    "army": ((15, 15, 15), (245, 245, 245), 0.02),
    "diplomatic": ((20, 60, 160), (245, 245, 245), 0.03),
}

DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SERIES = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # I and O are not issued


def _pack_states() -> list[str]:
    data = yaml.safe_load(PACK.read_text())
    s = data["sets"]["STATE"]
    return list(s["items"]) + [k for k in (s.get("soft") or {})]


class TextSampler:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        states = _pack_states()
        w = np.array([STATE_WEIGHTS.get(s, 0.7) for s in states], dtype=np.float64)
        self.states, self.state_p = states, w / w.sum()
        names = [k for k, v in LAYOUT_WEIGHTS.items() if v > 0]
        p = np.array([LAYOUT_WEIGHTS[k] for k in names])
        self.layouts, self.layout_p = names, p / p.sum()

    def _d(self, n: int) -> str:
        return "".join(self.rng.choice(list(DIGITS), n))

    def _s(self, n: int) -> str:
        return "".join(self.rng.choice(list(SERIES), n))

    def sample(self) -> tuple[str, list[str], str]:
        """-> (label, groups for rendering, layout name)"""
        lay = str(self.rng.choice(self.layouts, p=self.layout_p))
        st = str(self.rng.choice(self.states, p=self.state_p))
        rto = f"{int(self.rng.integers(1, 100)):02d}"
        num = f"{int(self.rng.integers(1, 10000)):04d}"
        if lay == "std2":
            g = [st, rto, self._s(2), num]
        elif lay == "std1":
            g = [st, rto, self._s(1), num]
        elif lay == "std3":
            g = [st, rto, self._s(3), num]
        elif lay == "std0":
            g = [st, rto, num]
        elif lay == "dl":
            g = ["DL", str(int(self.rng.integers(1, 14))), self._s(int(self.rng.integers(1, 3))), num]
            if len("".join(g)) > 10:
                g[2] = g[2][:1]
        elif lay == "bh":
            g = [f"{int(self.rng.integers(21, 27)):02d}", "BH", num, self._s(int(self.rng.integers(1, 3)))]
        elif lay == "va":
            g = [st, "VA", self._s(2), num]
        elif lay == "army":
            g = [f"{int(self.rng.integers(0, 27)):02d}", str(self.rng.choice(list(LETTERS))), self._d(6), str(self.rng.choice(list(LETTERS)))]
        else:  # diplomatic
            g = [str(int(self.rng.integers(1, 200))), str(self.rng.choice(["CD", "CC", "UN"])), str(int(self.rng.integers(1, 9999)))]
        return "".join(g), g, lay


def _fonts() -> list[str]:
    found = [str(p) for p in sorted(FONT_DIR.glob("*.ttf"))] if FONT_DIR.exists() else []
    found += [f for f in FALLBACK_FONTS if os.path.exists(f)]
    if not found:
        raise SystemExit("no fonts found: run training/synth/fetch_fonts.sh")
    return found


def _fit_font(path: str, text: str, max_w: int, max_h: int) -> ImageFont.FreeTypeFont:
    size = max_h
    while size > 6:
        f = ImageFont.truetype(path, size)
        x0, y0, x1, y1 = f.getbbox(text)
        if x1 - x0 <= max_w and y1 - y0 <= max_h:
            return f
        size -= 2
    return ImageFont.truetype(path, 6)


def _draw_line(draw: ImageDraw.ImageDraw, text: str, box: tuple[int, int, int, int], font_path: str, color, rng) -> None:
    x0, y0, x1, y1 = box
    f = _fit_font(font_path, text, x1 - x0, y1 - y0)
    bx0, by0, bx1, by1 = f.getbbox(text)
    tw, th = bx1 - bx0, by1 - by0
    jitter = int(rng.integers(-max(1, (x1 - x0 - tw) // 6), max(2, (x1 - x0 - tw) // 6 + 1))) if x1 - x0 > tw else 0
    x = x0 + (x1 - x0 - tw) // 2 - bx0 + jitter
    y = y0 + (y1 - y0 - th) // 2 - by0
    draw.text((x, y), text, font=f, fill=color)


def render(groups: list[str], layout: str, rng: np.random.Generator, fonts: list[str], two_line: bool) -> np.ndarray:
    """Render an RGBA plate image."""
    if layout == "army":
        scheme = "army"
    elif layout == "diplo":
        scheme = "diplomatic"
    else:
        names = [n for n in SCHEMES if n not in ("army", "diplomatic")]
        p = np.array([SCHEMES[n][2] for n in names])
        scheme = str(rng.choice(names, p=p / p.sum()))
    bg, fg, _ = SCHEMES[scheme]
    hsrp = rng.random() < 0.85 and scheme not in ("army", "diplomatic")

    if two_line:
        W, H = (400, 200) if rng.random() < 0.6 else (340, 200)
    elif not hsrp and rng.random() < 0.4:
        W, H = 560, 80  # long thin aftermarket strip (scooters, EVs): about 7:1
    else:
        W, H = 520, 112
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    radius = int(0.06 * H)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=radius, fill=bg + (255,))
    border = max(2, H // 40)
    if rng.random() < 0.9:
        d.rounded_rectangle([border, border, W - 1 - border, H - 1 - border], radius=radius, outline=fg + (255,), width=border)

    left = int(0.03 * W)
    if hsrp:  # chromium hologram + vertical "IND" in blue at the left edge
        hx = int(0.045 * W) if not two_line else int(0.07 * W)
        hr = int(0.11 * H) if not two_line else int(0.07 * H)
        d.ellipse([hx - hr, int(0.25 * H) - hr, hx + hr, int(0.25 * H) + hr], fill=(175, 185, 200, 255))
        ind_font = ImageFont.truetype(fonts[0], max(8, int(0.16 * H) if not two_line else int(0.1 * H)))
        for i, ch in enumerate("IND"):
            d.text((hx - int(0.35 * hr), int(0.42 * H) + i * int(0.16 * H if not two_line else 0.1 * H)), ch, font=ind_font, fill=(40, 80, 170, 255))
        left = int(0.1 * W) if not two_line else int(0.13 * W)
        if rng.random() < 0.6:  # laser-etched serial along the bottom edge
            sf = ImageFont.truetype(fonts[0], max(6, int(0.07 * H)))
            d.text((int(0.35 * W), int(0.9 * H) - 2), "".join(rng.choice(list(DIGITS), 10)), font=sf, fill=fg + (160,))
    elif not two_line and rng.random() < 0.35:  # aftermarket emblem at the left edge (a ring with a mark)
        er = int(0.24 * H)
        ex, ey = int(0.035 * W) + er, H // 2
        d.ellipse([ex - er, ey - er, ex + er, ey + er], outline=fg + (255,), width=max(2, er // 4))
        d.arc([ex - er // 2, ey - er // 2, ex + er // 2, ey + er // 2], 200, 340, fill=fg + (255,), width=max(1, er // 5))
        left = ex + er + int(0.02 * W)

    font_path = str(rng.choice(fonts))
    # separators between groups: none, space, hyphen, or the dots of aftermarket and EV plates (KA·53HS·3074);
    # the label never contains them, so the OCR learns to read through them
    sep = str(rng.choice(["", " ", "-", "\u00b7", "\u2022"], p=[0.3, 0.45, 0.08, 0.1, 0.07]))
    if sep and len(groups) == 4 and layout != "bh" and rng.random() < 0.4:
        groups = [groups[0], groups[1] + groups[2], groups[3]]  # KA·53HS·3074: RTO and series run together
    right = W - int(0.03 * W)
    if two_line:
        if layout == "bh":
            top, bot = "".join(groups[:2]), "".join(groups[2:])
        elif len(groups) == 4:
            top, bot = sep.join(groups[:3]), groups[3]
        else:
            top, bot = sep.join(groups[:-1]), groups[-1]
        _draw_line(d, top, (left, int(0.08 * H), right, int(0.47 * H)), font_path, fg + (255,), rng)
        _draw_line(d, bot, (left, int(0.53 * H), right, int(0.92 * H)), font_path, fg + (255,), rng)
    else:
        _draw_line(d, sep.join(groups), (left, int(0.14 * H), right, int(0.86 * H)), font_path, fg + (255,), rng)

    arr = np.array(img)
    if rng.random() < 0.15:  # italic / stylised aftermarket fonts
        sh = rng.uniform(-0.25, 0.25)
        M = np.float32([[1, sh, -sh * H / 2], [0, 1, 0]])
        arr = cv2.warpAffine(arr, M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--two-line-frac", type=float, default=0.25)
    ap.add_argument("--hard", type=float, default=0.6, help="0..1 strength of geometric distortion")
    ap.add_argument("--overhead-frac", type=float, default=0.5,
                    help="fraction seen from a camera above the lane (toll plaza / gantry geometry)")
    ap.add_argument("--ir-frac", type=float, default=0.15, help="fraction rendered as infrared night frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", type=int, default=0, help="also write a grid of N samples")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    fonts = _fonts()
    sampler = TextSampler(rng)
    out = Path(a.out)
    previews = []
    for split, count in (("train", a.n), ("val", a.val)):
        if count <= 0:
            continue
        img_dir = out / split / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        with open(out / split / "annotations.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["image_path", "plate_text"])
            for i in range(count):
                label, groups, layout = sampler.sample()
                two = rng.random() < a.two_line_frac
                plate = render(groups, layout, rng, fonts, two)
                crop = degrade_plate(plate, rng, a.hard, overhead=rng.random() < a.overhead_frac,
                                     ir=rng.random() < a.ir_frac)
                name = f"{split}_{i:07d}.jpg"
                cv2.imwrite(str(img_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                w.writerow([f"images/{name}", label])
                if len(previews) < a.preview:
                    previews.append((crop, label))
                if (i + 1) % 5000 == 0:
                    print(f"{split}: {i + 1}/{count}", flush=True)
    if previews:
        tiles = []
        for crop, label in previews:
            t = cv2.resize(crop, (256, 64))
            cv2.putText(t, label, (2, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
            tiles.append(t)
        cols = 4
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[r * cols : (r + 1) * cols]) for r in range(len(tiles) // cols)])
        cv2.imwrite(str(out / "preview.jpg"), grid)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
