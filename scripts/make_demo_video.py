"""Synthesise an 'approaching vehicle' CCTV clip from a still photo.

A virtual camera window starts wide (plate tiny) and closes in on a target point,
with per-frame blur, sensor noise, JPEG artefacts and down/up-sampling. Early
frames are illegible on their own, which is what makes the clip useful for
testing tracking and multi-frame fusion.

    python scripts/make_demo_video.py photo.jpg out.mp4 --target 1338 433 --frames 75
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np


def degrade(frame: np.ndarray, rng: np.random.Generator, level: float) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame
    s = 1.0 - 0.45 * level
    small = cv2.resize(out, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    sigma = rng.uniform(0.4, 0.6 + 1.2 * level)
    out = cv2.GaussianBlur(out, (0, 0), sigma)
    if rng.random() < 0.4 * level:  # horizontal motion blur
        k = int(rng.integers(3, 7))
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0 / k
        out = cv2.filter2D(out, -1, kern)
    noise = rng.normal(0, 3 + 7 * level, out.shape).astype(np.float32)
    out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    q = int(rng.integers(int(60 - 40 * level), int(75 - 40 * level)))
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, max(8, q)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def make_clip(img: np.ndarray, target: tuple[float, float], frames: int, out_wh: tuple[int, int], level: float,
              end_width: float, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    H, W = img.shape[:2]
    ow, oh = out_wh
    aspect = oh / ow
    out = []
    for i in range(frames):
        t = i / max(frames - 1, 1)
        e = t * t * (3 - 2 * t)  # smoothstep: slow start, slow finish
        win_w = W * (1 - e) + end_width * e
        win_h = win_w * aspect
        cx = W / 2 * (1 - e) + target[0] * e
        cy = H / 2 * (1 - e) + target[1] * e
        x1 = float(np.clip(cx - win_w / 2, 0, W - win_w))
        y1 = float(np.clip(cy - win_h / 2, 0, H - win_h))
        M = np.array([[ow / win_w, 0, -x1 * ow / win_w], [0, oh / win_h, -y1 * oh / win_h]], dtype=np.float32)
        frame = cv2.warpAffine(img, M, (ow, oh), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
        out.append(degrade(frame, rng, level))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("output")
    ap.add_argument("--target", type=float, nargs=2, required=True, help="x y of the plate in the photo")
    ap.add_argument("--frames", type=int, default=75)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--size", type=int, nargs=2, default=[1280, 720])
    ap.add_argument("--level", type=float, default=0.8, help="degradation 0..1")
    ap.add_argument("--end-width", type=float, default=900, help="final window width in photo pixels")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    img = cv2.imread(a.image)
    frames = make_clip(img, tuple(a.target), a.frames, tuple(a.size), a.level, a.end_width, a.seed)
    vw = cv2.VideoWriter(a.output, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, tuple(a.size))
    for f in frames:
        vw.write(f)
    vw.release()
    print(f"wrote {len(frames)} frames to {a.output}")


if __name__ == "__main__":
    main()
