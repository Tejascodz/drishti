"""Capture-chain degradations for synthetic plates.

The order mirrors a real camera: geometry (perspective, roll) -> optics (defocus,
motion blur) -> illumination (shadow, glare, night) -> sensor (low resolution,
noise) -> codec (JPEG). Sampling is biased towards the hard end because that is
where recognition systems fail in the field.
"""

from __future__ import annotations

import cv2
import numpy as np


def perspective(img: np.ndarray, rng: np.random.Generator, strength: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    """Random perspective warp. Returns warped image and the plate's 4 corners in it."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jit = rng.uniform(-strength, strength, (4, 2)) * np.float32([w, h])
    dst = src + jit
    dst -= dst.min(axis=0)
    ow, oh = int(np.ceil(dst[:, 0].max())), int(np.ceil(dst[:, 1].max()))
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    out = cv2.warpPerspective(img, M, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return out, dst


def overhead_view(img: np.ndarray, rng: np.random.Generator, pitch=(12.0, 45.0), yaw=(-18.0, 18.0),
                  roll=(-4.0, 4.0), depth=(3.0, 10.0)) -> tuple[np.ndarray, np.ndarray]:
    """Plate seen by a camera mounted above the lane (toll plaza / gate gantry).

    The plate is rotated in 3D and projected with a pinhole camera: looking down by
    `pitch` degrees squashes it vertically and makes the top edge wider than the
    bottom; `yaw` covers cameras offset to the side of the lane; `depth` (distance in
    plate widths) sets how strong the perspective is. Returns (image, corners)."""
    h, w = img.shape[:2]
    th, ph, ro = (np.radians(rng.uniform(*r)) for r in (pitch, yaw, roll))
    Rx = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    Ry = np.array([[np.cos(ph), 0, np.sin(ph)], [0, 1, 0], [-np.sin(ph), 0, np.cos(ph)]])
    Rz = np.array([[np.cos(ro), -np.sin(ro), 0], [np.sin(ro), np.cos(ro), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    corners = np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]], dtype=np.float64)
    z0 = rng.uniform(*depth) * w
    cam = corners @ R.T
    f = z0
    proj = np.stack([f * cam[:, 0] / (z0 + cam[:, 2]), f * cam[:, 1] / (z0 + cam[:, 2])], axis=1)
    proj -= proj.min(axis=0)
    ow, oh = int(np.ceil(proj[:, 0].max())) + 1, int(np.ceil(proj[:, 1].max())) + 1
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, proj.astype(np.float32))
    out = cv2.warpPerspective(img, M, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return out, proj


def infrared(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Night frame from an IR-illuminated camera: monochrome, retro-reflective plate
    much brighter than the vehicle, heavy noise, occasional headlight bloom."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = np.clip((g - g.mean()) * rng.uniform(1.2, 2.0) + rng.uniform(90, 150), 0, 255)
    if rng.random() < 0.4:
        h, w = g.shape
        cx, cy = rng.integers(0, w), rng.integers(h // 2, h)
        yy, xx = np.mgrid[0:h, 0:w]
        g += rng.uniform(60, 160) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (rng.uniform(0.15, 0.4) * w) ** 2))
    g += rng.normal(0, rng.uniform(4, 14), g.shape)
    return cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def place_on_background(plate_rgba: np.ndarray, rng: np.random.Generator, pad: tuple[float, float]) -> np.ndarray:
    """Composite a plate (with alpha) onto a vehicle-like background with padding."""
    h, w = plate_rgba.shape[:2]
    px, py = int(w * pad[0]), int(h * pad[1])
    H, W = h + 2 * py, w + 2 * px
    base = rng.integers(10, 245, 3).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    grad = (xx / W * rng.uniform(-60, 60) + yy / H * rng.uniform(-60, 60))[..., None]
    bg = np.clip(base[None, None, :] + grad + rng.normal(0, rng.uniform(2, 12), (H, W, 3)), 0, 255)
    if rng.random() < 0.5:  # bumper / grille stripes
        for _ in range(int(rng.integers(1, 4))):
            y = int(rng.integers(0, H))
            cv2.rectangle(bg, (0, y), (W, y + int(rng.integers(2, max(3, H // 6)))), rng.integers(0, 255, 3).tolist(), -1)
    alpha = plate_rgba[..., 3:4].astype(np.float32) / 255.0
    roi = bg[py : py + h, px : px + w]
    bg[py : py + h, px : px + w] = roi * (1 - alpha) + plate_rgba[..., :3].astype(np.float32) * alpha
    return bg.astype(np.uint8)


def illumination(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.astype(np.float32)
    h, w = out.shape[:2]
    if rng.random() < 0.35:  # cast shadow with a soft edge
        pts = np.int32([[rng.integers(0, w), 0], [rng.integers(0, w), h], [w, h], [w, 0]])
        mask = np.zeros((h, w), np.float32)
        cv2.fillPoly(mask, [pts], 1.0)
        mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, w / 40))
        out *= (1 - rng.uniform(0.25, 0.6) * mask)[..., None]
    if rng.random() < 0.15:  # specular glare / headlight bloom
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        rad = rng.uniform(0.1, 0.5) * w
        yy, xx = np.mgrid[0:h, 0:w]
        bloom = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rad**2))
        out += rng.uniform(80, 200) * bloom[..., None]
    if rng.random() < 0.12:  # night
        out *= rng.uniform(0.2, 0.5)
    gain, bias = rng.uniform(0.6, 1.3), rng.uniform(-35, 35)
    out = out * gain + bias
    gamma = rng.uniform(0.7, 1.5)
    out = 255.0 * (np.clip(out, 0, 255) / 255.0) ** gamma
    return np.clip(out, 0, 255).astype(np.uint8)


def dirt_and_occlusion(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    if rng.random() < 0.25:
        for _ in range(int(rng.integers(3, 15))):
            c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            r = int(rng.integers(1, max(2, h // 8)))
            col = rng.integers(40, 140, 3).tolist()
            cv2.circle(out, c, r, col, -1)
    if rng.random() < 0.06:  # rope, tow bar or garland across the plate
        y = int(rng.integers(int(0.2 * h), int(0.8 * h)))
        cv2.line(out, (0, y + int(rng.integers(-h // 4, h // 4))), (w, y), rng.integers(0, 255, 3).tolist(), int(rng.integers(1, max(2, h // 10))))
    return out


def optics(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img
    if rng.random() < 0.6:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 1.6))
    if rng.random() < 0.3:
        k = int(rng.integers(3, 9))
        kern = np.zeros((k, k), np.float32)
        ang = rng.uniform(-20, 20)
        cv2.line(kern, (0, k // 2), (k - 1, k // 2), 1.0, 1)
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), ang, 1.0)
        kern = cv2.warpAffine(kern, M, (k, k))
        kern /= max(kern.sum(), 1e-6)
        out = cv2.filter2D(out, -1, kern)
    return out


def sensor_and_codec(img: np.ndarray, rng: np.random.Generator, min_h: int = 14, max_h: int = 72) -> np.ndarray:
    h, w = img.shape[:2]
    # bias towards small plates: they are the common failure mode
    target_h = int(np.clip(rng.gamma(2.2, 11) + min_h, min_h, max_h))
    scale = target_h / h
    small = cv2.resize(img, (max(8, int(w * scale)), target_h), interpolation=cv2.INTER_AREA)
    noise = rng.normal(0, rng.uniform(0, 9), small.shape)
    small = np.clip(small.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    q = int(np.clip(rng.normal(55, 22), 12, 95))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else small


def degrade_plate(plate_rgba: np.ndarray, rng: np.random.Generator, hard: float = 0.6, overhead: bool = False,
                  ir: bool = False) -> np.ndarray:
    """Full chain: plate (RGBA, any size) -> realistic crop as a detector would deliver it.

    overhead: camera above the lane looking down (toll plaza geometry)
    ir:       night frame from an infrared-illuminated camera"""
    img = plate_rgba
    if overhead:
        img, _ = overhead_view(img, rng)
    elif rng.random() < 0.85:
        img, _ = perspective(img, rng, strength=0.03 + 0.07 * hard)
    if rng.random() < 0.5:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-7, 7), 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    pad = (rng.uniform(0.0, 0.18), rng.uniform(0.0, 0.3))
    img = place_on_background(img, rng, pad)
    img = dirt_and_occlusion(img, rng)
    img = infrared(img, rng) if ir else illumination(img, rng)
    img = optics(img, rng)
    return sensor_and_codec(img, rng)
