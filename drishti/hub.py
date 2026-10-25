"""Model registry and downloader.

Hub names resolve to files in the cache directory; anything else is treated as a
local path. All default weights are permissively licensed (MIT / Apache-2.0).
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .config import expand_path

log = logging.getLogger(__name__)

_OIM = "https://github.com/ankandrew/open-image-models/releases/download/assets"
_FPO = "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates"

# SHA-256 of every file the hub downloads. A download that does not match is discarded,
# so a replaced or tampered release asset is never loaded.
_SHA256 = {
    "cct_s_v1_global.keras": "6960906f834b5acedb80897c1db9dda0444ec03236be00c3461051a843383b90",
    "cct_s_v1_global.onnx": "5c95b3231fff415b05cb48a3a39fab7c009364c2ee441bef092e0162ab75ea74",
    "cct_s_v1_global_plate_config.yaml": "8ade332ef026520a3360a7c154cc10be664bee25482f3997bce34a7007b4786e",
    "cct_s_v2_global.keras": "dc6db494714d88e1fb0a21b4e4e96089842dd5440418262abfc382454453c2c8",
    "cct_s_v2_global.onnx": "384bbbd2cea3ef54761d3df70822ef3a349ee1a112aeafddbe0e3ba06bc6e47b",
    "cct_s_v2_global_plate_config.yaml": "0335c74a305173bb6f393efed0fde03cadeaa0b649ed8e19f431016d8232d0a6",
    "cct_xs_v1_global.keras": "fa773df224bb04af602f308188d8db1af5ab45f4929a19ad16ad7a2e2c70bba9",
    "cct_xs_v1_global.onnx": "c1ae2ae031b10f8ab7ed6e3cdaf8d9c753477e299dbad3579c8f705a7d882134",
    "cct_xs_v1_global_plate_config.yaml": "8ade332ef026520a3360a7c154cc10be664bee25482f3997bce34a7007b4786e",
    "cct_xs_v2_global.keras": "0716717772b1f8d25b3c227e1e65e7f42e63900ec017059b4a32155488735ffd",
    "cct_xs_v2_global.onnx": "8031afb5fdc6b4d80462c9d542f1284ebd2cfddf5dbacd62609848d7e2855f44",
    "cct_xs_v2_global_plate_config.yaml": "0335c74a305173bb6f393efed0fde03cadeaa0b649ed8e19f431016d8232d0a6",
    "rf-detr-large-704-coco.onnx": "9897fcdf13040c229d8adb0ac8d0dd8e445d12fa8c37eac7bc91f37d361b44bc",
    "rf-detr-medium-576-coco.onnx": "6a740726ea074cb8bcf0fab34c7b3e8e80a13939d8557d4fe50a59b35f2455ca",
    "rf-detr-nano-384-coco.onnx": "fdcdd1c566ed4ab4d3700ca1ce6f03edadfec2a0655432e40e4c2734623feaec",
    "rf-detr-small-512-coco.onnx": "674fb2fe151f49f569de3f3e91f512cb2b8159ad37c65223334a5cc3dc40febd",
    "yolo-v9-s-608-license-plates-end2end.onnx": "2b878b38d9aa07b6ddc3ea75c4ffcb39869bc5c218e0a14002f60ab2f7b0be9a",
    "yolo-v9-t-256-license-plates-end2end.onnx": "938a34ce868da69432fd5a872f24337b86e3dc8753426e16ae18163f615c2a2c",
    "yolo-v9-t-384-license-plates-end2end.onnx": "888397b96d761c89db40bc9c305838e8652660f5e282c2cadebbe8d2951a77a8",
    "yolo-v9-t-416-license-plates-end2end.onnx": "0469d81f176081671ca4caca1e05f94758eae2f9f659bb8a6c410216f2173fe4",
    "yolo-v9-t-512-license-plates-end2end.onnx": "746fdd358ec110418775d7c9d8d07910d48b1a21471f92bf4421f6510d6daade",
    "yolo-v9-t-640-license-plates-end2end.onnx": "c3c1026ca7d0585dd88084d68182dd897113712fa734ae1557ca70174440c076",
}


@dataclass(frozen=True)
class ModelSpec:
    kind: str  # plate_detector | vehicle_detector | ocr
    files: dict[str, str]  # role -> url
    fmt: str
    license: str
    notes: str = ""
    meta: dict = field(default_factory=dict)


REGISTRY: dict[str, ModelSpec] = {}

for _size, _res in [("s", 608), ("t", 640), ("t", 512), ("t", 416), ("t", 384), ("t", 256)]:
    _name = f"yolo-v9-{_size}-{_res}-license-plate-end2end"
    REGISTRY[_name] = ModelSpec(
        kind="plate_detector",
        files={"model": f"{_OIM}/yolo-v9-{_size}-{_res}-license-plates-end2end.onnx"},
        fmt="yolov9-e2e",
        license="MIT (open-image-models)",
        notes="Global plate detector, NMS embedded in graph.",
    )

for _name, _res in [("nano", 384), ("small", 512), ("medium", 576), ("large", 704)]:
    REGISTRY[f"rf-detr-{_name}-{_res}-coco"] = ModelSpec(
        kind="vehicle_detector",
        files={"model": f"{_OIM}/rf-detr-{_name}-{_res}-coco.onnx"},
        fmt="rfdetr",
        license="Apache-2.0 (RF-DETR, ICLR 2026)",
        notes="COCO detector used for car / motorcycle / bus / truck.",
    )

for _name, _stem in [
    ("cct-s-v2-global-model", "cct_s_v2_global"),
    ("cct-xs-v2-global-model", "cct_xs_v2_global"),
    ("cct-s-v1-global-model", "cct_s_v1_global"),
    ("cct-xs-v1-global-model", "cct_xs_v1_global"),
]:
    REGISTRY[_name] = ModelSpec(
        kind="ocr",
        files={
            "model": f"{_FPO}/{_stem}.onnx",
            "config": f"{_FPO}/{_stem}_plate_config.yaml",
            "keras": f"{_FPO}/{_stem}.keras",
        },
        fmt="slot-ocr",
        license="MIT (fast-plate-ocr)",
        notes="Compact Convolutional Transformer, fixed slots, 65+ countries.",
    )


def cache_root(cache_dir: str | os.PathLike | None = None) -> Path:
    root = expand_path(cache_dir or os.environ.get("DRISHTI_CACHE", "~/.cache/drishti"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", url)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    try:
        digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=120) as resp, os.fdopen(fd, "wb") as out:
            while chunk := resp.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
        want = _SHA256.get(dest.name)
        if want is None:
            raise ValueError(f"no pinned SHA-256 for {dest.name}; refusing to use an unverified download")
        if digest.hexdigest() != want:
            raise ValueError(f"SHA-256 mismatch for {url}: got {digest.hexdigest()}, expected {want}")
        os.replace(tmp, dest)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def fetch(name: str, role: str = "model", cache_dir: str | os.PathLike | None = None) -> Path:
    """Return a local path for `role` of hub model `name`, downloading on first use."""
    spec = REGISTRY[name]
    url = spec.files[role]
    dest = cache_root(cache_dir) / name / url.rsplit("/", 1)[-1]
    if not dest.exists():
        _download(url, dest)
    return dest


def resolve(name_or_path: str, role: str = "model", cache_dir: str | os.PathLike | None = None) -> Path:
    """Hub name -> cached file; existing path -> itself."""
    p = expand_path(name_or_path)
    if p.exists():
        return p
    if name_or_path in REGISTRY:
        return fetch(name_or_path, role, cache_dir)
    raise FileNotFoundError(f"'{name_or_path}' is neither a file nor a known model ({', '.join(sorted(REGISTRY))})")


def spec_format(name_or_path: str) -> str | None:
    spec = REGISTRY.get(name_or_path)
    return spec.fmt if spec else None
