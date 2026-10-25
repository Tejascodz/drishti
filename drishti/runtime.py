"""ONNX Runtime session factory with hardware-aware provider selection."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import onnxruntime as ort

from .config import expand_path

log = logging.getLogger(__name__)

_ALIASES = {
    "tensorrt": "TensorrtExecutionProvider",
    "trt": "TensorrtExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "directml": "DmlExecutionProvider",
    "dml": "DmlExecutionProvider",
    "cpu": "CPUExecutionProvider",
}
_AUTO_ORDER = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",
    "OpenVINOExecutionProvider",
    "DmlExecutionProvider",
]


def resolve_providers(pref: str | list[str] | None, allow_coreml: bool = True) -> list[str]:
    available = set(ort.get_available_providers())
    if pref in (None, "auto"):
        chosen = [p for p in _AUTO_ORDER if p in available]
    else:
        names = pref if isinstance(pref, list) else [s.strip() for s in str(pref).split(",")]
        chosen = []
        for n in names:
            full = _ALIASES.get(n.lower(), n)
            if full in available:
                chosen.append(full)
            else:
                log.warning("provider %s not available; skipping", full)
    if not allow_coreml:
        chosen = [p for p in chosen if p != "CoreMLExecutionProvider"]
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return chosen


def make_session(model_path: str | os.PathLike, runtime_cfg: dict | None = None, allow_coreml: bool = True) -> ort.InferenceSession:
    cfg = runtime_cfg or {}
    providers = resolve_providers(cfg.get("providers", "auto"), allow_coreml=allow_coreml)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    threads = int(cfg.get("intra_op_threads", 0) or 0)
    if threads > 0:
        opts.intra_op_num_threads = threads
    opts.log_severity_level = 3

    provider_args: list = []
    for p in providers:
        if p == "TensorrtExecutionProvider":
            cache = expand_path(cfg.get("trt_cache_dir", "~/.cache/drishti/trt"))
            cache.mkdir(parents=True, exist_ok=True)
            provider_args.append(
                (
                    p,
                    {
                        "trt_fp16_enable": bool(cfg.get("trt_fp16", True)),
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": str(cache),
                        "trt_timing_cache_enable": True,
                    },
                )
            )
        else:
            provider_args.append(p)
    sess = ort.InferenceSession(str(Path(model_path)), sess_options=opts, providers=provider_args)
    log.info("%s -> %s", Path(model_path).name, sess.get_providers())
    return sess


def _meta_imgsz(sess: ort.InferenceSession) -> tuple[int, int] | None:
    """Training image size recorded by Ultralytics in ONNX metadata, e.g. imgsz='[960, 960]'."""
    try:
        raw = sess.get_modelmeta().custom_metadata_map.get("imgsz")
        if raw:
            vals = [int(float(v)) for v in raw.strip("[]() ").split(",") if v.strip()]
            if len(vals) == 1:
                return vals[0], vals[0]
            if len(vals) >= 2:
                return vals[0], vals[1]
    except Exception:
        pass
    return None


def input_hw(sess: ort.InferenceSession, default: tuple[int, int] = (640, 640)) -> tuple[int, int]:
    """(height, width) of an NCHW / NHWC image input. Dynamic dims fall back to the
    size stored in the model's metadata, then to `default`."""
    meta = _meta_imgsz(sess)
    if meta:
        default = meta
    shape = sess.get_inputs()[0].shape
    dims = [d if isinstance(d, int) else None for d in shape]
    if len(dims) == 4:
        if dims[1] in (1, 3):  # NCHW
            h, w = dims[2], dims[3]
        else:  # NHWC
            h, w = dims[1], dims[2]
        return (h or default[0], w or default[1])
    return default


def has_dynamic_batch(sess: ort.InferenceSession) -> bool:
    return not isinstance(sess.get_inputs()[0].shape[0], int)
