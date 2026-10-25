"""Evaluation on labelled images and confidence calibration.

Labels CSV: `image_path,plate_text` (paths relative to the CSV; several plates in
one image separated by `|`; empty plate_text = image with no readable plate).

Metrics follow the ALPR literature: exact-match plate recall (the "recognition
rate"), precision, character accuracy, and the confidence gap (mean confidence
of correct reads minus mean confidence of incorrect reads) that the ICPR 2026
low-resolution plate competition used as its tie-breaker.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np

from .fusion import levenshtein
from .watchlist import normalize_plate


def _read_labels(path: str | Path) -> list[tuple[Path, list[str]]]:
    path = Path(path)
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            img = (path.parent / row["image_path"]).resolve()
            plates = [normalize_plate(p) for p in (row.get("plate_text") or "").split("|") if p.strip()]
            out.append((img, plates))
    return out


def evaluate(engine, labels_csv: str | Path, iou_free: bool = True) -> dict:
    items = []
    lat = []
    for img_path, gts in _read_labels(labels_csv):
        img = cv2.imread(str(img_path))
        if img is None:
            items.append({"image": str(img_path), "error": "unreadable"})
            continue
        t = time.perf_counter()
        res = engine.recognize(img)
        lat.append((time.perf_counter() - t) * 1000)
        preds = []
        for r in res:
            txt = r.read.text or ""
            best_gt = min(gts, key=lambda g: levenshtein(g, txt)) if gts else None
            preds.append({
                "text": r.read.text,
                "confidence": r.read.confidence,
                "correct": bool(txt and txt in gts),
                "edit_distance": levenshtein(best_gt, txt) if best_gt is not None else None,
                "gt_match": best_gt,
                "features": {
                    "prob": r.read.string_prob,
                    "agreement": r.read.agreement,
                    "n_obs": 1,
                    "quality": r.quality,
                    "penalty": r.read.format_penalty,
                },
            })
        items.append({"image": str(img_path), "gt": gts, "preds": preds, "latency_ms": lat[-1]})

    gt_total = sum(len(i.get("gt", [])) for i in items)
    hits = sum(sum(1 for g in i.get("gt", []) if any(p["text"] == g for p in i["preds"])) for i in items)
    all_preds = [p for i in items for p in i.get("preds", [])]
    emitted = [p for p in all_preds if p["text"]]
    correct = [p for p in emitted if p["correct"]]
    wrong = [p for p in emitted if not p["correct"]]
    char_acc = []
    for i in items:
        for g in i.get("gt", []):
            cands = [p["text"] for p in i["preds"] if p["text"]]
            d = min((levenshtein(g, c) for c in cands), default=len(g))
            char_acc.append(max(0.0, 1 - d / max(len(g), 1)))
    summary = {
        "images": len(items),
        "gt_plates": gt_total,
        "recognition_rate": round(hits / gt_total, 4) if gt_total else None,
        "precision": round(len(correct) / len(emitted), 4) if emitted else None,
        "char_accuracy": round(float(np.mean(char_acc)), 4) if char_acc else None,
        "abstained": sum(1 for p in all_preds if not p["text"]),
        "mean_conf_correct": round(float(np.mean([p["confidence"] for p in correct])), 4) if correct else None,
        "mean_conf_wrong": round(float(np.mean([p["confidence"] for p in wrong])), 4) if wrong else None,
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 1) if lat else None,
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 1) if lat else None,
    }
    if correct and wrong:
        summary["confidence_gap"] = round(summary["mean_conf_correct"] - summary["mean_conf_wrong"], 4)
    return {"summary": summary, "items": items}


def _features(f: dict) -> list[float]:
    p = min(max(f["prob"], 1e-6), 1 - 1e-6)
    return [math.log(p / (1 - p)), f["agreement"], math.log(max(f["n_obs"], 1)), f["quality"], f["penalty"]]


def fit_calibration(items: list[dict], l2: float = 1e-2, iters: int = 3000, lr: float = 0.1) -> dict:
    """Logistic regression (numpy, no extra deps) mapping read features to P(correct)."""
    X, y = [], []
    for it in items:
        for p in it.get("preds", []):
            if p.get("text") and p.get("features"):
                X.append(_features(p["features"]))
                y.append(1.0 if p["correct"] else 0.0)
    if len(X) < 10 or len(set(y)) < 2:
        raise ValueError("need at least 10 reads with both correct and incorrect examples to calibrate")
    X = np.array(X)
    y = np.array(y)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w = np.zeros(Z.shape[1])
    b = 0.0
    for _ in range(iters):
        z = Z @ w + b
        p = 1 / (1 + np.exp(-z))
        g = p - y
        w -= lr * (Z.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    # fold standardisation back into raw-feature weights
    wr = w / sd
    br = b - float((w * mu / sd).sum())
    names = ["w_logit", "w_agree", "w_logn", "w_quality", "w_penalty"]
    out = {n: float(v) for n, v in zip(names, wr)}
    out["bias"] = float(br)
    out["n_samples"] = int(len(y))
    out["positive_rate"] = float(y.mean())
    return out
