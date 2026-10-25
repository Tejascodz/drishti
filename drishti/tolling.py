"""Toll transaction rules for barrier-free (MLFF / ORT) lanes, camera only.

Follows the confidence-reporting scheme of the Indian MLFF ANPR specification (IHMCL):

* every read carries a confidence level: Good, Average or Poor;
* only Good or Average reads may drive an automated transaction; Poor reads go to
  manual review;
* the camera's registration number (VRN) drives a transaction on its own only if its
  confidence is above 95%.

Confidence thresholds are only meaningful once confidence is calibrated on labelled
frames from the lane camera (`drishti eval` + `drishti calibrate`).
"""

from __future__ import annotations

GOOD, AVERAGE, POOR = "Good", "Average", "Poor"


def confidence_level(conf: float | None, levels: dict | None = None) -> str:
    lv = levels or {}
    c = 0.0 if conf is None else float(conf)
    if c >= float(lv.get("good", 0.95)):
        return GOOD
    if c >= float(lv.get("average", 0.80)):
        return AVERAGE
    return POOR


def _fields(item):
    """Plate text / confidence / width / duplicate link from a PlateEvent or PlateResult."""
    if hasattr(item, "read"):  # PlateResult
        r = item.read
        return r.text, r.confidence, int(round(item.bbox.w)), None
    return item.text, item.confidence, getattr(item, "plate_width_px", None), getattr(item, "duplicate_of", None)


def toll_decision(engine, item) -> dict:
    tc = engine.cfg.get("tolling") or {}
    text, conf, width, dup = _fields(item)
    level = confidence_level(conf if text else 0.0, tc.get("levels") or {})
    out = {"action": "manual_review", "vrn": None, "plate": text,
           "confidence": None if conf is None else round(float(conf), 4), "level": level, "reasons": []}
    if dup:
        out.update(action="duplicate", reasons=[f"same vehicle already recorded as {dup} by an overlapping camera"])
        return out
    min_w = tc.get("min_plate_width_px")
    if width is not None and min_w and width < float(min_w):
        out["reasons"].append(f"plate {width} px wide, below the {min_w} px guideline: check camera zoom or position")
    auto_min = float(tc.get("auto_min", tc.get("anpr_only_min", 0.95)))
    if text and conf is not None and conf > auto_min and level in (GOOD, AVERAGE):
        out.update(action="auto", vrn=text)
    elif not text:
        out["reasons"].insert(0, "no readable plate")
    else:
        out["reasons"].insert(0, f"plate confidence {conf:.2f} is not above {auto_min:.2f}")
    return out
