"""Compare every detector / OCR combination, with and without the local VLM, on one labelled set.

    python scripts/compare_setups.py samples/labels.csv
    python scripts/compare_setups.py my_photos/labels.csv --vlm-model qwen3-vl:2b

Setups (a setup whose model file is missing is skipped):
  public detector + public OCR      the models `drishti models pull` downloads
  public detector + your OCR
  your detector   + public OCR
  your detector   + your OCR
  your detector   + your OCR + VLM  low-confidence reads checked by Ollama (skipped if Ollama is not running)

Labels CSV: image_path,plate_text (paths relative to the CSV; several plates separated by |).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drishti.config import load_config  # noqa: E402
from drishti.evaluate import evaluate  # noqa: E402
from drishti.ocr.vlm import from_config as vlm_from_config  # noqa: E402
from drishti.pipeline import Drishti  # noqa: E402

PUBLIC_DET = "yolo-v9-s-608-license-plate-end2end"
PUBLIC_OCR = "cct-s-v2-global-model"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", help="labels CSV, e.g. samples/labels.csv")
    ap.add_argument("--config", default="models/drishti.yaml", help="base config shared by every setup")
    ap.add_argument("--detector", default="models/plate_yolo26s_in_960.onnx", help="your plate detector")
    ap.add_argument("--ocr", default="models/ocr_in/best.onnx", help="your OCR")
    ap.add_argument("--ocr-config", default="models/ocr_in/best_plate_config.yaml")
    ap.add_argument("--vlm-model", default=None, help="Ollama model for the VLM setup (default: from config)")
    ap.add_argument("--providers", default="cpu")
    a = ap.parse_args()

    labels = Path(a.labels)
    with open(labels, newline="", encoding="utf-8") as fh:
        missing = [r["image_path"] for r in csv.DictReader(fh) if not (labels.parent / r["image_path"]).exists()]
    if missing:
        print(f"missing images listed in {labels}: {', '.join(missing)}")
        print("for samples/, run: bash scripts/fetch_samples.sh samples")
        return 1

    rel = lambda p: Path(p) if Path(p).is_absolute() else ROOT / p  # noqa: E731
    det, ocr, ocr_cfg, base = rel(a.detector), rel(a.ocr), rel(a.ocr_config), rel(a.config)
    dets = [("public", PUBLIC_DET)] + ([("your", str(det))] if det.exists() else [])
    ocrs = [("public", PUBLIC_OCR, None)] + ([("your", str(ocr), str(ocr_cfg))] if ocr.exists() else [])
    if not det.exists():
        print(f"note: {det} not found, skipping setups with your detector")
    if not ocr.exists():
        print(f"note: {ocr} not found, skipping setups with your OCR")

    setups = [(f"{dn} detector + {on} OCR", d, o, oc, False) for dn, d in dets for on, o, oc in ocrs]
    if det.exists() and ocr.exists():
        base_cfg = load_config([str(base)] if base.exists() else None)
        vcfg = dict(base_cfg.vlm)
        if a.vlm_model:
            vcfg["model"] = a.vlm_model
        info = vlm_from_config(vcfg, base_cfg.recognition.region).check()
        if info.get("reachable") and info.get("model_present"):
            setups.append((f"your detector + your OCR + VLM ({vcfg['model']})", str(det), str(ocr), str(ocr_cfg), True))
        else:
            print(f"note: VLM setup skipped ({info.get('hint') or info.get('error') or 'model not installed'})")

    rows, reads = [], {}
    for name, d, o, oc, vlm in setups:
        print(f"running: {name} ...", flush=True)
        over = {"runtime": {"providers": a.providers},
                "models": {"plate_detector": d, "ocr": o, "ocr_config": oc, "vehicle_detector": None},
                "vlm": {"enabled": vlm, **({"model": a.vlm_model} if a.vlm_model else {})}}
        cfg = load_config([str(base)] if base.exists() else None, over)
        rep = evaluate(Drishti(cfg), a.labels)
        rows.append((name, rep["summary"]))
        for it in rep["items"]:
            got = [p["text"] for p in it.get("preds", []) if p["text"]]
            reads.setdefault(Path(it["image"]).name, {"truth": "|".join(it.get("gt", []))})[name] = got

    def pct(v):
        return "-" if v is None else f"{100 * v:.0f}%"

    def num(v):
        return "-" if v is None else f"{v:.2f}"

    print("\n| setup | plates read right | precision | characters right | confidence when right | ms per photo |")
    print("|---|---|---|---|---|---|")
    for name, s in rows:
        print(f"| {name} | {pct(s['recognition_rate'])} | {pct(s['precision'])} | {pct(s['char_accuracy'])} | "
              f"{num(s['mean_conf_correct'])} | {s['latency_ms_p50']} |")

    print("\nReads per photo (a read matching the truth is marked ok):")
    for img, r in reads.items():
        truth = r.pop("truth")
        print(f"\n  {img}   truth: {truth or '(none)'}")
        gts = set(truth.split("|")) if truth else set()
        for name, got in r.items():
            marks = [f"{t} {'ok' if t in gts else 'x'}" for t in got] or ["nothing read"]
            print(f"    {name:<40} {', '.join(marks)}")
    print("\nplates read right = labelled plates read exactly; precision = share of emitted reads that are right")
    return 0


if __name__ == "__main__":
    sys.exit(main())
