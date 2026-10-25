"""Check a plate detector + OCR pair before using it, then test them together.

    python scripts/check_models.py
    python scripts/check_models.py --detector models/plate_yolo26s_in_960.onnx \
        --ocr models/ocr_in/best.onnx --ocr-config models/ocr_in/best_plate_config.yaml

Steps:
  1. each .onnx opens and its inputs/outputs look like what Drishti expects
  2. the OCR's output size matches its plate config (slots x alphabet)
  3. the OCR reads labelled plate crops; compared with the public global OCR it
     was fine-tuned from, which tells a trained model from an untrained or copied one
  4. the full pipeline (detector -> OCR -> format grammar) on labelled photos
Exit code is 1 if any check fails.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drishti import hub  # noqa: E402
from drishti.formats.grammar import FormatGrammar  # noqa: E402
from drishti.fusion import levenshtein  # noqa: E402

BASELINE = "cct-s-v2-global-model"
failures: list[str] = []
# CPU by default: CoreML (picked automatically on a Mac) can stall for minutes on the OCR
RUNTIME = {"providers": "cpu"}


def ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def bad(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    failures.append(msg)


def warn(msg: str) -> None:
    print(f"  [warn] {msg}")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def open_onnx(path: Path):
    import onnxruntime as ort

    if not path.exists():
        bad(f"{path} does not exist")
        return None
    size = path.stat().st_size
    if size < 100_000:
        bad(f"{path.name} is only {size} bytes: too small to hold model weights (a config or a failed conversion?)")
        return None
    try:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as e:  # noqa: BLE001
        bad(f"{path.name} is not a loadable ONNX model: {e}")
        return None
    ok(f"{path.name} loads ({size / 1e6:.1f} MB, sha {sha(path)})")
    for i in sess.get_inputs():
        print(f"         input  {i.name}: {i.shape} {i.type}")
    for o in sess.get_outputs():
        print(f"         output {o.name}: {o.shape} {o.type}")
    return sess


def check_detector(path: Path) -> bool:
    print(f"\n1. Plate detector: {path}")
    sess = open_onnx(path)
    if sess is None:
        return False
    shape = sess.get_inputs()[0].shape
    if len(shape) != 4:
        bad(f"detector input should be NCHW (4 dims), got {shape}")
        return False
    h, w = shape[2], shape[3]
    if isinstance(h, int) and isinstance(w, int):
        ok(f"input size {w}x{h}")
    dummy = np.zeros((1, 3, h if isinstance(h, int) else 640, w if isinstance(w, int) else 640), np.float32)
    try:
        sess.run(None, {sess.get_inputs()[0].name: dummy})
        ok("runs on a blank image")
    except Exception as e:  # noqa: BLE001
        bad(f"detector fails on a blank image: {e}")
        return False
    return True


def check_ocr_file(path: Path, cfg_path: Path) -> bool:
    print(f"\n2. OCR: {path}\n   config: {cfg_path}")
    if not cfg_path.exists():
        bad(f"{cfg_path} does not exist")
        return False
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    slots, alpha = int(cfg["max_plate_slots"]), len(cfg["alphabet"])
    ok(f"config: {slots} slots x {alpha} characters, input {cfg['img_width']}x{cfg['img_height']} {cfg.get('image_color_mode', 'rgb')}")
    sess = open_onnx(path)
    if sess is None:
        return False
    inp = sess.get_inputs()[0]
    channels = 3 if str(cfg.get("image_color_mode", "rgb")).lower() == "rgb" else 1
    expect_in = [cfg["img_height"], cfg["img_width"], channels]
    got_in = inp.shape[1:]
    if [d for d in got_in] != expect_in:
        bad(f"OCR input {inp.shape} does not match the config (N, {', '.join(map(str, expect_in))})")
        return False
    ok(f"input matches config: {inp.shape}")
    dummy = np.zeros((1, *expect_in), np.float32 if "float" in inp.type else np.uint8)
    out = sess.run(None, {inp.name: dummy})[0]
    if out.size != slots * alpha:
        bad(f"OCR outputs {out.size} values per plate but the config needs {slots}x{alpha}={slots * alpha}: "
            "the model and the config are from different trainings")
        return False
    ok(f"output size matches config ({slots}x{alpha})")
    try:
        base = hub.fetch(BASELINE)
        if sha(base) == sha(path):
            bad(f"{path.name} is byte-identical to the public global model {BASELINE} (not trained on Indian plates): "
                "no fine-tuned weights are in it")
            return False
    except Exception:  # noqa: BLE001
        pass
    return True


def read_crops(model: str, config: str | None, rows: list[tuple[Path, str]], label: str) -> dict:
    from drishti.ocr import SlotOCR

    ocr = SlotOCR(model, config, RUNTIME)
    print(f"   {label}: running on {ocr.session.get_providers()[0]}", flush=True)
    g = FormatGrammar.from_pack("india", ocr.alphabet, ocr.pad_char)
    exact = chars = 0
    examples, all_probs = [], []
    for i in range(0, len(rows), 64):
        chunk = rows[i : i + 64]
        imgs = [cv2.imread(str(p)) for p, _ in chunk]
        probs, _ = ocr(imgs)
        all_probs.append(probs)
        for (p, gt), pr in zip(chunk, probs):
            text = g.decode(np.log(pr + 1e-9), mode="auto").text
            exact += text == gt
            chars += max(0.0, 1 - levenshtein(gt, text) / max(len(gt), 1))
            if len(examples) < 8:
                examples.append((gt, text))
        print(f"\r   {label}: {min(i + 64, len(rows))}/{len(rows)} crops", end="", flush=True)
    print()
    n = len(rows)
    return {"exact": exact / n, "chars": chars / n, "examples": examples, "probs": np.concatenate(all_probs)}


def check_ocr_accuracy(path: Path, cfg_path: Path, crops_csv: Path, limit: int) -> None:
    print(f"\n3. OCR accuracy on labelled crops: {crops_csv}")
    if not crops_csv.exists():
        warn("no labelled crops CSV (make one with `drishti label`); skipped")
        return
    rows = []
    with open(crops_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            p = crops_csv.parent / r["image_path"]
            if r.get("plate_text") and p.exists():
                rows.append((p, r["plate_text"].strip().upper()))
    rows = rows[:limit]
    if not rows:
        warn("no readable crops found; skipped")
        return
    mine = read_crops(str(path), str(cfg_path), rows, "your OCR")
    print(f"   your OCR     : exact {mine['exact']:.1%}   characters {mine['chars']:.1%}   ({len(rows)} crops)")
    try:
        base = read_crops(BASELINE, None, rows, "global OCR")
        print(f"   global OCR   : exact {base['exact']:.1%}   characters {base['chars']:.1%}   (what you started from)")
    except Exception as e:  # noqa: BLE001
        base = None
        warn(f"could not load the global OCR for comparison ({e})")
    print("   samples (truth -> your OCR):")
    for gt, txt in mine["examples"]:
        print(f"     {gt:<12} -> {txt}")
    if mine["chars"] < 0.3:
        bad("your OCR reads almost nothing right: the weights are untrained/random, "
            "or the model and config do not belong together")
    elif base and base["probs"].shape == mine["probs"].shape and np.abs(base["probs"] - mine["probs"]).max() < 1e-3:
        bad("your OCR gives exactly the same outputs as the global OCR: it is the public model re-exported, "
            "not your fine-tuned one")
    elif base and mine["exact"] + 0.10 < base["exact"]:
        bad("your OCR is clearly worse than the model it was fine-tuned from: keep using the global OCR")
    else:
        ok("your OCR is a different, working model (these auto labels are where the global OCR already "
           "agreed with the VLM, so the global one scores ~100% here by construction)")
    print("   (these crops were also in training if you uploaded data/ocr_real, so treat the numbers as optimistic)")


def check_pipeline(det: Path, ocr: Path | None, cfg_path: Path | None, labels: Path) -> None:
    print(f"\n4. Full pipeline on {labels}")
    from drishti.config import load_config
    from drishti.evaluate import evaluate
    from drishti.pipeline import Drishti

    models = {"plate_detector": str(det), "vehicle_detector": None}
    if ocr is not None:
        models.update(ocr=str(ocr), ocr_config=str(cfg_path))
    else:
        models.update(ocr=BASELINE, ocr_config=None)
        print("   (using the global OCR because yours failed the checks above)")
    base_cfg = ROOT / "models" / "drishti.yaml"
    cfg = load_config([str(base_cfg)] if base_cfg.exists() else None, {"models": models, "runtime": RUNTIME})
    report = evaluate(Drishti(cfg), labels)
    for it in report["items"]:
        got = [p["text"] for p in it.get("preds", [])]
        print(f"     {Path(it['image']).name:<22} truth {'|'.join(it.get('gt', [])):<12} read {got}")
    s = report["summary"]
    print(f"   recognition rate {s['recognition_rate']}  char accuracy {s['char_accuracy']}  "
          f"latency p50 {s['latency_ms_p50']} ms")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", default="models/plate_yolo26s_in_960.onnx")
    ap.add_argument("--ocr", default="models/ocr_in/best.onnx")
    ap.add_argument("--ocr-config", default="models/ocr_in/best_plate_config.yaml")
    ap.add_argument("--crops", default="data/ocr_real/annotations_auto.csv", help="labelled crops CSV (image_path,plate_text)")
    ap.add_argument("--images", default="samples/labels.csv", help="labelled photos CSV for the full pipeline")
    ap.add_argument("--limit", type=int, default=500, help="max crops for step 3")
    ap.add_argument("--providers", default="cpu", help="cpu (default) | auto | coreml | cuda ...")
    a = ap.parse_args()
    RUNTIME["providers"] = a.providers
    rel = lambda p: Path(p) if Path(p).is_absolute() else ROOT / p  # noqa: E731

    det, ocr, cfg = rel(a.detector), rel(a.ocr), rel(a.ocr_config)
    det_ok = check_detector(det)
    ocr_ok = check_ocr_file(ocr, cfg)
    if ocr_ok:
        n = len(failures)
        check_ocr_accuracy(ocr, cfg, rel(a.crops), a.limit)
        ocr_ok = len(failures) == n
    if det_ok:
        check_pipeline(det, ocr if ocr_ok else None, cfg, rel(a.images))

    print()
    if failures:
        print(f"{len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
