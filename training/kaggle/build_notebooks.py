"""Generates the Kaggle notebooks in this folder.

    python training/kaggle/build_notebooks.py

Both notebooks also run outside Kaggle: set KAGGLE_INPUT / KAGGLE_WORK to local
folders, and SMOKE=1 for a tiny end-to-end run on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def nb(cells: list[tuple[str, str]]) -> dict:
    out = []
    for kind, src in cells:
        src = src.strip("\n") + "\n"
        lines = src.splitlines(keepends=True)
        if kind == "md":
            out.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            out.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines})
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_EVAL = r'''
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0

def match(preds, gts, thr=0.5):
    """Greedy one-to-one matching by score. Returns (tp, fp, fn)."""
    used, tp = set(), 0
    for p in sorted(preds, key=lambda p: -p[4]):
        best, bj = thr, None
        for j, g in enumerate(gts):
            if j not in used and iou(p, g) >= best:
                best, bj = iou(p, g), j
        if bj is not None:
            used.add(bj); tp += 1
    return tp, len(preds) - tp, len(gts) - tp

def evaluate_detector(predict, img_dir, lbl_dir, limit=None):
    """predict(bgr) -> [(x1, y1, x2, y2, score)]. Recall / precision / F1 at IoU 0.5 plus small-plate recall."""
    import cv2
    tp = fp = fn = 0; small_tp = small_n = 0
    files = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in IMG_EXT)[:limit]
    for p in files:
        img = cv2.imread(str(p))
        if img is None: continue
        H, W = img.shape[:2]
        gts = []
        for c, cx, cy, w, h in read_boxes(Path(lbl_dir) / (p.stem + ".txt")):
            gts.append(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
        preds = predict(img)
        a, b, c = match(preds, gts)
        tp += a; fp += b; fn += c
        for g in gts:  # recall on plates under 24 px high: the CCTV-distance case
            if g[3] - g[1] < 24:
                small_n += 1
                small_tp += any(iou(pp, g) >= 0.5 for pp in preds)
    r = tp / max(tp + fn, 1); pr = tp / max(tp + fp, 1)
    return {"images": len(files), "recall": round(r, 4), "precision": round(pr, 4),
            "f1": round(2 * r * pr / max(r + pr, 1e-9), 4),
            "small_plate_recall": round(small_tp / small_n, 4) if small_n else None, "small_plates": small_n}
'''

READ_BOXES = r'''
def read_boxes(lbl_path):
    """Clean YOLO boxes: class forced to 0, clipped to the image, degenerate boxes dropped."""
    rows = []
    try:
        text = Path(lbl_path).read_text()
    except (FileNotFoundError, UnicodeDecodeError):
        return rows
    for line in text.splitlines():
        v = line.split()
        if len(v) < 5:
            continue
        try:
            cx, cy, w, h = (float(x) for x in v[1:5])
        except ValueError:
            continue
        x1, y1, x2, y2 = max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(1.0, cx + w / 2), min(1.0, cy + h / 2)
        if x2 - x1 < 0.002 or y2 - y1 < 0.002:
            continue
        rows.append((0, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return rows
'''

DETECTOR = [
    ("md", r'''
# Drishti · plate detector training (YOLO26) on Kaggle

Trains a YOLO26 plate detector for Indian roads and exports an ONNX model that Drishti loads directly.

**Before running**
1. *Settings → Accelerator:* **GPU T4 x2** (T4 is preferred over P100; recent PyTorch builds drop P100 support).
2. *Settings → Internet:* **On** (needs a phone-verified Kaggle account).
3. *Add Input → Datasets:*
   - **`kedarsai/indian-license-plates-with-labels`** (CC0, ~2k Indian plates) - required
   - your upload of **`lpr_v13_unique_yolo.zip`** (Roboflow v13, ~10k unique photos, CC BY 4.0) - optional, adds volume

**What it does:** cleans the labels, removes near-duplicate photos across the train/val split, oversamples the
Indian images, trains, compares the new model with Drishti's current default detector on the same Indian
validation images, exports ONNX, and zips everything to `/kaggle/working/drishti_detector.zip`.

Ultralytics (YOLO26) is AGPL-3.0. That is fine for personal and research use; a closed commercial product needs
their enterprise licence.
'''),
    ("code", r'''
import os
SMOKE = os.environ.get("SMOKE") == "1"          # tiny CPU run for testing the notebook itself

MODEL = "yolo26n.pt" if SMOKE else "yolo26s.pt"  # n = fastest, s = small (recommended), m = more accurate
IMGSZ = 320 if SMOKE else 960                    # plates are small: resolution matters more than model size
EPOCHS = 1 if SMOKE else 80
BATCH = 4 if SMOKE else 16
INDIAN_OVERSAMPLE = 3                            # repeat Indian training images so they are not drowned out
VAL_FRACTION = 0.15
DUP_HAMMING = 6                                  # dHash distance below which two photos count as duplicates
SEED = 0
INPUT = os.environ.get("KAGGLE_INPUT", "/kaggle/input")
WORK = os.environ.get("KAGGLE_WORK", "/kaggle/working")
'''),
    ("code", r'''
%pip install -q "ultralytics>=8.4.0" open-image-models onnxruntime
'''),
    ("code", r'''
import json, random, shutil
from pathlib import Path
import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def find_yolo_sets(root):
    """Every folder named `images` that has a sibling `labels` folder."""
    found = []
    for img_dir in sorted(Path(root).rglob("images")):
        lbl_dir = img_dir.parent / "labels"
        if img_dir.is_dir() and lbl_dir.is_dir():
            n = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
            if n:
                found.append({"images": img_dir, "labels": lbl_dir, "n": n})
    return found

sets = find_yolo_sets(INPUT)
for s in sets:
    s["split"] = next((k for k in ("train", "valid", "val", "test") if k in s["images"].parts), None)
    s["indian"] = "indian" in str(s["images"]).lower()
    print(f'{s["n"]:6d}  indian={s["indian"]!s:5}  split={s["split"]}  {s["images"]}')
indian_sets = [s for s in sets if s["indian"]]
extra_train = [s for s in sets if not s["indian"] and s["split"] in (None, "train")]
extra_test = [s for s in sets if not s["indian"] and s["split"] == "test"]
assert indian_sets, "Add the dataset kedarsai/indian-license-plates-with-labels as an input"
'''),
    ("code", READ_BOXES + r'''

def dhash(path, size=8):
    im = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).ravel()

items, missing, empty, broken = [], 0, 0, 0
for s in indian_sets:
    for p in sorted(s["images"].iterdir()):
        if p.suffix.lower() not in IMG_EXT:
            continue
        lp = s["labels"] / (p.stem + ".txt")
        if not lp.exists():
            missing += 1; continue           # unlabelled, not "no plate": leave out
        boxes = read_boxes(lp)
        if not boxes:
            empty += 1; continue
        try:
            items.append({"img": p, "boxes": boxes, "hash": dhash(p)})
        except Exception:
            broken += 1
print(f"Indian: {len(items)} usable images ({missing} without labels, {empty} empty, {broken} unreadable)")

# group near-duplicate photos so copies never straddle train and val
H = np.array([it["hash"] for it in items])
parent = list(range(len(items)))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i
for i in range(len(items) - 1):
    d = (H[i + 1:] != H[i]).sum(axis=1)
    for j in np.where(d <= DUP_HAMMING)[0] + i + 1:
        parent[find(int(j))] = find(i)
groups = {}
for i in range(len(items)):
    groups.setdefault(find(i), []).append(i)
print(f"{len(groups)} distinct photos after merging {len(items) - len(groups)} near-duplicates")

rng = random.Random(SEED)
keys = sorted(groups); rng.shuffle(keys)
n_val = int(round(VAL_FRACTION * len(items))); val_idx = set()
for k in keys:
    if len(val_idx) >= n_val:
        break
    val_idx.update(groups[k])

hs = np.array([b[4] for it in items for b in it["boxes"]])
area = np.array([max(b[3] * b[4] for b in it["boxes"]) for it in items])
print("plate height as fraction of image  p10/p50/p90:", np.round(np.percentile(hs, [10, 50, 90]), 3))
print(f"tight plate crops (box > 60% of image): {(area > 0.6).mean():.1%}")
'''),
    ("code", r'''
DATA = Path(WORK) / "data"
if DATA.exists():
    shutil.rmtree(DATA)
TIGHT_MAX = 0.6   # plate fills >60% of the image: a close-up crop, OCR material rather than detector material

def put(split, name, img, boxes):
    (DATA / split / "images").mkdir(parents=True, exist_ok=True)
    (DATA / split / "labels").mkdir(parents=True, exist_ok=True)
    dst = DATA / split / "images" / (name + img.suffix.lower())
    os.symlink(img.resolve(), dst)
    (DATA / split / "labels" / (name + ".txt")).write_text(
        "\n".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for _, cx, cy, w, h in boxes) + "\n")

n_tr = n_val = n_tight = 0
for i, it in enumerate(items):
    if max(b[3] * b[4] for b in it["boxes"]) > TIGHT_MAX:
        n_tight += 1
        continue
    if i in val_idx:
        put("val", f"in_{i:05d}", it["img"], it["boxes"]); n_val += 1
    else:
        for r in range(INDIAN_OVERSAMPLE):
            put("train", f"in_{i:05d}_r{r}", it["img"], it["boxes"]); n_tr += 1

def add_sets(sets_, split, prefix):
    n = 0
    for k, s in enumerate(sets_):
        for j, p in enumerate(sorted(s["images"].iterdir())):
            if p.suffix.lower() in IMG_EXT:
                b = read_boxes(s["labels"] / (p.stem + ".txt"))
                if b:
                    put(split, f"{prefix}{k}_{j:06d}", p, b); n += 1
    return n

n_extra = add_sets(extra_train, "train", "x")
extra_valid = [s for s in sets if not s["indian"] and s["split"] in ("valid", "val")]
n_xval = add_sets(extra_valid, "val", "v")
add_sets(extra_test, "extra_test", "t")

(DATA / "data.yaml").write_text(f"path: {DATA}\ntrain: train/images\nval: val/images\nnames:\n  0: plate\n")
print(f"left out {n_tight} Indian close-up crops (plate fills the image)")
print(f"train: {n_tr} Indian (x{INDIAN_OVERSAMPLE}) + {n_extra} extra   val: {n_val} Indian + {n_xval} extra")
'''),
    ("md", r'''
## Baseline: Drishti's current default detector
`yolo-v9-s-608-license-plate-end2end` (open-image-models, MIT), scored on the same Indian validation images.
'''),
    ("code", COMMON_EVAL + r'''

from open_image_models import LicensePlateDetector
base_det = LicensePlateDetector(detection_model="yolo-v9-s-608-license-plate-end2end", conf_thresh=0.3)
def predict_baseline(img):
    return [(d.bounding_box.x1, d.bounding_box.y1, d.bounding_box.x2, d.bounding_box.y2, d.confidence) for d in base_det.predict(img)]

baseline = evaluate_detector(predict_baseline, DATA / "val" / "images", DATA / "val" / "labels")
print("baseline on Indian val:", baseline)
'''),
    ("code", r'''
import torch
from ultralytics import YOLO

model = YOLO(MODEL)
model.train(
    data=str(DATA / "data.yaml"), imgsz=IMGSZ, epochs=EPOCHS, batch=BATCH,
    device=0 if torch.cuda.is_available() else "cpu", workers=4, seed=SEED,
    project=f"{WORK}/runs", name="plates", exist_ok=True,
    cos_lr=True, patience=25, plots=not SMOKE,
    # plates are rigid and roughly frontal; keep geometry modest, vary scale and light a lot
    degrees=5.0, shear=2.0, perspective=0.0005, translate=0.1, scale=0.6, fliplr=0.3,
    mosaic=1.0, close_mosaic=10, hsv_h=0.02, hsv_s=0.6, hsv_v=0.5,
)
best = Path(model.trainer.best)
print("best weights:", best)
'''),
    ("code", r'''
new = YOLO(str(best))
def predict_new(img):
    r = new.predict(img, imgsz=IMGSZ, conf=0.3, verbose=False)[0]
    return [(*b.xyxy[0].tolist(), float(b.conf[0])) for b in r.boxes]

finetuned = evaluate_detector(predict_new, DATA / "val" / "images", DATA / "val" / "labels")
report = {"indian_val": {"baseline_yolov9s": baseline, "new_" + Path(MODEL).stem: finetuned}}
if (DATA / "extra_test").exists():
    report["extra_test"] = {
        "baseline_yolov9s": evaluate_detector(predict_baseline, DATA / "extra_test" / "images", DATA / "extra_test" / "labels", limit=500),
        "new_" + Path(MODEL).stem: evaluate_detector(predict_new, DATA / "extra_test" / "images", DATA / "extra_test" / "labels", limit=500),
    }
m = new.val(data=str(DATA / "data.yaml"), imgsz=IMGSZ, split="val", verbose=False)
report["ultralytics_val"] = {"mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4)}
print(json.dumps(report, indent=2))
'''),
    ("code", r'''
import onnxruntime as ort
onnx_path = Path(new.export(format="onnx", imgsz=IMGSZ, dynamic=False, simplify=True))
sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
print("ONNX input:", sess.get_inputs()[0].shape, " outputs:", [o.shape for o in sess.get_outputs()])

OUT = Path(WORK) / "drishti_detector"
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
name = f"plate_{Path(MODEL).stem}_in_{IMGSZ}"
shutil.copy(onnx_path, OUT / f"{name}.onnx")
shutil.copy(best, OUT / f"{name}.pt")
for f in ("results.csv", "results.png", "confusion_matrix.png", "BoxPR_curve.png"):
    if (best.parent.parent / f).exists():
        shutil.copy(best.parent.parent / f, OUT / f)
(OUT / "metrics.json").write_text(json.dumps(report, indent=2))
(OUT / "drishti.yaml").write_text(
    "# put the .onnx in models/ and run:  drishti --config drishti.yaml image photo.jpg\n"
    f"models:\n  plate_detector: models/{name}.onnx\n"
    "ocr:\n  # four crop views (one tight, three padded): best on real Indian photos with this detector\n"
    "  views: [[0.02, -0.06], [0.04, 0.10], [0.10, 0.18], [0.17, 0.27]]\n")
shutil.make_archive(str(Path(WORK) / "drishti_detector"), "zip", OUT)
print("download:", Path(WORK) / "drishti_detector.zip")
'''),
]

OCR = [
    ("md", r'''
# Drishti · Indian plate OCR fine-tune on Kaggle

Fine-tunes the global CCT plate OCR (fast-plate-ocr, MIT) on Indian plates and exports ONNX for Drishti.

**Before running**
1. *Settings → Accelerator:* **GPU T4** · *Internet:* **On**
2. *Add Input → Datasets:*
   - **`drishti-anpr`**: upload the Drishti project zip as a private dataset - required
     (synthetic Indian plate generator, format grammar, evaluation)
   - **`drishti-ocr-real`** (optional, strongly recommended): the output folder of
     `drishti label ... --out data/ocr_real` after review (`crops/` + the `annotations.csv` downloaded from
     `review.html`), zipped and uploaded as a private dataset.

**What it does:** renders synthetic Indian plates in parallel, splits the real crops by plate number so no
vehicle is in both train and validation, scores the current OCR, fine-tunes from the public global weights,
scores again on the same real validation plates, and zips the ONNX model to `/kaggle/working/drishti_ocr.zip`.

Without real crops the run still works, but validation then uses synthetic plates, which says little about
real roads.
'''),
    ("code", r'''
import os, sys
print("DRISHTI OCR v2  (40 epochs, real + 3000 hard synthetic validation plates)")
SMOKE = os.environ.get("SMOKE") == "1"

N_SYNTH = 800 if SMOKE else 120000   # synthetic Indian plates
N_SYNTH_VAL = 200 if SMOKE else 3000 # hard synthetic validation plates (all layouts, toll angle, night IR)
EPOCHS = 1 if SMOKE else 40          # ~9.5 min each on Kaggle: fits the 12 h limit with room to spare
BATCH = 32 if SMOKE else 256
LR = 5e-4
REAL_OVERSAMPLE = 6                  # each real crop counts this many times against the synthetic ones
VAL_FRACTION = 0.15
MODEL_CFG = "cct_s_v2"               # cct_xs_v2 = faster for edge devices
BACKEND = os.environ.get("KERAS_BACKEND", "tensorflow")   # Kaggle images ship TensorFlow with GPU
SEED = 0
INPUT = os.environ.get("KAGGLE_INPUT", "/kaggle/input")
WORK = os.environ.get("KAGGLE_WORK", "/kaggle/working")
SCRATCH = os.environ.get("KAGGLE_SCRATCH", "/tmp/drishti")  # big intermediate data stays out of the saved output
'''),
    ("code", r'''
import shutil, subprocess
from pathlib import Path

def find_repo(root):
    for pp in Path(root).rglob("pyproject.toml"):
        if 'name = "drishti-anpr"' in pp.read_text(errors="ignore"):
            return pp.parent
    raise SystemExit("Add the drishti-anpr dataset (the Drishti project upload) as an input")

src = find_repo(INPUT)
REPO = Path(WORK) / "drishti"
if src.resolve() != REPO.resolve():
    shutil.copytree(src, REPO, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.onnx", "*.keras", "__pycache__"))
os.environ["KERAS_BACKEND"] = BACKEND
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO), "fast-plate-ocr[train]>=1.1", "albumentations"], check=True)
subprocess.run(["bash", str(REPO / "training/synth/fetch_fonts.sh")], check=True)
sys.path.insert(0, str(REPO))
print("repo:", REPO, " keras backend:", BACKEND)
'''),
    ("code", r'''
import csv, os
render = REPO / "training/synth/render_plates.py"
K = max(1, min(os.cpu_count() or 1, 8))
SYN = Path(SCRATCH) / "synth"
procs = []
for k in range(K):
    n = N_SYNTH // K + (1 if k < N_SYNTH % K else 0)
    procs.append(subprocess.Popen([sys.executable, str(render), "--out", str(SYN / f"s{k}"), "--n", str(n), "--val", "0",
                                   "--seed", str(1000 + k)], stdout=subprocess.DEVNULL))
# separate seed: validation plates never appear in training
procs.append(subprocess.Popen([sys.executable, str(render), "--out", str(SYN / "val"), "--n", "0", "--val", str(N_SYNTH_VAL),
                               "--seed", "999"], stdout=subprocess.DEVNULL))
for p in procs:
    assert p.wait() == 0, "synthetic generation failed"
print(f"rendered {N_SYNTH} synthetic training plates and {N_SYNTH_VAL} validation plates with {K} workers")
'''),
    ("code", r'''
import random
OCR = Path(WORK) / "ocr"
OCR.mkdir(parents=True, exist_ok=True)

def rows_from(csv_path):
    base = Path(csv_path).resolve().parent
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [((base / r["image_path"]).resolve(), r["plate_text"].strip().upper()) for r in csv.DictReader(f) if r.get("plate_text")]

def write_csv(path, rows):
    # fast-plate-ocr joins paths to the CSV folder as text, so keep them relative to it
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["image_path", "plate_text"])
        for img, txt in rows:
            w.writerow([os.path.relpath(img, path.parent), txt])

synth = [r for k in range(K) for r in rows_from(SYN / f"s{k}/train/annotations.csv")]

real = []
for ann in Path(INPUT).rglob("annotations*.csv"):
    if src in ann.parents or REPO in ann.parents:
        continue  # the project's own files, not data
    rs = [r for r in rows_from(ann) if r[0].exists()]
    if rs:
        print(f"real crops: {len(rs)} from {ann}")
        real += rs
real = list({str(p): (p, t) for p, t in real}.values())  # annotations.csv and annotations_auto.csv overlap

synth_val = rows_from(SYN / "val/val/annotations.csv")
rng = random.Random(SEED)
real_val = []
if real:
    plates = sorted({t for _, t in real}); rng.shuffle(plates)
    val_plates = set(plates[: max(1, int(VAL_FRACTION * len(plates)))])
    real_val = [r for r in real if r[1] in val_plates]
    real_train = [r for r in real if r[1] not in val_plates]
    train = synth + real_train * REAL_OVERSAMPLE
    print(f"train: {len(synth)} synthetic + {len(real_train)} real (x{REAL_OVERSAMPLE})")
else:
    train = synth
    print("WARNING: no real crops found")
# Model selection uses real + hard synthetic plates together. Real crops labelled automatically are the
# easy ones (the current OCR already reads them), so on their own they sit at 100% and cannot pick a model.
val = real_val + synth_val
print(f"validation: {len(real_val)} real + {len(synth_val)} hard synthetic plates")
rng.shuffle(train)
write_csv(OCR / "train.csv", train)
write_csv(OCR / "val.csv", val)
write_csv(OCR / "val_real.csv", real_val)
write_csv(OCR / "val_synth.csv", synth_val)
'''),
    ("code", r'''
import numpy as np, cv2
from drishti.ocr import SlotOCR
from drishti.formats import FormatGrammar

def eval_ocr(model, config=None, csv_path=OCR / "val.csv", batch=128):
    ocr = SlotOCR(model, config)
    g = FormatGrammar.from_pack("india", ocr.alphabet, ocr.pad_char)
    rows = rows_from(csv_path)
    free = strict = 0; char_ok = char_n = 0
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        probs, _ = ocr([cv2.imread(str(p)) for p, _ in chunk])
        for (p, truth), pr in zip(chunk, probs):
            d = g.decode(np.log(pr + 1e-9), mode="auto")
            free += d.raw_text == truth; strict += d.text == truth
            char_ok += sum(a == b for a, b in zip(d.text, truth)); char_n += max(len(truth), len(d.text))
    n = max(len(rows), 1)
    return {"plates": len(rows), "exact_free": round(free / n, 4), "exact_with_grammar": round(strict / n, 4),
            "char_accuracy": round(char_ok / max(char_n, 1), 4)}

def eval_all(model, config=None):
    out = {"synthetic_hard": eval_ocr(model, config, OCR / "val_synth.csv")}
    if real_val:
        out["real"] = eval_ocr(model, config, OCR / "val_real.csv")
    return out

baseline = eval_all("cct-s-v2-global-model")
print("current global OCR:", baseline)
'''),
    ("code", r'''
from drishti import hub
weights = hub.fetch(f"{MODEL_CFG.replace('_', '-')}-global-model", "keras")
subprocess.run([sys.executable, str(REPO / "training/ocr/make_augmentation.py"), str(OCR / "augment.yaml")], check=True)
RUNS = Path(WORK) / "ocr_runs"
cmd = ["fast-plate-ocr", "train",
       "--model-config-file", str(REPO / f"training/ocr/{MODEL_CFG}.yaml"),
       "--plate-config-file", str(REPO / "training/ocr/india_plate_config.yaml"),
       "--annotations", str(OCR / "train.csv"), "--val-annotations", str(OCR / "val.csv"),
       "--augmentation-path", str(OCR / "augment.yaml"), "--weights-path", str(weights),
       "--lr", str(LR), "--warmup-fraction", "0.05", "--label-smoothing", "0.05",
       "--batch-size", str(BATCH), "--epochs", str(EPOCHS), "--workers", "4",
       "--early-stopping-patience", "10", "--early-stopping-metric", "val_plate_acc",
       "--output-dir", str(RUNS)]
print(" ".join(cmd))
subprocess.run(cmd, check=True, env={**os.environ, "KERAS_BACKEND": BACKEND})
run_dir = sorted(p for p in RUNS.iterdir() if (p / "best.keras").exists())[-1]
print("run:", run_dir)
'''),
    ("code", r'''
import json
EXP = Path(WORK) / "drishti_ocr"
if EXP.exists():
    shutil.rmtree(EXP)
EXP.mkdir(parents=True)
name = f"ocr_{MODEL_CFG}_in"
shutil.copy(run_dir / "best.keras", EXP / f"{name}.keras")  # kept even if the export below fails
shutil.copy(REPO / "training/ocr/india_plate_config.yaml", EXP / f"{name}_plate_config.yaml")
exp = subprocess.run(["fast-plate-ocr", "export", "--model", str(run_dir / "best.keras"), "--format", "onnx",
                      "--plate-config-file", str(REPO / "training/ocr/india_plate_config.yaml"), "--save-dir", str(EXP),
                      "--dynamic-batch"], env={**os.environ, "KERAS_BACKEND": BACKEND})
onnx_out = next(iter(sorted(EXP.glob("*.onnx"))), None)
if exp.returncode == 0 and onnx_out is not None:
    onnx_out.rename(EXP / f"{name}.onnx")
    finetuned = eval_all(str(EXP / f"{name}.onnx"), str(EXP / f"{name}_plate_config.yaml"))
else:
    finetuned = "ONNX export failed; the trained .keras model is in the zip and can be exported later"
    print("WARNING:", finetuned)
report = {"validation": "real" if real else "synthetic", "before": baseline, "after": finetuned}
print(json.dumps(report, indent=2))
(EXP / "metrics.json").write_text(json.dumps(report, indent=2))
for f in ("training_log.csv", "hyper_params.json"):
    if (run_dir / f).exists():
        shutil.copy(run_dir / f, EXP / f)
(EXP / "drishti.yaml").write_text(
    "# put both files in models/ and run:  drishti --config drishti.yaml image photo.jpg\n"
    f"models:\n  ocr: models/{name}.onnx\n  ocr_config: models/{name}_plate_config.yaml\n")
shutil.make_archive(str(Path(WORK) / "drishti_ocr"), "zip", EXP)
print("download:", Path(WORK) / "drishti_ocr.zip")
'''),
]


def main() -> None:
    for name, cells in (("drishti_detector_kaggle.ipynb", DETECTOR), ("drishti_ocr_kaggle.ipynb", OCR)):
        (HERE / name).write_text(json.dumps(nb(cells), indent=1), encoding="utf-8")
        print("wrote", HERE / name)


if __name__ == "__main__":
    main()
