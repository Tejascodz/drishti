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

Fine-tunes the public global CCT plate OCR (fast-plate-ocr, MIT) on Indian plates, including green EV plates,
dotted separators, emblems and thin aftermarket strips, and exports ONNX for Drishti.

**Before running**
1. *Settings → Accelerator:* **GPU T4** · *Internet:* **On**
2. *Add Input → Datasets:* your private dataset with
   - the Drishti project folder (required: synthetic plate generator, format grammar, evaluation), and
   - your reviewed real plate crops (`crops/` + `annotations.csv` from `drishti label`), strongly recommended.
3. *Save Version → Save & Run All (Commit)*. Nothing needs to stay open.

**What it does:** checks everything it needs in the first minutes and stops with a plain message if something
is missing, renders synthetic Indian plates, splits the real crops by plate number so no vehicle is in both
train and validation, scores the public OCR, fine-tunes from its weights, scores again on the same plates
(real, hard synthetic and green EV) and zips the result to `/kaggle/working/drishti_ocr.zip`.

Training stops on its own before Kaggle's 12 hour limit and keeps the best checkpoint, so a long run never
loses its result.
'''),
    ("code", r'''
import os, sys, time
T0 = time.time()
print("DRISHTI OCR v3  (green EV plates, fail-fast checks, time guard)")
SMOKE = os.environ.get("SMOKE") == "1"      # tiny end-to-end check of the notebook itself

N_SYNTH = 300 if SMOKE else 120000          # synthetic Indian plates for training
N_SYNTH_VAL = 60 if SMOKE else 3000         # hard synthetic validation plates (all layouts, toll angle, night IR)
N_GREEN_VAL = 40 if SMOKE else 1000         # green EV validation plates: shows whether the new styles are learned
GREEN_IN_SELECTION = 20 if SMOKE else 500   # how many of those also steer which epoch is kept
EPOCHS = 1 if SMOKE else 40
BATCH = 16 if SMOKE else 256
LR = 5e-4
PATIENCE = 10                               # stop early after this many epochs without improvement
REAL_OVERSAMPLE = 6                         # each real crop counts this many times against the synthetic ones
VAL_FRACTION = 0.15
MODEL_CFG = "cct_s_v2"                      # cct_xs_v2 = faster for edge devices
BACKEND = os.environ.get("KERAS_BACKEND", "tensorflow")   # Kaggle images ship TensorFlow with GPU
SEED = 0
TIME_LIMIT_H = 11.3                         # Kaggle stops a run at 12 h; training ends before this...
EXPORT_RESERVE_MIN = 25                     # ...leaving this long for export, evaluation and the zip
INPUT = os.environ.get("KAGGLE_INPUT", "/kaggle/input")
WORK = os.environ.get("KAGGLE_WORK", "/kaggle/working")
SCRATCH = os.environ.get("KAGGLE_SCRATCH", "/tmp/drishti")  # big intermediate data stays out of the saved output

def elapsed():
    return f"{(time.time() - T0) / 60:.0f} min"
'''),
    ("code", r'''
# Pre-flight: everything that could fail later is checked here, in the first few minutes.
import shutil, subprocess, urllib.request
from pathlib import Path

class Stop(RuntimeError):
    """A problem with the setup, explained in plain words."""

def run(cmd, what, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print((r.stdout or "")[-3000:], (r.stderr or "")[-3000:], sep="\n")
        raise Stop(f"{what} failed (exit {r.returncode}); the output above says why")
    return r.stdout

def find_repo(root):
    found = [pp.parent for pp in Path(root).rglob("pyproject.toml") if 'name = "drishti-anpr"' in pp.read_text(errors="ignore")]
    def up_to_date(d):
        r = d / "training/synth/render_plates.py"
        return r.exists() and "--schemes" in r.read_text(errors="ignore")
    found.sort(key=lambda d: (up_to_date(d), (d / "training/synth/render_plates.py").stat().st_mtime), reverse=True)
    for d in found:
        print("project found:", d, "" if up_to_date(d) else "(old version)")
    if found and up_to_date(found[0]):
        return found[0]
    # the uploaded copy is older than this notebook: take the current code from GitHub instead
    fresh = Path(SCRATCH) / "drishti_src"
    shutil.rmtree(fresh, ignore_errors=True)
    r = subprocess.run(["git", "clone", "--depth", "1", "https://github.com/Tejascodz/drishti", str(fresh)],
                       capture_output=True, text=True)
    if r.returncode == 0 and up_to_date(fresh):
        print("using the current project from GitHub:", fresh)
        return fresh
    why = "the GitHub copy is old too" if r.returncode == 0 else f"GitHub could not be reached: {r.stderr.strip()[-300:]}"
    raise Stop(f"the Drishti project in your dataset is an old version and {why}. Upload the current project")

try:
    urllib.request.urlopen("https://pypi.org/simple/fast-plate-ocr/", timeout=20)
except Exception as e:
    raise Stop(f"no internet ({e}): Settings -> Internet -> On")

src = find_repo(INPUT)

gpus = shutil.which("nvidia-smi") and subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
print("GPU:", gpus or "none")
if not gpus and not SMOKE:
    raise Stop("no GPU: Settings -> Accelerator -> GPU T4, then run again (on CPU this would take days)")

REPO = Path(WORK) / "drishti"
if src.resolve() != REPO.resolve():
    if REPO.exists():
        shutil.rmtree(REPO)  # never mix files from an older copy
    shutil.copytree(src, REPO, ignore=shutil.ignore_patterns("*.onnx", "*.keras", "__pycache__", ".git"))
os.environ["KERAS_BACKEND"] = BACKEND

pkgs = ["fast-plate-ocr[train]==1.1.0", "albumentations>=2.0.5,<3"]
try:
    run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO), *pkgs], "pip install")
except Stop:
    # the [train] extra pulls exporters for other platforms; training and ONNX export need only these
    print("full install failed, installing the parts training needs")
    run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO), "fast-plate-ocr==1.1.0", "albumentations>=2.0.5,<3",
         "keras>=3.10,<4", "click", "pandas", "pydantic>=2.5.2,<3", "scikit-learn", "matplotlib", "tensorboard",
         "tf2onnx", "onnxslim>=0.1.56"], "pip install (minimal)")
sys.path.insert(0, str(REPO))

# the trainer must see the GPU, or 40 epochs would run on the CPU and never finish
seen = run([sys.executable, "-c", "import keras, tensorflow as tf; print(keras.__version__, tf.__version__, "
            "len(tf.config.list_physical_devices('GPU')))"], "TensorFlow check", env={**os.environ, "KERAS_BACKEND": BACKEND}).split()
print(f"keras {seen[-3]}, tensorflow {seen[-2]}, GPUs visible to training: {seen[-1]}")
if BACKEND == "tensorflow" and seen[-1] == "0" and not SMOKE:
    raise Stop("TensorFlow cannot see the GPU after installing packages; restart the session and run again")

r = subprocess.run(["bash", str(REPO / "training/synth/fetch_fonts.sh")], capture_output=True, text=True)
fonts = sorted((REPO / "training/synth/fonts").glob("*.ttf"))
print(f"plate fonts: {len(fonts)}", "" if r.returncode == 0 else "(some downloads failed; using what is there)")
if not fonts and not any(Path(f).exists() for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",)):
    raise Stop("no plate fonts: training/synth/fetch_fonts.sh could not download any")

from drishti import hub
WEIGHTS = hub.fetch(f"{MODEL_CFG.replace('_', '-')}-global-model", "keras")   # starting point
BASE_OCR = "cct-s-v2-global-model"
hub.fetch(BASE_OCR, "model")                                                   # the model to beat

real_csvs = [a for a in Path(INPUT).rglob("annotations*.csv") if src not in a.parents]
print("real label files:", [str(a) for a in real_csvs] or "NONE (training on synthetic plates only)")
print(f"pre-flight passed in {elapsed()}")
'''),
    ("code", r'''
import csv
render = REPO / "training/synth/render_plates.py"
K = max(1, min(os.cpu_count() or 1, 8))
SYN = Path(SCRATCH) / "synth"
if SYN.exists():
    shutil.rmtree(SYN)
SYN.mkdir(parents=True)
jobs = []
def start(out, args):
    log = open(SYN / f"{out}.log", "w")
    jobs.append((out, log, subprocess.Popen([sys.executable, str(render), "--out", str(SYN / out), *map(str, args)],
                                            stdout=log, stderr=subprocess.STDOUT)))
for k in range(K):
    start(f"s{k}", ["--n", N_SYNTH // K + (1 if k < N_SYNTH % K else 0), "--val", 0, "--seed", 1000 + k])
# separate seeds: validation plates never appear in training
start("val", ["--n", 0, "--val", N_SYNTH_VAL, "--seed", 999])
# green EV plates in daylight colours only (no infrared), so this score is about the green style
start("green", ["--n", 0, "--val", N_GREEN_VAL, "--seed", 998, "--ir-frac", 0, "--schemes", "ev_private,ev_commercial",
                "--preview", 16])
for out, log, p in jobs:
    code = p.wait(); log.close()
    if code != 0:
        print((SYN / f"{out}.log").read_text()[-3000:])
        raise Stop(f"synthetic plate generation ({out}) failed; the log above says why")
print(f"rendered {N_SYNTH} training, {N_SYNTH_VAL} hard and {N_GREEN_VAL} green validation plates with {K} workers in {elapsed()}")
'''),
    ("code", r'''
import random, yaml, cv2
OCR = Path(WORK) / "ocr"
OCR.mkdir(parents=True, exist_ok=True)
PLATE_CFG = yaml.safe_load((REPO / "training/ocr/india_plate_config.yaml").read_text())
ALPHABET = set(PLATE_CFG["alphabet"]) - {PLATE_CFG["pad_char"]}

def rows_from(csv_path):
    base = Path(csv_path).resolve().parent
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [((base / r["image_path"]).resolve(), r["plate_text"].strip().upper().replace(" ", ""))
                for r in csv.DictReader(f) if (r.get("plate_text") or "").strip()]

def clean(rows, what, check_images=False):
    # a label the model cannot represent, or an image that does not open, would stop training hours in
    ok, bad = [], []
    for p, t in rows:
        good = 0 < len(t) <= PLATE_CFG["max_plate_slots"] and set(t) <= ALPHABET and p.exists()
        if good and check_images:
            good = cv2.imread(str(p)) is not None
        (ok if good else bad).append((p, t))
    if bad:
        print(f"{what}: skipped {len(bad)} unusable rows, e.g. {[(p.name, t) for p, t in bad[:3]]}")
    return ok

def write_csv(path, rows):
    # fast-plate-ocr joins paths to the CSV folder as text, so keep them relative to it
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["image_path", "plate_text"])
        for img, txt in rows:
            w.writerow([os.path.relpath(img, path.parent), txt])

synth = clean([r for k in range(K) for r in rows_from(SYN / f"s{k}/train/annotations.csv")], "synthetic")
synth_val = clean(rows_from(SYN / "val/val/annotations.csv"), "synthetic validation")
green_val = clean(rows_from(SYN / "green/val/annotations.csv"), "green validation")

real = []
for ann in real_csvs:
    rs = clean(rows_from(ann), f"real {ann.name}", check_images=True)
    if rs:
        print(f"real crops: {len(rs)} from {ann}")
        real += rs
real = list({str(p): (p, t) for p, t in real}.values())  # annotations.csv and annotations_auto.csv overlap

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
    print("WARNING: no real crops found, validation uses synthetic plates only")
# The kept epoch is picked on real + hard synthetic + green plates together. Real crops labelled automatically
# are the easy ones (the current OCR already reads them), so on their own they cannot pick a model.
val = real_val + synth_val + green_val[:GREEN_IN_SELECTION]
print(f"validation: {len(real_val)} real + {len(synth_val)} hard synthetic + {min(len(green_val), GREEN_IN_SELECTION)} green plates")
if len(train) < BATCH or not val:
    raise Stop(f"too little data to train ({len(train)} training rows, {len(val)} validation rows)")
rng.shuffle(train)
write_csv(OCR / "train.csv", train)
write_csv(OCR / "val.csv", val)
write_csv(OCR / "val_real.csv", real_val)
write_csv(OCR / "val_synth.csv", synth_val)
write_csv(OCR / "val_green.csv", green_val)
'''),
    ("code", r'''
import json
import numpy as np
from drishti.ocr import SlotOCR
from drishti.formats import FormatGrammar

def eval_ocr(model, config, csv_path, batch=128):
    ocr = SlotOCR(model, config)
    g = FormatGrammar.from_pack("india", ocr.alphabet, ocr.pad_char)
    rows = rows_from(csv_path)
    free = strict = 0; char_ok = char_n = 0
    for i in range(0, len(rows), batch):
        chunk = [(cv2.imread(str(p)), t) for p, t in rows[i:i + batch]]
        chunk = [(im, t) for im, t in chunk if im is not None]
        if not chunk:
            continue
        probs, _ = ocr([im for im, _ in chunk])
        for (_, truth), pr in zip(chunk, probs):
            d = g.decode(np.log(pr + 1e-9), mode="auto")
            free += d.raw_text == truth; strict += d.text == truth
            char_ok += sum(a == b for a, b in zip(d.text, truth)); char_n += max(len(truth), len(d.text))
    n = max(len(rows), 1)
    return {"plates": len(rows), "exact_free": round(free / n, 4), "exact_with_grammar": round(strict / n, 4),
            "char_accuracy": round(char_ok / max(char_n, 1), 4)}

SETS = {"real": OCR / "val_real.csv", "synthetic_hard": OCR / "val_synth.csv", "green_ev": OCR / "val_green.csv"}

def eval_all(model, config=None):
    out = {}
    for k, path in SETS.items():
        if k == "real" and not real_val:
            continue
        try:
            out[k] = eval_ocr(model, config, path)
        except Exception as e:  # a scoring problem must never cost the trained model
            out[k] = f"evaluation failed: {e!r}"
    return out

baseline = eval_all(BASE_OCR)
print("public global OCR:", json.dumps(baseline, indent=1))
'''),
    ("code", r'''
import signal, threading
subprocess.run([sys.executable, str(REPO / "training/ocr/make_augmentation.py"), str(OCR / "augment.yaml")], check=True)
RUNS = Path(WORK) / "ocr_runs"
RUNS.mkdir(exist_ok=True)

def runs_with(name):
    return sorted((p for p in RUNS.iterdir() if (p / name).exists()), key=lambda p: p.name)

def train(batch):
    """Train until done, early-stopped or out of time. Returns the trainer's exit code and its output tail."""
    deadline = T0 + TIME_LIMIT_H * 3600 - EXPORT_RESERVE_MIN * 60
    budget = deadline - time.time()
    if budget < 600:
        raise Stop("not enough time left to train")
    cmd = ["fast-plate-ocr", "train",
           "--model-config-file", str(REPO / f"training/ocr/{MODEL_CFG}.yaml"),
           "--plate-config-file", str(REPO / "training/ocr/india_plate_config.yaml"),
           "--annotations", str(OCR / "train.csv"), "--val-annotations", str(OCR / "val.csv"),
           "--augmentation-path", str(OCR / "augment.yaml"), "--weights-path", str(WEIGHTS),
           "--lr", str(LR), "--warmup-fraction", "0.05", "--label-smoothing", "0.05",
           "--batch-size", str(batch), "--epochs", str(EPOCHS), "--workers", "4", "--seed", str(SEED),
           "--early-stopping-patience", str(PATIENCE), "--early-stopping-metric", "val_plate_acc",
           "--output-dir", str(RUNS)]
    print(" ".join(cmd), flush=True)
    print(f"training may run for up to {budget / 3600:.1f} h", flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                         start_new_session=True, env={**os.environ, "KERAS_BACKEND": BACKEND})
    timed_out = threading.Event()
    def stop_in_time():
        timed_out.set()
        print("\ntime limit reached: stopping training and keeping the best epoch so far", flush=True)
        os.killpg(p.pid, signal.SIGINT)  # like Ctrl+C: lets Keras finish what it is writing
        try:
            p.wait(300)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
    timer = threading.Timer(budget, stop_in_time); timer.daemon = True; timer.start()
    tail = []
    with open(OCR / "train.log", "a") as log:
        for line in p.stdout:
            log.write(line)
            tail = (tail + [line])[-60:]
            if "\r" not in line and "━" not in line:  # skip per-step progress bars, keep epoch summaries
                print(line, end="", flush=True)
    code = p.wait(); timer.cancel()
    return (0 if timed_out.is_set() else code), "".join(tail)

code, tail = train(BATCH)
if not runs_with("best.keras"):
    print(tail)
    oom = any(s in tail for s in ("OOM", "ResourceExhausted", "out of memory"))
    if code != 0 and time.time() - T0 < (TIME_LIMIT_H - 2) * 3600:
        print(f"training stopped without a checkpoint (exit {code}{', out of GPU memory' if oom else ''}); "
              f"trying once more with batch {BATCH // 2}")
        code, tail = train(max(8, BATCH // 2))
if not runs_with("best.keras") and not runs_with("last.keras"):
    print(tail)
    raise Stop(f"training produced no model (exit {code}); the output above says why. Full log: {OCR / 'train.log'}")
if code != 0:
    print(f"WARNING: training ended with exit {code}, using the checkpoints it saved")
run_dir = (runs_with("best.keras") or runs_with("last.keras"))[-1]
print("run:", run_dir, " finished at", elapsed())
'''),
    ("code", r'''
# Export, score and package. Whatever fails here, the trained .keras model still ends up in the zip.
EXP = Path(WORK) / "drishti_ocr"
if EXP.exists():
    shutil.rmtree(EXP)
EXP.mkdir(parents=True)
name = f"ocr_{MODEL_CFG}_in"
plate_cfg = EXP / f"{name}_plate_config.yaml"
shutil.copy(REPO / "training/ocr/india_plate_config.yaml", plate_cfg)
report = {"validation": "real + synthetic" if real else "synthetic", "before": baseline, "after": None}
try:
    for ckpt in ("best.keras", "last.keras"):
        if not (run_dir / ckpt).exists():
            continue
        shutil.copy(run_dir / ckpt, EXP / f"{name}.keras")
        tmp = EXP / "export"
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
        r = subprocess.run(["fast-plate-ocr", "export", "--model", str(run_dir / ckpt), "--format", "onnx",
                            "--plate-config-file", str(plate_cfg), "--save-dir", str(tmp), "--dynamic-batch"],
                           capture_output=True, text=True, env={**os.environ, "KERAS_BACKEND": BACKEND})
        onnx_out = next(iter(sorted(tmp.glob("*.onnx"))), None)
        if r.returncode == 0 and onnx_out is not None:
            onnx_out.rename(EXP / f"{name}.onnx"); shutil.rmtree(tmp, ignore_errors=True)
            report["checkpoint"] = ckpt
            report["after"] = eval_all(str(EXP / f"{name}.onnx"), str(plate_cfg))
            break
        print(f"ONNX export of {ckpt} failed:", (r.stdout + r.stderr)[-2000:])
    if report["after"] is None:
        report["after"] = "ONNX export failed; the trained .keras model is in the zip and can be exported later"
except Exception as e:
    report["after"] = f"export or scoring failed: {e!r}; the trained .keras model is in the zip"
finally:
    for f in ("training_log.csv", "hyper_params.json"):
        if (run_dir / f).exists():
            shutil.copy(run_dir / f, EXP / f)
    for f in ("train.log", "augment.yaml"):
        if (OCR / f).exists():
            shutil.copy(OCR / f, EXP / f)
    if (SYN / "green/preview.jpg").exists():
        shutil.copy(SYN / "green/preview.jpg", EXP / "green_plates_preview.jpg")
    report["minutes"] = round((time.time() - T0) / 60)
    (EXP / "metrics.json").write_text(json.dumps(report, indent=2))
    (EXP / "drishti.yaml").write_text(
        "# put the .onnx and _plate_config.yaml in models/ and run:  drishti --config drishti.yaml image photo.jpg\n"
        f"models:\n  ocr: models/{name}.onnx\n  ocr_config: models/{name}_plate_config.yaml\n")
    shutil.make_archive(str(Path(WORK) / "drishti_ocr"), "zip", EXP)

def score(r, k):
    v = r.get(k) if isinstance(r, dict) else None
    return v["exact_with_grammar"] if isinstance(v, dict) else None

print(f"\n{'plates':<16}{'public OCR':>12}{'fine-tuned':>12}")
for k in SETS:
    b, a = score(baseline, k), score(report["after"], k)
    if b is not None or a is not None:
        fmt = lambda x: "-" if x is None else f"{100 * x:.1f}%"
        print(f"{k:<16}{fmt(b):>12}{fmt(a):>12}")
if not isinstance(report["after"], dict):
    print("\n", report["after"])
print("\ndownload:", Path(WORK) / "drishti_ocr.zip", f" (total {elapsed()})")
'''),
]


def main() -> None:
    for name, cells in (("drishti_detector_kaggle.ipynb", DETECTOR), ("drishti_ocr_kaggle.ipynb", OCR)):
        (HERE / name).write_text(json.dumps(nb(cells), indent=1), encoding="utf-8")
        print("wrote", HERE / name)


if __name__ == "__main__":
    main()
