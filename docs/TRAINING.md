# Training Drishti for Indian roads

Two models are worth specialising, in this order of payoff:

1. **The OCR**: biggest gain. The public global model has never seen HSRP fonts,
   the `IND` mark, two-line two-wheeler plates or yellow/green/black schemes.
2. **The plate detector**: Indian scenes, far and small plates, autos and trucks.

Training runs on Kaggle's free GPUs (T4). Labelling runs on your own machine with
a local vision model in Ollama. Nothing needs a paid service.

```
 your Mac                                   Kaggle (GPU)
 ────────                                   ────────────
 drishti label  ──► review.html ──► crops + annotations.csv ──► OCR notebook ──► ocr_*.onnx
 (OCR + Ollama VLM)  (fix disagreements)        (upload as dataset)                 │
                                                                                    ▼
 Kaggle Indian plates (CC0) + Roboflow unique ─────────────► detector notebook ──► plate_*.onnx
                                                                                    │
 drishti --config models/drishti.yaml ... ◄──────────── download both zips ◄────────┘
```

## 1. Set up the local vision model (Ollama)

```bash
ollama pull qwen3-vl:2b        # 1.9 GB; qwen3-vl:4b (3.3 GB) reads better if you have 16 GB RAM
drishti vlm-check              # reachable? model installed?
drishti vlm-check --image some_plate_crop.jpg   # times one real read
```

`llama3.2` (text-only) cannot read images; a vision model is required.

## 2. Label real Indian plates (text) with OCR + Ollama

Any folder of images works. If YOLO box files sit in a sibling `labels/` folder
(the Kaggle CC0 Indian set has them), those boxes are used; otherwise Drishti's
detector finds the plates.

```bash
# kaggle datasets download kedarsai/indian-license-plates-with-labels --unzip -p data/kaggle_in
drishti label data/kaggle_in/images --out data/ocr_real          # resumable; --limit 200 for a first look
open data/ocr_real/review.html
```

For each plate, Drishti's OCR and the vision model read it **independently**.
When both agree on a valid Indian registration mark, the label is accepted
automatically; everything else appears in `review.html`. There you press
<kbd>Enter</kbd> to accept, type a correction, or <kbd>Esc</kbd> for unreadable.
Then click **Download annotations.csv** and save it into `data/ocr_real/`.

Folders of **already-cropped plates** (for example the 1,182 real Indian crops in the
"Indian Plates" folder of the plate `archive.zip` set) use `--crops`, so the whole
image is treated as the plate:

```bash
drishti label "data/archive/in/in" --crops --out data/ocr_real_crops
```

Zip `data/ocr_real/` (it holds `crops/` and the CSV) and upload it to Kaggle as a
private dataset, e.g. `drishti-ocr-real`. Several labelled folders can be uploaded
together; the OCR notebook picks up every `annotations*.csv` it finds.

Agreement between two different readers is a strong signal but not proof. Skim
the *Auto-accepted* tab too; a few minutes there is worth it.

## 3. Train on Kaggle

Upload once, as private Kaggle datasets:

| Dataset | What | Size |
|---|---|---|
| `drishti-anpr` | this project zipped | small |
| `drishti-ocr-real` | output of step 2 | small |
| `lpr-v13-unique` | `lpr_v13_unique_yolo.zip` (Roboflow v13, one copy per photo, leaks removed) | 335 MB |

Or do the upload and start the OCR run in one command (Kaggle CLI, API token from
kaggle.com → Settings → API):

```bash
pip install kaggle
export KAGGLE_API_TOKEN=...
bash scripts/kaggle_train_ocr.sh <kaggle-username> data/ocr_real
```

Then *File → Import notebook* and run:

**`training/kaggle/drishti_detector_kaggle.ipynb`**: YOLO26-s at 960 px on the
CC0 Indian set (+ Roboflow). It cleans labels, keeps near-duplicate photos on the
same side of the train/val split, oversamples Indian images, and **reports the
current detector and the new one on the same Indian validation images**
(recall, precision, F1, recall on plates under 24 px). Output:
`drishti_detector.zip`.

**`training/kaggle/drishti_ocr_kaggle.ipynb`**: renders synthetic Indian
plates in parallel, splits real crops by plate number (no vehicle in both train
and val), scores the current OCR, fine-tunes from the public global weights,
scores again on the **same real validation plates** and on 3,000 hard synthetic plates
(all layouts, toll-camera angle, night infrared). Output: `drishti_ocr.zip`.

Settings for both: *Accelerator* **GPU T4**, *Internet* **On**, then
*Save Version → Save & Run All (Commit)* so the run continues after you close the browser.

Check the OCR log starts with `DRISHTI OCR v2` and reports
`validation: … real + 3000 hard synthetic plates`. Version 1 validated on the real plates
only; the public OCR already scores 100% on those easy, auto-labelled plates, so every
epoch tied, early stopping fired and the first, under-trained epoch was kept.

## 4. Use the models

Unzip both into `models/`, check them, and compare them with the public models before
switching:

```bash
python scripts/check_models.py --detector models/plate_yolo26s_in_960.onnx \
    --ocr models/ocr_cct_s_v2_in.onnx --ocr-config models/ocr_cct_s_v2_in_plate_config.yaml
python scripts/compare_setups.py my_labelled_photos/labels.csv \
    --ocr models/ocr_cct_s_v2_in.onnx --ocr-config models/ocr_cct_s_v2_in_plate_config.yaml
```

Point `models/drishti.yaml` at whichever detector and OCR win:

```yaml
models:
  plate_detector: models/plate_yolo26s_in_960.onnx
  ocr: models/ocr_cct_s_v2_in.onnx
  ocr_config: models/ocr_cct_s_v2_in_plate_config.yaml
vlm:
  enabled: true          # optional second opinion on low-confidence reads
  model: qwen3-vl:2b
```

```bash
drishti --config models/drishti.yaml image photo.jpg
drishti --config models/drishti.yaml video clip.mp4 --save-vis out.mp4
```

## 5. Measure and calibrate

Keep a labelled set that is never trained on, for example the 200-image
DataCluster sample (CC BY-NC-ND: evaluation only, no training, no redistribution):

```bash
python scripts/voc_to_csv.py data/datacluster/annotations data/datacluster/labels.csv
drishti --config models/drishti.yaml eval data/datacluster/labels.csv --report eval.json
drishti calibrate eval.json --out calibration.json     # recognition: {calibration: calibration.json}
```

`eval` reports the exact-match recognition rate, precision, character accuracy
and the confidence gap (mean confidence of correct reads minus wrong ones).

## Toll-plaza cameras

A camera above the lane sees plates vertically squashed, with the top edge wider
than the bottom, and at night in infrared. The synthetic generator renders this:
by default half of the plates use the overhead geometry
(`--overhead-frac`, pitch 12-45°, yaw ±18°) and 15% are infrared night frames
(`--ir-frac`). For a dedicated toll deployment, raise `--overhead-frac` to 0.8.
The best data of all is a few hundred labelled crops from your own lane camera:
record a day of video, run `drishti video ... --store`, and label the saved
evidence crops with `drishti label --crops`.

## Plate styles the public OCR has not seen

The public fast-plate-ocr models read standard white and yellow plates well but not green EV plates
with separator dots and emblems (`KA·53HS·3074` on a scooter): every one of its seven models misread
such a plate, even from hand-placed tight crops and colour-inverted copies. The synthetic generator
therefore renders, alongside HSRP layouts:

* green EV plates (white or yellow characters) at a higher share,
* separators: spaces, hyphens and dots, including `KA·53HS·3074` grouping,
* aftermarket emblems at the left edge and long thin strips (about 7:1),

always labelled with the plain characters, so a fine-tuned OCR learns to read through them.

## Data rules that matter more than the model

* **Split by plate (vehicle), not by image.** Frames of the same car leak
  between train and val and inflate scores. The notebooks do this for you.
* **Validate on real plates.** Synthetic validation accuracy says little about roads.
* **Crop like inference crops.** `drishti label` and `make_crops.py` cut crops
  with the same padding and rectification the pipeline uses.
* **Count unique photos, not files.** Augmented exports (14 copies per photo in
  Roboflow v13) add little; YOLO augments on its own.

## Extras

* Corner (keypoint) detector for perspective correction:
  `training/detector/convert.py` (CCPD / LabelMe polygons) + `train.py` (YOLO26-pose).
  Needs 4-corner labels, which the Kaggle and Roboflow sets do not have.
* Local GPU instead of Kaggle: `training/ocr/finetune.sh` and
  `training/detector/train.py` run the same recipes from the shell.
* Licences: Kaggle Indian set CC0; Roboflow v13 CC BY 4.0 (attribute);
  Ultralytics YOLO26 AGPL-3.0 (enterprise licence for closed products);
  fast-plate-ocr and open-image-models MIT.
