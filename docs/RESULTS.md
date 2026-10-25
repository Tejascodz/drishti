# Measured results

Unless a section says otherwise, numbers come from the default public models on a
2-core CPU. The trained YOLO26 detector and the first fine-tuned OCR are compared in
their own sections. Reproduce any row with `drishti eval` or `scripts/compare_setups.py`.

## Plate detection

Roboflow "License Plate Recognition" v13, **test split (1,020 unique photos,
mostly Vietnamese)**, IoU ≥ 0.5, confidence ≥ 0.3:

| Detector | Recall | Precision | F1 | Recall, plates < 24 px tall (229) | Time / 1,020 images |
|---|---|---|---|---|---|
| yolo-v9-s-608 (default) | **94.8%** | **98.0%** | **0.964** | 83.4% | 191 s |
| yolo-v9-t-384 (edge preset) | 92.1% | 96.9% | 0.944 | 75.1% | 38 s |

Takeaway: plate finding is already strong; the weak spot is small, distant
plates. Training should target those, which is what the 960 px YOLO26 recipe
in the Kaggle notebook does.

### Trained detector (Kaggle, YOLO26-s, 960 px, 115 epochs)

Same images for both models; IoU ≥ 0.5. Validation = 37 Indian + 2,045 Roboflow photos
never trained on; extra test = 500 further held-out photos.

| Set | Detector | Recall | Precision | F1 | Recall, small plates |
|---|---|---|---|---|---|
| Validation (2,082) | yolo-v9-s-608 (default) | 93.6% | 96.4% | 0.950 | 79.3% (487) |
| Validation (2,082) | **YOLO26-s 960 (trained)** | **96.1%** | **97.5%** | **0.968** | **86.0%** |
| Extra test (500) | yolo-v9-s-608 (default) | 97.5% | 98.2% | 0.978 | 91.7% (84) |
| Extra test (500) | **YOLO26-s 960 (trained)** | **98.0%** | 98.0% | **0.980** | **92.9%** |

mAP50 0.973, mAP50-95 0.680. Its boxes are looser than the default detector's, so its
config adds a fourth, tight crop view (`ocr.views: [[0.02,-0.06],[0.04,0.10],[0.10,0.18],[0.17,0.27]]`).
Chosen on 6 real Indian photos (incl. a night infrared toll frame and an overhead
road camera) and the 3 clips:

| Crop views | Real photos correct (sum of confidences) | Clips correct |
|---|---|---|
| default 3 views | 6 of 6 (5.33) | 1 of 3 |
| tight 3 views (tuned on clips only) | 5 of 6 (4.04): night IR plate misread | 2 of 3 |
| **4 views (tight + default)** | **6 of 6 (5.32)** | **2 of 3** |

Lesson kept: tune on real images, not on synthetic clips.

### Bad pixel correction and edge enhancement

6 real photos (4 road photos, a night infrared toll frame, an overhead road camera), read
clean and with about 2% of the plate's pixels made dead / hot (a third as 2x2 demosaic
clusters); plus the 3 clips. Trained YOLO26 detector, 4 crop views.

| Setting | Clean photos correct (sum of conf.) | Photos with bad pixels correct (sum) | Clips correct |
|---|---|---|---|
| no correction | 6 of 6 (5.32) | 2 of 6 (0.98) | 2 of 3 |
| **bad pixel correction (default)** | **6 of 6 (5.34)** | **4 of 6 (3.31)** | 2 of 3 |
| edge-enhanced view only | 6 of 6 (5.27) | 2 of 6 (0.98) | 2 of 3, wrong clip more confident |
| both | 6 of 6 (5.29) | 4 of 6 (3.31) | 2 of 3, wrong clip more confident |

Bad pixel correction is on by default; the edge-enhanced view gave no gain and is off
(`imaging.edge_enhance: on` to try it on your own camera).

## Plate reading

| Test | Result |
|---|---|
| 4 real Indian road photos, 5 plates (GJ, TN, DL, HR) | all 4 single-line plates read correctly (0.83 to 1.00 confidence); the 5th, a 32 px two-line scooter plate, is withheld instead of guessed. Detector + OCR without the grammar and multi-view reading gets 3 of 4 (`HR85A039`). |
| 600 synthetic Indian plates with heavy degradation, same OCR | free decoding **37.2%** → with the Indian format grammar **50.2%** exact match |
| 6 real dealer-style Indian crops (dotted / italic fonts) | 1 confident correct read; the other 5 are low-confidence and go to review, none wrongly auto-accepted. This is the gap Indian OCR training closes. |

### Detector × OCR combinations (`scripts/compare_setups.py`)

The 4 labelled road photos (`scripts/fetch_samples.sh`), every combination of the public
models and the trained ones, on an Apple-silicon MacBook (CPU). "Precision" counts every
emitted read; the trained detector also finds smaller, unlabelled plates in the
background (a two-line scooter plate, a second car), which count as wrong.

| Detector | OCR | Labelled plates read | Precision | Confidence when right | ms / photo |
|---|---|---|---|---|---|
| public yolo-v9-s-608 | public CCT-S v2 | 4 / 4 | 100% | **0.95** | 299 |
| public yolo-v9-s-608 | fine-tuned | 4 / 4 | 100% | 0.68 | 931 |
| **trained YOLO26-s 960** | **public CCT-S v2** | 4 / 4 | 80% | **0.95** | 853 |
| trained YOLO26-s 960 | fine-tuned | 4 / 4 | 67% | 0.65 | 818 |
| trained YOLO26-s 960 | fine-tuned + Qwen3-VL 2B check | 4 / 4 | 67% | 0.65 | 1,983 |

The first fine-tuned OCR reads the same strings but with far lower confidence, which
would send every plate to manual review (automatic tolling needs > 0.95). Its training
run selected the best epoch on 58 easy validation plates that every epoch scored 100%
on, so it kept an early, under-trained epoch; notebook v2 validates on 3,000 hard
synthetic plates as well. Until a v2 model beats the public OCR, `models/drishti.yaml`
uses the trained detector with the public OCR. The VLM check changed no read here and
adds about a second per photo. Four photos are a smoke test, not a benchmark.

## Video (tracking + multi-frame fusion)

Synthetic "vehicle approaching the camera" clips built from the real photos:
60 frames, zooming from far to near with blur, noise and JPEG compression.

| Clip | Single-frame reads correct | Track-level result |
|---|---|---|
| GJ02CG3862 | 8 of 36 reads (22%) | **GJ02CG3862**, 0.85 |
| DL1CR9541 | 21 of 30 reads (70%) | **DL1CR9541**, 0.99 |
| HR85A0398 | 1 of 8 reads (12%) | HR5A0398 (one character missing), flagged low confidence 0.34 |

## Latency (2-core CPU, 1920×1080)

| Stage | Time |
|---|---|
| Plate detector (yolo-v9-s-608) | ≈150 ms |
| OCR, one batch of crop views | ≈30 ms |
| Full image (detect + 3-4 views per plate + vehicle) | 300-650 ms |

A GPU or Apple-silicon CoreML run is several times faster; `drishti bench`
measures your machine.
