# State of the art in plate recognition (October 2026) and how Drishti uses it

This is the survey behind Drishti's design choices: what the strongest systems
and papers do today, and where each idea lives in the code.

## 1. Where the field is

* **Detection is close to saturated on clean images.** Organisers of the ICPR
  2026 low-resolution plate competition note that YOLO-family detectors have
  pushed plate detection near its ceiling under normal imaging, and that the
  open problem is *reading* degraded plates.
* **Low resolution is the real frontier.** The ICPR 2026 LRLPR competition
  (LRLPR-26: 20,000 training tracks of five low- and five high-resolution frames
  each; 269 teams from 41 countries) was won at an 82.13% track-level
  recognition rate, with four teams above 80% and the top ten within about three
  points. Even the winner misread roughly one track in six.
* **No single architecture wins.** The top five used very different models
  (joint super-resolution + OCR with a teacher-student scheme, STN + SE-ResNet +
  Transformer + CTC ensembles, SVTRv2 with character voting, DINOv2/v3 feature
  ensembles, a 300M-parameter ViT with temporal fusion). What they shared was
  **exploiting several frames of the same plate** and **using the plate's
  layout constraints** during decoding.
* **Confidence quality is now measured.** The competition broke ties on the
  *confidence gap* between correct and incorrect reads, because calibrated
  confidence is what lets a deployment auto-accept most reads and send the rest
  to a human.

## 2. Components

### Detectors
| Model | Notes | In Drishti |
|---|---|---|
| **Ultralytics YOLO26** (Jan 2026, v8.4.0) | NMS-free end-to-end, DFL removed, up to 43% faster on CPU, detect/pose/OBB/seg. AGPL-3.0. | `training/detector/train.py` trains YOLO26-**pose** with 4 plate corners; `(N, 300, 6+3K)` output auto-detected |
| **RF-DETR** (Roboflow, ICLR 2026) | First real-time detector above 60 AP on COCO; DINOv2 backbone; strong transfer (RF100-VL). Apache-2.0. | default **vehicle** detector (nano 384, ONNX) |
| **DEIMv2** (2025) | DINOv3 features in a real-time DETR; DEIMv2-S is the first sub-10M model above 50 AP. | compatible alternative for custom training |
| **YOLOv9-t/s end-to-end plate models** (open-image-models, MIT) | Global plate detectors with NMS in the graph. | default **plate** detector |

### Plate OCR
| Model | Notes | In Drishti |
|---|---|---|
| **fast-plate-ocr CCT v2** (MIT) | Compact Convolutional Transformer, fixed character slots, 65+ countries, sub-millisecond on GPU. | default OCR; we use its **full per-slot probability matrix**, not only its arg-max |
| SVTRv2 (ICCV 2025), PARSeq, CLIP4STR | Strong scene-text recognisers; SVTRv2-AR was the 3rd-place LRLPR backbone; a PARSeq + CLIP4STR vote with HAT super-resolution was used at the ICIP 2026 extreme plate SR challenge. | slot OCR is swappable; any model exporting (S, V) probabilities works |
| Vision-language models (Qwen3-VL, Qwen3.5) | Can read plates zero-shot, but published comparisons swing widely across model versions and they can hallucinate. | optional **verifier** for low-confidence reads, constrained to the decoder's candidates |

### Multi-frame restoration and fusion
* **MF-LPR²** (CVIU 2025) aligns neighbouring frames with filtered optical flow
  and aggregates them instead of hallucinating detail; it reported 86.4%
  recognition where the best single-frame method managed 14%.
* **UFPR-SR-Plates** (2025) showed super-resolution plus majority-vote fusion
  over five super-resolved images raises recognition sharply on real
  low-resolution captures.
* Drishti: per-track **logit-level fusion** (weighted geometric mean of slot
  distributions over layout-consistent frames, frames chosen by image quality)
  plus **ECC-registered multi-frame stacking** of the best crops, read again
  through the same multi-view path. An optional SR model hook covers HAT /
  MambaIRv2-style restorers.

### Tracking
ByteTrack and BoT-SORT remain the production defaults (both build plates
datasets like LRLPR-26 itself). Drishti uses ByteTrack's two-stage association
with buffered IoU (C-BIoU) so small, fast plates at low frame rates still match.

### Barrier-free tolling
India's first multi-lane free-flow (MLFF) gantries (NH-48, 2026) read vehicles at highway
speed with gantry ANPR cameras. Drishti's highway preset handles plates seen in only a few
frames, and its toll rules follow the IHMCL confidence scheme (below).

### Camera-side image quality (IHMCL specification)
The Indian MLFF ANPR camera specification lists 2 MP per lane, global shutter, >50 dB SNR,
H.264, IR night vision, bad pixel correction, edge enhancement and mandatory confidence
reporting, with camera-only transactions allowed above 95% confidence. Plates should be
roughly 100-150 px wide for reliable OCR. Drishti adds the software half: ISP-style
defective pixel correction (dynamic outlier test against the 3x3 and 5x5 neighbour range,
plus a learned static map of hot, dead and stuck pixels), an edge-enhanced OCR view,
Good/Average/Poor levels, pixel-density reporting and camera health monitoring.

### Legibility
Fine-grained legibility classification (LPLCv2, 2026) treats "can this plate be read at all?"
as its own task. Drishti's equivalent is abstention: unreadable plates are withheld and frames
are chosen by image quality rather than OCR confidence.

## 3. Indian specifics encoded in the grammar
* State codes after the recent changes (TS→TG in 2024, OR→OD, UA→UK, DN merged
  into DD), with the old codes still accepted at a small cost.
* Series letters never use **I** or **O**; the number is always four digits with
  leading zeros on HSRP plates; Delhi drops the RTO's leading zero.
* Bharat series `YY BH #### XX`, vintage `SS VA XX ####`, temporary, armed forces
  and diplomatic layouts.
* Colour schemes (white private, yellow commercial, green EV, black rental) and
  two-line two-wheeler / goods-vehicle plates in the synthesiser.

## Sources
- [ICPR 2026 Competition on Low-Resolution License Plate Recognition (arXiv 2604.22506)](https://arxiv.org/pdf/2604.22506)
- [ICPR 2026 LRLPR competition site](https://icpr26lrlpr.github.io/)
- [Ultralytics YOLO26 release (v8.4.0)](https://github.com/ultralytics/assets/releases/tag/v8.4.0)
- [Ultralytics end-to-end detection guide](https://docs.ultralytics.com/guides/end2end-detection)
- [RF-DETR: Neural Architecture Search for Real-Time Detection Transformers (ICLR 2026)](https://arxiv.org/html/2511.09554v2)
- [DEIMv2: Real-Time Object Detection Meets DINOv3](https://arxiv.org/html/2509.20787v4)
- [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) and [fast-alpr](https://github.com/ankandrew/fast-alpr)
- [MF-LPR²: Multi-Frame License Plate Image Restoration and Recognition using Optical Flow](https://arxiv.org/abs/2508.14797v1)
- [Toward Advancing License Plate Super-Resolution in Real-World Scenarios (UFPR-SR-Plates)](https://journals-sol.sbc.org.br/index.php/jbcs/article/download/5159/3661/34656)
- [HAT super-resolution and a PARSeq+CLIP4STR voting ensemble (ICIP 2026 XLPSR)](https://arxiv.org/pdf/2607.08896)
- [Advancing multinational LPR through synthetic and real data fusion](https://arxiv.org/html/2601.07671v1)
- [NHAI launches barrier-free MLFF tolling on NH-48](https://letsdatascience.com/news/nhai-launches-ai-powered-barrier-free-tolling-on-nh-48-3344a5cf)
- [NHAI barrier-free toll collection at Shahjahanpur, NH-48 (Aug 2026)](https://newsonair.gov.in/nhai-introduces-barrier-free-toll-collection-at-shahjahanpur-toll-plaza-on-delhi-jaipur-section-of-nh-48/)
- [LPLCv2: fine-grained license plate legibility classification (arXiv 2604.08741)](https://arxiv.org/abs/2604.08741)
- [An embedded real-time LPR system for complex traffic scenes (arXiv 2606.27772)](https://arxiv.org/abs/2606.27772)
- D. Kumar, "Why Camera Design Matters in Open Road Tolling (ORT) and Multi-Lane Free Flow (MLFF)", e-con Systems blog, 15 July 2026 (IHMCL ANPR camera requirements)
- [Vehicle registration plates of India](https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_India)
- [Indian Licence Plate Dataset in the wild](https://arxiv.org/pdf/2111.06054)
- [PaddleOCR 3.x / PP-OCRv5](https://pypi.org/project/paddleocr/)
- [Open-LPR (Qwen3-VL based plate reader)](https://github.com/faisalthaheem/open-lpr)
