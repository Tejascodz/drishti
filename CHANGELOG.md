# Changelog

All notable changes to this project. Versions follow [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added
- `scripts/check_models.py`: validates a detector + OCR pair (loads, shapes, OCR vs its
  plate config, comparison with the public OCR) and runs them together.
- `scripts/review_reads.py`: HTML page to check reads by eye (annotated photo, enlarged
  crop, text, confidence, alternatives).
- `scripts/compare_setups.py`: every public/own detector and OCR combination, plus the
  local VLM, on one labelled set.
- `scripts/make_readme_images.py`: README results image from real photos.
- Optional REST API key (`DRISHTI_API_KEY`), pinned SHA-256 for hub downloads,
  non-root Docker image, GitHub Actions CI, security policy.

### Added (recognition)
- Plates under 90 px wide are read from tight crops only (`ocr.small_plate_px`, `ocr.small_views`).
- Deblocking retry for unreadable plates: 3x upscale + non-local-means denoising.
- The local VLM reads plates the OCR cannot read at all (`vlm.read_unreadable`); answers must be
  valid plates and are capped at Poor confidence for manual review.
- Super-resolution on by default (`realesr-general-x4v3`, release v0.1.0, BSD-3-Clause): restored
  crops are an extra opinion on small or unsure plates, never a replacement. 48 shrunk/compressed
  sample cases: 4 -> 5 read; clean photos unchanged.
- OCR training notebook v3: pre-flight checks (GPU, internet, project version, fonts), unusable labels
  skipped, a green EV validation set, colour/polarity augmentation, a time guard under Kaggle's 12 h
  limit, one retry on a failed run, and the model zipped even if export fails.
- `render_plates.py --schemes`: render only chosen colour schemes (e.g. `ev_private,ev_commercial`).
- `drishti image --debug DIR` and best guesses for unreadable plates.
- The VLM is shown only the characters' band (densest vertical edges), as dark characters on a light
  plate: green EV, black rental and army plates become greyscale negatives.
- Synthetic plates for OCR training add green EV share, separator dots (`KA·53HS·3074`), aftermarket
  emblems and long thin strips; labels stay plain characters.
- `scripts/benchmark_degraded.py`: accuracy on shrunk, JPEG-compressed, blurred copies of labelled
  photos.

### Changed
- The default plate detector is Drishti's YOLO26-s model trained for Indian roads
  (`drishti-yolo26s-in-960`, release v0.1.0, AGPL-3.0); the OCR stays the public CCT-S v2.
  Default crop views add the tight first view this detector needs.
- The OCR never runs on CoreML (it could stall for minutes on Apple silicon).
- `models/drishti.yaml` is a template for your own model files.
- Overlay labels stay inside the frame.
- Model downloads show progress; an interrupted download leaves no partial file.

### Removed
- Plate datasets, third-party sample photos and camera frames from the repository.

## [0.1.0] - 2026-10-24

First release: detection, multi-view slot OCR with Indian format grammar, tracking and
multi-frame fusion, CLI, REST API, event store, watchlist, tolling rules, camera health,
labelling workflow and Kaggle training notebooks.
