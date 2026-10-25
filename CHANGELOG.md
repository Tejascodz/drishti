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

### Changed
- The OCR never runs on CoreML (it could stall for minutes on Apple silicon).
- `models/drishti.yaml` pairs the trained YOLO26 detector with the public OCR, which
  read the labelled samples equally well at higher confidence (see `docs/RESULTS.md`).
- Overlay labels stay inside the frame.

### Removed
- Plate datasets, third-party sample photos and camera frames from the repository.

## [0.1.0] - 2026-10-24

First release: detection, multi-view slot OCR with Indian format grammar, tracking and
multi-frame fusion, CLI, REST API, event store, watchlist, tolling rules, camera health,
labelling workflow and Kaggle training notebooks.
