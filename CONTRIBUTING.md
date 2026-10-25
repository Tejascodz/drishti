# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[server,track,dev]"
drishti models pull
bash scripts/fetch_samples.sh samples
```

End-to-end tests render a plate with DejaVu Sans Condensed Bold and skip without it
(`sudo apt install fonts-dejavu-extra` on Debian/Ubuntu).

## Before opening a pull request

```bash
ruff check drishti scripts tests training
pytest
```

CI runs both on Python 3.10 and 3.12.

## Guidelines

* Keep changes focused; one topic per pull request.
* Add or update a test for behaviour changes. Recognition changes should state their
  effect on a labelled set (`drishti eval labels.csv` or `scripts/compare_setups.py`).
* Never commit plate data, camera frames, model weights or databases. Use only images
  you may publish in documentation.
* A new country is a region pack in `drishti/formats/packs/` plus tests in
  `tests/test_grammar.py`.
