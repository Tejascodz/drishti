#!/usr/bin/env bash
# Fine-tune the global CCT plate OCR on Indian plates (real crops + synthetic).
#
#   bash training/ocr/finetune.sh data/ocr/train/annotations.csv data/ocr/val/annotations.csv runs/ocr_in
#
# Recommended data mix (see docs/TRAINING.md): every real crop you have, plus
# synthetic plates at roughly 1:1. Validate on REAL plates only.
# GPU box: pip install "fast-plate-ocr[train]" "jax[cuda12]" albumentations
set -euo pipefail
TRAIN_CSV=${1:?train annotations.csv}
VAL_CSV=${2:?val annotations.csv}
OUT=${3:-runs/ocr_in}
EPOCHS=${EPOCHS:-60}
BATCH=${BATCH:-128}
LR=${LR:-0.0005}
MODEL_CFG=${MODEL_CFG:-training/ocr/cct_s_v2.yaml}
HERE="$(cd "$(dirname "$0")" && pwd)"
export KERAS_BACKEND=${KERAS_BACKEND:-jax}

WEIGHTS=${WEIGHTS:-}
if [ -z "$WEIGHTS" ]; then
  # public global weights (MIT) as the starting point
  WEIGHTS=$(python3 -c "from drishti import hub; print(hub.fetch('cct-s-v2-global-model', 'keras'))")
fi
mkdir -p "$OUT"
python3 "$HERE/make_augmentation.py" "$OUT/augment.yaml"

fast-plate-ocr train \
  --model-config-file "$MODEL_CFG" \
  --plate-config-file "$HERE/india_plate_config.yaml" \
  --annotations "$TRAIN_CSV" \
  --val-annotations "$VAL_CSV" \
  --augmentation-path "$OUT/augment.yaml" \
  --weights-path "$WEIGHTS" \
  --lr "$LR" --warmup-fraction 0.05 --label-smoothing 0.05 \
  --batch-size "$BATCH" --epochs "$EPOCHS" \
  --early-stopping-patience 15 --early-stopping-metric val_plate_acc \
  --output-dir "$OUT"

echo "trained models are in $OUT; export the best one with training/ocr/export.sh"
