#!/usr/bin/env bash
# Export a fine-tuned OCR to ONNX (+ CoreML for Apple silicon) next to its plate config.
#
#   bash training/ocr/export.sh runs/ocr_in/<run>/best.keras models/ocr_in
#
# Then point Drishti at it:
#   models:
#     ocr: models/ocr_in/best.onnx
#     ocr_config: models/ocr_in/best_plate_config.yaml
set -euo pipefail
KERAS_MODEL=${1:?path to .keras}
DEST=${2:-models/ocr_in}
HERE="$(cd "$(dirname "$0")" && pwd)"
export KERAS_BACKEND=${KERAS_BACKEND:-jax}
mkdir -p "$DEST"
fast-plate-ocr export --model "$KERAS_MODEL" --format onnx --plate-config-file "$HERE/india_plate_config.yaml" \
  --save-dir "$DEST" --dynamic-batch
stem=$(basename "$KERAS_MODEL" .keras)
cp "$HERE/india_plate_config.yaml" "$DEST/${stem}_plate_config.yaml"
if [ "${COREML:-0}" = "1" ]; then
  fast-plate-ocr export --model "$KERAS_MODEL" --format coreml --plate-config-file "$HERE/india_plate_config.yaml" --save-dir "$DEST"
fi
ls -la "$DEST"
