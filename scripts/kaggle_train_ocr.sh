#!/usr/bin/env bash
# Fine-tune the OCR on Kaggle's free GPU in one command: uploads this project and your labelled
# real plates as a private dataset, pushes the OCR notebook (GPU T4, internet on) and starts it.
#
#   pip install kaggle
#   export KAGGLE_API_TOKEN=...        # kaggle.com -> Settings -> API -> Generate New Token
#   bash scripts/kaggle_train_ocr.sh   [kaggle-username] [path/to/labelled/plates]
#
# Afterwards:
#   kaggle kernels status  <user>/drishti-ocr-v3
#   kaggle kernels output  <user>/drishti-ocr-v3 -p models/ocr_in     # when it says COMPLETE
set -euo pipefail
USER_NAME=${1:-${KAGGLE_USER:-tejasss21}}
REAL=${2:-data/ocr_real}
DATASET="$USER_NAME/drishti"
KERNEL="$USER_NAME/drishti-ocr-v3"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

command -v kaggle >/dev/null || { echo "install the Kaggle CLI first: pip install kaggle"; exit 1; }
[ -n "${KAGGLE_API_TOKEN:-}" ] || [ -f ~/.kaggle/access_token ] || [ -f ~/.kaggle/kaggle.json ] || {
  echo "set KAGGLE_API_TOKEN (kaggle.com -> Settings -> API)"; exit 1; }

echo "1/3 packing the project and $REAL"
mkdir -p "$STAGE/data/drishti"
git -C "$ROOT" archive HEAD | tar -x -C "$STAGE/data/drishti"
if [ -d "$ROOT/$REAL" ]; then
  cp -r "$ROOT/$REAL" "$STAGE/data/ocr_real"
  echo "    real plates: $(ls "$STAGE/data/ocr_real"/annotations*.csv 2>/dev/null | wc -l) label file(s)"
else
  echo "    no $REAL folder: training on synthetic plates only"
fi
cat > "$STAGE/data/dataset-metadata.json" <<EOF
{"title": "drishti", "id": "$DATASET", "licenses": [{"name": "other"}]}
EOF

echo "2/3 uploading dataset $DATASET (private)"
if kaggle datasets status "$DATASET" >/dev/null 2>&1; then
  kaggle datasets version -p "$STAGE/data" -m "Drishti $(git -C "$ROOT" rev-parse --short HEAD)" --dir-mode zip
else
  kaggle datasets create -p "$STAGE/data" --dir-mode zip
fi
echo "    waiting for Kaggle to process the dataset"
for _ in $(seq 1 60); do
  kaggle datasets status "$DATASET" 2>/dev/null | grep -qi ready && break
  sleep 10
done

echo "3/3 pushing and starting notebook $KERNEL (GPU T4, internet on)"
mkdir -p "$STAGE/kernel"
cp "$ROOT/training/kaggle/drishti_ocr_kaggle.ipynb" "$STAGE/kernel/"
cat > "$STAGE/kernel/kernel-metadata.json" <<EOF
{
  "id": "$KERNEL",
  "title": "drishti-ocr-v3",
  "code_file": "drishti_ocr_kaggle.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["$DATASET"],
  "competition_sources": [],
  "kernel_sources": []
}
EOF
kaggle kernels push -p "$STAGE/kernel" --accelerator NvidiaTeslaT4

cat <<EOF

Started. Training takes about 7 hours; nothing needs to stay open.
  status : kaggle kernels status $KERNEL
  log    : https://www.kaggle.com/code/$KERNEL
  result : kaggle kernels output $KERNEL -p models/ocr_in     (when the status is COMPLETE)
EOF
