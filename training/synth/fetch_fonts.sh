#!/usr/bin/env bash
# Download open-licensed (SIL OFL) condensed fonts used by the plate synthesiser.
# HSRP plates use a narrow sans-serif; these cover a realistic spread of shapes.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)/fonts"
mkdir -p "$DIR"
BASE="https://raw.githubusercontent.com/google/fonts/main/ofl"
FONTS=(
  "barlowcondensed/BarlowCondensed-SemiBold.ttf"
  "barlowcondensed/BarlowCondensed-Bold.ttf"
  "robotocondensed/RobotoCondensed%5Bwght%5D.ttf"
  "oswald/Oswald%5Bwght%5D.ttf"
  "archivonarrow/ArchivoNarrow%5Bwght%5D.ttf"
  "sairacondensed/SairaCondensed-SemiBold.ttf"
  "asapcondensed/AsapCondensed-SemiBold.ttf"
  "pathwaygothicone/PathwayGothicOne-Regular.ttf"
)
for f in "${FONTS[@]}"; do
  # variable fonts are saved as Name-Variable.ttf: brackets are rejected by some uploaders (Kaggle)
  name="$(basename "$f" | sed 's/%5Bwght%5D/-Variable/')"
  [ -s "$DIR/$name" ] && continue
  curl -fsSL -o "$DIR/$name" "$BASE/$f" && echo "fetched $name" || echo "skipped $name"
done
echo "fonts in $DIR"
