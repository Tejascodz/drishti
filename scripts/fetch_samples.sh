#!/usr/bin/env bash
# Download four real Indian road photos (from the Indian_LPR project's demo folder)
# and a labels file for a quick sanity check. Only an annotated preview is stored
# (docs/images/results.jpg); the photos themselves are downloaded from the source.
set -euo pipefail
DIR=${1:-samples}
mkdir -p "$DIR"
BASE="https://raw.githubusercontent.com/sanchit2843/Indian_LPR/main/demo_images"
fetch() { curl -fsSL -o "$DIR/$2" "$BASE/$1" || curl -fsSL -o "$DIR/$2" "${BASE/main/master}/$1"; }
fetch "20201031_133155_3220.jpg" in_gj_street.jpg
fetch "VID_20201027_154255-1_3396.jpg" in_tn_highway.jpg
fetch "NO20200731-075035-000056-converted_267.png" in_dl_dashcam.png
fetch "final_merged_219468.jpg" in_hr_traffic.jpg
cat > "$DIR/labels.csv" <<'EOF'
image_path,plate_text
in_gj_street.jpg,GJ02CG3862
in_tn_highway.jpg,TN22DA5047
in_dl_dashcam.png,DL1CR9541
in_hr_traffic.jpg,HR85A0398
EOF
echo "samples in $DIR  ->  drishti image $DIR   |   drishti eval $DIR/labels.csv"
