#!/usr/bin/env bash
# Downloads the OpenDroneMap Brighton Beach drone dataset (BSD-2-Clause, 18 images, ~62 MB)
# and writes a ready-to-run config with the orthophoto enabled.
#
#   tools/download_brighton_beach.sh [target_dir]      (default: ~/sfm_data/brighton_beach)
#   build/sfm_main ~/sfm_data/brighton_beach/config.yaml
set -euo pipefail

DIR="${1:-$HOME/sfm_data/brighton_beach}"
REPO="OpenDroneMap/drone_dataset_brighton_beach"
RAW="https://raw.githubusercontent.com/$REPO/master"
mkdir -p "$DIR/images"
cd "$DIR"

curl -sfL "https://api.github.com/repos/$REPO/git/trees/HEAD?recursive=1" |
  python3 -c "
import json, sys
for x in json.load(sys.stdin)['tree']:
    if x['path'].startswith('images/') or x['path'] in ('brighton_beach.jpg', 'LICENSE', 'README.md'):
        print(x['path'])" > files.txt
xargs -P 8 -I{} curl -sfL -o {} "$RAW/{}" < files.txt
echo "Downloaded $(ls images | wc -l) images to $DIR/images (reference ortho: brighton_beach.jpg)"

cat > config.yaml <<'EOF'
# OpenDroneMap Brighton Beach (BSD-2-Clause), DJI Phantom 3 (FC300S), 18 nadir images 4000x2250.
io:
  image_dir: images
  output_dir: out

cameras:
  - id: 0
    fx: 2340.0          # 3.61 mm focal / 6.17 mm sensor width * 4000 px (EXIF)
    fy: 2340.0
    cx: 2000.0
    cy: 1125.0
    dist: [0.0, 0.0, 0.0, 0.0, 0.0]   # uncalibrated; refined in the final BA

features:
  max_image_dim: 2000   # SIFT on half resolution, keypoints mapped back to full resolution
  max_features: 8000

matching:
  mode: exhaustive

final:
  ba:
    refine_intrinsics: true
    refine_focal_length: false   # flat nadir scene: focal and flying height are ambiguous

ortho:
  enabled: true
  max_dimension: 8000
EOF
echo "Wrote $DIR/config.yaml"
