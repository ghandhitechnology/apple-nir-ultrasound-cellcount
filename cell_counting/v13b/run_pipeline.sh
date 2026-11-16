#!/bin/sh
# Runs the full v13b cell count on a folder of photos: v11 segmentation, pale-gap splitting, features and
# ResNet-18 embeddings, then the trained per-object classifier. Usage: run_pipeline.sh <work_dir>, where
# <work_dir>/full holds the .jpg photos; set PY and PYT to Python environments if python is not enough.
set -e
W=$1
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PY=${PY:-python}
PYT=${PYT:-$PY}
export CC_FULL="$W/full"

$PY "$ROOT/cell_counting/v11/count_cells.py" "$W/full" --out "$W/rD" --area full --labels \
  --set grow=5 seed=8 waist=0.95 hmax_h=2.5 faint_T=2 faint_gap=10 > "$W/rD.log" 2>&1
$PY "$HERE/coresplit.py" "$W/rD" "$W/rDs_0.5_3" 0.5 3
$PY "$HERE/feats2.py" "$W/rDs_0.5_3"
$PYT "$HERE/embed.py" "$W/rDs_0.5_3" > "$W/embed.log" 2>&1
$PY "$HERE/apply_fixed_pca.py" "$W/rDs_0.5_3" "$ROOT/weights/model_v13b.joblib" "$W/v13b" --embed --all-in-frame
