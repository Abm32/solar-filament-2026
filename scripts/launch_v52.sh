#!/usr/bin/env bash
set -euo pipefail
cd ~/solar-filament-2026
ANN=data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
IMG=data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/train_images
run() {
  local g=$1 d=$2 b=$3 r=$4 w=$5 e=$6
  mkdir -p outputs/$d
  CUDA_VISIBLE_DEVICES=$g nohup .venv/bin/python scripts/train_v52_cldice_semantic.py \
    --mode train --annotations "$ANN" --images "$IMG" \
    --work outputs/$d --output outputs/$d/train.json \
    --epochs $e --batch 4 --res $r --base $b --lr 1e-3 --cldice-w $w --fliplr --workers 4 \
    > outputs/$d/train.log 2>&1 &
  echo "launched gpu=$g dir=$d pid=$!"
}
run 0 v52_cldice      32 1024 0.5 120
run 1 v52_cldice_w1   32 1024 1.0 120
run 2 v52_cldice_b48  48 1024 0.5 120
run 3 v52_cldice_r768 32 768  0.5 160
