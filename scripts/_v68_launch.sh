#!/usr/bin/env bash
# V68: launch one fold-0 arm detached on one GPU:  bash scripts/_v68_launch.sh <recipe> <gpu>
cd "$HOME/solar-filament-2026"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup .venv/bin/python scripts/train_v61_fold.py \
  --fold 0 --recipe "$1" --work /mnt/v68 --device "$2" > "outputs/v68/train_$1_fold0.log" 2>&1 &
