#!/usr/bin/env bash
# Per-GPU hand-off as each p2first_1280 fold finishes (no idle gap, so V76's idle-GPU poll can't grab
# a planned GPU): GPU 0 (fold 1) -> p2first_1280 fold-99 all-data model; GPUs 1-3 (folds 2-4) -> V73.
cd "$HOME/solar-filament-2026"
(
  while [ ! -f outputs/v68/chain_p2first_1280_fold1.done ]; do sleep 30; done
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/train_v61_fold.py \
    --fold 99 --recipe p2first_1280 --work /mnt/v68 --device 0 \
    > outputs/v68/train_p2first_1280_fold99.log 2>&1
) &
FORCE=1 GPUS="1 2 3" WAIT_V68="1:p2first_1280_fold2 2:p2first_1280_fold3 3:p2first_1280_fold4" \
  bash scripts/_v73_launch.sh > outputs/v73/pipeline.log 2>&1
wait
