#!/bin/bash
# usage: _v91_chain.sh GPU FOLD [FOLD...]  -- train+infer each fold sequentially on one GPU
cd ~/solar-filament-2026
G=$1; shift
for F in "$@"; do
  CUDA_VISIBLE_DEVICES=$G .venv/bin/python scripts/v91_spine.py train --fold $F > outputs/v91/train_f$F.out 2>&1 || exit 1
  CUDA_VISIBLE_DEVICES=$G .venv/bin/python scripts/v91_spine.py infer --fold $F > outputs/v91/infer_f$F.out 2>&1 || exit 1
done
