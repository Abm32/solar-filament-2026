#!/usr/bin/env bash
# V84: V82 soft refiner folds 1-4 on GPUs 0/1/2 (never 3); fold 4 on whichever GPU frees first.
set -u
cd "$HOME/solar-filament-2026"
PY=.venv/bin/python; O=outputs/v84
status() { echo "$(date -Is) $*" >> "$O/STATUS"; }
run() {  # gpu fold
  status "start soft fold $2 on gpu $1"
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/train_v82_refiner_targets.py fold --arm soft --fold "$2" \
    --out $O > "$O/soft_fold$2.log" 2>&1 && status "done fold $2" || status "FAIL fold $2"
  echo "$1" >> "$O/.free"
}
: > "$O/.free"
run 0 1 & run 1 2 & run 2 3 &
until [ -s "$O/.free" ]; do sleep 30; done
run "$(head -1 $O/.free)" 4
wait
status "train all done"
