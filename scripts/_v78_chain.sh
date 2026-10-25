#!/usr/bin/env bash
# V78 label-transfer arm: wait for frames -> train fold 0 -> infer held-out -> compare vs p2first.
#   bash scripts/_v78_chain.sh <recipe> <gpu>
set -euo pipefail
R="$1"; G="$2"; W="/mnt/v78_$R"
cd "$HOME/solar-filament-2026"
PY=.venv/bin/python
mkdir -p outputs/v78
exec >>"outputs/v78/chain_$R.log" 2>&1
echo "[$(date -u +%FT%TZ)] chain $R gpu=$G waiting for frames"
while [ ! -f /mnt/v78_frames/READY ]; do sleep 60; done
echo "[$(date -u +%FT%TZ)] train"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY scripts/train_v61_fold.py --fold 0 --recipe "$R" \
  --work "$W" --device "$G" > "outputs/v78/train_$R.log" 2>&1
echo "[$(date -u +%FT%TZ)] infer"
CUDA_VISIBLE_DEVICES=$G $PY scripts/eval_v61_oof.py infer --fold 0 --recipe "$R" \
  --weights "$W/${R}_fold0/mask_ratio1/final.pt"
echo "[$(date -u +%FT%TZ)] compare"
$PY scripts/eval_v61_compare.py --fold 0 --a p2first --b "$R" --noise p2first_seed2 \
  --out "outputs/v78/compare_$R.json"
mkdir -p outputs/v78_persistent/$R && cp -n "$W/${R}_fold0/mask_ratio1/final.pt" "$W/${R}_fold0/fold_result.json" outputs/v78_persistent/$R/
echo "[$(date -u +%FT%TZ)] DONE"; touch "outputs/v78/chain_$R.done"
