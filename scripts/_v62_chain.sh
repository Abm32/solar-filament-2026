#!/usr/bin/env bash
# V62 pipeline chain for ONE fold (run detached, one per GPU):
#   wait for p2first fold K  -> inference p2first fold K
#   -> train baseline_seed2 fold K (the per-fold seed-noise control)
#   -> inference baseline_seed2 fold K
# Each step only starts when the previous one's completion marker exists, and aborts on any
# failure (set -e) so a broken step is visible, never silently skipped.
#
#   bash scripts/_v62_chain.sh <fold> <gpu>
set -euo pipefail
K="$1"
G="$2"
cd "$HOME/solar-filament-2026"
PY=.venv/bin/python
LOG=outputs/v62_chain_fold${K}.log
exec >>"$LOG" 2>&1
echo "[$(date +%H:%M:%S)] chain fold=$K gpu=$G start"

while [ ! -f "/mnt/v61/p2first_fold${K}/fold_result.json" ]; do sleep 60; done
echo "[$(date +%H:%M:%S)] p2first fold$K trained -> inference"
CUDA_VISIBLE_DEVICES=$G $PY scripts/eval_v61_oof.py infer --fold "$K" --recipe p2first \
    --weights "/mnt/v61/p2first_fold${K}/mask_ratio1/final.pt"

echo "[$(date +%H:%M:%S)] training baseline_seed2 fold$K"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY scripts/train_v61_fold.py --fold "$K" \
    --recipe baseline_seed2 --work /mnt/v61 --device "$G"

echo "[$(date +%H:%M:%S)] baseline_seed2 fold$K trained -> inference"
CUDA_VISIBLE_DEVICES=$G $PY scripts/eval_v61_oof.py infer --fold "$K" --recipe baseline_seed2 \
    --weights "/mnt/v61/baseline_seed2_fold${K}/mask_ratio1/final.pt"
echo "[$(date +%H:%M:%S)] chain fold=$K DONE"
touch "outputs/v62_chain_fold${K}.done"
