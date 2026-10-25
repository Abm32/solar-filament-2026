#!/usr/bin/env bash
# V68b: p2first_1280 OOF fold K on GPU G (5-fold continuation after the fold-0 pass).
#   bash scripts/_v68b_1280_fold.sh <fold> <gpu> [wait-for-file]
set -euo pipefail
K="$1"; G="$2"; WAIT="${3:-}"
cd "$HOME/solar-filament-2026"
PY=.venv/bin/python; R=p2first_1280
exec >>"outputs/v68/chain_${R}_fold$K.log" 2>&1
echo "[$(date -u +%FT%TZ)] fold $K gpu $G wait=${WAIT:-none}"
if [ -n "$WAIT" ]; then while [ ! -f "$WAIT" ]; do sleep 60; done; fi
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY scripts/train_v61_fold.py --fold "$K" \
  --recipe $R --work /mnt/v68 --device "$G" > "outputs/v68/train_${R}_fold$K.log" 2>&1
echo "[$(date -u +%FT%TZ)] trained -> infer"
CUDA_VISIBLE_DEVICES=$G $PY scripts/eval_v61_oof.py infer --fold "$K" --recipe $R \
  --weights "/mnt/v68/${R}_fold$K/mask_ratio1/final.pt" --imgsz 1280
mkdir -p outputs/v68_persistent/${R}_fold$K
cp -n /mnt/v68/${R}_fold$K/mask_ratio1/final.pt /mnt/v68/${R}_fold$K/fold_result.json outputs/v68_persistent/${R}_fold$K/
echo "[$(date -u +%FT%TZ)] DONE"; touch "outputs/v68/chain_${R}_fold$K.done"
