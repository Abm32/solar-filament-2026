#!/usr/bin/env bash
# V68 chain for one arm: wait for training -> fold-0 held-out inference -> compare vs p2first.
#   bash scripts/_v68_chain.sh <recipe> <gpu> [imgsz]
set -euo pipefail
R="$1"; G="$2"; S="${3:-1024}"
cd "$HOME/solar-filament-2026"
PY=.venv/bin/python
exec >>"outputs/v68/chain_$R.log" 2>&1
echo "[$(date -u +%FT%TZ)] chain $R gpu=$G imgsz=$S start"
while [ ! -f "/mnt/v68/${R}_fold0/fold_result.json" ]; do sleep 120; done
echo "[$(date -u +%FT%TZ)] trained -> infer"
CUDA_VISIBLE_DEVICES=$G $PY scripts/eval_v61_oof.py infer --fold 0 --recipe "$R" \
  --weights "/mnt/v68/${R}_fold0/mask_ratio1/final.pt" --imgsz "$S"
echo "[$(date -u +%FT%TZ)] compare"
$PY scripts/eval_v61_compare.py --fold 0 --a p2first --b "$R" --noise p2first_seed2 \
  --out "outputs/v68/compare_$R.json"
mkdir -p outputs/v68_persistent/$R && cp -n /mnt/v68/${R}_fold0/mask_ratio1/final.pt /mnt/v68/${R}_fold0/fold_result.json outputs/v68_persistent/$R/
echo "[$(date -u +%FT%TZ)] DONE"; touch "outputs/v68/chain_$R.done"
