#!/usr/bin/env bash
# V81 thin, fold K: [wait for file] -> train -> infer fold K held-out -> persist.
#   bash scripts/_v81_fold.sh <fold> <gpu> [waitfile]
set -euo pipefail
K="$1"; G="$2"; WAIT="${3:-}"; R=p2first_thin; W=/mnt/v81
cd "$HOME/solar-filament-2026"
PY=.venv/bin/python
exec >>"outputs/v81/chain_${R}_fold$K.log" 2>&1
echo "[$(date -u +%FT%TZ)] fold $K gpu=$G wait=$WAIT"
[ -n "$WAIT" ] && while [ ! -f "$WAIT" ]; do sleep 60; done
echo "[$(date -u +%FT%TZ)] train"
CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY scripts/train_v61_fold.py --fold "$K" \
  --recipe $R --work $W --device 0 > "outputs/v81/train_${R}_fold$K.log" 2>&1
echo "[$(date -u +%FT%TZ)] infer"
CUDA_VISIBLE_DEVICES=$G $PY scripts/eval_v61_oof.py infer --fold "$K" --recipe $R \
  --weights "$W/${R}_fold$K/mask_ratio1/final.pt"
mkdir -p outputs/v81_persistent/${R}_fold$K && cp -n "$W/${R}_fold$K/mask_ratio1/final.pt" "$W/${R}_fold$K/fold_result.json" outputs/v81_persistent/${R}_fold$K/
echo "[$(date -u +%FT%TZ)] DONE"; touch "outputs/v81/chain_${R}_fold$K.done"
