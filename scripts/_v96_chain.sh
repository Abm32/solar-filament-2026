#!/usr/bin/env bash
# V96 chain: wait for a free GPU -> for each style arm: continuation fine-tune of p2first fold 0 on that
# annotator subset -> held-out inference -> refined fold-0 gate per annotator subset (V81 harness).
#   setsid nohup bash scripts/_v96_chain.sh > /dev/null 2>&1 &
set -euo pipefail
cd "$HOME/solar-filament-2026"
PY=.venv/bin/python
ARMS="${V96_ARMS:-p2first_hard p2first_g01 p2first_more}"
mkdir -p outputs/v96
exec >>outputs/v96/chain.log 2>&1
log() { echo "[$(date -u +%FT%TZ)] $*"; }
log "start arms: $ARMS"
[ -f outputs/v96/prereg.sha256 ] || sha256sum docs/v96_style_finetune_preregistration_2026-10-22.md > outputs/v96/prereg.sha256
# ponytail: same free-GPU poll as V94 (<1000 MiB used for 10 min); V96_GPU=N skips the poll.
G="${V96_GPU:-}"
while [ -z "$G" ]; do
  for g in 0 1 2 3; do
    used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$used" -lt 1000 ]; then
      sleep 600
      used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
      [ "$used" -lt 1000 ] && G=$g && break
    fi
  done
  [ -z "$G" ] && sleep 300
done
log "gpu $G"
for ARM in $ARMS; do
  if [ ! -f "outputs/v96/work/${ARM}_fold0/fold_result.json" ]; then
    log "train $ARM"
    CUDA_VISIBLE_DEVICES=$G $PY scripts/train_v61_fold.py --fold 0 --recipe "$ARM" --work "$HOME/solar-filament-2026/outputs/v96/work" \
      --device 0 > "outputs/v96/train_${ARM}.log" 2>&1
  fi
  log "infer $ARM"
  CUDA_VISIBLE_DEVICES=$G $PY scripts/eval_v61_oof.py infer --fold 0 --recipe "$ARM" \
    --weights "$HOME/solar-filament-2026/outputs/v96/work/${ARM}_fold0/mask_ratio1/final.pt" > "outputs/v96/infer_${ARM}.log" 2>&1
  log "views $ARM: $(ls outputs/v61_oof_cache/$ARM/fold0 | wc -l)"
done
log "refined gate"
PROCS=20 CUDA_VISIBLE_DEVICES= $PY scripts/v96_refined_fold0.py $ARMS > outputs/v96/refined.log 2>&1
grep -A4 '"verdict"' outputs/v96/refined_fold0.json || true
touch outputs/v96/done
log done
