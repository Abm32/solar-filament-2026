#!/usr/bin/env bash
# V76 chain: frames synced -> CPU pseudo-labels (fold-0 teacher) -> wait for an idle GPU ->
# train p2first_ns fold 0 -> held-out inference -> compare vs p2first.
#   setsid nohup bash scripts/_v76_chain.sh > /dev/null 2>&1 &
set -euo pipefail
cd "$HOME/solar-filament-2026"
PY=.venv/bin/python
mkdir -p outputs/v76
exec >>outputs/v76/chain.log 2>&1
log() { echo "[$(date -u +%FT%TZ)] $*"; }
log start
while [ ! -f /mnt/v76/frames/FETCH_DONE ]; do sleep 120; done
log "frames: $(ls /mnt/v76/frames/*.jpg | wc -l) -> pseudo-labels"
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 $PY scripts/v76_pseudo.py --fold 0 --frames /mnt/v76/frames \
    --out /mnt/v76/pseudo_fold0 --shard-index $i --num-shards 8 > outputs/v76/pseudo_$i.log 2>&1 &
done
wait
log "pseudo kept: $(ls /mnt/v76/pseudo_fold0/labels | wc -l)"
# ponytail: polls for a GPU nobody uses for 10 min; a scheduler would be cleaner if this repeats.
G=""
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
log "train on gpu $G"
CUDA_VISIBLE_DEVICES=$G $PY scripts/train_v61_fold.py --fold 0 --recipe p2first_ns --work /mnt/v76 \
  --device 0 > outputs/v76/train_p2first_ns_fold0.log 2>&1
mkdir -p outputs/v76/persistent && cp -n /mnt/v76/p2first_ns_fold0/mask_ratio1/final.pt \
  /mnt/v76/p2first_ns_fold0/fold_result.json /mnt/v76/yolo_fold0/pseudo_manifest.json outputs/v76/persistent/
log infer
CUDA_VISIBLE_DEVICES=$G $PY scripts/eval_v61_oof.py infer --fold 0 --recipe p2first_ns \
  --weights /mnt/v76/p2first_ns_fold0/mask_ratio1/final.pt
log compare
$PY scripts/eval_v61_compare.py --fold 0 --a p2first --b p2first_ns --noise p2first_seed2 \
  --out outputs/v76/compare_p2first_ns.json
log DONE; touch outputs/v76/chain.done
