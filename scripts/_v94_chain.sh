#!/usr/bin/env bash
# V94 chain: limb-divide 707 train images (CPU) -> wait for a free GPU -> train p2first_limb fold 0
# -> held-out inference on limb images -> refined fold-0 gate vs p2first (V81 harness).
#   setsid nohup bash scripts/_v94_chain.sh > /dev/null 2>&1 &
set -euo pipefail
cd "$HOME/solar-filament-2026"
PY=.venv/bin/python
TRAIN=data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/train_images
mkdir -p outputs/v94
exec >>outputs/v94/chain.log 2>&1
log() { echo "[$(date -u +%FT%TZ)] $*"; }
log start
[ -f outputs/v94/prereg.sha256 ] || sha256sum docs/v94_limb_division_preregistration_2026-10-19.md > outputs/v94/prereg.sha256
$PY scripts/patch_v94.py
$PY scripts/v94_limb.py --selfcheck "$TRAIN/20110109104734Ch.jpeg"
[ -f outputs/v94/images/DONE ] || { $PY scripts/v94_limb.py --src "$TRAIN" --out outputs/v94/images --procs 24 && touch outputs/v94/images/DONE; }
log "images: $(ls outputs/v94/images/*.png | wc -l)"
# ponytail: same free-GPU poll as V76 (<1000 MiB used for 10 min); p2first peaks ~13.9 GiB on a T4.
G="${V94_GPU:-}"   # V94_GPU=2 skips the poll (user decision 2026-10-19: share GPU 2)
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
CUDA_VISIBLE_DEVICES=$G $PY scripts/train_v61_fold.py --fold 0 --recipe p2first_limb --work "$HOME/solar-filament-2026/outputs/v94/work" \
  --device 0 > outputs/v94/train_fold0.log 2>&1
log infer
CUDA_VISIBLE_DEVICES=$G $PY scripts/eval_v61_oof.py infer --fold 0 --recipe p2first_limb \
  --weights "$HOME/solar-filament-2026/outputs/v94/work/"p2first_limb_fold0/mask_ratio1/final.pt --image-dir outputs/v94/images \
  > outputs/v94/infer_fold0.log 2>&1
log "views: $(ls outputs/v61_oof_cache/p2first_limb/fold0 | wc -l)"
log "refined gate"
PROCS=20 CUDA_VISIBLE_DEVICES= $PY scripts/v94_refined_fold0.py > outputs/v94/refined.log 2>&1
grep -E '"verdict"' outputs/v94/refined_fold0.json
touch outputs/v94/done
log done
