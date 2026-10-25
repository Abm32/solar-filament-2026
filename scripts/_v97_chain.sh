#!/bin/bash
# V97 chain (rogii): train cond (GPU 1) + soft_g01 (GPU 2) -> 6 CPU extracts -> gate.
#   setsid nohup bash scripts/_v97_chain.sh > /dev/null 2>&1 &
cd ~/solar-filament-2026 || exit 1
source .venv/bin/activate
mkdir -p outputs/v97
exec >>outputs/v97/chain.log 2>&1
log() { echo "[$(date -u +%FT%TZ)] $*"; }
S=scripts/v97_style_refiner.py
log start
CUDA_VISIBLE_DEVICES=1 python $S fold --arm cond --fold 0 > outputs/v97/train_cond.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=2 python $S fold --arm soft_g01 --fold 0 > outputs/v97/train_soft_g01.log 2>&1 &
P2=$!
wait $P1; log "train cond rc $?"
wait $P2; log "train soft_g01 rc $?"
export CUDA_VISIBLE_DEVICES=
python $S extract soft soft > outputs/v97/extract_soft.log 2>&1; log "extract soft rc $?"
python $S extract soft_g01 outputs/v97/soft_g01_fold0_refiner.pt > outputs/v97/extract_soft_g01.log 2>&1; log "extract soft_g01 rc $?"
for k in 0 1 2; do
  python $S extract cond@0$k outputs/v97/cond_fold0_refiner.pt --token $k > outputs/v97/extract_cond0$k.log 2>&1; log "extract cond@0$k rc $?"
done
python $S extract cond@oracle outputs/v97/cond_fold0_refiner.pt --oracle > outputs/v97/extract_condoracle.log 2>&1; log "extract cond@oracle rc $?"
python $S gate soft soft_g01 cond@00 cond@01 cond@02 cond@oracle > outputs/v97/gate.log 2>&1; log "gate rc $?"
grep -A8 '"verdict"' outputs/v97/gate.json | head -12
log done
touch outputs/v97/done
