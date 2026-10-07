#!/usr/bin/env bash
# V64 copy-paste fold-0 gate on rogii GPU 3 (waits for V63's r3 to release it).
# Pre-registration: docs/v64_preregistration_2026-10-07.md. Status: outputs/v64/STATUS
#   setsid nohup bash scripts/run_v64_cp.sh > outputs/v64/pipeline.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
OUT=outputs/v64
PY=.venv/bin/python
GPU=3
mkdir -p "$OUT"
status() { echo "$(date -Is) $*" | tee -a "$OUT/STATUS"; }
step() {  # name cmd...
  local name=$1; shift
  status "start $name"
  if "$@" > "$OUT/$name.log" 2>&1; then status "done  $name"; else status "FAIL  $name (see $OUT/$name.log)"; exit 1; fi
}

status "waiting for GPU $GPU (V63 r3)"
while pgrep -f "train_v63_refiner.py --recipe p2first --tag r3" > /dev/null; do sleep 120; done
export CUDA_VISIBLE_DEVICES=$GPU
step train $PY scripts/train_v61_fold.py --fold 0 --recipe p2first_cp --work /mnt/v61 --device 0
step infer $PY scripts/eval_v61_oof.py infer --fold 0 --recipe p2first_cp \
  --weights /mnt/v61/p2first_cp_fold0/mask_ratio1/final.pt
step structural $PY scripts/analyze_v62_structural.py --fold 0 --a p2first --noise p2first_seed2 \
  --b p2first_cp --out "$OUT/structural_fold0.json"
step compare $PY scripts/eval_v61_compare.py --fold 0 --a p2first --b p2first_cp \
  --noise p2first_seed2 --out "$OUT/compare_fold0.json"
status "pipeline end"
