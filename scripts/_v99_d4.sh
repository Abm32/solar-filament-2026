#!/usr/bin/env bash
# V99 arm d4 (8-view detector TTA). Prereg: docs/v99_arms_preregistration_2026-10-25.md
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; OUT=outputs/v99; mkdir -p $OUT
echo "$(date -Is) d4 start" | tee -a $OUT/chain.log
$PY scripts/v99_d4_views.py demo
CUDA_VISIBLE_DEVICES=${GPU:-3} $PY scripts/v99_d4_views.py > $OUT/d4_views.log 2>&1
echo "$(date -Is) d4 views cached: $(ls outputs/v61_oof_cache/p2first_d4/fold0 | wc -l)" | tee -a $OUT/chain.log
cp outputs/v84/soft_fold0_refiner.pt $OUT/d4_refiner.pt
SOLAR_RECIPE=p2first_d4 FLOOR=0.10 PROCS=${PROCS:-22} CUDA_VISIBLE_DEVICES= $PY scripts/v99_screen.py extract d4 $OUT/d4_refiner.pt > $OUT/extract_d4.log 2>&1
PRIMARY=t050_c025 $PY scripts/v99_screen.py gate d4 > $OUT/gate_d4.log 2>&1
echo "$(date -Is) d4 done" | tee -a $OUT/chain.log; touch $OUT/done_d4
