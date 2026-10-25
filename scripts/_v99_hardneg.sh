#!/usr/bin/env bash
# V99 arm hardneg. Prereg: docs/v99_arms_preregistration_2026-10-25.md
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; OUT=outputs/v99; mkdir -p $OUT
echo "$(date -Is) hardneg start" | tee -a $OUT/chain.log
$PY scripts/v99_hardneg.py demo
CUDA_VISIBLE_DEVICES=${GPU:-2} $PY scripts/v99_hardneg.py fold --arm hardneg --fold 0 --out $OUT > $OUT/train_hardneg.log 2>&1
echo "$(date -Is) hardneg trained $(sha256sum $OUT/hardneg_fold0_refiner.pt | cut -c1-16)" | tee -a $OUT/chain.log
PROCS=${PROCS:-22} CUDA_VISIBLE_DEVICES= $PY scripts/v99_screen.py extract hardneg $OUT/hardneg_fold0_refiner.pt > $OUT/extract_hardneg.log 2>&1
for s in 43 44 45; do ln -sf ../v89/extract_ctrl$s.pkl $OUT/extract_s$s.pkl; done
PRIMARY=t050_c030_veto $PY scripts/v99_screen.py gate hardneg s43 s44 s45 > $OUT/gate_hardneg.log 2>&1
echo "$(date -Is) hardneg done" | tee -a $OUT/chain.log; touch $OUT/done_hardneg
