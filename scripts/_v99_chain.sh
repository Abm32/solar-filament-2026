#!/usr/bin/env bash
# V99 arm soft1280: V84 soft refiner retrained (stage A+B) with p2first_1280 fold-0 OOF proposals, judged with
# 1280 views, vs the deployed V84 soft refiner on p2first views. Prereg: docs/v99_soft1280_preregistration_2026-10-25.md
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; OUT=outputs/v99; mkdir -p $OUT
export SOLAR_RECIPE=p2first_1280
echo "$(date -Is) start" | tee -a $OUT/chain.log
CUDA_VISIBLE_DEVICES=${GPU:-1} $PY scripts/train_v82_refiner_targets.py fold --arm soft --fold 0 --out $OUT > $OUT/train_soft1280.log 2>&1
echo "$(date -Is) trained $(sha256sum $OUT/soft_fold0_refiner.pt | cut -c1-16)" | tee -a $OUT/chain.log
cp outputs/v84/soft_fold0_refiner.pt $OUT/ctrl1280_refiner.pt   # deployed refiner, no sibling json -> no cross-check
PROCS=${PROCS:-22} CUDA_VISIBLE_DEVICES= $PY scripts/v99_screen.py extract soft1280 $OUT/soft_fold0_refiner.pt > $OUT/extract_soft1280.log 2>&1
PROCS=${PROCS:-22} CUDA_VISIBLE_DEVICES= $PY scripts/v99_screen.py extract ctrl1280 $OUT/ctrl1280_refiner.pt > $OUT/extract_ctrl1280.log 2>&1
for s in 43 44 45; do ln -sf ../v89/extract_ctrl$s.pkl $OUT/extract_s$s.pkl; done
$PY scripts/v99_screen.py gate soft1280 ctrl1280 s43 s44 s45 > $OUT/gate.log 2>&1
echo "$(date -Is) done" | tee -a $OUT/chain.log; touch $OUT/done
