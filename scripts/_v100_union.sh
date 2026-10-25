#!/usr/bin/env bash
# V100 arm union: p2first detector trained on union-of-annotators labels (V81 arm, never launched), fold 0,
# judged refined through the deployed V84 soft fold-0 refiner at t0.50 (v99_screen). Prereg: docs/v100_preregistration_2026-10-25.md
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; OUT=outputs/v100; W=$OUT/work; mkdir -p $W
echo "$(date -Is) union start" | tee -a $OUT/chain.log
CUDA_VISIBLE_DEVICES=${GPU:-3} $PY scripts/train_v61_fold.py --fold 0 --recipe p2first_union --work $W --device 0 > $OUT/train_union.log 2>&1
CK=$(ls $W/p2first_union_fold0/mask_ratio1/final.pt 2>/dev/null || find $W -path "*union*" -name final.pt | grep mask_ratio1 | head -1)
echo "$(date -Is) union trained $CK $(sha256sum $CK | cut -c1-16)" | tee -a $OUT/chain.log
CUDA_VISIBLE_DEVICES=${GPU:-3} $PY scripts/eval_v61_oof.py infer --fold 0 --recipe p2first_union --weights $CK > $OUT/infer_union.log 2>&1
echo "$(date -Is) union views: $(ls outputs/v61_oof_cache/p2first_union/fold0 | wc -l)" | tee -a $OUT/chain.log
mkdir -p outputs/v99; cp outputs/v84/soft_fold0_refiner.pt outputs/v99/union_refiner.pt
SOLAR_RECIPE=p2first_union PROCS=${PROCS:-22} CUDA_VISIBLE_DEVICES= $PY scripts/v99_screen.py extract union outputs/v99/union_refiner.pt > $OUT/extract_union.log 2>&1
for s in 43 44 45; do ln -sf ../v89/extract_ctrl$s.pkl outputs/v99/extract_s$s.pkl; done
PRIMARY=t050_c030 $PY scripts/v99_screen.py gate union s43 s44 s45 > $OUT/gate_union.log 2>&1
cp outputs/v99/screen_union.json $OUT/
echo "$(date -Is) union done" | tee -a $OUT/chain.log; touch $OUT/done_union
