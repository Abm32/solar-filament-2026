#!/usr/bin/env bash
# After p2first_1280 folds 1-4 are inferred: pooled 5-fold verdict vs p2first (primary cut 0.35).
# No p2first seed-2 run exists beyond fold 0, so --noise is empty (bar = +0.010); the baseline's
# 5-fold seed difference (baseline_seed2 - baseline) is computed separately as a noise reference.
cd "$HOME/solar-filament-2026"
for k in 1 2 3 4; do while [ ! -f outputs/v68/chain_p2first_1280_fold$k.done ]; do sleep 60; done; done
PY=.venv/bin/python
CUDA_VISIBLE_DEVICES="" nice -n 5 $PY scripts/eval_v62_5fold.py --a p2first --b p2first_1280 --noise "" \
  --out outputs/v68/v68b_5fold_p2first_1280.json > outputs/v68/v68b_5fold.log 2>&1
CUDA_VISIBLE_DEVICES="" nice -n 5 $PY scripts/eval_v62_5fold.py --a baseline --b baseline_seed2 --noise "" \
  --out outputs/v68/v68b_5fold_seednoise_baseline.json > outputs/v68/v68b_seednoise.log 2>&1
touch outputs/v68/v68b_verdict.done
