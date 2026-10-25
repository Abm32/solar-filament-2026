#!/usr/bin/env bash
# V82 fold-0 refiner target arms on GPUs 0/1/2 (never 3), then base_rerun on GPU 0, then CPU gate.
set -u
cd "$HOME/solar-filament-2026"
PY=.venv/bin/python; O=outputs/v82
status() { echo "$(date -Is) $*" >> "$O/STATUS"; }
run() {  # gpu arm outdir
  status "start $2 on gpu $1"
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/train_v82_refiner_targets.py fold --arm "$2" --fold 0 \
    --out "$3" > "$3/$2_fold0.log" 2>&1 && status "done $2" || status "FAIL $2"
}
(run 0 soft $O; run 0 base outputs/v82_rerun) &
run 1 union $O &
run 2 qweight $O &
wait
status "gate start"
PROCS=24 $PY scripts/v82_refined_fold0.py > "$O/refined_fold0.log" 2>&1 && status "gate done" || status "FAIL gate"
