#!/usr/bin/env bash
# V65 RESUME (pre-registration: docs/v65_preregistration_2026-10-07.md; unchanged gate).
# The original run_v65.sh fold jobs all died with CUDA OOM in stage 2 (mask_ratio=1) at epoch
# 11-12 of 20. Stage 1 (80 ep) had finished for folds 1-4, so this re-runs ONLY stage 2 from
# base/final.pt with a chunked+checkpointed UPSTREAM BCE mask loss (train_v61_fold.py wires it for
# every *_cp recipe; value+gradient identical to upstream, tested; peak 9.2 GiB vs 14+ GiB).
# Everything after the folds (direct 5-fold, V11/r3 refiner on p2first_cp proposals, compare) is
# the SAME commands as run_v65.sh. Never submits.
#   setsid nohup bash scripts/run_v65_resume.sh > outputs/v65/resume_pipeline.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
OUT=outputs/v65
PY=.venv/bin/python
mkdir -p "$OUT/refined"
status() { echo "$(date -Is) $*" | tee -a "$OUT/STATUS"; }
run() {  # name gpu cmd...
  local name=$1 gpu=$2; shift 2
  status "start $name (GPU $gpu)"
  if CUDA_VISIBLE_DEVICES=$gpu "$@" > "$OUT/$name.log" 2>&1; then status "done  $name"
  else status "FAIL  $name (see $OUT/$name.log)"; return 1; fi
}

fold_job() {  # gpu fold
  local gpu=$1 fold=$2
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True run "resume_train_fold$fold" "$gpu" \
      $PY scripts/train_v61_fold.py --fold "$fold" --recipe p2first_cp --work /mnt/v61 --device 0 \
  && run "resume_infer_fold$fold" "$gpu" $PY scripts/eval_v61_oof.py infer --fold "$fold" \
      --recipe p2first_cp --weights "/mnt/v61/p2first_cp_fold$fold/mask_ratio1/final.pt"
}

status "resume pipeline start"
fold_job 0 1 & fold_job 1 2 & fold_job 2 3 & fold_job 3 4 &  # (gpu fold): folds 1-4, fold 0 done
wait
if grep -q "FAIL  resume_" "$OUT/STATUS"; then status "a resumed fold failed; stack test skipped"; exit 1; fi

cp outputs/v63/v11_baseline.json outputs/v63/v11_p2first.json outputs/v63/r3.json "$OUT/refined/"
run five_fold_direct 0 $PY scripts/eval_v62_5fold.py --a p2first --b p2first_cp --noise "" \
    --out "$OUT/5fold_p2first_cp.json" &
run refined_v11_cp 1 $PY scripts/train_v63_refiner.py --recipe p2first_cp --tag v11_p2first_cp \
    --no-train --out "$OUT/refined" &
run refined_r3cp 2 $PY scripts/train_v63_refiner.py --recipe p2first_cp --tag r3cp \
    --epochs 12 --lr 1e-4 --out "$OUT/refined" &
wait
run compare 0 $PY scripts/compare_v63.py --dir "$OUT/refined" --pairs v11_baseline:r3cp \
    r3:r3cp v11_baseline:v11_p2first_cp v11_p2first:v11_p2first_cp
status "resume pipeline end"
