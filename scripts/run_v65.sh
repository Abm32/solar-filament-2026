#!/usr/bin/env bash
# V65 on rogii (pre-registration: docs/v65_preregistration_2026-10-07.md). Status: outputs/v65/STATUS
#  A (GPU 1): r3-recipe refiner on ALL 707 p2first OOF proposal sets -> test export with the
#     all-data p2first detector -> merged CSV -> repo validator. NEVER submits.
#  B (GPUs 0-3): p2first_cp folds 1-4 (train + OOF infer), then the stack test on the 141 files:
#     r3cp = r3 recipe re-fit on p2first_cp proposals; PRIMARY r3cp - v11_baseline.
#   setsid nohup bash scripts/run_v65.sh > outputs/v65/pipeline.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
OUT=outputs/v65
PY=.venv/bin/python
DET=/mnt/v61/p2first_fold99/mask_ratio1/final.pt
TEST=data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images
mkdir -p "$OUT/export" "$OUT/refined"
status() { echo "$(date -Is) $*" | tee -a "$OUT/STATUS"; }
run() {  # name gpu cmd...
  local name=$1 gpu=$2; shift 2
  status "start $name (GPU $gpu)"
  if CUDA_VISIBLE_DEVICES=$gpu "$@" > "$OUT/$name.log" 2>&1; then status "done  $name"
  else status "FAIL  $name (see $OUT/$name.log)"; return 1; fi
}

export_track() {
  run export_refiner 1 $PY scripts/train_v63_refiner.py --recipe p2first --tag r3all --train-all \
      --epochs 12 --lr 1e-4 --out "$OUT/export" || return 1
  local ref="$OUT/export/r3all_refiner_state.pt"
  local dsha rsha; dsha=$(sha256sum "$DET" | cut -d' ' -f1); rsha=$(sha256sum "$ref" | cut -d' ' -f1)
  for s in 0 1 2 3; do
    run "export_shard$s" 1 $PY scripts/export_v62_shard.py --shard-index $s --num-shards 4 \
        --detector "$DET" --detector-sha "$dsha" --refiner "$ref" --refiner-sha "$rsha" \
        --mask-threshold 0.45 --confidence 0.30 --output-dir "$OUT/export" || return 1
  done
  $PY -c "
import pandas as pd
df = pd.concat([pd.read_csv(f'$OUT/export/shard_{s}.csv') for s in range(4)], ignore_index=True)
assert df.filament_id.is_unique
df.to_csv('$OUT/export/submission_v65_p2first_r3all.csv', index=False)
print(len(df), 'rows')" > "$OUT/merge.log" 2>&1 || { status "FAIL  merge"; return 1; }
  $PY scripts/validate_submission.py "$OUT/export/submission_v65_p2first_r3all.csv" \
      --test-dir "$TEST" --report "$OUT/export/validation.json" > "$OUT/validate.log" 2>&1 \
    && status "done  export validated: $OUT/export/submission_v65_p2first_r3all.csv (NOT submitted)" \
    || status "FAIL  validation (see $OUT/validate.log)"
}

fold_job() {  # gpu fold
  local gpu=$1 fold=$2
  while nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -q .; do
    sleep 120; done
  run "train_fold$fold" "$gpu" $PY scripts/train_v61_fold.py --fold "$fold" --recipe p2first_cp \
      --work /mnt/v61 --device 0 \
  && run "infer_fold$fold" "$gpu" $PY scripts/eval_v61_oof.py infer --fold "$fold" \
      --recipe p2first_cp --weights "/mnt/v61/p2first_cp_fold$fold/mask_ratio1/final.pt"
}

status "pipeline start"
( export_track; fold_job 1 4 ) &
fold_job 0 1 & fold_job 2 2 & fold_job 3 3 &
wait
if grep -q "FAIL  \(train\|infer\)_fold" "$OUT/STATUS"; then status "a fold failed; stack test skipped"; exit 1; fi

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
status "pipeline end"
