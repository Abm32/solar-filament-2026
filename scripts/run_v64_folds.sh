#!/usr/bin/env bash
# V64 follow-on: after run_v64_cp.sh ends, if BOTH fold-0 gates pass, train+infer p2first_cp
# folds 1-4 (one per GPU, each GPU waits until it is idle), then the 5-fold pooled verdict and the
# refined 141-file test with the frozen V11 refiner. Gate fail -> stop, nothing else runs.
#   setsid nohup bash scripts/run_v64_folds.sh > outputs/v64/folds.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
OUT=outputs/v64
PY=.venv/bin/python
status() { echo "$(date -Is) [folds] $*" | tee -a "$OUT/STATUS"; }

until grep -qE "pipeline end|FAIL" "$OUT/STATUS" 2>/dev/null; do sleep 300; done
if grep -q "FAIL" "$OUT/STATUS"; then status "fold-0 pipeline failed; not continuing"; exit 1; fi

if ! $PY - <<'EOF'
import json, sys
s = json.load(open("outputs/v64/structural_fold0.json"))["gate"]["p2first_cp"]["structural_pass"]
v = json.load(open("outputs/v64/compare_fold0.json"))["verdict"]
print(f"structural_pass={s} pq_verdict={v}")
sys.exit(0 if s and v == "continue" else 1)
EOF
then status "fold-0 gates NOT passed; copy-paste stops here"; exit 0; fi
status "fold-0 gates passed; launching folds 1-4"

fold_job() {  # gpu fold
  local gpu=$1 fold=$2
  while nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -q .; do
    sleep 300; done
  status "start fold $fold on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/train_v61_fold.py --fold "$fold" --recipe p2first_cp \
      --work /mnt/v61 --device 0 > "$OUT/train_fold$fold.log" 2>&1 \
    && CUDA_VISIBLE_DEVICES=$gpu $PY scripts/eval_v61_oof.py infer --fold "$fold" \
      --recipe p2first_cp --weights "/mnt/v61/p2first_cp_fold$fold/mask_ratio1/final.pt" \
      > "$OUT/infer_fold$fold.log" 2>&1 \
    && status "done  fold $fold" || status "FAIL  fold $fold (see $OUT/*_fold$fold.log)"
}
fold_job 0 1 & fold_job 1 2 & fold_job 2 3 & fold_job 3 4 &
wait
if grep -q "\[folds\] FAIL" "$OUT/STATUS"; then status "a fold failed; skipping evaluation"; exit 1; fi

$PY scripts/eval_v62_5fold.py --a p2first --b p2first_cp --noise "" \
    --out "$OUT/5fold_p2first_cp.json" > "$OUT/5fold.log" 2>&1 \
  && status "done  5-fold pooled verdict: $OUT/5fold_p2first_cp.json" || status "FAIL  5-fold (see $OUT/5fold.log)"

mkdir -p "$OUT/refined"
cp outputs/v63/v11_baseline.json outputs/v63/v11_p2first.json "$OUT/refined/"
CUDA_VISIBLE_DEVICES=1 $PY scripts/train_v63_refiner.py --recipe p2first_cp --tag v11_p2first_cp \
    --no-train --out "$OUT/refined" > "$OUT/refined.log" 2>&1 \
  && $PY scripts/compare_v63.py --dir "$OUT/refined" --pairs v11_baseline:v11_p2first_cp \
       v11_p2first:v11_p2first_cp > "$OUT/refined_compare.log" 2>&1 \
  && status "done  refined test: $OUT/refined/compare.json" || status "FAIL  refined (see $OUT/refined*.log)"
status "pipeline end"
