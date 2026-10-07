#!/usr/bin/env bash
# V63 unattended pipeline on rogii. One job queue per GPU; GPU 0 waits for the all-data p2first
# (fold 99) training to finish first. Status: outputs/v63/STATUS, results: outputs/v63/compare.json
#   nohup bash scripts/run_v63.sh > outputs/v63/pipeline.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
OUT=outputs/v63
PY=.venv/bin/python
mkdir -p "$OUT"
status() { echo "$(date -Is) $*" | tee -a "$OUT/STATUS"; }

job() {  # gpu tag recipe [extra args]
  local gpu=$1 tag=$2 recipe=$3; shift 3
  status "start $tag on GPU $gpu"
  if CUDA_VISIBLE_DEVICES=$gpu $PY scripts/train_v63_refiner.py --recipe "$recipe" --tag "$tag" \
       --out "$OUT" "$@" > "$OUT/$tag.log" 2>&1; then
    status "done  $tag"
  else
    status "FAIL  $tag (see $OUT/$tag.log)"
  fi
}

status "pipeline start"
( job 1 v11_baseline baseline --no-train; job 1 r2 baseline ) &
( job 2 v11_p2first p2first --no-train; job 2 r1 p2first ) &
( job 3 r3 p2first --epochs 12 --lr 1e-4 ) &
(
  while pgrep -f "train_v61_fold.py --fold 99" > /dev/null; do sleep 300; done
  status "GPU 0 free (fold 99 training finished)"
  job 0 r1s2 p2first --seed 20261007
) &
wait
$PY scripts/compare_v63.py --dir "$OUT" > "$OUT/compare.log" 2>&1 \
  && status "compare done: $OUT/compare.json" || status "FAIL compare (see $OUT/compare.log)"
status "pipeline end"
