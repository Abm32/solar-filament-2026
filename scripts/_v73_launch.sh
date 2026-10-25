#!/usr/bin/env bash
# V73 5-fold OOF refiner + arms (docs/v73_refiner_preregistration_2026-10-10.md). 20 jobs
# (4 trained arms x 5 folds), one GPU per job; one worker per GPU pulls from a shared queue,
# longest arm first. Resumable: a fold whose JSON exists is skipped. Ends with `compare`.
#   nohup bash scripts/_v73_launch.sh > outputs/v73/pipeline.log 2>&1 &
#   SMOKE=1 bash scripts/_v73_launch.sh      # CPU: 2 files/fold, 1 epoch/stage -> outputs/v73_smoke
# Refuses to start while any listed GPU holds >1 GB (V68); FORCE=1 overrides. GPUS="0 1" narrows.
set -u
cd "$HOME/solar-filament-2026"
PY=.venv/bin/python
if [ "${SMOKE:-0}" = 1 ]; then
  OUT=outputs/v73_smoke; EXTRA="--limit 2 --epochs-a 1 --epochs-b 1"; GPUS="${GPUS:-cpu cpu}"
else
  OUT=outputs/v73; EXTRA=""; GPUS="${GPUS:-0 1 2 3}"
  if [ "${FORCE:-0}" != 1 ]; then
    for g in $GPUS; do
      used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
      [ "$used" -lt 1000 ] || { echo "GPU $g busy (${used} MiB); FORCE=1 to override"; exit 1; }
    done
  fi
fi
rm -rf "$OUT/claims"; mkdir -p "$OUT/claims"
status() { echo "$(date -Is) $*" | tee -a "$OUT/STATUS"; }

JOBS=()
for arm in crop512 rn34 base r3c; do for k in 0 1 2 3 4; do JOBS+=("$arm $k"); done; done

worker() {  # $1 = GPU index or "cpu"
  local dev=$1 arm k
  [ "$dev" = cpu ] && dev=""
  # WAIT_V68="gpu:recipe ...": hold this worker until that GPU's V68 chain has finished
  local w; w=$(tr " " "\n" <<< "${WAIT_V68:-}" | grep "^$1:" | cut -d: -f2)
  if [ -n "$w" ]; then
    status "worker $1 waiting for V68 $w"
    while [ ! -f "outputs/v68/chain_$w.done" ]; do sleep 60; done
  fi
  for job in "${JOBS[@]}"; do
    read -r arm k <<< "$job"
    mkdir "$OUT/claims/${arm}_fold$k" 2>/dev/null || continue  # atomic claim
    status "start $arm fold $k on ${dev:-cpu}"
    # shellcheck disable=SC2086
    if CUDA_VISIBLE_DEVICES=$dev $PY scripts/train_v73_refiner.py fold --arm "$arm" --fold "$k" \
         --out "$OUT" $EXTRA > "$OUT/${arm}_fold$k.log" 2>&1; then
      status "done  $arm fold $k"
    else
      status "FAIL  $arm fold $k (see $OUT/${arm}_fold$k.log)"
    fi
  done
}

status "pipeline start (GPUS=$GPUS)"
for g in $GPUS; do worker "$g" & done
wait
$PY scripts/train_v73_refiner.py compare --out "$OUT" > "$OUT/compare.log" 2>&1 \
  && status "compare done: $OUT/compare.json" || status "FAIL compare (see $OUT/compare.log)"
status "pipeline end"
