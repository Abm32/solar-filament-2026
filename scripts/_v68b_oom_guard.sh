#!/usr/bin/env bash
# If p2first_1280 fold K falls back to batch 1 (CUDA OOM), restart its stage 2 at batch 2 with the
# chunked loss (same procedure as folds 2 and 3).   bash scripts/_v68b_oom_guard.sh <fold> <gpu> [wait]
cd "$HOME/solar-filament-2026"
K="$1"; G="$2"; WAIT="${3:-}"; L=outputs/v68/train_p2first_1280_fold$K.log
echo "[$(date -u +%FT%TZ)] guard fold $K" >> outputs/v68/oom_guard.log
while [ ! -f outputs/v68/chain_p2first_1280_fold$K.done ]; do
  if grep -aq "CUDA out of memory" "$L" 2>/dev/null; then
    C=$(pgrep -f "^bash scripts/_v68b_1280_fold.sh $K $G"); T=$(pgrep -f "train_v61_fold.py --fold $K --recipe p2first_1280" | head -n 1)
    kill $C; sleep 1; kill $T; sleep 20
    mv /mnt/v68/p2first_1280_fold$K/mask_ratio1/run /mnt/v68/p2first_1280_fold$K/mask_ratio1/run_oom_batch1_aborted
    mv "$L" outputs/v68/train_p2first_1280_fold$K.oom_attempt.log
    setsid nohup bash scripts/_v68b_1280_fold.sh "$K" "$G" $WAIT >/dev/null 2>&1 < /dev/null &
    echo "[$(date -u +%FT%TZ)] fold $K OOM -> restarted (chain $C, train $T killed)" >> outputs/v68/oom_guard.log
    exit 0
  fi
  sleep 30
done
