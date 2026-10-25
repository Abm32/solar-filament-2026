#!/usr/bin/env bash
# V69 neighbour-frame check: wait for the V68 flip chain to free GPU 3, then run fold-0 p2first
# on every neighbour bucket (images in /mnt/v69_nbr/<bucket>/<kaggle name>) and compare vs p2first.
# Each bucket B is scored on S_B = files having both a B frame and an o0 frame, twice:
# nbr_B and the o0 control on the same S_B (before-frames exist for only ~1/3 of files).
set -euo pipefail
cd "$HOME/solar-filament-2026"
PY=.venv/bin/python; G=3
exec >>outputs/v69_nbr/chain.log 2>&1
echo "[$(date -u +%FT%TZ)] waiting for GPU $G"
while [ ! -f outputs/v68/chain_p2first_flip.done ] || [ ! -f /mnt/v69_nbr/READY ]; do sleep 120; done
for B in o0 m2 p2 m10 p10 m30 p30 xsite; do
  echo "[$(date -u +%FT%TZ)] infer $B"
  CUDA_VISIBLE_DEVICES=$G $PY scripts/eval_v61_oof.py infer --fold 0 --recipe "p2first_nbr_$B" \
    --weights /mnt/v61/p2first_fold0/mask_ratio1/final.pt --image-dir "/mnt/v69_nbr/$B"
done
for B in m2 p2 m10 p10 m30 p30 xsite; do
  S=/mnt/v69_nbr/subset_$B.txt
  comm -12 <(ls /mnt/v69_nbr/$B | sort) <(ls /mnt/v69_nbr/o0 | sort) > "$S"
  echo "[$(date -u +%FT%TZ)] compare $B on $(wc -l < "$S") files"
  $PY scripts/eval_v61_compare.py --fold 0 --a p2first --b "p2first_nbr_$B" --only "$S" \
    --out "outputs/v69_nbr/compare_$B.json" || echo "compare $B failed"
  $PY scripts/eval_v61_compare.py --fold 0 --a p2first --b p2first_nbr_o0 --only "$S" \
    --out "outputs/v69_nbr/compare_o0_on_$B.json" || echo "compare o0_on_$B failed"
done
echo "[$(date -u +%FT%TZ)] DONE"; touch outputs/v69_nbr/chain.done
