#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/solar-filament-2026"; PY=.venv/bin/python; O=outputs/v95/oof; mkdir -p $O
exec >>outputs/v95/oof_run.log 2>&1
SRC=outputs/v92/oof_v84.csv
run() { n=$1; shift; [ -f $O/$n.csv ] || $PY scripts/v95_probe_ops.py $SRC $O/$n.csv "$@"; }
run minarea500 minarea 500; run minarea1000 minarea 1000
run topk3 topk 3; run topk5 topk 5
run merge10 merge 10; run merge25 merge 25
run maxrr090 maxrr 0.90; run maxrr080 maxrr 0.80
for s in Bh Ch Lh Mh Th Uh; do run drop$s dropsite $s; done
echo "[$(date -u +%FT%TZ)] ops done"
$PY scripts/v95_score_oof.py outputs/v95/oof_scores.json $SRC $O/*.csv
echo "[$(date -u +%FT%TZ)] scoring done"
