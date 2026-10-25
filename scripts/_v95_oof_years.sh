#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/solar-filament-2026"; PY=.venv/bin/python; O=outputs/v95/oof; SRC=outputs/v92/oof_v84.csv
exec >>outputs/v95/oof_run.log 2>&1
until grep -q "scoring done" outputs/v95/oof_run.log; do sleep 60; done
$PY scripts/v95_probe_ops.py $SRC $O/dropY1112.csv dropyears 2011,2012
$PY scripts/v95_probe_ops.py $SRC $O/dropY1314.csv dropyears 2013,2014
$PY scripts/v95_probe_ops.py $SRC $O/dropY1517.csv dropyears 2015,2016,2017
$PY scripts/v95_probe_ops.py $SRC $O/dropY1822.csv dropyears 2018,2019,2020,2021,2022
$PY scripts/v95_score_oof.py outputs/v95/oof_scores.json $O/dropY*.csv
echo "[$(date -u +%FT%TZ)] years scoring done"
