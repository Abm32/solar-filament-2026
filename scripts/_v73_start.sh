#!/usr/bin/env bash
cd "$HOME/solar-filament-2026"
mkdir -p outputs/v73
FORCE=1 WAIT_V68="0:p2first_m 1:p2first_1280 2:p2first_long 3:p2first_rot" \
  setsid nohup bash scripts/_v73_launch.sh > outputs/v73/pipeline.log 2>&1 < /dev/null &
