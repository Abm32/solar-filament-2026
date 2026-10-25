#!/usr/bin/env bash
cd "$HOME/solar-filament-2026"
s() { setsid nohup "$@" >/dev/null 2>&1 < /dev/null & }
s bash scripts/_v68b_1280_fold.sh 1 0
s bash scripts/_v68b_1280_fold.sh 2 1
s bash scripts/_v68b_1280_fold.sh 3 2 outputs/v68/chain_p2first_long.done
s bash scripts/_v68b_1280_fold.sh 4 3 outputs/v68/chain_p2first_rot.done
# V73 restarts on all GPUs once every 1280 fold is done (claims dir is reset by the launcher)
s bash -c 'for k in 1 2 3 4; do while [ ! -f outputs/v68/chain_p2first_1280_fold$k.done ]; do sleep 120; done; done; FORCE=1 bash scripts/_v73_launch.sh > outputs/v73/pipeline.log 2>&1'
