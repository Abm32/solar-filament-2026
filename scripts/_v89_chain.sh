#!/usr/bin/env bash
# V89 fold-0 chain (GPUs 0/1/2 only): SSL arm done -> V84-soft fine-tune fold 0 -> CPU extract; seed-44 control on GPU 2.
set -u
cd "$HOME/solar-filament-2026"
PY=.venv/bin/python; O=outputs/v89
status() { echo "$(date -Is) $*" >> "$O/STATUS"; }
arm() {  # gpu arm kind
  until [ -f "$O/ssl_$2/DONE" ]; do sleep 60; done
  mkdir -p "$O/ft_$2"
  status "ft $2 start gpu $1"
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/v89_finetune.py --init "$O/ssl_$2/ssl_state.pt" --init-kind "$3" \
    fold --arm soft --fold 0 --out "$O/ft_$2" > "$O/ft_$2/soft_fold0.log" 2>&1 && status "ft $2 done" || status "FAIL ft $2"
  PROCS=16 $PY scripts/v89_screen.py extract "$2" "$O/ft_$2/soft_fold0_refiner.pt" > "$O/extract_$2.log" 2>&1 \
    && status "extract $2 done" || status "FAIL extract $2"
}
ctrl() {  # gpu seed (after ctrl43 is written)
  until [ -f "$O/ctrl43/soft_fold0.json" ]; do sleep 60; done
  PROCS=16 $PY scripts/v89_screen.py extract ctrl43 "$O/ctrl43/soft_fold0_refiner.pt" > "$O/extract_ctrl43.log" 2>&1
  mkdir -p "$O/ctrl$2"
  status "ctrl$2 start gpu $1"
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/v89_finetune.py fold --arm soft --fold 0 --seed "$2" --out "$O/ctrl$2" \
    > "$O/ctrl$2/soft_fold0.log" 2>&1 && status "ctrl$2 done" || status "FAIL ctrl$2"
  PROCS=16 $PY scripts/v89_screen.py extract "ctrl$2" "$O/ctrl$2/soft_fold0_refiner.pt" > "$O/extract_ctrl$2.log" 2>&1
}
arm 0 mim full & arm 1 moco encoder & ctrl 2 44 &
wait
status "chain done"
