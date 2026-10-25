#!/usr/bin/env bash
# V89: wait for a full (40k-step) SSL arm, fine-tune fold 0 (V84 soft recipe) on the least-loaded of GPUs 0-2, extract.
#   scripts/_v89_ft.sh ARM KIND      (mim full | moco encoder)
set -u
cd "$HOME/solar-filament-2026"
PY=.venv/bin/python; O=outputs/v89; A=$1
status() { echo "$(date -Is) $*" >> "$O/STATUS"; }
until grep -q '"steps": 40000' "$O/ssl_$A/DONE" 2>/dev/null; do sleep 60; done
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1<3' | sort -t, -k2 -n | head -1 | cut -d, -f1)
mkdir -p "$O/ft_$A"
status "ft $A start gpu $G"
CUDA_VISIBLE_DEVICES=$G $PY scripts/v89_finetune.py --init "$O/ssl_$A/ssl_state.pt" --init-kind "$2" \
  fold --arm soft --fold 0 --out "$O/ft_$A" > "$O/ft_$A/soft_fold0.log" 2>&1 && status "ft $A done" || status "FAIL ft $A"
PROCS=16 $PY scripts/v89_screen.py extract "$A" "$O/ft_$A/soft_fold0_refiner.pt" > "$O/extract_$A.log" 2>&1 \
  && status "extract $A done" || status "FAIL extract $A"
