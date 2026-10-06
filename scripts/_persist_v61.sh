#!/usr/bin/env bash
# Copy every COMPLETED V61/V62 fold run from the ephemeral /mnt/v61 to the persistent OS disk.
# Idempotent (cp -n never overwrites). Writes a SHA256SUMS manifest. Safe to re-run as folds finish.
set -euo pipefail
DEST="$HOME/solar-filament-2026/outputs/v61_persistent"
mkdir -p "$DEST"
n_done=0
for d in /mnt/v61/*_fold[0-9]; do
  [ -f "$d/fold_result.json" ] || continue
  name=$(basename "$d")
  mkdir -p "$DEST/$name"
  cp -n "$d/fold_result.json" "$DEST/$name/fold_result.json"
  cp -n "$d/mask_ratio1/final.pt" "$DEST/$name/final.pt"
  cp -n "$d/base/final.pt" "$DEST/$name/base_final.pt"
  n_done=$((n_done + 1))
done
# The 2 init checkpoints for the P2 architectures and the fold manifest are small and essential.
mkdir -p "$DEST/_inputs"
cp -n "$HOME/solar-filament-2026/outputs/v61_folds.json" "$DEST/_inputs/" 2>/dev/null || true
cp -n "$HOME/solar-filament-2026/outputs/v62_init/"*.pt "$DEST/_inputs/" 2>/dev/null || true
cp -n "$HOME/solar-filament-2026/outputs/v16_arm_r1_seed1/detector.pt" \
      "$DEST/_inputs/v16_arm_r1_seed1_detector.pt" 2>/dev/null || true
(cd "$DEST" && find . -type f \( -name '*.pt' -o -name 'fold_result.json' -o -name '*.json' \) \
    ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
echo "persisted runs: $n_done ; total size: $(du -sh "$DEST" | cut -f1)"
ls "$DEST"
