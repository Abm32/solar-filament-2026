# V94 pre-registration: limb-darkening division as detector input (2026-10-19, before any result)

Hash recorded in `outputs/v94/prereg.sha256` before the chain runs.

## Why
The 2026-10-19 competitor scan (`memory: leaderboard-and-competitor-scan-oct19`) found that the two best clean
public repos (theainerd OOF 0.452, ameyypawar 0.421) feed the detector a limb-darkening-corrected image, and
that CLAHE hurts in every controlled ablation. Our only prior attempt (V23/E035, radial flattening + CLAHE) was
killed at a memory gate before producing a PQ. The detector has only ever seen raw JPEGs. Recall is the bound
(FN >> FP in every clean pipeline), and faint filaments near the limb are where the raw intensity gradient is
largest, so this is the one untested preprocessing lever with external evidence.

## Fixed design (no tuning after the first run)
- Transform (`scripts/v94_limb.py`): V23 `fit_disk` + `radial_profile` unchanged (Otsu disk, 256-bin radial
  median profile, Gaussian sigma 15 bins), divide by the interpolated profile, rescale to the profile median,
  p2–p98 per-image scaling to uint8, zero outside r/R > 0.98. **No CLAHE, no blur.** Written as PNG
  (lossless) under `outputs/v94/images/`. One transform, no variants.
- Arm `p2first_limb` = the p2first recipe exactly (same init, same two stages, same seed 42, same
  augmentation, imgsz 1024, batch 4) with the training images replaced by their transformed PNGs
  (`scripts/patch_v94.py` registers the recipe; dataset dir `yolo_fold0_limb`). Labels unchanged.
- Fold 0 only (141 held-out files), one T4. Held-out inference on the transformed images with the standard
  4-flip view cache (`eval_v61_oof.py infer --image-dir outputs/v94/images`).
- Gate harness: `scripts/v94_refined_fold0.py` = `v81_refined_fold0.py` with recipes [p2first, p2first_limb]
  and seed 94: flip consensus -> V73 base fold-0 refiner on the **raw** image crops (k7, t0.45) -> cut 0.30
  -> per-record PQ. The refiner is not retrained and never sees transformed pixels. The harness must first
  reproduce V73 base fold 0 (PQ 0.453259) from the p2first views or the verdict is REPRO_FAIL.

## Gate (refined fold 0, p2first_limb vs p2first; same as V77/V81)
CONTINUE iff d >= +0.010 and P(d>0) >= 0.90 (paired file bootstrap, 2000 draws).
KILL iff d < +0.005. Otherwise INCONCLUSIVE (reported, nothing else authorised).
Secondary (reported only): the same comparison with the MIN200 area filter; direct (unrefined) fold-0 PQ via
`eval_v61_oof.py pool` is reported for the record but does not enter the verdict (V77/V81: direct gains
do not survive the refiner).

CONTINUE authorises only: train folds 1–4 with the same recipe and run the 5-fold refined comparison through
the V84 bundle (soft refiner + keep rule) before any export. No submission without the user's go-ahead.
Not authorised by this document: channel stacks (raw, limb, r/R), CLAHE, other profile fits, threshold changes.

## Compute
CPU transform of 707 images (~minutes, Pool 24). Training ~3.5 h on one free T4 (p2first fold 0 took 1.86 h +
1.70 h; peak 13.9 GiB, so the chain waits for a GPU with < 1 GB in use). Inference ~15 min. Gate CPU ~20 min.
Chain: `scripts/_v94_chain.sh`, log `outputs/v94/chain.log`, verdict `outputs/v94/refined_fold0.json`.

## Amendment 2026-10-19 09:25 UTC (before training started; no result exists)
GPUs 0/1/3 are held by another project; the user chose to train on GPU 2 next to a 3.8 GB job. To fit in the
remaining 12.5 GB the p2first_limb arm uses the V65/V68b chunked+checkpointed upstream BCE
(`v61_seg_loss.install_chunked_upstream`, identical loss value and gradient, less memory), as p2first_1280 did.
Nothing else changes. Second hash line in `outputs/v94/prereg.sha256`.
