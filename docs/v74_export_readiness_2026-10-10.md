# V74 export readiness: p2first_1280 (2026-10-10)

Status: **export path ready, final model missing.** Nothing was submitted. No GPU was used: all tests ran on CPU on rogii. No labels were read.

## What V65 used (reference graph, unchanged in V74)
- Detector: `/mnt/v61/p2first_fold99/mask_ratio1/final.pt` (sha `7ff39296…`), imgsz 1024.
- Views: 4 flips through `v61_toolkit.generate_views` (box iou 0.3, retina masks, agnostic NMS, floor 0.1).
- Consensus: `detector_tta.cluster_views` flip_consensus, match iou 0.5.
- Refiner: `outputs/v65/export/r3all_refiner_state.pt` (sha `17b6280f…`). This is the r3 recipe trained on all 707 p2first OOF proposal sets (`train_v63_refiner.py --recipe p2first --tag r3all --train-all --epochs 12 --lr 1e-4`). Inference uses flip TTA.
- Ownership: `clean_refined_predictions` refined_constrained. Score-ordered pixel ownership, with an internal floor of 20 px.
- Thresholds: the V17 cell, **t 0.45 / c 0.30**, and min area 20.
- Output: `outputs/v65/export/submission_v65_p2first_r3all.csv`, 1224 rows, sha `530e89ea…`.

## New files
- `scripts/export_v74.py` is the V65 graph with these parameters:
  - `--detector`, `--imgsz {1024,1280}`
  - `--refiner` (defaults to the V65 r3all refiner)
  - `--min-area` (default 200)
  - `--device`, `--limit`
  - sharding via `--shard-index/--num-shards`

  MIN200 drops final kept masks under 200 px, after ownership and confidence. This matches the V72 measurement. With `--min-area 20` it does nothing. The script writes a per-shard CSV and a manifest that records the SHAs, the recipe and the number of masks dropped by min area.
- `scripts/_v74_export.sh <final.pt>` runs the full export:
  1. 4-GPU sharded export
  2. checks that every shard manifest is complete
  3. merges the shards
  4. validates with `validate_submission(min_area=200)` and the 180-stem check
  5. writes `outputs/v74/submission_v74.csv`, `.sha256` and `validation.json`

  Env overrides: IMGSZ, MIN_AREA, REFINER, GPUS, DEVICE, OUT.

## Tests (CPU)
1. **Parity:** I ran 1024, the p2first fold-99 checkpoint, min-area 20 on shard 0, first 5 images (`outputs/v74/parity_1024_min20`). The 34 rows are byte-identical to the same rows of the V65 CSV. CPU inference reproduces the GPU output exactly.
2. **1280 smoke:** I ran the fold-0 checkpoint on 3 images (`outputs/v74/smoke_1280_fold0`). It passed the validator: 19 rows, no overlaps, min area 815.
3. **End-to-end dry run:** I ran `_v74_export.sh` with the fold-0 checkpoint on all 180 images on CPU (`GPUS="x x x x" DEVICE=cpu`), which took about 5.5 min. It produced 1238 rows over 177 images, the same 3 empty images as V65, min area 216, and 17 masks dropped by MIN200. All validator checks passed. Output: `outputs/v74/dryrun_fold0/`. **This file is NOT a submission candidate**, because the fold-0 model was trained on only 566 files.

## Missing prerequisite: p2first_1280 all-data model (fold 99)
```
sudo mkdir -p /mnt/v74 && sudo chown azureuser: /mnt/v74   # /mnt is root-owned
CUDA_VISIBLE_DEVICES=<free gpu> setsid nohup .venv/bin/python scripts/train_v61_fold.py \
  --fold 99 --recipe p2first_1280 --work /mnt/v74 --device 0 > outputs/v74/train_fold99.log 2>&1 < /dev/null &
bash scripts/_v74_export.sh /mnt/v74/p2first_1280_fold99/mask_ratio1/final.pt
```
- **Time:** fold 0 took 10656 s (base) + 9834 s (mask) = 5.7 h on 566 files on one T4. Scaled to 707 files, fold 99 should take about **7 h**.
- **Disk:** copy the final.pt off /mnt, which is ephemeral.
- **Export time:** the GPU export should take minutes.

## Refiner retrained on 1280 proposals: not required, optional
- The CropUNet reads grayscale image crops only. Proposals affect it only through the crop box and the dilated constraint, so the r3all refiner runs unchanged on 1280 proposals. The parity and dry-run tests confirm it runs.
- Retraining could help a little because the crop-box distribution shifts. It needs the 5-fold p2first_1280 OOF caches from the V68 run, then:

  ```
  train_v63_refiner.py --recipe p2first_1280 --tag r3all1280 --train-all --epochs 12 --lr 1e-4
  ```

  Adopt it only if r3 refit on p2first_1280 OOF beats r3 on the 141-file paired comparison. Pass the result as `REFINER=...`.
