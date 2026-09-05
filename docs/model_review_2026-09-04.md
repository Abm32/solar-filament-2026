# Model Review - 2026-09-04

## Live Submission Status

- Checked at `2026-09-04 12:52 IST`.
- Completed submission: ref `56007181`, file `submission_nonoverlap.csv`, public score `0.31`.
- Previous raw submissions errored:
  - ref `56006718`, `submission.csv`
  - ref `56006788`, `submission.csv`
- Visible leaderboard at check time: top public score `0.56`; many teams are clustered at `0.55`.

## Current Pipeline

1. Convert COCO polygons to YOLO-seg labels.
2. Pick one annotator set per duplicated image using annotation count, then total area.
3. Train `yolo11s-seg.pt` at `imgsz=1024`, `epochs=30`, `batch=4`, `mask_ratio=2`, `overlap_mask=False`.
4. Train a ResNet18 encoder crop U-Net on ground-truth filament crops.
5. Use YOLO proposals at test time, refine each proposal crop, RLE encode each instance.
6. Postprocess overlapping masks before submission because Kaggle rejects overlapping raw instance masks.

## Findings

- The first live valid score is `0.31`, substantially behind the visible `0.56` top score.
- YOLO validation training was still improving at epoch 30. The final mask mAP50 was about `0.343`, and the best row was the final epoch, so early stopping did not exhaust the model.
- The kernel validation PQ was not trustworthy: `pairwise_iou` used `uint8 @ uint8`, which can overflow when intersections exceed 255 pixels. That explains the recorded `best_pq=0.0028504863186230847` despite a public score of `0.31`.
- The raw refiner submission likely failed because independent crop masks can overlap. The accepted non-overlap CSV solved validity but may have removed useful pixels after masks had already expanded into neighboring instances.
- Current test output predicts 1,810 instances over 178 of 180 test images, median 10 instances per populated image. Two test images have no predicted rows.
- The pulled public YOLO + U-Net notebook uses almost the same architecture and reports stronger public performance, so the immediate gap is likely implementation details, validation reliability, postprocessing, training duration, and threshold selection rather than needing a completely different model.

## V4 Changes

- Fixed validation IoU overflow by using `int32` intersections.
- Added score-ordered non-overlap resolution directly in the Kaggle kernel.
- Added validation comparison across:
  - `yolo_only`
  - `refined_constrained`
  - `refined_free`
- Constrained refiner masks to the original YOLO mask dilated by 3 pixels for the main refined variant.
- Wrote `validation_variants.txt` so the chosen inference mode and confidence threshold are auditable.
- Increased YOLO patience from 5 to 10 while keeping epochs at 30 for a controlled next run.
- Pushed as Kaggle kernel version `4`; status was `KernelWorkerStatus.COMPLETE` at `2026-09-04 15:31 IST`.
- V4 validation summary:
  - `best_conf=0.25`
  - `best_mode=refined_constrained`
  - `best_pq=0.40872896213406407`
- V4 submission ref `56010439` completed with public score `0.34`.
- V4 generated fewer test instances than the accepted V3 non-overlap run: `1,293` rows over `177` populated images versus `1,810` rows over `178` images. This should reduce false positives, but it may also reduce recall if public test images have more weak filaments than the validation split.

## V5 Changes

- Increased YOLO training from `30` to `80` epochs.
- Increased YOLO patience from `10` to `20`.
- Widened threshold sweep to `0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45`.
- Added `submission_diagnostics.txt` to record row count, populated images, missing images, and instance count distribution summary.
- Kept `yolo11s-seg.pt` before trying `yolo11m-seg.pt` because V4 still peaked at the final epoch: mask mAP50 was `0.34306` at epoch 30, so training duration is the cleaner first variable.
- Pushed as Kaggle kernel version `5`; status was `KernelWorkerStatus.RUNNING` at `2026-09-04 15:36 IST`.

## V6 Prepared Changes

- Train on all non-leaking train annotator variants instead of only the richest annotation set per train file.
- Keep validation as one best annotation set per file. This avoids training on one annotator's labels for a file and validating on another annotator's labels for the same image.
- Generate unique YOLO filenames for duplicate train annotator variants, so labels do not overwrite each other.
- Switch YOLO to single-class segmentation:
  - the competition submission has no class column;
  - the metric evaluates instance masks, not chirality labels;
  - raw categories are imbalanced and `Ambiguous` has zero annotations;
  - a single foreground class gives the detector more examples for one mask/object concept.
- Data impact:
  - current V4/V5 train image-label pairs: `566`;
  - V6 train image-label pairs: `937`;
  - current V4/V5 train annotations: `4,452`;
  - V6 train annotations: `6,606`;
  - annotation increase: `2,154`, or `48.4%`.

## Next Optimization Queue

1. Wait for V4 public score and compare it to validation PQ. If it lands near `0.40-0.45`, the validation split is predictive enough to keep optimizing against it.
2. Run V5 longer training. If mask mAP50 still peaks at the final epoch, extend again or switch to `yolo11m-seg.pt`.
3. Run V6 if V5 is still below the leaderboard cluster. It attacks data volume and class-target mismatch without increasing model size.
4. If V6 public score improves but remains below `0.55`, test `yolo11m-seg.pt` with 50-60 epochs and the same postprocess.
5. Add per-image validation diagnostics: TP/FP/FN, missing-image list, low-PQ image list, and predicted/ground-truth count deltas.
6. Test a recall-preserving variant if V4/V5 over-prune: keep `conf=0.20` predictions but drop tiny/low-score masks by area and score instead of confidence alone.
