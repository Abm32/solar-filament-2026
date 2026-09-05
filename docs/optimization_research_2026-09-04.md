# Optimization Research - 2026-09-04

## Live Score State

- Competition: `filament-segmentation-2026`
- Current best submitted score after V6: `0.35` public
- Previous accepted scores: `0.31`, `0.33`, and `0.34` public
- Visible leaderboard top at latest check: `0.56`; many leading submissions are clustered at `0.55`
- Visible rank-19 score at latest check: `0.40`
- V5 kernel status: `KernelWorkerStatus.COMPLETE`; public score `0.33`
- V6 kernel status: `KernelWorkerStatus.COMPLETE`; public score `0.35`
- V7 kernel status: `KernelWorkerStatus.RUNNING`
- Public `lamhuy8904/solar-filament-unet-segmentation-0-55` notebook reviewed on `2026-09-05`; details in `docs/public_notebook_lamhuy_055_review_2026-09-05.md`

## External Research Signals

The competition page confirms that the challenge focuses exclusively on solar filament segmentation masks, even though MAGFiLO includes extra metadata such as boxes, spines, and class labels. This supports the V6 decision to simplify YOLO to a single segmentation class.

The official evaluation rubric is not leaderboard-only. Quantitative scoring is based on Panoptic Quality plus distributions of Dice, IoU, one-to-many, and many-to-one relationships. Qualitative judging also reviews morphology, pipeline documentation, and code quality. This means the repo needs both score improvement and a reproducible, well-documented method.

Public discussion reports that many test filenames overlap with the public MAGFiLO release and that some leaderboard scores may be inflated by leakage or evaluator issues. The host response in that thread says leaderboard rank is only a preliminary filter and that final evaluation rewards a decent strategy and segmentation quality. Because of that, this project should not chase leaked annotations or empty-mask leaderboard tricks. The optimization target remains a legitimate model pipeline with strong validation diagnostics.

Useful public benchmark signal:

- A public `YOLO + U-Net` notebook reports `0.69` public score with `yolo11s-seg.pt`, `30` epochs, `imgsz=1024`, `batch=4`, `mask_ratio=2`, and a ResNet18 crop U-Net refiner.
- The current live visible leaderboard top is lower at `0.56`, suggesting the public score landscape has shifted or old scores may not be directly comparable.
- Our implementation is now close to that public architecture, so the remaining gap is more likely from data handling, postprocessing validity, training sufficiency, and hidden/public split mismatch than from needing an unrelated architecture immediately.

## Score Jump: 0.31 To 0.34

The `0.31` submission was the first valid postprocessed output after two raw submissions failed with `SubmissionStatus.ERROR`. The raw crop-refiner output could create overlapping instance masks because every crop was refined independently. Kaggle accepts one row per instance, but overlapping instance masks are unsafe for the competition evaluator/submission validator.

The `0.34` V4 submission changed three important things:

1. It fixed the validation IoU calculation.
   - Previous validation used `uint8 @ uint8` for mask intersections.
   - Any intersection larger than 255 pixels could overflow.
   - That made the reported validation PQ near zero even though the public submission scored `0.31`.
   - V4 casts masks to `int32` before matrix multiplication, producing a usable validation PQ of `0.408729`.

2. It moved non-overlap handling into the inference pipeline.
   - V3 generated masks first and required an external non-overlap cleanup.
   - V4 sorts masks by confidence and removes pixels already claimed by higher-confidence instances.
   - This makes the generated `submission.csv` valid by construction.

3. It constrained the crop refiner to the YOLO proposal support.
   - The free crop refiner can expand into neighboring filament regions.
   - V4 keeps refined pixels only inside a slightly dilated YOLO mask.
   - Validation showed `refined_constrained` was best: `0.408729` at confidence `0.25`.

The score only moved from `0.31` to `0.34` because this was mainly a correctness/precision fix, not a stronger detector. It reduced noisy overlap and false positives, but it also reduced the test submission from `1,810` rows to `1,293` rows. That likely improved precision while losing some recall.

## Current Model Anatomy

The active pipeline is:

1. Parse MAGFiLO COCO polygons.
2. Choose one annotation set per duplicated train image.
3. Convert polygons to YOLO segmentation labels.
4. Train `yolo11s-seg.pt` at `imgsz=1024`, `batch=4`, `mask_ratio=2`.
5. Train a ResNet18 encoder crop U-Net on individual filament crops.
6. Run YOLO on each test image.
7. Refine each YOLO mask crop.
8. Force score-ordered non-overlap.
9. Encode each instance as COCO compressed RLE.

## Evidence From V4

Validation sweep:

| Mode | Best confidence | Best validation PQ |
| --- | ---: | ---: |
| `yolo_only` | `0.25` | `0.379349` |
| `refined_constrained` | `0.25` | `0.408729` |
| `refined_free` | `0.25` | `0.406808` |

Detection training:

- Best mask mAP50 occurred at epoch `30`.
- Final epoch mask mAP50: `0.34306`.
- Since the best epoch was the last epoch, the model had not exhausted training.

Submission shape:

| Run | Rows | Populated test images | Missing test images | Median masks/image |
| --- | ---: | ---: | ---: | ---: |
| V3 non-overlap | `1,810` | `178` | `2` | `10` |
| V4 constrained | `1,293` | `177` | `3` | `7` |

Mask area:

| Run | Mean area | Median area | P10 | P90 |
| --- | ---: | ---: | ---: | ---: |
| V3 non-overlap | `1716.7` | `1016.0` | `272.9` | `3929.0` |
| V4 constrained | `1966.2` | `1277.0` | `437.2` | `4341.2` |

Interpretation: V4 produces fewer, larger, cleaner masks. That explains the modest public score improvement, but the leaderboard gap implies the detector still misses too many real filaments or produces masks with insufficient IoU.

## Highest-Leverage Improvement Areas

### 1. Train Longer Before Changing Architecture

V4 ended while the YOLO segmentation metric was still improving. V5 increases YOLO training to `80` epochs with patience `20`.

Expected benefit:

- better proposal recall;
- better mask contours before refinement;
- less dependence on the crop refiner to recover weak instances.

Risk:

- overfitting the small 707-image dataset;
- longer Kaggle runtime;
- if validation and public split differ heavily, local validation may overstate gains.

### 2. Use All Non-Leaking Train Annotator Variants

The dataset contains duplicated image files labeled by multiple annotators:

- Unique train files in current split: `566`
- Current best-annotator train annotations: `4,452`
- All train annotator annotations without validation leakage: `6,606`
- Gain: `2,154` annotations, or `48.4%`

Current pipeline discards these extra annotations. V6 prepares a controlled change: keep validation file-level, but train on every annotator variant for train files using unique YOLO filenames that point to the same source image.

Expected benefit:

- more examples of weak, ambiguous, and boundary-varied filaments;
- improved detector recall;
- regularization from human label variation.

Risk:

- conflicting masks on the same image may blur the training target;
- if one annotator is consistently noisier, recall may improve while precision drops.

### 3. Train As Single-Class Segmentation

Submission rows contain only `filament_id` and `segmentation_rle`. The metric does not use the categories `Left`, `Right`, `Unidentifiable`, or `Ambiguous`.

Raw category distribution:

- `Left`: `2,535`
- `Right`: `2,590`
- `Unidentifiable`: `3,074`
- `Ambiguous`: `0`

The current four-class YOLO head spends capacity predicting labels that do not affect score. V6 switches to `single_cls=True` and writes all labels as class `0`.

Expected benefit:

- simpler optimization target;
- more positive examples per class;
- fewer class-specific NMS/training artifacts.

Risk:

- if category indirectly captures morphology, removing it could slightly reduce proposal quality.

### 4. Add Per-Image Error Diagnostics

Current validation gives global TP/FP/FN but not which images fail. The next diagnostic pass should write:

- per-image PQ;
- ground-truth count;
- predicted count;
- TP, FP, FN;
- low-PQ image names;
- missing-prediction images.

This is needed before more postprocessing work, because over-splitting and under-detection require opposite fixes.

### 5. Postprocess By Recall-Preserving Filters

V4 improved score while reducing row count sharply. If V5/V6 still lag, test a lower-confidence inference mode, such as `conf=0.20`, then remove predictions by area/score heuristics after seeing per-image diagnostics.

Expected benefit:

- recover faint filaments;
- preserve precision through targeted filtering instead of global confidence pruning.

Risk:

- false positives can damage PQ quickly because each unmatched prediction contributes `0.5` FP penalty.

### 6. Try Larger YOLO Only After V5/V6

`yolo11m-seg.pt` is likely stronger, but it should come after the data/label fixes because it increases runtime and memory pressure. If V5 improves and still peaks at the final epoch, try:

- `yolo11m-seg.pt`
- `imgsz=1024`
- `epochs=50-60`
- `batch=2`
- same validation sweep and non-overlap logic

## Prepared Next Runs

### V5

Purpose: test whether YOLO was undertrained.

Changes:

- `epochs=80`
- `patience=20`
- threshold sweep `0.10-0.45`
- same single-best annotator data as V4
- same constrained refiner

Result:

- Validation: `best_conf=0.30`, `best_mode=refined_constrained`, `best_pq=0.396896`.
- Training: early-stopped after `73` epochs. Ultralytics selected best overall weights from epoch `53`; the highest raw mask mAP50 was `0.3408` at epoch `46`, and the final epoch fell to `0.30444`.
- Submission shape: `1,240` rows across `176` populated test images.
- Public score: `0.33`, below V4's `0.34`.

Interpretation: longer training did not help this exact data setup. It reduced predicted rows versus V4 and likely moved further toward precision/under-recall. Do not extend the same V5 recipe again without changing data, class setup, model size, or postprocessing.

### V6

Purpose: use more labels and remove irrelevant class prediction.

Changes prepared locally:

- train on all annotator variants for train files;
- keep validation as one best annotation set per file to avoid leakage;
- generate unique YOLO image/label names for duplicate annotator variants;
- train as single-class filament segmentation;
- keep V5 longer training and wider threshold sweep.

Result:

- Validation: `best_conf=0.40`, `best_mode=refined_free`, `best_pq=0.426821`.
- Best validation counts: `tp=617`, `fp=252`, `fn=527`.
- YOLO training: best mask mAP50 `0.59715` at epoch `76`; final epoch mask mAP50 `0.59006`.
- Submission shape: `1,094` rows across `176` populated test images.
- Public score: `0.35`, the current best submitted score.

Interpretation: all-annotator single-class training helped, especially in detector mAP and validation PQ, but the public score moved only from `0.34` to `0.35`. The chosen `conf=0.40` output is conservative, with fewer rows than V4/V5. The remaining gap is likely proposal recall, public/validation mismatch, and/or missing morphology-aware postprocessing rather than simple training duration.

### V7

Purpose: test detector capacity and reduce submission-selection risk.

Changes prepared locally:

- switch from `yolo11s-seg.pt` to `yolo11m-seg.pt`;
- reduce batch from `4` to `2` for T4 memory headroom;
- use `60` epochs and `15` patience because V5 showed the small model peaked before epoch 80;
- keep V6 all-annotator single-class training;
- write `validation_candidates.csv` so each mode/confidence pair has PQ, TP, FP, FN, precision, and recall;
- write `validation_image_diagnostics.csv` for the best validation candidate;
- write the best `submission.csv` plus near-best `submission_candidate_*.csv` files.

Status: pushed as Kaggle kernel version `7` at `2026-09-04 23:17 IST`; status `KernelWorkerStatus.RUNNING`.

Decision rule: after V7 completes, pull `validation_candidates.csv`, `validation_image_diagnostics.csv`, and all generated submission candidate diagnostics. Because only one Kaggle submission remained after V6 upload, submit V7 only if its validation PQ and diagnostics clearly dominate V6 or if a near-best candidate shows materially better recall without an extreme FP increase.

### V8

Purpose: reduce final-submission selection risk without changing the model.

Changes prepared locally:

- keep V7's `yolo11m-seg.pt`, all-annotator, single-class training recipe;
- emit top `5` validation-ranked submission CSVs instead of top `3`;
- widen the near-best validation margin from `0.015` to `0.03`;
- always include recall-hedged, precision-hedged, and best `yolo_only` candidates when available;
- add `scripts/analyze_kaggle_outputs.py` to summarize candidate metrics and submission row counts quickly after a kernel run.

Status: prepared locally only. Do not push while V7 is running. Use this if V7's limited candidate set is insufficient or if a later daily submission window allows another run.

### Public 0.55 Notebook Signal

The reviewed `lamhuy8904` notebook is not a reproducible U-Net improvement. It trains a small demonstration U-Net, then creates the high-scoring submission from an embedded compressed CSV payload of precomputed RLE masks.

The payload is still useful as a public-score shape signal:

- decoded payload rows: `1,342`;
- populated images before fallback: `178`;
- median instances per populated image: `7.0`;
- median mask area: `1,294`;
- V6 rows: `1,094`;
- V6 populated images: `176`;
- V6 median instances per populated image: `5.5`;
- V6 median mask area: `1,429`.

Interpretation: our masks are not obviously too small, but V6 is likely under-recalling instances. For V7 candidate selection, avoid spending the final daily submission on another very conservative `~1,100` row file unless validation PQ is dramatically better. A near-best validation candidate with `1,250-1,400` rows and controlled FP may be a better public-score bet.

## Decision Rules

1. If V5 score is above `0.40`, submit/keep it and then run V6.
2. If V5 score is flat near `0.34`, V6 becomes urgent because training duration alone is not enough.
3. If V6 improves validation but lowers public score, revert single-class first and keep all-annotator training for one isolated test.
4. If V6 improves public score but remains below `0.50`, try `yolo11m-seg.pt`.
5. If public score reaches `0.45+`, spend time on per-image diagnostics and targeted postprocessing before larger models.

## Sources Checked

- Kaggle competition evaluation page: `https://www.kaggle.com/competitions/filament-segmentation-2026/overview/evaluation`
- Kaggle competition leaderboard/final submission page: `https://www.kaggle.com/competitions/filament-segmentation-2026/overview/leaderboard-ranking`
- Public YOLO + U-Net notebook: `https://www.kaggle.com/code/ektarr/yolo-u-net-solar-filament-segmentation`
- Public discussion about test overlap: `https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/728062`
