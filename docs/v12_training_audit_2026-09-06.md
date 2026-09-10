**V12 training audit — 2026-09-06**

This is a preparation handoff for a conditional V12 experiment after V11
results. No V12 training, Kaggle write, submission, or large download was
performed during this audit. The proposed configuration below has not been
smoke-tested on a GPU and has no demonstrated score gain.

The repository root for every local path below is
`/home/abhimanyu/Desktop/digital_agency/kaggle/solar/solar-filament-2026`.
The experiment sequence and its selection limitations are described in the
[score research report](score_050_research_2026-09-06.md).

**Observed V6 configuration and provenance**

The authoritative saved run arguments are
[`outputs/kaggle_kernel_v6/runs_filament/yolo_seg_optimized/args.yaml`](../outputs/kaggle_kernel_v6/runs_filament/yolo_seg_optimized/args.yaml).
They establish:

| Setting | Observed V6 value | Saved argument lines |
| --- | --- | --- |
| Task and architecture | Segmentation, `yolo11s-seg.pt` | 1–3 |
| Training duration | 80 epochs, patience 20 | 5–7 |
| Global batch and image size | 4, 1024 | 8–9 |
| GPU selection | `0,1` | 13 |
| Optimizer argument | `auto` | 20 |
| Seed and determinism | 42, true | 22–23 |
| Classes | Single class | 24 |
| Schedule | Linear, `cos_lr=False`, `lrf=.01` | 26, 76 |
| Mosaic shutdown | Last 5 epochs | 27 |
| Resume and AMP | False, true | 28–29 |
| Multiscale | Disabled | 33 |
| Target masks | `overlap_mask=False`, `mask_ratio=2` | 36–37 |
| Weight decay | `.0005` | 78 |
| Warmup duration | 3 epochs | 79 |
| Nominal accumulated batch | `nbs=64` | 95 |

The original augmentation arguments are `hsv_h=0`, `hsv_s=0`, `hsv_v=.3`,
`degrees=10`, `translate=.1`, `scale=.3`, `shear=0`, `perspective=0`,
`flipud=0`, `fliplr=0`, `bgr=0`, `mosaic=.3`, `mixup=0`, `cutmix=0`, and
`copy_paste=0` (saved argument lines 96–110). The general YAML also contains
classification settings such as `auto_augment` and `erasing`; their presence
does not establish that classification augmentation was applied to this
segmentation run.

[`results.csv`](../outputs/kaggle_kernel_v6/runs_filament/yolo_seg_optimized/results.csv)
contains 80 completed epochs. Its last recorded elapsed time is 5,904.68
seconds. Epoch 76 has the highest recorded sum of box and mask mAP50–95
(`.36179 + .18655`), consistent with the detector's built-in segmentation
fitness criterion. This is not canonical competition PQ.
[Pinned metric implementation](https://github.com/ultralytics/ultralytics/blob/v8.4.138/ultralytics/utils/metrics.py#L1247)

The version is independently present in both checkpoint pickle metadata and
[`v6_provenance.json`](../outputs/v6_inference_source/v6_provenance.json):
Ultralytics **8.4.138**. The provenance identifies Kaggle kernel version
`abhimanyu122/solar-filament-yolo-refiner/6`, seed 42, and historical
checkpoint selection by detector mAP and refiner Dice on these validation
filenames. That historical public submission scored `.35`; later frozen
inference/refiner changes produced the `.38` control.

**Recorded arguments differ from the effective optimizer**

The YAML's `lr0=.01` and `momentum=.937` are defaults ignored by
`optimizer=auto`. For the V6 dataset, the pinned implementation selects
AdamW with LR `.002` and betas `(.9, .999)`; it also changes warmup bias LR
to zero. Nominal weight decay remains `.0005` with the original accumulated
batch configuration. The saved checkpoint arguments contain the changed
warmup bias value. The recorded LR sequence—`.00066383`, `.00131403`,
`.00194773`, `.00192575` for epochs 1–4—corroborates the effective LR.
[Pinned optimizer implementation](https://github.com/ultralytics/ultralytics/blob/v8.4.138/ultralytics/engine/trainer.py#L1025)

There is no original V6 run log available locally. The optimizer conclusion
combines version-pinned implementation, actual sample counts, checkpoint
metadata, and the recorded LR sequence; it is not a claim to have read that
missing log.

**Exact checkpoint and source fingerprints**

The following SHA-256 values were recomputed locally during this audit:

| Repository-relative file | SHA-256 |
| --- | --- |
| `outputs/v6_inference_source/v6_provenance.json` | `ceb09f980d5be195db35f2954c26fd19ea7ffc0f318b2bae5b7c803504b17122` |
| `outputs/v6_inference_source/solar_base.py` | `f0e9e8d1e54c05908692118876f24c4caae46e02669070a5efc8a392232e57e5` |
| `outputs/v6_inference_source/yolo_best.pt` | `0abf568f5ffa504979a44c6ac4997c231adc38cb3c6d998461295c4dad3a38b8` |
| `outputs/kaggle_kernel_v6/yolo_best.pt` | `0abf568f5ffa504979a44c6ac4997c231adc38cb3c6d998461295c4dad3a38b8` |
| `outputs/kaggle_kernel_v6/runs_filament/yolo_seg_optimized/weights/best.pt` | `0abf568f5ffa504979a44c6ac4997c231adc38cb3c6d998461295c4dad3a38b8` |
| `outputs/kaggle_kernel_v6/runs_filament/yolo_seg_optimized/weights/last.pt` | `0635783a3d8e4c43a456f660c09ae6d6f7b457814e00f9ae8ecb32996f2ffb96` |
| `outputs/kaggle_kernel_v6/runs_filament/yolo_seg_optimized/args.yaml` | `5b712f3dadf94b98ae9a43e30d8a662151439efd7e9e6c9a1bace1f89a9158b0` |
| `outputs/kaggle_kernel_v6/runs_filament/yolo_seg_optimized/results.csv` | `f159190b96efa9ec47be05335e1545c4f02f71c259be92c7c75c94a5bc65b453` |
| `outputs/kaggle_kernel_v6/yolo_dataset/labels/train.cache` | `2f43a3b721c46ec59cf94f02357e9c3ed4acfb1316f3a8cb003225d921e63f84` |

The canonical training annotation path is
`data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json`.
Its provenance-recorded SHA-256 is
`5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1`;
recheck it as part of a future launch preflight.

The frozen `solar_base.py` is an inference helper source, **not the historical
V6 training recipe**. Its lines 38–54 contain V8-era defaults:
`yolo11m-seg.pt`, batch 2, 60 epochs, patience 15, and horizontal/vertical
flips `.5`. Its `main()` also labels its manifest as V8. The file's matching
provenance hash does not make those training defaults appropriate for V12.
Initialize from the pinned V6 checkpoint and supply explicit training
arguments.

Checkpoint inspection used the ZIP container's `data.pkl` with
`pickletools.genops`; it did not execute checkpoint objects or require
PyTorch. Both V6 best and last checkpoints contain `epoch=-1`,
`best_fitness=None`, `optimizer=None`, and `scaler=None`. The best checkpoint
records date `2026-09-04T14:07:55.496426+00:00`; last records
`2026-09-04T14:12:42.355885+00:00`. These are stripped export checkpoints.
They support weight warm-starting, not recovery of the original optimizer
state. **Use `resume=False`.**

**Dataset reconstruction**

Use the exact `training_files` and `validation_files` lists in
`outputs/v6_inference_source/v6_provenance.json`. They represent the grouped
split produced by shuffling sorted filenames with `random.Random(42)` and
reserving `round(.2 * 707) = 141` filenames.

| Partition | Unique filenames | Annotator image entries | Instances |
| --- | ---: | ---: | ---: |
| Training | 566 | 937 | 6,606 |
| Canonical external validation | 141 | 217 | 1,593 |
| All permitted training data | 707 | 1,154 | 8,199 |

The saved YOLO `train.cache` was inspected locally: its `results` tuple is
`(937, 0, 0, 0, 937)`, it contains 937 labels covering 566 original filename
stems, and all 6,606 classes are zero. Its warning list is empty. The
cache version is `1.0.4`; its internal dataset hash is
`fc320abd9009d1b3133d8c0a534a67549f548d12f0f9014491ecea4367ac4b97`.

The downloaded export directory is incomplete: only 391 training label
text files and no validation label text files are present. Its images also
refer to Kaggle paths. **Do not use this directory directly for training.**
Regenerate the export from the complete annotations and original JPEGs.

The existing exporter in
[`kaggle_kernel/train_and_submit.py`](../kaggle_kernel/train_and_submit.py)
shows the required scheme: normalized polygon coordinates with six decimal
places (lines 107–111), unique `_ann<image_id>` names for training annotator
variants (114–139), and filename grouping before assigning partitions
(142–175). All 8,199 source annotations were inspected and currently contain
one polygon each. Preserve this export behavior for the resolution comparison;
do not simultaneously change instance sampling or polygon handling.

The exporter's built-in detector validation uses one chosen annotator per
filename, maximizing annotation count then total area. Canonical external PQ
must evaluate all 217 annotator entries separately. Never union competing
annotator masks. The exporter deletes `CFG.yolo_dir` before rebuilding, so
configure a new isolated V12 directory rather than reusing historical outputs.

**Proposed matched-budget screening configuration**

The following values are a new conservative screening proposal, not observed
V6 settings or a validated optimum. In particular, the 20-epoch duration,
lower LR, shorter warmup, and final LR ratio are deliberate continuation
choices applied identically to both arms.

```python
common = dict(
    epochs=20,
    patience=0,
    resume=False,
    optimizer="AdamW",
    lr0=1e-4,
    lrf=0.1,
    momentum=0.9,
    weight_decay=5e-4,
    cos_lr=False,
    warmup_epochs=1.0,
    warmup_bias_lr=0.0,
    nbs=64,
    seed=42,
    deterministic=True,
    single_cls=True,
    mask_ratio=2,
    overlap_mask=False,
    amp=True,
    rect=False,
    multi_scale=0.0,
    fraction=1.0,
    freeze=None,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.3,
    degrees=10,
    translate=0.1,
    scale=0.3,
    shear=0.0,
    perspective=0.0,
    fliplr=0.0,
    flipud=0.0,
    bgr=0.0,
    mosaic=0.3,
    close_mosaic=5,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
    save_period=5,
)
```

Initialize the 1024 and 1536 arms independently from the pinned V6 best
checkpoint. Do not initialize one from the other's trained weights. Use
separate run directories. Keep the duration fixed; equal wall-clock limits
would give the slower resolution fewer optimization updates. Saving every
five epochs supports diagnosis; predeclare the comparison checkpoint policy
before inspecting scores rather than expanding the selection search afterward.

First smoke-test 1536 with global batch 4 on two T4 GPUs. If it does not fit,
use global batch 2 **for both arms**. Keep GPU count and local microbatch
identical between arms. With `nbs=64`, batch 4 accumulates 16 microbatches
after warmup; batch 2 accumulates 32. Different microbatches still change
BatchNorm statistics and warmup update counts. Record actual optimizer steps,
LRs, sample counts, and final batch sizes; matching `nbs` alone is insufficient.
[Pinned accumulation and warmup implementation](https://github.com/ultralytics/ultralytics/blob/v8.4.138/ultralytics/engine/trainer.py#L258)

Pin and log optional augmentation dependencies. The version-pinned
segmentation pipeline optionally inserts Albumentations Blur, MedianBlur,
ToGray, and CLAHE with probability `.01` each. Adjacent V5 and V8 run logs
show them active, but the absent V6 log prevents direct confirmation for V6.
Keep the same explicitly recorded transforms in both new arms. The YAML's
augmentation list alone does not capture these optional transforms.
[Pinned augmentation implementation](https://github.com/ultralytics/ultralytics/blob/v8.4.138/ultralytics/data/augment.py#L1898)

**Evaluation and acceptance**

Evaluate each trained checkpoint at both 1024 and 1536 inference resolution:
a small 2×2 comparison separates the effect of training resolution from the
effect of inference resolution. Keep raw-proposal caching keyed by checkpoint,
resolution, and inference configuration so historical caches cannot be reused
accidentally. V9 changed inference resolution alone and does not answer this
training comparison.

Report direct detector segmentation and an unchanged refiner/postprocessing
pipeline separately. Freeze whichever refiner is selected after V11 before
starting this detector comparison. Keep four-view transforms, proposal
grouping, thresholds, support constraints, minimum prediction area, and pixel
ownership fixed for the primary comparison. Any recalibration should be a
small predeclared tuning-only follow-up applied equally to both arms.

Use canonical global PQ, separate annotator entries, strict IoU `> .5`, and
all nonempty ground-truth instances. Preserve the established 94 tuning / 47
audit filenames. The old audit has already influenced earlier selections;
it is reused screening evidence, not an untouched holdout. Built-in detector
mAP is useful diagnostics but does not replace final-image PQ. The research
report's `+.01` tuning-PQ and nonnegative audit-delta screen is a practical
heuristic, not a confidence guarantee or evidence of a public `.50` score.

Only consider 2048 after a useful 1536 result and a successful memory/runtime
smoke test. If 2048 requires another batch size or GPU topology, compare it
with a control using that same topology. Native tiling is a separate branch
requiring cut-instance and duplicate handling; it should not be introduced
inside this resolution-only comparison.

**Checks required before a V12 launch**

1. Recheck annotation and checkpoint hashes, Ultralytics version, and the
   exact training/validation filename lists.
2. Assert 937 training examples, 566 unique training filenames, 6,606 class-zero
   instances, and no filename overlap with the 141 validation filenames.
   Verify the external evaluator sees all 217 validation annotator entries.
3. Verify the loaded architecture is `yolo11s` segmentation with `nc=1`,
   checkpoint weights actually load, and `resume=False` is explicit.
4. Smoke-test dense-instance training and validation batches at the intended
   resolution and global batch. Record peak memory, finite losses, successful
   backward/optimizer steps, and resolved batch/augmentation settings.
5. Assert equal training sample counts, GPU topology, effective batch,
   optimizer-step schedules, and checkpoint cadence between arms. Treat
   automatic batch reduction or an OOM retry as a changed experiment.
6. Reproduce the frozen canonical control before comparing new predictions;
   verify cache provenance and the final ownership/evaluation rules.

**Fresh-fold confirmation and the final all-707 fit**

After selecting a recipe, confirmation folds must exclude the same filenames
from both detector and refiner training. A V6 checkpoint has already seen many
possible new-fold images and cannot initialize an allegedly unseen fold.
Start confirmation models from the permitted common pretrained initialization,
with all subsequent data-dependent fitting restricted to their training folds.

Freeze the selected hyperparameters, duration, checkpoint policy, and
postprocessing before the final fit. Regenerate 1,154 annotator examples /
8,199 instances over all 707 filenames. Prefer recreating the selected full
training schedule from the common pretrained initialization, followed by its
selected continuation. Simply adding 20 epochs to the old V6 checkpoint gives
the formerly excluded images much less training exposure than the original
566; it would be a different adaptation recipe and should be labeled as such.

Refit the selected refiner on all permitted training filenames as well,
preserving the validated proposal-generation and target rules. Training-only
out-of-fold proposals remain preferable where the selected recipe requires
realistic detector errors. Once the previous validation images enter training,
their score is a training-set diagnostic, not held-out evidence. The final
submission must retain its own hashes and provenance; this audit does not
authorize bypassing the V11 result gate or imply a score improvement.
