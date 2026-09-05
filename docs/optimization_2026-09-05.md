# Inference optimization and annotation rasterization correction

## Confirmed baseline

Kaggle submission `56026663` (COCO refiner with detector consensus TTA) is now the
best accepted model at **0.38**, confirmed live. COCO refiner without detector TTA
(`56025722`) scored **0.36**; V6 (`56017396`) scored **0.35**.
V7 submission `56024292` scored **0.34**. Both scores were checked live in this session.
V8 completed; its original selected submission `56027025` scored **0.35**.
V8 + COCO/TTA `56027435` completed scoring at **0.37**, below the best 0.38.
V10 full version 2 subsequently completed: selected `quality_rank_rescue40`
improved local all-file PQ from 0.437873 to 0.439246, with modest gains whose
paired intervals include zero. Its 991-mask / 176-populated-image CSV passed
external validation; it is **unsubmitted, with no public score**. Preflight
reported zero submissions remaining on this date. No new weights were trained;
proposal-aware training is deferred pending candidate/public evidence. See the
[V10 execution record](v10_execution_2026-09-05.md).

## Annotation target defect

`scripts/audit_annotation_rasterization.py` checked all **8,199** training annotations:

| Rasterization | Areas exactly matching supplied annotations | Median area / supplied area | Median absolute area error |
| --- | ---: | ---: | ---: |
| COCO polygons | 8,199 / 8,199 | 1.000000 | 0 pixels |
| Rounded OpenCV polygons, used previously | 0 / 8,199 | 1.106202 | 131 pixels |

Full per-annotation results are in `outputs/annotation_rasterization/areas.csv`.
The old rasterization enlarged thin filaments, affecting both crop-refiner targets and
local evaluation. `kaggle_kernel/train_and_submit.py:annotation_mask` now uses COCO
polygon conversion and merges components belonging to the same instance. The running
V8 still uses the earlier targets; this fix applies to subsequent training runs.
Canonical evaluation retains all nonempty annotations, including those smaller than
the 20-pixel prediction filter. Historical comparison retains the old GT filter.

The exact agreement with the supplied areas supports using COCO targets. It does not
prove the implementation of the hidden competition evaluator. COCO's reference
conversion is [annToRLE/annToMask](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py).

## V9 experiment

Private kernel: `abhimanyu122/solar-filament-v9-inference`.
Private checkpoint dataset: `abhimanyu122/solar-filament-v6-checkpoints`.

The checkpoint package contains the detector, refiner, inference helper source,
SHA-256 hashes, split filenames and source annotation hash. Checkpoint metadata
records Ultralytics **8.4.138**, which is now pinned for inference and future training.

The sweep holds weights fixed and tests:

- Input resolutions: 1024, 1536 and 2048.
- Detector confidence: 0.10 through 0.45, step 0.05.
- Mask thresholds: 0.50, 0.65 and 0.80.
- Original crop prediction and an average of four flip predictions.
- Free and proposal-constrained refinement, plus detector-only masks.

This gives **312 candidates**. Detector and crop probabilities are reused across
confidence/threshold settings. Score-ordered overlap removal preserves the equality
between cached confidence cuts and independently processed predictions.
The corrected run exports the tune-selected winner and V6 control. During export,
proposals below the selected confidence are discarded before crop refinement.

The first kernel version was launched before discovering the rasterization defect.
Its results use the old targets and must not be treated as corrected evaluation.
The corrected version selects on COCO targets and additionally records the original
best-annotator/OpenCV score in `legacy_pq` to reproduce previous comparisons. A push
of the corrected version was initially rejected because both GPU slots were occupied.

Version 1 completed and was archived to `outputs/kaggle_inference_v9_legacy` before
version 2 was pushed. Its control CSV is byte-for-byte identical to accepted V6
(SHA-256 `02d88d1d0f35f1818aa315d6fa5bc50746843c2b37602195f95863d93438dbf4`).
The tune-selected candidate was 1024px, free refinement, threshold 0.50, four-view
crop averaging, confidence 0.40. On historical targets it improved all-annotator
PQ from 0.428245 to 0.431663. Audit improvement was only 0.000476, with paired
95% interval [-0.006108, 0.006915], so this is weak evidence of a useful gain.
The best historical 1536px and 2048px candidates scored 0.420024 and 0.373842.

Version 2 completed and is archived at `outputs/kaggle_inference_v9`. On canonical
targets the tune-selected configuration is 1024px/free/0.65/four-view averaging/
confidence 0.40. PQ is 0.422512 versus 0.420713 for the original V6 settings.
Audit delta is -0.000019 (95% interval [-0.004810, 0.004114]). The all-file interval
also includes zero, so no public submission was spent on this weak inference-only
gain. Canonical best-per-resolution PQ is 0.422512, 0.410397, 0.366374 respectively.

Selection uses 94 of the 141 validation files. The remaining 47 are an audit of the
new inference choices. They were previously used for V6 checkpoint selection, so
this is **not** an untouched holdout. All annotators for a filename stay in the same
partition and are evaluated separately. `scripts/analyze_inference_sweep.py` uses
paired bootstrap resampling by filename, keeping annotator variants together.

## Runtime improvement

Refinement now resolves overlap within each crop and encodes its location directly
into full-image COCO RLE. This avoids creating and scanning full 2048×2048 arrays for
every mask/threshold variant. Tests compare decoded outputs against the dense
implementation, including overlaps, empty-refinement fallback, edge crops and
full-height runs. `outputs/postprocessing_benchmark.json` records a local synthetic
benchmark; it measures CPU postprocessing, not end-to-end GPU speed.

The corrected run also writes compressed prediction caches for later analysis
without rerunning the detector. Its manifest distinguishes COCO scores from the
historical OpenCV scores.

Measured full validation time fell from **2225.8 seconds to 534.2 seconds** (4.17×).
Both runs covered all 312 configurations; the new run additionally evaluated COCO
targets and saved compressed predictions. Every configuration's historical PQ,
TP, FP and FN remained exactly equal, and the V6 control CSV retained the original
SHA-256. The separate synthetic CPU postprocessing benchmark measured 23.32×.

## Reproduction

```bash
.venv/bin/python scripts/audit_annotation_rasterization.py
.venv/bin/python scripts/prepare_inference_source.py
.venv/bin/python -m pytest tests/test_inference_optimization.py tests/test_inference_comparison.py tests/test_kernel_metrics.py tests/test_rle.py -q
# After GPU output download:
.venv/bin/python scripts/analyze_inference_sweep.py outputs/kaggle_inference_v9
```

Do not regenerate the checkpoint dataset during an active experiment: each run's
manifest records the source hashes actually used. The lightweight local environment
does not contain PyTorch; the existing full `test_pq_eval.py` suite requires it.

## Canonical refiner fine-tune

Prepared by `scripts/prepare_refiner_experiment.py`, using the shared inference
implementation with four fine-tuning epochs and a 1024px sweep. It starts from V6's
refiner, trains only on the original 566 training files, uses COCO masks and the
existing crop augmentation, and uses AdamW at 5e-5 with cosine decay. The detector
is frozen. Epoch zero is retained as a fallback; epochs are selected by crop soft
Dice on the 94 tuning files. Final selection and reporting still require end-to-end
PQ and the 47-image audit. No audit file participates in fine-tuning or epoch selection.

Use `scripts/analyze_inference_sweep.py <fine-tuned-output> --baseline-dir <version-2-output>`
to compare with the previously tune-selected model. The analysis refuses mismatched
partitions/target definitions and requires an external baseline for fine-tuned weights.

Launched as private kernel `abhimanyu122/solar-filament-coco-refiner`, version 1.

### Completed fine-tune evaluation

Kaggle confirmed COMPLETE. Checkpoint, metrics, submissions and logs are archived
in `outputs/kaggle_coco_refiner`; the separate metrics download and comparison
report are in `outputs/kaggle_coco_refiner_metrics`.
Epoch 3 had the highest tuning crop soft Dice (0.803138 versus 0.795638 at epoch 0).
The tune-selected end-to-end candidate is 1024px, constrained refinement,
threshold 0.50, four-view averaging, confidence 0.40.

| Canonical PQ | V6 control | V9 selected | COCO fine-tune |
| --- | ---: | ---: | ---: |
| Tune (94 files) | 0.416407 | 0.418946 | 0.420230 |
| Audit (47 files) | 0.431289 | 0.431270 | 0.435251 |
| All (141 files) | 0.420713 | 0.422512 | 0.424581 |

Paired filename bootstrap versus V9: audit delta +0.003980, 95% interval
[0.001142, 0.008151]; all-file delta +0.002069, interval [-0.000451, 0.004578].
Versus V6: audit delta +0.003961, interval [-0.000997, 0.008099]; all-file delta
+0.003868, interval [0.001180, 0.006697]. This is a modest promising gain,
not evidence of a large leaderboard improvement. The audit was already used in
V6 checkpoint selection. Historical OpenCV PQ regresses to 0.424471 from V6's
0.426821; canonical and historical metrics must remain distinguished.

The exported submission has 1087 instances across 176 test images and passed full
RLE decoding, ID, area and non-overlap checks. It was submitted on user approval
as `56025722` at `2026-09-05 05:38:51.393000 UTC`; final status is COMPLETE,
with public score **0.36** (versus V6's 0.35).
Kaggle reported three submissions remaining today. Submitted SHA-256:
`22e36c30e2953cf6f1fec4d9cccc5a2299a8348e4f42ef64a9b4960fb3d91e43`.
V8 was checked separately and remains RUNNING; it is not a dependency for this candidate.

## Detector flip and threshold experiment

Private kernel `abhimanyu122/solar-filament-detector-tta`, version 1,
explicitly requested T4 and is COMPLETE. V8 has also completed; its original
submission is documented below.
Local verification: 32 focused tests passed, covering geometry, grouping, control
reproduction, component splitting, sparse RLE postprocessing and paired comparison.
The lightweight environment still lacks PyTorch for the unrelated full evaluator suite.

The accepted canonical configuration has 903 TP, 418 FP and 690 FN across all
held-out annotator entries. Its winning mask threshold (0.50) was at the boundary
of the previous grid. This motivates two inference ablations with frozen weights:

- Local threshold calibration: 0.35 through 0.60, step 0.05.
- Four detector views: identity, horizontal flip, vertical flip, both flips.
  Restore masks to original coordinates, group cross-view duplicates at IoU >= 0.5,
  retain the strongest proposal mask, then use either maximum confidence or sum
  of member confidences divided by four (missing views contribute zero).

Including the original single-view detector yields 126 settings with confidences
0.25 through 0.55, step 0.05. Crop refinement stays constrained with four-view crop
averaging at 1024px detector resolution. Grouping cannot merge two proposals from
the same view, and matches fixed anchors to avoid chains across neighboring objects.
The inference API restores full-image masks with `retina_masks=True`; see the
[Ultralytics segmentation predictor source](https://docs.ultralytics.com/reference/models/yolo/segment/predict).
The grouping/confidence policies are our experimental choices, not a documented
guarantee of improvement.

The runner validates the refiner SHA-256, original source hashes, annotation hash,
and exact tune/audit partitions. It must reproduce every accepted-control per-image
PQ count and the accepted submission SHA-256 before its exports are considered valid.
All detector proposals are cached at a fixed confidence floor of 0.1; confidence
filtering happens after grouping and refinement, identically in validation and export.
The winner is selected only on tuning PQ. No additional public submission has been
made; compare the paired audit before deciding whether another upload is useful.

Reproduce staging and analyze outputs:

```bash
.venv/bin/python scripts/prepare_detector_tta.py
.venv/bin/python -m pytest tests/test_detector_tta.py tests/test_inference_comparison.py -q
# Once the completed artifacts are downloaded:
.venv/bin/python scripts/analyze_inference_sweep.py outputs/kaggle_detector_tta --baseline-dir outputs/kaggle_coco_refiner
```

### Completed detector-TTA evaluation

Artifacts are archived in `outputs/kaggle_detector_tta`. All 358 original-control
diagnostic rows match the accepted COCO run, and `submission_control.csv` has the
identical SHA-256 `22e36c30e2953cf6f1fec4d9cccc5a2299a8348e4f42ef64a9b4960fb3d91e43`.
The run log ends at 1458.1 seconds (about 24.3 minutes, including validation,
control/candidate test exports and notebook conversion).

The tune-selected winner uses four-view detector consensus, mask threshold 0.40,
confidence 0.40, constrained refinement and crop flip averaging at 1024px.

| Canonical PQ | Accepted COCO refiner | Detector consensus | Delta |
| --- | ---: | ---: | ---: |
| Tune (94 files) | 0.420230 | 0.432823 | +0.012593 |
| Audit (47 files) | 0.435251 | 0.450130 | +0.014879 |
| All (141 files) | 0.424581 | 0.437873 | +0.013292 |

Paired filename bootstrap 95% intervals: tune [-0.001779, 0.026337], audit
[-0.010319, 0.038885], all files [0.000384, 0.025455]. The audit gain is encouraging
but uncertain; it wins on 26 audit files and loses on 21. The combined interval
includes tuning files used to select among 126 settings and is not an independent
confirmation. The audit was also used in original V6 checkpoint selection.

Across all annotator entries, TP changes 903 -> 891, FP 418 -> 299 and FN 690 -> 702.
The main gain is a reduction of 119 false positives (28.5%), with 12 fewer matches.
Historical OpenCV PQ also improves, from 0.424471 to 0.434366.
The best single-view calibration has all-file PQ 0.425431, while maximum-confidence
flip grouping has 0.424615; their audit PQ values are 0.434976 and 0.427617.
This suggests that agreement-based confidence contributes more than flips alone.

The candidate export has 982 instances across 176 test images, SHA-256
`bf11690e9bcce73b0c627096cc1198d9fb0916650f7fa4b3f08f1167bb76bde9`.
Full RLE decoding, ID, minimum-area and non-overlap validation passed;
the smallest mask has 27 pixels. Report: `outputs/kaggle_detector_tta/submission_validation.json`.
Submitted on user approval as `56026663` at `2026-09-05 06:36:46.980000 UTC`.
Final status is COMPLETE with public score **0.38**, up from 0.36.
Two ordinary CLI uploads stalled at zero bytes and were cancelled;
submission listings verified that neither created a submission. A bounded IPv4
upload returned HTTP 200 in 1.9 seconds and successfully created this one submission.
The running V8 kernel did not prevent submission and was not modified.

## V8 original submission

Kernel `abhimanyu122/solar-filament-yolo-refiner/8` completed. Its original outputs
were downloaded explicitly from version 8 into `outputs/kaggle_kernel_v8`.
The selected original configuration is constrained refinement, confidence 0.25,
refiner threshold 0.65, with 1124 predicted masks across 175 test images.
Full RLE decoding, ID, minimum-area and non-overlap checks passed; minimum area 21px.
SHA-256: `2a843dac7f03877a3ad8397ba93a4a04a1c7e4b1871c3ccbc45219c1b621a0a6`.

Submitted as requested before COCO/TTA changes: `56027025`, timestamp
`2026-09-05 07:00:24.773000 UTC`, final status COMPLETE, public score **0.35**. Bounded IPv4 upload
succeeded (HTTP 200 in 2.3 seconds).

V8's own validation reports PQ 0.402650, TP 858, FP 501 and FN 734 over 217
annotator entries. This run still used historical rounded-OpenCV targets and
the old small-GT filter; do not compare that PQ directly with canonical TTA PQ.
The next combination requires V8 detector weights, canonical evaluation, and
threshold calibration against the accepted 0.38 pipeline.

## V8 detector with accepted COCO refiner and TTA

User explicitly requested generating and submitting this combination.
`abhimanyu122/solar-filament-v8-coco-tta`, version 1, completed on T4. The runner pairs the
V8 detector with the existing canonical refiner (`8d008053...d01fb1`) and uses
four-view detector consensus plus four-view crop averaging. It does not retrain
the detector or refiner. The selected configuration uses tuning PQ only.

- V8 detector SHA-256: `19c713e327e7bf39e839d428712d243d25ce746628237663a64abbf58f6e9359`.
- V8 manifest SHA-256: `5cc17143eceed96ccba9a13b662641fc2be77d660a02920b7f1c459afe42606f`.
- Verified 566 training and 141 held-out files, including all 217 held-out annotator entries.
- V8 was trained with Ultralytics 8.4.140; this run pins 8.4.140 instead of the V6 helper's 8.4.138.
- 48 settings: mask thresholds 0.35–0.60 and confidences 0.10–0.45, both step 0.05.
- Prediction caches are separated by detector SHA-256. Refiner/source/annotation hashes are checked.
- 36 focused tests passed, including rejection of split leakage, missing annotators,
  wrong package version, changed weights and unknown image IDs.

Staging: `.venv/bin/python scripts/prepare_detector_tta.py --v8`.
Live progress was monitored with `kaggle kernels logs -f abhimanyu122/solar-filament-v8-coco-tta`.
Outputs and comparison reports are archived in `outputs/kaggle_v8_coco_tta`.
Reproduce comparison with
`.venv/bin/python scripts/analyze_inference_sweep.py outputs/kaggle_v8_coco_tta --baseline-dir outputs/kaggle_detector_tta`.

Selected configuration: detector consensus, constrained COCO refiner with crop TTA,
mask threshold **0.45**, confidence **0.25**, 1024px detector input.

| Canonical PQ | Accepted 0.38 pipeline | V8 COCO/TTA | Delta |
| --- | ---: | ---: | ---: |
| Tune | 0.432823 | 0.401104 | -0.031720 |
| Audit | 0.450130 | 0.429404 | -0.020725 |
| All files | 0.437873 | 0.409296 | -0.028577 |

The paired audit interval is [-0.044842, 0.005686]; the all-file interval is
[-0.040278, -0.017163]. The new combination is weaker on validation. Canonical
counts are TP 869, FP 457, FN 724. The user was informed before upload; submission
proceeded under the explicit request to submit V8 with COCO/TTA.

The generated CSV contains 1070 masks across 175 images. Full RLE decode, ID,
minimum area (23px), and non-overlap checks passed. SHA-256:
`370a9a3c56876ca44dd4202cc649a4ac3d334357e5387fae4c4ffa387be8ee71`.
Submitted as **56027435**, timestamp `2026-09-05 07:24:56.903000 UTC`, final
status COMPLETE with public score **0.37**. Bounded IPv4 upload returned HTTP
200 in 2.0 seconds. The best known public score remains **0.38**; this V8
combination did not beat it.

## Cached component-splitting ablation (rejected)

Three tuning-image overlays in `outputs/coco_failure_samples` show missed faint
structures and mismatched instance grouping. The diagnosis covers selected difficult
images, not a representative error-rate estimate. A prediction in
`20160411134234Ch.jpeg` contains multiple substantial disconnected components.
This motivated an isolated splitting experiment on all 141 held-out files.

`scripts/analyze_component_split.py` evaluates the accepted masks unchanged versus
8-connected components with minimum area 20px, 10% of original area (and 20px), or
25% (and 20px). Relative policies retain the original mask unless at least two
components qualify. The original per-image canonical counts match E014 exactly.

| Policy | Tune PQ | Audit PQ | All PQ |
| --- | ---: | ---: | ---: |
| Accepted masks unchanged | 0.420230 | 0.435251 | 0.424581 |
| Split components >=25% | 0.414798 | 0.428745 | 0.418833 |
| Split components >=10% | 0.410060 | 0.424600 | 0.414249 |
| Split components >=20px | 0.388756 | 0.395064 | 0.390596 |

Every splitting policy regresses. Retain intact masks; no submission generated.
Results and full cached validation masks are in `outputs/coco_component_analysis`.
Reproduction: `.venv/bin/python -m scripts.analyze_component_split outputs/coco_component_analysis/prediction_cache`.
