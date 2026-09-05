# Next score-improvement experiments

Status (2026-09-05 completion handoff): V10 full version 2 is COMPLETE and
downloaded to `outputs/kaggle_v10_quality`. Full V6 control hash/counts and the
V8 source-control counts reproduced. The selected `v10_quality_rank_rescue40`
candidate passed the local promotion gate and external full CSV validation
(exit 0): **991 masks across 176 populated test images, not 180**.

| Canonical PQ | Control | V10 | Delta |
| --- | ---: | ---: | ---: |
| Tune | 0.4328232774 | 0.4340642398 | +0.0012409624 |
| Audit | 0.4501295961 | 0.4518370900 | +0.0017074939 |
| All files | 0.43787276745 | 0.43924560663 | +0.00137283918 |

All-file counts changed by +8 TP, +8 FP, -8 FN. All paired 95% delta intervals
include zero: this is a modest local gain, not statistically decisive evidence
or a promised public gain. Only 9 of 15 predeclared possibilities were evaluated;
component eligibility excluded six combinations. The three rejection-only gates
had zero tuning deltas; quality ordering plus conservative V8 rescue supplied
the gain. See the [execution record](v10_execution_2026-09-05.md) for exact
hashes, confidence intervals, ablations, and validation evidence.

**Unsubmitted; no V10 public score.** Preflight reported zero submissions
remaining on 2026-09-05. Preserve the public 0.38 baseline and the validated
candidate for a later authorized submission. The original phases below remain
the design record: phases 1–2 completed as this frozen-weight run; proposal-aware
training is deferred pending candidate/public evidence. No training or learned
scorer fitting occurred, and this document does not authorize auto-submission.

## Phase 0 — Evidence and documentation discovery

Verified public results: V6 original 0.35, V7 original 0.34, V8 original 0.35,
V6 + canonical COCO refiner + custom TTA 0.38 (56026663), V8 + the same refiner/TTA
0.37 (56027435). The best pipeline is V6-based, not V7-based.

Preserve the best pipeline's weights, four-view detector consensus, four-view crop
averaging, canonical COCO targets, 1024px detector input, mask threshold 0.40,
confidence 0.40, and score-ordered non-overlap cleanup as the exact control.

Full canonical validation, 141 files / 217 annotator entries:

| Metric | V6 COCO/TTA | V8 COCO/TTA |
| --- | ---: | ---: |
| PQ | 0.437873 | 0.409296 |
| TP | 891 | 869 |
| FP | 299 | 457 |
| FN | 702 | 724 |
| Matched-mask mean IoU | 0.6838 | 0.6874 |

The immediate opportunity is better detection selection, not evidence of worse
V8 mask boundaries. V8 confidence is lower on matching proposals. Identical
TTA/refinement/evaluation function bodies and identical COCO checkpoint hashes
were verified against the scripts recorded in both run manifests.

New read-only complementarity check on the 94 tuning files only: of 1,134
annotated instances, the union of four raw detector views at proposal floor 0.1
and mask IoU > 0.5 covers 790 with both detectors, 116 with V6 only, 33 with V8
only, and neither covers 195. These are per-annotator oracle coverage counts,
before grouping, refinement, rejection, or one-to-one matching; they are NOT
33 guaranteed extra true positives or an ensemble score.

### Verified interfaces and references

- `kaggle_inference/detector_tta.py:67`, `cluster_views(views, policy, match_iou=0.5)`:
  fixed-anchor, cross-view grouping; use this geometry as the starting pattern.
- `kaggle_inference/detector_tta.py:140`, `tta_prediction_variants(...)`:
  detector proposals, crop refinement, and constrained output variants.
- `kaggle_inference/optimize_inference.py:118`, `crop_probabilities(...)`:
  returns crop bounds and normal/flip-averaged refiner probabilities.
- `kaggle_inference/optimize_inference.py:153`, `clean_refined_predictions(...)`:
  mask constraints, empty-refinement fallback, score ordering, non-overlap cleanup.
- `kaggle_inference/optimize_inference.py:227`, `finetune_refiner(...)`:
  canonical-target training on original training files only.
- `kaggle_inference/optimize_inference.py:296` and `:382`, `sweep(...)` and
  `export_submission(...)`: shared validation/export behavior and caches.
- `scripts/analyze_inference_sweep.py:20`, `compare_group(...)`:
  paired filename-level comparisons, keeping annotator variants together.
- `scripts/validate_submission.py`: full RLE, ID, area, and non-overlap validation.
- Existing tests: `tests/test_detector_tta.py`, `tests/test_inference_optimization.py`,
  `tests/test_inference_comparison.py`, `tests/test_submission_validation.py`.
- Research motivation: [Mask Scoring R-CNN](https://arxiv.org/abs/1903.00241)
  distinguishes detection confidence from mask quality. Its COCO AP findings do
  not prove gains for our PQ task; a lightweight quality heuristic here is an
  experiment, not an implementation of that paper's learned head.

The make-plan workflow informed the explicit control/ablation/verification
structure. Delegation tools were unavailable; discovery was performed directly.

## Phase 1 — V10 frozen-weight precision experiment (first priority)

What to implement, after approval to execute:

1. Reuse the existing V6 proposal caches and the exact COCO refiner checkpoint.
2. Copy the current crop-probability and cleanup patterns, adding diagnostic
   fields for detector view support, refiner foreground support, and the fraction
   of the original proposal retained by refinement.
3. Evaluate the unchanged control first. Then test a small, predeclared set of
   quality-based rejection/ranking variants, each independently.
4. Include one isolated test of rejecting an empty refined mask instead of
   restoring the original detector proposal. The current fallback is not proven
   erroneous: it can preserve genuine faint filaments, so measure the tradeoff.
5. Tune on the 94 tuning files with end-to-end canonical PQ, TP, FP, and FN.

Verification:

- Reproduce the selected 0.38 control's per-image counts and submission hash
  `bf11690e9bcce73b0c627096cc1198d9fb0916650f7fa4b3f08f1167bb76bde9`.
- Test empty predictions, weak true filaments, duplicate masks, and adjacent instances.
- If quality changes ordering, redo non-overlap cleanup from untrimmed masks;
  do not re-rank already clipped cached masks and call it equivalent.
- Use identical scoring, thresholding, and cleanup in validation and export.

Guards: no detector training, no default bigger resolution, no automatic
component splitting, no arbitrary area filter that removes thin real filaments,
and no large search over the repeatedly inspected validation set.

## Phase 2 — Conservative V8 rescue, not equal-weight fusion

What to implement:

1. Preserve V6's accepted instances as anchors.
2. Reuse `cluster_views` grouping behavior and the common refiner to identify
   V8 candidates not duplicating accepted V6 instances.
3. Add only V8 candidates satisfying a separately calibrated confidence/quality
   gate. Do not assume V6 and V8 numeric confidence scores are comparable.
4. Keep V6 mask ownership when resolving overlaps; avoid allowing added weak
   V8 masks to erase accepted V6 pixels.
5. Compare control, Phase-1-only, V8-rescue-only, then a combination only if the
   individual ablations justify it. Export only a tune-selected candidate.

Verification: report incremental TP, FP, FN, and PQ, not just extra mask count;
test duplicate rejection, distinct neighboring filaments, and masks with partial
overlap. Use the same split and paired comparison as Phase 1.

Guards: never concatenate submission CSVs, union neighboring instances, or apply
equal-weight score averaging by default. The 33 raw oracle opportunities justify
a bounded test, not an assumption that an ensemble will win.

## Phase 3 — Proposal-aware COCO refiner training, if inference gains stall

Status: deferred pending V10 candidate/public evidence; not trained or launched.

What to implement:

1. Extend the existing `finetune_refiner` training pattern to include imperfect
   detector-derived crops, along with the existing ground-truth-centered crops.
2. Add conservative background negatives and positive examples of faint filaments.
   Retain separate annotator targets; do not union conflicting annotations.
3. Treat a proposal as a background negative only when it lacks meaningful
   overlap with every available annotator's instances for that image. Ignore
   ambiguous examples rather than labeling another annotator's filament negative.
4. Train only on the original training files. If adding a learned proposal-quality
   scorer, prefer out-of-fold training predictions to avoid in-sample calibration bias.
5. Select with full-image PQ, not crop Dice alone. Keep the frozen old refiner control.

Verification: assert no training/tuning/audit filename leakage; check class balance,
annotation matching and negative examples visually; compare complete-image PQ.

Guards: starting from V8's own refiner is a separate ablation, not a presumed fix.
The current COCO refiner was trained on annotation-centered crops, not V6 proposals;
its V6 initialization alone does not establish incompatibility with V8.

## Final phase — Promotion and independent confirmation

- Select all parameters on tuning files. Inspect the 47-file audit once per frozen
  candidate as a regression check, without retuning to its failures.
- Require a meaningful tuning improvement with no material audit regression;
  report paired uncertainty and object-count tradeoffs. The audit has already
  influenced earlier checkpoint choices and diagnostics, so it is not untouched.
- For a promising recipe, use a freshly trained independent fold before making
  strong generalization claims. Never call a reshuffle of already-trained-on
  images an independent validation set.
- Run focused tests, verify source/weight/split hashes, then fully validate the CSV.
- Before any authorized submission, check current submission allowance and check
  the candidate has not already been uploaded. Keep 56026663 as the baseline
  until a new public score actually exceeds 0.38.

No promised public score. Preserve the completed, validated V10 candidate while
awaiting submission allowance and authorization. Proposal-aware training, larger
detectors, longer training, and a new V8 refiner remain deferred follow-ups.
