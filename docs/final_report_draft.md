# Final technical report: DRAFT (2026-10-10)

Draft for the Solar Filament Segmentation Challenge 2026 (`filament-segmentation-2026`).
Not submitted, not published. Deadline: 2026-11-15 06:00 UTC.

**Host format (Overview > Final Submission, Evaluation, Prizes).** The report is 4 pages, uses the host's
Overleaf template (https://www.overleaf.com/read/vjztpvrnbrdh) and is submitted as one PDF through the
Google form, with a public repo link. The template's own section list is not saved locally:
**[TBD: open the template and map these sections onto it]**. The Prizes page asks the report to cover
"data preprocessing, feature engineering (if applicable), model architecture, training procedure,
post-processing steps", in enough detail to reproduce the result. Rubric: 70% quantitative (PQ, plus the
Dice distribution, the IoU distribution, and the distribution of one-to-many / many-to-one relations)
and 30% qualitative (pipeline description, apparent morphology on H-alpha, code quality). The abstract
also mentions fine structures (barbs), fragmentation/over-merging, and end-to-end compute efficiency.

Convention: every number cites its source doc in `docs/`. **[TBD]** marks anything not yet measured or
not documented locally. "OOF" = out-of-fold. "Refined" = detector + crop refiner (the deployed path).

---

## 1. Problem and data

The task is instance segmentation of solar filaments on 2048x2048 grayscale GONG H-alpha JPEGs (MAGFiLO).
The score is Panoptic Quality (PQ) with strict IoU > 0.5 matching. Train data: 707 files, 1154 annotator
records, 8199 annotations. Test: 180 images (`SESSION_CONTEXT_2026-10-06.md` §1). Records per file:
411 files x1, 145 x2, 151 x3, so 64% of the metric weight is on multiply-annotated images
(`v69_kaggle_metric_audit_2026-10-10.md` §2).

**Scoring unit = annotator record.** Each prediction is matched separately against every record of its
image, and TP/FP/FN are pooled over all (image, record) pairs (`v69_kaggle_metric_audit` §2;
`v69_error_analysis_2026-10-10.md` §0). A filament drawn by k of n annotators and predicted at IoU s
gives k TP and (n-k) FP. Predicting it pays off roughly when k/n > PQ/(2s) ≈ 0.31 (`v69_error_analysis` §0).
The target is therefore "would a random annotator draw this", not "is this a filament". Our local scorer
matches the host notebook exactly: identical TP/FP/FN on all 707 files, PQ 0.4060209295 on both
(`v69_kaggle_metric_audit` §3).

**Sparse, inconsistent labels set a low ceiling** (`v69_error_analysis` §3, 296 files with ≥2 records):

| predictor, scored against a held-out annotator record | PQ | SQ |
|---|---|---|
| another single annotator (1196 ordered pairs) | **0.340** | 0.634 |
| union of the other annotators (leave-one-out) | 0.345 | 0.630 |
| **our direct detector (p2first OOF), same records** | **0.411** | 0.656 |

The model already beats a held-out human by about 0.066 PQ. On multi-record files, 66% of FN are
filaments that only one annotator drew, and recall is 0.293 on those versus 0.783 on filaments matched in
another record (`v69_error_analysis` §2). Human boundary agreement (SQ ≈ 0.63) caps the realistic
mask quality, and the refined pipeline's SQ of 0.689 is already above it (`v69_error_analysis` §1).

## 2. Method

The pipeline has three stages, in this order: a detector, a 4-flip consensus, then a crop refiner.

**2.1 Detector: YOLO11s-seg with a P2 level and stride-2 prototypes ("p2first").** Source:
`SESSION_CONTEXT` §7, `scripts/v62_arch.py`, `configs/yolo11s-p2first-seg.yaml`.
- We append Upsample(layer 16) → Concat(backbone P2) → C3k2[128] to the stock yolo11-seg graph. The
  Segment head reads [P2, P3, P4, P5].
- Ultralytics builds mask prototypes from the first input. Putting P2 first gives stride-2 prototypes
  (512x512 at 1024 input) and adds detection at stride 4.
- Init: COCO YOLO11s-seg. Head tensors are remapped because the inserted layers shift the head index
  (509/639 tensors transfer at nc=1). The cls branch cannot transfer, which is a disclosed init
  disadvantage (`scripts/v62_init.py`).
- The validator's hard-coded 4x prototype scale is patched to 2x (`scripts/v62_patch.py`). One GPU per
  run is required, because DDP workers lose the in-process patch.
- Recipe (= V17/V21; `SESSION_CONTEXT` §5): 80 epochs at mask_ratio=2, then 20 epochs at mask_ratio=1
  (AdamW 1e-4); imgsz 1024; batch 4; single class; final fixed checkpoint.
- Pre-processing: none beyond Ultralytics resizing **[TBD: confirm and state normalisation/letterbox
  exactly]**. The final detector is retrained on all 707 files (`p2first_fold99`, `scripts/run_v65.sh`).

**2.2 4-flip consensus and pixel ownership.** Source: `SESSION_CONTEXT` §3/§5,
`v66_preregistration_2026-10-07.md`.
- Inference runs on the identity, horizontal, vertical and both-axes flips.
- Masks are clustered across views with the `flip_consensus` rule (`detector_tta.cluster_views`;
  anchor = highest-score mask, match IoU 0.5). A cluster's score is the sum of its view scores divided
  by the number of views.
- Score-ordered pixel ownership makes the predictions pixel-disjoint, because the host rejects
  overlapping instances (`v69_kaggle_metric_audit` §2, §5).

**2.3 Crop refiner (r3 recipe).** Source: `v73_refiner_preregistration_2026-10-10.md`,
`v63_preregistration_2026-10-07.md`.
- Model: ResNet18 U-Net on 256 px proposal-centred crops (4-flip TTA). It is trained on the detector's
  own OOF proposals, with targets from `v11.match_positive_proposals` (IoU ≥ 0.30, one positive per GT
  per record).
- Training: 12 epochs, AdamW 1e-4, cosine schedule, 0.5 BCE + 0.5 Dice, batch 16.
- Init lineage: ImageNet ResNet18 → GT-crop training → COCO-mask fine-tune → r3.
- Output: `refined_constrained` clamps each refined mask to a 7x7-ellipse dilation of its seed.
- Operating cell: mask threshold t = 0.45, confidence c = 0.30 ("V17 cell").
- The deployed refiner `r3all` is trained on the OOF proposals of all 707 files (`scripts/run_v65.sh`).

**2.4 Export.** Masks are produced natively at 2048 and encoded as pycocotools RLE counts, one row per
filament, validated by `scripts/validate_submission.py` (`v69_kaggle_metric_audit` §5).
**[TBD: whether the final export uses min-area 200; see §4]**.

**2.5 1280 variant: [TBD, pending].** p2first at imgsz 1280 (prototypes 640x640, batch 2; disclosed
confound) gave PQ 0.4134 vs 0.3967 on fold 0, d = +0.0143, CI [+0.0041, +0.0250], P = 0.998
(`v68_preregistration_2026-10-10.md`, results section). It is in 5-fold OOF now.
**[TBD: 5-fold pooled result (gate ≥ +0.010); refined-pipeline test against p2first+r3; final choice.]**

## 3. Validation protocol

- **Folds.** 5-fold OOF over all 707 files, split by file (`scripts/make_v61_folds.py`): files
  141/141/142/142/141, about 1640 annotations per fold, group_days = 0 after measuring that test files
  sit as close to train files as train files do to each other (`SESSION_CONTEXT` §5). Every file is
  predicted by a model that never saw it.
- **Refined evaluation.** The refiner was historically fit on 566 canonical files, so refined numbers use
  the 141 held-out canonical files, where both detector and refiner are unseen (`v62_refiner_test`).
  Their SE is about 0.005, so a true +0.008 passes the +0.010 gate only about 35% of the time
  (`v69_history_audit_2026-10-10.md` §3). A leak-free 5-fold refiner that scores the refined pipeline on
  all 707 files is running (V73). **[TBD: V73 result.]**
- **Pre-registration.** Each experiment writes its gate before any number exists (doc sha256 recorded,
  e.g. V70/V71/V72). Typical CONTINUE rule: pooled d ≥ +0.010, P(d>0) ≥ 0.90, CI lower bound > 0,
  ≥ 4/5 folds positive, FP/proposal volume ≤ 1.30x. KILL if d < +0.005 (`v62_5fold_preregistration`,
  `v66_preregistration`).
- **Statistics.** Paired file bootstrap (2000-4000 resamples). Confidence cuts are cross-fitted across
  folds when cuts are compared (`v66_preregistration`).
- **Seed noise.** Measured, not assumed: pooled 5-fold A' - A = +0.0015 (`v62_5fold_result`).
- **Public leaderboard.** It shows 2 decimals on about 90 test images, so it cannot resolve the
  effects above (`v69_kaggle_metric_audit` §1.1). V17 scored 0.39 and V65 (p2first + r3) scored 0.39
  (`v66_ensemble_result`).

## 4. Results and ablations

**Main OOF results** (direct detector on 707 files at conf 0.35: `v62_5fold_result`; refined on the 141
files at t0.45/c0.30: `v62_refiner_test`, `v66_ensemble_result`, `v69_error_analysis` §1):

| pipeline | files | PQ | SQ | TP / FP / FN |
|---|---|---|---|---|
| baseline YOLO11s-seg (V17 recipe), direct | 707 | 0.3856 | 0.6485 | 4503 / 2442 / 3696 |
| **p2first, direct** | 707 | **0.4060** | 0.6567 | 4737 / 2387 / 3462 |
| baseline + V11 refiner (public-V17 style) | 141 | 0.4517 | 0.6886 | 1004 / 464 / 589 |
| p2first + V11 refiner | 141 | 0.4566 | 0.6876 | 1016 / 451 / 577 |
| **p2first + r3 refiner (deployed, V65)** | 141 | **0.4595** | 0.6894 | 1020 / 448 / 573 |
| p2first_1280 (+ refiner) | 707 | [TBD] | [TBD] | [TBD] |

**Positive ablations**

| change | Δ PQ (95% CI) | level | source |
|---|---|---|---|
| P2 level + stride-2 prototypes (p2first vs baseline) | **+0.0204** [+0.0151, +0.0256], 5/5 folds | direct, 707 | `v62_5fold_result` |
| ... small-object recall | 0.411 → 0.460 | direct, 707 | `v62_5fold_result` |
| crop refiner on top of the detector | +0.059 (baseline), +0.040 (p2first) | 141 | `v62_refiner_test` |
| r3 refiner refit on p2first proposals vs V11 | 0.4566 → 0.4595 | refined, 141 | `v69_history_audit` §3 |
| p2first + r3 vs public-V17 pipeline | +0.0078, P 0.92 (below the +0.010 bar) | refined, 141 | `v66_ensemble_result`, `v65_preregistration` |
| imgsz 1280 (fold 0) | +0.0143 [+0.0041, +0.0250] | direct, fold 0 | `v68_preregistration` |
| min mask area 200 px (export), on p2first + r3 | +0.0018 [+0.0005, +0.0034], removes FP only | refined, 141 | `v72_bundle_result` |

**Key negatives** (all pre-registered and closed)

| change | Δ PQ | verdict | source |
|---|---|---|---|
| p2first gain after the refiner | +0.0049 [-0.0052, +0.0153] (≈80% absorbed) | not distinguishable from 0 | `v62_refiner_test` |
| p2last (P2 detection, stride-4 prototypes) | structural recall +0.008 | fail | `SESSION_CONTEXT` §7 |
| BCE + Focal + Dice mask loss | +0.0047 vs seed noise +0.0049 (fold 0) | kill | `SESSION_CONTEXT` §6 |
| size-gated hybrid (skip refiner < 1000 px) | -0.0137 [-0.0247, -0.0022] | fail | `v62_size_hybrid` |
| voting ensemble of 4 OOF detectors | -0.0026 [-0.0069, +0.0017] | kill | `v66_ensemble_result` |
| YOLO11m scale (fold 0) | +0.0036 | kill | `v68_preregistration` |
| train-time flips (fold 0) | +0.0068 [-0.0043, +0.0176] | inconclusive | `v68_preregistration` |
| learned full-pool re-scorer (agreement targets) | -0.0099 [-0.018, -0.002] refined (+0.0161 direct) | kill | `v70_rescorer_result` |
| post-refiner FP suppression rules | +0.0028 [-0.0002, +0.0061] | kill | `v71_postrefiner_suppression_result` |
| bundle MIN200 + R3 + R2 | +0.0034 (bar +0.004) | not adopted | `v72_bundle_result` |
| 1 px erosion of refined masks | -0.024 (p2first, V17 cell) | never erode refined | `v69_kaggle_metric_audit` §3.3 |
| labels carried to GONG frames ±10 min | -0.098 vs archive copy | fail, closed | `v69_neighbour_frame_result` |
| DenseCRF; one-to-one tile loss; Mask R-CNN; Mask2Former | below V17 | closed | `SESSION_CONTEXT` §3-4 |

Caveat for honesty: several earlier closures (V44-V56) used a non-canonical harness and are not comparable
with post-Phase-A numbers (`v69_history_audit` §1, §4).

## 5. Error analysis and rubric distributions

Source: `v69_error_analysis` (CPU, cached OOF, reproduces PQ 0.4060 and 0.4595 exactly).

- **FN: the score or selection is wrong, not the mask.** In 58% of direct FN and 62% of refined FN, a
  mask with IoU > 0.5 already exists in the candidate pool (below the cut, or a non-anchor view).
  Mask near-misses (IoU 0.3-0.5) account for 27% / 23%. True recall misses are 14-15%, and 95% of the
  "no proposal" FN on multi-record files are single-annotator filaments (§2).
- **FP: about one third are another annotator's TP.** 850 of 2387 direct FP (35.6%) and 149 of 448
  refined FP (33.3%) match a GT in another record of the same image. Only 459 (19%) are pure
  hallucinations (§2).
- **Hardest GT.** Area ≤ 500 px: recall 0.315 (25% of FN). Lowest-contrast quintile: 0.355 (30.5% of
  FN). Width ≤ 8.3 px: recall 0.423, matched IoU 0.598, which is the fine-structure/barb regime.
  Chirality "Unidentifiable": 46.5% of FN (§2).
- **Image conditions are not predictive** (Spearman with disk std, sharpness and year all within ±0.06).
  The annotator record ID is the dominant file-level factor: direct PQ ranges 0.32-0.50 by record ID (§3, §5).
- **IoU distribution of TPs** (direct, 707 files): 0.5-0.6: 1347; 0.6-0.7: 1868; 0.7-0.8: 1317;
  0.8-0.9: 205; > 0.9: 0 (§1).
  **[TBD: host self-evaluation plots (Dice histogram, IoU histogram, 1:n / n:1 bar chart) for the final
  pipeline, on OOF files. Use the host notebook's `plot_distribution` / `plot_m2n_counts`.]**
- **One-to-many / many-to-one** (direct): 122 GT (1.5%) are split across ≥ 2 kept predictions; 155 FP
  blobs cover ≥ 2 GTs; 22 FN are swallowed by a prediction that matched a neighbour. In the refined
  pipeline, fragments are 89/448 = 20% of FP (§4). Topology errors are a minor share of the loss.
- **Fine structures / barbs.** Thin-filament recall did not improve with p2first (431 GT: 0.548 → 0.550;
  `v62_5fold_result`). The gain was in small objects. **[TBD: qualitative figure of barbs on 2-3
  OOF images, refined vs GT from ≥ 2 annotators.]**

## 6. Data use and rule compliance

- **Inference uses only the supplied test H-alpha image.** No neighbouring frames and no magnetograms
  are used at inference (host rulings 743385/744047; `kaggle_discussion_scan_2026-10-04.md`).
- **No test labels.** The public MAGFiLO v1.0 Dataverse file contains test GT. It was set aside and never
  used for training or tuning (`SESSION_CONTEXT` §2-3). No test pseudo-labels were used.
- **No extra ground truth.** The host banned SWEFil on 2026-10-06 (thread 746318). Before that ruling,
  SWEFil was used only in the V58/V59 verifier experiment, which was closed (-0.070 on V17) and is not
  part of any submitted pipeline (`SESSION_CONTEXT` §3-4; `v69_kaggle_metric_audit` §0.5).
  **[TBD: state this disclosure in the final text, and ensure no SWEFil artifact is in the repo.]**
- **Magnetograms: training-side analysis only.** Co-temporal GONG magnetograms for the 707 train images
  were fetched for the V67 diagnostic, which asks whether they carry information about detector errors.
  The host allows this (thread 743521). No test-time magnetogram was fetched, and no model in the
  pipeline uses them (`v67_preregistration_2026-10-07.md`).
  **[TBD: V67 verdict is not documented locally.]**
- **Neighbour GONG frames: diagnostic only, train files only.** Archive frames near 141 fold-0 train files
  were used to test label transfer (FAIL). They were not used for training or inference
  (`v69_neighbour_frame_result`). A label-transfer question is pending with the host (thread 748045).
- **External weights.** COCO-pretrained YOLO11s-seg (Ultralytics) and ImageNet ResNet18 (torchvision).
  **[TBD: licence statement for the Ultralytics AGPL-3.0 dependency versus the OSI-licence winner
  obligation.]**

## 7. Limitations

- **Small evaluation set for the refined pipeline.** Refined comparisons use 141 files (CI about ±0.010).
  The deployed p2first + r3 gain over the public-V17 pipeline (+0.0078) is below our own +0.010 bar
  (`v66_ensemble_result`, `v69_history_audit` §3).
- **Untested refiner lineage.** The deployed refiner is trained on OOF proposals but serves the all-data
  detector. Calibration shift between the two was not measured (`v69_history_audit` §4.6).
- **Offline gains do not show publicly.** Local gains have not moved the 2-decimal public score
  (V17 = V65 = 0.39). Clean teams report a 0.05-0.06 OOF-to-public drop (`v69_kaggle_metric_audit` §1.2).
- **Thin structures and barbs are not improved** by the architecture change (§5).
- **The remaining loss is mostly annotator disagreement.** A learned agreement re-scorer did not help
  (`v70_rescorer_result`). Modelling annotator propensity is open.
- **Compute efficiency is unmeasured.** **[TBD: end-to-end s/image for detector 4-flip + refiner on the
  Kaggle GPU; training GPU-hours (stage 2 of p2first ≈ 298 s/epoch on a T4, `SESSION_CONTEXT` §7).]**
- **Earlier closures may be weak.** Some were made on a confounded harness (`v69_history_audit` §1).
