# Research toward public PQ 0.50 — 6 September 2026

**The best-supported next experiment is proposal-aware, instance-conditioned
refiner training.** Follow it with a controlled detector training-resolution
experiment and, after selecting the recipe, a final fit using all competition
training images. V10's frozen-weight changes have reached a practical plateau:
the submitted file scored **0.38**, exactly matching our previous best.
There is no verified, fully reproducible public recipe in this review that
guarantees 0.50. The target remains a research objective, not an expected result
from a small threshold adjustment.

Research used the authenticated Kaggle public leaderboard export at
**2026-09-06 02:21:04 UTC / 07:51:04 IST**, all **58** competition-linked public
notebook metadata entries sorted by score, update time, and votes, and static
inspection of **17** downloaded public sources. No downloaded notebook was
executed. Original notebook files and source hashes are retained in the ignored
`outputs/research_2026-09-06/` directory. Private training code, checkpoints, and
other teams' private submission files are not available through this review.

**What the highest scores actually establish.**

| Live leaderboard group | Score | Evidence |
| --- | ---: | --- |
| Rank 1: NIklitaCheporev | 0.56 | Public best submission 55993614, submitted 3 September 19:44:41 UTC; no competition-linked public notebook found for this account |
| Ranks 2–21: 20 teams | 0.55 | Includes HDJoJo and Lâm Huy; their public code exposes very different submission paths |
| Ranks 22–27: six teams | 0.40 | No scores from 0.41 through 0.54 in this snapshot |
| Next 12 teams | 0.39 | The nearest score group above ours |
| Our team, WeAreBatman: rank 43 of 517 | 0.38 | V10 submission 56044785; ten submissions in the snapshot |

These are current rescored leaderboard values, not notebook-title claims.
The exported table and top-team public submission metadata are saved locally.
[Kaggle leaderboard](https://www.kaggle.com/competitions/filament-segmentation-2026/leaderboard)

The competition changed its principal metric from Dice to PQ on 7 August and
announced the rescore on 12 August. Its 20 August announcement also cautions
about submissions exploiting the system. This warrants checking provenance;
it does **not** establish that any particular private model is invalid.
Old 0.69/0.70/0.90 notebook titles are not evidence of comparable current PQ.
The competition also restricts additional ground-truth metadata and requires
reproducible source for final review.
[Organizer announcements and evaluation](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/prizes)

**HDJoJo provides the most relevant high-score modeling lead.** The public
notebook reports **0.55, version 2**, and runs YOLOv8l segmentation at `imgsz=2048`
with `conf=0.3`, `iou=0.0`, and a five-pixel minimum. It loads a private checkpoint
named `magfilo_260825_yolov8l_seg_imgsz2048_nc1.pt`. Training data selection,
augmentation, validation, and checkpoint provenance are unpublished. The name
suggests training resolution but does not prove it. Public best submission
55904210 was made on 31 August; we cannot establish that its bytes equal the
notebook output. There is no embedded prediction payload in the inspected
inference source. This supports testing native-resolution training ourselves,
not attributing the entire score to a larger YOLO.
[HDJoJo notebook](https://www.kaggle.com/code/hdjojo/solar-filament-seg-inference)

One concrete issue prevents copying that inference code blindly: it passes
`result.boxes.cls` into its overlap-ordering function instead of `boxes.conf`.
Class IDs do not provide confidence ordering in a single-class model. Also,
`iou=0.0` is an aggressive box-suppression threshold for this YOLO family, not
an instruction to disable suppression. Our implementation should preserve
actual confidence values and validate suppression on neighboring filaments.
[Ultralytics prediction arguments and result fields](https://docs.ultralytics.com/modes/predict/)

**Lâm Huy's 0.55 source does not expose a trainable winning model.** Its final
submission reads an embedded compressed CSV; the small U-Net demonstration is
not used to generate those predictions. The payload contains 1,342 rows over
178 images, with synthetic fallback masks for the other two images. Today's
download has exactly the same cell source as our previous inspection. Its
origin is unverified. We should neither copy those predictions nor treat its
instance counts as ground truth or a target submission size.
[Public notebook](https://www.kaggle.com/code/lamhuy8904/solar-filament-unet-segmentation-0-55),
[earlier local inspection](public_notebook_lamhuy_055_review_2026-09-05.md)

The leaderboard clustering alone does not tell us how the other nineteen
high-scoring teams produced their results. The 0.56 leader, and the checked
0.40 accounts `jab100`, `welshonionman`, and `talhabacak`, exposed no notebook
through the competition-filtered author query. This is an access limitation,
not evidence that no code exists elsewhere.

**Recent public work offers useful mechanisms, but little evidence that changing
backbones alone will beat our current score.** In the table below, “team best”
comes from the live leaderboard and may belong to a different/private run.
It must not be attributed to the downloaded notebook. Dates are source metadata
last-run dates; indexed notebook scores are identified separately.

| Public source | Last run | Live team best | Useful finding from actual source |
| --- | --- | ---: | --- |
| [Tim Gluhkikh: YOLO detect + seed refiner](https://www.kaggle.com/code/timgluhkikh/best-score-magfilo-yolo11s-detect-crop-u-net) | Aug 16 | 0.37 | YOLO11s detection at 1280; 512-pixel ResNet34 U-Net with raw image, CLAHE, and a padded box seed. Training jitters the seed box. Inference defaults to attached weights. |
| [Ektarr: New Metric](https://www.kaggle.com/code/ektarr/new-metric-solar-filament) | Aug 8 | 0.38 | Five grouped folds; YOLO11s-seg plus image/CLAHE/seed refiner. Uses perturbed GT-mask seeds in training and exact GT seeds during crop validation: crop scores are not deployment evidence. |
| [MAGFiLO V9](https://www.kaggle.com/code/muhammadshoaibaltaf/magfilo-v9-evaluation-aligned-solar-filament-s) | Aug 29 | 0.36 | YOLO26l detector at 1280 and 512-pixel EfficientNet-B4 U-Net++; still GT-centered crop training. Indexed latest/older pages report 0.35/0.36. |
| [Blended-Tile U-Net](https://www.kaggle.com/code/anthonytherrien/magfilo-blended-tile-u-net-for-solar-filaments) | Aug 25 | 0.36 | Latest code actually uses one centered 1888-pixel crop with EfficientViT-B2 and eight-view TTA. Indexed notebook score 0.33, best 0.36 V10. The title no longer describes the current implementation. |
| [Siva: improved pipeline](https://www.kaggle.com/code/sivapnlr80/solar-filament-segmentation-improved-pipeline) | Aug 30 | 0.33 | Five 1280-pixel Mask R-CNN models, cross-fold mask voting, three-fold agreement. Its claimed validation PQ 0.4367 is not comparable to ours; see issues below. |
| [Ruhul: U-Net++ / EfficientNet-B4](https://www.kaggle.com/code/ruhul20/solar-segmentation-u-net-efficientnet-b4) | Sep 1 | 0.29 | Raw/CLAHE/local-contrast channels, EMA and TTA; selection uses a weighted objective combining image-mean PQ, Dice, and morphology rather than our canonical global PQ. |
| [EdgeAttNet](https://www.kaggle.com/code/koushikkumardinda/solar-filament-segmentation-2026-edgeattnet) | Sep 4 | 0.26 | Newest competition-linked source; semantic edge/attention model and morphology. Indexed notebook score 0.25. A newer architecture is not a demonstrated improvement. |
| [RF-DETR baseline](https://www.kaggle.com/code/iamprateek/rf-detr-baseline-submission) | Aug 28 | 0.26 | Public inference using an attached RF-DETR checkpoint; useful alternative detector family, presently weak score evidence. |
| [RF-DETR + crop refiner](https://www.kaggle.com/code/jirkaborovec/filamentsegm-eda-rf-detr-seg-ptl) | Jul 27 | Not resolved | Grouped split, RF-DETR small, six detector epochs, eight refiner epochs. An implementation reference, not a verified high-score recipe. |
| [SFC–MAgo](https://www.kaggle.com/code/trishyagupta/sfc-mago) | Sep 2 | Not resolved | Preprocessing/CLAHE exploration; no comparable high-scoring trained submission established. |

Additional source checks change how we should interpret those methods:

- Siva's [training source](https://www.kaggle.com/code/sivapnlr80/filament-maskrcnn-kfold-v2)
  keeps only categories 1 and 2, excluding category 3. Its
  [validation sweep](https://www.kaggle.com/code/sivapnlr80/filament-v41-sweep-val)
  evaluates all five fold checkpoints on a new random validation subset. That
  is not out-of-fold ensemble validation: training folds overlap the evaluated
  files. It also rasterizes polygons with integer OpenCV filling. These issues
  prevent treating its 0.4367 as an independent performance benchmark.
- Tim's jitter modifies the seed box, but the training crop remains centered
  on the GT box. We should perturb the actual crop origin and scale too, or use
  detector-generated crops, to reproduce inference errors more faithfully.
- The [grouped-validation writeup](https://www.kaggle.com/code/dattadhebe/solar-filament-grouped-cv-and-strategy-writeup)
  correctly highlights repeated physical observations, but its 0.66 score and
  penalized-Dice results predate the metric reset. We already group by filename.
- The [0.70 topology handoff](https://www.kaggle.com/code/phuongncn/lb-0-70-topology-safe-solar-filaments-handoff)
  contains genuine detector/refiner inference, but its score claim is explicitly
  dated 5 August. Its author's live team best is 0.36. Our own component-splitting
  experiments have already regressed, so generic topology postprocessing is low priority.
- [Please fix the Leaderboard](https://www.kaggle.com/code/artkomissar/please-fix-the-leaderboard)
  is a diagnostic of the former scoring defect, not a model-training lead.

**Our measured bottleneck is the proposal-to-instance decision, alongside mask
geometry.** V10's completed canonical validation is:

| Metric | V6 control | V10 |
| --- | ---: | ---: |
| Public score | 0.38 | 0.38 |
| Full canonical validation PQ | 0.437873 | 0.439246 |
| True positives / false positives / false negatives | 891 / 299 / 702 | 899 / 307 / 694 |
| Mean IoU on matched instances | 0.683838 | 0.683787 |
| Test submission masks | 982 | 991 |

V10 added eight validation matches and eight false positives while leaving
matched-mask quality essentially unchanged. Its paired-bootstrap gain intervals
include zero. The three isolated rejection gates changed no tuning predictions.
The evidence supports moving beyond frozen mean-probability gates and tiny
rescue changes. [V10 execution and exact artifacts](v10_execution_2026-09-05.md)

I ran a new CPU-only diagnosis on the **94 tuning filenames / 1,134 separate
annotator instances**. All 188 V6/V8 raw proposal cache hashes match the frozen
V10 manifest. Canonical COCO masks reproduce annotation areas exactly. No new
audit-set candidate search was performed.

| Tuning-only geometric diagnostic | GT instances with opportunity |
| --- | ---: |
| Some V6 raw four-view proposal has mask IoU > 0.5 | 906 / 1,134 (79.9%) |
| Some retained V6 group anchor has mask IoU > 0.5 | 841 / 1,134 (74.2%) |
| Some anchor survives consensus >= 0.40 and has mask IoU > 0.5 | 563 / 1,134 (49.6%) |
| Accepted V6 proposal's dilated support contains >50% of GT pixels | 746 / 1,134 (65.8%) |
| Accepted V6 proposal's full context crop contains >50% of GT pixels | 835 / 1,134 (73.6%) |
| At least 90% of GT is inside some accepted dilated support / full crop | 580 / 770 instances |

These are **per-GT oracle opportunities**, not PQ, one-to-one recall, or promised
recoverable true positives. Multiple GT masks can select the same proposal.
The first three rows measure raw detector shapes; the last three allow a
hypothetically perfect refiner. They are not a single decreasing recall funnel.
For reference, actual control refinement/cleanup produces **626 tuning TP**,
and V10 produces **632**.

The specific losses are actionable: selecting the highest-confidence fixed
anchor loses **65** raw mask-match opportunities; applying the acceptance cutoff
loses a further **278** grouped opportunities. Lowering the cutoff alone is not
a solution—our previous sweep selected 0.40 because additional predictions also
create false positives. A trained proposal-quality decision could recover a
different subset than a global confidence threshold.

Separately, **89 GT instances** have >50% coverage in an accepted context crop
but <=50% coverage inside its permitted dilated mask. Even a perfect refiner
cannot match those instances while confined to that support. This motivates
learning instance identity from a seed and permitting controlled mask growth.
It does not justify removing the constraint from the existing unconditioned
refiner, which could merge neighboring filaments.

Small and ambiguous objects deserve explicit training diagnostics. Among the
451 tuning instances with area 200–999 pixels, raw V6 match opportunity is
71.2%, falling to 35.9% after grouping/cutoff. Category 3, “Unidentifiable,” has
469 tuning instances and only 38.4% accepted-anchor match opportunity, compared
with 55.8% and 59.5% for categories 1 and 2. Those category names describe
chirality, not whether an annotation is valid. Retain all three categories.
The raw V6/V8 union remains the previously observed 790 both / 116 V6-only /
33 V8-only / 195 neither; V8 offers limited complementary coverage.

Reproduce the diagnostic with:

```bash
.venv/bin/python scripts/analyze_proposal_research.py
```

The [script](../scripts/analyze_proposal_research.py),
[per-instance CSV](../outputs/research_2026-09-06/proposal_diagnostics.csv), and
[summary with hashes](../outputs/research_2026-09-06/proposal_diagnostics_summary.json)
record the method. The script calls the exact local grouping and crop helpers
without importing GPU dependencies; it never writes submission predictions.

**What reaching 0.50 would require.** V10 has local recognition quality 0.64237
and matched segmentation quality 0.68379, whose product is PQ 0.43925. Holding
other quantities fixed gives these useful counterfactuals:

| Local-validation counterfactual | Result |
| --- | --- |
| Delete all 307 FP, keep every TP and IoU | PQ 0.49336 |
| Keep counts fixed, reach PQ 0.50 through better masks | Mean matched IoU must rise to 0.77836 |
| Add matches with IoU 0.70, create no FP, keep existing masks | About 189 extra TP are needed for local PQ 0.50 |

These are algebraic scale estimates, not feasible oracle operations or gain
forecasts. Public 0.50 is harder to infer: current local PQ already exceeds
public PQ by about 0.059. We cannot assume that gap is constant or that local
0.50 will produce public 0.50. A robust path needs better recall and boundaries,
plus validation that tracks generalization.

**Recommended experiment order.** The following is a proposed research plan;
none of these new training runs has been launched.

1. **V11: fix refiner training and instance conditioning while keeping V6 fixed.**
   Compare a matched-budget GT-centered training control, proposal-centered
   training with unchanged inputs, then the same training with an explicit
   proposal-seed channel. Initialize any added input-channel weights to zero
   so the initial image path is preserved. Use canonical, single-instance
   targets and noisy detector crops, including translation/scale errors. Prefer
   proposals generated out of fold within the original training set; in-sample
   proposals are a cheaper but optimistic approximation. First keep the current
   support constraint. Only for the conditioned model, add one predeclared
   full-context-crop variant to test the measured geometric limitation. Preserve
   pixel ownership and inspect neighboring-instance merges. Start at crop 256;
   make 512 a separate follow-up if boundary errors remain. Train-time CLAHE is
   another separate ablation, not a simultaneous change.

2. **V11 follow-up: learn which low-confidence proposals are useful.** Train an
   IoU/quality head or small scorer on training-only, preferably out-of-fold
   detections. Inputs can include mask/image features, view agreement, and
   refiner support. Compare it with raw consensus at the same accepted-proposal
   budget, then evaluate final PQ after ownership. The goal is to investigate
   the 278 cutoff losses without indiscriminately adding predictions. Do not
   label every unmatched proposal as background: incomplete annotations make
   that unsafe supervision. Verified negatives must be clean across all
   available annotators and visually unambiguous. Keep uncertain cases ignored.
   This is materially different from V10's untrained foreground-mean heuristic.

3. **V12: train at higher spatial resolution with a controlled comparison.**
   Compare a matched-update 1024 continuation against a 1536 continuation of
   V6's detector; progress to 2048 if the first comparison supports it and a
   GPU smoke run fits. Keep architecture, split, labels, augmentation, and
   effective batch controlled, using a smaller microbatch as needed. Evaluate
   direct segmentation and the unchanged refinement pipeline to localize gains.
   V9's unsuccessful 1536/2048 inference sweep did not train the weights at
   those resolutions and therefore does not settle this question. A native
   1024-tile branch is an alternative if full-frame training is impractical;
   it needs explicit handling of cut objects and tile-boundary duplicates.
   Test YOLOv8l or RF-DETR only after measuring whether spatial detail or
   detector capacity is the actual limiting factor.

4. **Confirm the strongest training recipe with genuinely excluded folds.**
   The old 47-image audit has influenced past work and is not untouched.
   Bootstrap intervals on repeatedly searched data remain descriptive. Freeze
   new grouped folds and retrain both detector and refiner excluding each fold;
   a V6 checkpoint trained on those filenames cannot seed an allegedly unseen
   fold. Use global canonical PQ, separate annotator entries, strict IoU >0.5,
   and identical postprocessing. Add site/year/size diagnostics to see whether
   a gain depends on one subset. Select by final-image PQ, not GT-crop Dice or
   an average of image-level PQ values. A practical V11 screening target is
   at least +0.01 tuning PQ with a nonnegative audit delta, followed by fresh-fold
   confirmation; this target is our heuristic, not a statistical guarantee.

5. **Fit the selected recipe on all 707 training images before the final export.**
   The current detector and refiner use 566 filenames and 6,606 annotations;
   the full permitted set contains 707 filenames and 8,199 annotations. That
   adds 141 images (24.9% relative to the training subset) and 1,593 annotations
   (24.1%). Freeze hyperparameters and training duration from excluded-fold
   evidence, then refit. Once those former validation images enter training,
   their score is not held-out evidence. If compute permits, compare a small
   independently trained seed/fold ensemble using valid excluded-fold predictions
   before deciding whether its added runtime is justified.

The rationale for instance conditioning and fine-detail recovery also has
primary research support, though none supplies a solar-competition gain
guarantee. [Deep Extreme Cut](https://arxiv.org/abs/1711.09081) conditions
segmentation on object-specific spatial input.
[Mask Transfiner](https://openaccess.thecvf.com/content/CVPR2022/html/Ke_Mask_Transfiner_for_High-Quality_Instance_Segmentation_CVPR_2022_paper.html)
targets high-frequency errors lost by coarse masks. A recent
[solar-filament study](https://www.sciencedirect.com/science/article/pii/S0094576526004649)
combines DETR with U-Net and multi-site image normalization. Its dataset and
metrics differ from this competition; it supports architectural/preprocessing
hypotheses, not importing external annotations or claiming equivalent PQ.

**Work to deprioritize now:** broad threshold grids, additional fixed-weight
upscaled inference, automatic component splitting, simply replacing YOLO11s
with a bigger model, dropping category 3, and targeting someone else's test-row
count. We have either contrary local evidence or insufficient reproducible
evidence for these as the next use of compute.

Research artifacts include `leaderboard_summary.json`, the original leaderboard
ZIP, source metadata and hashes, public best-submission metadata for the first
and third teams, and the proposal diagnostics under
`outputs/research_2026-09-06/`. The latest Lâm Huy cell-source SHA-256 is
`a3627800f5d114911588242b54f062b9b4d8bda9231e24d963e456dbaa58406b`.
That opaque third-party notebook remains ignored and is not part of our model.

A documentation error was also corrected in the V10 handoff/execution/research
notes: the frozen implementation stores zero foreground quality when refinement
is empty; it does not substitute original-proposal mean support. This was
verified against code and the executed manifest. No model weights, inference
behavior, or submitted CSV bytes changed during this research.
