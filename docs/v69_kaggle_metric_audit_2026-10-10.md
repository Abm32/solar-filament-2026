# V69: Kaggle intel, metric audit and submission-format check (2026-10-10)

Scope: research and audit. CPU only on rogii (V68 holds all 4 GPUs; nothing in /mnt/v68 or
outputs/v68* was touched). No submission. Train OOF caches only; no test labels and no MAGFiLO
Dataverse test GT were read.
Artifacts on rogii: `~/v69_kernels/` (discussions, competition pages, leaderboard, pulled notebooks,
audit outputs) and `~/v69_tools/` (the scripts that produced every number below).

## 0. TL;DR (ranked levers)
1. **Minimum mask area 200 px (currently 20) in the export.** At conf >= 0.35, every prediction
   under 200 px matched zero annotators. Gains: refined pipeline at the V17 cell (t0.45/c0.30, 141
   clean files) **+0.0037 baseline (CI [+0.0020, +0.0058])** and **+0.0016 p2first (CI [+0.0004,
   +0.0032])**; direct p2first on 707 files +0.0022 (CI [+0.0015, +0.0029], 5/5 folds). It is free,
   never negative in any cell, and needs no retraining. **This is the only post-process lever that
   survives for the deployed path.**
2. **Mask width: do NOT erode refined masks, but DO account for width in direct-detector screens.**
   Direct YOLO masks are about 1 px too wide: a 1-px erosion (3x3 cross) gives **+0.0226 for p2first
   (0.4060 -> 0.4285, CI [+0.019, +0.026], 5/5 folds)** and +0.031 for the baseline. 2-px erosion and
   1-px dilation are worse. On the **refined** pipeline the same erosion costs **-0.021 to -0.036**
   (all cells, CIs exclude 0): the V11 refiner already fixes the width, and that is part of its
   +0.04-0.06 lift. Consequences: (a) never add erosion to the refined export; (b) direct-detector
   comparisons (V62/V68 screens) mix width bias with detection. An arm that changes mask resolution
   (p2first, 1280) partly gains by being thinner, which is one reason "the refiner absorbs ~80% of
   the p2first gain". Score direct screens with erode1 as well, so they measure detection.
3. **Already right; leave alone:** local scorer (bit-for-bit equal to the host code: identical
   TP/FP/FN on 707 files), confidence cut (0.30-0.375 plateau = the Dinkelbach optimum u ≈ PQ/2),
   mask registration (all 1-px shifts cost 0.004-0.015), RLE encoding and pixel-disjoint export.
4. **Learned decision layer (tushhaaarrrr's û > tau = PQ/2):** the oracle headroom is large (0.49
   vs 0.406 direct), but a cross-fitted rule from conf x area x limb distance adds only +0.0008 (CI
   [-0.0015, +0.0031]). Confidence is already calibrated. It needs genuinely new evidence (a
   soft-target semantic map, a second model family), so it is a modelling project, not a postprocess.
5. **Rules/admin:** the host **banned SWEFil** (2026-10-06, "we do not allow using more
   ground-truth data"). Keep every SWEFil-trained artifact (the V58/59 verifier) out of the final
   pipeline and repo story. Make **both** final selections reproducible from the public repo
   (question 747960 is open). The rubric also scores Dice/IoU distributions and 1:n / n:1 relations.
6. **Calibrate expectations:** every clean team reports a local/OOF -> public drop of about 0.05-0.06
   (tushhaaarrrr 0.46 OOF -> 0.39-0.40; Spaghetto99 0.49 -> 0.41). The public board is about 90
   images at 2 decimals. Realistic clean frontier: 0.40-0.42 public.

## 1. Kaggle intel since 2026-10-04
Source: authenticated internal API (`discussions.DiscussionsService/GetTopicListByForumId` and
`GetForumTopicById`, `competitions.PageService/ListPages`, `LeaderboardService/GetLeaderboard`)
via `~/v69_tools/fetch_disc.py`; raw text is in `~/v69_kernels/discussions/` and
`~/v69_kernels/lb/pages.txt`.

### 1.1 Host rulings and announcements
- **SWEFil is NOT allowed** (topic 746318, host, 2026-10-06): "we do allow using more data for
  training (e.g., other instruments' observation of the sun), we do not allow using more
  ground-truth data." This reverses our 10-04 reading that SWEFil was firewall-clean. Unlabelled
  external images and other instruments (e.g. magnetograms) for training remain allowed. Inference
  may use only the supplied test H-alpha image.
- **Scoreboard** (747399, host, 10-08): "We will thoroughly examine all submissions with PQ score
  higher than 0.35". Final judging follows the rubric. Very high leaderboard scores are being
  examined; no new rescoring announcement.
- **Final submission** (Overview > Final Submission, unchanged since Aug 7): Google form, a 4-page
  Overleaf report, and a public git repo with `requirements.txt` and a Jupyter notebook that
  reproduces the submitted predictions "without additional files". An open question (747960,
  10-10, no answer yet) asks whether both selected submissions must be reproducible. **Safe
  assumption: make both selected submissions reproducible from the repo.**
- **Rubric** (Evaluation page): 70% quantitative = PQ **plus the distributions of Dice, of IoU and
  of one-to-many / many-to-one relations**; 30% qualitative = pipeline description, morphology and
  code quality. The rules page also mentions penalties for fragmentation/over-merging and
  "end-to-end computational efficiency".
- **Timeline:** deadline 2026-11-15 06:00 UTC; winners announced Nov 30. The team-merger deadline
  only points to the Timeline page (merging needs combined submissions <= 5/day x days). The
  announcements list nothing after Sep 20 (the SABID workshop reminder; the workshop deadline was
  extended to Oct 6 and has passed).
- **Leaderboard facts** (competition API): `leaderboardPercentage` = 50, so the public board is
  about 90 of the 180 test images and the private board is the other 90. The board shows 2
  decimals. The metric runs as Kaggle metric `panoptic_quality` (kernel 125319199, version 72842,
  private: 403). Five submissions per day; two final selections.
- The leaderboard is still polluted: top 20 at 0.55-0.63, with new 0.60-0.62 entries this week.
  No cleanup has happened.

### 1.2 What clean competitors report
- Several people report the same **local -> public gap of about 0.06**: Trishanth ("local ~0.46,
  Kaggle ~0.40"), Даша ("local ~46, stuck ~40"), Spaghetto99 (local 0.49 on a casual split ->
  **public 0.41**). tushhaaarrrr: "every reproducible public pipeline we could audit loses
  ≈0.05-0.06 from validation to the public board"; their strict OOF 0.4597 -> public 0.39-0.40.
  Our V17 (tune 0.449 -> 0.39) fits the same pattern. **Realistic clean public frontier: 0.40-0.42.**
- Spaghetto99 names "noisyislands with 0.49 PQ" as probably first (unverified). He calls a new
  model "massive"; no details.
- juanmirod (public 0.37): YOLO11m 1280 -> crop U-Net refiner 512 (prob >= 0.40), conf 0.35,
  **area >= 300**. Says operating-point tuning is worth +0.03 val; morphology is negative for them.
- timgluhkikh (~0.41 local): YOLO11s 1280 det -> crop U-Net [img, CLAHE, box seed], box-jitter
  TTA, conf 0.28, **MIN_AREA 200**.
- MAGFiLO V9 (muhammadshoaibaltaf): YOLO11s @1280-1536 + crop seg with BCE+Tversky (beta>alpha).
- topabaem: their CSV was **rejected for overlapping instances**, which confirms the host check.

### 1.3 Strongest public method: tushhaaarrrr "filseg v4" (OOF 0.4597, notebook `solar-filament-segmentation-2026-0-40`)
I read all of it (`~/v69_kernels/tushhaaarrrr_*`; code depends on private datasets, no leaked
payload). Key ideas:
- **Metric algebra:** pooled PQ = sum_i R_i sum_c u_i(c) / (0.5 sum_i R_i |P_i| + 0.5 N_G), where
  u = annotator-averaged matched IoU. By Dinkelbach, keep a candidate iff **u_hat > tau = PQ*/2**,
  with one global tau (not per image). They use tau = 0.26. This is the same derivation as §3.2.
- Semantic soft target p(x) = mean of the annotator masks (BCE + 0.5 soft Dice, plus a spine head
  at weight 0.3), physics flat-field normalisation (limb fit, CLV, background), 3-member U-Net/UNet++
  ensemble, 5 folds x D4 TTA.
- **Component tree:** 15 threshold levels 0.05-0.80 at native 2048, plus a LightGBM hurdle scorer
  u_hat = h_hat*m_hat over 60 descriptors (RF-DETR evidence +0.0034), plus an exact DP over
  antichains. +0.065 over fixed thresholds **for a semantic map**.
- Stage-2 crop refiner (rotation-normalised 384x128, 5 channels), threshold **t* = J/(1+J) ≈ 0.40**,
  MST joining of refined pieces (for the 1:n rubric).
- Their negative results: dilate/erode ±1 px costs 0.025 *for their refined masks*; clDice/skeleton
  -0.045; magnetogram auxiliary heads n.s.; temporal TTA -0.02; 1024-px masks upsampled -0.006 vs
  native.
- Human-vs-human PQ is 0.340, so the remaining recall loss is mostly annotator disagreement.

## 2. Official metric: what it does exactly
Source: host Self_Evaluation_Notebook (re-pulled today, md5 unchanged since 09-13), the Evaluation
and Submission pages, and the data page. The Kaggle metric kernel itself is private.
- **Unit = annotator record**, not image. GT `filament_id` = `<annotator>-<image>_<k>`. For each
  annotator-image, every prediction of that image is compared with that annotator's filaments.
  TP/FP/FN and matched IoUs are **pooled (micro) over all (image, record) pairs**, so one PQ comes
  from global sums. There is no per-image or per-record averaging and no best-annotator matching.
  - Consequence: an image with R annotators counts each prediction R times. An unmatched
    prediction costs 0.5·R in the denominator; a prediction matched by m of R records adds
    sum(IoU) to the numerator and 0.5·R to the denominator. Train data: 411 files x1, 145 x2,
    151 x3 records, so 64% of the weight sits on multiply-annotated images.
- Matching: `hit = iou > 0.5` (strict), no Hungarian step. TP = number of hits, FP = prediction
  columns with no hit, FN = GT rows with no hit. Unique matching holds only for pixel-disjoint
  predictions, and **the host rejects overlapping predictions** (host ruling plus topabaem's
  rejection), so double-TP is impossible.
- IoU in float32 torch matmul over 2048x2048 (exact integer counts below 2^24 per mask).
- Empty cases: a record with GT and no predictions gives all FN; a record with no GT rows does not
  exist in the solution, so predictions on an image that has no GT record are **ignored**. Train has
  0 empty records and 0 zero-area annotations. "Ignore regions" do not exist. No predictions for
  an image is valid (no rows).
- 5 GT pairs overlap within a record (1 with IoU > 0.2), so their effect is negligible.

## 3. Local scorer vs host: exact match
`~/v69_tools/v69_metric_audit.py` re-implements the host notebook logic verbatim (torch
float32, the same hit/FP/FN rules, GT built from every annotation row) and runs it next to
`scripts/v61_toolkit.score_file` on the same OOF predictions (all 707 files, conf 0.35):

| recipe | toolkit PQ | host PQ | TP/FP/FN equal | max per-file IoU-sum diff |
|---|---|---|---|---|
| p2first | 0.4060209295 | 0.4060209295 | yes, all 707 files | 1.3e-6 (float32) |
| baseline | 0.3856469140 | 0.3856469142 | yes | ~1e-6 |

**Mismatch = 0.** Two divergences are possible in principle but neither occurs in the data: we
drop area-0 GT (there are none), and we keep empty records (there are none).

### 3.2 Metric-exploiting (legitimate) levers, measured on p2first direct OOF (707 files)
Because predictions are disjoint, a prediction's contribution does not depend on the other
predictions, so any selection or mask transform can be scored exactly offline.

| lever | PQ @0.35 | best cut | delta vs base | verdict |
|---|---|---|---|---|
| base (min area 20) | 0.4060 | 0.4062 @0.375 | - | |
| **erode 1 px (3x3 cross)** | **0.4285** | 0.4285 @0.35 | **+0.0226** [0.019, 0.026], 5/5 folds | **adopt** |
| erode 1 px, 3x3 square | 0.4252 | | +0.019 | cross is better |
| erode 2 px | 0.3982 | | -0.008 | too much |
| erode only if >= 50% area survives | 0.4277 | | +0.022 | no benefit |
| dilate 1 px (re-owned) | 0.3490 | | -0.057 | masks already too wide |
| shift x+1 / x-1 / y+1 / y-1 | 0.3985 / 0.3953 / 0.3912 / 0.4020 | | -0.004 .. -0.015 | registration is right |
| keep largest connected component | 0.4042 | | -0.0018 [-0.003, -0.000] | no |
| **min area 200** | 0.4082 | | **+0.0022** [0.0015, 0.0029], 5/5 | **adopt** |
| min area 300 / 500 | 0.4083 / 0.4044 | | +0.0023 / -0.0016 | 200-300 |
| erode1 + min area 200 | **0.4303** | | **+0.0243** [0.021, 0.028], 5/5 | **adopt** |
| cross-fitted fixed cut | 0.4050 | cuts 0.30-0.375 | - | |
| cross-fitted û>tau (conf x area x limb bins) | 0.4058 | tau 0.16-0.21 | +0.0008 [-0.0015, +0.0031] | n.s. |
| oracle u>tau (uses labels) | 0.489-0.491 | | +0.085 | upper bound only |

Baseline recipe, erode1: 0.3856 -> 0.4162 (+0.031), so the effect is not specific to p2first.

Calibration (mean annotator-averaged IoU u per conf bin): 0.30-0.35 -> 0.214, 0.35-0.40 -> 0.223,
0.40-0.45 -> 0.284. With PQ ≈ 0.41 the break-even is u = PQ/2 ≈ 0.20-0.21, which sits at conf
≈ 0.30-0.35. **The 0.35 cut is therefore already at the Dinkelbach optimum.** Per-image thresholds
cannot beat one global tau (the R_i factor out); only better u-features can.

Area: predictions below 200 px have u = 0 (n = 53), 200-400 -> 0.27, 400-800 -> 0.37,
>1600 -> 0.49.

Why the masks are too wide (hypothesis, untested): YOLO-seg thresholds bilinearly upsampled
prototype logits at 0 (p = 0.5), and thin ribbons (median width 11 px) blur outward. A 1-px
boundary error on an 11-px ribbon changes IoU by about 0.1, which pushes many matches across 0.5.

### 3.3 Refined pipeline (the deployed path)
Same harness as `eval_v62_refiner.py` (141 canonical held-out files, OOF proposals ->
flip_consensus -> V11 refiner -> refined_constrained), refiner run on **CPU** (`v69_refined_morph.py`;
CUDA hidden, device strings remapped). **Parity:** the CPU run reproduces `v62_refiner_test.json`
exactly (baseline t0.45/c0.30 0.4517, 1004/464/589; p2first 0.4566).

| recipe, cell | base | erode1 | min area 200 | erode1 + min200 |
|---|---|---|---|---|
| baseline t0.45 c0.30 (V17 cell) | 0.4517 | 0.4303 (-0.021 [-0.028,-0.014]) | **0.4554 (+0.0037 [+0.0020,+0.0058])** | 0.4333 |
| baseline t0.50 c0.30 | 0.4519 | 0.4256 | **0.4556 (+0.0037)** | 0.4289 |
| p2first t0.45 c0.30 | 0.4566 | 0.4329 (-0.024) | **0.4583 (+0.0016 [+0.0004,+0.0032])** | 0.4343 |
| p2first t0.50 c0.35 | 0.4581 | 0.4302 (-0.028) | **0.4592 (+0.0011)** | 0.4318 |
| p2first t0.55 c0.35 | 0.4582 | 0.4229 (-0.035) | **0.4593 (+0.0011)** | 0.4241 |

Erosion hurts every refined cell (-0.021 to -0.036, all CIs below 0) and hurts more at higher t,
because higher t already thins the mask. The refiner's threshold t is the width knob. min-area 200
helps every cell (+0.0003 to +0.0037), removing only FP (TP and FN unchanged).
Full grid: `~/v69_kernels/audit/refined_morph.json`.

## 5. Submission format check (scripts/validate_submission.py, src/rle.py, export_v21_shard.py)
- RLE: pycocotools compressed counts of a Fortran-ordered 2048x2048 uint8 mask; size omitted;
  no quotes. Round trip (bytes -> str -> decode) is bit-identical for all exported masks of 707
  files. Counts characters (ASCII 48-111) contain no comma or quote, so the CSV is safe.
- Overlaps: the exporter does score-ordered pixel ownership and the validator asserts disjointness,
  which matches the host's rejection rule. No points lost.
- IDs: `<stem>_<k>` with sequential k; the host parses the image id as `split('_')[0]`. Test stems
  contain no `_`, so this is safe.
- Resolution: masks are produced at 2048 natively (retina masks / crop paste-back). No 1024 round trip.
- Empty images: allowed (no rows). Predictions on unannotated test images would be ignored, which
  is harmless.
- **Encoding loses nothing.** The only free points are the small-mask filter (§3.3). Width
  matters only for direct (unrefined) masks.

## 6. Recommendations
1. Export: set the minimum area to 200 (one constant, `MIN_AREA` in the refined export and in
   `clean_refined_predictions`; also the validator's `min_area`). Re-validate on the 141 clean
   files, expect +0.002-0.004. No retraining, no GPU.
2. Keep the refined masks as they are (no morphology). If a refiner-free pipeline is ever
   considered, apply erode1 to it (direct p2first + erode1 + min200 = 0.4303 on 707 OOF; on the
   141 files about 0.44, still below refined 0.457).
3. Re-score the V68 direct screens with erode1 too (CPU, cached views: `VARIANTS=base,erode1
   v69_metric_audit.py <recipe>`). A real detection gain should survive the width correction.
4. If time allows a modelling bet: follow tushhaaarrrr's direction. A soft annotator-mean semantic
   map (BCE + Dice) is the source of new u-evidence for a û > PQ/2 decision layer; their
   +0.065 is over fixed thresholds of a semantic map, not over a detector. The oracle gap here is
   0.085, but conf/area/position alone capture almost none of it.
5. Admin: SWEFil out; both final selections reproducible; a notebook that reproduces predictions;
   report the Dice/IoU/1:n distributions (the host's self-evaluation plots) in the technical report.

## 7. Reproduce
```
# on rogii, CPU only
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv/bin/python ~/v69_tools/v69_metric_audit.py ~/solar-filament-2026 p2first ~/v69_kernels/audit 24
.venv/bin/python -I ~/v69_tools/v69_audit_analyze.py ~/v69_kernels/audit/p2first_perfile.jsonl ~/v69_kernels/audit/p2first_analysis.json
.venv/bin/python -I ~/v69_tools/v69_fold_delta.py ~/v69_kernels/audit/p2first_perfile.jsonl
CUDA_VISIBLE_DEVICES= .venv/bin/python ~/v69_tools/v69_refined_morph.py ~/solar-filament-2026 ~/v69_kernels/audit/refined_morph.json
```
The scripts live in `~/v69_tools/` (also copied to `scripts/v69_*.py` in the local mirror).
