# V67 pre-registration: does a co-temporal magnetogram carry information about filament errors? (2026-10-07, before any result)

## Rules and data (firewall)
- Host ruling (discussion 743521): co-temporal GONG magnetograms are allowed as a TRAINING-ONLY auxiliary
  signal, never at inference. Neighbouring frames at inference are forbidden. This diagnostic uses
  magnetograms only to ANALYSE train files; it is not an inference path.
- Only the 707 TRAIN images' magnetograms are fetched (`scripts/fetch_v67_magnetograms.py` reads the train
  directory only, asserts so). No test image, test label or test-time magnetogram is touched.
- Source: GONG `zqa` full-disk LOS Bz (Gauss), same GONG site as the H-alpha frame (site-matched), nearest
  time, quick-reduce `bqa` fallback. Registered to the 2048 H-alpha grid by disk-circle similarity
  (`scripts/v67_register.py`; limb fitted per frame by steepest-descent along 72 rays).
- Registration was validated on 11 smoke files BEFORE this document: correlation of |Bz| with H-alpha plage
  excess +0.057 (correct) vs -0.047 (rows flipped) vs -0.008 (60 px shift); overlay of |Bz|>150 G contours
  sits on plage. That validation is the only thing looked at so far.

## Why this is a diagnostic first
Training an auxiliary head costs ~3.5 GPU-h per 5-fold arm. A magnetogram helps a training-only aux head only if
it carries information about the targets that image + size + confidence do not already carry. If it carries
none, no aux head can help, and the line is killed without any GPU. Passing the diagnostic is NECESSARY, not
sufficient: the aux still has to survive the real pipeline (detector + V11/r3 refiner) at +0.010.

## Frame exclusion (geometry only, fixed BEFORE any D1/D2/control number)
The H-alpha frames are pre-normalised to a common disk: 672 of 691 fitted disks sit at centre
(1023.5 +- 1, 1024.3 +- 1) with radius 904.6 +- 2. The other 19 are failed limb fits (radius 861..1036, centres
off by up to 80 px), so their registration is wrong. Rule: exclude a frame iff its fitted radius differs from
the median (904.6) by more than 6 px or its centre is more than 6 px from (1023.5, 1024.3). This uses only the
fit geometry, never labels or detector output. Also reported: results with the frames included (not gating).
Magnetogram coverage: 691 of 707 train files (16 days have none); 575 within 15 min and 623 within 30 min of
the H-alpha frame; 35 between 30 and 120 min; 33 further (up to 22 h). All used (quiet-sun structure changes
slowly; large gaps are reported as a covariate), none dropped for gap.

## Features (per mask, computed at 512x512 = 2048/4; mask area-downsampled)
Registered Bz, smoothed (Gaussian sigma 2 px at 512). From it: |Bz|; polarity masks pos = (Bz > +25 G), neg = (Bz < -25 G). PIL set = pixels within 6 px of BOTH a pos and a
neg pixel (dilation radius 6 at 512 px). d_PIL = Euclidean distance (px) from each pixel to the nearest PIL
pixel, clipped at 40 (also 40 everywhere if the frame has no PIL). (Wording made exact before any result.)
Per mask: mean/max |Bz|; fraction of pixels with |Bz| > 150 G (sunspot-like strong
flux); mean d_PIL; fraction of pixels with d_PIL <= 5 px; mean signed Bz and its absolute value (unipolar =
sunspot-like). Nothing is tuned: thresholds above are fixed here.

## Positive control (validity gate, not a result)
For each GT filament instance with a registered magnetogram, compare mean d_PIL of the instance with the same
mask translated by a random vector that stays on-disk (20 random placements, seeded). AUC(real mask closer to
a PIL than its random translates) must be >= 0.60. If not, the features or registration cannot see filaments
at all and the diagnostic is INVALID (fix registration; do not read D1/D2).

## D1: do magnetogram features separate true from false detections?
Proposals: p2first 5-fold OOF direct detector (`outputs/v61_oof_cache/p2first`), flip-consensus + pixel
ownership, all proposals with score >= 0.20. Label TP if IoU > 0.5 with any GT instance of that file (any
annotator record), else FP.
Baseline model: logistic regression on [score, log area, elongation (major/minor of the mask)].
Augmented model: baseline + the magnetogram features above (standardised). Both trained with the V61 folds
(file-grouped 5-fold cross-fit; fit on 4 folds, predict the 5th). Metric: out-of-fold ROC-AUC; paired file
bootstrap (2000) of AUC(augmented) - AUC(baseline).

## D2: do magnetogram features separate detected from missed GT filaments?
GT instances; label DETECTED if some proposal with score >= 0.35 has IoU > 0.5, else MISSED (per annotator
record). Baseline: log area, elongation, mean darkness-contrast is NOT used (kept to geometry only to avoid a
second image model): [log area, elongation]. Augmented: + magnetogram features. Same cross-fit and bootstrap.

## Gate (mechanical)
Positive control must pass first. Then:
- CONTINUE to a training-only auxiliary experiment iff, for D1 OR D2, delta-AUC >= +0.030 and the 95% bootstrap
  CI lower bound > 0, and the sign is positive in at least 4 of the 5 folds.
- KILL the magnetogram line (no training, no GPU) iff neither D1 nor D2 reaches delta-AUC >= +0.015.
- Anything in between: inconclusive; reported, not acted on, no re-reading.
Also reported, not gating: coverage (files with a magnetogram within 30 min), AUC of each single feature,
and the FP/TP separation restricted to proposals within the confidence band 0.20-0.45 (where errors live).

## Not allowed after seeing results
No new features, thresholds, models or subsets chosen after the numbers; no relabelling of a reported-only
quantity as primary; a pass authorises designing the auxiliary-head experiment (own pre-registration, own
gate vs seed noise and a seed-only rerun), not a submission.
