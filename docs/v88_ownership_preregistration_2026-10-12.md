# V88 pre-registration: pixel-ownership alternatives (2026-10-12)

Written BEFORE any V88 number exists. CPU only on rogii, no GPU, no training of networks, no submission.
No test data, no test labels, no MAGFiLO Dataverse test GT, no leaked CSV, no SWEFil.
Code: `scripts/v88_extract.py` (stage 1, refiner forward + ownership arms), `scripts/v88_premodel.py`
(ordering model for arm B_yhat), `scripts/v88_score.py` (stage 2). Outputs: `outputs/v88/`.

## Motivation
V85 (`docs/v85_error_analysis_2026-10-12.md`, ownership drill-down): 610 FN (19%) are lost at the greedy
score-order ownership step (430 clipped, 180 killed below MIN_AREA). In 71% a higher-scored kept neighbour
overlaps the GT at IoU 0.05-0.5. Naive upper bound +0.036, V85 "realistic" +0.006..+0.012.

## Pipeline (unchanged except the ownership step)
p2first OOF 4-flip views -> flip_consensus groups -> V84 soft fold refiner (k7 clamp, t0.50) -> per-group
pre-ownership crop `pre_i` (exactly as v85_extract) -> **ownership** -> alive masks (>= core.MIN_AREA)
-> features (`v83_extract.refine_feats` body, recomputed on the arm's final masks; `retain` = final/pre area)
-> candidate rows (score >= 0.15) -> cut 0.30 -> MIN200 -> cross-fitted demote (V84 rule).

**Demote handling:** ownership changes the features, so for EVERY arm the features are recomputed on that
arm's own masks and the demote LightGBM + lambda are REFIT cross-fitted by fold on that arm
(`v84_bundle.arrays/crossfit/run_arm`, unchanged code). Ownership is not applied after demote.

**Scope of the change:** the alternative rule acts only among groups with detector score >= 0.30 (the only ones
that can be kept). Groups with score < 0.30 are then processed greedily by score on the leftover pixels, as
today (in V84 they are already processed after every >= 0.30 group, so this keeps them identical in role:
they feed demote training rows and neighbour features only).

## Arms (all fixed now)
- **G (control)**: greedy by detector score (V84). Must reproduce V84 OOF PQ **0.45898** (assert |d| < 5e-5)
  and its candidate IoU columns must equal `outputs/v84/extract_soft.pkl[0.5]` (max |d| < 1e-6).
- **A_argmax**: per-pixel argmax of refiner probability among the >= 0.30 groups whose pre crop contains the
  pixel (tie -> higher detector score). Masks below MIN_AREA are killed and argmax recomputed among survivors,
  iterated to a fixed point. Then MIN200 / demote as usual.
- **B_pin**: greedy ownership, order = mean refiner probability over the pre crop (descending).
- **B_yhat**: greedy ownership, order = predicted y-hat of a demote-style LightGBM (V83 features/params)
  computed on the pre-ownership crops. The fold-k model is trained on the V84 extract rows of folds != k only
  (`v88_premodel.py`), so the order of fold-k files never uses fold-k labels.
- **C_x (keep the clipped mask, trim the neighbour)**: greedy by score; if candidate i would lose more than a
  fraction x of its pre area to already-owned pixels, then for each occupier j the contested pixels go to the
  smaller mask by pre area (if i is smaller, j is trimmed; else i is clipped). After the >= 0.30 pass, any mask
  trimmed below MIN_AREA is killed (its pixels freed). Grid x in {0.1, 0.2, 0.3, 0.5, 0.7}.
  **x is cross-fitted**: for held-out fold k, x_k = argmax pooled PQ over the other 4 folds; fold k is scored
  with x_k. The reported arm is **C_cf**. (Caveat, accepted: the 4-fold PQ used to pick x_k uses that grid
  point's OOF demote, whose models for folds j != k saw fold k; this only affects the choice among 5 values.)
- **D_small (my variant, fixed before results)**: greedy ownership, order = pre area ascending (thin, small masks
  claim first; targets the "fat neighbour swallows the filament" pattern).

## Primary and decision
- Primary metric: 5-fold OOF pooled PQ (707 files, records pooled) of each arm vs control G (0.45898).
- Paired bootstrap by file (2000 draws, `v83_rule.boot`, seed 88): delta, 95% CI, P(d > 0); per-fold deltas.
- **All arms are reported.** The candidate = arm with the largest delta among {A_argmax, B_pin, B_yhat, C_cf,
  D_small} (the 5 C grid points are diagnostic only and cannot be the candidate).
- **CONTINUE** iff the candidate has d >= +0.005 AND P(d > 0) >= 0.95 AND delta > 0 in >= 4/5 folds.
  Otherwise **KEEP GREEDY**. A non-candidate arm that also clears all three bars is noted, not promoted.
- CONTINUE means: implement it in the export path and do a GPU-free parity check before any submission
  decision (separate step). Expected public effect at the 0.85 ratio is small (V85 section 3).

## Secondary (diagnostic only)
TP / FP / FN, kept count, number of the V85 ownership FN recovered.
