# V83 pre-registration: agreement-calibrated keep rule on the refined output (2026-10-11)

Written BEFORE any V83 number exists. CPU only (`CUDA_VISIBLE_DEVICES=""`, <= 24 processes). No detector or
refiner training, no test data, no external GT, no submission.
Code: `scripts/v83_extract.py` (features + targets), `scripts/v83_rule.py` (model, rule, evaluation).
Outputs: `outputs/v83/`.

## Why this is not V70 / V69
- V70 re-scored the PRE-refiner flip-view pool (it also changed the seed masks, i.e. vote75 thinning) and picked
  one score cut from a grid. The refiner then undid the gain. V83 works on the FINAL refined, post-ownership
  masks. The masks are never changed, only kept or dropped. Its features include refiner-internal signals
  that V70 did not have.
- V69's audit rule used conf x area x limb bins on the direct detector. V83 uses gradient boosting on about 25
  features, on the refined output.
- The keep threshold is not a tuned score cut. It is the Dinkelbach fixed point lambda/2 (lambda = the
  pooled PQ of the kept set), estimated on training folds only.

## Data
- The real pipeline, as in `scripts/audit_v80_extract.py`: p2first OOF 4-flip views, flip_consensus groups
  (score = sum/4), V73 `base` fold refiner (leak-free, k7 clamp, t0.45), and greedy pixel ownership by
  consensus score over ALL groups. It is re-run with extra instrumentation.
- Run-time assertion: at cut 0.30 the pooled PQ reproduces V73 base 0.4539 (max per-file count diff <= 1e-3).
- **Candidates:** every surviving (post-ownership) refined prediction with consensus score >= 0.15.
  - Baseline = candidates with score >= 0.30, which is the current pipeline.
  - Every candidate set is pixel-disjoint by construction, because ownership ran over all groups in score
    order. So any keep subset is a valid non-overlapping submission. Re-running ownership by y_hat is not done.
  - Disclosed simplification: when a mask is demoted, its pixels are not handed back to lower masks
    (conservative).

## Target (per candidate j in a file with R records)
- y_j = (sum over records r of the IoU of j with the GT instance it matches at IoU > 0.5) / R = q_j * s_j.
- Including j adds 0.5*R to the pooled PQ denominator and R*y_j to the numerator. So the optimal pooled
  rule is: keep iff y_j > lambda*/2.
- Sample weight = R. The payoff of a decision scales with R, so the weighted fit estimates E[R y | x] / E[R | x].
- R is a target and weight ONLY. It is never a feature.

## Features (fixed list; all available at test time)
- **Detector:** consensus score, n_views, mean / max / min member view score, score rank within the file.
- **Refiner:**
  - mean refined probability inside the final mask; max probability inside it;
  - mean probability inside the seed mask;
  - boundary margin = mean probability in a 2 px inner band minus mean in a 2 px outer band;
  - fallback flag (the refiner returned nothing, so the seed was used);
  - seed area; area ratio pre-ownership refined / seed; ownership retention post / pre;
    IoU of seed vs final mask.
- **Shape:** log area (final), elongation (minAreaRect long side^2 / area), width (area / long side).
- **Photometry / position:**
  - limb distance (centroid distance / Otsu disk radius);
  - local contrast (bg - fg) / bg, with the ring = 25x25 dilation minus 7x7 dilation;
  - darkness = fg / disk mean;
  - disk mean, disk std, low-frequency CV.
- **Neighbours:**
  - number of other candidates within 25 px; max score among them;
  - number of candidates with score >= 0.30 in the file;
  - number of candidates in the file.

## Model (fixed, no tuning)
- LightGBM 4.5.0 from `outputs/v70/pylib`. sklearn and LightGBM are absent from `.venv`, so this is the same
  disclosed install as V70.
- Objective `cross_entropy` on y in [0,1], weight R. Parameters: num_leaves 31, learning_rate 0.05,
  300 rounds, min_data_in_leaf 50, bagging_fraction 0.8 (freq 1), feature_fraction 0.8, seed 83.
- **Cross-fitting by the V61 folds** (`outputs/v61_folds.json`): fold k is predicted by a model trained on
  the other 4 folds.
- **lambda_k** is estimated on the 4 training folds only, from nested OOF predictions (for each training fold
  j, a model trained on the 3 remaining folds), by Dinkelbach:
  - start lambda = training-fold baseline PQ;
  - iterate lambda <- PQ_train(kept set);
  - stop when the change is < 1e-6 or after 50 iterations.
- Fold k then keeps candidates with y_hat > lambda_k/2.

## Arms
- **PRIMARY (gated), "full":** any candidate (score >= 0.15) is kept iff y_hat > lambda/2. The rule can
  promote and demote.
- **Secondary, "demote-only":** only score >= 0.30 candidates are eligible; those are kept iff y_hat > lambda/2.
- **Secondary, "promote-only":** all score >= 0.30 candidates are kept; 0.15-0.30 candidates are added iff
  y_hat > lambda/2.
- For both secondary arms, lambda uses the same arm-specific Dinkelbach on the training folds.
- **Oracle (headroom only, not cross-fitted):** the true y is known, keep iff y > lambda*/2, with Dinkelbach
  on all 707 files.

## Metric and gate
- Pooled PQ over all annotator records, 707 files, exact per-record matching (`pq_counts` on the kept
  columns of the stored per-record IoU matrices).
- Comparison is against the baseline (cut 0.30, 0.4539). Paired file bootstrap: 2000 resamples, seed 83,
  `v71_suppress.boot` style. Also reported: per-fold deltas, TP/FP/FN, promoted and demoted counts,
  feature importance.
- **CONTINUE** iff d >= +0.010 AND P(d>0) >= 0.90 AND >= 4/5 folds positive. **KILL** iff d < +0.005.
  Otherwise INCONCLUSIVE.
- Any change after seeing a number is a new labelled experiment, not V83.
