# V70 pre-registration: re-scorer over the full flip-view pool (2026-10-10)

Written BEFORE any V70 number exists. Motivation: V69 (`docs/v69_error_analysis_2026-10-10.md`)
found 62% of refined-pipeline FN already have an IoU>0.5 mask in the candidate pool (180 below the
cut, 175 single-view masks consensus did not pick); score-oracle bound +0.048. The metric scores
each annotator record separately; including a candidate adds 0.5*n_rec to the PQ denominator
whatever happens, so its pooled-PQ value is positive iff 2*p*e > PQ, where p = fraction of the
file's records it matches and e = its expected matched IoU. Hence the target/score below.

CPU only (`CUDA_VISIBLE_DEVICES=""`, `nice -n 10`). No training of detectors/refiner, no test data,
no submission. LightGBM is not in `.venv` and sklearn is not either: LightGBM 4.5.0 is installed
`--no-deps` into `outputs/v70/pylib` (isolated; `.venv` untouched) and added via PYTHONPATH.
Code: `scripts/v70_rescorer.py`. Outputs: `outputs/v70/`.

## Candidates (per file, from `outputs/v61_oof_cache/p2first/fold*`, 707 files)
- Clusters = the flip_consensus groups (V17 `cluster_views`, match IoU 0.5, at most one mask per
  view per group, strongest mask as anchor) - this is the IoU de-duplication.
- Candidates = every member view mask of every cluster (score floor 0.1 as cached) PLUS, for
  clusters with >=2 members, one fused "consensus" mask = pixels in >=75% of member masks (V69
  vote75; skipped if <20 px).

## Features (fixed, 21)
cand_score (view score; fused = max member score), is_fused, is_anchor, grp_n_views,
grp_cons_score (sum/4 = V17 score), grp_max_score, grp_min_score, grp_score_std,
cand_mean_iou_members (mean IoU to the cluster's other member masks; 0 if none),
grp_mean_pairwise_iou, log_area, elong (minAreaRect length^2/area), width (area/length),
contrast ((ring-fg)/ring, ring = 21x21 dilation minus candidate), darkness (fg mean / disk mean),
rdist (centroid distance / disk radius, Otsu disk), nbr_max_iou (max IoU to other clusters'
anchors), nbr_n_overlap (# other anchors intersecting), nbr_cov_stronger (fraction of candidate
pixels covered by anchors of clusters with higher grp_cons_score), file_n_groups_035 (# clusters
with grp_cons_score >= 0.35), disk_mean.

## Targets
For each annotator record r of the file: m_r = max IoU of the candidate with r's GT.
p = mean_r [m_r > 0.5]; e = mean of m_r over records with m_r > 0.5 (defined only when p > 0).

## Model (fixed, no tuning)
LightGBM: model P = objective `cross_entropy` on p (all candidates); model E = `regression` (L2)
on e (candidates with p>0). Both: num_leaves 31, learning_rate 0.05, n_estimators 300,
min_child_samples 100, subsample 0.8 (bagging_freq 1), colsample_bytree 0.8, seed 70.
Score s = clip(P_hat,0,1) * clip(E_hat,0,1).

## Cross-fitting
- PRIMARY: V61 file folds (`outputs/v61_folds.json`): a file's candidates are scored by models
  trained on the other four folds only.
- REPORTED (not gated): date-grouped 5 folds - unique days (first 8 chars of the file name),
  sorted, assigned largest-first to the fold with the fewest files. Detector OOF stays V61-fold
  (detector-side same-day leakage is unchanged and disclosed).

## Selection
Per cluster keep the single candidate with the highest s. Direct pipeline: clusters in
descending s; drop a candidate if >50% of its pixels are already owned, else clip to unowned
pixels, drop if <20 px; then cut s >= c. Refined pipeline: the per-cluster selected masks with
score s go through the r3 refiner and the UNCHANGED `clean_refined_predictions` (t = 0.45,
refined_constrained, ownership by descending s), then cut s >= c.
Cut grid for s: 0.050, 0.075, ..., 0.500 (19 values). The cut used for a file in fold k is the
one maximising pooled PQ over the evaluated files NOT in fold k (cross-fitted, never on the
scored fold). The same cross-fitting is applied to every comparator.

## Metrics
- PRIMARY (gated): real pipeline on the 141 canonical held-out files (V63/V65 setup; r3 refiner
  `outputs/v63/r3_refiner.pt`, run on CPU): re-scorer arm vs r3 baseline arm (consensus
  proposals, V17 scores, c from `v50.CONFIDENCES`), both cuts cross-fitted by V61 fold.
  Run-time check: the baseline arm must reproduce `outputs/v63/r3.json` per file at t0.45/c0.30
  (0.4595). Also reported: vs the fixed V17 cell 0.4595.
- SECONDARY: direct detector, all 707 files: re-scorer (cross-fitted cut) vs consensus at the
  fixed 0.35 (0.4060); also vs consensus with cross-fitted cut.
- Paired file bootstrap (2000, seed 70): dPQ, 95% CI, P(d>0). Also reported: TP/FP/FN, SQ,
  per-fold deltas, date-grouped results, feature importances.

## Gate (on PRIMARY, V61-fold cross-fit)
CONTINUE iff refined dPQ >= +0.010 AND P(d>0) >= 0.90 AND CI lower bound > 0.
KILL iff refined dPQ < +0.005. Otherwise INCONCLUSIVE.
Any change to features/model/targets/selection after seeing a number is a new, labelled
experiment, not V70.
