# V75 pre-registration: E_p2x2 voting ensemble through the r3 refiner (2026-10-10, before any V75 number)

## Why
V66 closed voting ensembles on a false premise ("E_all4 is the upper bound on any subset";
`v69_history_audit` 1f). E_p2x2 = {p2first, p2first_cp} scored direct crossfit +0.0063, CI
[+0.0025, +0.0101], 5/5 folds. The refiner absorbs ~80% of detector SQ gains, but a voting
ensemble mostly changes ranking/RQ, which the V62 hybrid test showed survives refinement. This
tests whether the gain survives r3.

## Harness (CPU only, no training, no test data, no submission)
`scripts/v75_ensemble_refined.py`. Identical to the V71 r3 cache (`v71_suppress.cache_work`):
141 canonical held-out files, own-fold OOF views, `F.groups_of` consensus with score =
sum(view scores) / n_views, r3 refiner (`outputs/v63/r3_refiner.pt`) with flip TTA,
`clean_refined_predictions` refined_constrained, t0.45 / c0.30. The ONLY change: views are the
concatenation of p2first's and p2first_cp's 4 flip views (8 views), exactly as V66 defines an
ensemble. Reference = `outputs/v71/cache.pkl` (r3 on p2first, 0.4595, 1020/448/573).
Self-check: members=[p2first] must reproduce the reference counts per file.

## Primary
Refined PQ at t0.45/c0.30, E_p2x2 minus r3, paired file bootstrap (2000, seed 75), 141 files.
- CONTINUE iff dPQ >= +0.010 and P(d>0) >= 0.90.
- KILL iff dPQ < +0.005.
- Otherwise INCONCLUSIVE.
Secondary (descriptive, not gated): same with MIN200 applied; per-fold deltas.

## Known limits, stated now
- r3 was trained on single-model p2first proposals; ensemble proposals have different scores.
  A refiner retrained on ensemble OOF proposals is not tested here.
- 141 files: SE ~0.005, so a true +0.008 passes CONTINUE about a third of the time.
- Deployment would need a p2first_cp all-data (fold-99) model, which does not exist yet.
