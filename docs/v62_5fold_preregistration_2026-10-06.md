# V62 `p2first` 5-fold decision rule (written 2026-10-06 ~07:55 IST, BEFORE folds 1-4 exist)

Status at writing: fold 0 done for baseline (A), baseline_seed2 (A'), p2first (seed 42, 2); folds 1-4
of p2first are training; no held-out number exists for folds 1-4 of ANY recipe except the baseline
(A), whose pooled Phase-A PQ (0.3856) is already fixed. Nothing below has been tuned on folds 1-4.

## Comparison
B = p2first (seed 42) vs A = baseline (seed 42), pooled over all 707 held-out files, direct
detector (no refiner), same scorer / TTA / consensus / pixel ownership, paired by file.
Noise control = baseline_seed2 (A') on every fold; n = |PQ(A') - PQ(A)| pooled over the 5 folds at
the baseline's pooled best confidence.
Cut: the baseline's pooled best confidence (0.35 from Phase A, fixed in advance) is the PRIMARY
cut; B's own best cut is reported, not used for the gate (removes any threshold-selection edge).

## CONTINUE (-> matched full pipeline: refiner on p2first proposals, then all-data retrain) iff ALL:
1. pooled d = PQ(B) - PQ(A) at conf 0.35  >=  max(+0.010, 2n);
2. paired bootstrap over files (2000 resamples): P(d > 0) >= 0.90 AND 95% CI lower bound > 0;
3. at least 4 of 5 folds have d_fold > 0 at conf 0.35 (effect is not one lucky split);
4. small-object recall (GT category `small`) rises pooled, AND B's own-best-cut d > 0;
5. proposals/image and FP/image at conf 0.35 are each <= 1.30x the baseline's (no cloud).
## KILL iff pooled d < +0.005, or fewer than 3 of 5 folds positive.
## Otherwise INCONCLUSIVE: no further compute on p2first; record and move on.

## Explicitly NOT part of this gate
- Thin-object recall (fold 0 showed no gain; n is small): reported, not gated.
- Any refiner / all-data / submission step. A pass here authorises BUILDING the matched pipeline,
  not submitting. Submission requires that pipeline to be evaluated out-of-fold first.
- Head/neck tuning, a third architecture, or weight changes.

## Expectation stated in advance (so it can be checked)
Fold 0 gave +0.0227 (seed 42) and +0.0128 (seed-matched seed 2). I expect the pooled effect to be
smaller than the fold-0 headline, roughly +0.010 to +0.018. A pooled d near +0.010 would sit on
the gate; I will not re-read the gate if it does.
