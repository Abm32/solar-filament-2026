# V61 Phase B pre-registration (written 2026-10-05 ~11:30 IST, BEFORE any OOF number exists)

Status at writing: baseline folds 0-3 training (epoch ~35/80), no held-out inference run,
no PQ computed on any V61 fold. Nothing below was tuned on held-out data.

## Formulation B (frozen)
B = V17 recipe (identical architecture, data, augmentation, schedule, seeds, 2 stages) with the
Ultralytics mask loss replaced by  BCE + 1.0*Focal(gamma=2) + 1.0*Dice  (scripts/v61_seg_loss.py).
- Applied in BOTH stages (80 ep mask_ratio=2, then 20 ep mask_ratio=1): B is a different recipe,
  not a different continuation.
- Weights w_focal=1.0, w_dice=1.0 chosen a priori (same scale as the BCE term; Dice is in [0,1]
  per instance, BCE term is O(0.1-1) per instance). NOT to be changed before the decision below.
- Box/cls/dfl losses untouched. The seg term keeps Ultralytics' hyp.box gain (7.5).
- Verified: zero weights reproduce upstream bit-for-bit (tests/test_v61_seg_loss.py), gradients
  finite, patch is invoked during real training (936 calls in a 1-epoch probe).

## First comparison (fold 0 only)
- A = baseline fold 0, B = focal_dice fold 0 (same files, scorer, TTA, consensus, ownership).
- Noise control: A' = baseline fold 0 retrained with a different seed (20261005). The A-vs-A' paired
  delta prices retraining noise on THIS fold; B is judged against it, not against zero.
- Direct detector only (no V11 refiner). Confidence grid is the declared 8-value grid; each recipe
  is scored at ITS OWN best confidence AND at the baseline's best confidence (to separate a real
  objective effect from a threshold shift).

## Reported for A, A', B (fold 0)
direct PQ, TP/FP/FN, mean matched IoU (iou_sum/TP), recall by GT category, per-file paired dPQ
with 2000-resample file bootstrap. Categories reuse diagnose_v59_0_recall exactly (priority order):
thin = elongation>=4 ; else small = area<1000px (full-res) ; else faint = GT mean intensity in the
bottom quartile of the image's disk ; else normal. GT measured at 512px as in that script.

## Decision rules (fixed now)
Let d = PQ(B) - PQ(A) at each recipe's own best confidence, and n = |PQ(A') - PQ(A)| (noise).
- CONTINUE to full 5-fold B only if ALL hold:
  (1) d >= +0.010 ; (2) d >= 2*n ; (3) bootstrap P(d>0) >= 0.90 ; (4) the gain is not explained by a
  threshold shift (d at the baseline's best confidence is also > 0) ; (5) recall gain on
  small+thin+faint pooled is at least as large as on normal (direction check, not a gate on size).
- KILL the loss line if d < +0.005 or d < n.
- Anything in between: report as inconclusive; do NOT sweep weights on this fold. A weight sweep is
  only justified after a CONTINUE verdict and must be judged on folds not used to choose it.
A single fold has ~141 files, so even a CONTINUE verdict is a direction, not a result; the full
5-fold paired bootstrap is what decides promotion.
