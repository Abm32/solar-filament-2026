# V66 pre-registration: consensus-VOTING ensembles of OOF detectors (2026-10-07, before any result)

## Why the earlier ensembles failed, and what is different here
Earlier union ensembles (E028, V49, V53) pooled every member's proposals and de-duplicated with NMS.
A detection made by ONE member therefore kept its full confidence, so the union raised recall and
false positives together (union beat the best single source at 1 of 7 confidences, max +0.0005).
Here the members VOTE: a detection's score is (sum of its matched view scores) / (number of views),
with zero for every view that missed it. A one-member detection can score at most 1/N; a detection
all members agree on keeps its full score. This is exactly the existing `flip_consensus` rule
(`detector_tta.cluster_views`, anchor = highest-score mask, match IoU 0.5), applied to the
concatenation of every member's four flip views (4N views) instead of one member's four.

## Evidence that there is variance to remove (measured, not assumed)
Same architecture, different training seed, fold 0: aggregate PQ moved +0.0049 but 25 of 141 files
moved by more than 0.10 PQ (per-file sd 0.083), and the moves were balanced (49.6% of files up).
Large per-file disagreement with no aggregate difference is the setting where averaging members helps.

## Members (all out-of-fold on the same 707 files, same V17-validated views/scorer)
B = p2first, Bcp = p2first_cp, A = baseline, A2 = baseline_seed2, B2 = p2first_seed2 (folds 1-4
are training now; fold 0 exists). Invariant tested in code: an ensemble of a model with itself
reproduces that model exactly (same proposals, same scores).

## Ensembles (equal weights; no weight is tuned)
| name | members | compared against |
|---|---|---|
| E_all4 | B, Bcp, A, A2 | B   **(PRIMARY, stage 1)** |
| E_p2x2 | B, Bcp | B |
| E_mix | B, A | B |
| E_base2 | A, A2 | A   (pure seed-variance control: no architecture gain involved) |
| E_p2seed | B, B2 | B   **(stage 1b, the deployable pattern; needs B2 folds 1-4)** |
| E_p2all | B, B2, Bcp | B |

## Measurement (stage 1, direct detector, all 707 files)
Cut selection is CROSS-FITTED: for fold k the confidence cut is the pooled-PQ maximiser over the OTHER
four folds (declared grid 0.20..0.55), then applied to fold k; the same for every recipe and for the
single-model references, so an ensemble cannot win by having its cut tuned on the files it is scored
on. Reported too: PQ at the fixed cut 0.35 and each recipe's own best cut (not gating).
Paired bootstrap over files (2000 resamples) of ensemble minus reference; FP ratio from the pooled
per-record counts.

## Gate
STAGE 1 (E_all4 - B):  CONTINUE iff pooled d >= +0.010 AND P(d>0) >= 0.90 AND 95% CI lower bound > 0
AND >= 4 of 5 folds positive AND FP ratio <= 1.30.  KILL iff d < +0.005 or < 3 of 5 folds positive.
Otherwise inconclusive.  E_all4 is the UPPER BOUND on what any cheaper subset can give at equal
weights; if it fails, voting ensembles are closed (no subset is expected to beat the full set).
STAGE 1b (E_p2seed - B), same rule: decides whether the deployable seed-ensemble of the winning
architecture is worth building.
STAGE 2 (only after a stage-1/1b CONTINUE): the refined pipeline on the 141 canonical held-out files
(V17 cell t0.45/c0.30, V11 refiner and the r3-recipe refiner): ensemble+r3 vs p2first+r3 (0.4595);
PASS iff delta >= +0.010 AND P(delta>0) >= 0.90. The direct-detector gains of p2first and
copy-paste were absorbed by the refiner, so a direct gain alone does NOT authorise a submission.

## Not allowed after seeing results
No weighting of members, no dropping a member, no changing the grid or the cross-fit rule, no
re-labelling a reported-only ensemble as primary. A pass authorises BUILDING the all-data
ensemble (members must be retrained on all 707 files), not submitting.
