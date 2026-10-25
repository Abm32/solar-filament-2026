# V70 result: full-pool re-scorer is a KILL (2026-10-10)

**Verdict: KILL.** On the real pipeline the re-scorer lowered PQ by 0.0099 (95% CI [−0.018, −0.002], P(d>0) = 0.009). The pre-registered kill line was dPQ < +0.005.

**Setup**
- **Pre-registration:** `docs/v70_rescorer_preregistration_2026-10-10.md` (sha256 ebbc606b…), written 07:23 UTC, before any V70 number.
- **Code:** `scripts/v70_rescorer.py`. Outputs are in `outputs/v70/`: `features.pkl`, `scores.pkl`, `direct.json`, `refined.json`, `importance.json`.
- **Rules followed:** CPU only, nothing submitted, no test data read.
- **Disclosed deviations:**
  - LightGBM 4.5.0 was installed `--no-deps` into `outputs/v70/pylib`; `.venv` was not touched.
  - sklearn is absent, so the models were trained with the native `lgb.train` API. The pre-registered parameters map 1:1.
- **Candidates:** 36,863 across 706 files. One file has no proposals. 55.9% of candidates match at least one record.

## Primary metric: real pipeline (p2first OOF + r3 refiner, 141 canonical files)
The refiner ran on CPU on the re-scorer's selected masks, so nothing was approximated. The r3 baseline arm reproduced `outputs/v63/r3.json` file by file (asserted).

| arm | PQ | SQ | TP | FP | FN | cuts per fold |
|---|---|---|---|---|---|---|
| r3, fixed V17 cell | 0.4595 | 0.689 | 1020 | 448 | 573 | – |
| r3, cross-fitted cut | 0.4572 | 0.690 | 997 | 419 | 596 | 0.35 / 0.30 / 0.30 / 0.35 / 0.30 |
| **re-scorer, V61-fold cross-fit** | **0.4473** | 0.694 | 956 | 416 | 637 | 0.275 / 0.25 / 0.275 / 0.275 / 0.275 |
| re-scorer, date-grouped | 0.4460 | 0.693 | 970 | 450 | 623 | – |

| comparison | dPQ | 95% CI | P(d>0) |
|---|---|---|---|
| **re-scorer (V61) − r3 (cross-fitted), gated** | **−0.0099** | [−0.0182, −0.0021] | 0.009 |
| re-scorer (V61) − r3 (fixed cell) | −0.0121 | [−0.0203, −0.0045] | 0.002 |
| re-scorer (date) − r3 (cross-fitted) | −0.0112 | [−0.0204, −0.0026] | 0.007 |

The re-scorer is worse on all 5 V61 folds (per-fold dPQ −0.022 to −0.002).

## Secondary metric: direct detector, 707 files
| arm | PQ | dPQ vs consensus @0.35 (95% CI) |
|---|---|---|
| consensus, fixed 0.35 | 0.4060 | – |
| consensus, cross-fitted cut | 0.4046 | – |
| re-scorer, V61 cross-fit | 0.4221 | +0.0161 [+0.012, +0.020], P = 1.00 |
| re-scorer, date-grouped | 0.4233 | +0.0173 [+0.013, +0.021] |

The date-grouped cross-fit matches the V61 cross-fit, so same-day leakage did not inflate the re-scorer.

## Why the direct gain does not transfer
- **The direct gain is mostly mask thinning, which the refiner already does.**
  - Of the selected candidates above the cut, 70.6% are the fused vote75 masks. 18.3% are a non-anchor view mask, and only 11.1% are the original anchor.
  - V69 already showed that vote75 thinning is worth +0.020 on the direct detector and nothing after the refiner.
- **The thinner seeds cost recall after refinement.**
  - `refined_constrained` limits each refined mask to a 7 px dilation of its seed mask.
  - The refiner was also trained on anchor seeds.
  - Result: TP fell from 997 to 956.
- **The ranking itself added nothing.**
  - Spearman correlation of the re-scorer score with the target p is 0.581, versus 0.582 for the V17 consensus score.
  - Feature importance is dominated by the consensus score (37% of gain). Contrast, limb distance, shape and neighbour features each add 1–6%.
  - The agreement-fraction target appears to be mostly unpredictable from these features. This is consistent with V69: missed filaments are mostly drawn by only one annotator.

## Status
- The re-scoring and selection line is closed with these features and this candidate pool.
- **Not run, and would need its own pre-registration:** re-scoring with the anchor masks kept as they are (score-only re-ranking, i.e. "V70b"). The flat Spearman result above suggests it is unlikely to reach +0.010.
- The V69 levers #2 (fragment and hallucination suppression after the refiner) and #3 remain open.
