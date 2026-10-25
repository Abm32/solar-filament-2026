# V72 pre-registration: "stack the small positives" post-process bundle (2026-10-10)

Written BEFORE any V72 number. CPU only, no training, no test data, no submission, no SWEFil.
Code: `scripts/v72_bundle.py`. Outputs: `outputs/v72/`.

## Why
Three free post-processes each gave a small positive on the refined pipeline:
the min-area-200 rule (V69 metric audit, +0.0016 to +0.0037), V71 R3 (+0.0030) and V71 R2 (+0.0012).
Each alone is below any submission bar. V72 asks whether they stack. Every parameter is fixed
from prior docs. Nothing is re-tuned here.

## Pipeline under test (primary)
- p2first OOF proposals → r3 refiner (CPU) → V17 cell (t 0.45, `refined_constrained`, cut c 0.30).
- Population: the 141 canonical held-out files (all 5 V61 folds).
- Input: the V71 cache `outputs/v71/cache.pkl`. It holds the kept predictions (score ≥ 0.30)
  with area, score and contrast. V71 asserted that it reproduces `outputs/v63/r3.json` per file
  (PQ 0.4595). V72 re-asserts the unmodified arm equals 0.4595, 1020/448/573.

## Bundle components (drop-only; applied after the refiner's ownership step, to the final kept masks)
- **(a) MIN200.** Drop the prediction if its final mask area is < 200 px.
  - Source: V69 metric audit.
  - Applied post hoc, exactly as audited. In-loop `MIN_AREA` stays 20, so ownership is unchanged.
- **(b) R3.** Drop if contrast < 0.06 and score < 0.5.
  - τ = 0.06 was V71's cross-fitted choice in all 5 folds.
- **(c) R2.** Drop if area < 300 and score < 0.5.
  - (300, 0.5) was V71's choice in 4 of 5 folds.
  - Fold 4 chose (150, 0.5). That per-fold V71 cross-fit is reported as a sensitivity arm, "bundle-xfit".
- All rules are evaluated on the original kept set. Their union is dropped, so order does not matter.

**Disclosed optimism.** τ and (A, S) were selected by V71 on these same 141 files. The selection
was cross-fitted, but the single fixed value used for export is the mode across folds. MIN200 was
chosen on 707 direct-detector files and the area-sweep cells. This bias is small, and it is the
reason the bundle-xfit arm is reported.

## Arms (all vs the r3 baseline at the V17 cell, 0.4595)
- **PRIMARY:** bundle = a + b + c with the fixed values above.
- Leave-one-out: b + c, a + c, a + b.
- Singles: a, b, c (descriptive).
- bundle-xfit: a + b + c, with V71's per-fold R2 choice.

## Statistics
- Paired file bootstrap, 2000 resamples, seed 72.
- Reported per arm: dPQ, 95% CI, P(d>0), and SQ/TP/FP/FN.
- Per-fold dPQ for the primary arm.

## Interaction check
- Per-prediction match status per annotator record (IoU > 0.5; the masks are pixel-disjoint, so
  dropping one prediction does not change the others' matches).
- Venn counts of the predictions dropped by a, b and c, with the TP-records and FP-records removed
  in each region.
- Additivity: the sum of the single-rule dPQ compared with the bundle dPQ.

## Context arm (not gated)
- Baseline detector + V11 refiner (public V17 style): the `tta.cluster_views` consensus, V17 cell,
  the same 141 files.
- Run on CPU into `outputs/v72/cache_baseline.pkl`.
- Must reproduce 0.4517 (1004/464/589) from the V69 metric audit (asserted).
- The same bundle, leave-one-out and singles arms are applied, with the same bootstrap.

## Gate (primary arm only)
- **ADOPT** into the final export iff dPQ ≥ +0.004 AND P(d>0) ≥ 0.90. Otherwise do not adopt.
- This bar is deliberately lower than the experiment gate (V70/V71 CONTINUE: +0.010, CI > 0).
  The reason: these are free, deterministic post-processes with no retraining, no extra inference
  cost and fixed parameters. Their downside risk is bounded to removing a handful of predictions.
- Leave-one-out results are informational. If the gate passes, the whole pre-registered bundle is
  adopted; it is not pruned after the fact.
- Any parameter change after a number is seen is a new, labelled experiment, not V72.
