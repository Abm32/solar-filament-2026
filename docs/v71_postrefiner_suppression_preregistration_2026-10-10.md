# V71 pre-registration: post-refiner suppression of fragments, merges and hallucinations (2026-10-10)

Written BEFORE any V71 number. It follows V69 lever #2: in the real pipeline, 299 of 448 FP match no annotator, and fragments are 20% of FP. V70 (the re-scorer) was a KILL. CPU only, no model training, no test data, no submission. Code: `scripts/v71_suppress.py`. Outputs: `outputs/v71/`.

## Pipeline under test
- p2first OOF proposals go through the r3 refiner (`outputs/v63/r3_refiner.pt`, run on CPU), then the V17 cell: t = 0.45, `refined_constrained`, cut c = 0.30.
- Population: the 141 canonical held-out files.
- Run-time check: the unmodified output must reproduce `outputs/v63/r3.json` per file at t0.45/c0.30 (PQ 0.4595).
- The rules act on the kept predictions (score ≥ 0.30). These are already pixel-disjoint after the refiner's ownership step.
- Per-prediction quantities:
  - area, score;
  - principal-axis angle and elongation, from mask second moments: elong = λ1/λ2 of the covariance;
  - contrast = (ring − fg)/ring, where ring = a 21×21 dilation minus the mask, on the grayscale image.

## Rules (fixed family, small fixed grids)
All rules are non-iterative and are evaluated against the original kept set.

**R1 fragment.**
- A prediction q is a fragment if some other kept prediction P meets all of:
  - higher score than q;
  - larger area than q;
  - area(q)/area(P) ≤ ρ;
  - adjacent: q dilated by a 5 px radius (11×11 ellipse) touches P;
  - along the same spine: principal axes differ by < 30°, or q has elong < 2 (axis undefined).
- If several P qualify, the highest-scoring one is used.
- Action: drop q, or merge q into P (P's mask becomes P ∪ q and P keeps its score).
- Grid: action {drop, merge} × ρ {0.25, 0.5} (4 points).

**R2 small/low-score.**
- Drop q if area < A and score < S.
- Grid (A, S): (150, 0.5), (300, 0.5), (150, 1.01), (300, 1.01) (4 points). S = 1.01 makes it an unconditional minimum-area rule.

**R3 hallucination/contrast.**
- Drop q if contrast < τ and score < 0.5.
- Grid τ: {0.02, 0.04, 0.06} (3 points).

**Merge-split rule: not implemented.** Splitting a blob into two elongated parts needs junction or skeleton analysis, which is not cheap. Merges are only 27 of the 448 refined FP (V69).

Each rule's grid also includes "off". When rules are combined they are applied in the order R1 → R2 → R3.

## Selection (cross-fitted, never on the scored files)
- The 141 files carry their V61 fold labels (`outputs/v61_folds.json`; they are spread over all 5 folds).
- For each fold k, and for each rule separately, pick the grid point or "off" that maximises pooled PQ on the 141 files NOT in fold k, with that rule applied alone.
- Fold k's files are then scored with the combination of the chosen points.
- Ties go to "off", then to the earlier grid point in the order listed above.

## Metrics and gate
- **PRIMARY:** combined cross-fitted rules vs r3 at the V17 cell, on the same 141 files. Paired file bootstrap (2000, seed 71): dPQ, 95% CI, P(d>0).
- **Reported:** each rule alone (cross-fitted the same way), every grid point pooled (descriptive only), TP/FP/FN/SQ, chosen points per fold.
- **Gate:**
  - CONTINUE iff dPQ ≥ +0.010 AND P(d>0) ≥ 0.90 AND CI lower bound > 0.
  - KILL iff dPQ < +0.005.
  - Otherwise INCONCLUSIVE.
- Any rule or grid change after a number is seen is a new, labelled experiment, not V71.
