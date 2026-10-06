# V62 `p2first` with the V11 refiner: the gain shrinks (2026-10-07)

Harness: `scripts/eval_v62_refiner.py` -> `outputs/v62_refiner_test.json`;
`scripts/analyze_v62_subset_direct.py` -> `outputs/v62_subset_direct.json`.
Population: the 141 canonical held-out files (94 tune + 47 audit). The V11 refiner was trained on the
other 566 canonical files, so on these 141 files BOTH the refiner and (because proposals come from
the OOF cache) the detector have never seen the labels. Proposals: V61 OOF cache -> flip_consensus
-> V11 refiner -> V17 scoring (`refined_constrained`, thresholds t x confidences c, same grid as V17).
Not computed on the other 566 files: the refiner saw their labels.

## Results (141 files)
| pipeline | cell | PQ | TP | FP | FN |
|---|---|---|---|---|---|
| baseline + V11 refiner | V17 cell t0.45/c0.30 | 0.4517 | 1004 | 464 | 589 |
| p2first + V11 refiner | V17 cell t0.45/c0.30 | 0.4566 | 1016 | 451 | 577 |
| baseline + V11 refiner | own best t0.55/c0.30 | 0.4519 | 1004 | 465 | 589 |
| p2first + V11 refiner | own best t0.55/c0.35 | 0.4582 | 976 | 365 | 617 |
- With the refiner: B - A = **+0.0049** at the V17 cell (95% CI [-0.0052, +0.0153], P(>0) 0.815) and
  +0.0062 at each recipe's own best cell (CI [-0.0051, +0.0169], P 0.857). Not distinguishable from 0.
- Without the refiner, on the SAME 141 files (direct detector, conf 0.35): baseline 0.3923,
  p2first 0.4162, **+0.0238** (CI [+0.0126, +0.0360]). The other 566 files: +0.0196; all 707: +0.0204.
  So these 141 files are typical for the direct-detector gain; the shrinkage is not a subset effect.

## What this says
1. The V11 refiner absorbs most of the direct-detector gain: +0.024 before, +0.005 after (about
   80% gone). The refiner is also the single biggest component of the pipeline: on these files it
   lifts the baseline from 0.3923 (direct) to 0.4517 (+0.059) and p2first from 0.4162 to 0.4566
   (+0.040).
2. CORRECTION to my own earlier statement: I said the refiner adds a stable ~+0.016 and so expected
   roughly 0.40 for a refined OOF baseline. That was inferred, and it was wrong for this
   comparison: the measured refiner lift is about +0.06 on the baseline. The direct-detector OOF
   numbers (0.3856 / 0.4060) are therefore far from the refined pipeline's level.
3. Direct p2first (0.4162 on these files) is well BELOW the refined baseline (0.4517). Submitting
   a direct-detector p2first would be expected to score below the current 0.39, not above.
4. Why the gain may be absorbed (hypotheses, not tested): the refiner re-predicts each mask from a
   256px crop and `refined_constrained` clamps to a dilated proposal, so it can overwrite the
   finer stride-2 masks p2first produces; and the V11 refiner was trained on proposals from a
   different (older, stride-4-prototype) detector.
5. The structural gain p2first provides (small-object recall +0.049 before refinement) may still
   be useful; what is unmeasured is whether a refiner trained on p2first's own proposals keeps it.

## Not established
- Whether a refiner trained on p2first proposals recovers the gain (untested; the proposals exist
  as OOF caches for all 707 files, which would allow a clean train/evaluate split).
- Statistical power: 141 files; the refined +0.005 has a CI of about +/-0.010. "No gain" and
  "+0.010 gain" are both consistent with these data.
- The own-best-cell numbers are selected on the same files and should not be read as evidence.

## Status of the plan
- All-707-file `p2first` is training (GPU 0, started ~00:20 IST, ~3.5 h) - harmless to continue, and
  needed for any later export.
- Export path validated (scripts/export_v62_shard.py, smoke shard passes the repo validator).
- NOT justified: submitting p2first now, direct or with the existing refiner (+0.005, CI spans 0).
- Candidate next step (needs your decision): train a refiner matched to p2first proposals
  (V11 proposal-centered continuation on OOF p2first proposals of the 566 refiner-training files),
  then re-run this exact 141-file test. If the refined gain does not reach the pre-registered
  bar there, p2first is a real architecture result that does not transfer to the submitted
  pipeline.
