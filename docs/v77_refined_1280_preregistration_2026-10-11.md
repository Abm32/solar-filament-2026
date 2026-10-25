# V77 pre-registration: p2first_1280 proposals through the r3 refiner (2026-10-11, before any result)

## Why
V68b: p2first_1280 beat p2first by +0.0142 direct over 5 folds (CI [+0.0090, +0.0192]). The refiner absorbed
~80% of the earlier p2first gain, so this checks whether the 1280 gain survives the refiner.

## Method
`scripts/v77_refined_1280.py` = the V75 harness (`v75_ensemble_refined.py`) with members=[p2first_1280]:
the same canonical 141-file population (outputs/v71/cache.pkl), the same r3 refiner weights, flip-vote grouping,
clean_refined_predictions, and cut. The proposals come from the 1280 OOF caches (each file scored by the model of its
own fold). The reference is current p2first+r3 (TP/FP/FN 1020/448/573). The refiner is NOT retrained on 1280 proposals. CPU only.

## Gate (same as V75)
CONTINUE iff d >= +0.010 and P(d>0) >= 0.90; KILL iff d < +0.005; otherwise inconclusive.
Secondary (reported only): the same comparison with the MIN200 area filter.
Whatever the outcome, no submission without the user's go-ahead.
