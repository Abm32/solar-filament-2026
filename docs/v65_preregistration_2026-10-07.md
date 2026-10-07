# V65 pre-registration: stack the near-misses + prepare an r3-style export (2026-10-07)

Written after V62-V64 results were seen; this is a post-hoc combined hypothesis and is labelled so.

## Context
On the 141 canonical held-out files at the V17 cell, against the current pipeline (baseline + V11,
0.4517): p2first +0.0049, r1 +0.0059, r3 (12 epochs, lr 1e-4) +0.0078 (P 0.92). Copy-paste on
fold 0 gave +0.0064 direct (P 0.88) but failed its structural gate. Refiner seed noise ~0.001.

## B. Stack test (`scripts/run_v65.sh`, GPUs 0-3)
p2first_cp folds 1-4 (fold 0 exists), OOF inference, then on the 141 files:
- `v11_p2first_cp`: frozen V11 refiner on p2first_cp proposals.
- `r3cp`: r3 recipe re-fit on p2first_cp OOF proposals of the 566 fit files.

PRIMARY: `r3cp - v11_baseline`. PASS iff delta >= +0.010 AND P(delta>0) >= 0.90.
Reported only: r3cp - r3 (does copy-paste add anything after the refiner), v11_p2first_cp vs
v11_baseline and v11_p2first, 5-fold direct p2first_cp vs p2first.

## A. Export (GPU 1, never submits)
r3 recipe refiner trained on all 707 p2first OOF proposal sets + the all-data p2first detector
(fold 99) -> 180-image test CSV at t0.45/c0.30 -> `validate_submission.py`. Its local evidence is
r3: +0.0078, below our bar. It is prepared so a submission decision can be made quickly; the
decision to submit is the user's.
