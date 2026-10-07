# V63 pre-registration: refiner re-fit on OOF proposals (2026-10-07, before any result)

## Why
- V62 hybrid (`v62_size_hybrid_2026-10-07.md`): the V11 refiner is the biggest measured lever
  (+0.06 PQ) and equalises p2first's mask-quality edge. The refiner was trained on V6-detector
  proposals, never on the proposal distribution it now refines.
- Data search: the 71 external MAGFiLO v1.0 images outside Kaggle train/test carry **zero**
  annotations (all 10,244 labels sit on the 887 Kaggle images). The GONG archive
  (gong2.nso.edu, nispdata.nso.edu) times out from the Azure VM, so archive pseudo-labelling is
  blocked. No new labelled data exists; the detector-with-more-data track is dropped.

## Arms (`scripts/train_v63_refiner.py`, runner `scripts/run_v63.sh`, GPUs 0-3 on rogii)
Training: the 566 canonical FIT files only, their V61/V62 OOF proposals, V11 matching and recipe
(canonical refiner init, AdamW, cosine, 0.5 BCE + 0.5 Dice, batch 16, 256 proposal-centred crops).
Scoring: the 141 canonical held-out files, V17 cell t0.45/c0.30.

| tag | proposals | refiner |
|---|---|---|
| v11_baseline | baseline OOF | frozen V11 (control = current V17-equivalent, 0.4517) |
| v11_p2first | p2first OOF | frozen V11 (0.4566) |
| r2 | baseline OOF | re-fit, 4 epochs, lr 5e-5, seed 42 |
| r1 | p2first OOF | re-fit, 4 epochs, lr 5e-5, seed 42 |
| r1s2 | p2first OOF | as r1, seed 20261007 (refiner seed noise) |
| r3 | p2first OOF | re-fit, 12 epochs, lr 1e-4 |

## Gate (`scripts/compare_v63.py`)
PRIMARY: r1 - v11_baseline. PASS iff delta >= +0.010 AND file-bootstrap P(delta>0) >= 0.90.
Reported, never gating: r1-r2, r2-v11_baseline, r1-r1s2, r3-r1, r3-v11_baseline. r3 cannot
replace r1 as primary after the fact. A PASS authorises an all-707 refiner re-fit on the fold-99
p2first model and a test export; it does not by itself authorise a submission.
