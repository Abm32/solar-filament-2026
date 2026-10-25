# V78 pre-registration: label transfer to nearby GONG frames, fold-0 screen (2026-10-11, before any result)

## Why
Host ruling (thread 748045, 2026-10-10 22:40Z): copying the competition's own training labels onto nearby
archive frames of the same site is allowed for training. GONG frames are co-registered with the Kaggle images
(V69: shift < 0.6 px), but appearance changes within minutes (V69: -0.044 PQ at +2 min at inference). Training on
neighbour frames that carry the original labels is real-data augmentation (seeing, transparency, small
evolution), with label noise that grows with the time offset.

## Data
`scripts/fetch_v78_lt.py`, fetched on the local machine (archive blocked from the VM), rsynced to /mnt/v78_frames.
For each of the 566 fold-0 TRAIN files: the nearest same-site frame within +-30 s of t0 +-1, +-2, +-5 min, never
the labelled frame itself. **Frames within +-1 h of any test-image timestamp are dropped** (host condition). The test
names are used only for their timestamps. No held-out (fold-0) file, test image, or test label is used.

## Arms (p2first graph, V21/V17 recipe, fold 0; one T4 each)
| arm | frames | per labelled record |
|---|---|---|
| p2first_lt1 | +-1 min | one frame (random side if both exist, seed 78), labels copied unchanged |
| p2first_lt2 | +-2 min | same |
| p2first_lt5 | +-5 min | same |
Records with no frame in the arm's window get none. Training data are roughly 2x (like V76), with the same epoch count,
so roughly 2x the training time. Labels are copied without a rotation correction (+-5 min of solar rotation is < 1 px at disk centre).

## Measurement and gate (same as V68)
`eval_v61_oof.py infer` (V17 cell, 4-flip consensus), then `eval_v61_compare.py --fold 0 --a p2first --b <arm> --noise p2first_seed2`.
CONTINUE iff d >= +0.010, P(d>0) >= 0.90, and d > |seed noise|. KILL iff d < +0.005. Otherwise inconclusive.
Three arms are screened (a dose series), so a single pass is a weaker claim. Known risk (V77): the r3 refiner absorbed
all of 1280's direct +0.014, so a continuing arm must also pass a refined-pipeline test at +0.010 before
any 5-fold run or export. No submission without the user's go-ahead. Disclosed in the technical report.

## Launch note (2026-10-11 03:58 UTC, before any result)
The first launch at 03:55 failed immediately (PermissionError creating /mnt work dirs). The dirs were created with sudo and all three arms relaunched at 03:57. Frames: 2166 fetched for 566 files. Transferred records: lt1 887, lt2 833, lt5 806 of 935.
