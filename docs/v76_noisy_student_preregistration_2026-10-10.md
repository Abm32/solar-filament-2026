# V76 pre-registration: noisy-student on unlabelled GONG archive frames, fold-0 screen (2026-10-10, before any V76 number)

## Why
`v69_history_audit` 2a: semi-supervised data scale is the root-cause doc's #1 lever and was never
run. The host allows unlabelled GONG H-alpha frames for training only (thread 744047); extra
ground truth is banned (746318), so no human labels are added. The archive is blocked from rogii
but reachable locally (memory: gong-archive-blocked-from-vm).

## Data (fixed now)
- `fetch_v76_unlabelled.py` (local, `python -I`): days sampled per year in proportion to the 707
  train files (2011-2022, seed 76), **excluding every train and test date +-1 day**; one frame per
  day, random site among the six train sites (B C L M T U), from the middle half of that site's
  frames that day. About 1,350 days requested. Rsynced to `/mnt/v76/frames`.
- No test image, test label or test date is used. The teacher never sees the frames' labels
  (there are none).

## Teacher and pseudo-labels (`scripts/v76_pseudo.py`)
- Teacher = p2first **fold-0** detector (`/mnt/v61/p2first_fold0`), which never saw fold-0 labels,
  so the fold-0 screen stays leak-free. No refiner (r3 is not fold-aligned).
- 4-flip views exactly as `eval_v61_oof` (V17 cell); flip-consensus groups, score = sum/4.
- Keep score >= **0.35** (the teacher's operating point), erode 1 px (3x3 cross; direct masks are
  ~1 px too wide), score-ordered pixel ownership, drop < 200 px. Frames with no kept mask are
  skipped. YOLO polygon labels.
- (A 0.50 cut was smoke-tested on 4 frames before this was written: it kept only 2-3 masks per
  frame, so 0.35 was chosen to avoid teaching that real filaments are background. No PQ was seen.)

## Student (`train_v61_fold.py --recipe p2first_ns`)
- Exactly the p2first recipe and init (stage_args identical, asserted), fold 0, trained on the
  566 fold-0 train files (935 annotator records) **plus 935 pseudo-labelled frames** (random
  sample, seed 76, if more were kept). Ultralytics augmentation is the "noise". One round only.

## Measurement and gate (same as V68)
`eval_v61_oof.py infer --fold 0 --recipe p2first_ns`, then
`eval_v61_compare.py --fold 0 --a p2first --b p2first_ns --noise p2first_seed2`.
- CONTINUE iff d >= +0.010, P(d>0) >= 0.90 and d > |seed noise| -> 5-fold OOF (fold-K teachers),
  then the refined-pipeline test (r3 refit on student OOF proposals) at +0.010 before any export.
- KILL iff d < +0.005. Otherwise INCONCLUSIVE.

## Known limits, stated now
- Archive frames are on average lower quality than the selected labelled frames (V69: the same
  detector loses 0.04-0.2 PQ on neighbour frames). Pseudo-labels on them may be noisier than the
  teacher's held-out PQ suggests.
- One fold, one seed; the refiner has absorbed ~80% of every direct detector gain so far, so a
  direct +0.010 may be worth ~+0.002 deployed.
- GPU time: ~7 h on one T4, queued behind V68b / fold-99 / V73; it only takes an idle GPU.
