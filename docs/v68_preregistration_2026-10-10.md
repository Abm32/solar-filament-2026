# V68 pre-registration: scaling the one axis that worked (p2first), fold-0 screen (2026-10-10, before any result)

## Why
Every post-hoc lever V58-V67 closed. The only detector-level gain measured so far is V62 p2first
(stride-2 mask prototypes + stride-4 detection): direct PQ +0.020 on fold 0, +0.0238 pooled 5-fold.
V68 asks whether pushing further along that axis (capacity, input resolution, schedule, train-time
flips) gives another detector-level step. Four arms run in parallel, one per T4.

## Arms (all = p2first graph, V21/V17 recipe, fold 0, 566 train / 141 held-out files)
| arm | change vs p2first | note |
|---|---|---|
| p2first_m | YOLO11m scale (COCO yolo11m-seg init, same head remap) | batch 2 (T4 memory); nbs=64 keeps effective batch |
| p2first_1280 | imgsz 1280 train + inference (prototypes 640x640) | batch 2 (T4 memory) |
| p2first_long | 2x schedule: 160 ep mask_ratio 2 + 40 ep mask_ratio 1 | same batch |
| p2first_flip | fliplr 0.5, flipud 0.5 in both stages | inference unchanged (4-flip consensus) |
Batch 2 for m/1280 is a disclosed confound (forced by 16 GB).

## Measurement
Direct detector, `eval_v61_oof.py infer` (V17 cell, 4-flip consensus, pixel ownership; 1280 arm uses
`--imgsz 1280`), then `eval_v61_compare.py --fold 0 --a p2first --b <arm> --noise p2first_seed2`.
Reference: p2first fold 0 PQ 0.3967 @ conf 0.35. Noise: |p2first_seed2 - p2first| on fold 0.

## Gate (mechanical, per arm, primary cut 0.35)
- CONTINUE (5-fold OOF for that arm) iff d(arm - p2first) >= +0.010, P(d>0) >= 0.90, and
  d > |seed noise|. Four arms are screened, so a single +0.010 is a weaker claim than in V62;
  a continuing arm must reproduce >= +0.010 pooled over 5 folds before anything else.
- KILL iff d < +0.005. Otherwise inconclusive (reported, no further compute on that arm alone).
- If two or more arms pass: one combined arm (passing changes together) gets its own fold-0 run
  before any 5-fold; no other combinations.

## Known risk, stated before results
The refiner absorbed ~80% of p2first's direct gain (+0.0238 direct -> +0.0049 refined). A direct
+0.010 may be worth ~+0.002 deployed. A continuing arm therefore also needs a refined-pipeline test
(r3 recipe on its OOF proposals) at +0.010 vs current before any submission. Nothing here authorises
a submission.

## Addendum (2026-10-10 ~16:30 IST, after the p2first_flip result, before any rot number)
p2first_flip fold 0: +0.0068 vs p2first (CI [-0.004, +0.018], P 0.89) = inconclusive. Note: the
compare script evaluates at the reference's best cut (0.30), not 0.35 as written above; the same cut
rule is used for every V68 arm.
Fifth arm on the freed GPU 3, **p2first_rot**: degrees=180 (full-angle rotation) + fliplr/flipud 0.5,
both stages, everything else = p2first. Same gate. Note that this was chosen after seeing flip's
direction (positive, not significant), so it is a follow-up, not an independent arm.

## Fold-0 results (2026-10-10 ~18:25 IST)
| arm | PQ | d vs p2first | 95% CI | P(d>0) | verdict |
|---|---|---|---|---|---|
| p2first_1280 | 0.4134 | +0.0143 | [+0.0041, +0.0250] | 0.998 | **continue** |
| p2first_flip | 0.4058 | +0.0068 | [-0.0043, +0.0176] | 0.891 | inconclusive |
| p2first_m | 0.4027 | +0.0036 | [-0.0084, +0.0163] | 0.721 | kill |
p2first_long and p2first_rot are still running. Per the gate, p2first_1280 goes to 5-fold OOF (folds
1-4, `scripts/_v68b_1280_fold.sh`), with priority over V73 (V73 paused and re-queued after it).
It still needs the 5-fold pooled >= +0.010 and a refined-pipeline test before any submission.
| p2first_long | 0.3936 | -0.0055 | [-0.0148, +0.0040] | 0.128 | kill |
After 1280 folds 1-4 finish: the p2first_1280 fold-99 all-data model trains on GPU 0 (for the V74 export; it's used only if 5-fold and the refined test pass), and V73 runs on GPUs 1-3 (`scripts/_v68b_after.sh`).
| p2first_rot | 0.3960 | -0.0030 | [-0.0160, +0.0099] | 0.339 | kill |

## Incident: 1280 fold 2 stage 2 OOM (2026-10-10 ~15:50 UTC)
Fold 2 stage 2 hit CUDA OOM at batch 2; Ultralytics silently fell back to batch 1 (a recipe change).
The run was stopped before epoch 1 finished and stage 2 restarted at batch 2 from the finished
stage-1 checkpoint. `p2first_1280` now uses the chunked, activation-checkpointed upstream mask loss
(`v61_seg_loss.install_chunked_upstream`, identical value and gradient, tested in V65) for any new run:
the fold-2 restart and fold 99. Folds 0, 1, 3 and 4 run in already-started processes with the plain loss.
The aborted attempt is kept in `mask_ratio1/run_oom_batch1_aborted` and `train_p2first_1280_fold2.oom_attempt.log`.
Any other fold that logs the batch-1 fallback gets the same restart.
Fold 3 also OOMed at the start of stage 2 (batch-1 fallback, 16:2x UTC); it was restarted the same way at batch 2 with the chunked loss. `scripts/_v68b_oom_guard.sh` now does this automatically for folds 1 and 4 if they OOM.
Fold 1 crashed with a hard CUDA OOM at stage-2 epoch ~10 (17:1x UTC); the guard restarted stage 2 with the chunked loss. Fold 4 was restarted proactively at stage-2 epoch 1 (15.5 GB peak on the plain loss). So folds 1-4 all run stage 2 with the chunked loss (identical value and gradient); fold 0 used the plain loss.
