# V99 pre-registration, arms `hardneg` and `d4` (2026-10-25)

Written before any number for these arms. No test images, test labels, MAGFiLO release or leaked CSV are read.
No submission. Source: `docs/v99_pipeline_audit_2026-10-25.md` §3 rows 1 and 4. Companion arm `soft1280`:
`docs/v99_soft1280_preregistration_2026-10-25.md` (running).

## Arm `hardneg` (GPU 2, `scripts/v99_hardneg.py`)
The refiner has only ever been trained on proposals matched to a GT (IoU ≥ 0.30, V11 matcher); an empty refined
mask falls back to the seed, so every seed survives refinement and junk is left to the cut and the demote rule.
V85: 811 kept masks match no GT of any record (+0.037 exact if all were dropped); V83 oracle selection 0.528.
Arm: V84 soft recipe (stage A 8 ep 3e-4 GT crops; stage B 12 ep 1e-4; seed 42; ResNet18 CropUNet 256 px) with
stage B extended by negatives: fold-0 OOF p2first proposals with consensus score ≥ 0.20 whose max IoU to every GT
of every record of the file is < 0.10, target all zeros, subsampled to 25 % of the positive count (seed 99).
Loss, crops, jitter unchanged (negatives use unjittered bounds because the empty-target guard fires).
Judge: `v99_screen.py extract hardneg` on p2first views (cross-checked against the job's own scoring), gate vs
`v84_soft`, `s43 s44 s45`. Cells: `t050_c030` (as V89), `t050_c030_veto` (also drop candidates with
`fallback = 1`, i.e. refined mask empty → seed), `full_v84`, `full_v84_veto`.
Primary: `t050_c030_veto` vs `v84_soft t050_c030`. CONTINUE d ≥ +0.008 & P ≥ 0.90; INCONCLUSIVE +0.003 ≤ d < +0.008
(enters the stack bundle); KILL d < +0.003. Reported: `t050_c030` without veto (did negatives hurt positives?),
`v84_soft t050_c030_veto` (is the deployed refiner's fallback already a usable junk signal?), fallback rates.

## Arm `d4` (GPU 3, `scripts/v99_d4_views.py`)
Inference averages 4 flip views; the disk image is also invariant to 90° rotations, which no run has used at test
time (`p2first_rot` was train-time rotation). Arm: the existing p2first fold-0 detector run on the 8 dihedral
views (4 rotations × {identity, LR flip}; the 4 flip views are a subset), cached as recipe `p2first_d4`, then
the unchanged deployed V84 soft fold-0 refiner (copied without its json so the extract skips the cross-check),
`FLOOR = 0.10` for the extract because `cluster_views` normalises by 8 views.
Judge: gate vs `v84_soft`; cells `t050_c015 … t050_c030` (cut sweep) and `full_v84` (reported only; demote
features shift with 8 views). Primary: best of `t050_c020`/`t050_c025`/`t050_c030` chosen on fold 0 is an
optimistic read, so the rule is: CONTINUE only if `t050_c025` **and** `t050_c030` both have d ≥ +0.008 and
P ≥ 0.90; INCONCLUSIVE if either has d ≥ +0.003; else KILL. Secondary: kept-count-matched cut (the cut at which
`d4` keeps the same number of candidates as `v84_soft` at 0.30), reported with its d.

## Cost
hardneg: one refiner training (~25 min) + one CPU extract (~6 min). d4: 141 × 8 detector passes (~5 min GPU) +
one CPU extract. Disk < 1 GB (41 GB free after the dense-CRF pickle was removed).

---
Result `d4` (2026-10-25 06:45Z): **KILL.** 8 dihedral views reproduce the 4-flip proposal set almost exactly at every
cut (fold 0, 141 files; v84_soft → d4 at t0.50): c0.15 0.4366 → 0.4378 (+0.0012, P .65), c0.20 0.4480 → 0.4481,
c0.25 0.4545 → 0.4551 (+0.0005 [−.007, +.008], P .55), c0.30 0.4555 → 0.4556 (+0.0001), full V84 path 0.4590 → 0.4566
(−0.0024). TP/FP/FN differ by < 1 % at each cut: the rotated views detect the same masks at the same confidence, so
/8 consensus equals /4 consensus. Rotation TTA adds no information for this detector. `outputs/v99/screen_d4.json`.

Result `hardneg` (2026-10-25 07:00Z): **KILL.** Fold-0 fit set had only 647 junk proposals at score ≥ 0.20 with
max IoU < 0.10 to every record (10.6 % of the 6079 positives; all used). Cell `t050_c030`: 0.4555 → 0.4536
(−0.0020 [−.004, +.000], P .04); full V84 path 0.4590 → 0.4538 (demote features shifted, not refit). The veto never
fired: fallback rate is 0.000 for both the deployed and the hardneg refiner at every candidate, i.e. the refiner
always keeps some pixels inside the k7 clamp even on junk crops, so "empty output" is not a usable junk signal.
Junk separability of refiner-internal features moved only slightly (AUC at cut 0.30, fold 0: p_in 0.683 → 0.704,
margin 0.608 → 0.629, iou_seed_post 0.658 → 0.676); a file-grouped 5-fold LightGBM refit of the demote rule on
fold-0 candidates gives AUC 0.745 → 0.748 and best-λ PQ 0.4622 (control) vs 0.4614 (hardneg). Negative crops do
not teach this refiner to reject junk; the demote rule already extracts what those features carry.
`outputs/v99/screen_hardneg.json`, `outputs/v99/hardneg_fold0_refiner.pt` (sha 6f936905…).

All three V99 arms KILL (soft1280, d4, hardneg). The weakly-closed lines in the audit §2 rows 1–3 are now closed
on direct evidence; `p2first_flip` / `p2first_cp` follow by the soft1280 mechanism and were not run.
