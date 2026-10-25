# V99 pre-registration, arm `soft1280`: refiner co-trained on the 1280 detector's proposals (2026-10-25)

Written before any V99 number. No test images, test labels, MAGFiLO release or leaked CSV are read. No submission.
Source: `docs/v99_pipeline_audit_2026-10-25.md` §2 row 1. V68b: p2first_1280 beat p2first by +0.0142 direct over
5 folds (CI [+0.009, +0.019]); V77 read −0.0029 refined through the r3 refiner trained on p2first-1024 proposals
and wrote "the refiner was not retrained on 1280 proposals; that variant is untested". This arm tests that variant.

## Arm
`soft1280`: the V84 soft recipe exactly (`train_v82_refiner_targets.py fold --arm soft --fold 0`, stage A 8 ep 3e-4
on GT crops, stage B 12 ep 1e-4, seed 42, ImageNet ResNet18 CropUNet, 256 px crops) with `SOLAR_RECIPE=p2first_1280`,
so stage B's proposals, the in-job scoring and the extract all use the p2first_1280 fold-0 OOF views
(`outputs/v61_oof_cache/p2first_1280/fold0`, 141 files). One code change: `train_v73_refiner.RECIPE` reads the env var.

## Judge
`scripts/v99_screen.py` = `v89_screen.cmd_extract/cmd_gate` with outputs under `outputs/v99`. Arms in the gate:
- `v84_soft` (control, built in): deployed V84 soft fold-0 refiner on p2first views (`outputs/v84/extract_soft.pkl`).
- `ctrl1280`: the same deployed refiner on 1280 views (the V77 configuration with the soft refiner): separates
  "better proposals" from "co-training".
- `soft1280`: the arm.
- `s43 s44 s45`: V89 fresh-seed V84 refiners on p2first views (symlinked from `outputs/v89`), to price seed noise.
Cells: `t050_c030` (t0.50, cut 0.30; primary, demote-free because the demote model was fit on p2first features),
`full_v84` (plus V84 demote fit on folds 1–4; reported). Paired per-file bootstrap, 2000 draws, seed 89 (V89 code).

## Pre-stated reading (primary = `t050_c030`, soft1280 vs v84_soft)
- CONTINUE: d ≥ +0.008 and P(d>0) ≥ 0.90. Follow-up: 5-fold soft1280 (folds 1–4, caches exist), then the all-data
  1280 detector (fold 99, ~7 h on a T4; the V74 run was stopped at epoch 10) and a demote refit on 1280 features.
- INCONCLUSIVE: +0.003 ≤ d < +0.008, or P < 0.90 with d ≥ +0.008: the arm enters the V99 stack bundle (5-fold,
  gated once at P ≥ 0.95 on 707 files) rather than being closed.
- KILL: d < +0.003. Also report d vs the fresh-seed mean (s43–s45) and `ctrl1280` vs `v84_soft`: if soft1280 ≈
  ctrl1280 the co-training adds nothing; if soft1280 > ctrl1280 ≥ v84_soft the "absorbed" reading of V77 was an
  artefact of the frozen refiner and the same test is owed to `p2first_flip` and `p2first_cp`.

## Cost
One refiner training (~22 min, GPU 1), two CPU extracts (~6 min each, 22 processes), one gate. Disk: < 0.5 GB.

---
Result (2026-10-25 06:49Z): **KILL.** Fold 0, 141 files, cell `t050_c030` (t0.50, cut 0.30, no demote):
v84_soft 0.4555; ctrl1280 (deployed refiner, 1280 views) 0.4529 (−0.0026 [−.011, +.006], P .27);
soft1280 (refiner co-trained on 1280 proposals, 1280 views) 0.4531 (−0.0025 [−.011, +.006], P .26).
Full V84 path: 0.4590 / 0.4573 / 0.4580. soft1280 = ctrl1280 to 0.0002: co-training the refiner on the new
detector's proposals changes nothing, so V77's "absorbed" reading was not an artefact of the frozen refiner.
The 1280 detector's +0.014 direct gain is width/SQ that the refiner redraws either way; the same holds by
inference for `p2first_flip` and `p2first_cp` (not run). Fresh-seed refiners at this cell: s43 0.4535, s44 0.4556,
s45 0.4553 (mean 0.4548, seed 42 = 0.4555): at the demote-free cell seed 42 is not lucky; the +0.0025 seed effect
seen in V89 lives in the demote stage (λ and features fit on seed-42 candidates). `outputs/v99/screen_soft1280.json`.
