# V93 pre-registration: V91 spine-net instances as extra candidates inside the V84 stack (2026-10-19)

Written before any V93 number exists. Hash recorded in `outputs/v93/prereg.sha256`.

## Question
V91 (spine-first dense U-Net, `docs/v91_spine_first_result_2026-10-12.md`) matches V84 recall and recovers
138 of V84's 658 fold-0 FN pairs, but add-only fusion with a 5-feature logistic score gained +0.0002 because the
additions were mostly junk. Does routing V91 instances through the V84 soft crop refiner and a V84-style
LightGBM quality rule (same target $q\cdot s$, same Dinkelbach decision $\hat y > \lambda/2$) turn that
orthogonal recall into PQ?

## Fixed design (no tuning after the first run)
- Base: V84 bundle OOF kept set per fold, exactly the rows of `outputs/v87/fold{k}_v84.csv` (parity-checked
  export; pooled PQ must reproduce 0.45898 ± 5e-5 before anything is added).
- V91 models: `outputs/v91/f{k}/model.pt`, folds 0–4 (f0–f2 trained 2026-10-12, f3–f4 trained today with the
  same script/seed/steps). Predictions `/mnt/v91/pred_f{k}.u8` on the fold's held-out files only.
- Candidate generation: `v91_spine.decode(pf, ps, tf=0.4, ts=0.3)` (single grid point, the middle of the V91 grid;
  not selected on any result). Candidate j is dropped before refinement if (a) its area < 200 px, or (b) more than
  10 % of its pixels lie inside the union U of the V84 kept masks (the V91 fusion rule). Remaining candidates are
  clipped to ~U.
- Refinement: V84 soft refiner of the same fold (`outputs/v84/soft_fold{k}_refiner.pt`), t = 0.50, 7×7 seed
  clamp, 4-flip TTA, `v83_extract.refine_feats` with `occupied` initialised to U so V91 masks never take V84
  pixels; ownership among V91 candidates in order of their mean foreground probability. Post masks < 200 px drop.
- Features (24): the 21 refiner/shape/photometry features of V83 (`p_in … lowfreq_cv`, `area`) plus the five
  V91 decode features (mean fg prob, mean spine prob, max spine prob, log area, log skeleton length) and two
  context counts (V91 candidates in file, V84 kept masks in file). No detector features exist for these masks.
- Target and model: as V84 (`v83_rule.fit`, same LightGBM params, target min(1, Σ_rec IoU·[IoU>0.5]/R),
  weight R), cross-fitted: outer fold-k predictions from a model fit on folds ≠ k.
- Decision rule, primary arm A: keep V91 candidate iff $\hat y > \lambda_k/2$ with $\lambda_k$ = the V84
  per-fold lambda already fixed in `outputs/v84/export/parity/demote_fold{k}.json`.
- Secondary arm B: $\lambda$ re-solved by Dinkelbach on the combined (V84 kept + V91 eligible) set with inner
  cross-fitted predictions, as `v84_bundle.run_arm`. Reported, not used for the verdict.
- Scoring: host-exact pooled per-record PQ on all 707 files; combined columns = V84 kept ∪ V91 kept.
  Paired file bootstrap (2000 draws, seed 93) vs the V84 base, per-fold deltas.

## Gate (arm A vs V84 0.45898)
CONTINUE iff d ≥ +0.005 and P(d>0) ≥ 0.95 and ≥ 4/5 folds positive. Otherwise KILL; no re-tuning of the grid
point, overlap rule, features or λ. CONTINUE authorises: train a V91 all-data model (fold 99, same recipe),
run it on the 180 test images, add the arm-A rule to the V84 export, submit once, and record the public score.

## Compute
CPU extraction (refiner on CPU, Pool 22) over 707 files; V91 fold 3/4 training 31 min each on one T4.
Scripts: `scripts/v93_v91_candidates.py` (extract, score); `scripts/v91_spine.py` (patched today to build its
own `imgs.u8` instead of reading the wiped `/mnt/v89`). Outputs under `outputs/v93/`.
