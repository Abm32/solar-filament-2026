# V84 pre-registration: the "polish bundle" on the refined OOF (2026-10-11)

Written BEFORE any V84 number exists (no soft fold 1-4 refiner trained, no V84 extract run).
GPUs 0-2 only (never 3), CPU <= 24 processes. No submission, no test data, no test labels, no external GT.
Code: `scripts/_v84_train.sh` (soft folds 1-4), `scripts/v84_extract.py`, `scripts/v84_bundle.py`. Outputs: `outputs/v84/`.

## Components (each individually measured before, each small)
1. **Soft refiner** (V82 arm A): stage-B target = pixel mean over the image's records of each record's matched GT.
   Fold 0 = `outputs/v82/soft_fold0_refiner.pt` (+0.0019 at t0.45). Folds 1-4 are trained with
   `scripts/train_v82_refiner_targets.py fold --arm soft` with exactly the V73 base recipe/seed (imagenet/resnet18/256,
   p2first, seed 42, 8 A + 12 B epochs). A CPU base-target smoke (V82 script `--arm base` vs `train_v73_refiner.py`,
   2 files, 1+1 epochs) must reproduce V73 losses first, as in V82.
2. **Refiner threshold t = 0.50** instead of 0.45 (V82 sweep: every refiner about +0.001 at 0.50-0.55).
3. **MIN200**: drop final kept masks with area < 200 px (V69 audit / V72; +0.0016..+0.0037, removes FP only).
4. **Demote-only keep rule** (V83 arm "demote": LightGBM q*s regressor on the V83 features, Dinkelbach
   lambda/2 threshold, may only drop score >= 0.30 masks). It is REFIT, cross-fitted by V61 fold, on each
   configuration's own refined candidates. The model and lambda for fold k use only folds != k (lambda from
   inner cross-fit predictions, as in V83). Held-out labels of the fold being scored are never used. With MIN200
   on, the base set for lambda and the eligible set exclude masks below 200 px.

## Pipeline and reproduction
- The V83 extract path (`audit_v80_extract.py` real pipeline: p2first OOF 4-flip views, flip_consensus groups,
  fold refiner, k7 clamp, greedy ownership over all groups, cut 0.30). It is run with each refiner family
  (V73 base, V82 soft), and each run saves both t0.45 and t0.50 from the same probabilities.
- Assertion: V73 base at t0.45, cut 0.30, no MIN200 reproduces pooled PQ 0.45393 on 707 files
  (max per-file count diff <= 1e-3 vs `outputs/v73/base_fold*.json`). Soft-refiner per-file counts are also checked
  against each training job's own scoring (t0.45 and t0.50, c0.30).

## Primary (one comparison)
- **Baseline** = current deployed-equivalent refined OOF: V73 base refiner, t0.45, cut 0.30, no MIN200, no rule
  (= 0.45393, 707 files).
- **Bundle** = soft refiner + t0.50 + MIN200 + cross-fitted demote rule.
- Metric: pooled per-record PQ (record-level TP/FP/FN/IoU sums). d = PQ(bundle) - PQ(baseline). 95% CI and P(d>0)
  come from a paired file bootstrap (2000 resamples, seed 84). Per-fold d is reported over the 5 V61 folds.
- **Gate: CONTINUE (adopt the bundle for the final export) iff d >= +0.005 AND P(d>0) >= 0.90 AND >= 4/5 folds
  have d > 0.** Otherwise keep the current pipeline plus MIN200 only (already recommended by V72).
- Why 0.005: the components' separate fold-0 / 5-fold estimates sum to about +0.008 (soft +0.0019, t0.50 about
  +0.001, MIN200 about +0.002, demote +0.0030). V72 showed that small post-process gains overlap: its bundle
  (+0.0034) missed a +0.004 bar while its singles summed to +0.0066. A bundle that keeps under 5/8 of the summed
  gain is too fragile to justify a new refiner in the export.

## Secondary (reported only, no decision)
- Leave-one-out: the bundle minus each of soft, t0.50, MIN200 and demote, each vs the baseline (the rule is refit
  for every configuration that includes it).
- Each component alone vs the baseline.
- Same bootstrap and per-fold reporting.

## Disclosed simplifications
- Demoted or MIN200-dropped masks do not hand their pixels back to lower-ranked masks (as in V72 and V83).
- LightGBM comes from `outputs/v70/pylib`, as in V70 and V83. The feature set and hyperparameters are V83's,
  unchanged.
