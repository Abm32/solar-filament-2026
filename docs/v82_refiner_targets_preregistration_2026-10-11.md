# V82 pre-registration: agreement-aware training targets for the crop refiner (2026-10-11)

Written before any V82 training run. Fold-0 screen only. Nothing is submitted; test images and
labels are never read; no external GT (SWEFil, MAGFiLO Dataverse test GT). Code:
`scripts/train_v82_refiner_targets.py` (hooks into `train_v73_refiner.cmd_fold`, unchanged),
gate harness `scripts/v82_refined_fold0.py` (the V81 harness with the refiner swapped in).

## Why
- The refiner is the binding component. Detector upgrades gave +0.01..+0.027 direct but ~0 after
  the refiner (outputs/v81/refined_fold0.json, outputs/v76/refined_fold0.json, docs/v77_*).
  Capacity, crop size and seed dilation are exhausted (docs/v73_refiner_result_2026-10-11.md).
- The metric pools PQ over annotator records (1-4 per image), each scored against the same
  predictions. Human-vs-human PQ is 0.34; refined masks are ~8% larger than GT (area ratio 1.08);
  the union-of-annotators oracle (0.663) beats single-annotator (0.605) and majority (0.592)
  (outputs/v79_audit/labels_report.json). V73 trains each proposal crop against ONE record's GT,
  so a proposal matched by 3 records is pulled toward 3 different outlines with equal weight and
  a proposal matched by 1 of 4 records is trained as fully positive. Agreement-aware targets
  should teach the shape that maximises expected IoU across records.

## Fixed (identical to V73 `base` fold 0)
Fit = the 566 non-fold-0 files, p2first OOF proposals, `cluster_views(flip_consensus, 0.5)`;
stage A 8 epochs lr 3e-4 GT-centred crops (unchanged in all arms); stage B 12 epochs lr 1e-4,
AdamW wd 1e-4, cosine, batch 16, fp32, 256 crops, ImageNet ResNet18, seed 42; same sample list
(V11 matcher `match_positive_proposals`, IoU>=0.30 + margin rules, one positive per GT per
record; 6,079 stage-B samples), same jitter/augmentation RNG. Inference: 4-flip TTA, 7x7 clamp,
t0.45, cut 0.30. Only the stage-B target (and for C, the per-crop loss weight) changes.

"Record r matches proposal p" = the V11 matcher assigns one of r's GTs to p (the matcher used
to build the V73 cohort; deviation from a plain IoU>0.5 rule so the sample set stays identical).
R = number of records of the image (records with no annotations count, contributing 0).
q_p = (#records matching p) / R.

## Arms (one GPU each, fold 0)
| arm | stage-B target per sample (r, p, g) | loss |
|---|---|---|
| A `soft` | pixel mean over the R records of the GT each matched to p (0 if none) | 0.5 BCE + 0.5 soft Dice, soft targets |
| B `union` | pixel max over the matched GTs of all records | as V73 |
| C `qweight` | g (as V73) | per-crop (0.5 BCE + 0.5 Dice) weighted by q_p, normalised by sum of weights per batch |

Reported control (not gated): `base_rerun` = V73 base retrained by the V82 script (w=1, binary
target), to size retraining noise vs the stored V73 base refiner.

## Gate (per arm, mechanical, applied once)
`scripts/v82_refined_fold0.py` -> `outputs/v82/refined_fold0.json`. 141 fold-0 files,
refined PQ (per-record pooled) at t0.45/c0.30 with the arm's refiner vs the stored V73 base
refiner (`outputs/v73/base_fold0_refiner.pt`), same proposals; the harness must reproduce V73
base fold 0 = 0.453259 exactly, else REPRO_FAIL. Paired file bootstrap (2,000 resamples, seed 82):
- **CONTINUE** iff d >= +0.010 and P(d>0) >= 0.90;
- **KILL** iff d < +0.005;
- otherwise **INCONCLUSIVE**.
Secondary (exploratory only, never gates): refiner threshold sweep t in {0.35, 0.40, 0.45, 0.50,
0.55} per arm (soft targets shift calibration); best-t PQ and its delta vs base at t0.45 and vs
base at the same t.
