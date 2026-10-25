# V82 result: agreement-aware refiner targets, fold-0 screen (2026-10-11)

Pre-registration: `docs/v82_refiner_targets_preregistration_2026-10-11.md`
(sha256 `88a4ad48...ffdbf`, `outputs/v82_prereg.sha256`, recorded before any training).
Code: `scripts/train_v82_refiner_targets.py`, `scripts/v82_refined_fold0.py`, `scripts/_v82_launch.sh`,
`scripts/v82_cohort_stats.py`. Outputs: `outputs/v82/{soft,union,qweight}_fold0{.json,.log,_refiner.pt}`,
`outputs/v82_rerun/base_fold0*` (noise control), `outputs/v82/refined_fold0.json` (gate), `outputs/v82/cohort_stats.json`.

## Sanity
- CPU smoke (2 files, 1+1 epochs): the V82 script with `--arm base` matches `train_v73_refiner.py`
  (loss to 1e-7, same records_a/records_b and the same args: imagenet/resnet18/256/p2first/seed 42).
  Full runs used the same files (566 fit / 141 scored), the same epochs (8 A + 12 B) and 6,079 stage-B samples.
- The gate harness reproduces V73 base fold 0 = 0.453259 exactly (max per-file diff 0). The arm
  per-file counts match each training job's own GPU scoring (max diff ≤ 4e-4).
- Cohort: 6,079 samples, mean q 0.90. 21% have q<1 (371 at 1/3, 376 at 1/2, 542 at 2/3), and 37% come from
  single-record images, where every arm equals base.

## Primary (141 fold-0 files, k7, t0.45, c0.30, arm minus stored V73 base, 2,000 bootstraps, seed 82)
| arm | PQ | TP/FP/FN | d | 95% CI | P(d>0) | verdict |
|---|---|---|---|---|---|---|
| base (V73) | 0.4533 | 1028/431/612 | – | – | – | – |
| A soft | 0.4552 | 1031/425/609 | +0.0019 | [-0.0004, +0.0043] | 0.95 | **KILL** |
| B union | 0.4338 | 1004/451/636 | -0.0195 | [-0.0262, -0.0125] | 0.00 | **KILL** |
| C qweight | 0.4532 | 1027/429/613 | -0.0000 | [-0.0023, +0.0024] | 0.48 | **KILL** |
| base_rerun (control) | 0.4531 | 1027/429/613 | -0.0001 | [-0.0022, +0.0022] | 0.43 | (noise only) |

Retraining noise is about 0.0001 PQ, so the small soft gain (+0.002, P 0.95) is probably real but is
5x below the kill line.

## Secondary: threshold sweep (PQ, exploratory only)
| arm | 0.35 | 0.40 | 0.45 | 0.50 | 0.55 | best t: d vs base@0.45 / vs base@same t |
|---|---|---|---|---|---|---|
| base | 0.4501 | 0.4519 | 0.4533 | 0.4542 | 0.4545 | – |
| soft | 0.4529 | 0.4539 | 0.4552 | **0.4555** | 0.4543 | 0.50: +0.0023 (P 0.91) / +0.0014 (P 0.85) |
| union | 0.4235 | 0.4291 | 0.4338 | 0.4376 | **0.4413** | 0.55: -0.0120 / -0.0132 |
| qweight | 0.4505 | 0.4516 | 0.4532 | 0.4544 | **0.4547** | 0.55: +0.0015 (P 0.83) / +0.0002 |
| base_rerun | 0.4502 | 0.4513 | 0.4531 | 0.4530 | 0.4540 | 0.55: +0.0007 / -0.0005 |

## Reading
- Union targets make masks larger, which lowers SQ (0.669 vs 0.683) and turns about 24 TPs into FN and FP.
  This matches the audit finding that masks are already about 8% too large. The union-oracle gain
  comes from instance selection, not from mask extent.
- Soft and q-weighting barely move the refiner. Agreement only differs from the per-record target on about 21%
  of crops, and the matched GTs largely overlap. The refiner's target definition is not the bottleneck
  at fold-0 resolution.
- Every arm (base included) prefers t of 0.50 to 0.55, but that is worth only about 0.001 PQ. Nothing reaches the gate.
