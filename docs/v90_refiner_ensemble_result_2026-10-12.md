# V90 refiner-seed ensemble: pre-registration (2026-10-12, written before any V90 number)

Motivation: V89 showed fold-0 refined PQ moves +-0.003 with the refiner seed alone (seeds 42-45, V84 soft recipe),
and seed 42 (deployed V84) is the best of four. All earlier ensembles were over detectors; refiner members were
never averaged. Averaging independent refiners should cut that variance.

Method: per crop, mean of member sigmoid probabilities (4-flip TTA per member as today), then the unchanged V84
path (t0.50, cut 0.30, MIN200, V84 fold-0 demote model and lambda). Fold 0, 141 files, `scripts/v90_refiner_ens.py`
+ `scripts/v89_screen.py gate`. No refit of anything; no tuning.

Arms (fixed now):
- `ens4`: V84 soft seeds 42, 43, 44, 45 (identical recipe).
- `ens6`: ens4 + V89 `mim` + `moco` fine-tunes (same recipe, SSL inits).

Primary: `ens4` vs V84 seed 42 (0.45899), full V84 path, paired bootstrap by file.
Gate (same as V89): d >= +0.008 and P(d>0) >= 0.90 -> 5-fold (train seeds 43-45 for folds 1-4, ~15 min each,
GPU), 5-fold gate d >= +0.010 and P >= 0.90 vs 0.45898. Secondary only: `ens6`; each arm vs mean of fresh seeds.
Caveat stated up front: the demote model was fitted on single-refiner candidate features; ensemble probabilities
may shift those features. Not refit in the screen.

## Result (2026-10-12 15:15 UTC)

| arm | PQ (full V84 path) | TP/FP/FN | d vs V84 seed 42 | 95% CI | P(d>0) | refined-only d |
|---|---|---|---|---|---|---|
| V84 soft seed 42 | 0.45899 | 982/312/658 | - | - | - | - |
| ens4 (seeds 42-45) | 0.45929 | 978/300/662 | +0.0003 | [-0.0038, +0.0044] | 0.55 | +0.0011 |
| ens6 (+mim, moco) | 0.45788 | 975/308/665 | -0.0011 | [-0.0050, +0.0028] | 0.28 | +0.0007 |

**Verdict: FAIL (gate +0.008) -> KILL.** Secondary: ens4 is +0.0028 vs the mean of fresh single seeds (0.45645),
i.e. averaging removes the seed lottery but the expected single-seed refiner plus variance reduction only matches the
lucky deployed seed. No 5-fold run.
