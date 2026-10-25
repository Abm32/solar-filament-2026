# V88 result: pixel-ownership alternatives (2026-10-12)

Pre-registration: `docs/v88_ownership_preregistration_2026-10-12.md`, written and uploaded before any number existed.
The run was CPU only on rogii. No GPU, no submission, and no test or external labels were used.
Code:
- `scripts/v88_premodel.py`
- `scripts/v88_extract.py`
- `scripts/v88_score.py`

Outputs: `outputs/v88/{extract.pkl, result.json, extract.log, score.log}`.

## Controls
- **Arm G (greedy by detector score) reproduces V84 exactly.** On all 707 files, its IoU columns and its 28 features plus area match `outputs/v84/extract_soft.pkl[0.5]` with max |d| = 0.0. Its PQ is **0.45898**, which passes the assertion.
- **Demote was refit on every arm.** Each arm's features were recomputed on that arm's own masks. The LightGBM and λ were then refit cross-fitted by fold, using the unchanged `v84_bundle.arrays/crossfit/run_arm`. This is the preferred route; ownership was not applied after demote.
- **The alternative rules act only on groups with score ≥ 0.30.** Groups below 0.30 are still filled greedily on the leftover pixels.

## Arms (5-fold OOF, 707 files; paired file bootstrap, 2000 draws, seed 88)

| arm | PQ | TP / FP / FN | d vs G | 95% CI | P(d>0) | per-fold d (f0..f4) | pass |
|---|---|---|---|---|---|---|---|
| G (control, greedy by score) | 0.45898 | 5050 / 1881 / 3149 | – | – | – | – | – |
| A_argmax (per-pixel refiner-prob argmax) | 0.45686 | 5001 / 1841 / 3198 | −0.0021 | [−0.0044, +0.0001] | 0.031 | +.0015 −.0036 −.0046 −.0032 −.0007 | no |
| B_pin (order by mean refiner prob) | 0.45567 | 5011 / 1921 / 3188 | −0.0033 | [−0.0057, −0.0009] | 0.006 | +.0002 −.0086 −.0057 −.0009 −.0015 | no |
| **B_yhat** (order by pre-ownership ŷ, fold-clean model) | 0.45727 | 5035 / 1907 / 3164 | **−0.0017** | [−0.0037, +0.0004] | 0.062 | +.0008 −.0048 −.0056 +.0026 −.0015 | no |
| C_cf (trim the neighbour; x cross-fitted) | 0.45531 | 5000 / 1921 / 3199 | −0.0037 | [−0.0062, −0.0012] | 0.002 | +.0001 −.0075 −.0075 −.0022 −.0011 | no |
| D_small (smallest pre-area claims first) | 0.45564 | 5006 / 1917 / 3193 | −0.0033 | [−0.0059, −0.0008] | 0.005 | −.0021 −.0049 −.0069 −.0000 −.0027 | no |

For C, the cross-fitted x per fold was 0.2, 0.7, 0.5, 0.2, 0.2. The individual grid points are diagnostic only:

| x | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 |
|---|---|---|---|---|---|
| d | −0.0038 | −0.0028 | −0.0028 | −0.0029 | −0.0030 |

All of them are negative.

## Verdict: **KEEP GREEDY**
- The candidate, chosen as the largest d among the primary arms, is B_yhat at d = −0.0017. It fails all three bars: d ≥ +0.005, P(d>0) ≥ 0.95, and positive in ≥ 4/5 folds.
- No arm has a positive pooled delta.
- Three arms are significantly worse than greedy: B_pin, C_cf and D_small.

## Reading
- **The 610 "ownership FN" from V85 are not recoverable by re-assigning pixels.** Every alternative rule loses more TP than it gains: FN goes up by 15 to 50, and TP goes down by the same amount.
- **Higher detector score is the better arbiter of contested pixels.** Giving the pixels to the clipped mask breaks the neighbour, and the neighbour is often a TP for this or another record (V85: 41% of FP are another annotator's TP).
- **A_argmax trades TP for FP.** It cuts FP by 40, but loses 49 TP.
- **The V85 naive +0.036 upper bound was misleading for this lever.** It held the neighbour fixed, but changing ownership hurts the neighbour.
- The ownership class can be closed for rule-based fixes. Any remaining gain would need a learned per-conflict decision, for example inside the V86 pooled selector, and is unlikely to be large.
