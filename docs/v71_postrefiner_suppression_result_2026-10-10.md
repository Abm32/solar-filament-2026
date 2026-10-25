# V71 result: post-refiner suppression is a KILL (2026-10-10)

**Verdict: KILL.** The combined cross-fitted rules gain only +0.0028 PQ, below the +0.005 kill line. The run cleared none of the CONTINUE conditions, though the effect is real but small.

- **Pre-registration:** `docs/v71_postrefiner_suppression_preregistration_2026-10-10.md` (sha256 26a5aede…), written 07:45 UTC, before any V71 number.
- **Code:** `scripts/v71_suppress.py`. Outputs: `outputs/v71/{cache.pkl,result.json,eval.log}`.
- **Run:** CPU only, nothing submitted, no test data read.
- **Reproduction check:** the r3 refiner ran on CPU and reproduced `outputs/v63/r3.json` per file at t0.45/c0.30 (0.4595; asserted).
- **Folds:** the 141 files span all five V61 folds, so rules were chosen with a true V61-fold cross-fit. No fallback split was needed.

## Primary result (141 files, V17 cell, paired bootstrap 2000)
| arm | PQ | SQ | TP | FP | FN |
|---|---|---|---|---|---|
| r3 baseline | 0.4595 | 0.6894 | 1020 | 448 | 573 |
| **combined, cross-fitted** | **0.4623** | 0.6896 | 1011 | 412 | 582 |

- dPQ = **+0.0028**, 95% CI [−0.0002, +0.0061], P(d>0) = 0.966.
- Per-fold dPQ: +0.006, −0.000, +0.005, +0.002, +0.002.
- Rules chosen in the cross-fit:
  - R1: drop at ρ = 0.5 in 4 folds, merge at ρ = 0.25 in 1.
  - R2: (A = 300, S = 0.5) in 4 folds, (150, 0.5) in 1.
  - R3: τ = 0.06 in all 5.

## Each rule alone (cross-fitted)
| rule | dPQ | 95% CI | P(d>0) | TP/FP/FN change |
|---|---|---|---|---|
| R1 fragment (adjacent, aligned, smaller) | +0.0006 | [0.000, +0.0017] | 0.88 | 0 / −4 / 0 |
| R2 small and low-score | +0.0012 | [−0.0007, +0.0031] | 0.90 | −2 / −13 / +2 |
| R3 low contrast, score < 0.5 | +0.0030 | [+0.0003, +0.0060] | 0.99 | −7 / −32 / +7 |

Pooled results over the whole grid (descriptive only): the best single point is R3 τ = 0.06 at +0.0030. Every grid point lies between +0.0006 and +0.0030.

## Why so small
- **R1 catches almost nothing (4–5 FP).** V69's refined "fragment" FPs (89) are pieces lying mostly inside an annotator's GT. They are not adjacent to a larger, higher-scoring prediction on the same axis; they are usually alone, or sit beside an unmatched filament.
- **The removable FPs are few and cheap.** Low-contrast and small low-score predictions are only about 35 FP. Removing them also removes about 7 TP, because faint and small TPs look the same.
- **About a third of refined FPs (149/448) match another annotator's GT.** No rule can remove those without losing TP in that other record.

## Status
- The post-refiner suppression line is closed.
- R3 (low-contrast drop) is the only rule whose CI excludes 0, but at +0.003 it is below any submission bar.
