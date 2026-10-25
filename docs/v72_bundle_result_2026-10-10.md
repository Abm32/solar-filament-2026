# V72 result: the post-process bundle does not stack. DO NOT ADOPT (2026-10-10)

**Verdict: DO NOT ADOPT.** On the real pipeline (p2first + r3 at the V17 cell), the bundle gains +0.0034 PQ with P(d>0) = 0.984. That misses the pre-registered +0.004 bar. The rules mostly remove the same false positives (FPs), so their gains do not add up.

- **Pre-registration:** `docs/v72_bundle_preregistration_2026-10-10.md` (sha256 adc7f871…), written 10:59 UTC, before any V72 number.
- **Code:** `scripts/v72_bundle.py`.
- **Outputs:** `outputs/v72/{cache_baseline.pkl,result.json,eval.log,cache_base.log}`.
- **Run conditions:** CPU only, no training, no submission, no test data.
- **Reproduction checks:**
  - The r3 arm reproduces 0.4595 (1020/448/573) from the V71 cache, which V71 had already asserted per file against `outputs/v63/r3.json`.
  - The baseline + V11 context arm was rebuilt on CPU and reproduces 0.4517 (1004/464/589), matching the V69 metric audit. This was asserted.
- **One fix before any V72 number was seen:** the first baseline cache run failed its reproduction assert (TP 496). The probe module stubs `v11.build_model` to an untrained network. `init_base` now loads the canonical + V11 weights exactly as `reproduce_v17_from_cache` does. No rule or parameter changed.

## Bundle and parameters (all fixed from prior docs)
- **a. MIN200:** drop if final area < 200.
- **b. R3:** drop if contrast < 0.06 and score < 0.5.
- **c. R2:** drop if area < 300 and score < 0.5.
- All three are drop-only and act on the final kept masks.

## Primary: p2first + r3, V17 cell, 141 files (paired bootstrap 2000, seed 72)
| arm | PQ | SQ | TP | FP | FN | dPQ | 95% CI | P(d>0) |
|---|---|---|---|---|---|---|---|---|
| r3 baseline | 0.4595 | 0.6894 | 1020 | 448 | 573 | – | – | – |
| **bundle a+b+c** | **0.4629** | 0.6896 | 1011 | 408 | 582 | **+0.0034** | [+0.0003, +0.0068] | 0.984 |
| without MIN200 (b+c) | 0.4629 | 0.6896 | 1011 | 408 | 582 | +0.0034 | [+0.0003, +0.0068] | 0.984 |
| without R3 (a+c) | 0.4613 | 0.6892 | 1018 | 431 | 575 | +0.0018 | [−0.0003, +0.0040] | 0.954 |
| without R2 (a+b) | 0.4629 | 0.6898 | 1013 | 413 | 580 | +0.0034 | [+0.0007, +0.0064] | 0.994 |
| only MIN200 | 0.4613 | 0.6894 | 1020 | 436 | 573 | +0.0018 | [+0.0005, +0.0034] | 0.999 |
| only R3 | 0.4625 | 0.6898 | 1013 | 416 | 580 | +0.0030 | [+0.0005, +0.0059] | 0.989 |
| only R2 | 0.4613 | 0.6892 | 1018 | 431 | 575 | +0.0018 | [−0.0003, +0.0040] | 0.954 |
| bundle, V71 per-fold R2 (xfit) | 0.4623 | 0.6896 | 1011 | 412 | 582 | +0.0028 | [−0.0002, +0.0059] | 0.967 |

- Per-fold dPQ for the bundle: +0.006, −0.000, +0.005, +0.002, +0.004.
- **Gate:** dPQ +0.0034 < +0.004, so DO NOT ADOPT. The P(d>0) condition passes.
- The less optimistic per-fold cross-fit arm (xfit) gives +0.0028.

## Interaction (r3): the rules remove the same FPs
- The singles sum to +0.0066, but the bundle gives only +0.0034.
- Venn of the 30 predictions dropped. Each prediction counts once per annotator record, so the numbers are TP-records / FP-records:

| region | preds | TP-records lost | FP-records removed |
|---|---|---|---|
| a ∩ b ∩ c | 6 | 0 | 9 |
| a ∩ c only | 1 | 0 | 3 |
| b only | 17 | 7 | 23 |
| c only | 6 | 2 | 5 |

- **MIN200 is a strict subset of R2.** Every kept prediction under 200 px on r3 also has score < 0.5. Once R2 is applied, MIN200 adds nothing, which is why the "without MIN200" arm equals the full bundle.
- R2's extra band (200–300 px, score < 0.5) removes 5 FP-records but costs 2 TP-records. That is roughly neutral: without R2 the bundle still gives +0.0034.
- The only independent gain is R3's 17 predictions that only it drops (23 FP-records, but 7 TP-records lost). Effectively, the bundle is R3 + MIN200.

## Context: baseline detector + V11 refiner (public V17 style; not gated)
| arm | PQ | TP | FP | FN | dPQ | 95% CI | P(d>0) |
|---|---|---|---|---|---|---|---|
| baseline + V11 | 0.4517 | 1004 | 464 | 589 | – | – | – |
| bundle a+b+c | 0.4564 | 989 | 404 | 604 | +0.0047 | [+0.0013, +0.0079] | 0.997 |
| without MIN200 | 0.4558 | 989 | 408 | 604 | +0.0041 | [+0.0007, +0.0072] | 0.994 |
| without R3 | 0.4554 | 998 | 426 | 595 | +0.0037 | [+0.0010, +0.0065] | 1.000 |
| without R2 | 0.4564 | 995 | 417 | 598 | +0.0047 | [+0.0019, +0.0075] | 1.000 |
| only MIN200 / R3 / R2 | 0.4554 / 0.4549 / 0.4548 | | | | +0.0037 / +0.0032 / +0.0031 | | |
| bundle xfit | 0.4563 | 989 | 405 | 604 | +0.0046 | [+0.0012, +0.0077] | 0.997 |

- On the weaker baseline pipeline, the bundle would clear +0.004. Overlap is again heavy: the singles sum to +0.010, the bundle gives +0.0047.
- Here MIN200 is not fully inside R2: 3 predictions are dropped by MIN200 alone, all FP.
- R2 adds nothing over a+b, as on r3.

## Status
- **V72:** do not adopt the bundle into the r3 final export. The stack is about R3 + MIN200, and on r3 that totals +0.003.
- **MIN200 alone:**
  - It stays the metric audit's standalone recommendation. That is a separate decision, not part of the V72 gate.
  - On r3 it is +0.0018 (P = 0.999) and never removes a TP.
- **The post-refiner suppression line is exhausted.** Any further variant (e.g. R3 alone, or other τ) is a new experiment.
