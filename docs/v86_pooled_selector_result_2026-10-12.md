# V86 result: pooled multi-detector candidates + learned selector (2026-10-12)

Pre-registration: `docs/v86_pooled_selector_preregistration_2026-10-12.md` (sha256 in `outputs/v86_prereg.sha256`,
written before any number). Code: `scripts/v86_extract.py`, `scripts/v86_select.py`. Outputs: `outputs/v86/`
(`extract.pkl`, `result.json`, `extract.log`, `select.log`). CPU only (26 procs), no GPU used, no test data.
Runtime: extract 30 min (5 sources x 707 files, refiner on CPU), selector 14 min (5 arms).

## Verdict: **KILL**
Primary P3 (p2first + p2first_1280 + baseline) vs V84: d = +0.0003, 95% CI [-0.0032, +0.0039], P(d>0) = 0.56.
Gate (d >= +0.010 and P(d>0) >= 0.90) not met. The pooled selector is not adopted; V84 stays.

## Checks
- Sources with full 5-fold OOF view caches (`outputs/v61_oof_cache`): p2first, p2first_1280, baseline, p2first_cp,
  baseline_seed2. p2first_ns (V76) has fold 0 only and was excluded.
- p2first extract reproduces `outputs/v84/soft_fold*.json` t0.50_c0.30 per-file counts (max diff 0.0008 <= 1e-3).
- Selection-mechanics parity: P1 pool, detector-score order, cut 0.30, MIN200 = 0.456729, exactly V84 `loo_demote`.
- V84 baseline per-file counts (bundle_ref.pkl) = 0.45898 (asserted).

## Results (707 files, pooled per-record PQ; baseline V84 = 0.45898)

| Arm | Sources | Candidates | PQ | d vs V84 | 95% CI | P(d>0) | TP / FP / FN | Oracle PQ (ceiling d) |
|---|---|---|---|---|---|---|---|---|
| P1 (control) | p2first | 6344 | 0.45758 | -0.0014 | [-0.0037, +0.0010] | 0.14 | 5175 / 2165 / 3024 | 0.5285 (+0.070) |
| P2 | + p2first_1280 | 12602 | 0.46095 | +0.0020 | [-0.0013, +0.0055] | 0.88 | 5318 / 2337 / 2881 | 0.5444 (+0.086) |
| **P3 (primary)** | + baseline | 18850 | 0.45924 | +0.0003 | [-0.0032, +0.0039] | 0.56 | 5301 / 2366 / 2898 | 0.5510 (+0.092) |
| P4 | + p2first_cp | 25479 | 0.45815 | -0.0008 | [-0.0046, +0.0032] | 0.33 | 5282 / 2343 / 2917 | 0.5579 (+0.099) |
| P5 | + baseline_seed2 | 31822 | 0.46001 | +0.0010 | [-0.0030, +0.0054] | 0.69 | 5312 / 2339 / 2887 | 0.5610 (+0.102) |

Per-fold d (folds 0-4): P1 [+0.0007, -0.0071, -0.0016, -0.0007, +0.0020]; P2 [+0.0035, -0.0028, +0.0005, +0.0034,
+0.0053]; P3 [+0.0023, -0.0081, -0.0003, +0.0028, +0.0048]; P5 [+0.0049, -0.0085, -0.0022, +0.0080, +0.0028].
P3 vs P1 (gain from extra sources, same selector): +0.0017, CI [-0.0016, +0.0049], P = 0.84.
Fold lambdas (P3): 0.458, 0.462, 0.454, 0.459, 0.461.

P3 top LightGBM gain: agree_mean_score 0.38, retain 0.09, score 0.05, msc_baseline 0.03, iou_baseline 0.03,
agree_max_pin 0.02, ... (cross-source agreement dominates the model).

## Reading
- The pool does raise the ceiling: oracle selection goes from +0.070 (p2first only) to +0.092 (P3) and +0.102 (P5).
  The extra sources contribute ~+0.02-0.03 of reachable PQ, mostly recall (oracle).
- The learned selector captures almost none of it: TP rises by ~125-145 vs P1 but FP rises by ~170-200; SQ is flat
  (0.687). Adding sources turns into about as many new FP as new TP, the same failure as V52/V53 and V66 voting.
- The control P1 (new selector over all score >= 0.15 candidates with greedy ownership by y-hat) is 0.0014 below V84:
  the selector change itself is neutral to slightly negative, so V84's demote rule remains the better single-pool rule.
- Gains are not monotone in sources (P2 > P3 < P5) and fold 1 is negative in every arm: noise-level differences.
- Nothing here clears even a +0.005 bar; no follow-up on pooling is recommended without a much stronger candidate
  quality signal (the q*s regressor's ranking is the bottleneck, not the pool).
