# V66 result: consensus-VOTING ensembles of OOF detectors. CLOSED (2026-10-07)

Pre-registration: `docs/v66_preregistration_2026-10-07.md` (written before any ensemble number).
Evaluator: `scripts/eval_v66_ensemble.py`; tests: `tests/test_v66_ensemble.py` (4 pass, incl. the
"ensemble of one model == that model" invariant and the 1/n cap on single-member detections).
Direct detector, all 707 files, V17-validated scorer; confidence cut CROSS-FITTED per fold (chosen on
the other four folds); paired file-bootstrap (2000). CPU only; nothing submitted.

## Stage 1 result

| ensemble | members | vs | dPQ | 95% CI | folds + | FP ratio | verdict |
|---|---|---|---|---|---|---|---|
| **E_all4 (primary)** | p2first, p2first_cp, baseline, baseline_seed2 | p2first | **-0.0026** | [-0.0069, +0.0017] | 1/5 | 0.97 | **kill** |
| E_mix | p2first, baseline | p2first | -0.0041 | [-0.0078, -0.0002] | 1/5 | 1.15 | kill |
| E_p2x2 | p2first, p2first_cp | p2first | +0.0063 | [+0.0025, +0.0101] | 5/5 | 1.00 | inconclusive |
| E_base2 (seed control) | baseline, baseline_seed2 | baseline | +0.0040 | [+0.0006, +0.0074] | 5/5 | 1.01 | kill (< +0.005) |

References: p2first 0.4046 (cross-fitted cut), 0.4060 (fixed 0.35); baseline 0.3848 / 0.3856.
At the fixed 0.35 cut: E_all4 -0.0040, E_p2x2 +0.0049, E_mix -0.0065, E_base2 +0.0037.

## What it means
- The pre-registered primary (E_all4 - p2first) is a kill. E_all4 was the declared upper bound for any
  equal-weight subset, so voting ensembles of these members are closed; no weight tuning is allowed.
- Voting fixes the old failure mode (union ensembles inflated FP: here FP ratio ~1.00), and it does
  average away seed variance (+0.004 for two baseline seeds, +0.006 for two p2first-family members,
  both 5/5 folds). It does NOT beat the bar, and mixing a weaker architecture at equal weight lowers PQ.
- The +0.0063 of E_p2x2 is below +0.010 on the DIRECT detector. The refiner absorbed ~80% of the
  direct gain of p2first (+0.0238 direct -> +0.0049 refined), so a deployable seed ensemble would be
  expected to add roughly +0.001-0.002 on the refined pipeline. That is a forecast, not a measurement.

## Not run (and why)
Stage 1b (E_p2seed = p2first + p2first_seed2) needs p2first_seed2 trained on folds 1-4 (~3.5 GPU-h).
Not run: E_p2x2 already bounds the same-architecture two-member pattern at +0.006 < +0.010, and the
refiner absorption makes a refined pass implausible. No weight sweep and no re-reading of the gate.

## Line status
Ensembling: CLOSED. Best measured real pipeline remains p2first + r3 refiner, 0.4595 vs 0.4517
(+0.0078, P 0.92, below the +0.010 bar). V65 public score (that pipeline): 0.39, tied with V17.
