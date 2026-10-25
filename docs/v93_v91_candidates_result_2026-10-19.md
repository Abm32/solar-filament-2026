# V93 result: V91 instances as extra candidates inside the V84 stack (2026-10-19)

**Status: KILL.** Arm A d = +0.0011 vs V84 0.45898, 95% CI [−0.0010, +0.0031], P(d>0) = 0.85, 3/5 folds
positive. Gate was d ≥ +0.005, P ≥ 0.95, ≥ 4/5 folds. Nothing exported or submitted.

- Pre-registration: `docs/v93_v91_candidates_preregistration_2026-10-19.md` (sha256 `d3c6a7af…`, in
  `outputs/v93/prereg.sha256`).
- Code: `scripts/v93_v91_candidates.py`; `scripts/v83_extract.refine_feats` gained an optional `occupied`
  argument; `scripts/v91_spine.py` now builds its own `imgs.u8` and accepts `--fold 99`.
- V91 folds 3 and 4 were retrained today (same recipe, 31 min each); folds 0–2 reused the Oct 12 models.
  All five fold predictions were regenerated into `/mnt/v91` after the Oct 17 wipe.
- Outputs: `outputs/v93/{extract.pkl,result.json,extract.log,score.log}`.

## Numbers (707 files, host-exact pooled PQ)

| | PQ | d vs V84 | 95% CI | P(d>0) | per-fold d | masks added | TP rows / FP rows added |
|---|---:|---:|---|---:|---|---:|---|
| V84 base | 0.45898 | — | — | — | — | — | — |
| A: λ = V84 fold λ | 0.46010 | +0.0011 | [−0.0010, +0.0031] | 0.85 | +.0043 −.0060 +.0030 −.0003 +.0050 | 285 | 175 / 300 |
| B: λ re-solved (Dinkelbach) | 0.46011 | +0.0011 | [−0.0009, +0.0031] | 0.86 | same | 284 | 174 / 298 |

Candidate funnel: 9863 V91 instances decoded at (t_f 0.4, t_s 0.3) → 2012 eligible (area ≥ 200 px, ≤ 10 %
inside the V84 union) → 1992 survive refinement → 285 kept by the rule.

## Reading
- The refiner + quality rule does better than the V91 logistic fusion (+0.0011 vs +0.0002) but the kept
  additions are still net-neutral: 175 TP rows against 300 FP rows, almost exactly the break-even ratio
  (an added mask pays iff its TP rows ≥ ~PQ/2 ≈ 0.23 of its rows... in pooled terms the 285 masks add
  +175 TP and +300 FP, which at PQ 0.46 is a wash).
- The eligible pool is small (2012) because the V91 instances that V84 misses mostly overlap V84 masks
  that V84 already kept or rejected; the orthogonal recall from V91 lives in single-annotator filaments
  the quality model cannot tell from junk, the same wall as V83 promote and V86.
- Fold 1 is negative in both arms, as in V86: the noise floor of these selection levers is ±0.005 per fold.

## Decision
Closed under the pre-registered rule. No grid, overlap, feature or λ re-tuning. The V91 all-data model
(`outputs/v91/f99`, started before the verdict) is kept only as an artefact; it is not exported.
