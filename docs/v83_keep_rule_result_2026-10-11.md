# V83 result: the agreement-calibrated keep rule is a KILL (2026-10-11)

**Verdict: KILL.** The primary "full" arm gains dPQ = +0.0030 (95% CI [+0.0001, +0.0058], P(d>0) = 0.978, 5/5 folds positive). This is below the pre-registered kill line of +0.005.

**Setup**
- **Pre-registration:** `docs/v83_keep_rule_preregistration_2026-10-11.md`. Its sha256 (`8da3d9e8…`) is stored in `outputs/v83_prereg.sha256`. It was written at 14:28 UTC, before any V83 number existed.
- **Code:** `scripts/v83_extract.py`, then `scripts/v83_rule.py`.
- **Outputs:** `outputs/v83/extract.pkl` and `result.json`.
- **Rules followed:** CPU only, at most 20 worker processes, no submissions, no test data.
- **Same disclosed install as V70:** sklearn is absent, so LightGBM 4.5.0 is loaded from `outputs/v70/pylib`.

## Pipeline reproduction
- The extract re-runs the V73 base refined pipeline on all 707 files.
- At cut 0.30 it gives PQ 0.45393. This matches V73 base 0.4539; the largest per-file count difference is 1.1e-4.
- Candidates: 6,339 refined, post-ownership masks with score ≥ 0.15. Of these, 4,733 are in the baseline set (score ≥ 0.30).

## Results (707 files, pooled PQ over records, cross-fitted by V61 fold)

| arm | PQ | TP / FP / FN | promoted / demoted | dPQ [95% CI] | P(d>0) | per-fold dPQ |
|---|---|---|---|---|---|---|
| baseline (cut 0.30) | 0.4539 | 5270 / 2444 / 2929 | – | – | – | – |
| **full (gated)** | **0.4569** | 5162 / 2157 / 3037 | 274 / 504 | **+0.0030** [+0.0001, +0.0058] | 0.978 | +.0036 +.0002 +.0007 +.0042 +.0065 |
| demote-only | 0.4569 | 5015 / 1868 / 3184 | 0 / 505 | +0.0030 [+0.0007, +0.0052] | 0.995 | +.0021 +.0038 +.0010 +.0043 +.0038 |
| promote-only | 0.4540 | 5418 / 2735 / 2781 | 277 / 0 | +0.0001 [−0.0017, +0.0020] | 0.54 | +.0013 −.0027 −.0009 +.0003 +.0027 |
| oracle (true q·s, not cross-fitted) | 0.5283 | 5536 / 633 / 2663 | – | +0.0744 | – | – |

- **λ per fold:** for the full arm, the Dinkelbach λ on the training folds ranged from 0.452 to 0.461.
- **Ranking quality:** Spearman correlation with the true q·s is 0.590 for ŷ and 0.570 for the consensus score alone.
- **Feature gain:** the consensus score has 31%, mean member score 6%, ownership retention 5% and mean refined probability inside the mask 5%. Every other feature has 4% or less.

## Reading
- **Headroom is large, but the features cannot reach it.**
  - The oracle shows +0.074 is available just from choosing which existing masks to keep.
  - The model barely ranks better than the score alone (Spearman +0.02).
  - The refiner-internal signals add little beyond the consensus score.
- **All of the realised gain comes from demotion.**
  - Removing about 505 of 4,733 kept masks is worth +0.003, and that gain is consistent across folds.
  - Promotion is worth nothing. The 0.15–0.30 extras are, on average, single-annotator hits that look the same as the false positives.
  - This is the same wall that V69 and V70 hit: agreement is not predictable from image or refiner evidence.
- **Not tested here:** a demote-only rule worth about +0.003 could be stacked with other small gains. It does not pass this gate on its own, and any adoption would need a new pre-registration.
- **Disclosed simplification:** demoted masks do not hand their pixels back to lower-ranked masks. The candidate set is pixel-disjoint by construction, because ownership runs over all groups in score order.

The keep/drop re-scoring line on the refined output is closed.
