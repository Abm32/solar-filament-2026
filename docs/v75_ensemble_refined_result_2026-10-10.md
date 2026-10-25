# V75 result: E_p2x2 through r3 — KILL (2026-10-10)

Pre-registration: `docs/v75_ensemble_refined_preregistration_2026-10-10.md` (sha256 4735c802…).
Code: `scripts/v75_ensemble_refined.py`. Output: `outputs/v75/{result.json,cache.pkl,run.log}`.
CPU only; no training, no test data, no submission. Self-check: members=[p2first] reproduced the
V71 r3 counts on 12/12 files; the full run asserted r3 = 1020/448/573.

| arm (141 files, t0.45/c0.30) | PQ | SQ | TP | FP | FN |
|---|---|---|---|---|---|
| r3 (p2first) | 0.4595 | 0.6894 | 1020 | 448 | 573 |
| E_p2x2 (p2first + p2first_cp) + r3 | 0.4611 | 0.6912 | 1030 | 465 | 563 |

- Primary dPQ **+0.0016**, 95% CI [-0.0060, +0.0094], P(d>0) 0.64. **< +0.005, so KILL.**
- Per fold: -0.006, -0.003, +0.005, +0.003, +0.010 (3/5 positive).
- Secondary, both arms with MIN200: +0.0025, CI [-0.0057, +0.0111], P 0.71.

Reading: the direct +0.0063 (V66, 707 files, 5/5 folds) mostly does not survive the refiner, the
same ~75-80% absorption seen for p2first. The ensemble adds TP (+10) but also FP (+17). The
"voting changes ranking, which survives refinement" argument from `v69_history_audit` 1f did not
hold here. Not tested: a refiner retrained on ensemble proposals. Line closed; no fold-99
p2first_cp model is needed.
