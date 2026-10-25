# V67 result: magnetogram diagnostic. INVALID (positive control failed), line closed (2026-10-07)

Pre-registration: `docs/v67_preregistration_2026-10-07.md`. Output: `outputs/v67_diag.log`,
`outputs/v67_diag.json` on rogii (written up 2026-10-10 from those files). CPU only, train files only.

- Frames: 691 with a magnetogram; 672 pass the geometry rule (19 excluded). 5400 proposals, 7841 GT.
- **Positive control: AUC 0.589 (n 7841) < 0.60, so the result is a FAIL and the diagnostic is INVALID.**
  The registered PIL-distance feature cannot reliably tell real filament positions from random on-disk translates.
- Reported only, not to be read as a result per the pre-registration:
  - D1 (TP vs FP): AUC base 0.782, augmented 0.781, d -0.0006, CI [-0.0021, +0.0008], 3/5 folds.
  - D2 (detected vs missed): AUC 0.631 vs 0.634, d +0.0031, CI [-0.0006, +0.0067], 4/5 folds.
- Verdict (mechanical): `invalid_positive_control_failed`. Neither D-test would have reached the +0.015
  kill line anyway. No auxiliary-head training was run. Magnetograms were used only for this train-set
  analysis, never for training a submitted model or at inference.
