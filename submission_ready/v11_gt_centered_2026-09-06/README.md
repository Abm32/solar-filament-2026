# V11 `gt_centered_e4` submission package — 6 September 2026

Ready-to-upload file: `submission_v11_gt_centered.csv`.

- SHA-256: `97e6db03a9ad7cc1e15750d40fd4ebcdeb77504b704dd050ca9a963f73c72a6c`
  (verify with `sha256sum -c SHA256SUMS` from this directory).
- Original: `outputs/kaggle_v11_export_gt_centered/submission.csv`; byte-identical copy.
- Candidate: `v11_gt_centered_e4_refined_constrained`.
- 980 masks; 176 populated image IDs; all 180 test images processed.
- No detections on `20170708164010Mh`, `20200310113230Ch`, `20200806072410Th`,
  `20211214174010Mh` — the identical absent-image set to the accepted control and
  to the 0.39 `proposal_centered` submission.
- Independent validation PASSED: full RLE decode, unique/known IDs, minimum mask
  area 29, non-overlap. See `submission_validation.json`.

## Why this exists: a disambiguation measurement, not a promotion

This is **not** a new model and **not** an attempt to improve the score. It is a
controlled measurement that answers one open question left by the V11 result.

V11's `proposal_centered_e4` scored **0.39** public (submission `56049359`), a new
best over the long-standing 0.38. But locally it beat the matched `gt_centered_e4`
control by only **+0.00069** tuning PQ, while both arms gained roughly +0.0045 over
the frozen control. So the public gain has two competing explanations:

1. **Proposal centering** (the V11 hypothesis) is what helped, or
2. **Continuation training** — simply training the refiner four more epochs at all —
   is what helped, and the crop distribution is irrelevant.

Submitting `gt_centered_e4` separates them:

| Observed public score | Verdict |
| --- | --- |
| **0.39** | The gain is **continuation training**. Crop distribution is not the lever. Cheap continuation headroom likely remains in the refiner track. |
| **0.38** | Proposal centering carries the gain despite a +0.00069 local margin, which would mean local tuning PQ badly under-resolves what transfers publicly. |

Either outcome is informative and de-risks the E-C (learned proposal-quality head)
budget, which is the next planned GPU block.

### Measured: 0.39 — the gain is continuation training

Submitted once as **56050554** at **2026-09-06T08:18:14.807 UTC (13:48 IST)**,
status **COMPLETE**, public score **0.39**. It **ties** `proposal_centered_e4`
(`56049359`, also 0.39) despite scoring 0.000688 *lower* on local tuning PQ.

**Verdict: continuation training, not proposal centering.** Training the refiner
four more epochs from the COCO checkpoint is what moved the public score; which
crop distribution those epochs used did not matter. The V11 headline hypothesis
therefore has no public support either — proposal centering joins instance
conditioning as an unsupported V11 idea, while the incidental continuation is the
part that worked.

Two secondary conclusions:

- Public scoring at two-decimal rounding **cannot resolve** a +0.00069 local
  difference. Do not spend further submissions separating arms this close.
- The local→public calibration set now has three points: +0.0012 → 0.38,
  **+0.0045 → 0.39**, +0.0052 → 0.39. Local gains around +0.004–0.005 do appear
  to move the public score, which further confirms the +0.01 screening gate was
  miscalibrated. Three points still do not make a transfer function.

## This candidate also FAILED the promotion gate

Same gate, same verdict as `proposal_centered_e4`. The V11 experiment required
**at least +0.01 tuning PQ over frozen V6**; the run achieved **+0.005177**, so
`promotable` is false in every artifact. `gt_centered_e4`'s own tuning PQ is
**0.437313** (+0.004489 vs frozen V6), i.e. *below* the selected arm. It is
uploaded to measure, not to promote. Nothing here changes the accepted baseline,
submission **56026663** at 0.38.

## Provenance

Produced by the export-only notebook
`abhimanyu122/solar-filament-v11-export-gt-centered` **version 1**, COMPLETE,
staged source SHA-256
`af0559af9b2e3dd01b797cf41ec42f9fcfa4c9895c0deecdd9196b51543818ba`.
It performs **no training**. It recovers the already-trained checkpoint
`gt_centered_e4.pt` (SHA-256
`80c17bc1ef822de5c45b07c14770b6e9c2c7b2449aa854d0250877a8754d96d9`, confirmed
equal to the local copy in `outputs/kaggle_v11_full/checkpoints/`) from the
completed experiment run `abhimanyu122/solar-filament-v11-refiner` version 1,
run identity `7c45acf26f210acce00f1bff581288bc2eb6121485237c7c028588d1de69164c`,
and reuses that run's frozen pre-audit selection. Export identity
`ff1ade381967a39768be177fb4210c00151109212b85677aaab45b919084d56e`.

Runtime fidelity was proven before exporting the candidate: the frozen control
export reproduced the accepted 0.38 submission **byte-for-byte**, SHA-256
`bf11690e9bcce73b0c627096cc1198d9fb0916650f7fa4b3f08f1167bb76bde9`, 982 rows,
and is byte-identical to the control emitted by the earlier V11 export run.
All 321 frozen raw detector caches, V6 source artifacts, canonical annotations,
crop geometry, threshold 0.40, confidence 0.40, minimum area 20, and pixel
ownership are unchanged from the accepted pipeline. This is a **single-variable**
comparison against the 0.39 submission: only the refiner checkpoint differs.

Environment: Ultralytics 8.4.138, torch 2.10.0+cu128, pycocotools 2.0.11,
NumPy 2.0.2.

Reproduce the package locally with:

```bash
.venv/bin/python scripts/validate_submission.py \
  outputs/kaggle_v11_export_gt_centered/submission.csv
```

## Upload

See `submission_receipt.json` in this directory for the submission ID,
description, and latest observed status. Do not upload this file again.
