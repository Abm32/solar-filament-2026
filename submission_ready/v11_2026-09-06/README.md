# V11 submission package — 6 September 2026

Ready-to-upload file: `submission_v11.csv`.

- SHA-256: `9dcda2ed8f864dca83f5d4104425948615de933ee19c04b229bc1a81f3a09172`
  (verify with `sha256sum -c SHA256SUMS` from this directory).
- Original: `outputs/kaggle_v11_export/submission.csv`; this copy is byte-identical.
- Candidate: `v11_proposal_centered_e4_refined_constrained`.
- 980 masks; 176 populated image IDs; all 180 test images processed.
- No detections on `20170708164010Mh`, `20200310113230Ch`, `20200806072410Th`,
  `20211214174010Mh` — the identical absent-image set to the accepted control.
- Independent validation PASSED: full RLE decode, unique/known IDs, minimum mask
  area 24, non-overlap. See `submission_validation.json`.

## This candidate FAILED its promotion gate

Read this before treating the result as an improvement. The V11 experiment
required **at least +0.01 tuning PQ over frozen V6** to promote. This candidate
achieved **+0.005177**, so `promotable` is false in every artifact.

| Gate delta | Value | Required |
| --- | ---: | --- |
| tune vs frozen V6 | +0.005177338559703704 | >= +0.01 (**failed**) |
| tune vs V10 | +0.003936376179929257 | >= 0 (passed) |
| audit vs frozen V6 | +0.004754753668608613 | >= 0 (passed) |
| audit vs V10 | +0.003047259766300547 | >= 0 (passed) |

It is uploaded to **measure** whether small positive local deltas transfer to the
public leaderboard, not to promote V11. The accepted baseline remains submission
**56026663** at 0.38 regardless of this result. A tie or a regression here does
not change the baseline; a gain does not retroactively pass the gate.

Local tuning PQ 0.438001 (629 TP / 206 FP / 505 FN) and audit PQ 0.454884
(268 / 87 / 191). The 47-image audit is reused screening evidence, not an
untouched holdout. Per-epoch scores were non-monotonic across all three V11 arms,
so these deltas may be close to run-to-run noise.

## Provenance

Produced by the export-only notebook `abhimanyu122/solar-filament-v11-export`
**version 1**, COMPLETE, staged source SHA-256
`c2dcfbf8270d57426b83ab5677c0bcf31e2319e23f11b066eabcc643a90ef10a`.
It performs **no training**. It recovers the already-trained checkpoint
`proposal_centered_e4.pt`
(SHA-256 `fae496b9d00819f49c5278e81e41a7917b31d18ea80f827e162746d840d14cd2`)
from the completed experiment run `abhimanyu122/solar-filament-v11-refiner`
version 1, run identity
`7c45acf26f210acce00f1bff581288bc2eb6121485237c7c028588d1de69164c`, and reuses
that run's frozen pre-audit selection. No candidate re-selection was possible.

Runtime fidelity was proven before exporting the candidate: the frozen control
export reproduced the accepted 0.38 submission **byte-for-byte**, SHA-256
`bf11690e9bcce73b0c627096cc1198d9fb0916650f7fa4b3f08f1167bb76bde9`, 982 rows.
All 321 frozen raw detector caches, V6 source artifacts, canonical annotations,
crop geometry, threshold 0.40, confidence 0.40, minimum area 20, and pixel
ownership are unchanged from the accepted pipeline.

Reproduce the package locally with:

```bash
.venv/bin/python scripts/prepare_v11_export.py
.venv/bin/python scripts/validate_submission.py outputs/kaggle_v11_export/submission.csv
```

## Upload

Submitted once on 6 September 2026; see `submission_receipt.json` in this
directory for the submission ID, description, and latest observed status.
Do not upload this file again.
