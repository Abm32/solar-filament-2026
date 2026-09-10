# V10 — ready for submission on 2026-09-06

**Already submitted on 2026-09-06; do not upload again.** Kaggle submission
**56044785**, created at **02:14:51.233 UTC / 07:44:51.233 IST**, is **COMPLETE**.
Public score: **0.38**, tying the previous best; confirmed at **02:18:43 UTC /
07:48:43 IST**. [Submission receipt](submission_receipt.json).
The instructions below describe the frozen package and original upload procedure.

Upload **`submission_v10.csv` only** to
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).
Do not upload the JSON reports, this README, or the control CSV.

Status when packaged on 2026-09-05: **validated, NOT SUBMITTED**. No rerun,
retraining, regeneration, or dependency installation is needed to upload.

Candidate: `v10_quality_rank_rescue40` — best V6 detector + COCO refiner + our
custom detector/crop TTA, quality-based overlap ordering, conservative V8 rescue.
Kaggle notebook: `abhimanyu122/solar-filament-v10-quality-rescue`, completed
**version 2**, not the three-image version-1 smoke test.

## Integrity and validation

- SHA-256: `a39da47881431a22cd9b67acabeec624a06e2e2450487f92290048c3e72c2777`.
- 991 masks, 176 populated test images; all 180 test images were processed.
- Four images have no detections, exactly the same four as the accepted control.
- Full RLE decoding, known/unique IDs, minimum mask area and non-overlap passed.
- Source: `outputs/kaggle_v10_quality/submission.csv`, copied without changes.
- `submission_validation.json` describes that original source path; its checksum
  identifies this byte-identical packaged copy too.
- `selected_config.json` and `inference_comparison.json` preserve the selection
  and measured results. They are reference files, not upload files.

Run from this directory before upload:

```bash
sha256sum -c SHA256SUMS
```

Expected response: `submission_v10.csv: OK`. If the check fails, stop; do not
repair/re-encode the CSV or submit a differently named candidate by guesswork.

## When and how to upload

The expected daily reset is **Sunday 6 September 2026, 05:30 AM IST**
(2026-09-06 00:00 UTC). Kaggle's UTC-midnight convention is documented in
[this staff-confirmed discussion](https://www.kaggle.com/competitions/dunnhumbychallenge/discussion/901).
The submission page shows the live remaining allowance/reset countdown;
recheck it after the reset. The 2026-09-05 preflight showed 5 used, 0 remaining.

Use the competition's Submit Predictions button and select `submission_v10.csv`.
Suggested description:

> V10 quality_rank_rescue40 - V6 COCO custom TTA + conservative V8 rescue

Before uploading, check My Submissions in case this candidate was already
submitted manually. After upload, save the submission ID, status and public
score in [the handoff](../../docs/handoff_2026-09-06.md).

The previous best public score remains **0.38**. V10 local validation PQ is
0.439246 vs 0.437873, with uncertainty intervals including zero: a higher public
score is not guaranteed. Keep the old best submission.

There is no scheduled automatic upload. The user will upload manually or ask
the assistant to submit after returning.
