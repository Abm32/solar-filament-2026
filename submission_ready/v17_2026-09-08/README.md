# V17 submission package — `mask_ratio=1` detector x V13's measured recipe

**Prepared 2026-09-08. Submitted as `56103401`. Public score returned: 0.39 —
a tie, the fifth consecutive 0.39.** See
[`submission_receipt.json`](submission_receipt.json) and ledger entry E027.

## Result

**0.39 — no movement.** V17 was +0.006926 tune / +0.006119 audit above V13
locally and +0.015879 tune above the 0.38-scoring E015 baseline, and the public
score did not change.

This submission was worth making and it answered its question. Its stated purpose
(`docs/v17_design_2026-09-08.md`) was to be "a calibration point on whether a
proposal-stage gain transfers where inference-stage gains did not." **It does not.**
V17's advantage was mechanistically grounded and replicated across two training
seeds (E026's raw proposal IoU), unlike V13's inference-side rearrangement, and it
transferred no better.

**It also falsifies the arithmetic the whole project was steering by.**
`ceiling_analysis_2026-09-07.md` §1 claimed a stable local→public offset of
−0.050 ± 0.003 and concluded "public 0.55 requires local tune PQ ≈ 0.60". Under
that model V17 predicted public 0.3959–0.4014. Running
`scripts/analyze_public_transfer.py` over all five submissions:

- the feasible **constant-offset** region is now **empty** under both
  round-to-nearest and truncation display models — no single offset fits all five;
- for affine models `public = a·local + b`, feasible slopes are **a ∈ [0.01, 0.87]**
  and **a = 1.00 is ruled out**;
- at the most favourable feasible slope (0.87), local 0.60 yields public ≈ **0.527**,
  and public 0.55 would require local ≥ **0.6269**.

So the local target was understated, and local tune PQ is a **weaker** predictor of
public score than assumed. Do not quote the −0.050 offset again.

## What this is

Composes V16's `mask_ratio=1` detector retrain (replicated across two independent
training seeds in E026) with V13's measured recipe: box `iou` swept over
`{0.3, 0.7}`, V11 `proposal_centered_e4` refiner, an 84-candidate grid (2 cells x
6 thresholds x 7 confidences), evaluated at 2 seeds. **Trains nothing** — both
`arm_r1` detector checkpoints, the V11 refiner, and every cached proposal are
pinned by SHA-256 and already existed on disk before this run, so E026's
detector-training noise floor (±0.007677 tune PQ) does not apply here.

Selected candidate: `mr1_iou030/t0.45_c0.30` (box `iou=0.3`, threshold 0.45,
confidence 0.30), from seed 1 (the pre-registered primary seed).

| | tune PQ | audit PQ | Public |
| --- | ---: | ---: | ---: |
| E015 control (accepted) | 0.432823 | 0.450130 | 0.38 |
| V11 `proposal_centered_e4` | 0.438001 | 0.454884 | 0.39 |
| V13 compose (best measured to date) | 0.441777 | 0.464644 | 0.39 |
| V16 `arm_r1` seed1 (detector alone, default `iou=0.7`) | 0.431604 | 0.452410 | not measured |
| V16 `arm_r1` seed2 | 0.434795 | 0.448140 | not measured |
| **V17 (this package), seed1** | **0.448703** | **0.470763** | **0.39** |
| V17, seed2 (corroboration only, did not select) | 0.444860 | 0.468300 | not measured |

V17 beats **every** parent on **both** groups, in **both** seeds:

| vs | tune delta | audit delta |
| --- | ---: | ---: |
| E015 control | +0.015879 | +0.020634 |
| V11 alone | +0.010702 | +0.015879 |
| V13 compose (the 0.39) | **+0.006926** | +0.006119 |
| V16 `arm_r1` seed1 | +0.017099 | +0.018354 |
| V16 `arm_r1` seed2 | +0.013907 | +0.022623 |

## Gates, both predeclared before any V17 number was observed

- **SUBMIT_BAR**: tune >= 0.441777 (V13) and audit >= 0.450130 (E015 floor).
  **CLEARED** — tune +0.006926, audit +0.006119 above the respective floors.
- **PROMOTION_GATE**: tune >= V13 + 0.010 and audit >= V13's 0.464644. **NOT
  CLEARED** — tune delta is +0.006926, short of the +0.010 requirement (audit
  delta alone, +0.006119, does clear its half).

So this is authorized to export and measure, but not to be called an established
gain over V13 on local metrics alone. It is being submitted anyway per
`docs/v17_design_2026-09-08.md`'s own reasoning: this is the first candidate whose
edge traces to a *replicated, mechanistic* proposal-quality improvement (raw
proposal IoU ceiling broken in two independent training seeds, E026) rather than
another inference-side rearrangement (V13, E-A) — worth one submission as a
calibration point on whether that class of gain transfers where V13's did not
(E024: V13 cleared both its bars and the public score did not move).

## Honest caveat: seed spread

Seed1 tune 0.448703 vs seed2 tune 0.444860 — a spread of **0.003843**. Smaller
than E026's measured training-noise floor (0.007677) but not negligible relative
to the +0.006926 margin over V13. Both seeds independently clear SUBMIT_BAR
(seed2: tune +0.003083, audit +0.003656 vs V13), so the composition's benefit over
V13 is not an artifact of one lucky seed — but the exact size of that benefit is
seed-sensitive, and selection used seed1 only per the predeclared protocol (audit
never selects, seed2 never selects).

## Verification

- **Frozen control reproduced E015 exactly:** tune `[426.5473399071962, 626, 211,
  508]`, audit `[182.75261599994462, 265, 88, 194]`. Control CSV sha256
  `bf11690e9bcce73b0c627096cc1198d9fb0916650f7fa4b3f08f1167bb76bde9` — matches the
  accepted 0.38 file byte-for-byte.
- **Independent CSV validation PASSED** (`scripts/validate_submission.py`, run
  outside the kernel): 1,215 masks, 177 populated images, non-overlapping, every
  RLE decodes, min mask area 20.
- Both `arm_r1` detector checkpoints (seed 42, seed 20260908), the V11 refiner
  checkpoint, and the frozen V6 detector are pinned by SHA-256 in
  `v17_manifest.json`; no training occurred in this run (`trains: false`).
- Raw proposal IoU ceiling: mean 0.5693/0.5733 (seed1/seed2), max 0.9119/0.9093 —
  both exceed `ceiling_analysis`'s prior 0.8734 cap, consistent with E026.

## Files

| File | Purpose |
| --- | --- |
| `submission_v17.csv` | the upload, sha256 `87b22b056ca830527664a92db8daa76e2c93660cd232ae45346b28d4c1cb922b` |
| `SHA256SUMS` | checksum to re-verify from the repository root with `sha256sum -c submission_ready/v17_2026-09-08/SHA256SUMS` |
| `v17_report.json` | both bar decisions and signed deltas vs all five parents |
| `v17_manifest.json` | runtime manifest, checkpoint/cache pins, versions |
| `v17_submission_validation.json` | independent structural validation |

Source kernel: `abhimanyu122/solar-filament-v17-compose` v1, smoke kernel
`abhimanyu122/solar-filament-v17-compose-smoke` v1 (both `COMPLETE`).
Design doc: `docs/v17_design_2026-09-08.md`.
