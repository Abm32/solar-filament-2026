# V13 submission package — E-A `iou=0.3` flags x V11 `proposal_centered_e4` refiner

**Prepared 2026-09-07. Submitted as `56076685`, public score 0.39 — TIES the
previous best, no public improvement.** First candidate in the project to clear a
**promotion gate** on local metrics, and the largest verified local gain so far,
yet the public score did not move. See "Outcome" below before reusing this.

## Outcome, stated plainly

| | tune PQ | audit PQ | Public |
| --- | ---: | ---: | ---: |
| E015 control (accepted) | 0.432823 | 0.450130 | 0.38 |
| V11 `proposal_centered_e4` | 0.438001 | 0.454884 | **0.39** |
| **V13 (this package)** | **0.441777** | **0.464644** | **0.39** |

V13 gained **+0.003776 tune / +0.009760 audit over V11** and still scored the same
0.39. That is the informative result: it adds a fourth point to the local→public
calibration, and the first *negative* one at this magnitude.

Local→public calibration to date: +0.0012 → 0.38, +0.0045 → 0.39, +0.0052 → 0.39,
and now **+0.0038 over the 0.39 → still 0.39**. Local tuning gains in the
+0.003–0.010 band are no longer reliable evidence of public movement. Either the
gain does not transfer, or two-decimal rounding is hiding it (0.39 could be
anything from 0.385 to 0.394, so a true +0.004 can be invisible).

**Do not promote V13 as a public improvement.** It is a local improvement with a
verified runtime and a neutral public result.

## What this is

| Parent | What it changed | tune PQ | audit PQ | Public |
| --- | --- | ---: | ---: | ---: |
| E015 control (accepted baseline) | — | 0.432823 | 0.450130 | 0.38 |
| E-A `iou=0.3` (E023, gate PASSED) | inference flags only | 0.436726 | 0.459054 | not measured |
| V11 `proposal_centered_e4` (E019) | refiner weights only | 0.438001 | 0.454884 | **0.39** |
| **V13 = both** | flags + refiner | **0.441777** | **0.464644** | **0.39** |

Selected candidate `t0.45_c0.30`: threshold 0.45, confidence 0.30,
686 TP / 309 FP / 448 FN on the 94 tuning files.

The composition beats **every** parent on **both** groups locally:

| vs | tune delta | audit delta |
| --- | ---: | ---: |
| E015 accepted control | **+0.008954** | **+0.014514** |
| E-A alone | +0.005052 | +0.005590 |
| V11 alone (the 0.39) | **+0.003776** | **+0.009760** |

The two levers are additive rather than redundant locally, which remains the
substantive finding: E-A recovers missed instances at the proposal stage, V11's
continuation training refines the masks that survive. The public split does not
reward it.

## Why this needed a fresh run instead of a re-export

V11's export path reads the 321 **frozen** raw detector caches, and those were
generated at the Ultralytics default box `iou=0.7`. `iou=0.3` changes the
detector's own NMS, so those proposals cannot be recovered from the frozen files.
V13 regenerates proposals at `iou=0.3` for all 141 validation and 180 test images.
Cache ids are **cell-scoped** (the cell name is hashed into the id) so `iou=0.3`
proposals can never collide with the frozen `iou=0.7` ones.

## Gates, both predeclared before any V13 number was observed

- **SUBMIT_BAR** (authorizes exporting a CSV at all): tune >= 0.436726 (the better
  parent, E-A) **and** audit >= 0.450130 (the accepted E015 floor). **CLEARED.**
- **PROMOTION_GATE** (authorizes replacing the baseline): tune >= V11 + 0.003
  **and** audit >= V11's audit. **CLEARED** (+0.003776 / +0.009760).

Selection was **tune-only** over the predeclared 42-candidate grid (6 thresholds x
7 confidences), bound at runtime to `detector_tta.LOCAL_THRESHOLDS` /
`LOCAL_CONFIDENCES` so the grid cannot widen. All 42 were evaluated; the selected
candidate is the tune-PQ maximum and is inside the declared grid. Audit never
selects.

## Verification

- **Frozen control reproduced the accepted 0.38 submission byte-for-byte:**
  `bf11690e9bcce73b0c627096cc1198d9fb0916650f7fa4b3f08f1167bb76bde9`. The control
  arm ran **first**; nothing downstream would have been trusted without it.
- **Control counts reproduced E015 exactly:** tune `[426.5473399071962, 626, 211, 508]`,
  audit `[182.75261599994462, 265, 88, 194]`.
- **Independent CSV validation PASSED** (`scripts/validate_submission.py`, run
  outside the kernel): 1,198 masks across 177 populated images, non-overlapping,
  every RLE decodes, min mask area 21.
- V11 checkpoint pinned by hash `fae496b9d00819f49c5278e81e41a7917b31d18ea80f827e162746d840d14cd2`,
  run identity verified as the completed V11 run, epoch 4.
- Local suite **186 passed**, including 22 V13 tests covering grid discipline,
  control verification, tune-only selection, and both bars.

## Honest caveat

A paired per-image bootstrap (10,000 resamples, seed 42) against the accepted E015
control gives:

| group | delta | 95% CI | P(delta>0) | P(delta>=0.003) |
| --- | ---: | --- | ---: | ---: |
| tune | +0.008954 | [-0.001171, +0.018949] | 0.960 | 0.879 |
| audit | +0.014514 | [-0.002776, +0.033161] | 0.951 | 0.901 |

The intervals still graze zero, so the local gain was never a certainty — though it
was much stronger evidence than E-A alone (P(delta>0) 0.96 vs 0.79). **The public
result then came back flat at 0.39**, which is consistent with either
interpretation: a real but small gain hidden by two-decimal rounding, or a local
gain that does not transfer. The bootstrap width was the right thing to worry
about.

## Files

| File | Purpose |
| --- | --- |
| `submission_v13.csv` | the upload, sha256 `96a71016cba3195ca2a651718a9e546db8efcb22875e0fd308545a2f9f13c7fa` |
| `SHA256SUMS` | checksum to re-verify before upload |
| `v13_report.json` | both bar decisions and signed deltas vs all three parents |
| `v13_candidates.csv` | all 42 predeclared candidates with counts |
| `v13_manifest.json` | runtime manifest, pins, versions |
| `submission_candidates.json` | control + candidate export records |
| `submission_validation.json` | independent structural validation |
| `staging_manifest.json` | staging pins (checkpoint, caches, detector, annotations) |
| `submission_receipt.json` | written after upload |

Source kernel: `abhimanyu122/solar-filament-v13-compose` v1, script sha256
`f0b941918e4cbac99d8060152b37869663d4a819761fd5e3da343925b5a02161`.
