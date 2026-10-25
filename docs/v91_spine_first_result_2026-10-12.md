# V91 "spine-first" dense instance segmentation: result (2026-10-12)

**Status: CLOSED AT G2.** Standalone PQ 0.4333 is below 0.45. Fusion with V84 gives d = +0.0002 with P = 0.55. G3 was not run, no CSV was exported, and nothing was submitted.

- Pre-registration: `docs/v91_spine_first_preregistration_2026-10-12.md` (sha256 `9628391c…`, recorded in `outputs/v91/prereg.sha256` before any result).
- Code: `scripts/v91_spine.py` (decoder self-check: `python scripts/v91_spine.py demo`).
- Diagnostics: `scripts/v91_g1diag.py`, `scripts/v91_s0probe.py`, `scripts/v91_scorer_check.py`.
- Outputs: `outputs/v91/` (`g1_oracle.json`, `g1_diag.json`, `f0_sanity/s0.json`, `g2.json`, `f*/train.log`). Caches are in `/mnt/v91`.
- Wall time: about 50 min (15:50–16:40 UTC). Fold-0 training took 31 min on one T4 (6,000 steps, batch 4 × 1024² tiles, 2.9 GB peak).

## Paradigm (as pre-registered)

- No YOLO, box detector, crop refiner or COCO/HF weights.
- Network: a ResNet18 U-Net (`CropUNet`) initialised from the in-domain MIM SSL state `outputs/v89/ssl_mim/ssl_state.pt`. It predicts, at native resolution:
  - soft foreground (mean over annotator records);
  - an expert-spine Gaussian heatmap (σ 2.5, mean over records).
- Losses: BCE + Dice on foreground; quality-focal loss on the spine, normalised by positive mass.
- Decoder:
  1. Spine components inside the foreground become seeds (skeleton ≥ 8 px).
  2. A flat-image watershed assigns foreground pixels to their geodesically nearest seed.
  3. Seedless foreground components become their own instances.
  4. A logistic match-probability score feeds a single keep threshold τ.
- The thresholds (t_f, t_s), the logistic model and τ are cross-fitted on held-out fold-0 halves A/B (by index parity).

## Scorer check

The per-file host-exact bookkeeping reproduces host V84 fold 0: 0.458987773574 vs 0.458987773562 (982 TP / 312 FP / 658 FN).

## G1: decoder oracle on GT targets (CPU, 707 files)

| variant | PQ | TP / FP / FN | 1-rec | 2-rec | 3-rec |
| --- | ---: | --- | ---: | ---: | ---: |
| per-record hard targets, scored vs own record | **0.986** | 8191 / 218 / 8 | 0.993 | 0.980 | 0.983 |
| soft mean targets t_f 0.5, t_s 0.5 (pre-registered) | 0.719 | 7258 / 2470 / 941 | 0.993 | 0.687 | 0.489 |
| soft mean targets t_f 0.5, t_s 0.3 | 0.761 | 7358 / 1637 / 841 | 0.993 | 0.695 | 0.574 |

The pre-registered G1 (≥ 0.85 on soft targets) **failed as written**, but the cause is not the decoder:
- The decoder reconstructs every single record almost perfectly (0.986 per record; 0.993 on the 411 single-record files).
- The gap is entirely in multi-record files. There, one consensus set of non-overlapping instances is scored against 2–3 disagreeing annotators.
- That consensus ceiling (~0.72–0.76) bounds any single-prediction method, V84 included.

Under the pre-registered rule, "fix the decoder" was the only allowed response, and nothing in the decoder needed fixing. Training therefore proceeded, and this deviation is recorded here.

## S0: engineering overfit (8 images, 600 steps)

- No collapse: on all 8 images, pixels with P_fg ≥ 0.5 were 0.91–1.75× the GT foreground (E044/E045 failed exactly here).
- Keep-all PQ was 0.26 (< 0.5). Seeds were mostly correct (39 seeds for 51 GT, 35 TP pairs). The FPs were low-score seedless foreground blobs, which the keep rule removes.
- S0 was pre-registered as non-blocking.
- On the full fold-0 run the foreground ratio stayed at 0.7–1.5 throughout, and every loss was finite.

## G2: fold-0 model on 141 held-out files (cross-fitted)

| arm | PQ | d vs V84 0.45899 | P(d > 0) | TP / FP / FN |
| --- | ---: | ---: | ---: | --- |
| p2first direct (reference) | ≈0.40 | | | |
| V84 full | 0.45899 | — | — | 982 / 312 / 658 |
| **V91 standalone** | **0.43328** | −0.02571 | 0.0005 | 991 / 455 / 649 |
| **V91 + V84 fusion** (add non-overlapping V91, < 10% overlap) | **0.45919** | **+0.00020** | 0.546 | 1045 / 422 / 595 |

- The selected grids were (0.4, 0.3)/(0.5, 0.3) with τ 0.34/0.39 for standalone, and (0.3, 0.3)/(0.5, 0.3) with τ 0.41/0.48 for fusion.
- Standalone fit-half PQ was 0.442/0.435 and transfer PQ 0.433, so the cross-fit is not badly optimistic.

**Orthogonality (standalone keep set):**
- V91 recovers **138 of V84's 658** FN (GT, record) pairs.
- 90 of V91's 236 never-TP FPs coincide (IoU > 0.5) with a V84 FP.

**Reading:**
- V91 standalone matches V84's recall (991 vs 982 TP; FN 649 vs 658). That makes it the first non-detector model in this project at V84-level recall, and it clearly beats p2first direct (~0.40).
- It loses on precision: +143 FP, and SQ ≈0.675 vs 0.686.
- It is substantially orthogonal: it recovers 21% of V84's misses.
- Simple add-only fusion converts that orthogonality into almost nothing. It gains +63 TP but adds +110 FP, because the non-overlapping V91 additions are mostly instances V84 already rejected for good reason, or instances that are ambiguous between annotators.

## Verdict

- **G2 FAIL → STOP.** Standalone 0.4333 < 0.45, and fusion d = +0.0002 is far below +0.008 (P 0.55).
- My own fold 3/4 trainings, launched only for wall-time savings, were stopped. The fold-1/2 models and predictions exist (`outputs/v91/f1`, `f2`, `/mnt/v91/pred_f{1,2}.u8`) but were never decoded or scored, as pre-registered.

## What would be worth testing next (not done)

- The recall-level parity suggests V91 is a competitive candidate generator. The weak link is the precision of its keep score: five hand features with a logistic score.
- One option is to use V91 as an extra candidate source inside the existing V84 selector, scored by the refiner/selector features, instead of add-only fusion. That reuses the old stack, so it was out of scope for this from-scratch paradigm.
- Within the paradigm, the obvious upgrades are:
  - a learned per-instance quality head;
  - longer training (the loss was still falling at 6k steps);
  - flip TTA.
- None is pre-registered here.
