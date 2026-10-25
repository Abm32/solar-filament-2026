# V94 result: limb-darkening division as detector input (2026-10-19)

**Status: KILL.** Refined fold 0, p2first_limb vs p2first: d = −0.0073, 95% CI [−0.0180, +0.0032],
P(d>0) = 0.09. Gate was d ≥ +0.010 and P ≥ 0.90 (KILL below +0.005). Nothing trained beyond fold 0,
nothing exported or submitted.

- Pre-registration: `docs/v94_limb_division_preregistration_2026-10-19.md` (two hashes in
  `outputs/v94/prereg.sha256`: original, and the 09:25 UTC amendment that moved the arm onto the chunked
  upstream loss so it fits next to a 3.8 GB job on GPU 2; both written before training started).
- Code: `scripts/v94_limb.py` (transform), `scripts/patch_v94.py` (recipe registration),
  `scripts/_v94_chain.sh`, `scripts/v94_refined_fold0.py` (V81 harness, seed 94),
  `scripts/v94_direct_fold0.py` (record-only direct PQ, added after the verdict; not part of the gate).
- Outputs: `outputs/v94/{refined_fold0.json,direct_fold0.json,chain.log,train_fold0.log,refined.log,
  transform_diag_train.json}`; checkpoint `outputs/v94/work/p2first_limb_fold0/mask_ratio1/final.pt`
  (sha256 `d5554da8…`); transformed PNGs `outputs/v94/images/` (1.3 GB); views
  `outputs/v61_oof_cache/p2first_limb/fold0`.
- Harness reproduced V73 base fold 0 exactly (PQ 0.453259, max per-file diff 0.0).
- Training: 80 + 20 epochs on GPU 2 shared with another job, 09:23 → 13:21 UTC (~4 h). Final stage-2 val:
  mask mAP50 0.894, mAP50-95 0.378.

## Numbers (fold 0, 141 held-out files, host-exact per-record PQ)

| | PQ | SQ | TP | FP | FN | d vs p2first | 95% CI | P(d>0) |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| p2first, refined (V73 base) | 0.45326 | 0.6832 | 1028 | 431 | 612 | — | — | — |
| p2first_limb, refined | 0.44601 | 0.6845 | 1023 | 477 | 617 | −0.0073 | [−0.0180, +0.0032] | 0.09 |
| p2first, refined + MIN200 | 0.45546 | 0.6832 | 1028 | 416 | 612 | — | — | — |
| p2first_limb, refined + MIN200 | 0.44885 | 0.6845 | 1022 | 455 | 618 | −0.0066 | [−0.0172, +0.0038] | 0.11 |
| p2first, direct (conf 0.30) | 0.39908 | — | 953 | 513 | 687 | — | — | — |
| p2first_limb, direct (conf 0.30) | 0.39689 | — | 958 | 556 | 682 | −0.0022 | [−0.0147, +0.0104] | 0.37 |

## Reading
- The limb arm changes almost nothing on the recall side (TP 1023 vs 1028 refined, FN 617 vs 612; direct FN
  682 vs 687) and adds false positives (+46 refined, +43 direct). Flattening the limb did not surface new
  faint filaments; it surfaced more non-filament dark structure that the refiner then kept.
- SQ is unchanged (0.6845 vs 0.6832), consistent with the refiner re-drawing every mask from raw crops: the
  detector input only moves which candidates exist, and here the extra candidates were junk.
- Unlike V77/V81 there is no direct gain to be absorbed: the transform is flat direct (−0.002) and flat-to-
  negative refined. The external evidence (theainerd 0.452, ameyypawar 0.421 both using limb correction) does
  not transfer to this detector; their gains likely come from elsewhere in their stacks.
- One seed, one fold: the CI is wide (±0.011), so a true effect of a few thousandths either way cannot be
  excluded, but the pre-registered CONTINUE threshold (+0.010) is outside the interval.

## Decision
Closed under the pre-registered rule. Not authorised and not run: channel stacks (raw + limb + r/R), CLAHE,
other profile fits, threshold changes, folds 1–4. The `p2first_limb` recipe stays registered in
`train_v61_fold.py` (additive; the p2first path is byte-identical to the pre-V94 backup for every other recipe).
