# V60A candidate-conditioned mask refiner — result (2026-10-04)

## Setup (controlled A/B, FIT-only OOF)
Compact 2-ch U-Net (1.9M params), input = [1-ch grayscale 1024 crop resized to 256, candidate
seed mask], outputs mask head + spine head. Loss = 1.0*Dice + 0.5*Focal(mask) + spine_w*BCE(spine).
Trained per-fold (4 OOF folds), 25 epochs, AdamW 1e-3, batch 16. Two variants, identical except
spine: S1 spine_w=0, S2 spine_w=0.2.

Seed construction (FIXED a critical bug first): the original fixed-7px erosion left 92% of seeds
EMPTY (median GT mask only ~67px at 256) and eroded the rest to 2.3% area — the refiner would
have learned "segment from grayscale", not "expand a seed". Replaced with distance-ranked
trimming (keep nearest ~0.69-of-area pixels to centroid, trim far extensions/barbs = the real
V17 failure mode). Verified: 0% empty, median seed/mask ratio 0.692 (matches measured V17
under-seg 0.686). This made the experiment valid.

## Result (fold-0 weights on the 94-file tune split; tune is OOF for all folds)
NORTH STAR (reservoir = 132 tune V17 candidates at full-res IoU 0.3-0.5):
| metric                         | S1 (no spine) | S2 (spine 0.2) |
|--------------------------------|---------------|----------------|
| reservoir pushed to IoU>=0.5   | 17 / 132      | 17 / 132       |
| median IoU gain on reservoir   | -0.026        | -0.029         |
| median area ratio before->after| 0.65 -> 0.86  | 0.65 -> 0.84   |
| good V17 (>=0.5) DAMAGED        | 172 / 607     | 168 / 607      |
| best selective-replacement PQ  | 0.338 (OFF)   | 0.338 (OFF)    |

baseline V17 PQ (same harness, 512px) = 0.3382 (tp815/fp1344/fn319). Bar = 0.4487.

## Verdict: NEGATIVE. Best replacement rule is "off" (= baseline). Lever does not clear the bar.
- The refiner DOES expand under-seg masks the right direction (area 0.65->0.86 toward 1.0) and
  pushes 17/132 reservoir instances over 0.5 — a real but small positive.
- BUT median reservoir IoU gain is NEGATIVE (-0.026): it hurts more than it helps across the set.
- It DAMAGES 172/607 (28%) of already-good >=0.5 masks (over-expands correct masks).
- EVERY blind area-growth replacement rule loses ~0.10 PQ (tp 815 -> ~620). No rule nets +.
- S1 ~= S2: spine aux made no meaningful difference (17 vs 17 pushed). The hoped-for
  "S2 >> S1 on barb recovery" did NOT appear.

## Root cause
TRAIN/INFERENCE GAP. Trained on (under-seg seed -> full GT), the refiner learned to EXPAND the
seed toward a plausible full filament. At inference it expands V17's masks INDISCRIMINATELY,
including the 607 already-good ones, where expansion = damage. The output carries no reliable
"I was under-segmented" signal correlated with GT improvement, so selective replacement can't
separate the 17 wins from the 172 harms.

## Cheap root-cause fix option (NOT yet run — one train+eval cycle)
Train with a MIXED seed distribution: ~50% under-segmented seeds (as now) + ~50% IDENTITY seeds
(seed = full correct mask, target = same mask). This teaches "if the seed is already good, leave
it alone", directly attacking the 172-damage failure mode. It is a targeted fix of the measured
cause, not a random tweak. If after this the refiner preserves >=0.5 masks (damage << 50) while
still pushing a meaningful share of the reservoir, selective replacement could net positive.
If it still can't separate win-from-harm, reservoir 1 is closed too and V60 consolidates.

## Standing
Both V60 reservoirs now measured: reservoir 2 (tile-only, 149 GT) CLOSED (3 converging negatives,
oracle not FP-safely harvestable); reservoir 1 (189 under-seg) NEGATIVE as built, one cheap
root-cause fix (identity-seed mix) remains before closing. V17 0.39 public remains best. Firewall
intact, no submission, rogii not deallocated. The honest North-Star metric did its job: training
loss ~0.08 looked great but the reservoir recovery is net-negative once good-mask damage is counted.


---

# v2: identity-seed mix + correction-needed head (2026-10-05) — KILL, reservoir 1 CLOSED

## What changed vs v1 (S1/S2)
- Seed mixture: 50% identity (seed = GT, label 0), 15% mild (~0.90 area), 30% regime (~0.69,
  the measured V17 under-seg), 5% severe (~0.40). Correction label = 1 iff IoU(seed, GT) < 0.8.
- New head: correction-needed logit from pooled bottleneck features (mean+max), BCE weight 0.5.
- No spine (S1 only, per the gate). Everything else identical. Two training seeds (mixA, mixB),
  fold 0, 25 epochs. Replacement gate uses ONLY model outputs (corr prob, refined-mask mean
  prob), never GT and never area growth.
- Verified before training on real data: 0% empty seeds, identity ~0.48, label-1 ~0.37,
  seed/GT ratio p25/p50/p75 = 0.69/0.91/1.0, per-epoch seed resampling actually varies
  (a shared default_rng(0) forked into workers would have replayed one stream every epoch;
  replaced with a per-sample RNG seeded from torch.initial_seed()).

## Result (tune, 94 files, 132 reservoir / 607 good; populations and baseline identical to v1)
| | v1 S1 | mixA | mixB |
|---|---|---|---|
| AUC(corr: reservoir vs good) | n/a | 0.512 | 0.505 |
| mean corr reservoir / good | n/a | 0.204 / 0.187 | 0.243 / 0.233 |
| replace-all: reservoir pushed to >=0.5 | 17 | 6 | 4 |
| replace-all: good masks damaged | 172 | 107 | 106 |
| replace-all PQ | 0.238-0.23 | 0.2628 | 0.2617 |
| best gated PQ | 0.3382 (off) | 0.3386 (corr0.9: replaces 0 masks) | 0.3383 (3 replaced) |
Baseline V17 PQ (same harness) 0.3382, bar 0.4487.

## Verdict: kill_head_cannot_separate
- The correction head cannot tell a reservoir candidate from a good one on real V17 masks
  (AUC ~0.51 on two independent seeds). Mean corr is ~0.2 for BOTH populations.
- The gate therefore has nothing to select on: at corr>=0.5 it replaces ~10 of 132 reservoir
  and ~30 of 607 good masks (same ratio as chance), pushes 0-2, damages 8-9.
- The identity mix did not add selectivity; it made the refiner more conservative across the
  board. Replace-all damage fell 172 -> ~107 but recovery fell 17 -> 4-6, so pushed/damaged got
  WORSE (0.10 -> 0.04-0.06). Fewer harms came with fewer wins.
- The head only reached corr_acc ~0.70-0.72 on its own synthetic training distribution
  (majority-class rate ~0.62), so it barely learned even the synthetic task. Whatever it
  learned did not transfer to real V17 masks, whose errors are not centroid-distance trims.

## What this does and does not establish
- Established: the two-seed, fold-0, tune-only test shows no selectivity signal from this
  head, and the pixel head did not become selective either.
- Not established: that NO correction signal exists. The head is a small global-pool MLP on a
  256px downscaled crop trained on synthetic degradations. A head trained on REAL V17 failures
  (OOF V17 candidates labelled by measured IoU) is a different experiment that was not run.
  Only fold 0 was trained for v2; the n is small (132 reservoir).

## Decision
Per the agreed gate, reservoir 1 (under-segmented V17 near-misses) is CLOSED for post-hoc
refinement. Reservoir 2 (tile-only) was closed earlier (3 negatives). Nothing cheap remains on
the V17 candidate set. No submission; V17 (public 0.39) stays the best clean result.
