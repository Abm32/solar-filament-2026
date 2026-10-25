# V62 pre-registration (written 2026-10-05 ~21:30 IST, BEFORE any V62 OOF number exists)

Status at writing: V62 architectures built and smoke-tested (1 tiny epoch + predict); no V62 model
has been trained on a fold or scored on held-out data. Reference = Phase-A baseline, fold 0:
A (seed 42) PQ 0.3763 @ conf 0.35, A' (seed 20261005) PQ 0.3813, i.e. seed noise |A'-A| = 0.0049.

## Question
Does adding a P2/stride-4 head level to the V17 detector (YOLO11s-seg) make fine structure that the
P3/stride-8 detector misses REACHABLE, without recreating the tile detector's proposal cloud?

## Treatments (both appended to the stock yolo11s-seg graph; layers 0-22 and weights unchanged)
- `p2last`  : Segment inputs [P3,P4,P5,P2]. Prototypes still from P3 (stride 4 masks, 256x256 @1024).
              Adds detection/coefficients at stride 4. Isolates "detect at P2".
- `p2first` : Segment inputs [P2,P3,P4,P5]. Prototypes from P2 (stride 2 masks, 512x512 @1024).
              Adds the above PLUS higher-resolution mask prototypes.
Everything else is the V17 recipe exactly (80 ep mask_ratio=2, 20 ep mask_ratio=1, same seed 42,
same data/augmentation/schedule). Init: COCO yolo11s-seg weights REMAPPED so all tensors that exist
in both graphs start from COCO values (scripts/v62_init.py; the stock loader would silently drop
the whole head). Honest init difference, not removable: `p2first` cannot transfer the
classification branch (its first input has 64 channels vs COCO's 128) - 509/639 tensors transfer at
nc=1 vs 555/639 for p2last and 555/561 for the baseline; the P2 level and (p2first) proto.cv1 are
new/random. This is part of the treatment and is disclosed with the result.

## Fold and controls
Fold 0 only, paired against A and A' on the SAME 141 held-out files, same scorer/TTA/consensus
(eval_v61_compare.py + a new oracle/cloud analysis). Compute: p2last and p2first in parallel, one
GPU each. No 5-fold before the structural gate below passes.

## Structural metrics (reported first, before PQ)
From the cached pre-consensus 4-view pool at the detector floor conf 0.1 and from the final
post-consensus prediction set:
 (S1) oracle recall @IoU>0.5 = GT matched by ANY proposal, by category (thin/small/faint/normal,
      categories as in diagnose_v59_0_recall), from the consensus pool at conf floor 0.1;
 (S2) proposals/image and FP/image (unmatched proposals) at the baseline's operating cut 0.35
      and at each recipe's own best cut;
 (S3) small+thin oracle recall gain over A, and over A'.

## Gate (fixed now; no edits after seeing results)
STRUCTURAL PASS requires ALL of:
 1. pooled small+thin oracle recall gain over A >= +0.030 (absolute) AND >= 2x the A'-A gain on
    the same metric (seed noise on the structural metric is measured, not assumed);
 2. proposals/image at conf 0.35 <= 1.30 x A's (no cloud): the gain must not be bought with
    volume; equivalently gain per added proposal is reported;
 3. FP/image at conf 0.35 <= 1.30 x A's.
Only a structural pass earns a PQ read. PQ PASS (-> full 5-fold OOF) requires, in addition:
 4. direct PQ gain d >= +0.010 at conf 0.35 (the baseline's own best cut) AND d >= 2 x 0.0049
    (= 0.0098) AND bootstrap P(d>0) >= 0.90;
 5. the result is not explained by a threshold shift (d at the recipe's own best cut and at 0.35
    both > 0).
Decision tree: structural fail -> kill that variant; structural pass but PQ fail -> report, no
5-fold (a reachable-but-unselectable gain is the tile failure mode); both pass -> 5-fold OOF of
that variant with a second baseline seed for noise.
If both variants pass, prefer the one that passes with the smaller model/compute delta; do not
pick on PQ among them beyond the gate.

## Not allowed after seeing results
No head-weight or neck-width tuning, no second architecture variant, no confidence-grid widening,
no re-reading the gate. Magnetogram work stays behind V62.

## Caveats stated in advance
- One fold, one seed per treatment: a kill is reliable, a pass is a direction only.
- p2first changes two things (P2 detection AND stride-2 prototypes) plus a weaker init; a p2first
  win does not isolate which part helped. p2last vs p2first is the only attribution we get.
- Mask IoU may change via prototype resolution independent of detection; mean matched IoU is
  reported alongside PQ.
