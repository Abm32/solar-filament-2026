# V69 history audit: V1-V68 reviewed by a skeptical outside reader (2026-10-10)

Read-only audit. No compute, no files changed except this one. Sources: `docs/experiments.md`
(E001-E053), `SESSION_CONTEXT_2026-10-06.md`, all V42-V68 result and preregistration docs,
`annotation_sparsity_finding`, `root_cause_0.39_plateau`, `kaggle_discussion_scan`, git log, and
these scripts: `train_v21_full_detector.py`, `fetch_v67_magnetograms.py`, `train_v60d_onetoone.py`,
`diagnose_v60d_empty_tiles.py`, `train_v63_refiner.py`, `kaggle_inference/v14_dualhead.py`.
When I say something is "not documented", I mean no doc in the local mirror records it. A doc may
exist only on `rogii`.

## 0. Bottom line

1. The pre-registration regime is not the main reason for the plateau. If every "inconclusive
   positive" were stacked, the refined pipeline would gain about +0.004 to +0.010 PQ (section 3).
   That is real, but it is not the 0.39 -> 0.46 gap.
2. The main losses are **wrong-comparator closures in the V44-V56 era**, **resource kills that a
   later infrastructure fix removed**, and **axes that were never tested at all**: semi-supervised
   data scale, train-time flips, a better refiner, and size-aware thresholds.
3. The refined pipeline is judged on 141 files. Its 95% CI is about +/-0.010 and the gate requires
   a point estimate of at least +0.010. A true +0.008 improvement passes about 35% of the time.
   Every decision now depends on refined deltas of about 0.005, so this measuring setup cannot make
   them.

## 1. Lines closed on weak or confounded evidence

| # | Line (where closed) | What the closure rested on | Specific flaw | Reopen? |
|---|---|---|---|---|
| 1a | **Native-2048 / single YOLO (V44, V45, V49)** | V44 yolo11s-seg@2048 scored 0.351 "vs V17 0.4487", so it was declared "-0.09 below V17" | **Wrong comparator.** In the same harness, V49 scored the V21 V17-recipe detector at **0.351** too, and V21 was trained on all 707 files, so those tune files were *in-sample* for it. The harness has no refiner, no 4-flip consensus and single-record GT. Phase A later measured the same recipe OOF in the canonical harness at 0.3856. So V44 **tied** V17's own detector and never went through refiner and consensus. V46 and V49 also attribute V17's score to a "six-source ensemble". That is false: V17 is one detector (`arm_r1` seed 1) plus a refiner. Six sources appear only in the E028 *analysis*. | **YES, strong** |
| 1b | **V29 YOLOv8l @ native 2048** (E041) | 20 epochs: refined 0.4464 vs 0.4487 | The gap of -0.0024 is well inside the measured +/-0.007 training noise. The planned 40-epoch continuation **never ran**: Ultralytics restored the 20-epoch contract, a tooling bug. The line was closed as "not reproduced", not as "measured worse". | YES (merge with 1a) |
| 1c | **V18 1536 + mask_ratio=1** (E028c) | OOM in `single_mask_loss` BCE at batch 1 | V65 later added a chunked, checkpointed upstream BCE mask loss that keeps the same value and gradient, with a peak of 9.2 GiB. That fix removes exactly this failure. **The fix came after the kill and the line was never re-run.** | YES |
| 1d | **V23 radial flattening + CLAHE** (E035) | Closed at the resource gate: peaks of 14,355-14,775 MiB vs a self-imposed limit of <14,336 | It was killed by **19-439 MiB** of self-imposed headroom on a 15,360 MiB T4, and was never trained. The V48 re-test ran in the broken 1a harness and gave 0.357 vs 0.351 (**+0.006**), which was reported as "0 lift". EdgeAttNet, the MAGFiLO SOTA, uses exactly this preprocessing. The chunked loss now makes memory a non-issue. | YES, medium |
| 1e | **Mask R-CNN (V56)** | Best 0.383 "vs V17 0.4487" | Same comparator flaw as 1a: single-record GT, no refiner or consensus, compared with a refined, cell-selected number on 94 files. 0.383 is in the range of V17's own OOF direct detector (0.3856). It is a plausible orthogonal ensemble member that was never put through the canonical harness. The "resolution scaling law" retraction is correct. | Low-medium |
| 1f | **Voting ensembles (V66)** | Primary E_all4 = -0.0026, so kill. "E_all4 is the UPPER BOUND on any equal-weight subset" | **That premise is false.** Adding weaker members lowers PQ, and E_mix shows exactly that (-0.0041). E_p2x2 = **+0.0063, CI [+0.0025, +0.0101], 5/5 folds**, and E_base2 = +0.0040, 5/5. Stage 1b (seed ensemble of p2first) was skipped on a *forecast* of 80% refiner absorption. That forecast came from p2first, whose direct gain was mostly SQ, and the refiner equalises SQ (`v62_size_hybrid`). A voting ensemble mostly changes **ranking/RQ**, and the hybrid test showed RQ survives refinement. | **YES** |
| 1g | **Copy-paste (V64)** | Failed the *structural* gate (small+thin oracle recall +0.004 < +0.030) | The **PQ outcome** was +0.0064 direct on one fold (P 0.88). The project killed a PQ-positive lever on a mechanism proxy. A 5-fold OOF later existed for E_p2x2, and the r3cp stack result is **not documented**. | YES (stack) |
| 1h | **Focal + Dice (V61-B)** | B-A +0.0047 vs seed noise n=0.0049 (fold-0, one pair) | The pooled 5-fold noise later measured **0.0015**, about 3x smaller. It was tested on the baseline, not on p2first, with one weight setting. Caveat: at the baseline cut B-A was only +0.0016 and thin recall fell, so the evidence against it is moderate. | Weak, stack-only |
| 1i | **Tile detector (V60, V60B/C)** | Add-only fusion -0.062, consensus -0.031 | The baseline tile trainer turned **28% of tiles (the empty ones) into full-tile "filament" masks**, and 22.5% of its detections were whole-tile blobs with 0 TP. V60D's `train_v60d_onetoone.py` fixed this (`legacy_empty=False`). **The fusion and consensus tests were not documented as re-run with the fixed tile model.** | Medium |
| 1j | **V14 dual-head refiner / IoU head** (E025) | PQ 0.245 | Three changes were confounded in one arm: ResNet34 encoder, **proposal clamp removed**, and an IoU head. The IoU head was trained on positives only (min IoU 0.30), so it emitted a near-constant ~0.62. Its diagnostic was misaligned (pearson -0.006). The doc correctly found both bugs and then closed the line on a noise-model projection, not a fixed run. The ResNet34 mask head *with* the clamp was never isolated. | YES (see 2c) |
| 1k | **Train resolution 1536 (E022)** | -0.0006 tune, -0.019 audit | It was a 20-epoch warm start from a 1024 checkpoint, used mask_ratio=2, one seed, and 94 files with +/-0.007 noise. That is not a test of "training at high resolution with high-resolution prototypes". | Merge with 1a |
| 1l | **Longer training (E007, V5)** | 30 -> 80 epochs, -0.012 public | This was one run on an early pipeline with patience and best-checkpoint selection. V68 `p2first_long` is the first clean test. | V68 in flight |
| 1m | **Consensus targets (V24) and ignore-region (V25)** | -0.0026 and -0.0054 vs control, 2 seeds | These were reasonably done (two seeds, matched controls). The ">=2-annotator" target and the true sparse-supervision mask (`annotation_sparsity` doc, "better remaining idea") were never run. | Low |

Closures that look sound to me: Mask2Former and CondInst (large gaps), DenseCRF, SWEFil verifier on
V17, graph/ridge/orientation opportunity gates (V38-V40), V60A mask refiner, and V62 size-hybrid
(a clean falsification).

## 2. Systematic blind spots (never tried, or only tried in a broken form)

a. **Semi-supervised data scale: the root-cause doc's own #1 lever, never executed.** The host
   allowed unlabeled GONG H-alpha for training (discussion 737215/744047/743385). V63 says "GONG
   archive times out from the Azure VM, so archive pseudo-labelling is blocked". But
   **`fetch_v67_magnetograms.py` pulled 680/707 magnetograms from `gong2.nso.edu/oQR` on the
   local machine** (`outputs/v67_fetch.log`, local manifest path). The archive is reachable: fetch
   it locally and scp it to the VM. Plan: noisy-student with the p2first+r3 teacher, plus
   thousands of archive frames (exclude a +/-1 day window around the 180 test timestamps to stay
   clearly firewall-clean), then student retrain and 5-fold OOF. **Pseudo-labelling the 180 test
   images** was never considered and was explicitly forbidden in V27. The rule text covers
   leaking test GT and extra observations *at inference*. Transductive self-training on the
   supplied test images is commonly allowed, but because judging is by committee, **ask the host
   before doing it.** Using test images with pseudo-labels is the higher-risk option; the archive
   is the safe one.

b. **Train/inference augmentation mismatch.** The detector recipe trains with
   `fliplr=0.0, flipud=0.0` and only `degrees=10` (`train_v21_full_detector.py` L143/L161). Every
   inference then runs **4-flip consensus**, so 3 of the 4 views are out of distribution. This
   setting was inherited from a public notebook (`research.md` L71). The only flip test (E011, V8)
   was confounded with yolo11m, 60 epochs, a different refiner and different scoring targets.
   V68 `p2first_flip` is the first clean test. If it is positive, add 90/180/270-degree rotations,
   which the full disk supports.

c. **The refiner is the biggest lever (+0.059 to +0.06 refined vs direct) and has had almost no
   architecture work.** It is still the public-notebook ResNet18 U-Net on 256px crops that are
   **resized**: large filaments are downsampled and thin ones up-sampled. Only its weights and
   training proposals changed (V11, V63 r1/r3). The refiner "absorbing" p2first's gain is a sign
   that it, not the detector, sets the SQ ceiling (~0.688). Untested: a bigger encoder with the
   clamp kept, native-resolution tiled crops for long filaments, a context margin, the proposal
   mask as an input channel *with* the clamp (V14 removed the clamp), and a properly trained IoU or
   score head using negatives (V28 showed OOF AUC +0.012 with negatives included).

d. **High-resolution prototypes at high input resolution.** The one axis that worked is mask
   resolution (stride-2 prototypes, +0.020 direct, 5/5 folds). The natural next step is
   p2first at 1536 or native 2048, or native 1024 tiles plus a full-frame pass, with the chunked
   loss. V68 `p2first_1280` is a timid step in this direction. See 1a-1c.

e. **Size-aware post-processing, zero GPU.** No document mentions per-size confidence or
   mask thresholds. Small-object recall is 0.46 vs 0.665 for normal objects, and confidence is
   strongly size-correlated. With **707-file OOF direct caches** already on disk, cross-fitted
   per-area-bin confidence cuts (and mask threshold `t`) cost CPU minutes. On the refined pipeline
   they can only be fitted on 141 files until 3a is done.

f. **Optimizer-side regularisation.** Ultralytics EMA is on by default, but there is no SWA or
   checkpoint averaging over the final N epochs, and the final fixed checkpoint is used. The voting
   ensemble showed variance is worth +0.004 to +0.006 direct. Weight averaging gets part of that
   for free at deploy time.

g. **Target choice.** Training repeats each image once per annotator record, with conflicting
   labels. Under per-record pooled PQ this is a defensible soft-target choice. Not tried: a single
   image instance with per-instance weights equal to the annotator-agreement fraction, or
   ">=2-annotator" positives combined with ignore regions for singletons.

h. **Local-to-public validation.** The scorer matches the organiser notebook, which is good. Two
   problems remain. The board was re-scored after a metric change (0.66 -> 0.31 for one file), so
   confirm that `src/pq_eval.py` matches the *current* metric code and that 0.39 is a post-rescore
   score. And since the final judgement is made by a committee on method, a clean OOF number on
   707 files plus an ablation table is worth more than chasing a two-decimal public score.

i. **SAM-family refinement** was dismissed by citation (arXiv 2412.04243, about *finding*
   low-contrast tree-like objects), not by test. As a *box-prompted refiner* of existing proposals
   it is a different question. Lower priority than 2c.

## 3. Process: did the +0.010 / P>=0.90 gate throw away compounding gains?

**Statistical power.** For the refined 141-file test, SE is about 0.0052. P(pass | true delta) =
P(d_hat >= 0.010):
- delta = +0.005: about 17%
- delta = +0.008: about 35%
- delta = +0.010: about 50%

On the 707-file direct OOF, SE is about 0.0027, so delta = +0.008 passes about 23% of the time,
because the +0.010 point floor binds and not P>=0.90. **The floor rejects most true effects
smaller than 0.010 on any population size.** That was deliberate after E024 (local gains of
+0.004 did not move public), but it means improvements in the +0.003 to +0.007 range can never be
stacked.

**Inconclusive or below-bar positives that were not shipped** (the shipped ones were E-A
`iou=0.3`, V11 continuation, mask_ratio=1, p2first and r3, all inside V17/V65):

| lever | measured delta | level / population | status |
|---|---|---|---|
| copy-paste 0.3 (V64) | +0.0064 (P 0.88) | direct, fold 0 | killed on structural proxy |
| p2first 2-member voting (V66 E_p2x2) | +0.0063, CI [+0.0025, +0.0101], 5/5 | direct, 707 | "inconclusive", line closed |
| seed voting baseline (E_base2) | +0.0040, CI [+0.0006, +0.0074], 5/5 | direct, 707 | kill (<0.005) |
| Focal+Dice (V61-B) | +0.0047 own cut / +0.0016 base cut | direct, fold 0, baseline arch | kill |
| r3 refiner refit over p2first+V11 | +0.0029 (0.4595 vs 0.4566) | refined, 141 | shipped in V65 |
| V52 clDice fusion (V53) | +0.0006 | refined, 94 | closed |
| V28 utility scorer over V26 | +0.0003 | refined, 94 | closed |

Estimate: the unshipped direct-level levers (copy-paste, seed voting, Focal+Dice on p2first) sum
to about +0.012 to +0.017 direct if they were additive. They are not fully additive, and copy-paste
and the ensemble partly overlap. After the refiner, assume 40-70% survives for RQ-type gains and
about 20% for SQ-type gains. That gives **about +0.004 to +0.010 refined PQ left on the table.**
This is worth harvesting, but it is not the plateau explanation.

**Recommended process changes**
- (i) Build a **5-fold refiner**: for each fold, train the r3 recipe on the other folds' OOF
  proposals. This makes the *refined* pipeline evaluable on all 707 files and halves the SE.
- (ii) Replace the +0.010 per-lever floor with a **stack-then-test** rule. Pre-register a bundle
  of levers that are individually positive with P>=0.80, and gate the bundle once at P>=0.95 on
  707 refined files.
- (iii) Gate on PQ, never on mechanism proxies (V64).
- (iv) Do not skip a stage on a forecast (V66 stage 1b).
- (v) When an infrastructure fix lands (chunked loss), re-open every line that was killed by the
  failure it fixes (V18, V23).

## 4. Infrastructure bugs and inconsistencies between docs

1. **V44-V56 harness vs canonical harness.** It scored single-record GT (`records_for()`, the max
   instance record) and the 141 tune+audit files, with no consensus and no refiner. It was
   compared against V17's per-record, refined, cell-selected 0.4487 on 94 files. The
   `annotation_sparsity` doc says this "matches V17's convention", but the canonical convention is
   all records (150 records on 94 files). An in-sample V21 model scored 0.351 in this harness. Any
   V44-V56 absolute number is not comparable to anything after Phase A.
2. **"V17 = six-source ensemble"** (V45/V46/V49) is a factual error that drove the conclusion that
   "single models cannot match V17".
3. **The audit split was supposed to be sealed** (E019-E042), but the V44-V56 work evaluated on
   all 141 files, including the 47 audit files. This is harmless for leakage into training, but
   the 94-vs-141 populations get mixed in comparisons.
4. **GONG archive is "unreachable" (V63) vs a successful GONG magnetogram fetch (V67)** from the
   same host (`gong2.nso.edu`).
5. **No flips in training vs 4-flip TTA at inference** (2b).
6. **The refiner was trained on proposals from a different detector.** V11 was trained on V6
   proposals and served on `arm_r1` (V17) and then p2first proposals until V63. The V65 r3 refiner
   was trained on OOF fold-model proposals and serves the all-data `fold 99` detector. The second
   case is acceptable, but nobody measured whether fold-99 proposals are distributed like the OOF
   proposals: confidence calibration shifts with 25% more training data. V21 (all-data refit)
   emitted **124 fewer masks** than V17 at the same cut and scored 0.38 vs 0.39, which is exactly
   this calibration-shift symptom.
7. **The cell t0.45/c0.30** was selected in September on 94 files for the V17 detector and V11
   refiner. It is still used for p2first+r3 (V65). p2first's own best on the 141 files was
   t0.55/c0.35, but that was selected in-sample.
8. **The V60 tile trainer's empty-tile bug** was fixed in V60D after the fusion verdicts. The
   re-fusion is not documented.
9. **Missing result docs in the local mirror:** V63 (only commit-message numbers), V64, V65 r3cp
   stack result, V67 magnetogram diagnostic verdict. V68 is in flight.
10. Minor: README.md's top section is stale (it still says "V41 running, 2026-09-14"), and
    `RESUME_TOMORROW.md` stops at 2026-09-15.

## 5. Ranked recommendations

1. **High resolution plus stride-2 prototypes, re-opened properly.** Train p2first at 1536 or
   native 2048, or on native 1024 tiles plus a full frame, with the chunked loss. Run 5-fold OOF
   through the canonical harness, then the refiner. Evidence: 1a-1c and 1k; mask resolution is
   the only axis that has ever worked.
2. **Refiner v2** (2c, 1j). The refiner is worth +0.06, has had essentially no architecture
   iteration, and is where detector SQ gains disappear.
3. **Semi-supervised with the GONG archive** (2a). The host allows it, it is reachable, and it is
   the root-cause doc's #1 lever, never run. Ask the host about test-image pseudo-labels
   separately.
4. **Fix the instrument and stack** (section 3): build the 5-fold refiner, then one bundle of
   p2first + copy-paste + 2-seed voting or SWA + r3, gated once on 707 refined files.
5. **Augmentation and input** (2b, 1d): train-time flips and rotations (V68 arm), and radial
   flattening + CLAHE now that memory is solved. Add the zero-GPU size-aware thresholds (2e) as a
   free side check.
