# V60 reframing — segmentation-first decoupled proposal pipeline (2026-10-04)

Supersedes the "train a V17-type detector on tiles" framing. Three independent signals now
converge on the SAME sharper hypothesis:
1. Our V59-0 proposal-oracle diagnostic: V17 never proposes ~27% of GT at IoU>0.5; the
   bottleneck is upstream representation, not score/topology/label-noise.
2. Competition host explicitly says to focus on FINE-SCALE structure, especially BARBS.
3. A competitor ("Spaghetto99") reports getting past ~0.44 with a DECOUPLED pipeline, and
   another reports **0.460 strictly OOF** over the whole train set (a far more credible
   target than the random-split 0.487 — split choice alone swings PQ ~0.01-0.03).

## Sharpened hypothesis
V17 is NOT mainly failing on confidence threshold, post-proc, or instance topology. It
fails because its DETECTOR REPRESENTATION discards fine filament structure (barbs, thin
extensions). A high-resolution SEGMENTATION-FIRST proposal model may recover that
structure; instance formation should then be handled as a SEPARATE stage. Our previous
ordering (detector -> mask refine) may be backwards for huge/irregular/thin/branched
filaments. Better: fine semantic segmentation -> structure extraction -> instance grouping.

## The TWO reservoirs (from our own oracle numbers — the key quantitative clue)
V17 proposal-oracle recall: IoU>0.5 = 0.728, >0.4 = 0.824, >0.3 = 0.892.
- ~116 GT: score-suppressed but already matchable >0.5 (recovering floods 7x FP — hard).
- **~109 GT: roughly localized at 0.3-0.5 but masks too POOR to cross 0.5** <- a MASK-QUALITY
  / fine-structure problem (barbs eroded by downscaling). A segmentation-first high-res
  model attacks THIS directly. Different, more tractable target than recognition.
- ~310 GT: not matchable at 0.5 at all (hardest; genuine never-seen).
The middle ~109 is the realistic near-term reservoir: V17 already localizes them, so it's
not a recognition/FP-flood problem — just getting the mask over 0.5.

## Target recalibration
Stop chasing 0.55+ public (leakage-polluted; host takes 0.41 seriously). The credible
clean target is ~0.46-0.47 local OOF PQ (competitor's 0.460 OOF). Our 0.4487 needs ~+0.01
to +0.02, NOT +0.1. That is a realistic, bounded goal.

## V60 experiment (segmentation-first), measured by ORACLE RECALL first (not PQ)
Fine-structure semantic model:
- native-resolution overlapping crops (768/1024)
- 1-CHANNEL grayscale input (host confirms RGB-replicate is allowed but ~3x compute for no
  real info; use native 1-ch unless ImageNet weights are needed)
- loss = Dice + lambda*Focal with explicit positive-class weighting (do NOT crank globally
  -> that recreates the measured low-threshold FP flood; keep it high-RECALL-proposal, judged
  by oracle recall)
- optional: spine auxiliary head (forces features to preserve thin central structure/barbs),
  deep supervision
THE TEST (before building any instance stage): of the 310 V17-never-proposed GT, how many
does the semantic model activate (connected-component overlap / fraction of GT pixels above
prob threshold)? And how many 0.3-0.5 near-misses become >0.5-plausible masks?
GATE: recover ~60-100 previously-inaccessible GT -> build the decoupled instance stage.
Near-none -> stop, consolidate (V17 is the clean ceiling for this family).

## GPU plan (4x T4) for the segmentation-first sweep
- G0-1: high-res tiled semantic (768/1024, 1-ch, Dice+Focal, pos-weighted, native)
- G2: same + spine auxiliary head (preserve barbs/thin center)
- G3: alternative strong multi-scale backbone
Compare on structure/proposal recall ONLY first.

## Status of the CURRENTLY-RUNNING V60A (instance tile detector)
The in-flight run (v60_tiling.py, maskrcnn on 768 tiles, 4 OOF folds, ep4/10, loss ~0.29)
is the WEAKER instance-first version. KEEP it running — its oracle-recall test still
answers "does native-res proposal generation find new objects" (a valid, cheaper data
point). After it reports, build this stronger segmentation-first version if the instance
tiler is promising OR if we want the cleaner test regardless.

Firewall unchanged: FIT-only OOF, 1-ch native crops from competition images, tune labels
for scoring only, no test, no submission. rogii not deallocated. Target: clean +0.01-0.02
local OOF PQ over V17's 0.4487, via recovering the ~109 mask-quality reservoir + some of
the 310.

Sources: competition discussion (host barbs/fine-scale focus; Spaghetto99 decoupled >0.44;
0.460 OOF; split swings 0.01-0.03; host takes 0.41 seriously; RGB-replicate allowed-but-3x);
our V59-0 oracle diagnostic.

---

## FINALISED V60A/B DESIGN (user adjustments, 2026-10-04)

Mask-quality diagnostic confirmed the target: 189 GT in IoU 0.3-0.5, median pred/GT area
0.686, 58.7% undersized (vs matched >=0.5: median 1.023, 18% undersized). 135/189 are
NORMAL-sized -> substantial filaments losing extremities/barbs, not just tiny ones.

### SPLIT the project (risk-ordered):
**V60A = candidate-conditioned MASK REFINER (low risk, build first).**
V17 finds the candidate -> crop a high-res window around it -> segmentation-first model
produces a REPLACEMENT/EXPANDED mask for THAT candidate. This is FP-safe: we refine
existing V17 proposals, we do NOT generate new standalone instances (correction: standalone
connected-components WOULD create FP; refining does not). Attacks the 189 under-segmented
masks surgically.
- Tile/crop: **1024** main (768 as comparison) — more structural context to preserve barbs.
- Overlap (for the discovery stage): 25-30%.
- Input: **1-channel native grayscale** (no RGB triplication; T4 memory).
- Loss: **1.0*Dice(mask) + 0.5*Focal(mask) + 0.15-0.30*Dice/BCE(spine)**. Bias to recall
  only mildly; do NOT heavy positive-weight (that recreates the measured FP flood).
- **Spine aux head from the start but MODEST** — purpose is feature preservation of thin
  central structure/barbs, NOT topology (topology already shown not the bottleneck).
- **SELECTIVE replacement:** do NOT replace every V17 mask. Produce an alternative, replace
  only when it improves (e.g. area-expansion + confidence), so the already-good >=0.5 masks
  (median 1.023) are not degraded.
- **Two clean controlled variants only:** V60-S1 (no spine) vs V60-S2 (+spine), identical
  otherwise -> clean causal read on whether spine preserves lost area. Use other GPUs for
  fold parallelism, not more architecture variables.
- **SELECT BY RESERVOIR METRICS, never pixel Dice:**
  * # of the 189 0.3-0.5 near-misses pushed to IoU>=0.5  <- NORTH STAR
  * median IoU gain on the 189
  * median pred/GT area after refinement
  * new FP components created (must stay low)
  * (and overall tune PQ vs V17 0.4487)

**V60B = proposal DISCOVERY (later, higher risk).** Whole-image tiled semantic activations
-> find GT V17 never proposed (the 120 <0.3 + 310 never-at-0.5). Only after V60A works.

### Firewall / target unchanged
FIT-only OOF (refiner trained on fold-held-out V17-style candidates vs GT), 1-ch native
crops, tune labels scoring only, no test/submission. Target clean +0.01-0.02 over 0.4487.

---

## V60A result (2026-10-04): oracle-recall BREAKTHROUGH

4 OOF tile detectors (native-res 768 crops, 30% overlap, Mask R-CNN, 10ep), evaluated on
the 94-file tune split; union proposal-oracle recall vs the V17 pool (512px scoring):

| IoU   | V17 pool | tile pool | union  |
|-------|----------|-----------|--------|
| >0.5  | 0.728    | 0.818     | 0.8598 |
| >0.4  | 0.824    | 0.895     | 0.922  |
| >0.3  | 0.892    | 0.939     | 0.957  |

- union@0.5 = 0.8598 clears the >0.82 "very strong" gate decisively (+0.132 over V17).
- tile-alone (0.818) already beats V17 (0.728).
- 149 tile-ONLY GT found (tiles hit @0.5, V17 misses entirely): mean area ~115px, mean
  elong ~1.9 = the small/compact filaments the downscaled full-disk detector never proposes.

This is the FIRST representation win of the whole V57-V60 arc. It confirms the sharpened
hypothesis: V17's ceiling is upstream representation, and native-res tiling recovers the
missing reservoir. CAVEAT: this is ORACLE recall (GT is somewhere in a 120-200-proposal/img
pool). The gate's purpose was to justify a fusion build — which 0.86 does.

## V60B result (2026-10-04): add-only fusion = DECISIVE NEGATIVE (not_selectable)

fuse_v60b.py — V17 trusted base (never modified) + add-only tile additions in weak/absent
regions; verifier applied ONLY to the tile stream; sweep conf{0.15,0.25,0.35,0.5} ×
ver{0,0.5,0.7} × overlap_max{0.1,0.25}; V17-identical per-annotator PQ at 512px.

- baseline V17-only PQ 0.3402 (tp815/fp1324/fn319).
- BEST policy (cf0.5/vt0.7/om0.1): PQ 0.2779 (tp816/fp2061/fn318) = +1 TP, +737 FP.
- EVERY policy LOST PQ (delta -0.062 to -0.105). Verdict: not_selectable (< +0.003 gate).

### The funnel — where the oracle gain leaks (distinct tile-only GT covered / props surviving)

| stage       | proposals | tile-only GT covered |
|-------------|-----------|----------------------|
| raw         | 16798     | 149                  |
| confidence  | 12541     | 129                  |
| verifier    | 8330      | 108                  |
| geometry    | 8272      | 108                  |
| cross-NMS   | 472       | **1**                |

ROOT CAUSE: the 149 tile-only GT are NOT isolated. Each true tile-only filament sits in a
cloud of overlapping near-duplicate tile proposals. Confidence-ranked greedy NMS picks a
spurious higher-confidence neighbor, which suppresses the correctly-localized proposal and
is itself admitted as an FP -> ~7 FP per recovered TP (the same FP-explosion branch V59-0
found for aggressive thresholding). The right mask EXISTS in the pool (oracle 0.86 is real)
but is not SELECTABLE by confidence+verifier+geometry+NMS. The failure axis is
LOCALIZATION/RANKING (which overlapping proposal is correctly placed), which is orthogonal
to the verifier's real-vs-junk axis — so the verifier only trims at the margin (129->108)
and cannot rescue selection.

### Implications
To harvest the oracle gain would require a different instrument:
(a) a per-cluster localization-quality ranker (pick the single best-placed proposal per
    tile cluster, not max-confidence);
(b) mask-fusion/consensus over each overlapping cluster;
(c) training the tile detector to be far less duplicative (in-detector NMS, higher conf).
All are larger builds with uncertain payoff vs V17's clean 0.39.

NOTE the V60A mask REFINER (189 under-segmented 0.3-0.5 near-misses) is INDEPENDENT of this
selection problem: it refines EXISTING V17 candidates and adds no new proposals, so it is
NOT blocked by this negative. That is the remaining primary lever.

---

## V60C-0 result (2026-10-04): consensus RANKING refuted; consensus MASK has partial signal

diagnose_v60c_consensus.py over 77 tile-only clusters (single-link IoU>=0.10 grouping on the
tile-only proposal pool; "correct proposal" = argmax IoU-to-GT in the cluster):

RANK of the correct proposal within its cluster (1 = would be picked):
| signal     | median rank | frac top-2 |
|------------|-------------|------------|
| confidence | 5.0         | 19.5%      |
| consensus  | 6.0         | 20.8%      |
| verifier   | 5.0         | 22.1%      |

=> Consensus does NOT rank the correct proposal better than confidence (6 vs 5, both ~rank-5
of the cluster, ~20% top-2). The predicted "confidence buries it, consensus surfaces it"
pattern DID NOT appear. A single-proposal localization ranker (handcrafted C1 or learned C2)
is CONTRAINDICATED — none of confidence/consensus/verifier contains the signal to pick the
correctly-localized proposal out of its cluster.

CONSENSUS MASK (pixel-occupancy >= tau over the whole cluster, no selection):
- best single proposal IoU-to-GT median = 0.636 (the right mask IS in the pool — oracle holds)
- consensus-mask IoU-to-GT median: tau0.2 0.478, tau0.3 0.468, tau0.4 0.457, tau0.5 0.440
- fraction of clusters reaching IoU>=0.5: tau0.2 0.416, tau0.3 0.416, tau0.4 0.377, tau0.5 0.338

=> Occupancy-consensus (MERGE the cluster instead of SELECT one) recovers ~42% of tile-only
GT at IoU>=0.5 (tau~0.25) WITHOUT needing to identify the single correct proposal — it
sidesteps the unsolvable ranking problem. BUT median consensus IoU (0.47) is BELOW the best
single proposal (0.636): fusing dilutes the good mask with noisy neighbors. It wins only
because it needs no selection.

VERDICT: c0_mask_recovers_build_c0 — but with the explicit caveat that recovery-count is NOT
PQ. The real gate is a scoped C0 PQ test: one consensus mask per cluster (tau 0.25), added to
the V17 base only in weak/absent regions, scored V17-identical tune PQ vs 0.4487. One mask per
cluster (not the 472 raw NMS survivors) MIGHT control FP where V60B add-only could not — must
be measured. If C0 >= +0.003 tune: tune it. If it floods FP like V60B: tile reservoir closed,
move to the V60A refiner (reservoir 1, the 189 under-segmented V17 near-misses).

---

## V60C-0 PQ test (2026-10-04): consensus MASKING = KILL. Tile reservoir CLOSED.

fuse_v60c0.py — V17 base + one occupancy-consensus mask per tile cluster (tau=0.25 exact,
single-link IoU>=0.10), add-only in weak/absent V17 regions; V17-identical per-annotator PQ
512px; sweep conf{0,0.15,0.25,0.35,0.5} x overlap_max{0.1,0.25}.

- baseline V17-only PQ 0.3402 (tp815/fp1324/fn319).
- BEST C0 (cf0.5/om0.1): PQ 0.3089 (tp817/fp1663/fn317). delta -0.0314; -0.1398 vs 0.4487 bar.
- THE RATIO (user's key metric): dTP +2..+3 for dFP +339..+490 -> dTP/dFP = 0.006-0.008.
- consensus-only TP = 2-3 (of 149 reachable); consensus instances added = 240-396.

C0 did NOT flip V60B's ~1-TP-per-hundreds-of-FP ratio (0.006-0.008, unchanged). Collapsing
each cloud into ONE object reduced the proposal COUNT (240 consensus masks vs thousands of raw
tiles) but those 240 masks are themselves almost all FP: only 2-3 land on a real tile-only GT.

WHY the diagnostic's 42% did not translate: the diagnostic measured consensus IoU ONLY on the
77 clusters KNOWN to correspond to a tile-only GT. At inference that knowledge is unavailable
-- the tile detector forms a consensus-worthy cluster around EVERY region it fires on, almost
all spurious. So C0 emits ~240-400 consensus masks to capture ~3 real ones. The 42% was
conditional on knowing the GT location; unconditionally, consensus masking is just another FP
generator. This is the oracle->PQ gap one layer deeper.

THREE converging negatives on converting the 0.86 tile oracle to PQ:
  1. V60B add-only selection:   -0.062  (cannot RANK the correct proposal)
  2. V60C-0 consensus ranking:  refuted (correct proposal ~rank 5-6 under every signal)
  3. V60C-0 consensus masking:  -0.031  (dTP/dFP ~0.007, unchanged; one FP generator -> another)

DECISION (user gate PQ<=0.4487 -> kill): TILE-CONSENSUS BRANCH CLOSED. No more NMS/verifier/
ranker/consensus tinkering on the tile pool. The representation win (masks exist, oracle 0.86)
is real but NOT FP-safely harvestable without a fundamentally LESS-DUPLICATIVE detector
(in-model NMS / higher conf / one-proposal-per-object training) -- a large build, deferred.

REMAINING LEVER: V60A high-res mask REFINER on reservoir 1 (the 189 under-segmented V17
candidates, IoU 0.3-0.5). Independent of tile selection -- it refines EXISTING V17 masks, adds
no proposals, so none of the above negatives apply. This is now the single remaining cheap-ish
lever. Then: PARKED magnetogram training-only aux (host-blessed) if detection recall is the
binding constraint and we commit to a new detector.
