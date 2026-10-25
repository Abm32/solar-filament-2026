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
