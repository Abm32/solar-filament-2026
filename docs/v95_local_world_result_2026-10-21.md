# V95 counterfactual (VM-only, no Kaggle): how readable are the remaining object-class probes? (2026-10-21)

The five object-class probes were **not** submitted (user instruction: no Kaggle submissions). Instead the
"H_local" world was scored on the VM: remove the same object class from the **train** GT (the world in which
test labels lack that class) and score V84 OOF and the probe against it with the host scorer
(`scripts/v95_local_world.py`, `outputs/v95/local_world_*.json`). Test labels, the MAGFiLO release and the
leaked CSV were not touched. `topk3/5` have no GT analogue and were not scored.

| probe | GT dropped | Δ_uniform = OOF_probe − OOF_v84 | Δ_local = PQ(probe\|GT\X) − PQ(V84\|GT\X) | separation ×0.85 | readable at 2 dec? |
|---|---|---|---|---|---|
| minarea500  | 1263/8199 | −0.0206 | +0.0165 | 0.031 | yes |
| minarea1000 | 3388/8199 | −0.0993 | +0.0805 | 0.153 | yes, decisive |
| maxrr080    |  592/8199 | −0.0104 | +0.0120 | 0.019 | borderline (≈ rule threshold) |

Predicted public reads, V84 = 0.39: under H_uniform 0.373 / 0.306 / 0.381; under "test lacks X"
≈ 0.39 + 0.85·Δ_local = 0.40 / 0.46 / 0.40.

Side observation: V84 scored against GT without masks < 1000 px drops 0.459 → 0.382 (−0.077), the same size
as the observed OOF → public gap (−0.069). GT without masks < 500 px drops only −0.011. So "test labels lack
small (<1000 px) filaments" is one H_local world that reproduces the gap by itself; the minarea1000 probe is
the single read that decides it (0.31 uniform vs 0.46 local). Nothing changes until a public read exists.

## 2026-10-22 addendum: full (GT min-area G) × (prediction min-area P) grid

`scripts/v95_minarea_grid.py` (one host overlap pass on V84 OOF vs train GT, then slicing; `outputs/v95/minarea_grid.json`,
`.log`). Cells (G=1000, P=0) = 0.3823 and (G=1000, P=1000) = 0.4628 reproduce `local_world_minarea1000.json` exactly.
Rows: world "test labels lack filaments < G px". Columns: submission filtered at min-area P. Base OOF 0.4590.

| G | GT kept | P=0 | 300 | 500 | 750 | 1000 | 1250 | 1500 | 2000 | gap at P=0 | best P | PQ at best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 8199 | .459 | .458 | .438 | .401 | .360 | .322 | .287 | .227 | 0 | 0 | .459 |
| 250 | 8022 | .461 | .462 | .444 | .406 | .365 | .327 | .292 | .231 | +.002 | 300 | .462 |
| 500 | 6936 | .448 | .450 | .464 | .442 | .401 | .361 | .324 | .259 | −.011 | 500 | .464 |
| 750 | 5796 | .416 | .419 | .444 | .464 | .441 | .405 | .366 | .296 | −.043 | 750 | .464 |
| 1000 | 4811 | .382 | .385 | .411 | .449 | .463 | .446 | .411 | .339 | −.077 | 1000 | .463 |
| 1250 | 4031 | .350 | .352 | .377 | .418 | .451 | .463 | .448 | .381 | −.110 | 1250 | .463 |
| 1500 | 3435 | .319 | .322 | .346 | .385 | .424 | .452 | .461 | .419 | −.140 | 1500 | .461 |
| 2000 | 2546 | .268 | .270 | .291 | .327 | .366 | .405 | .436 | .456 | −.191 | 2000 | .456 |

Reads:
1. **Gap match.** The observed OOF→public gap (−0.069) sits between G=750 (−0.043) and G=1000 (−0.077): a world where
   test labels lack filaments below ≈ 900 px reproduces it. G=500 does not (−0.011).
2. **Diagonal is flat.** For every G the best P equals G and PQ(G, G) ≈ 0.456–0.464. V84's quality is size-invariant;
   the whole gap in any such world is the size mismatch, not detection quality, and it is fully recoverable by one
   submission-side knob (min-area = G).
3. **Off-diagonal cost is steep both ways.** Under H_uniform (G=0), P=1000 costs −0.099; under G=1000, P=0 costs −0.077.
   No hedge P exists (P=500: −0.021 uniform / +0.029 at G=1000). The decision needs the public read.
4. Predicted public reads (public ≈ 0.85 × PQ): minarea1000 probe = **0.31 if H_uniform, 0.39–0.40 if G≈900, 0.46 if G=1000**
   (vs V84 0.39). Any read ≥ 0.40 on minarea1000 rejects H_uniform at 2 decimals. minarea750 would read 0.34 / 0.39 / 0.38
   — less separable; minarea1000 stays the probe to send first.

What this cannot do: distinguish "lacks small filaments" from any other test-label difference of the same PQ size
(annotator subset, expert-review pruning, limb exclusion). Only the Kaggle read does that. No submission made (paused).

## 2026-10-22 addendum 2: real annotators as worlds (`scripts/v95_annotator_world.py`, `v95_annotator_style.py`)

The host scores per annotator-image, so every train annotator is a real label-style world. V84 OOF against each
annotator's images only (`outputs/v95/annotator_world.json`, `annotator_style.json`; overlap table cached in
`outputs/v95/overlap_v84.pt`):

| annotator | images | GT/img | share GT < 900 px | V84 PQ | best pred min-area | recall | precision | other-annotator PQ on same images |
|---|---|---|---|---|---|---|---|---|
| 020403 | 42 | 6.0 | 0.25 | **0.560** | 300 (+0.002) | 0.82 | 0.76 | 0.441 |
| 050101 | 47 | 6.7 | 0.36 | 0.526 | 0 | 0.73 | 0.78 | 0.397 |
| 010403 | 77 | 5.6 | 0.26 | 0.449 | 500 (+0.004) | 0.69 | 0.63 | 0.314 |
| 010402 | 75 | 8.3 | 0.35 | 0.440 | 0 | 0.55 | 0.75 | 0.279 |
| 030101 | 48 | 10.5 | 0.43 | 0.415 | 0 | 0.46 | 0.82 | — |
| 010401 | 73 | 6.8 | 0.39 | **0.399** | 0 | 0.60 | 0.64 | 0.285 |
| 010101 | 72 | 8.4 | 0.37 | **0.399** | 0 | 0.50 | 0.72 | 0.305 |

By annotation group (first two code digits): 01 = 0.435 (516 annotator-images, 45 % of train), 02 = 0.516, 03 = 0.458,
04 = 0.454, 05 = 0.485, 06 = 0.469. Round-1 batches are hardest (0101 0.426, 0301 0.415, 0401 0.436).

Reads:
1. **The public number already exists inside train.** Annotators 010101 and 010401 (145 images, 13 % of train
   annotator-images) score exactly 0.399 with V84. A test set annotated in that style gives 0.39 with no size story.
2. **The small-filament world is not a real annotator style.** No annotator with ≥ 10 images has a small-filament share
   below 0.17, and prediction min-area never helps any real annotator by more than +0.004. The synthetic G ≈ 900 world
   from addendum 1 reproduces the gap arithmetically but matches nobody's labels. Prior on the minarea1000 probe drops;
   under the annotator-mix reading it would print ≈ 0.29–0.31 (P=1000 on 010101 = 0.294).
3. **Hard annotators are recall-limited by zero-overlap misses.** 010101: 303 FN, 204 with IoU 0 against every
   prediction (median area ≈ 840 px, 54 % < 900 px, 19 % ≥ 2000 px); FP are mostly partial overlaps (IoU 0.3–0.5),
   i.e. extent/splitting disagreements. Lower conf, wider masks and merging were all read on the public board (V92/V95)
   and lose, matching OOF — the misses are filaments the detector does not see at all (the V39–V41 finding).
4. **Human ceiling is below V84.** Other annotators, scored as predictions against each annotator's labels on the 296
   multi-annotator images, reach PQ 0.28–0.44; V84 beats every human on every annotator. A 0.39 public score is a
   model scoring one hard annotator-style better than the annotators score each other.

Consequence for "improve the score": no submission-side knob and no annotator-specific re-tuning exists (best
min-area is 0 for the hard annotators, conf/width/merge already read on public). Movement needs recall on
zero-overlap misses of dense annotators, which V39/V40/V41 found unsupported by image signal. OOF gains spread
across easy annotators do not show on a 2-decimal board scored in the hard style.
