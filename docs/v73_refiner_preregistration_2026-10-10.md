# V73 pre-registration: 5-fold OOF refiner + three one-change refiner upgrades (2026-10-10)

Written before any V73 GPU run. Nothing is submitted, test images and labels are never read, no
SWEFil. Code: `scripts/train_v73_refiner.py` (copied from `train_v63_refiner.py`, which is
unchanged), launcher `scripts/_v73_launch.sh`, check `tests/test_v73_refiner.py`.

## Why
- The refiner is the biggest component of the pipeline (+0.04 to +0.06 PQ) and absorbs about 80%
  of detector gains (`v62_refiner_test`). Its architecture has never been iterated: it is still
  the public-notebook ResNet18 U-Net on resized 256 px crops (`v69_history_audit` 2c, 1j).
- Every refined number so far uses the 141 canonical held-out files (SE about 0.005). A true +0.008
  passes our gate about 35% of the time (`v69_history_audit` §3). The audit's process fix (i) is a
  5-fold refiner, which scores the real pipeline on all 707 files.

## How the existing numbers are built (verified in code)
- **141 refined-eval files** = `optimize_inference.validation_split`: the 707 files shuffled with
  `random.Random(42)`, first 20% (94 tune + 47 audit), checked against
  `outputs/v6_inference_source/v6_provenance.json`. They are **not** V61 fold 0. They are spread
  over all five V61 folds (fold sizes 141/141/142/142/141). Each file is scored with proposals from
  the fold model that did not train on it (`outputs/v61_oof_cache/p2first/fold{K}`), then
  `cluster_views(flip_consensus, 0.5)` -> refiner crop probabilities (4-flip TTA) ->
  `clean_refined_predictions(refined_constrained)` -> per-record PQ counts at t0.45/c0.30.
- **r3 training proposals** = the other 566 canonical FIT files, each with its own V61-fold OOF
  p2first proposals. Targets come from `v11.match_positive_proposals` (IoU >= 0.30, margin rules,
  one positive per GT per annotator record): 6,115 records. Crops are 256 px, centred on the
  proposal, with jitter and rotation/flip/gamma augmentation. Training is 12 epochs, AdamW 1e-4,
  cosine, 0.5 BCE + 0.5 Dice, batch 16, fp32.
- **r3's initialisation** is `outputs/kaggle_coco_refiner/best_refiner_coco.pt`. Its lineage is
  V6, an ImageNet ResNet18 trained 8 epochs on GT crops of the **566 FIT files**, followed by the
  COCO-mask fine-tune on the same 566. So r3 is clean on the 141 files and **in-sample on the
  other 566**. A naive 5-fold r3 (canonical init) would leak on 566 of the 707 files.
- "7 px seed restriction" = `cv2.dilate(proposal, 7x7 ellipse)`, a radius of 3 px.

## Harness (per V61 fold K = 0..4)
Train on the fold != K files and their own-fold OOF p2first proposals. Refine and score the fold-K
files with fold-K proposals. Pool the per-file counts over K to get 707 files, all clean: the
refiner never sees fold-K labels, and the detector proposals are OOF. Leak-free lineage:
- stage A (V6-like): ImageNet encoder, GT-centred crops of the fit files, 8 epochs, lr 3e-4,
  cosine, last epoch (no validation selection);
- stage B (r3): OOF proposal crops, 12 epochs, lr 1e-4, the same as r3.
Scorer parity: with `outputs/v63/r3_refiner.pt` on CPU, the V73 scorer reproduces `v63/r3.json`
per-file counts **exactly** on 20/20 canonical files (`train_v73_refiner.py parity`).

## Arms (each ONE change vs `base`)
| row | change | rationale |
|---|---|---|
| `base_k7` | none: ImageNet ResNet18, 256 crops, 7x7 clamp | leak-free r3 baseline |
| `crop512_k7` | 512 px crops (train + inference) | large filaments are downsampled into 256; 23% of refined FN are mask near-misses (`v69_error_analysis` §7.3) |
| `rn34_k7` | ImageNet ResNet34 encoder (same widths, same decoder) | stronger encoder with the clamp kept; never isolated (V14 confounded it) |
| `base_k15` (`dil15`) | base's own models, 15x15 clamp at inference | lets the refiner recover recall on thin/under-segmented seeds; zero extra training |
| `r3c_k7` | control: canonical init, stage B only (= r3 exactly) | reproduction check and size of the leak; **never a primary baseline** |

timm and segmentation_models_pytorch are not installed. The encoder arm uses torchvision only.

## Gate (written now, applied once, `train_v73_refiner.py compare` -> `outputs/v73/compare.json`)
PRIMARY, for each of `crop512_k7`, `rn34_k7` and `base_k15`: pooled 707-file refined PQ at
t0.45/c0.30, arm minus `base_k7`, paired file bootstrap (4,000 resamples, seed 73).
- **CONTINUE** iff dPQ >= +0.010 AND P(dPQ>0) >= 0.90 AND 95% CI low > 0 AND >= 4/5 folds positive.
- **KILL** iff dPQ < +0.005.
- Otherwise **INCONCLUSIVE**: it may enter a stack-then-test bundle (audit §3 ii) but is not
  shipped alone.
No cell re-selection: the fixed t0.45/c0.30 cell only. Three primaries, no multiplicity correction
(documented here). A CONTINUE authorises an all-707 export re-fit of that arm. It does not
authorise a submission.

Reported, never gating:
- `r3c - base` on the 141 canonical files: does the leak-free lineage match r3's quality?
  If base is below r3c by more than 0.010 there, the arms are judged against a weaker baseline.
  This is stated in the result and the arm deltas are not reinterpreted.
- `r3c - base` on the other 566 files (LEAKY): the size of the canonical-init leak.
- 5-fold `r3c` vs `v63/r3.json` on the 141 files: reproduction of r3's 0.4595. Expect |d| < 0.005
  (the training set differs by about 20%; r1 seed noise was about 0.001).
- Per-arm grids at every (t, c) and both kernels are in the fold JSONs.

## Cost (T4, fp32; scaled from r3 = 52 s/epoch at 6.1k records, CPU smoke ratios)
| arm | per fold | 5 folds |
|---|---|---|
| base (also yields dil15) | ~22 min | ~1.9 GPU-h |
| rn34 | ~30 min | ~2.5 GPU-h |
| crop512 | ~75 min (CPU ratio 3.3x/epoch) | ~6.3 GPU-h |
| r3c | ~14 min | ~1.2 GPU-h |
| **total** | | **~12 GPU-h, ~3.2 h wall on 4 T4** |
512 px memory is estimated at about 4-6 GiB (batch 16, fp32). It was not measured on GPU because
the GPUs were busy. `peak_mib` is logged per epoch.

## Launch (after V68 frees the GPUs, about 20:45 IST)
`cd ~/solar-filament-2026 && nohup bash scripts/_v73_launch.sh > outputs/v73/pipeline.log 2>&1 &`
(`mkdir -p outputs/v73` first.) The launcher refuses to start while any GPU holds >1 GB. It is
resumable, and progress goes to `outputs/v73/STATUS`. The CPU smoke run of every code path is
`SMOKE=1 bash scripts/_v73_launch.sh`, which passed on 2026-10-10 (20 jobs + compare, into
`outputs/v73_smoke`).

## Caveats
- Stage A is a simplification of the lineage: no COCO fine-tune and no V11 continuation stage,
  and the last epoch is used instead of a Dice-selected one. The `r3c` control measures the cost.
- Second-order stacking leak: fold-j proposals used for training come from a detector that saw
  fold K. This is standard nested-CV practice and is not removed.
- One seed per arm. Refiner seed noise was about 0.001 on 141 files.
