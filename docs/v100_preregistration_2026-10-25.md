# V100 pre-registration: image-based selector, union-label detector, aggregate probe (2026-10-25)

Written before any V100 number. No test labels, MAGFiLO release or leaked CSV are read. Goal set by the user:
public > 0.39. Measured headroom (V83/V85/V86): oracle selection on our own candidate pool 0.528 (+0.07 over V84),
811 kept masks match no annotator, 964 TP-quality masks are dropped; the deployed selector is a 28-feature LightGBM
(AUC 0.745, V99). Recall gains survive the refiner, width gains do not (V62 size hybrid, V99 soft1280).

## Arm A `selector` (image-based candidate scorer; GPUs 1–2)
Data: `scripts/v100_extract.py` = the V84 soft extract (asserted to reproduce `extract_soft.pkl` per-file counts) plus a
5-channel 128 px crop per candidate (gray, refiner probability, candidate mask after ownership, union of other alive
candidates, detector seed), all 707 files, 6344 candidates at t0.50 / score ≥ 0.15. Target y and weight w exactly as
`v84_bundle.arrays` (fraction of the file's records the candidate matches at IoU > 0.5, weight = #records).
Model: torchvision ResNet18 (ImageNet init, conv1 widened to 5 channels by copying the mean RGB filter), 1 logit,
BCE with soft target y, weight w, AdamW 3e-4 wd 1e-4, cosine, 12 epochs, batch 64, dihedral augmentation on the
crop stack; the 28 tabular features are concatenated to the pooled embedding through a 2-layer MLP (tabular+image).
Cross-fit: outer V61 folds (model k trained on folds ≠ k, predicts fold k) and the inner fold-k-free fits that
`v84_bundle.run_arm` needs for the Dinkelbach λ; seed 100. Two seeds averaged if time allows (reported).
Cells (all 707 files, V84 path: cut 0.30 + MIN200 + ŷ > λ/2, λ the inner Dinkelbach fixed point):
 `cnn` (image+tabular model as ŷ), `mix` (mean of `cnn` ŷ and the V84 LightGBM cross-fitted ŷ), `v84` (control,
 must reproduce 0.458975). Secondary: `promote` (eligible = all candidates ≥ 0.15, keep iff ŷ > λ/2), AUC of each ŷ.
Primary: `mix` vs `v84`, paired per-file bootstrap, 2000 draws, seed 100. CONTINUE d ≥ +0.008 and P ≥ 0.95
(707 files, so the bar is 4× the pooled seed noise); INCONCLUSIVE +0.003 ≤ d; KILL otherwise. CONTINUE → all-data
fit (train on all 6344, 2 seeds), export with ŷ_mix replacing the demote rule, λ from the pooled cross-fit, submit.

## Arm B `union` (GPU 3, `scripts/_v100_union.sh`)
V81's `p2first_union` recipe (one union-of-annotators instance set per file, V79 clustering), fold 0, 80 + 20 epochs,
unchanged schedule, never launched before. Judge: OOF views cached as recipe `p2first_union`, deployed V84 soft fold-0
refiner, `v99_screen` cells, vs `v84_soft` and fresh seeds. Primary `t050_c030`: CONTINUE d ≥ +0.008 & P ≥ 0.90;
INCONCLUSIVE ≥ +0.003; KILL below. Mechanism expected if positive: more TP at the same cut (recall), not SQ.

## Arm C `probe_half` (one public submission, from the V84 CSV; `scripts/v100_probe_half.py`)
Keep the V84 rows of the 90 test images with the most kept masks (ties by total mask area), drop every row of the
other 90. Reads the private kernel's aggregate: pooled counts vs per-image mean. OOF prediction is computed before
the submission from `extract_soft.pkl` per-file counts with the same rule (densest half by kept count), both as
pooled PQ with the blanked half counted as FN and as per-image mean with blanked images at 0, scaled by
0.39 / 0.4590. The reading is whichever prediction the public number is within 0.02 of; if both or neither,
inconclusive. The number also bounds any missing-image penalty (a read far below both).

## Cost
Extract ~35 min CPU (22 procs). Selector: 5 outer + 20 inner fits × ~4 min on a T4 ≈ 100 min across two GPUs.
Union: ~4 h GPU 3 + 10 min infer + 6 min extract. Probe: minutes, one submission.

---
Arm C prediction, written before submission (2026-10-25, `outputs/v100/probe_half_prediction.json`): OOF densest half
holds 70.0 % of GT and 73.2 % of kept masks. OOF: pooled with the other half all-FN 0.3800 → **public 0.32 if the
kernel pools counts**; per-image mean with blanked images at 0: 0.2361 → **public 0.20 if the kernel averages
per image**. File `outputs/v100/probe_half.csv`, 819 rows, 90 images, sha256 `4aeb5715…`. Reading rule as above;
a read below 0.17 would mean a missing-image penalty beyond FN.

---
Result arm A `selector` (2026-10-25, seed 100, 707 files, `outputs/v100/gate.json`): **KILL.** Outer-fold AUC of the
image+tabular model 0.826–0.861 (all candidates), 0.780 at the cut vs 0.753 for the V84 LightGBM, so the crops do
carry ranking information. PQ through the V84 path: v84 0.4590 (reproduced), cnn 0.4584 (−0.0006 [−.003, +.002],
P .31), mix 0.4595 (+0.0005 [−.001, +.002], P .75). Secondary promote cells (eligible = all candidates ≥ 0.15):
cnn 0.4595, mix 0.4606 (+0.0016). A +0.03 AUC gain in agreement ranking buys ≈ +0.001 PQ: the candidates that
decide PQ sit at matched IoU 0.5–0.6 where keep/drop is near-neutral, and the oracle headroom (0.528) is
annotator-level noise that no image evidence resolves. Selection is closed with image evidence as well.

Result arm C `probe_half` (2026-10-25 10:44Z, Kaggle submission of `probe_half.csv`): **public 0.31.** Pre-stated
pooled read 0.32, per-image-mean read 0.20. The private kernel pools TP/FP/FN over images like our scorer and the
public notebook; images with no rows cost only their FN (no extra penalty). The λ target aggregate is the right one.

## Arm D `split` (CPU, `scripts/v100_split.py`; added 2026-10-25 after arms A/C, before any number)
V85: 305 FN + 21 FP come from one kept mask covering two GT filaments of the same record (+0.019 exact if every
such mask were split correctly). Nothing in the pipeline splits a kept mask. Rule: for each kept candidate i
(t0.50, cut 0.30, MIN200), find the best-scoring other proposal j (any view group, score ≥ 0.10, not itself kept)
with containment |j ∧ i| / |j| ≥ 0.70 and share |j ∧ i| / |i| in [0.15, 0.60]; split i into (i ∧ j) and (i \ j),
keep both if each ≥ 200 px (i \ j reduced to its largest component if it has > 2 components). Split iff
score_j ≥ τ. τ is chosen per fold on the other four folds (τ ∈ {0.10, 0.15, 0.20, 0.25, 0.30}); the honest
number is the cross-selected PQ on all 707 files vs the same cell unsplit. CONTINUE d ≥ +0.008 & P ≥ 0.95;
INCONCLUSIVE ≥ +0.003; KILL below. Reported: number of splits, TP/FP/FN change, best in-sample τ.

Result arm D `split` (2026-10-25, 707 files, `outputs/v100/split.json`): **KILL.** 4686 kept masks, 312 had an inner
proposal meeting the containment/share rule. Every τ loses in-sample (τ 0.10: 312 splits, 0.4385; 0.20: 169, 0.4474;
0.30: 74, 0.4539 vs unsplit 0.4567); cross-selected τ = 0.30 on all folds, d = −0.0028 [−.005, −.001], P(d>0) 0.002.
At τ 0.30 the 74 splits gain 4 TP and add 111 FP: the inner proposal's boundary is not where the second
annotator's filament ends, and most swallowed pairs have no inner proposal at all. Splitting along proposal
geometry is dead; a learned split would need a signal the candidate pool does not contain.

Result arm B `union` (2026-10-25 13:49Z, fold 0, 141 files, `outputs/v100/screen_union.json`, checkpoint sha
`99c5484c…`): **KILL.** Detector trained on union-of-annotators labels (p2first recipe, 80+20 epochs, GPU 0 with
chunked loss) reaches val mask mAP50 0.933 / mAP50-95 0.407 vs 0.937 / 0.455 for the per-record p2first fold 0
(looser boxes, as expected from union masks). Through the frozen soft refiner at the primary cell t050_c030:
0.4375 vs 0.4555 (d = −0.0180 [−.028, −.008], P(d>0) 0.001); TP +75, FP +286, FN −75. Every cut in the sweep
loses by more (c015 0.4035, c020 0.4164, c025 0.4258). With the fold-0 demote stage (`full_v84` cell): 0.4549 vs
0.4590 (−0.0041). Fresh-seed controls s43/s44/s45: −0.0021 / +0.0001 / −0.0002, so the union loss is 9× the seed
spread. Union labels teach the detector to propose what any one annotator drew; under per-record pooled
PQ each extra proposal is a FP for every record that did not draw it, and the extra recall (+75 TP) does not pay
for it. The demote stage recovers most but not all of the damage. Arm closed.

## V100 verdict
All four arms resolved under the pre-registered rules: A selector KILL (+0.0005), B union KILL (−0.0180),
C probe public 0.31 (pooled kernel confirmed), D split KILL (−0.0028). Together with V99 (soft1280, d4, hardneg
KILL) every stage of the V84 family — detector labels, detector TTA, refiner training set, candidate selection
with image evidence, mask splitting — has now been tested against the deployed model with fresh-seed controls and
none moves fold-0 or 707-file PQ by more than the seed spread. The deployed V84 (public 0.39) stays selected.
