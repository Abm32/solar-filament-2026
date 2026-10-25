# V97 pre-registration: annotator-style crop refiner (2026-10-24)

Written before any V97 training. No test images, test labels, MAGFiLO release or leaked CSV are read. No Kaggle
submission is part of this experiment (submissions are paused by user instruction).

## 1. Where every earlier line leads (the chain this experiment sits on)

| line | what was shown | doc |
|---|---|---|
| Export / metric / sampling | test export == OOF path byte-for-byte; host scorer reproduced; 90-image public sampling cannot give 0.39 from OOF 0.459 (P = 0 in 20k draws) | v87, v69 metric audit, v85 §3 |
| Public probes | public reacts to every knob like OOF: ±1 px width 0.36/0.37, conf 0.20 no-demote 0.38, conf 0.40 0.39, fold-0 models 0.39. The −0.06 offset is label content, not an operating point | v92 |
| Detector changes | every direct gain is absorbed by the refiner: imgsz 1280, thin labels, noisy student, limb division, ensembles, spine-net source (all refined d in ±0.007) | v77, v81, v76, v94, v66/v86, v93 |
| Selection | agreement fraction is not predictable from 28 candidate features (re-scorer −0.010, promote +0.0001, pooled selector +0.0003, rule-based ownership −0.002..−0.004) | v70, v83, v86, v88 |
| Refiner recipe | soft (annotator-mean) targets +0.002, crop512 / RN34 / dil15 / SSL init / union all ≤ +0.001 | v73, v82, v84, v89 |
| Who the public score is | V84 scores 0.399 on train annotators 010101/010401 and 0.435 on group 01 (45 % of train); 0.52 on group 02; the public 0.39 is a hard-annotator style, not a size story (no real annotator matches the min-area world) | v95 addendum 2 |
| Style via the detector | fine-tuning p2first on hard / group-01 records lowers PQ on every subset, including its own (hard −0.014, g01 −0.001, control −0.003) | v96 |

What the hard annotators actually lose (V84 OOF, `outputs/v95/annotator_style.json`):

| annotator | PQ | FN | FN iou 0 | FN iou .3–.5 | FP | FP iou .3–.5 |
|---|---|---|---|---|---|---|
| 010101 | 0.399 | 303 | 204 | 60 | 116 | 60 |
| 010401 | 0.399 | 201 | 99 | 74 | 166 | 72 |
| 020403 (easiest) | 0.560 | 46 | 27 | 17 | 66 | 15 |

Two loss classes. (a) Zero-overlap FN: filaments no kept prediction touches; V39–V41 found no image signal for them, and
lowering the cut loses on the public board. (b) The IoU 0.3–0.5 band: the same filament drawn with a different
extent, counted once as FN and once as FP. For 010401 this band is 37 % of FN and 43 % of FP; flipping half of
these pairs to TP is worth ≈ +0.03 on that annotator. Extent is drawn by the crop refiner (V69 §6, V70: the
refiner sets width and length; detector shape changes do not survive it). The refiner is trained on the
annotator-mean target (V84 soft), i.e. the average style. No experiment has yet asked the refiner to draw a
specific annotator group's extent. That is the one untested place where style can enter the pipeline.

**Question.** Does a crop refiner trained toward group-01 extent raise refined PQ on group-01 records
(held out, fold 0) relative to the V84 soft refiner, and what does it cost on the other groups?

## 2. Arms (fold 0, V84 soft recipe: stage A 8 ep 3e-4 on GT crops, stage B 12 ep 1e-4 on p2first fold-0 OOF
proposals, seed 42, ImageNet ResNet18 CropUNet, 256 px crops; `scripts/v97_style_refiner.py`)

- `soft` (control): the existing V84 soft fold-0 refiner `outputs/v84/soft_fold0_refiner.pt` (sha256 `5b1362f1…`),
  re-extracted with the V97 extractor (must reproduce `outputs/v84/extract_soft.pkl` fold-0 counts exactly).
- `soft_g01`: stage A unchanged; stage B records restricted to group-01 annotator records (428 of 935 fit records),
  target = mean of the GT masks that the image's group-01 records matched to the proposal (0 for group-01 records
  with no match). The V84 soft target with the mean taken over group 01 only.
- `cond`: all records; CropUNet with a 7-token embedding (0 = pooled, 1–6 = annotation groups 01–06) applied as
  zero-initialised FiLM at the bottleneck (512 ch) and after the 1/4-resolution decoder block (64 ch). Per
  training sample (record r, proposal p): with probability 0.5 token = group(r), target = GT of r; otherwise token 0,
  target = V84 soft target. Stage A the same with GT targets. At token 0 the model is asked to be the V84 soft
  refiner; at token k it is asked to draw group k's extent. Evaluated at token 1 (`cond@01`), token 0 (`cond@00`,
  sanity), token 2 (`cond@02`, the easy style, reported), and `cond@oracle` (each record scored against the
  kept set produced with its own group's token; an upper bound that assumes the style is known).

## 3. Judge

`v97_style_refiner.py extract` = the V84 extractor (`v84_extract.work`: fold-0 p2first views → flip consensus →
refiner k7 → t0.50 → candidates + V83 features + per-record IoU) with per-record annotator tags and near-miss
geometry added. `gate` = `v89_screen.cmd_gate` extended to annotator subsets: hard (010101 + 010401; 26 fold-0
records), g01 (group 01; 88), rest (131), all (219). Cells: `t050_c030` (cut 0.30 + MIN200, no demote) and
`full_v84` (plus the V84 demote rule fit on folds 1–4, λ = V84's fold-0 λ; features shift with the refiner, so
this cell is secondary). Deltas vs `soft`, per-file paired bootstrap (2000 draws, seed 97). Control must reproduce
the V84 bundle fold-0 counts exactly (assert, as in V89).

**Primary:** `t050_c030` on the g01 subset, for `soft_g01` and `cond@01`. Per arm: CONTINUE if Δ ≥ +0.020 and
P(Δ > 0) ≥ 0.90; KILL if Δ < +0.005; otherwise INCONCLUSIVE.

**Reported, not gated:** hard, rest and all subsets (the price on other styles); `cond@00` vs `soft` on all
(sanity: |Δ| should be < 0.010, else the conditioning harmed the pooled model); `cond@oracle` on all (ceiling if
the test style were known); `cond@02` on rest; per-subset counts of FN/FP in the IoU 0.3–0.5 band and the mean
containment of near-miss pairs (intersection / pred area, intersection / GT area), before and after, which say
whether the arm moved extent in the intended direction; SQ per subset.

## 4. Pre-stated reading

- CONTINUE on `cond@01` (or `soft_g01`) with `rest` ≤ −0.01: the refiner can draw a group's extent. The public
  gain then depends on the test labels being group-01-like, which only a public submission can read; that
  decision is the user's and is outside this experiment. A CONTINUE authorises 5-fold + all-data fits of the
  arm (no submission).
- CONTINUE with `rest` ≥ 0: the arm is a plain improvement; same follow-up.
- KILL on both: extent style is not learnable by this refiner from these labels. With V96 this closes
  "annotator style" on both the detector and the refiner; the hard-annotator loss is then the zero-overlap recall
  that V39–V41 found unsupported by image signal, and 0.39 is the ceiling of this pipeline family on the public
  set absent a probe.
- `cond@oracle` ≫ `cond@01` while `cond@01` KILLs: style is learnable but group 01 is not internally consistent
  enough to help blind; per-annotator tokens would be the next (and last) refinement, not pre-registered here.

## 5. Cost

Two refiner trainings (~22 min each on a T4, GPUs 1 and 2 are free; GPU 0 holds another project's job), six CPU
extracts (~6 min each, 22 processes), one gate. Nothing beyond fold 0 is trained.

---
Result (2026-10-24 19:08Z): KILL on soft_g01 and cond@01; cond@oracle = control. See
`docs/v97_style_refiner_result_2026-10-24.md`.
