# V97 result: annotator-style crop refiner (2026-10-24)

**Status: KILL on both arms; the oracle ceiling is zero.** A refiner trained toward group-01 extent (`soft_g01`),
and a FiLM-conditioned refiner asked for group 01 (`cond@01`), both lower refined PQ on the group-01 records they
were trained for. Scoring every record with its *own* group's style (`cond@oracle`) is indistinguishable from
the pooled V84 soft refiner (all subset −0.0002). Extent style is not something this refiner can learn from
these labels. Gate: CONTINUE d ≥ +0.020 & P ≥ 0.90, KILL d < +0.005, on `t050_c030` / g01. Nothing trained
beyond fold 0, nothing exported or submitted.

- Pre-registration: `docs/v97_style_refiner_preregistration_2026-10-24.md`, sha256 `cff465c8…`
  (`outputs/v97/prereg.sha256`, written before training). Section 1 of it is the ledger of every earlier line.
- Code: `scripts/v97_style_refiner.py` (fold / extract / gate / demo), `scripts/_v97_chain.sh`.
- Outputs: `outputs/v97/{gate.json,chain.log,train_*.log,extract_*.log,gate.log,cond_fold0.json,soft_g01_fold0.json}`
  (VM + local); checkpoints `outputs/v97/{cond,soft_g01}_fold0_refiner.pt` (sha256 `27c5bd05…`, `fe568cb3…`);
  extracts `outputs/v97/extract_*.pkl` (VM).
- Harness: V97 extractor reproduced `outputs/v84/extract_soft.pkl` fold-0 counts, IoU tables and feature rows
  exactly; control reproduced the V84 bundle fold-0 per-file counts (max diff 0.0); demote refit deterministic.
- Training: V84 soft recipe (stage A 8 ep on 6559 GT crops, stage B 12 ep on 6079 OOF-proposal records; soft_g01
  stage B on the 2612 group-01 records), seed 42, GPUs 1 and 2, 18:39 → 18:59 UTC. Sanity: `cond@00` vs soft on
  all = +0.0010 [−0.001, +0.004] (the conditioning did not harm the pooled model).

## Numbers (fold 0, 141 files, 219 annotator records: hard 26, g01 88, rest 131; cut 0.30 + MIN200, no demote)

PQ per subset:

| arm | all | hard | g01 | rest |
|---|---:|---:|---:|---:|
| V84 soft (control) | 0.4573 | 0.4096 | 0.4318 | 0.4741 |
| soft_g01 | 0.4553 | 0.4035 | 0.4306 | 0.4715 |
| cond@00 (pooled token) | 0.4583 | 0.4076 | 0.4319 | 0.4756 |
| cond@01 (group-01 token) | 0.4556 | 0.4004 | 0.4288 | 0.4731 |
| cond@02 (group-02 token) | 0.4561 | 0.4042 | 0.4269 | 0.4751 |
| cond@oracle (own group) | 0.4571 | 0.4004 | 0.4288 | 0.4757 |

Deltas vs control, per-file bootstrap (2000, seed 97); primary cell in bold:

| arm | subset | d | 95% CI | P(d>0) | d with V84 demote |
|---|---|---:|---|---:|---:|
| soft_g01 | **g01** | **−0.0012** | [−0.008, +0.006] | 0.34 | −0.0028 |
| soft_g01 | hard | −0.0061 | [−0.017, +0.003] | 0.08 | −0.0079 |
| soft_g01 | rest | −0.0026 | [−0.006, +0.000] | 0.03 | −0.0078 |
| soft_g01 | all | −0.0021 | [−0.005, +0.001] | 0.10 | −0.0058 |
| cond@01 | **g01** | **−0.0030** | [−0.012, +0.006] | 0.26 | −0.0031 |
| cond@01 | hard | −0.0092 | [−0.023, +0.002] | 0.05 | −0.0092 |
| cond@01 | rest | −0.0010 | [−0.004, +0.002] | 0.24 | −0.0008 |
| cond@01 | all | −0.0018 | [−0.006, +0.002] | 0.20 | −0.0018 |
| cond@02 | rest | +0.0011 | [−0.003, +0.005] | 0.70 | +0.0033 |
| cond@oracle | all | −0.0002 | [−0.005, +0.004] | 0.47 | +0.0012 |
| cond@oracle | g01 | −0.0030 | [−0.012, +0.006] | 0.26 | −0.0031 |

Mechanism check, g01 subset (what the arms were supposed to move):

| arm | FN iou 0 | FN iou .3–.5 | FP iou .3–.5 | near-miss pairs | inter / pred area | inter / GT area |
|---|---:|---:|---:|---:|---:|---:|
| V84 soft | 150 | 75 | 76 | 77 | 0.71 | 0.58 |
| soft_g01 | 150 | 73 | 75 | 75 | 0.71 | 0.59 |
| cond@01 | 150 | 79 | 82 | 82 | 0.70 | 0.60 |
| cond@oracle | 150 | 79 | 82 | 82 | 0.70 | 0.60 |

## Reading

- **The near-miss band did not move.** On group-01 records the IoU 0.3–0.5 band stays at 73–82 FN and 75–82 FP
  for every arm, and the containment of near-miss pairs stays at ≈0.70 of the prediction inside the GT and ≈0.60
  of the GT inside the prediction. The near misses are not "we draw too short" or "too long": they are partial,
  offset overlaps in both directions at once, which no systematic extent shift fixes. That is also why ±1 px
  width (V92) and the clamp/crop/encoder arms (V73) never moved anything.
- **Zero-overlap FN are untouched by construction** (150 on g01 in every arm): the refiner only redraws
  proposals that exist.
- **The oracle is the decisive number.** Even with each record's own group known, the conditioned refiner scores
  the same as the pooled one (−0.0002 on all). The per-group label differences are not a learnable extent style
  at the crop level; they are per-filament choices (which faint ends, which fragments) that look like noise to a
  256-px crop model. With V96 (detector side) this closes "annotator style" as a lever anywhere in the pipeline.
- The group-02 token helps group-02-heavy `rest` slightly (+0.001, +0.003 with demote) and hurts g01 (−0.005): the
  model learned *something* per group, but it is smaller than the per-fold noise floor (±0.005) and does not
  transfer to the hard subset.

## Consequence

Every pipeline-side lever is now closed by a pre-registered experiment: detector architecture/input/labels/
ensembles (V66–V94), selection (V70/V83/V86/V88), refiner recipe (V73/V82/V89/V90), submission knobs on the
public board (V92/V95), annotator style via detector (V96) and via refiner (V97). The remaining gap to the
public board is the recall on zero-overlap, mostly single-annotator filaments (V85: 95 % single-annotator;
V39–V41: no image signal). The only instrument that can still move the public number is the public board
itself (the V95 object-class probes, paused), and the only pipeline change with a non-zero prior is a new
proposal source for faint filaments, which V91 (spine net) already tried at the noise floor.

Not touched: test images, test labels, MAGFiLO release, leaked CSV. No Kaggle submission.
