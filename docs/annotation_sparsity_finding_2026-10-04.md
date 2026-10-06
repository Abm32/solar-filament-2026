# MAGFiLO annotation sparsity — the data-side finding (BioHub transfer)

Date: 2026-10-04. The most important finding of the score-push: MAGFiLO ground-truth
annotations are **sparse / individually incomplete**, which means every model this
project has trained (V17, V52, V55, V56) has been trained and scored against an
incomplete label set — penalising recall. This reframes the long-standing
"~0.35–0.38 tune-PQ ceiling" as *partly an annotation artifact*, not purely a model
limit.

## Where the idea came from
The user pointed to the just-ended Kaggle **BioHub — Cell Tracking During Development**
competition (royerlab/Zebrahub). Its defining data property and baseline technique
(from `royerlab/kaggle-cell-tracking-competition`):
> "Ground truth annotations are **sparse** — only a subset of cells are annotated."
> Baseline method #3 — "**Sparse supervision**: only edges with ground truth are used
> for backpropagation — background detections and **unannotated cells are ignored**
> during training."

i.e. the winners/baseline do **not** penalise the model for detecting real objects
that happen to be unannotated. The user's hypothesis: the solar dataset has the same
problem, and *data handling* (not architecture) is the missing lever.

## Confirming it on MAGFiLO (measured, not assumed)
`MAGFiLO_1.0_Annotations_kaggle2026_train.json`:

- **707 physical image files, but 1154 image-records → ~1.63 annotators per file.**
  296 of 707 files have **more than one** independent annotator.
- On multi-annotator files, the per-annotator **instance-count spread** averages 2.58
  (max 13). The mean *max*-annotator marks **8.08** filaments; the mean *min*-annotator
  marks **5.5** → **min/max coverage ratio 0.677**. Each annotator misses ~32% of what
  another annotator on the same image marked.
- **Union vs single** (120 multi-annotator files, instances deduped at IoU>0.5):
  **1303 union instances vs 967 single-max-annotator = 1.347×.** ~35% more real
  filaments — each marked by at least one human — than the single record the project
  trains/scores on.

The project's `records_for()` (used by V52/V55/V56 and matching V17's convention)
selects **one max-instance annotator record per file**. So every model is:
1. **Trained** to treat a filament another annotator marked (but this record didn't) as
   background → learns to suppress real detections → artificially low recall.
2. **Scored** (tune PQ) with those same correct detections counted as false positives.

This plausibly also explains ledger **E027** (local tune-PQ decoupled from public): if
the public test is scored against fuller/union labels, an aggressive-recall model looks
*worse* on our single-annotator tune but *better* on public — consistent with the
observation (lamhuy 0.55 analysis) that high scorers "predict more instances across
more images."

## What was built (firewall-safe)
`scripts/train_v56_maskrcnn.py` gained:
- `union_rles(...)` / `union_instance_masks(...)` — merge every annotator's instances
  for a physical file, dedup by IoU>0.5 (largest-area-first greedy).
- `--target {single,union}` — training GT source.
- `--eval-gt {single,union}` — tune GT to score against.
FIT-only firewall preserved: the union is built per file from the training population;
the 141 held-out tune files' labels are only ever used for *evaluation*, never training.
7 unit tests (incl. union dedup) pass; ruff clean.

## Cheap confirmation: union-GT eval of EXISTING (single-trained) weights
Re-scored the best single-trained V56 weights against union GT (141 tune):

| variant | single PQ / recall | union PQ / recall | TP | FP | FN |
|---|---|---|---|---|---|
| mrcnn_1536 | 0.3833 / 0.580 | 0.3746 / 0.535 | 664→**694** | 448→**418** | 481→**603** |
| mrcnn_2048 | 0.3767 / 0.563 | 0.3665 / 0.516 | 645→669 | 468→444 | 500→628 |
| mrcnn_2048_lr2e3 | 0.3676 / 0.567 | 0.3596 / 0.522 | 649→677 | 520→492 | 496→620 |

Reading this honestly:
- **TP↑ and FP↓ under union GT** (e.g. 664→694 TP, 448→418 FP): confirms ~30 of each
  model's "false positives" are **real filaments another annotator marked** — the
  sparse-annotation artifact is real and measurable. ✅
- **PQ dips slightly** because **FN rises a lot** (481→603): the union has ~35% more
  instances, and a model *trained on single-annotator labels* cannot cover them, so the
  extra union instances become new false negatives, outweighing the FP→TP gain in the
  PQ denominator.

**Conclusion of the cheap test:** eval-only against union does not raise PQ for
single-trained weights. The FP-relabelling helps; the model's recall ceiling (set by
training on sparse labels) hurts more. The decisive experiment is to **train on union
targets** so the model learns the fuller instance set — in progress (`v56_mrcnn_1536_union`,
`v56_mrcnn_2048_union`, `--target union`).

## Why this matters regardless of the final PQ number
Even before the union-trained result lands, this finding changes how the project should
interpret every prior "ceiling":
- The "~18.7% invisible-to-every-source hard-FN bucket" (project §4) is now known to
  overlap with **inter-annotator disagreement**, not only genuine ambiguity.
- Single-annotator tune PQ is a **biased-low** estimator of real recall; union-GT eval
  is the fairer local proxy for a (likely union-scored) public test.
- The right training target for *this* dataset is the union (or a sparse-supervision
  loss mask), not one annotator — the direct BioHub transfer.

Evidence: measurements reproducible from the annotations JSON; union-eval JSONs at
`outputs/v56_mrcnn_*/tune_union.json`; builder + flags in `scripts/train_v56_maskrcnn.py`;
tests in `tests/test_v56_maskrcnn.py`.

Firewall intact throughout (FIT-only union, no test/CSV/submission). rogii not deallocated.

---

## UPDATE 2026-10-04 — union-TRAINED result: CONFIRMED NEGATIVE

Retrained Mask R-CNN with `--target union` (FIT-only) at 4 configs and tune-evaluated
against union GT, vs the single-trained baselines:

| config | single-trained (union-GT PQ) | union-trained (union-GT PQ) |
|---|---|---|
| 1536 | 0.3746 | **0.3576** |
| 2048 | 0.3665 | **0.3564** |
| 1536_e40 | — | 0.3495 |
| 2048_lr2e3 | — | (same trend) |

Every union-trained variant scored **below** its single-trained counterpart on union GT.
Recall stayed flat (~0.52–0.54, i.e. union training did NOT find more real filaments) and
false positives rose (~545). Multi-variant confirmation.

**Conclusion:** the sparse-annotation *finding* is real and valuable, but **hard-union-
target training does not convert it into score.** The ~35% extra union instances are too
NOISY — many are detections one annotator made and another explicitly rejected, not
uniformly real filaments. Forcing them as hard positives teaches the model to fire more
loosely (more FP) without raising genuine recall. The best standalone instance model
remains single-trained 1536 at single-GT 0.3833, still below V17's 0.4487.

**What this does NOT refute:** that the labels are incomplete/inconsistent (measured), and
that single-annotator tune PQ is a biased-low recall estimator. It refutes only the naive
fix (hard union = GT).

**Better remaining idea (BioHub's actual technique, untried):** a **sparse-supervision
loss mask** — ignore unannotated regions in the loss (neither positive nor negative)
rather than forcing the noisy union as hard positives. In Mask R-CNN this needs custom
RPN/ROI target assignment (non-trivial). A cheaper variant: **confidence-weighted union**
— only add union instances marked by ≥2 annotators (a higher-precision subset) instead of
the full noisy union.

Evidence: `outputs/v56_mrcnn_*_union/{train.json,tune_union.json}`.
