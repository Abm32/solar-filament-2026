# V81 pre-registration: label geometry (thin polygons, union-of-annotators), fold-0 screen (2026-10-11, before any training)

## Why
V79 audit (`outputs/v79_audit/labels_report.json`):
1. **Fat labels.** Ultralytics rasterises our polygons (`polygon2mask`, cv2 fillPoly incl. boundary pixels) about 1 px/side
   too fat at 2048. Median area ratio vs the pycocotools GT is 1.217 at 1024 (stage 2, mask_ratio 1) and 1.325 at 512
   (stage 1, mask_ratio 2). Mean IoU vs GT is 0.810 / 0.739. Median filament width is 9.5 px at 2048, so +2 px is about +20 %.
   The detector learns to predict this bias.
2. **Missing-annotation conflict.** Each annotator record is a separate training image. A filament drawn by A but not by B
   is a positive in A's copy and background in B's copy of the same pixels. Oracle PQ: union of annotators 0.663 >
   single annotator 0.605 > majority 0.592.

## Arms (p2first graph and init, V21/V17 schedule 80 ep mask_ratio 2 + 20 ep mask_ratio 1, seed 42, fold 0, one T4 each)
Built by `scripts/v81_labels.py` (installed by `scripts/patch_v81.py`) into `<work>/yolo_fold0_<kind>`. The baseline
`yolo_fold0` dir is never touched. Only the 566 fold-0 train files are used. No held-out file, test image or external GT.

| arm | training images | labels |
|---|---|---|
| p2first_thin | 935 (one per record, as baseline) | every polygon vertex moved 1.0 px inward along its mitre (cap 2 px), the same instances |
| p2first_union | 566 (one per file) | across-record clusters (IoU>0.5, max one per record, greedy as V79 `multi_oracles`). Each cluster gives one instance, the polygon of its lower-median-area member. Unmatched instances are kept. |

Shrink value fixed on CPU (2,000 random train annotations, seed 81, audit_v79 `yolo1024` path):

| labels | area ratio @1024 | mean IoU @1024 | area ratio @512 | mean IoU @512 |
|---|---|---|---|---|
| original | 1.216 | 0.810 | 1.325 | 0.739 |
| shrink 0.75 px | 1.040 | 0.878 | 1.144 | 0.800 |
| **shrink 1.0 px (chosen)** | **0.987** | **0.876** | 1.084 | 0.809 |
| mask erode 1 px + contour | 0.926 | 0.829 | n/a | n/a |

Union on fold-0 train (as built by `v81_labels.build`): 935 records with 6,559 instances become 566 images with 5,145
instances. Instances per image go from 7.01 to 9.09. 243 multi-record files: mean 6.69 per record vs about 10.9 in the union (1.33x the
largest record). 1,414 duplicates merged. 1,558 single-annotator instances in multi-record files. 2,764
positive-vs-background conflicts removed (instance x other-record copies). Greedy clustering depends on order: a
json-order pass gives 5,128 / 1,431 / 2,713, so the build (sorted record id) is the reference.

**Known confound (union):** with fixed epochs, one image per file means 566 instead of 935 images per epoch, so about 40 % fewer
optimizer steps, and single-record files get relatively more weight. A null result does not refute the label change. Any
follow-up repeats each union image once per record (same 935 steps), which isolates the labels.

**Not run: p2first_union_w** (records kept separate, an instance's region ignored as negative in copies that lack it).
Ultralytics has no per-pixel or per-region ignore mask in the seg loss, and the background for the cls/DFL assigners is
implicit. It would need a patched `v8SegmentationLoss` plus a dataset field threaded through collate. Feasible but
invasive (an in-process loss patch like `v61_seg_loss`), so it is deferred until union shows signal.

**Difference from the earlier negative union result (V56, 2026-10-04):** V56 trained a Mask R-CNN on `--target union`
and tune-evaluated it against union GT. PQ fell (1536: 0.358 vs 0.375 single-trained), recall stayed flat (about 0.52-0.54), and FP rose (about 545).
That was a different model (a weaker, about 0.37-level pipeline with no consensus or refiner), and it was scored only
against union GT. V81 changes only the labels of the current best graph (p2first YOLO11s-seg, same schedule, 4-flip
consensus, r3 refiner). It merges duplicates explicitly, and evaluation is unchanged: per record against each annotator,
which is the Kaggle metric. **Prior is weak:** the V56 failure mode (looser firing, more FP, flat recall) is the expected
risk here too. The test is whether recall on single-annotator filaments (V79: recall_single 0.16-0.35 vs agreed
0.53-0.85) rises more than per-record precision falls.

## Measurement and gate (same as V68 / V78)
`eval_v61_oof.py infer` (V17 cell, 4-flip consensus) on the 141 fold-0 files, then
`eval_v61_compare.py --fold 0 --a p2first --b <arm> --noise p2first_seed2`.
CONTINUE iff d >= +0.010, P(d>0) >= 0.90, and d > |seed noise|. KILL iff d < +0.005. Otherwise inconclusive.
Two arms are screened, and neither is selected by the other. **Refined-pipeline requirement (V77 lesson):** the r3 refiner
already corrects mask geometry (refiner256 IoU 0.938), so thin may help the raw detector and vanish after refining. A
continuing arm must also pass the V75/V77 harness (same 141 files, same r3 weights, not retrained, same cut) at
d >= +0.010 vs p2first+r3 before any 5-fold run or export. No submission without the user's go-ahead. Disclosed in the
technical report.

## Launch (after `python scripts/patch_v81.py` on the real file; one free GPU each)
```
CUDA_VISIBLE_DEVICES=<g> nohup .venv/bin/python scripts/train_v61_fold.py --fold 0 --recipe p2first_thin  --work /mnt/v81 --device 0 > /mnt/v81/thin.log 2>&1 &
CUDA_VISIBLE_DEVICES=<h> nohup .venv/bin/python scripts/train_v61_fold.py --fold 0 --recipe p2first_union --work /mnt/v81 --device 0 > /mnt/v81/union.log 2>&1 &
```

## Launch scope (2026-10-11, before any V81 run)
Only p2first_thin runs now, on GPU 3 after V76 finishes (scripts/_v81_chain.sh). p2first_union is NOT launched in this form: because of the 40% fewer-steps confound it will be rebuilt with each union image repeated once per record (same 935 images/epoch) under its own addendum before it runs.

## Fold-0 result (2026-10-11, mechanical)
p2first_thin 0.4263 vs p2first 0.3991 @0.30: d +0.0272, CI [+0.0177, +0.0372], P 1.000; seed noise -0.0050. Verdict: CONTINUE.
Deviation, stated before any further result: folds 1-4 train now (scripts/_v81_fold.sh; GPUs 0-2, fold 4 after fold 1) in parallel with the fold-0 refined test, to save wall time. The refined-pipeline test (+0.010) still gates any export; the 5-fold direct result also needs pooled >= +0.010.
p2first_union not run yet (see launch scope).

## Refined fold-0 test (2026-10-11, mechanical): KILL
Pipeline reproduces V73 base fold 0 exactly (0.453259). p2first+refiner 0.4533 (1028/431/612) vs thin+refiner 0.4516 (1032/447/608): d -0.0017, CI [-0.0102, +0.0071], P 0.35. MIN200: -0.0018. The refiner removes the thinner geometry (matched area/GT 1.083 vs 1.074). Folds 1-4 stopped at about 1 h of training (user decision precedent: fold-99 after V77). outputs/v81/refined_fold0.json.
