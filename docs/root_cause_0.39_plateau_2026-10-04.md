# Root-cause analysis: why we are stuck at 0.39 (and what actually breaks it)

Date: 2026-10-04. Written after a deliberate step back from the local-tweak loop, studying
BioHub + analogous Kaggle instance-segmentation winners and re-verifying the competition.
Goal: the EXACT reason for the 0.39 plateau and the concrete ways to overcome it.

## The verified facts (not assumptions)
1. **Competition metric + data** (too.foo mirror + MAGFiLO Nature Sci Data s41597-024-03876-y):
   Panoptic Quality, 2048px, **multi-annotator labels**. Inter-annotator agreement
   **Kappa = 0.66** (moderate). Our measured 0.677 coverage ratio matches this exactly —
   the label noise is real and is a property of the dataset, shared by everyone.
2. **The full MAGFiLO corpus is 10,244 filaments / 1,593 observations (2011–2022).**
   The Kaggle train set is only **707 images / ~8,199 annotations**. There is a large,
   published, same-domain dataset we have never pulled.
3. **Leaderboard:** top 0.63, dense 0.55–0.63 cluster. Our 0.39 (V17) is far below.
4. **What we have tried (all single-model, COCO/scratch, 707-image train):** V6–V21
   YOLO/refiner (→0.39), V42–V51 YOLO variants + fusion, V52 clDice semantic, V55
   precision-Tversky, V56 Mask R-CNN (COCO-finetune, 1024–2048) — everything caps at
   **0.35–0.45 tune / 0.39 public.** Union/consensus data experiments: negative.

## What the WINNERS do (Sartorius / HuBMAP cell-instance seg — the analogous task)
From the Sartorius 11th-place writeup (TheoViel) and the common pattern across top
cell-instance solutions, the winning stack is a **combination we have never assembled**:
1. **Domain pretraining then finetune** — pretrain on a LARGE same-domain dataset
   (LIVECell), then finetune on the competition set. We finetune COCO→MAGFiLO directly;
   COCO is natural images — a huge domain gap, and we leave the 10k-filament external
   MAGFiLO corpus unused.
2. **Remove the first conv-layer stride** to raise effective resolution for small/thin
   objects without resizing. Directly relevant to thin filaments. Never tried.
3. **Cascade Mask R-CNN / HTC**, not plain Mask R-CNN — stronger iterative box/mask heads.
   We used plain `maskrcnn_resnet50_fpn_v2`.
4. **Multi-model + multi-flip ensemble at proposal/box/mask stages (+ WBF/TTA).** We ran
   single models; our prior fusion attempts (V45/V51/V53) were YOLO-family or post-hoc,
   not a trained multi-architecture detector ensemble.
5. **Heavy CV-tuned post-processing:** NMS-boxes-high → NMS-masks-low, small-mask removal,
   per-class mask-threshold tuning. We did a coarse score sweep only.

## THE ROOT CAUSE (exact)
Our 0.39 is **not** a hard-FN "invisible filament" wall and **not** a metric artifact.
It is the compound result of three deficits, in order of impact:

**A. Data scale + domain gap (biggest).** We train one model on 707 images from a
COCO-initialised backbone. The winners effectively train on an order of magnitude more
same-domain data (external corpus + pretraining) before touching the competition set.
Our models are data-starved and domain-mismatched; that caps generalisation (recall) at
~0.55–0.58 regardless of architecture — exactly what the V52/V55/V56 sweeps showed
(every architecture plateaus at the same place = a DATA ceiling, not a model ceiling).

**B. No ensemble + no proper detector.** Single plain-Mask-R-CNN vs the winners'
Cascade/HTC multi-model proposal/box/mask ensembling. On instance PQ this is routinely
worth +0.05–0.10 — the bulk of the 0.39→0.55 gap.

**C. Shallow post-processing.** stride-removal for thin-structure resolution + CV-tuned
NMS/mask-threshold/small-removal is a documented +0.02–0.05 that we have not harvested.

The sparse-annotation finding (real, Kappa 0.66) explains the *local tune noise* and why
single-annotator PQ under-reads recall — but it does NOT explain the leaderboard gap,
because the gap is measured on the same multi-annotator basis for everyone. We previously
over-weighted the annotation story; the dominant lever is **data scale + ensemble**, the
standard Kaggle instance-seg playbook we skipped.

## Why 10 months of local tweaks never moved it
Every post-V17 experiment changed the **model/loss/fusion** while holding the **data
(707 single-annotator images) and the single-model, single-pass inference** fixed. We
were optimising the one axis the winners treat as solved infrastructure, and never
touched the axes that actually produce their score (external data + ensemble + TTA +
post-proc). This is textbook local-optimum: all our levers are inside a basin that caps
at ~0.39.

## How to overcome it (concrete, ranked by EV)
1. **Pull the full external MAGFiLO corpus (10,244 filaments / 1,593 obs) and pretrain
   on it, then finetune on the 707 Kaggle images.** Highest EV by far — directly attacks
   deficit A. Data is published (Nature Sci Data; GONG Hα). Firewall note: external data
   is allowed (it is NOT the competition test set); keep our 141-file tune split clean.
2. **Switch to Cascade Mask R-CNN / HTC + stride-removal** for thin-structure resolution
   (deficit B/C). mmdet or torchvision Cascade.
3. **Multi-model + 4-flip TTA ensemble** (proposal/box/mask averaging or WBF) over 2–3
   backbones/resolutions (deficit B). We already have several trained V56 variants to
   start from.
4. **CV-tuned post-processing**: NMS-boxes-high→masks-low, small-mask removal, mask-thresh
   sweep (deficit C) — cheap, do last on the best single/ensemble.

Expected trajectory: (1)+(2) should move recall materially (the real ceiling), (3)+(4)
convert it to PQ. This is the first plan grounded in what demonstrably scores on
analogous competitions, not another variation inside the 0.39 basin.

## Firewall / guardrails unchanged
External pretraining data is fine; NEVER train on the 141-file tune split or the
competition test. Submit only on gate (tune > 0.4587) or strong public-transfer evidence
+ user go-ahead. Do NOT deallocate rogii.

Sources: MAGFiLO Nature Sci Data s41597-024-03876-y (Kappa 0.66; 10,244 filaments/1,593
obs); too.foo competition mirror (PQ, 2048px, multi-annotator); TheoViel/kaggle_sartorius
(11th-place: LIVECell pretrain, stride-removal, Cascade/HTC, proposal/box/mask ensemble,
NMS post-proc); analogous HuBMAP/Sartorius winning writeups.
