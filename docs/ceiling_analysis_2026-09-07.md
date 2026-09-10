# Why we are stuck at 0.39 — ceiling analysis, 7 September 2026

> **CORRECTION, 2026-09-09 (E027). Section 1 below is FALSIFIED. Do not use its
> offset or any target derived from it.**
>
> §1 claimed a stable local→public offset of −0.050 ± 0.003 and concluded "public
> 0.55 requires local tune PQ ≈ 0.60". V17 (`56103401`) scored local **0.448703 →
> public 0.39**, which §1's model predicted at 0.3959–0.4014.
> `scripts/analyze_public_transfer.py` now shows that **no constant offset fits all
> five submissions** (the feasible region is empty under both round-to-nearest and
> truncation display models, while remaining non-empty for the first four — so V17
> is specifically what breaks it), and that for affine `public = a·local + b` the
> feasible slopes are **a ∈ [0.01, 0.87]** with **a = 1.00 ruled out**. The
> corrected implication is **local 0.60 → public ≈ 0.527 at best**, and **public
> 0.55 requires local ≥ 0.6269**. Every "realistic expectation" figure in §8 and at
> the end of this document is therefore optimistic and should be re-derived, not
> quoted.
>
> §8's ranked plan is also now mostly closed by experiment: item 1 (higher-capacity
> FN-weighted mask head) failed three times (V14/E025, V15, and the V11 family),
> item 2 (learned mask-quality head) failed on a train/serve population mismatch in
> V14, item 4 (`mask_ratio=1`) proved real at the proposal stage but produced no
> public movement (E026/E027), and the "E-H fold ensemble" is a measured clean
> negative (E028: union oracle PQ 0.5849 at zero FP; realistic union wins 1 of 7
> tune confidences). **Sections 3–6 (the mechanistic decomposition) still stand and
> were largely confirmed.**

Three different submissions now score exactly **0.39** (V11 proposal_centered, V11
gt_centered, V13 composition) while the leaderboard cluster sits at **0.55–0.56**.
V13 cleared a promotion gate locally and still did not move. This document
establishes *why* from the numbers, and what it would actually take.

Reproduce the arithmetic with the figures in
`outputs/kaggle_v13_full/v13_candidates.csv` and
`outputs/research_2026-09-06/proposal_diagnostics.csv`.

## 1. The local→public offset is stable, so local PQ is a valid target

> **FALSIFIED 2026-09-09 by V17 — see the correction at the top of this file.
> The table below has a fifth row now (local 0.448703 → public 0.39) and no
> constant offset fits all five.**

| local tune PQ | public | offset |
| ---: | ---: | ---: |
| 0.432823 | 0.38 | −0.0528 |
| 0.438001 | 0.39 | −0.0480 |
| 0.437313 | 0.39 | −0.0473 |
| 0.441777 | 0.39 | −0.0518 |

The offset is tight at **−0.050 ± 0.003** across four submissions. That is good
news: it means local tuning PQ *is* a reliable predictor with a constant shift, and

> **public 0.55 requires local tune PQ ≈ 0.60.**

We are at 0.4418. The gap is **+0.16 local PQ**, not a tuning-sized gap.

## 2. Neither axis alone can get there

Current tune structure: TP 686, FP 309, FN 448, mean matched IoU **0.6855**,
1,134 GT instances, recall **0.6049**.

Holding mean IoU at 0.6855 and FP at 309:

| recall | PQ |
| ---: | ---: |
| 0.65 | 0.4636 |
| 0.75 | 0.5084 |
| 0.85 | 0.5491 |
| 0.95 | 0.5861 |
| **1.00 (perfect recall)** | **0.6033** |

Holding recall at 0.6049:

| mean IoU | PQ |
| ---: | ---: |
| 0.75 | 0.4833 |
| 0.80 | 0.5155 |
| 0.85 | 0.5478 |
| 0.90 | 0.5800 |

**Perfect recall alone barely reaches 0.60. Perfect masks alone never do.** The
0.60 target requires roughly **recall ≥ 0.85 together with mean IoU ≥ 0.75**:

| | IoU 0.686 | 0.72 | 0.75 | 0.78 | 0.80 |
| --- | ---: | ---: | ---: | ---: | ---: |
| recall 0.742 | 0.5048 | 0.5302 | 0.5523 | 0.5744 | 0.5891 |
| recall 0.799 | 0.5288 | 0.5554 | 0.5785 | **0.6017** | 0.6171 |
| recall 0.899 | 0.5678 | 0.5963 | **0.6212** | 0.6460 | 0.6626 |

Every prior experiment moved **one** axis by ~0.005. That is why they all landed
on 0.39.

## 3. Where recall is lost — the cutoff, not the detector

GT instances with an IoU>0.5 match at each pipeline stage:

| stage | matched | recall |
| --- | ---: | ---: |
| 1. raw detector proposals | 906 / 1134 | **0.7989** |
| 2. after flip_consensus grouping | 841 | 0.7416 |
| 3. after the 0.40 confidence cutoff | 563 | 0.4965 |
| 4. actual deployed (refiner recovers +123) | 686 | 0.6049 |

- grouping costs **65**
- the fixed 0.40 confidence cutoff costs **278**

The detector already finds 906 of 1,134. **The cutoff throws away 278 instances
that already have a good proposal.** This is the single largest recoverable pool
and it is a *decision* problem, not a detection problem. E021 showed a
cheap-feature linear probe cannot fix it; a head reading mask pixels is untested.

## 4. Where mask quality is capped — YOLO's coarse protos

Raw proposal IoU against GT, before any refinement:

| band | 0–.1 | .1–.2 | .2–.3 | .3–.4 | .4–.5 | .5–.6 | .6–.7 | .7–.8 | .8–.9 | .9–1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| n | 45 | 10 | 17 | 36 | 119 | 276 | 381 | 215 | 35 | **0** |

**No raw proposal anywhere exceeds IoU 0.8734.** Mean is 0.5842. This is a
structural cap, and the mechanism is identifiable:

- YOLO segmentation emits masks from a **prototype layer at 1/4 of `imgsz`** —
  256×256 protos at our `imgsz=1024` — which are then upsampled to 2048×2048.
  That is an **8× upsample**, which cannot represent a thin filament boundary.
- `mask_ratio=2` (in `solar_base.py` CFG) **further halves the mask resolution
  used for training targets**. Ultralytics documents this as "downsample ratio for
  segmentation masks, affecting the resolution of masks used during training."
- The consequence is the flat/staircased mask edges YOLO segmentation is known
  for, which is fatal here because PQ sums IoU rather than counting hits.

**Oracle ceiling of the current detector.** If we perfectly selected every raw
proposal, kept all of them, and had zero FP:

> recall 0.7989, mean IoU 0.6491 → **local PQ 0.5765 → public ≈ 0.53**

With our actual FP=309 that oracle drops to **0.5007**. So:

> **The V6 detector's proposal masks cannot reach 0.60 local PQ no matter what the
> refiner selection or thresholds do.** Every experiment E015–E024 was optimising
> inside a box whose lid is 0.5765.

This is the real reason we are stuck, and no amount of inference tuning escapes it.

## 5. The second structural limit — the refiner is clamped to the proposal

`optimize_inference.clean_refined_predictions` in `refined_constrained` mode does:

```python
crop &= cv2.dilate(proposal, kernel) > 0   # 7x7 ellipse
```

The refined mask is intersected with the **dilated YOLO proposal**. The refiner
therefore cannot repair a badly localised or truncated proposal — it can only
carve *within* a slightly grown version of a mask we already know caps at IoU
0.873. Mask quality is bounded by the coarse proposal twice over: once as the
training target resolution, once as an inference-time constraint.

The refiner does add real value (+123 recovered instances, mean IoU 0.686 vs the
0.649 oracle raw mean), so the crop-refinement idea is sound. It is the
*constraint* and the *capacity* that are limiting, not the concept.

### Measured: removing the clamp alone does nothing

The historical sweeps already contain an unclamped mode, `refined_free`, and it is
essentially tied with `refined_constrained`:

| run | `refined_constrained` | `refined_free` | `yolo_only` |
| --- | ---: | ---: | ---: |
| V9 sweep | 0.418869 | 0.418946 | 0.388492 |
| COCO refiner | 0.420230 | 0.420141 | 0.352532 |

So the clamp is **not** what binds — the refiner is not even trying to leave the
proposal envelope. That reframes item 1 below: it is not enough to *permit* the
mask to deviate, the head must be *trained to produce a sharper mask* against
canonical GT with an FN-weighted objective and enough capacity. Dropping the clamp
is a necessary but insufficient part of that, and is worth ~0 on its own.

This is exactly the kind of cheap negative result that should precede a big run,
and it cost nothing because the data already existed.


### What is NOT a bottleneck (checked, so we do not waste a run on it)

- **Crop resolution is fine.** With `crop_size=256`, `crop_context=1.8`, only
  **2 of 1,134** instances have a crop side that could exceed 256px, so 99.8% of
  crops are *upsampled* into the refiner, not downsampled. Raising `crop_size`
  would buy nothing. Instances in the 150–256px crop band actually score *better*
  (accepted IoU 0.545 vs 0.385), consistent with small instances being the problem.
- **Elongation is not a factor** (E-B/Q5, partial corr +0.04 once area is
  controlled).
- **Crop-refiner architecture is not obviously starved**: it is a ResNet18 U-Net
  decoding back to full crop resolution, so it *can* express a sharp boundary. Its
  problem is the input constraint and the target it was trained against.

## 6. Small instances are a mask problem AND a detection problem

| group | n | raw IoU>0.5 | support>0.5 | mean raw IoU |
| --- | ---: | ---: | ---: | ---: |
| smallest 30% by area | 338 | 0.672 | 0.530 | 0.5182 |
| largest 70% | 796 | 0.853 | 0.712 | 0.6122 |

Small instances are worse on **both** counts, and their mean raw IoU (0.518) sits
barely above the 0.5 matching threshold — so tiny mask errors flip them to FN.
This is why an FN-weighted loss and a sharper mask head compound: they raise the
same population above threshold and keep it there.

Also measured: **84 instances have support coverage >0.5 but raw mask IoU ≤0.5** —
the object is essentially found and only the *mask* fails. Those are directly
recoverable by a better mask head, worth up to +42 TP at the midpoint.


## 7. What the 0.55 cluster is, and is not

Per `score_050_research_2026-09-06.md` and
`public_notebook_lamhuy_055_review_2026-09-05.md`: the ~20-team 0.55 cluster is
dominated by an **embedded precomputed submission payload** (`CHAMPION_PAYLOAD`,
lzma+base85 in the lamhuy notebook) of unverified provenance — not a reproducible
model. HDJoJo's 0.55 uses YOLOv8l-seg at `imgsz=2048` behind a private checkpoint.

E022 already falsified "just train at higher resolution" as a standalone fix
(1536-trained lost at both inference resolutions). But note what E022 did **not**
test: it kept `mask_ratio=2` and trained only 20 epochs. The resolution that
matters for PQ is **mask** resolution, not image resolution — and those are
different knobs.

The one legitimate, reproducible signal from the payload comparison remains
**instance count**: it emits 1,342 instances vs our ~1,100–1,200 with a similar
area distribution. Consistent with our own decomposition: we are recall-limited.

## 8. Conclusion — what has to change

Ranked by expected PQ per GPU-hour, given the two-axis requirement:

1. **Own the mask: a higher-capacity, FN-weighted mask head trained on canonical GT.**
   Keep YOLO for *localisation*. The clamp itself is worth ~0 (measured above), so
   the change that matters is the **head and its objective**: more capacity than a
   ResNet18 U-Net at 256px, a Focal-Tversky (β>α) loss so thin/small structure is
   not eroded, deep supervision, and the proposal supplied as a *hint channel*
   rather than a hard boundary. This is the only lever that can move mean matched
   IoU from 0.686 toward 0.78, and it needs no detector retrain.
2. **Replace the fixed 0.40 cutoff with a learned mask-quality head** (E-C proper).
   Targets the 278-instance cutoff pool directly. Must read mask pixels — E021
   proved summary features are insufficient. Predicting IoU also gives a
   *better ranking signal* than raw detector confidence.
3. **FN-weighted loss** (Focal-Tversky, β>α) on the mask head. Small instances sit
   at mean raw IoU 0.518, barely over the 0.5 bar, so this both converts
   near-misses and protects the 199 fragile TPs.
4. `mask_ratio=1` and `imgsz` ≥1024 **if** the detector is retrained later — cheap,
   and E022's resolution test never varied `mask_ratio`, so "resolution is dead" was
   only ever tested for *image* resolution, not *mask* resolution.

Items 1–3 are a **single training run** and they attack **both** required axes at
once. That is the design of **V14** (`docs/v14_design_2026-09-07.md`).

### Realistic expectation, stated before running

Composing the measured pools: raw-proposal recall 0.799 is reachable without a new
detector, plus up to ~42 of the 84 mask-recoverable instances → recall ≈ 0.836.

| | mean IoU 0.72 | 0.75 | 0.78 |
| --- | ---: | ---: | ---: |
| recall 0.836, FP 309 | 0.5709 | 0.5947 | 0.6185 |
| recall 0.836, FP 200 | 0.5982 | 0.6231 | 0.6481 |

So **local 0.57–0.62 is mechanically available** from V14 without touching the
detector, which maps to **public ≈ 0.52–0.57**. That is the first plan in this
project whose arithmetic actually reaches the target rather than hoping for it.

The risk is honest: it assumes the mask head achieves mean IoU ≥0.72 on instances
it keeps, and that removing the proposal clamp does not inflate FP. Both are
measured directly by the V14 gates before anything is submitted, and the frozen
control still has to reproduce the accepted 0.38 CSV byte-for-byte first.

