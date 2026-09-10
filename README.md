# Solar Filament Segmentation Challenge 2026

Repository for the Kaggle Solar Filament Segmentation Challenge 2026.

**Resume here (updated 2026-09-09):** the current public best is **0.39**,
confirmed by **four** independent submissions: V17 `mr1_iou030/t0.45_c0.30`
(submission **56103401**, [package](submission_ready/v17_2026-09-08/README.md)),
V13 `t0.45_c0.30` (**56076685**, [package](submission_ready/v13_2026-09-07/README.md)),
V11 `proposal_centered_e4` (**56049359**, [package](submission_ready/v11_2026-09-06/README.md))
and V11 `gt_centered_e4` (**56050554**,
[package](submission_ready/v11_gt_centered_2026-09-06/README.md)). Do not upload
any of them again. Deadline 2026-11-15. Full state and the current
recommendation: [`../RESUME_TOMORROW.md`](../RESUME_TOMORROW.md).

**The most important current result: local tune PQ has stopped predicting the
public score, and the model the project was steering by is falsified (E027).**
`ceiling_analysis_2026-09-07.md` §1 claimed a stable local→public offset of
−0.050 ± 0.003 and concluded "public 0.55 requires local tune PQ ≈ 0.60". Five
submissions now span local **0.432823 → 0.448703** (+0.015880) while the public
score moved **once**, by one display step:

| local tune PQ | public | submission |
| ---: | ---: | --- |
| 0.432823 | 0.38 | `56026663` |
| 0.437313 | 0.39 | `56050554` |
| 0.438001 | 0.39 | `56049359` |
| 0.441777 | 0.39 | `56076685` |
| **0.448703** | **0.39** | **`56103401`** |

`scripts/analyze_public_transfer.py` shows **no constant offset fits all five**
(the feasible region is empty under both round-to-nearest and truncation display
models, while being non-empty for the first four — so V17 is the falsifier), and
for affine `public = a·local + b` the feasible slopes are **a ∈ [0.01, 0.87]** with
**a = 1.00 ruled out**. Consequently **local 0.60 maps to public ≈ 0.527 at best,
and public 0.55 requires local ≥ 0.6269.** Do not quote the −0.050 offset again.

**And the last cheap lever is now measured and dead (E028).** The six proposal
populations on disk do miss *different* instances (union recall 0.8131 vs 0.7875
for the best single source), but **212 of 1134 tune instances (18.7%) are missed by
every source**, the union's *oracle* PQ is only **0.5849 at zero false positives**,
and a realistic cross-source-NMS ensemble **beats the best single detector at just
1 of 7 tune confidences** (max +0.0005). `ceiling_analysis` §8's E-H fold ensemble
is closed. Separately, V18 (`imgsz=1536` + `mask_ratio=1`) is **infeasible on a
Kaggle T4 at any batch size** — both runs verified the frozen control exactly and
then OOMed inside the Ultralytics segmentation loss.

**Current recommendation: stop optimising this pipeline and keep 0.39.** Not for
lack of runway (two months, 45k GPU-s) but because no remaining lever has a
measured positive expected value.

**V13 (E024) was the pivotal earlier result and it is a mixed one.** Composing
E-A's box `iou=0.3` detector flags with V11's refiner **cleared both predeclared
bars — the first candidate in the project to pass a promotion gate** — reaching
tune PQ 0.441777 / audit 0.464644 and beating all three parents on both groups
(+0.003776 tune / +0.009760 audit over the 0.39-scoring V11). **The public score
stayed at 0.39.** Two conclusions: E-A and V11 are genuinely **additive**
(orthogonal mechanisms, not two routes to one gain), and local tuning gains in the
**+0.003–0.010 band no longer predict public movement** — calibration is now
+0.0012 → 0.38, +0.0045 → 0.39, +0.0052 → 0.39, +0.0038 → still 0.39. Do not spend
further submissions on local deltas that size.

Both come from the [V11 run](docs/v11_execution_2026-09-06.md), which **failed its
predeclared promotion gate** (+0.005177 tuning PQ against a required +0.01), so
neither is **promoted**; they were exported by separate no-training notebooks after
each run's frozen control reproduced the accepted 0.38 CSV byte-for-byte.

The second submission resolved an open question: because `gt_centered_e4` ties 0.39
despite 0.000688 *lower* local tuning PQ, **the gain is continuation training, not
proposal centering**. Both of V11's substantive hypotheses — instance conditioning
and proposal centering — are therefore unsupported.

**Late on 6 September the two highest-ranked remaining bets were also run, and both
failed** (see the [experiment ledger](docs/experiments.md), E021 and E022):

- **E-C, cheap-feature proposal-quality ranker — NOT SUPPORTED.** At a matched
  accepted-proposal budget of 544, the best out-of-fold linear probe recovers one
  *fewer* true positive than the fixed 0.40 consensus gate. Only a CNN head reading
  real mask pixels remains untested.
- **E-D / V12, detector training resolution 1024 vs 1536 — gate FAILED.** Best arm
  `res1024_infer1536` scored tune PQ -0.000580 and audit -0.019284 against the
  frozen control; the 1536-trained arm lost at both inference resolutions.

**On 7 September E-A was run and it PASSED its gate (E023) — the first positive
result in three days, and the first of any kind to move recall.** Box `iou=0.3` on
the frozen detector recovers **58 of the 508 missed tune instances** (recall
0.5520 → 0.6032) at a *net-positive* PQ trade: tune **0.436726** (+0.003902 over
the accepted control) and audit **0.459054** (+0.008925). Every earlier recall
attempt gave the gain back in precision; this one does not. The frozen control
reproduced E015's accepted counts exactly. Nothing was submitted.

Three measured limits keep that in perspective: `agnostic_nms` is **inert** (the
detector is `single_cls=True`, so those arms are bit-identical and the 2x2 grid is
really 1x2); the gain requires the confidence floor to move 0.40 → 0.30, so it is a
flag+threshold interaction rather than a free flag win; and a paired bootstrap gives
a tune 95% CI of **[-0.005372, +0.013211]**, so the interval includes zero even
though the point estimate clears +0.003.

Next step is to compose E-A's `iou=0.3`/c0.30 inference with the V11
`proposal_centered_e4` refiner — the levers are orthogonal, and E-A is +0.004170
*ahead* of that checkpoint on audit while 0.001275 behind on tune — then spend one
submission as the tiebreaker. **Done: that is V13 / E024 above. The composition was
additive locally and cleared both bars, but the public score stayed at 0.39.**

The cheap inference-side levers are now exhausted. Remaining ranked options are
training-side: **E-C proper** (a CNN mask-scoring head reading real mask pixels,
aimed at the residual 448 tune FN), **E-F Focal-Tversky loss** (FN-weighted, now
more attractive since E-A proved the FN pool is reachable), then the more expensive
**E-E CLAHE** / **E-G PointRend** / **E-H fold ensemble** family. Recall remains
the largest bottleneck (448 tune FN after V13, from 508); see the
[PQ bottleneck decomposition](docs/pq_bottleneck_2026-09-06.md) and the
[0.55 optimization plan](docs/optimization_plan_055_2026-09-06.md).

The earlier [V10 package](submission_ready/v10_2026-09-06/README.md) was uploaded
on 6 September as **56044785** and scored **0.38**, tying the then-best; the
[V10 handoff](docs/handoff_2026-09-06.md) covers that step. Prior best was
**56026663** at 0.38.

[6 September score-0.50 research](docs/score_050_research_2026-09-06.md) compares
the live leaderboard, 17 public notebook sources, and fresh tuning-only
proposal diagnostics, with a prioritized training plan.

Current status and the V7/V8 model review are tracked in `docs/status_2026-09-05.md`.

The latest completed optimization is [V10 quality ordering and conservative V8 rescue](docs/v10_execution_2026-09-05.md).
Earlier COCO target rasterization correction, inference sweeps, and crop postprocessing
are recorded in [docs/optimization_2026-09-05.md](docs/optimization_2026-09-05.md).

The working strategy is a staged, reproducible pipeline:

1. Build a strong classical/CV baseline to validate RLE encoding and submission format.
2. Train a lightweight instance segmentation model using MAGFiLO masks.
3. Improve panoptic quality with post-processing that reduces fragmentation and over-merging.
4. Keep every experiment documented in `docs/experiments.md`.

Current conclusion: prioritize a detector-guided segmentation pipeline, not a plain semantic U-Net. The challenge is scored with Panoptic Quality, so instance separation and merge/split behavior matter as much as mask overlap.

## Setup

```bash
conda env create -f environment.yml
conda activate solar-filament-2026
```

## Data

Accept the Kaggle competition rules in the browser first, then run:

```bash
kaggle competitions download -c filament-segmentation-2026 -p data/raw
python scripts/unpack_data.py
```

Current local snapshot:

- `707` train images
- `180` test images
- `8,199` annotation entries
- `296` train files with duplicate annotator labels

## Baseline Submission

```bash
python -m src.infer_classical \
  --input-dir data/raw/test \
  --output outputs/submission_classical.csv
```

The submission CSV contains:

- `filament_id`
- `segmentation_rle`

RLE counts are encoded with `pycocotools` for 2048 x 2048 masks.

## Repository Status

GitHub: [Abm32/solar-filament-2026](https://github.com/Abm32/solar-filament-2026),
branch `main`. Code, experiment documentation, tests, and the prepared V10
submission package are versioned. Raw data, model weights, bulk run outputs,
credentials, and the downloaded third-party payload notebook remain excluded.

The prepared CSV can be uploaded without rerunning inference. Reproducing the
experiments additionally requires the data, checkpoints, and caches described
in the [handoff](docs/handoff_2026-09-06.md).

## Training Skeleton

```bash
python scripts/prepare_yolo_dataset.py --config configs/yolo_seg.yaml
python scripts/train_yolo_seg.py --config configs/yolo_seg.yaml
python -m src.train --config configs/baseline_unetplusplus.yaml
```

The YOLO segmentation path is the main track. The `src.train` semantic path is kept as a controlled baseline and expects prepared binary masks under `data/processed`.
