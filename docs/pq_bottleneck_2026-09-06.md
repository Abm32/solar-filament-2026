# PQ bottleneck decomposition — 6 September 2026

Quantitative decomposition of our own pipeline's PQ loss on the 94-file tuning
set, plus what the public 0.55 cluster actually represents. Reproduce with:

```bash
.venv/bin/python scripts/analyze_pq_bottleneck.py
```

Outputs: `outputs/pq_bottleneck_2026-09-06/pq_bottleneck_summary.json`.
No inference is re-run; only existing local artifacts are consumed.

## What was computed and verified

- Actual selected-model counts aggregated from
  `outputs/kaggle_v11_full/inference_image_diagnostics.csv`, candidate
  `v11_proposal_centered_e4_refined_constrained`, group `tune`.
  Reproduced the reference **exactly, zero discrepancy**:
  IoU sum `431.21160644179463`, TP `629`, FP `206`, FN `505`, **PQ 0.43800**.
- V10 tune cross-check reproduced exactly: IoU sum `430.3746937713222`,
  TP `632`, FP `217`, FN `502`.
- Per-instance geometry from `outputs/research_2026-09-06/proposal_diagnostics.csv`
  (1134 GT instances = 629 actual TP + 505 actual FN).

### Critical data caveat (stated, not hidden)

The per-image diagnostics store only per-image **sums** (`iou_sum, tp, fp, fn`),
not per-matched-instance IoU. The deployed refined-mask IoU distribution is
therefore **not stored locally** (no torch here to re-infer). Every per-instance
IoU result below uses `v6_accepted_iou` — the IoU of the best proposal that
survives grouping + the 0.40 cutoff and is fed to the refiner.

The refiner improves on that geometry: **563 oracle-accepted matches (IoU>0.5)
vs 629 actual TP** — a +66 net recovery. So `v6_accepted_iou` is a documented
**lower bound**; per-instance FN/TP proxies overstate FN by ~66 and are used for
relative attribution only, never as the actual PQ.

## Q1 — FN and lost-PQ by GT area decile

FN concentrate sharply in small instances. FN-proxy rate by area decile:

| Decile | Area range (px) | n | FN rate |
| ---: | --- | ---: | ---: |
| 0 | 103–423 | 115 | 0.78 |
| 1 | 426–596 | 112 | 0.72 |
| 2 | 597–800 | 114 | 0.56 |
| 3 | 802–997 | 114 | 0.51 |
| 4 | 1001–1243 | 112 | 0.52 |
| 5 | 1245–1576 | 113 | 0.42 |
| 6 | 1581–2118 | 114 | 0.47 |
| 7 | 2119–2852 | 113 | 0.30 |
| 8 | 2860–4422 | 113 | 0.33 |
| 9 | 4444–37739 | 114 | 0.41 |

- **41.2%** of all FN come from the bottom 30% by area; only **20.7%** from the top 30%.
- **36.2%** of the total IoU headroom (Σ(1−accepted_iou)) sits in the bottom 30% by area.

## Q2 — Matched-IoU distribution and the SQ→PQ counterfactual

Distribution over `v6_accepted_iou` (oracle accepted proposal geometry):

| Band | 0.0–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| n | 458 | 113 | 199 | 225 | 118 | 21 | 0 |

- **Fragile TPs (0.50–0.60): 199** — one small perturbation from dropping below threshold.
- **Near-misses (0.40–0.50): 113** — a small SQ gain converts these to TP.
- Actual mean matched IoU (from real V11 sums) = **0.6856**.

Counterfactuals priced against the **actual** V11 counts (algebraic, not forecasts):

- Uniform matched-IoU lift on existing TPs: **+0.05 → PQ 0.470**, **+0.10 → PQ 0.502**.
- Convert all 113 near-misses to TP at IoU 0.55 (FP fixed): **PQ 0.474** (TP 742, FN 392).

Note the near-miss pool is measured on oracle geometry; the actual refiner has
already converted ~66 of the sub-0.5 proposals, so the deployable near-miss pool
is smaller than 113.

## Q3 — FN cause funnel (nested raw → grouped → accepted)

Of the 571 oracle FN-proxy instances (accepted IoU ≤ 0.5):

| Cause | Count |
| --- | ---: |
| (a) No raw proposal at all (v6_raw_iou ≤ 0.5) | **228** |
| (b) Lost by grouping (raw>0.5, grouped≤0.5) | **65** |
| (c) Lost by the 0.40 acceptance cutoff (grouped>0.5, accepted≤0.5) | **278** |
| a+b+c (checks against total 571) | 571 ✓ |
| (d) Support geometry insufficient (support_coverage ≤ 0.5) among FN | 388 |

The **acceptance cutoff (278)** and **missing raw proposals (228)** dominate.
The cutoff loss is the single largest recoverable category: these instances
already have a grouped proposal with IoU>0.5 that the confidence threshold
discards. Lowering the threshold alone regressed before (it adds FP), which is
why a learned proposal-quality decision is the recommended lever.

## Q4 — Category and annotator-duplicate structure

| Category | n | Raw match rate | Accepted match rate | Median area |
| --- | ---: | ---: | ---: | ---: |
| 1 | 337 | 0.810 | 0.558 | 1558 |
| 2 | 328 | 0.845 | 0.595 | 1510 |
| 3 (Unidentifiable) | 469 | 0.759 | **0.384** | 945 |

Yes — we systematically lose category 3. But its **raw** match rate (0.759) is
close to categories 1/2; the gap opens **downstream** (grouping + cutoff) and it
skews small (median 945 px). So category 3 loss is mostly the same
small-instance + cutoff problem, not a distinct category-3 detector failure.
Dropping category 3 is not justified (chirality label, not annotation validity).

Annotator duplication: **35 of 94 files have >1 annotator**;
**632 / 1134 (55.7%) GT instances** live in multi-annotator files. PQ is scored
per annotator entry and never unions masks, so those files contribute their
GT — and their FN/TP — once per annotator to the global denominator. This is the
metric's design, not a pipeline bug, but it means small-instance FN in
multi-annotator files is effectively double-weighted.

## Q5 — Elongation vs matched IoU (hypothesis rejected)

Computed per-instance elongation from canonical GT masks (bbox aspect ratio and
PCA major/minor ratio), merged all 1134 instances.

- corr(PCA elongation, accepted IoU) = **+0.08**
- corr(bbox aspect ratio, accepted IoU) = **−0.08**
- corr(log area, accepted IoU) = **+0.32**
- **Partial** corr(elongation, IoU | log area) = **+0.04** (negligible)

Match rate by PCA-elongation bucket actually *rises* with elongation
(1–2: 33%, 3–5: 53%, 8+: 52%) because more elongated filaments are larger.
**Thin elongated filaments are not systematically worse; small ones are.**
Area is the driver; elongation adds nothing once area is controlled.

## Q6 — Algebraic PQ ceilings (not achievable targets)

Holding all other quantities fixed at the actual V11 tune counts:

| Ceiling | PQ | Assumption |
| --- | ---: | --- |
| Baseline (actual V11) | 0.438 | — |
| Perfect FP filter | **0.489** | Delete all 206 FP; keep every TP + IoU |
| Perfect masks, counts fixed | 0.639 | Every existing TP → IoU 1.0 |
| Full GT recovery @ current mean IoU | **0.628** | Every FN → TP at IoU 0.686; FP fixed |
| Perfect FP filter AND full recovery | 0.686 | Both of the above |

Recall recovery (0.628) dwarfs FP filtering (0.489). FP is a bounded lever
(≈ +0.05 PQ max); recall is the dominant one (≈ +0.19 PQ headroom).

## What the public 0.55 cluster actually is

Per the existing research (`docs/score_050_research_2026-09-06.md`,
`public_notebook_lamhuy_055_review_2026-09-05.md`): the 20-team 0.55 cluster is
dominated by an **embedded/compressed precomputed submission payload**
(`CHAMPION_PAYLOAD` in the lamhuy notebook) whose provenance is unverified — not
a reproducible trained model. HDJoJo's public 0.55 uses YOLOv8l seg at
`imgsz=2048` behind a **private** checkpoint; the recipe is not published. The
one reproducible, legitimate signal is **recall**: the 0.55 payload emits 1,342
instances across more images vs our ~1,094, with a similar area distribution.
This matches our own decomposition exactly — we are recall-limited, not
mask-quality-limited or FP-limited.

## Single change that would most increase PQ

**Recover missed instances — specifically the 278 grouped-but-cutoff-rejected
proposals — via a learned proposal-quality/IoU head instead of the fixed 0.40
confidence cutoff.**

Rationale, entirely from the numbers above:
- Recall ceiling (+0.19 PQ) is ~4× the FP ceiling (+0.05 PQ) and larger than any
  plausible SQ gain (+0.10 IoU → +0.064 PQ).
- The 278 cutoff losses already have a grouped proposal with IoU>0.5 — they are
  the most directly recoverable FN population. A naive threshold drop failed
  because it adds FP; a learned scorer targets the same pool without that cost.
- These losses concentrate in small instances (bottom 30% area = 41% of FN) and
  category 3, so the recovery mechanism should be size-aware.

Second-order lever: because 199 TPs are fragile (IoU 0.50–0.60), a modest
mask-quality (SQ) improvement compounds with recovery and guards existing TPs.
Elongation-specific handling is **not** warranted (Q5).
