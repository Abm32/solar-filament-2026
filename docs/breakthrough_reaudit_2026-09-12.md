# Breakthrough re-audit after V25 — 2026-09-12

## Result first

The 0.39 plateau is not explained by a wrong local metric, obvious hidden-domain shift, stale organizer notebook, or a fresh public high-scoring implementation. The highest-value remaining experiment is native sliced proposal discovery (V26), because it attacks a measured untested mechanism: small filaments are downsampled before every deployed detector sees them, while 226/228 exact no-proposal misses fit wholly in their deterministic native 1024 owner tile.

## Metric audit

The current organizer `azimahmadzadeh/self-evaluation-notebook` is byte-identical to the archived project copy:

`ae7c9445b4eee2b77b216df5c9772fdf92fda15611e98abd7ef5f02a4e380801`

It evaluates each annotator-image independently, computes all pairwise mask IoUs, declares hits at strict `IoU > 0.5`, then globally aggregates IoU sum, hit count, unmatched prediction columns, and unmatched GT rows. For disjoint predictions and threshold >0.5, one-to-one uniqueness is guaranteed geometrically, so this is equivalent to the project's canonical TP/FP/FN arithmetic.

The July leaderboard vulnerability—five omitted-image empty masks scoring 1.00—was a responsible disclosure against the pre-reset evaluator. The competition changed its primary metric to PQ on 7 August. It does not justify incomplete/empty submissions now; current valid V17/V21 submissions and notebook hash provide no evidence that exploit remains active.

## Alternative aggregation audit

Historical candidates E015, V10, V11 and V17 were rescored from per-annotator diagnostics under:

- organizer/global instance PQ;
- annotator-record macro PQ;
- inverse-annotator global PQ;
- physical-image macro PQ;
- mean annotator PQ within physical images;
- test-station reweighted PQ;
- test-year reweighted PQ.

V17 remains locally best under every aggregation. Metric weighting does not identify a hidden winner discarded by canonical selection. The five public labels are too censored at two decimals for meaningful correlation claims, but global PQ is at least as aligned with their ordering as the alternatives.

## Domain audit

Physical image distributions:

| split | files | stations | years |
| --- | ---: | --- | --- |
| train | 566 | all B/C/L/M/T/U | 2011-2022 |
| tune | 94 | all six | 2011-2022 except sparse years |
| audit | 47 | all six | 2011-2021 |
| test | 180 | all six, 26-34 each | 2011-2022 |

Downsampled intensity mean/std and in-disk p10/p50/p90 are closely aligned. Test has somewhat broader low-percentile variation but no large station/time/intensity shift explaining why local improvements disappear publicly.

## Fresh public notebook audit

Authenticated Kaggle listing contains two notebooks newer than the previous research snapshot:

- `deepmalyakoley/solar-filament-segmentation-characterization` (Sep 8): EDA, rasterization, spine visualization, grouped-fold advice, and proposed 512 tiling. It does not train a model or produce a scored method. Its rasterizer also uses only the first polygon component in an annotation.
- `trishyagupta/sfc-mago` (Sep 6): disk masking, CLAHE, and tile generation. It does not train/evaluate an instance model; it collapses annotations semantically and selects one annotator per filename.

No newly trained reproducible high scorer was found. Current leaderboard leaders at 0.55-0.56 still do not expose validated checkpoints/methods.

## Native-resolution opportunity

Every deployed V6-V17 detector uses full-frame input <=1024 for 2048 images. V18 attempted high-resolution full-frame training but OOMed; no native tile inference was run.

For a 3×3 grid of 1024 tiles at starts 0/512/1024:

- 1,130/1,134 tune GT fit wholly inside their nearest-center owner tile;
- 1,036/1,134 fit wholly in multiple tiles;
- 226/228 exact V6 no-raw-proposal misses fit their owner tile;
- 213/228 fit multiple tiles;
- missed median area is 826.5 px;
- missed bbox p90 is about 197×171 px.

The four owner-noncontained GTs are large; a full-image branch covers that failure mode.

SAHI (Akyon et al., 2022) provides primary external evidence: +5.1-6.8 AP from sliced inference alone and larger cumulative gains with sliced fine-tuning on high-resolution small-object datasets. It specifically recommends overlap plus full inference to balance small and large objects. V26 translates that mechanism into mask proposals and prices it with strict solar PQ.

## Ranked remaining mechanisms

1. **V26 native sliced proposal discovery — run next.** No training, direct 226/228 geometric coverage of the measured invisible pool, primary paper support, four-GPU tune-first execution.
2. **Sliced fine-tuning — conditional only.** Consider only if V26 inference materially improves raw recall but final PQ is limited by tile score calibration. It requires a separate frozen dataset/training design.
3. **Latitude/spine auxiliary supervision — low prior.** Public EDA and solar physics support spatial priors, but category/spine handling does not directly explain the measured proposal miss pool and risks optimizing training-only metadata absent from test labels.
4. **More missing-rater handling — closed.** V24 positive union and V25 uncertain-negative weighting recovered the same seven raw instances and both lowered final PQ.
5. **More generic architecture/preprocessing swaps — closed without a new mechanism.** Mask2Former, CondInst, PointRend, full refit, CLAHE/radial preprocessing, refiner capacity, threshold/ranker, ensemble, and detector resolution have measured negative or resource outcomes.

## Decision

Freeze and implement V26 exactly as `docs/v26_design_2026-09-12.md`. Preserve V17 submission `56103401` at public 0.39 until a candidate clears tune and audit gates. Do not submit based on sliced raw recall alone.
