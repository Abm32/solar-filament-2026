# Clean research paths beyond the 0.55 barrier — 2026-09-13

## Executive decision

The next experiment should be **native-tile, expert-spine-supervised skeleton-aware distance transform (SDT)**, not another YOLO/refiner iteration. It is the only high-ranked option that simultaneously:

1. uses a strong source of supervision no project model has consumed: a spine polyline for every one of the 8,199 competition annotations;
2. attacks thin-mask connectivity, touching-instance separation, and proposal recall in one representation;
3. avoids box/proposal assumptions that repeatedly failed on elongated filaments;
4. is feasible on four T4s using 1024 native tiles; and
5. can be evaluated with the exact organizer PQ before audit or test access.

All numerical gains below are **scenario estimates, not measured competition results**. Cross-domain papers do not report this competition's PQ.

## Important leaderboard integrity finding

The live leaderboard now has three teams at 0.56 and many at 0.55. This cluster is not reliable evidence that a clean model architecture reaches 0.55.

The public Kaggle dataset `esairlab/magfilo-v1-0-segmentation-of-solar-filaments` exposes the original full MAGFiLO annotation release. A filename-only audit found:

- public release: 704 labeled physical files / 8,212 annotations;
- competition train overlap: 565 of 707 filenames;
- **competition test overlap: 139 of 180 filenames**.

The public annotation file was used only to compute these counts and then deleted. It must not be used for training, validation, scoring, pseudo-labeling, model selection, ensembling, or submission. Report: `outputs/research_2026-09-13/public_magfilo_audit/overlap_report.json`.

This does not prove that any specific team used leaked labels. It does mean leaderboard scores and public solar checkpoints cannot establish clean method quality unless their provenance proves every competition test filename was excluded. The public “LB 0.70” handoff is also a pre-7-August-reset score and is not comparable to the current PQ leaderboard.

## Measured problem geometry

From official competition annotations only:

- all **8,199/8,199** annotations have expert spine polylines;
- canonical fit split: 566 physical files / 937 annotator records / 6,606 instances;
- median spine length: **111.3 px**; p90: 285.2 px; p99: 627.5 px;
- effective width (`area / spine length`) median: **11.1 px**; p90: 21.0 px;
- median bounding-box diagonal: 111.8 px; p99: 574.1 px;
- exact same-record instance overlap occurs in only **5/1,154 records** and 5/36,064 instance pairs.

No project Python training/inference pipeline reads the `spine` field. Existing public centerline notebooks compute a skeleton from a merged semantic mask, which discards expert instance identity and is not equivalent.

These measurements support one global SDT energy field per annotator record. Orientation-conditioned overlap handling is unnecessary as the primary representation because true same-record overlap is exceptionally rare.

Artifacts: `outputs/research_2026-09-13/spine_geometry.json` and `instance_overlap_geometry.json`.

## Why incremental improvement is insufficient

V17 tune is PQ 0.448703 with 704 TP / 306 FP / 430 FN and SQ about 0.683. At assumed SQ 0.69:

- local PQ 0.50 requires about 818 TP if FP stays 306, or 758 TP if FP falls to 200;
- local PQ 0.55 requires about 818 TP even if FP falls to 100;
- the prior censored-score affine bound of local 0.6269 would require about 944 TP even at zero FP.

Therefore another +0.005 calibration improvement cannot solve the task. A viable method must jointly recover many missing instances and avoid false-positive inflation.

## Ranked clean approaches

Ranking weights mechanism fit, primary evidence, T4 feasibility, and novelty relative to V17–V30. Gain ranges are deliberately speculative.

| Rank | Approach | Mechanism | Speculative tune-PQ delta vs V17 | T4 feasibility | Main risk |
| ---: | --- | --- | ---: | --- | --- |
| **1** | **Expert-spine SDT on native tiles** | Energy is maximal along each expert spine and minimal at instance boundaries; global seeded watershed reconstructs instances | **+0.025 to +0.080** | High | Energy decoding may create FP or fragmented seeds |
| **2** | Omnipose distance + flow | Boundary, distance field, and its gradient; suppressed Euler integration handles elongated/branched objects | +0.015 to +0.070 | High | Package refactor is WIP; very long objects need careful tiling |
| **3** | Orientation-to-overpass | Orientation branches separate crossing/nearby filaments, followed by terminus pairing | +0.010 to +0.050 | Medium | Exact overlaps are rare; original reconnection uses MATLAB and is slow |
| **4** | MaskDINO or RF-DETR-Seg | Detection-aware query initialization plus high-resolution mask embeddings | 0.000 to +0.060 | Medium-low | MaskDINO's published setup uses 40 GB A100s; still proposal/query based |
| **5** | TARDIS FNet + DIST-linear | Semantic mask → skeleton point cloud → learned pairwise graph → graph cut | -0.010 to +0.040 | Medium-low | Solar semantic errors propagate; public 2D filament path is less mature |

### 1. Skeleton-aware distance transform

Lin et al., *Structure-Preserving Instance Segmentation via Skeleton-Aware Distance Transform* (MICCAI 2023, arXiv:2310.05262), define

`E(x) = (d(x,boundary) / (d(x,spine) + d(x,boundary)))^alpha`, with `alpha=0.8`.

The representation preserves connectivity while keeping boundaries distinguishable. Quantized 10-bin classification outperformed regression in their ablation and achieved state-of-the-art gland instance segmentation, including IoU>0.5 F1. Their decoder thresholds high-energy skeletons into seeds and applies watershed on the reversed energy.

This competition is unusually well suited because it provides expert spines rather than requiring unstable morphological skeletonization. Median width is only 11 px, exactly where a boundary-only target can destroy connectivity.

### 2. Omnipose

Cutler et al., *Omnipose* (Nature Methods 2022, DOI 10.1038/s41592-022-01639-4), combine boundary probability, a smooth eikonal distance field, and the gradient of that field. Suppressed Euler integration prevents pixels from fragmenting along elongated-object skeletons. The paper reports large improvements for elongated and branched cells, including non-bacterial subjects.

Omnipose is the best fallback if SDT target decoding works but the discrete watershed is unstable. Its effective scale is driven by object width rather than length, matching the competition's ~11 px median width. Implementation risk is higher because the public repository is undergoing a refactor.

### 3. Orientation-aware overpass reconstruction

Liu et al., *Intersection to Overpass* (CVPRW 2019, DOI 10.1109/CVPRW.2019.00021), predict six orientation-associated layers and pair termini across layers. Their method reached F1 0.716 versus 0.27–0.29 for prior microtubule methods and is MIT licensed.

It is especially relevant to close parallel filaments and crossing-like structures, but lower-ranked because exact same-record overlaps are rare and the released reconnection stage depends on MATLAB. It is a conditional decoder if SDT diagnostics show merge concentration at near-crossings.

### 4. MaskDINO / RF-DETR-Seg

MaskDINO improves over Mask2Former by adding detection-aware query selection, mask-enhanced box initialization, denoising, and joint box/mask matching. The paper reports +2.6 to +2.7 mask AP over Mask2Former and 59.4 COCO PQ for its large model. This is materially different from the failed V19 Mask2Former, but its official large setup is expensive and still relies on object queries.

RF-DETR-Seg has a more practical DINOv2-based implementation and a downloaded public competition notebook, but no valid canonical-PQ result. Either is a detector-family fallback, not the first experiment.

### 5. TARDIS

TARDIS converts a semantic mask into a skeleton point cloud and uses the SO(n)-equivariant DIST transformer to predict pairwise connections before graph cut. It reports strong filament/membrane instance recovery and is MIT licensed. However, its current 2D filament support is less turnkey, and any missed semantic pixels become graph gaps. It is better suited after a strong semantic/spine model exists.

## Approaches not recommended now

- **Public full MAGFiLO labels or solar checkpoints without exclusion manifests:** contamination risk; forbidden for this project.
- **Plain semantic U-Net, CLAHE, radial preprocessing, more YOLO epochs/resolution, confidence sweeps, proposal union, V17 refiner capacity:** already tested or mechanistically closed.
- **clDice alone:** connectivity loss does not produce instance identity. Use it only as an auxiliary diagnostic/ablation.
- **SAM/SAM2 as primary:** low-contrast thin structures are a known hard case; prompts still require a detector and do not solve recognition.
- **More representative/consensus target variants:** V24, V25, V30 already show the recall/FP trade-off.
- **TARDIS/FFN/PatchPerPix as first run:** complex assembly and long inference with weaker direct evidence under this 2D PQ metric.

## Recommendation

Execute V31 exactly as `docs/v31_design_2026-09-13.md`. It is a four-arm matched experiment: standard distance transform versus expert-spine SDT, two seeds each. Fit only the canonical competition training split, blend native 1024 tile logits globally, and decode with a predeclared nine-cell grid. Open audit only if SDT reaches tune PQ >=0.50 and replicates above V17. Do not use the public MAGFiLO release, inspect audit early, infer test, or upload a candidate that fails those gates.

## Primary sources

- MAGFiLO dataset paper: https://doi.org/10.1038/s41597-024-03876-y
- Skeleton-aware distance transform: https://arxiv.org/abs/2310.05262
- Omnipose: https://doi.org/10.1038/s41592-022-01639-4
- Intersection to Overpass: https://doi.org/10.1109/CVPRW.2019.00021
- MaskDINO: https://arxiv.org/abs/2206.02777
- TARDIS: https://pmc.ncbi.nlm.nih.gov/articles/PMC11702698/
- FISBe thin-filament benchmark: https://arxiv.org/abs/2404.00130
- clDice: https://arxiv.org/abs/2003.07311
