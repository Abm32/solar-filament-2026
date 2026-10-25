# V69 error analysis: where the remaining PQ goes (2026-10-10)

This is CPU-only analysis on cached OOF predictions. No GPU was used, nothing was trained, and nothing was submitted. No test labels were read.
It is post-hoc and diagnostic. It does not authorise anything, and every lever below needs its own pre-registration.

## Sources
- **Direct detector:** p2first 5-fold OOF cache (`outputs/v61_oof_cache/p2first/fold*`), all 707 files, 1154 annotator records, 8199 GT, conf cut 0.35. The pipeline is the V17 scorer: flip consensus, then pixel ownership, then per-record `pq_counts`.
  - The collector asserts that every file's counts equal `v61_toolkit.score_file`. It reproduces PQ **0.4060** exactly (TP 4737 / FP 2387 / FN 3462). The baseline recipe reproduces 0.3856.
- **Real pipeline:** p2first OOF proposals go through the r3 refiner (`outputs/v63/r3_refiner.pt`, trained on the 566 fit files) and are scored at the V17 cell t0.45/c0.30. This covers the 141 canonical held-out files.
  - It runs on CPU by monkeypatching the CUDA-only loader. It reproduces `outputs/v63/r3.json` **per file** exactly (asserted): PQ **0.4595** (TP 1020 / FP 448 / FN 573).
- **Scripts:** `scripts/analysis_v69_{errors,report,extras,misc,fusion,refined_probe,refined_errors}.py`.
- **Outputs:** `outputs/v69_analysis/{p2first,baseline}/` (pickles, `report.txt`, `report.json`, `extras.json`, `fusion*.json`, `refined_probe.json`, `refined_errors.json`).

## 0. How the metric treats multiple annotators (this drives everything)
Each annotator record is its own scoring unit, in both the host `pq_score` (one `annotator_image` per row) and our `pq_counts`. The same prediction set is matched (IoU > 0.5) against every record of a file independently. TP, FP and FN are then summed over all records and PQ is computed once.

This has three consequences:
1. A file with 3 records weighs 3x. Each prediction is counted up to 3 times.
2. A filament drawn by only k of n annotators, predicted at IoU s, yields k TP and (n-k) FP. Leaving it out yields k FN.
   - Predicting it helps when roughly **k/n > PQ/(2·s) ≈ 0.41/(2·0.66) ≈ 0.31**. So anything at least 1 of 3 annotators would draw is (barely) worth predicting.
   - This means the ideal score is a calibrated "probability that a random annotator draws this filament", not "is this a filament".
3. Some FP and FN are structurally unavoidable: they are disagreements between annotators.

## 1. PQ decomposition (Q1)
| pipeline | files | PQ | SQ (mean matched IoU) | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| direct p2first @0.35 | 707 | 0.4060 | 0.6567 | 0.6183 | 4737 | 2387 | 3462 |
| direct p2first @0.35 | the 141 | 0.4162 | 0.6609 | – | 925 | 420 | 668 |
| p2first + r3 refiner (t0.45/c0.30) | the 141 | 0.4595 | 0.6894 | 0.666 | 1020 | 448 | 573 |

What-if scenarios. Each FN that is "halved" is converted to a TP at the current SQ:

| scenario | direct (707) | refined (141) |
|---|---|---|
| now | 0.406 | 0.460 |
| all TP IoU = 0.9 (perfect SQ) | 0.556 (+0.150) | 0.600 (+0.140) |
| SQ +0.05 | 0.437 (+0.031) | – |
| FN halved | 0.498 (+0.092) | 0.538 (+0.079) |
| FP halved | 0.440 (+0.034) | 0.496 (+0.036) |
| FP = 0 | 0.481 | – |

On paper, perfect SQ is worth the most. In practice it is **not reachable**:
- **Human boundary agreement is the SQ ceiling.** Human-vs-human SQ on matched pairs is only **0.634** pairwise and **0.630** leave-one-out (§3). The refined pipeline's SQ of 0.689 is already above both.
- **TP IoU distribution** (direct): 0.5–0.6: 1347, 0.6–0.7: 1868, 0.7–0.8: 1317, 0.8–0.9: 205, >0.9: 0.

**FN is the biggest realistic pool: halving it is worth about +0.08 to +0.09, versus about +0.035 for halving FP.**

## 2. FN and FP breakdown (Q2)

### FN categories (hierarchical, each FN assigned to its first matching category)
| category | direct 707 | share | refined 141 | share |
|---|---|---|---|---|
| a. **score**: a proposal (owned or refined) below the cut has IoU > 0.5 | 948 | 27.4% | 180 | 31.4% |
| b. ownership-clipped: a consensus mask at or above the cut matched, but ownership cut it | 85 | 2.5% | – | – |
| c. **pool-only**: some single flip-view mask has IoU > 0.5, but consensus kept another mask | 987 | 28.5% | 175 | 30.5% |
| d. near-miss: a kept prediction has IoU 0.3–0.5 | 480 | 13.9% | 72 | 12.6% |
| e. near-miss only below the cut or in views | 466 | 13.5% | 60 | 10.5% |
| f. weak overlap 0.05–0.3 | 217 | 6.3% | 32 | 5.6% |
| g. **no proposal at all** (IoU < 0.05 everywhere) | 279 | 8.1% | 54 | 9.4% |

Grouped:
- **Score or selection problems (a+b+c): 58% direct, 62% refined.** A mask good enough to be a TP already exists in the candidate pool.
- **Mask-quality near-misses (d+e): 27% direct, 23% refined.**
- **True recall misses (f+g): 14–15%.**

Detail on category c (direct):
- 448 of 987 have a view-mask score above 0.4.
- 527 have a kept prediction at IoU 0.3–0.5. In those cases the consensus "strongest mask" representative was the worse of the group's masks.

Near-miss shape (category d, direct):
- 240 predictions cover the GT but are too fat or too long.
- 176 are boundary disagreements.
- 64 predictions sit inside the GT (under-segmented).

Direct-detector recall by GT property, with each bin's share of all FN:

| factor | worst bin | best bin |
|---|---|---|
| area | ≤500 px: recall 0.315 (869 FN, 25%) | 2–8k px: 0.68 |
| width (area/length) | ≤8.3 px: 0.423, matched IoU 0.598 | >17.5 px: 0.657, matched IoU 0.698 |
| contrast vs a 10 px ring | lowest quintile: 0.355 (1057 FN, **30.5%**) | top quintile: 0.729 |
| limb distance r/R | 0.85–0.95: 0.374 (268 FN, 7.7%) | <0.7: about 0.60 |
| chirality label | "Unidentifiable" (cat 3): 0.476 (1610 FN, **46.5%**) | Left/Right: 0.63–0.65 |
| GT per record | 15–25: 0.470 | ≤5: 0.630 |
| **agreement** (multi-record files) | drawn by **one** annotator only: **0.293** (n = 1855) | matched in another record: **0.783** (n = 3138) |

Agreement is the strongest single factor:
- On multi-record files, **66% of FN (1312 / 1992) are filaments only one annotator drew**.
- Among "no proposal" FN on those files, the share is 95% (127 / 133). True recall misses are almost entirely idiosyncratic annotations.
- In the refined pipeline the same split holds: recall is 0.375 on single-annotator GT versus 0.818 on agreed GT.

### FP categories (per record; also flagged when the FP matches a GT in ANOTHER record of the same file at IoU > 0.5)
| category | direct 707 | of which match another record | refined 141 | of which match another record |
|---|---|---|---|---|
| merge (covers ≥30% of ≥2 GTs) | 155 (6.5%) | 61 | 27 | 11 |
| near-miss of a GT (IoU 0.3–0.5) | 937 (39.3%) | 381 | 148 | 59 |
| fragment or duplicate of a **matched** GT | 24 (1.0%) | 2 | 89 (both fragment kinds) | 19 |
| fragment of an **unmatched** GT | 275 (11.5%) | 90 | (included above) | |
| weak overlap | 145 (6.1%) | 18 | 14 | 3 |
| no overlap with this record's GT | 851 (35.7%) | 298 | 170 | 57 |
| **total** | 2387 | **850 (35.6%)** | 448 | **149 (33.3%)** |

- **About one third of all FP are another annotator's TP.** They are the price of predicting filaments that some, but not all, annotators drew.
- Of the 851 no-overlap FP, 392 overlap some other record's GT. Only **459 (19% of FP) are pure hallucinations**.
- FP scores are concentrated at the bottom: 837 FP score 0.35–0.45, against 503 TP in that range.
- Baseline recipe for comparison: the same pattern with FN shares 26 / 2 / 28 / 15 / 16 / 6 / 7%. p2first's gain over baseline is spread across categories.

## 3. Annotator disagreement and the human ceiling (Q3)
Files with ≥2 records: 296 files (145 with 2 records, 151 with 3).

| predictor (scored against a held-out annotator record) | PQ | SQ |
|---|---|---|
| another single annotator (1196 ordered pairs) | **0.340** | 0.634 |
| leave-one-out: union of the other annotators | **0.345** | 0.630 |
| leave-one-out: only filaments both other annotators agree on (3-record files) | 0.326 | – |
| **model (direct p2first) on the same records** | **0.411** | 0.656 |
| model on single-record files | 0.398 | – |

- **The direct detector already beats a held-out human annotator by about 0.066 PQ.** The refined pipeline beats one by more.
- The "record-as-prediction" oracle reaches 0.605 mean and 0.631 best. That figure includes scoring a record against itself, so it is not attainable.
- The realistic ceiling is set by annotator style, not by vision.
- **Style varies a lot by annotator record ID:**
  - Direct PQ by record ID ranges from **0.32** (010401, 73 records, 3.2 FP/record) to **0.50** (020403, 42 records).
  - GT per record ranges from 4.3 to 10.5 across IDs.
  - Prefix group "01" holds 45% of records and scores PQ 0.386; the other prefix groups score 0.39–0.44.
- **Matches to another record:** 850 FP (35.6%) match a GT in another record of the same file (refined: 149, 33%).

## 4. Fragmentation and merging (Q4), direct
- **GT split** across ≥2 kept predictions (each prediction ≥50% inside the GT and covering ≥10% of it): **122 GT (1.5%)**, of which 116 are FN.
- **Merges**: 155 FP blobs cover ≥2 GTs, and 22 FN are swallowed by a prediction that matched a neighbouring GT.
- **Fragment FP**: 299 (24 of a matched GT, 275 of an unmatched GT). In the refined pipeline fragments are 89/448 = 20% of FP, which is up from about 12.5% in the direct detector.
- Topology errors are a minor share of the loss, matching the earlier V58 estimate of about 2.5%.

## 5. Per-file variance (Q5), direct
- **The loss is diffuse.** The worst 10% of files hold 27.6% of the "perfect-file" gain, the worst 20% hold 45%, and the worst 50% hold 78%. The single worst file, 20110914063134Lh (3 records, 58 GT, PQ 0.25), is worth only +0.005.
- The worst files are crowded 3-record images with about 40 GT.
- **Image-quality covariates are not predictive:**
  - Spearman correlation of per-file PQ with disk std is −0.03, with Laplacian sharpness −0.03, with disk brightness +0.06, and with year −0.02.
  - The number of kept predictions is the strongest, at +0.12.
  - PQ by quintile of disk std, sharpness or brightness stays within 0.38–0.44 with no trend.
- **By site:** U (Udaipur) 0.424 is best; M (Mauna Loa) 0.391 is worst; B, C, L and T score 0.405–0.408.
- **By year:** 2011–2018 score 0.40–0.43. 2019–2021 score 0.34–0.38, but those years have only 9–28 files each.
- **By number of records:** 1 record 0.398, 2 records 0.406, 3 records 0.414.
- **By mean GT per record:** files with more than 10 GT per record have recall 0.51–0.55 versus about 0.61 below that. Crowded discs lose recall, not precision.
- **The dominant file-level factor is which annotator drew the record** (§3), not image conditions.

## 6. Probes run during the analysis (post-hoc, diagnostic)
**Direct-detector masks are about 1 px too fat.** On all 707 files at 0.35:

| change to the masks | ΔPQ | 95% CI |
|---|---|---|
| keep pixels in ≥75% of a consensus group's view masks (vote75) | **+0.0199** | [+0.017, +0.023] |
| 1 px erosion of the strongest mask | **+0.0183** | [+0.014, +0.023] |
| intersection of all view masks (vote100) | +0.015 | – |
| score-weighted 50% vote | +0.011 | – |
| plain 50% vote | +0.000 | – |
| 2 px erosion | −0.094 | – |

**The r3 refiner fully absorbs the 1 px fattening.** On the real pipeline (141 files, V17 cell):

| change | ΔPQ | 95% CI | P(Δ > 0) |
|---|---|---|---|
| vote75 masks fed to the refiner | −0.0014 | [−0.0047, +0.0019] | – |
| erode1 masks fed to the refiner | −0.0011 | – | 0.02 |
| 1 px square erosion of the refined output | −0.068 | – | – |
| 1 px cross erosion of the refined output | −0.031 | – | – |

- Refiner masks are already calibrated. The refiner threshold grid t0.35–0.60 is flat to within 0.002.
- **This closes the mask-thickness lever.** It also explains why direct-detector gains shrink by about 80% after the refiner: much of what the detector improves is boundary quality, which the refiner already provides.
- **Future detector changes should be judged after the refiner, not on the direct detector.**

## 7. Ranked levers (estimated recoverable PQ on the refined pipeline)
1. **Candidate ranking and selection over the whole 4-view pool.** Estimated +0.010 to +0.020.
   - **Evidence:** 62% of refined FN (355 / 573) already have an IoU > 0.5 mask in the pool. For 180 of them it is a refined prediction below the cut; for 175 it is a single-view mask that consensus did not pick.
   - **Upper bound:** the score oracle alone gives **0.508 (+0.048)**, with no new FP. Category c adds roughly as much again.
   - **Realistic estimate:** this is a learned re-scorer that sees every view mask as a candidate (not only the strongest) and is trained with **agreement-fraction targets** from the 296 multi-record files (the k/n rule in §0). It should recover 20–40% of the bound.
   - **Caveats:**
     - About 64% of score-class FN are single-annotator filaments, so they are intrinsically ambiguous.
     - Earlier verifiers failed: SWEFil −0.07, and V60 consensus ranking. They did not use multi-annotator targets or view-level candidates.
2. **Suppressing avoidable FP**, mainly fragments and isolated hallucinations. Estimated +0.008 to +0.012.
   - **Evidence:** in the refined pipeline, 299 of 448 FP match no annotator. Among those are 70 fragments, 16 merges and 113 no-overlap hallucinations.
   - **Upper bound:** removing all avoidable FP gives **0.509 (+0.050)**.
   - **Fragments:** they rose to 20% of refined FP, which points at the refiner/ownership step leaving pieces behind. A post-process rule (drop a low-score refined mask that is mostly adjacent to or inside a higher-score mask's dilation) is cheap and testable on CPU with the same harness (`analysis_v69_refined_probe.py`).
   - **Hallucinations:** they score low, so they overlap the score lever.
3. **Low-contrast, small and limb filaments: recall and near-misses.** Estimated +0.005 to +0.010.
   - **Evidence:** the lowest contrast quintile holds 30% of FN, GT ≤500 px hold 25%, and "Unidentifiable" filaments hold 46%.
   - **Refined near-misses:** 132 FN (23%) are near-misses. Fixing the 72 kept near-misses into TP would remove 72 FN and up to 72 FP.
   - **Ceiling:** SQ is already above human boundary agreement (0.689 vs 0.63), and most of these GT are single-annotator. Native-resolution or contrast-normalised input for small, faint candidates is the route, at a bounded gain.

**Not worth pursuing:**
- **Perfect SQ:** human boundary agreement caps it.
- **Split/merge topology:** split GT, merge FP and absorbed FN together are only about 1.5% of GT and 7% of FP.
- **Image-quality conditioning:** no correlation with PQ.
- **Detector-side mask sharpening:** the refiner absorbs it (§6).

**Expectation setting.** A held-out annotator scores 0.345 against the others, and the model already scores 0.41–0.46. The ceiling of this metric on this data is far below 1. The remaining gains look incremental (about +0.01 to +0.02 each). A large jump would need modelling annotator propensity (the k/n target) rather than better vision.
