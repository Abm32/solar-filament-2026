# V85 error analysis of the V84 pipeline (2026-10-12)

This is a CPU-only, post-hoc diagnostic on the 5-fold OOF predictions of the V84 bundle (707 train files, 1154 annotator records).
- No GPU was used, nothing was trained or submitted, and no test labels, Dataverse test GT, leaked CSV or SWEFil were used.
- Nothing here authorises a change. Every lever below needs its own pre-registration.
- It builds on `v69_error_analysis_2026-10-10.md`, which covered the p2first detector and the r3 refiner on 141 files. This report repeats that analysis on the current pipeline: soft refiner, t0.50, MIN200 and cross-fitted demote.

## Sources and reproduction
- **`scripts/v85_extract.py`** runs the V84 soft-refiner path (t0.50) on all 707 files. For every GT of every record it stores the IoU at each stage:
  - the 4-flip raw view masks;
  - the consensus representative, which is the refiner seed;
  - the refined mask before ownership;
  - the refined mask after ownership, for all alive proposals including those scoring below 0.15.
  - It also stores GT geometry and contrast (`audit_v80_extract.gt_feats`) and cross-record agreement.
  - **Assertion:** the candidate IoU columns equal `outputs/v84/extract_soft.pkl` exactly (max |Δ| = 0.0).
- **`scripts/v85_analyze.py`** takes the kept set from `v84_bundle.run_arm` (cross-fitted demote, as in `report/plots.py`).
  - **Assertion:** PQ = **0.45898** (TP 5050, FP 1881, FN 3149, SQ 0.6876).
  - Output: `outputs/v85_analysis/report.json`, plus per-row data in `rows.pkl`.
- **`scripts/v85_extra.py`** computes exact what-ifs, the ownership drill-down and the leaderboard-gap checks. Output: `outputs/v85_analysis/extra.json`.
- **How "PQ if fixed" is computed:** each FN in the class is turned into a TP at its recoverable IoU (or at the current SQ when no such IoU exists), with all else held fixed. This "naive" figure is an upper bound because it ignores the FP that adding a mask creates in other records.
  - Values marked **exact** instead re-score the actual candidate masks against every record.

## 1. Decomposition of the lost PQ

### FN (3149) by pipeline stage
The stages are applied hierarchically: each FN is assigned to the first stage where a mask with IoU > 0.5 existed and was lost.

| FN class | n | share | naive PQ if fixed | single-annotator share (multi-record files) |
|---|---|---|---|---|
| **Dropped by the score cut** (alive refined mask with IoU > 0.5, score < 0.30) | **870** | 27.6% | +0.047 | 65% |
| **Ownership clip/kill** (pre-ownership refined mask has IoU > 0.5; after ownership it does not) | **610** | 19.4% | +0.036 | 57% |
| Near-miss in the pool (best of any stage has IoU 0.3–0.5) | 338 | 10.7% | +0.020 | 85% |
| Near-miss on a kept mask (kept prediction has IoU 0.3–0.5) | 320 | 10.2% | +0.019 | 68% |
| No candidate at all (IoU < 0.05 at every stage) | 275 | 8.7% | +0.016 | **95%** |
| Dropped by demote (score ≥ 0.30, area ≥ 200, demoted) | 225 | 7.1% | +0.012 | 60% |
| Weak overlap only (IoU 0.05–0.3) | 181 | 5.7% | +0.011 | 87% |
| Refiner damaged it (seed has IoU > 0.5, refined mask does not) | 173 | 5.5% | +0.010 | 64% |
| Consensus picked the wrong representative (some single view has IoU > 0.5) | 156 | 5.0% | +0.009 | 59% |
| Dropped by MIN200 | 1 | 0.0% | 0 | – |

Grouped:
- **Candidate existed but was dropped:** 2035 FN (65%).
- **Shape near-miss:** 658 FN (21%).
- **No usable candidate:** 456 FN (14%).

**Exact re-scoring of dropped candidates.** Alive masks are non-overlapping, so they can be added back and every record re-scored.
- Adding back all 964 alive candidates that would match some FN: **+0.040 exact** (PQ 0.499).
- Only those with score < 0.30: **+0.033**.
- Only demote/MIN200 drops: **+0.0085**.
- **The cut cannot separate them by score.** In the score band 0.20–0.30 there are 323 TP-quality candidates and **658 junk** candidates.

**Ownership drill-down (610 FN).**
- 430 masks were clipped and 180 were killed (left below MIN_AREA after clipping). Their median area is 1579 px.
- For **433 of the 610 (71%)**, a higher-scored kept prediction overlaps the GT at IoU 0.05–0.5. A fatter or merged neighbour wins the pixels and is not itself a TP for this GT.
- This is the "swallowed" pattern: 305 FN sit under a kept prediction that covers ≥ 50% of them and ≥ 30% of another GT.

### FN by GT property

| factor | worst bin | recall | FN (share) | naive PQ if all fixed |
|---|---|---|---|---|
| **agreement, 3-record files** | drawn by 1 of 3 | **0.29** | 698 (22%) | +0.040 |
| agreement, 2-record files | drawn by 1 of 2 | 0.36 | 553 (18%) | +0.032 |
| (for contrast) | 3 of 3 / 2 of 2 | 0.91 / 0.85 | 107 / 166 | – |
| **contrast** | lowest quintile | **0.33** | 1098 (35%) | +0.062 |
| size | ≤ 500 px | 0.42 | 728 (23%) | +0.042 |
| size | 500–1k px | 0.61 | 837 (27%) | +0.048 |
| limb distance r/R | 0.85–0.95 | 0.38 | 240 (7.6%) | +0.014 |
| elongation (length/width) | < 10 (blobby) | 0.52 | 1363 (43%) | +0.076 |

Recall by number of records per file is 0.585 for 1 record, 0.638 for 2 and 0.634 for 3.

### FP (1881 record-level FP)

| FP class | n | share | of which match a GT in another record of the same image | PQ if removed in this record only |
|---|---|---|---|---|
| no overlap with this record | 790 | 42% | 309 | +0.025 |
| near-miss of a GT (IoU 0.3–0.5) | 735 | 39% | 338 | +0.023 |
| fragment (≥ 50% inside one GT) | 307 | 16% | 101 (16 are fragments of a matched GT) | +0.010 |
| weak overlap | 28 | 1.5% | 9 | +0.001 |
| merge (covers ≥ 30% of ≥ 2 GT) | 21 | 1.1% | 7 | +0.001 |

- **Disagreement FP:** 764 of 1881 (41%) are another annotator's TP. They cannot be removed without losing that TP.
- **Pure-junk predictions:** 811 kept predictions match no record at all, giving 1117 FP rows. Removing them is worth **+0.037 exact**.
  - By class: no-overlap 335, near-miss 305, fragment 147, merge 10, weak 14.
  - By size: ≤ 500 px 142, 500–1k 248, 1k–2k 221, larger 200.
- FP by prediction size: ≤ 500 px 19%, 500–1k 31%, 1k–2k 28%, 2k–4k 14%, larger 8%.

### SQ
- Mean IoU of TPs is flat across size bins: 0.684–0.699.
- TP IoU histogram: 0.5–0.6: 935, 0.6–0.7: 1669, 0.7–0.8: 2018, 0.8–0.9: 428, > 0.9: 0. There are 351 TPs at IoU 0.50–0.55, close to flipping.
- **Every TP at IoU + 0.05 would be worth +0.033.** This lever is not reachable: SQ 0.688 is already above the human-vs-human ceiling of 0.63 (V69 §3), and V69 §6 showed the refiner already absorbs mask-thickness gains.

## 2. Split and merge (final kept set)
- **GT split** across ≥ 2 kept predictions (each ≥ 50% inside the GT and covering ≥ 10% of it): **91 FN**. A perfect fix (FN to TP, fragment FP removed) is worth **+0.015**.
- **Merges:** 21 merge FP, and **305 FN swallowed** by a prediction that covers ≥ 50% of them and ≥ 30% of another GT. A perfect fix is worth **+0.019**. This overlaps heavily with the ownership class above.
- Topology errors are not separate from ownership. Most "merges" are a kept neighbour holding the pixels after the greedy ownership step, not one blob covering two TPs.

## 3. Per-image, records and leaderboard resolution

**Per-file PQ.**
- Quantiles: p5 0.00, p10 0.25, p25 0.37, median 0.47, p75 0.54, p90 0.62.
- 37 files score PQ 0.
- The loss is diffuse: the worst 10% of files hold 30% of the "perfect-file" gain, the worst 20% hold 49%, and the worst 50% hold 82%.

**By number of records.**

| records | files | PQ | FP per record | FN per record |
|---|---|---|---|---|
| 1 | 411 | 0.460 | 1.43 | 3.23 |
| 2 | 145 | 0.463 | 1.75 | 2.48 |
| 3 | 151 | 0.456 | 1.74 | 2.43 |

**By annotator code** (the image-id prefix, codes with ≥ 40 records):
- PQ runs from 0.399 (010401, 010101) to 0.560 (020403).
- Annotator 030101 marks 10.5 GT per record, with 270 FN and only 52 FP.
- No single annotator dominates. Removing any one code moves pooled PQ by at most +0.004.

**Can a 90-image public set explain 0.39?**

| simulation (20k draws) | std | p05 |
|---|---|---|
| 90 files without replacement | 0.012 | 0.439 |
| clustered by annotator code (6–10 codes) | 0.017 | 0.434 |

- In the file draws, P(PQ ≤ 0.395) is 0.
- In the annotator-clustered draws, P(PQ ≤ 0.40) is 0 and the minimum over 20k draws is 0.407.
- The test record-count mix cannot explain it either. Pooled PQ is 0.4555 if every file has 3 records, 0.4633 if every file has 2, and 0.459 for the train mix.

**The OOF-to-public gap of −0.07 is systematic, not sampling noise.** Candidate explanations that were checked and ruled out:
- **Same-day leakage across folds:** files with no same-day file in another fold score 0.456, against 0.478 for the rest. That accounts for at most −0.003.
- **Image-statistics shift:** V79 KS tests on site, year and image statistics all give p > 0.1.
- **Leaderboard sampling:** see the table above.

What remains is a difference in the test labels (annotator style or protocol) or something in the submission path.
- V65 and V84 both show 0.39 on the public board, even though OOF improved from one to the other.
- Consistent with this, 71 of the 72 non-leak teams sit at 0.39–0.41 (the one exception is at 0.49).
- A practical heuristic is public ≈ 0.85 × OOF.

**Resolution.**
- The paired noise on a Δ between two of our submissions over the same 90 images is about **0.004 (1σ)**. This is scaled from the V84 bootstrap: 707-file CI half-width 0.0027, times √(707/90).
- The board shows 2 decimals, so ±0.005 is hidden by rounding.
- **Going from 0.39 to 0.41 needs a real public gain of about +0.015–0.025.** At the 0.85 ratio that means roughly **+0.02–0.03 OOF**. A +0.005 OOF change (V84) is invisible.

## 4. Top 5 fixable error classes (ranked by realistic recoverable PQ)

| # | class | size | upper bound | realistic | method |
|---|---|---|---|---|---|
| 1 | **Ownership clip/kill and swallowing:** a higher-score neighbour takes the pixels | 610 FN; 71% have the neighbour overlapping the GT | naive +0.036 | **+0.006 to +0.012** | Replace greedy score-order ownership with pixel-wise argmax of refiner probability among overlapping proposals, or "keep the clipped mask if it is still ≥ X% of its pre area and drop the occupier's overlap". This is CPU-testable with the v85 extract path, because pre masks and probabilities are already computed. It has not been tried before; V69 §6 only tested thickness. |
| 2 | **Score-cut drops** (TP-quality masks below 0.30) | 870 FN; 768 candidates | exact +0.033 | **+0.004 to +0.010** | Feature promote failed in V83 (+0.0001), because in the 0.20–0.30 band junk outnumbers good 2:1. The extra signal must come from outside the 28 features: cross-detector agreement (the V86 pooled selector, already pre-registered) or a 1280 / second-detector vote. |
| 3 | **Pure-junk kept predictions** (match no annotator) | 811 predictions, 1117 FP rows | exact +0.037 | **+0.003 to +0.008** | Same selector as #2, in the demote direction, with explicit fragment features: inside-fraction relative to the nearest higher-score kept mask, and the gap to it. Fragments make up 147 of the 811 (18%). Demote is already applied, so only the residual is left. |
| 4 | **Representative or refiner choice** (consensus picked the wrong view, or the refiner degraded a good seed) | 329 FN | naive +0.020 | **+0.003 to +0.006** | Multi-hypothesis refinement: refine every view member, not only the strongest, and pick by refiner p_in. Score with agreement targets. This is a candidate-pool change, so it belongs inside the V86 pooled selector. |
| 5 | **Kept near-misses** (IoU 0.3–0.5, each one FN plus one FP) | 320 FN + 735 FP rows | naive about +0.04 | **+0.002 to +0.005** | 68% are single-annotator and SQ is above the human ceiling. The only cheap test is a refiner threshold or dilation tuned per size class. Expect little. |

**Not worth pursuing:**
- **SQ +0.05:** worth +0.033 on paper, but above the human ceiling.
- **"No candidate" FN:** 95% are single-annotator.
- **Split topology:** 91 GT, +0.015 only with perfect repair.
- **Image-quality conditioning:** shows no correlation with PQ.

**Bottom line.**
- Levers 1, 2 and 4 together realistically give about +0.015–0.025 OOF. That is borderline for a visible 0.39 → 0.40, and probably not enough for 0.41.
- The cheapest new lever is **#1 (ownership)**. It is untested and CPU-only.
- The 0.07 OOF-to-public gap is larger than every in-pipeline lever combined. Two unexplained items deserve a CPU check first:
  - Re-verify that the submission path equals OOF on more than 10 files: full-data or fold models, the λ, and the demote model actually used at test time.
  - The test-label difference itself.

## Public notebooks and discussion after 2026-10-10 (read only, nothing run)
- `kaggle kernels list` (sorted by run date) shows four notebooks after 10-10:
  - **hryhoriinazarenko/filaments:** EDA only.
  - **jamesmills1975/cage-filament-unet-public:** smp U-Net resnet34 at 1024 with connected components. The author states PQ 0.33.
  - **koushikkumardinda/...-edgeattnet:** U-Net with edge attention, no score disclosed.
  - **hodohabalotchikp/notebookc66249a467:** imports the kernel `rokaiyasomapti/lb-1st-solar-filament-segmentation-2026`, which is most likely a leak. It was not opened.
- None discloses a clean 0.40–0.49 method.
- The discussion pages need a login. The web fetch returned nothing and the web search found no threads, so nothing could be confirmed there.
