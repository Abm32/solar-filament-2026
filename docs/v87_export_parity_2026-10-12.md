# V87: end-to-end export-path parity on held-out train folds (2026-10-12)

**Question.** V84 scores OOF PQ 0.4590 but public 0.39. Is the gap caused by the test submission path doing something different from the OOF evaluation path?

**Answer: no.** The exact export path reproduces OOF on all 5 folds (707 files): identical views and identical TP/FP/FN, and the host scorer applied to the submission-format CSV gives the same PQ.
- Image properties of test and train are identical as read by the export.
- The gap is not a submission-path bug.

All work ran on rogii, GPUs 0-2 only. It used train labels only, no test labels, no external GT, and made no submission.

## Method
- **Script:** `scripts/v87_export_parity.py <fold>`.
  - For each held-out train file of fold k it runs the **test path verbatim**: `v61_toolkit.generate_views` on the image file (fresh detector inference, no cache), then `export_v74.refine_image`, then the `export_v74.main` CSV writer (`<stem>_<n>`, RLE counts).
  - It uses the fold-k detector `/mnt/v61/p2first_fold{k}/mask_ratio1/final.pt` at imgsz 1024, with 4 flips, box iou 0.3, conf floor 0.1, retina masks and agnostic NMS.
- **Arms:**
  - **v84:** fold-k V84 soft refiner, t0.50, conf 0.30, MIN200, and the fold-k cross-fitted demote model (`outputs/v84/export/parity/demote_fold{k}.json`). These are the exact `_v84_export.sh` flags.
  - **v65:** fold-k V73 base (r3) refiner, t0.45, conf 0.30, min-area 20, no demote. This is the V65 recipe; its OOF is the V73 base k7 cell.
- **Stage checks:**
  1. Fresh views compared with the OOF cache views (count, score and RLE per view).
  2. Per-file PQ counts compared with the OOF references: `outputs/v84/export/parity/bundle_ref.pkl` for v84, and `outputs/v73/base_fold{k}.json` per_file at k7/t0.45_c0.30 for v65.
  3. `scripts/v87_host_score.py` scores the CSVs with the **host self-evaluation code** (verbatim cells, as in V79) against host-format train GT, with one block per annotator record.
- **Outputs:** `outputs/v87/fold{k}.json`, `fold{k}_{v84,v65}.csv`, `host_score.json`, `props.json`.

## Result: export path = OOF

| fold | files | views identical | V84 export PQ | V84 OOF PQ | V65 export PQ | V65 OOF PQ |
|---|---|---|---|---|---|---|
| 0 | 141 | 141 | 0.458988 | 0.458988 | 0.453259 | 0.453259 |
| 1 | 141 | 141 | 0.452864 | 0.452864 | 0.443200 | 0.443200 |
| 2 | 142 | 142 | 0.473478 | 0.473478 | 0.470639 | 0.470639 |
| 3 | 142 | 142 | 0.455120 | 0.455120 | 0.452429 | 0.452429 |
| 4 | 141 | 141 | 0.454597 | 0.454597 | 0.450145 | 0.450145 |
| **pooled** | **707** | **707** | **0.4589754** | **0.4589754** | **0.4539309** | **0.4539309** |
| host scorer on CSV | | | 0.4589754 | | 0.4539309 | |

- **Detector stage:** the fresh views are bit-identical to the OOF cache on 707/707 files, covering scores and RLEs.
- **V65 arm:** identical counts on 707/707 files.
- **V84 arm:** identical TP/FP/FN on 707/707 files. Two fold-2 files (`20161223133734Ch`, `20170427154550Bh`) differ only in matched-IoU sum, by 4e-4 and 8e-4. That is a few pixels of refiner float nondeterminism. The pooled ΔPQ is 5e-8.
- **Host scorer:** host PQ on the CSVs equals the internal counts to 1e-9, so RLE orientation, ids and the writer are fine.

**There is no discrepancy, so there is no stage to localize.** The earlier 10-file V84 parity used cached views; this run closes the detector-inference and CSV stages as well.

What remains untested by construction:
- The test export uses the **all-data** detector (fold99), the all-data refiner and the all-data demote model, not fold models.
- These are single fits that are only checked to load and run. A bad all-data fit would show up as a *variable* gap, but V65 and V84 use different all-data refiners and demote settings and both score 0.39.

## Image properties as the export reads them
`scripts/v87_props.py` reads every file (707 train, 180 test) through cv2 colour (the detector), cv2 grayscale (the refiner) and the PIL header.
- **Identical properties for all 887 files:**
  - `.jpeg`, JPEG, PIL mode L (single-channel greyscale), 2048x2048, baseline (not progressive), no EXIF, no DPI.
  - cv2 colour is uint8 (2048,2048,3) with all three channels equal, and grayscale equals channel 0.
  - **No resizing happens anywhere in our code.** The detector letterboxes to 1024 internally, and the masks come back at 2048 (retina).
- **Distributions:**

| | train p5 / median / p95 | test p5 / median / p95 |
|---|---|---|
| bytes | 710k / 790k / 882k | 717k / 787k / 893k |
| gray mean | 79.9 / 82.5 / 89.4 | 80.1 / 82.8 / 91.7 |
| gray max | 229 / 234 / 255 | 229 / 234 / 255 |
| fraction of zero pixels (off-disk) | 0.198 / 0.206 / 0.230 | 0.198 / 0.205 / 0.228 |

**Examples (name: gray mean, max, zero fraction):**
- train
  - 20110109104734Ch: 85.3, 230, 0.229
  - 20120803164614Mh: 86.4, 254, 0.198
  - 20140122180214Mh: 81.6, 236, 0.210
  - 20150625082034Lh: 80.6, 255, 0.208
  - 20170329133730Ch: 81.2, 255, 0.220
- test
  - 20110120105534Ch: 79.4, 255, **0.384** (one test outlier with a smaller or cropped disk)
  - 20120710063134Lh: 80.7, 255, 0.215
  - 20140322200714Mh: 89.0, 255, 0.199
  - 20151227133734Ch: 79.0, 255, 0.225
  - 20171021223430Lh: 79.3, 229, 0.225

## OOF to public history
- Columns:
  - The local metric is the one recorded at the time, and its basis varies. V4-V8 use the kernel's own 141-file validation. E015-V17 use the 94-file canonical tune set. V65 and V84 use 5-fold OOF on 707 files.
  - The ranges in brackets reflect the board's 2-decimal display (±0.005).
- Correction: V7 public was **0.34**, not 0.30. The 0.31 entry is V3 (`56007181`), whose local PQ was not trustworthy because of the uint8 IoU-overflow bug.

| submission | local | public | gap (range) | public/local (range) |
|---|---|---|---|---|
| V4 | 0.4087 | 0.34 | 0.069 (0.064..0.074) | 0.832 (0.820..0.844) |
| V5 | 0.3969 | 0.33 | 0.067 (0.062..0.072) | 0.831 (0.819..0.844) |
| V6 | 0.4268 | 0.35 | 0.077 (0.072..0.082) | 0.820 (0.808..0.832) |
| V7 | 0.4060 | 0.34 | 0.066 (0.061..0.071) | 0.837 (0.825..0.850) |
| V8 | 0.4027 | 0.35 | 0.053 (0.048..0.058) | 0.869 (0.857..0.882) |
| V8+COCO/TTA (tune) | 0.4011 | 0.37 | 0.031 (0.026..0.036) | 0.922 (0.910..0.935) |
| E015 TTA (tune) | 0.4328 | 0.38 | 0.053 (0.048..0.058) | 0.878 (0.866..0.890) |
| V11 gt_centered (tune) | 0.4373 | 0.39 | 0.047 (0.042..0.052) | 0.892 (0.880..0.903) |
| V11 proposal_centered (tune) | 0.4380 | 0.39 | 0.048 (0.043..0.053) | 0.890 (0.879..0.902) |
| V13 (tune) | 0.4418 | 0.39 | 0.052 (0.047..0.057) | 0.883 (0.871..0.894) |
| V17 (tune) | 0.4487 | 0.39 | 0.059 (0.054..0.064) | 0.869 (0.858..0.880) |
| V65 (OOF 707) | 0.4539 | 0.39 | 0.064 (0.059..0.069) | 0.859 (0.848..0.870) |
| V84 (OOF 707) | 0.4590 | 0.39 | 0.069 (0.064..0.074) | 0.850 (0.839..0.861) |

**Reading the table.**
- **The gap is present in every submission and is never below 0.03.** It survives several complete rewrites of the export code: the Kaggle kernel at V4-V8, the V17 composer and the V62/V74 sharded export. The external teams quoted in V69 report the same 0.05-0.06 gap. A path bug common to all of these is implausible, and V87 now measures our current path as exact.
- **Neither the ratio nor the offset is constant**, ranging over 0.82-0.92 and 0.03-0.08. The mixed local bases (141 val, 94 tune, 707 OOF) explain part of this.
- **Within one basis** (tune V11 to V17, then OOF V65 to V84), local rose by 0.011 and 0.005 while public stayed at 0.39. In other words, the slope from local to public is below 1, and the public subset responds weakly to the gains we measure.
- **This pattern points to the test labels**, through annotator style or protocol, or a different distribution of filament types on the public 90 images. It does not point to a submission-path bug.
- It cannot be resolved further without test labels, which we do not use.

## Verdict
The V84 and V65 export path is not the cause of the gap: it is exact on 707/707 held-out files, it is host-scored on the CSV, and test images are byte-format identical to train. Close the "submission path bug" hypothesis.
- **Remaining explanation:** test-label or annotator differences on the public split. This is consistent with every clean team landing at 0.39-0.41.
- **Implication:** OOF gains will move public weakly, with a slope below 1. Prefer robustness and annotator-agnostic gains over small OOF increments, and don't expect a public jump from them.
