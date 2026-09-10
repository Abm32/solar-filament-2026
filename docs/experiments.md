
2026-09-10 latest: **V19 Mask2Former — the "genuinely different architecture
family" the project's own analysis said was the only remaining lever — ran to
completion and failed hard.** Best tune PQ **0.153578**, roughly a third of
SUBMIT_BAR (0.448703), audit PQ 0.150346. See E029 below.
# Experiments

Record every experiment here before and after running it.

2026-09-09 latest: **V17 scored 0.39 — the fifth consecutive 0.39 — and that
result falsifies the transfer model the project was steering by. Two further
zero-GPU analyses then closed the last cheap lever. The V18 detector retrain is
infeasible on the available hardware.** Three findings, in order of consequence.

**1. `ceiling_analysis` §1's constant-offset model is dead (E027).** That document
asserted a stable local→public offset of −0.050 ± 0.003 across four submissions and
derived the project's whole target from it: *"public 0.55 requires local tune PQ
≈ 0.60."* V17 is the fifth point and it breaks the model.
`scripts/analyze_public_transfer.py` (zero GPU) shows:

| local tune PQ | public | submission |
| ---: | ---: | --- |
| 0.432823 | 0.38 | `56026663` |
| 0.437313 | 0.39 | `56050554` |
| 0.438001 | 0.39 | `56049359` |
| 0.441777 | 0.39 | `56076685` |
| **0.448703** | **0.39** | **`56103401`** |

Local tune PQ rose **+0.015880** across this table while the public score moved
once, by one display step. The feasible constant-offset region is **empty** under
both round-to-nearest and truncation display models — and it was *non-empty* for
the first four points, so V17 specifically is what falsifies it (asserted by test).
For affine models `public = a·local + b`, feasible slopes are **a ∈ [0.01, 0.87]**
and **a = 1.00 is ruled out**. Consequence: at the most favourable feasible slope,
local 0.60 yields public ≈ **0.527**, and public 0.55 needs local ≥ **0.6269**.
**The target was understated and local PQ is a weaker predictor than assumed.**

**2. The detectors do miss different instances, but the ensemble does not pay
(E028).** `ceiling_analysis` §8's "E-H fold ensemble" was never run. Two new
zero-GPU analyses settle it on the six proposal populations already on disk
(V6@{0.7,0.3}, arm_r1 seed1/seed2 @{0.7,0.3}), same 141 files, same GT:

| tune group, grouped pre-refiner | recall | meanIoU |
| --- | ---: | ---: |
| best single source (`MR1s2_iou070`) | 0.7875 | 0.6451 |
| **UNION of all six (oracle)** | **0.8131** | 0.6522 |
| missed by **every** source | **212 of 1134 (18.7%)** | — |

So complementarity is real (+0.0256 oracle recall) but bounded, and **the union's
oracle PQ is 0.5849 at zero FP and 0.5092 at V17's actual FP=306 — both below the
0.60 that even the falsified transfer model asked for.** A *perfect* six-way
ensemble cannot reach the target.

The realistic version is worse. `scripts/analyze_union_ensemble.py` merges all six
pools with cross-source NMS at IoU 0.5 and sweeps the shipped confidence grid:
**UNION beats the best single source at 1 of 7 tune confidences (max gain
+0.0005) and 2 of 7 audit.** The extra recall does not survive its false
positives. **E-H is now measured and dead.**

**3. V18 (`imgsz=1536` + `mask_ratio=1`) is infeasible on a single T4, and both
runs errored.** Kernels `solar-filament-v18-native-2048` and `…-reva` both reached
`KernelWorkerStatus.ERROR`. Neither is a code bug: **both verified the frozen
control exactly first** (tune 0.432823, audit 0.450130, raw mean IoU 0.5253, raw
max 0.8734) and then died in epoch 1 with `torch.OutOfMemoryError` inside
Ultralytics' `single_mask_loss` → `binary_cross_entropy_with_logits`. The first run
used `GLOBAL_BATCH=2`, hit Ultralytics' auto-fallback to `batch=1` at step 7, and
still OOMed around step 555/937 (requested 1.17 GiB, 1.16 GiB free). The revA run
set `batch=1` from the start *and* `expandable_segments:True`, and still OOMed
(requested 2.85 GiB, 1.91 GiB free) at t=1842 s.

**Why the smoke test wrongly authorised it:** the probes ran at `fraction=0.2` and
`0.02` and saw at most **24** and **13** instances per batch; the full 937-image
run reaches **31**. At `mask_ratio=1` the loss keeps masks at full proto
resolution, so its memory scales with instances-per-image — the quantity the
subsampled probe understated. `batch=1` is already the floor, so **there is no
batch size at which this configuration fits 14.56 GiB.**

Cost: ~1,550 GPU-s across the two failed runs. Quota **45,292 s** of 108,000
remaining, refresh 2026-09-12. Artifacts: `outputs/public_transfer_2026-09-09.json`,
`outputs/v16_proposal_analysis/detector_complementarity.json`,
`outputs/v16_proposal_analysis/union_ensemble.json`. Local suite **385 passed,
1 skipped** (from 304), including `tests/test_transfer_and_ensemble_analyses.py`
(25 tests). Also fixed a repeat of E026's cache trap:
`outputs/kaggle_v17_full/proposal_cache` was an **incomplete** pull (323 of 744
signatures, partial per cell); re-pulled complete to
`outputs/kaggle_v17_cache_pull/proposal_cache` (744 files), and the analyses assert
completeness before computing anything.

---

2026-09-08: **V17 (mask_ratio=1 detector x V13 recipe) cleared SUBMIT_BAR,
not PROMOTION_GATE — submitted as `56103401`, scored 0.39 (see above).** Per
`docs/v17_design_2026-09-08.md`, composed V16's replicated `mask_ratio=1` detector
with V13's measured recipe (box `iou` swept over {0.3, 0.7}, V11
`proposal_centered_e4` refiner, 84-candidate grid, 2 seeds). Inference-only —
trains nothing — so E026's detector-training noise floor (±0.007677 tune PQ) does
not apply; this comparison is deterministic and exactly reproducible.

Selected: `mr1_iou030/t0.45_c0.30`, tune PQ **0.448703**, audit PQ **0.470763**
(seed 1, pre-registered primary). Beats every parent on both groups:

| vs | tune delta | audit delta |
| --- | ---: | ---: |
| E015 control | +0.015879 | +0.020634 |
| V11 `proposal_centered_e4` | +0.010702 | +0.015879 |
| V13 compose (best measured, 0.39 public) | **+0.006926** | +0.006119 |
| V16 `arm_r1` seed1 | +0.017099 | +0.018354 |
| V16 `arm_r1` seed2 | +0.013907 | +0.022623 |

`export_authorized: true` (clears SUBMIT_BAR: tune >= V13's 0.441777, audit >=
E015's 0.450130). `promotable: false` (misses PROMOTION_GATE: needs tune >= V13 +
0.010; got +0.006926). Frozen control reproduced E015's exact tune/audit counts
and its CSV hashed to the accepted 0.38 file byte-for-byte, so the harness is
trustworthy.

**Caveat the report itself flags:** seed spread is 0.0038 (seed1 tune 0.448703 vs
seed2 tune 0.444860) — smaller than E026's training-noise floor but non-trivial.
Both seeds independently clear SUBMIT_BAR, but selection used seed1 only per the
predeclared protocol; audit never selects. Read as: the composition is real and
beats all parents in both seeds, but the exact margin above V13 is seed-sensitive.

Submitted `outputs/kaggle_v17_full/submission.csv` (1,215 masks, 177 populated
images, validated non-overlapping, all RLEs decode,
sha256 `87b22b0...c1cb922b`) as **`56103401`**, 2026-09-08 17:24 UTC. This is the
first submission whose local edge traces to a *replicated, mechanistic*
proposal-quality gain (raw IoU ceiling broken in two independent training seeds,
E026) rather than an inference-side rearrangement — worth one submission as a
calibration point on whether that class of gain transfers where V13's did not.
Score not yet returned; see RESUME_TOMORROW.md for the outcome once scored.

Artifacts: [report](../outputs/kaggle_v17_full/v17_report.json),
[manifest](../outputs/kaggle_v17_full/v17_manifest.json), kernel
`abhimanyu122/solar-filament-v17-compose` v1, smoke kernel
`abhimanyu122/solar-filament-v17-compose-smoke` v1 (both `COMPLETE`).

2026-09-07 latest: **V13 cleared both predeclared bars — the first candidate ever
to pass a promotion gate — and the public score did not move (E024).** Composing
E-A's box `iou=0.3` flags with V11's `proposal_centered_e4` refiner reached tune
**0.441777** / audit **0.464644**, beating all three parents on both groups: vs the
accepted E015 control +0.008954 / +0.014514, vs E-A alone +0.005052 / +0.005590,
vs the 0.39-scoring V11 **+0.003776 / +0.009760**. Submitted as **`56076685`** →
**0.39**, tying the existing best. Public best stays **0.39**; 4 submissions
remained after the upload.

Two findings, one positive and one cautionary.

**E-A and V11 are additive.** The composition is meaningfully better than either
parent, so they are genuinely orthogonal mechanisms — E-A recovers missed instances
at the proposal stage, V11's continuation training refines the masks that survive —
not two routes to the same gain. That is worth knowing regardless of the public
result.

**Local gains in the +0.003–0.010 band are no longer trustworthy evidence of
public movement.** Calibration now has four points and the newest is the first
negative one: +0.0012 → 0.38, +0.0045 → 0.39, +0.0052 → 0.39, and **+0.0038 over
the 0.39 → still 0.39**. Two readings remain open and this run cannot separate
them: either the gain does not transfer to the public split, or two-decimal
rounding hides it (0.39 spans 0.385–0.394, so a true +0.004 can be invisible).
Either way, **stop spending submissions on local deltas of this size** — the next
one needs a materially larger local margin or a different mechanism.

Verification was clean throughout, which is what makes the flat result
interpretable rather than suspect: the control arm ran first, reproduced E015's
exact tune/audit counts, and its exported CSV hashed to the accepted 0.38 file
byte-for-byte; independent CSV validation passed (1,198 masks, 177 populated
images, non-overlapping, all RLEs decode). Local suite **186 passed**.

2026-09-07 earlier: **E-A PASSED its gate (E023) — recall finally moved.** Box
`iou=0.3` on the frozen detector recovers **58 of the 508 missed tune instances**
(recall 0.5520 -> 0.6032) at a net-positive PQ trade: tune **0.436726**
(+0.003902 over the control's 0.432823), audit **0.459054** (+0.008925). This is
the first lever since E018 to clear its predeclared bar, and the first of any kind
to convert recall into PQ rather than losing it back in precision. The frozen
control reproduced E015's accepted counts exactly. Public best is still **0.39**;
nothing was submitted and all 3 submissions are available.

Three honest limits on that result. `agnostic_nms` is **inert** — the detector is
`single_cls=True`, so the `agnostic`/`noagnostic` arms are bit-identical across all
42 candidates and the 2x2 grid is really 1x2 on the box `iou` axis. The gain is
**confidence-coupled**: at the control's own c0.40 the `iou=0.3` cell is slightly
*worse* (-0.000922), and it only wins once the confidence floor drops to 0.30, so
this is a flag+threshold interaction inside the predeclared grid, not a free flag
win. And the margin is **inside the noise**: a paired per-image bootstrap (10,000
resamples, seed 42) gives a tune 95% CI of **[-0.005372, +0.013211]** with
P(delta>=0.003)=0.57, so the point estimate clears the gate while the interval
does not exclude zero — the same caveat that applied to E018.

The first E-A attempt (04:47 UTC) errored on a **measurement bug, not a model
bug**: its control check diffed against the *single-view* sweep stored beside the
refiner checkpoint (candidate t0.50/c0.40) while E-A's control runs
**flip_consensus**. Confirmed locally that `single` matches that file byte-exactly
and `flip_consensus` differs at precisely the 2.7380-vs-2.1754 first value the
kernel reported. The check now pins the flip_consensus E015 counts as
`CONTROL_REFERENCE` (identical to `v12_resolution.V6_REFERENCE`), and the re-run
reproduced them exactly. Local suite: **164 passed**.

Next step: E-A and V11 are orthogonal levers (inference flags vs refiner weights),
and E-A's best is **+0.004170 ahead of V11's 0.39-scoring `proposal_centered_e4`
on audit** while 0.001275 behind on tune. Composing `iou=0.3`/c0.30 with that
refiner and exporting one CSV is the cheapest informative move; given the
bootstrap width, treat a single public submission as the tiebreaker rather than
trusting the local point estimate. **Done — see E024 above: the composition was
additive locally but the public score stayed at 0.39.**

2026-09-06: **both remaining high-value bets were run and both failed.**
E021 bounded the cheap-feature variant of E-C (learned proposal-quality head):
an out-of-fold linear probe over summary geometry features cannot beat the fixed
0.40 consensus gate at a matched proposal budget — `consensus_score` alone is the
best available ranker (OOF AUC 0.7252) and every geometry feature added to it
*reduces* recovered positives. E022 ran E-D (V12 detector training resolution,
1024 vs 1536, 2x2 train/infer matrix): the best arm `res1024_infer1536` scored
tuning PQ 0.432243 against the frozen control's 0.432823, a **tune delta of
-0.000580 with audit delta -0.019284**. Gate FAILED; nothing promoted; nothing
submitted from either.

The consequence is that **the two levers the 0.55 plan ranked highest are now
both measured and negative.** Recall is still the bottleneck (278 cutoff-lost,
228 missing-raw instances remain lost), but neither a cheap learned ranker nor
higher-resolution detector training reaches it. Public best remains **0.39** from
V11 continuation training. Remaining unrun levers are E-A (inference flags, cheap),
and the more expensive E-E/E-F/E-G family. **Superseded by E023 above: E-A was run
on 7 September and PASSED, reaching part of that recall pool.**

2026-09-06 earlier: **E020 disambiguation resolved — the V11 public gain is
continuation training, not proposal centering.** The already-trained
`gt_centered_e4` checkpoint was exported with no training and submitted as
**56050554**, scoring **0.39** — tying `proposal_centered_e4` (`56049359`)
despite 0.000688 *lower* local tuning PQ. Both V11 continuation arms therefore
land on the same public score, so the crop distribution is not the lever; four
more epochs of refiner training from the COCO checkpoint is. Proposal centering
joins instance conditioning as an unsupported V11 hypothesis. Public best remains
**0.39**; 2 submissions remain today.
[Package and receipt](../submission_ready/v11_gt_centered_2026-09-06/README.md).

Local→public calibration now has three points: +0.0012 → 0.38, +0.0045 → 0.39,
+0.0052 → 0.39. Local gains near +0.004–0.005 do move public score, confirming
the +0.01 screening gate was miscalibrated. Three points are still not a transfer
function. Also established: two-decimal public scoring **cannot resolve** local
differences of ~0.0007, so do not spend submissions separating arms that close.

2026-09-06 earlier: **V11 candidate submitted as 56049359, public score 0.39 —
a new best**, beating the long-standing 0.38. Submitted at 07:07:16.430 UTC,
COMPLETE at 07:11 UTC, 980 masks / 176 populated images, 3 submissions remaining.
The candidate **failed its predeclared promotion gate** (E019: +0.005177 tuning PQ
against a required +0.01) and remains **unpromoted**; it was exported by a separate
no-training notebook purely to measure public transfer. Its frozen control export
reproduced the accepted 0.38 CSV byte-for-byte, so the runtime is verified.
Instance conditioning was the weakest arm and is not carried forward.
[V11 execution and results](v11_execution_2026-09-06.md),
[submission receipt](../submission_ready/v11_2026-09-06/submission_receipt.json).

The +0.01 gate threshold now looks miscalibrated: V10 gained +0.0012 locally and
scored 0.38, V11 gained +0.005177 and scored 0.39. Recalibrate future gates
against this evidence. ~~Open follow-up: export and measure `gt_centered_e4` from
the same run to separate continuation training from proposal centering.~~
**Resolved by E020 above: continuation training.**

2026-09-06 earlier: **V10 submitted as 56044785**, `submission_v10.csv`, at
02:14:51.233 UTC / 07:44:51.233 IST. Status **COMPLETE**, public score **0.38**,
confirmed at 02:18:43 UTC. This ties the previous best; no public improvement is
established, and submission 56026663 remains the baseline.
Checksum matched the frozen package; one IPv4 upload succeeded, and **4 daily
submissions remain**. [Submission receipt](../submission_ready/v10_2026-09-06/submission_receipt.json).

2026-09-05 latest: detector consensus TTA submission `56026663` remains the public best at **0.38**. V8 original `56027025` scored **0.35**; V8 + COCO/TTA `56027435` completed scoring at **0.37**, below the best. V10 quality ordering + conservative V8 rescue completed with a modest local PQ gain and a validated 991-mask CSV, but is **unsubmitted, with no public score**. Preflight reported zero submissions remaining for this date. Results: [V8 optimization](optimization_2026-09-05.md), [V10 execution](v10_execution_2026-09-05.md).

| ID | Date | Method | Data | Key Params | Local Metric | Kaggle Score | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E001 | 2026-09-04 | Classical dark-structure baseline | Pending download | threshold + morphology | Pending | Pending | First submission-format validation. |
| E002 | 2026-09-04 | YOLOv11 segmentation prep | 707 train / 180 test images | 80/20 file split, best duplicate annotator | Prep only | Pending | Produced 566 train and 141 validation YOLO images. |
| E003 | 2026-09-04 | Kaggle GPU YOLOv11s-seg + crop refiner | Kaggle competition data | 30 YOLO epochs, 8 refiner epochs, validation confidence sweep | best_conf=0.15, best_pq=0.0028504863186230847 | Error | V1 fixed data path; V2 hit Kaggle P100/PyTorch CUDA incompatibility; V3 completed, but raw submission refs 56006718 and 56006788 failed with SubmissionStatus.ERROR. |
| E004 | 2026-09-04 | E003 non-overlap postprocess | Kaggle competition data | Remove/avoid overlapping predicted instances; 1,810 predicted filaments across 178 test images | Kernel validation was unreliable because of uint8 IoU overflow | 0.31 public | Submission ref 56007181 completed. This is live but below the 0.56 visible leaderboard top checked at 2026-09-04 12:52 IST. |
| E005 | 2026-09-04 | V4 validation/postprocess optimization | Kaggle competition data | Fix validation IoU overflow; compare yolo_only, refined_constrained, refined_free across conf 0.15-0.35; score-ordered non-overlap | Pending Kaggle run | Pending | Main hypothesis: free crop refiner expands masks beyond YOLO proposals and causes overlap/instance damage. Constraining refined masks to dilated YOLO proposals should preserve instance identity and produce valid CSVs. |
| E006 | 2026-09-04 | V4 submitted constrained refiner | Kaggle competition data | best_conf=0.25, best_mode=refined_constrained, validation PQ=0.408729; 1,293 predicted filaments across 177 test images | 0.408729 validation PQ | 0.34 public | Submission ref 56010439 completed. It improved E004 by 0.03, but the public gap to the 0.56 visible leader remains large. V4 likely improved precision while losing some recall. |
| E007 | 2026-09-04 | V5 longer YOLO training | Kaggle competition data | Increase YOLO from 30 to 80 epochs, patience 20, confidence sweep 0.10-0.45, same constrained refiner and non-overlap logic | best_conf=0.30, best_mode=refined_constrained, validation PQ=0.396896; 1,240 rows over 176 populated test images | 0.33 public | Submission ref 56012359 completed. Longer training alone underperformed V4, so the next useful variable is data/label handling rather than extending this exact setup again. |
| E008 | 2026-09-04 | V6 all annotators + single class | Kaggle competition data | Train on all non-leaking train annotator variants, write all labels as one class, keep V5 80 epochs/patience 20/wide confidence sweep | best_conf=0.40, best_mode=refined_free, validation PQ=0.426821; 1,094 rows over 176 populated test images | 0.35 public | Submission ref 56017396 completed. This is the current best public score, but the output is still conservative and remains below the visible 0.56 leader. |
| E009 | 2026-09-04 | V7 larger detector + candidate submissions | Kaggle competition data | `yolo11m-seg.pt`, batch 2, 60 epochs, patience 15, all train annotators, single class | 0.406043 PQ; refined_free conf 0.25; 1275 test masks | 0.34 public | Completed; submission 56024292 scored 2026-09-05. Below V6 on validation and public score. Larger detector alone did not help. |
| E010 | 2026-09-04 | V8 broader candidate export | Kaggle competition data | Keep V7 training recipe; emit top 5 validation candidates plus recall-hedged, precision-hedged, and best YOLO-only candidate within a wider validation margin | Prepared locally | Pending | Prepared while V7 is running. This does not change model training; it reduces submission-selection risk by preserving more alternatives from one expensive run. Do not push until V7 outputs are reviewed. |
| E011 | 2026-09-05 | V8 single-class flip augmentation and all-annotator validation | Same 566 train / 141 held-out files; evaluate all 217 held-out annotator entries | V7 training plus horizontal/vertical flips 0.5; constrained refiner conf 0.25 | Historical-target PQ 0.402650; 1124 masks across 175 test images | **0.35 public** (`56027025`) | COMPLETE. Original submission validated and uploaded before COCO/TTA changes. PQ targets differ from canonical evaluation. |
| E012 | 2026-09-05 | V9 V6 inference sweep, first run | Same V6 weights and file split; 94 tuning / 47 audit files | 312 resolution, flip-average, threshold, confidence and refinement variants | Historical all-annotator PQ 0.431663 vs 0.428245; audit gain 0.000476 | Not submitted | Complete; V6 control reproduced byte-for-byte. Higher resolutions regressed. Superseded for selection by canonical E013. |
| E013 | 2026-09-05 | Corrected V9 evaluation and crop postprocessing | COCO targets match all 8,199 supplied annotation areas exactly | Same 312 settings; canonical selection plus old legacy metric, compressed prediction cache, direct crop RLE | PQ 0.422512 vs 0.420713; audit delta -0.000019; validation 4.17× faster | Not submitted | Complete; unchanged historical metrics across all 312 settings. Inference gain too weak to promote. |
| E014 | 2026-09-05 | Canonical crop-refiner fine-tuning | V6 weights; original training files; 94 tune / 47 audit | Four epochs, AdamW 5e-5; epoch 3 selected; constrained threshold 0.50, crop TTA, confidence 0.40 | Canonical PQ 0.424581; audit 0.435251 | **0.36 public** | Submission 56025722; current best accepted model. Small canonical gain over V6/V9. |
| E015 | 2026-09-05 | Detector flips and local threshold calibration | Frozen E014 checkpoint and V6 detector; same tune/audit files | 126 settings; consensus selected, mask threshold 0.40, confidence 0.40 | PQ 0.437873 vs 0.424581; audit 0.450130 vs 0.435251 | **0.38 public** (`56026663`) | COMPLETE; current best public score. Exact E014 control counts and CSV SHA-256 reproduced. |
| E016 | 2026-09-05 | Disconnected-component splitting | Cached E014 predictions; full 141-file validation | Original masks versus minimum 20px components or two substantial components (10% / 25% area) | Best split PQ 0.418833 vs intact 0.424581; all policies regress | Not submitted | Rejected. Exact per-image baseline counts reproduced. |
| E017 | 2026-09-05 | V8 detector + accepted COCO refiner + consensus TTA | Same training/validation files; frozen V8 detector and E014 refiner | 48 settings; threshold 0.45, confidence 0.25; Ultralytics 8.4.140 | PQ 0.409296; audit 0.429404; weaker than E015 | **0.37 public** (`56027435`) | COMPLETE; 1070 masks across 175 test images validated and submitted as requested. Public score below E015's 0.38. |
| E018 | 2026-09-05 | V10 quality ordering + conservative V8 rescue | Frozen V6/V8 raw caches and E014 refiner; 94 tune / 47 audit files | `quality_rank_rescue40`; 9 of 15 declared possibilities evaluated; no training | PQ 0.439246 vs 0.437873; audit 0.451837 vs 0.450130; all +8 TP / +8 FP / -8 FN | **0.38 public** (`56044785`, submitted 2026-09-06) | COMPLETE; ties E015's displayed public score, so no public improvement established. Exact E015 control hash/counts reproduced, external CSV validation PASSED: 991 masks / 176 populated images. Frozen CSV submitted once; 4 daily submissions remain. Paired local 95% delta intervals include zero. Training follow-up deferred. |
| E019 | 2026-09-06 | V11 proposal-aware / instance-conditioned refiner continuation | Frozen V6 detector and 321 raw caches; 566 training filenames; 94 tune / 47 audit | 3 arms x 4 epochs from the COCO refiner checkpoint; FP32, batch 16, AdamW 5e-5, cosine, 0.5 BCE + 0.5 Dice; crop 256; 6,380 samples; 17 predeclared evaluations | Best `proposal_centered` e4: tune PQ 0.438001 (+0.005177 vs frozen V6, +0.003936 vs V10), audit 0.454884 (+0.004755 / +0.003047); 629 TP / 206 FP / 505 FN | **0.39 public** (`56049359`) — **new best** | **Gate FAILED** (required +0.01 tuning PQ); candidate remains **unpromoted**. The experiment run exported nothing; a separate no-training notebook (`solar-filament-v11-export` v1) measured the frozen checkpoint, reproducing the accepted 0.38 control CSV byte-for-byte first. 980 masks / 176 populated images, independent validation PASSED. Instance conditioning was the weakest arm, with e2/e3 below control; full-crop growth added only ~+0.0002. Proposal-centered beat matched GT-centered by just +0.00069, so the cause of the public gain is ambiguous: measure `gt_centered_e4` next. The +0.01 threshold looks miscalibrated given V10 (+0.0012 -> 0.38) and V11 (+0.0052 -> 0.39). |
| E020 | 2026-09-06 | V11 `gt_centered_e4` export-only disambiguation | Same frozen V6 detector, 321 raw caches, canonical annotations; no training | Export-only; checkpoint `gt_centered_e4.pt` pinned; threshold 0.40, confidence 0.40, min area 20, crop 256, 4+4 flips — identical to the 0.39 run except the refiner checkpoint | Tune PQ 0.437313 (+0.004489 vs frozen V6); 628 TP / 207 FP / 506 FN | **0.39 public** (`56050554`) | **Verdict: the V11 gain is continuation training, not proposal centering.** Ties `proposal_centered_e4`'s 0.39 despite 0.000688 lower local tune PQ, in a strict single-variable comparison. Gate still FAILED (same run, +0.005177 vs required +0.01); candidate **unpromoted**. Control export reproduced the accepted 0.38 CSV byte-for-byte and is byte-identical to the earlier export run's control. Checkpoint hash `80c17bc1…` confirmed against the local copy. 980 masks / 176 populated images, independent validation PASSED. Also establishes that two-decimal public scoring cannot resolve ~0.0007 local deltas. |
| E021 | 2026-09-06 | E-C feasibility: can a cheap-feature ranker beat the 0.40 consensus gate? | 94 tuning files; frozen V6 flip_consensus grouped proposals; 2,113 proposals (893 oracle-positive / 1,220 negative); 94 cache files hashed; annotations `5da9e92b…` | Two stages, no GPU, no training of any CNN: (1) fixed-feature AUC study over 9 summary features; (2) 5-fold out-of-fold linear probe grouped BY FILE, seed 42, at a **matched accepted-proposal budget of 544** (identical count to the 0.40 gate) | Gate baseline: 544 accepted, 406 positives, precision 0.7463, 487 positives rejected. Best single feature = `consensus_score` / `view_support`, AUC 0.7261, delta_vs_gate **0**. Best OOF model = `consensus_only`, AUC 0.7252, delta **-1**. `consensus_plus_geometry` delta **-235**; `geometry_only` delta **-249** | Not submitted | **NOT SUPPORTED (cheap-feature version).** No cheap geometric feature carries incremental signal over the consensus score, and combining them actively destroys ranking (AUC drops to 0.45 / 0.42 — worse than random, i.e. the geometry correlates with the wrong sign once fitted). At a matched budget the best learned probe recovers **one fewer** true positive than the fixed threshold it was meant to replace. Scope limit stated honestly: labels use **pre-refiner proposal geometry**, and a trained CNN mask-scoring head reading mask pixels could exceed a linear probe, so this bounds the cheap variant of E-C, **not every possible E-C**. Cost: minutes of CPU, zero GPU. |
| E022 | 2026-09-06 | E-D / V12 detector training-resolution comparison (1024 vs 1536) | Warm-started from pinned V6 (`0abf568f…`); dataset `65106740…` = 937 train / 141 val / 6,606 train instances; refiner `8d008053…`; 94 tune / 47 audit; 2 GPUs, global batch 4 | Two arms x 20 epochs, `resume=False`, AdamW lr0 1e-4, lrf 0.1, warmup 1, `cos_lr=False`, nbs 64, seed 42 deterministic, single class, `mask_ratio=2`, `overlap_mask=False`, AMP, identical augmentation to V6. Each checkpoint evaluated at **both** 1024 and 1536 inference (2x2) to separate train-res from infer-res, against one frozen control. Gate predeclared +0.01 tune with nonnegative audit; frozen before audit | Control tune 0.432823 / audit 0.450130. Best arm `v12_res1024_infer1536`: tune **0.432243** (delta **-0.000580**), audit **0.430845** (delta **-0.019284**), 637 TP / 249 FP / 497 FN. Others: `res1536_infer1536` tune 0.428023; `res1024_infer1024` 0.425146; `res1536_infer1024` 0.415018. Train cost 1,388.81 s (1024) + 2,759.04 s (1536) | Not submitted | **Gate FAILED, nothing promoted.** Higher-resolution *training* is not a lever: the 1536-trained arm lost to the 1024-trained arm at **both** inference resolutions, and the 1024 arm's best configuration still failed to beat the frozen control. Raising *inference* resolution to 1536 does buy recall (FN 508 -> 497, TP 626 -> 637) but pays for it in precision (FP 211 -> 249), netting negative PQ — the same recall/precision trade that killed the naive threshold drop. This falsifies the predeclared E-D hypothesis and removes the circumstantial "HDJoJo's advantage is 2048 training" explanation for the public 0.55 cluster. `res1536_infer1536` posted the run's best *audit* PQ (0.436854) while losing on tune; selection is tune-only by protocol, so this does not rescue the arm. Total ~6,240 GPU-s. |
| E023 | 2026-09-07 | E-A inference-flag ablation on the frozen detector (`retina_masks`, `agnostic_nms`, box `iou`)| Frozen V6 detector (`0abf568f…`) and canonical COCO refiner (`8d008053…`); 94 tune / 47 audit; no training | 4 predeclared cells (`retina_masks` always True; `agnostic_nms` x box `iou` 0.7/0.3), each swept over the existing 6 thresholds x 7 confidences = 42 candidates. Gate predeclared +0.003 tune with nonnegative audit | Control reproduced E015 exactly (tune `[426.5473399071961, 626, 211, 508]`, audit `[182.75261599994468, 265, 88, 194]`). Best cell box `iou=0.3` at t0.40/**c0.30**: tune **0.436726** (delta **+0.003902**), audit **0.459054** (delta **+0.008925**); 684 TP / 312 FP / 450 FN | Not submitted | **Gate PASSED — first lever since E018 to clear its bar, and the first of any kind to move recall.** Box `iou=0.3` recovers **58 of the 508 missed tune instances** (recall 0.5520 -> 0.6032) for +101 FP, and unlike every prior recall attempt the PQ trade is *net positive*. Audit agrees and is positive at **every** confidence, which the failed E022 arm never managed. Three caveats, recorded before any submission. (1) **`agnostic_nms` is inert**: the `agnostic`/`noagnostic` arms are bit-identical across all 42 candidates (maxdiff 0.0) because the detector is `single_cls=True`, making class-agnostic NMS a no-op by construction — the 2x2 grid is really 1x2 and only box `iou` carries information. (2) **The gain is confidence-coupled, not a pure flag win**: at the control's own c0.40 the `iou=0.3` cell is *worse* (-0.000922); it wins only after the confidence floor drops to 0.30. It is nonetheless stable along the threshold axis (`iou=0.3` wins at every threshold 0.35–0.60, always at c0.30). (3) **The margin is inside the noise**: paired per-image bootstrap (10,000 resamples, seed 42) gives tune 95% CI **[-0.005372, +0.013211]**, P(delta>0)=0.79, P(delta>=0.003)=0.57; audit CI [-0.007202, +0.026270], P(>0)=0.85. The point estimate clears the gate; the interval includes zero, as in E018. Calibration against the 0.39 submissions: **-0.001275 tune below** V11 `proposal_centered_e4` (0.438001) but **+0.004170 above it on audit**. The two levers are orthogonal (inference flags vs refiner weights), so composing them is the natural next step. Cost ~1,570 GPU-s. First attempt errored at 04:47 UTC on a mis-specified control assertion (diffed the flip_consensus control against the single-view t0.50 baseline); measurement was never at fault. |
| E024 | 2026-09-07 | V13 composition: E-A box `iou=0.3` flags x V11 `proposal_centered_e4` refiner | Frozen V6 detector (`0abf568f…`), V11 checkpoint (`fae496b9…`, run identity verified, epoch 4), canonical annotations; 94 tune / 47 audit; no training | Regenerated `iou=0.3` proposals for all 141 validation + 180 test images (the 321 frozen caches are `iou=0.7`, so they cannot express this cell); cell-scoped signed cache ids. Predeclared 42-candidate grid (6 thresholds x 7 confidences) bound at runtime to `detector_tta.LOCAL_*`; tune-only selection. Two predeclared bars: SUBMIT_BAR (tune >= E-A 0.436726, audit >= E015 0.450130) and PROMOTION_GATE (tune >= V11 + 0.003, audit >= V11) | Selected `t0.45_c0.30`: tune **0.441777**, audit **0.464644**, 686 TP / 309 FP / 448 FN. Beats **all three** parents on **both** groups — vs E015 +0.008954 / +0.014514, vs E-A +0.005052 / +0.005590, vs V11 **+0.003776 / +0.009760** | **0.39 public** (`56076685`) — **ties, no improvement** | **Both bars CLEARED — the first candidate in the project to pass a promotion gate — yet the public score did not move.** That is the result: the largest verified local gain to date (+0.0090 tune over the accepted baseline) produced no public change. **E-A and V11 are additive locally**, confirming they are genuinely orthogonal mechanisms (proposal recovery vs mask refinement) rather than two routes to the same gain; the composition is not merely the better parent. But the public split does not reward it. Calibration now has a fourth and first-negative point: +0.0012 → 0.38, +0.0045 → 0.39, +0.0052 → 0.39, **+0.0038 over the 0.39 → still 0.39**. Local tuning gains in the +0.003–0.010 band are no longer reliable evidence of public movement. Two readings remain open and this run cannot separate them: either the gain does not transfer, or two-decimal rounding hides it (0.39 spans 0.385–0.394, so a true +0.004 can be invisible). Verification was clean throughout: the control arm ran FIRST, reproduced E015's exact counts (tune `[426.5473399071962, 626, 211, 508]`, audit `[182.75261599994462, 265, 88, 194]`), and its exported CSV hashed to the accepted 0.38 file `bf11690e…` byte-for-byte; independent CSV validation PASSED with 1,198 masks / 177 populated images, non-overlapping, all RLEs decoding. Paired bootstrap vs E015 (10,000 resamples, seed 42): tune +0.008954 CI [-0.001171, +0.018949] P(>0)=0.960; audit +0.014514 CI [-0.002776, +0.033161] P(>0)=0.951 — much stronger than E-A alone (0.79) but still grazing zero, and the flat public result vindicates that caution. **V13 is a local improvement with a verified runtime and a neutral public result; do not promote it as a public gain.** Cost ~3,470 GPU-s. [Package and receipt](../submission_ready/v13_2026-09-07/README.md). |

| E025 | 2026-09-07 | V14 dual-head refiner: ResNet34 encoder + mask head (deep supervision) + IoU head replacing the fixed 0.40 confidence cutoff | Frozen V6 detector, V11 `proposal_centered_e4` warm start (`CHECKPOINT_SHA256` pinned), same 566 training files, 94 tune / 47 audit; cohort `8033aec4…`, 6,380 samples | Two arms x 8 epochs, identical cohort/order/seed asserted per epoch (`sample_order_sha256` matched across arms at every epoch). arm_a 0.5 BCE + 0.5 Dice; arm_b 0.5 BCE + Focal-Tversky (α 0.3 / β 0.7 / γ 0.75). IoU head = MSE on predicted-vs-observed mask IoU, weight 0.2, reading `x4.detach()`. 21 predeclared candidates per arm: `mask_threshold` {0.40,0.45,0.50} x `iou_gate` {0.20…0.50}. Proposal clamp removed; proposal supplied as a 4th hint channel | Control verified FIRST and reproduced E015 exactly (tune `[426.5473399071962, 626, 211, 508]`, audit `[182.75261599994462, 265, 88, 194]`). Best `arm_a/m0.50`: tune **0.244906**, audit **0.238574**, 725 TP / **2051 FP** / 409 FN, recall **0.6393** (+0.0344 vs V13), meanIoU **0.6604** (−0.0251 vs V13). Best arm_b `m0.50`: tune 0.231108. IoU-head AUC **0.5087** vs E021's `consensus_score` 0.7252 | Not submitted (SUBMIT_BAR not cleared; no CSV exported by design) | **FAILED HARD — tune PQ 0.2449 against V13's 0.4418, the largest regression in the project.** Two independent defects, both diagnosed offline from the returned artifacts at zero GPU cost. **(1) The `iou_gate` never fired.** All 21 candidates per arm collapse to just 3 distinct results, one per `mask_threshold`; every `iou_gate` from 0.20 to 0.50 is byte-identical. Measured from `v14_iou_head.csv`, predicted IoU spans only **[0.5934, 1.0000]** (arm_a) and **[0.5790, 0.7210]** (arm_b), so every declared gate filters **0 of 1773** / **0 of 1840** proposals. Nothing was ever filtered, all ~2,776 grouped proposals were kept, and FP exploded 309 → 2051. **Root cause: train/serve population mismatch.** The IoU head trains on `v11.build_training_cohort`, which by construction emits only positive-matched proposals (`match_positive_proposals` `min_iou=0.30`; measured 6,380 records, `matched_iou` min 0.301 / mean 0.618 / max 0.914). The head never saw a bad proposal, so it learned to emit a near-constant ~0.62–0.66 — exactly the degenerate case `ranking_auc`'s tie handling was written to expose. **(2) The head diagnostic was invalid, so the AUC never measured the head.** In `evaluate_arm` the head rows are built as `zip(predicted_iou[:len(actual)], actual)`, but `predicted_iou` is in raw proposal order while `actual` comes from `instances_for_candidate()`, which **sorts by −predicted_iou and drops entries** (<20px, occupancy). Predicted and actual are misaligned; measured pearson(pred,actual) is **−0.0062** (arm_a) / **+0.0397** (arm_b). The reported 0.5087 is not evidence about the head's ranking ability, only about the misalignment. **Offline ceiling analysis, and why the obvious fix is not worth GPU.** Holding tp/iou_sum fixed, clearing SUBMIT_BAR needs FP ≤ **309**, i.e. the gate must remove **85.0%** of the 2,051 FP; at exactly V13's FP=309 the PQ is **0.441690 vs V13's 0.441777 — a dead tie, not a win**, and only at E015's FP=211 does it reach 0.462600. An **oracle** gate ranking by true IoU (keep top 30%) reaches **0.5679**, so headroom exists in principle, but a noise-model sweep over the same population (score = true_iou + gaussian noise, 5 trials/point, tie-aware AUC) prices the requirement: AUC 0.9189 → PQ 0.4477 (bare V13 parity), AUC 0.9499 → 0.4767, while AUC 0.7414 → only 0.3349. **A fixed gate must hit AUC ≈0.92 merely to tie V13 and ≈0.95 to gain +0.035**, against E021's best measured cheap ranker of 0.7252. **(3) The encoder was never pretrained — the mask head was trained almost from scratch.** `build_dual_head` does `DualHeadRefiner(pretrained=warm_start is None)`, and a warm start *was* supplied, so `resnet34(weights=None)` — **ImageNet initialisation was silently disabled by the presence of the warm start.** The warm start then transferred only **131 of 298 tensors** (logged `warm_start_tensors_loaded 131 / skipped 167`), because V11's checkpoint is a ResNet18 U-Net and only shape-matching tensors move. So the ResNet34 encoder was mostly **random**, and it got 8 epochs. `v14_design_2026-09-07.md` explicitly specified the opposite ("the encoder is ImageNet-init and the decoder is fresh"); the code contradicts the design. **This is why meanIoU regressed (0.6604 vs V13's 0.6855), and it means the mask-quality hypothesis was never actually tested.** Accordingly: `ceiling_analysis_2026-09-07.md` item 1 (more capacity + FN-weighted loss moves meanIoU 0.686 → 0.78) is **NOT falsified — it is untested**, and **E-F / Focal-Tversky is also NOT retired**: arm_b lost to arm_a on every axis (tune, audit, recall, meanIoU, FP), which is a clean comparison because budgets were asserted identical, but both arms sat on a crippled random encoder, so the loss comparison is confounded by initialisation and says nothing about Focal-Tversky on a properly initialised model. Recall did move (+0.0344), confirming the 278-instance cutoff pool is reachable, but only by keeping everything. **Verdict: V14 measures three of its own bugs, not its hypotheses. Every headline conclusion is a defect report.** The correct next step is therefore **not** the detector `mask_ratio=1` retrain but a corrected re-run of this same design (ImageNet init + negatives in the scoring cohort + aligned diagnostic), because the expensive question it was built to answer is still open. Cost ~3,413 GPU-s. Artifacts `outputs/kaggle_v14_full/`, kernel `abhimanyu122/solar-filament-v14-dualhead` v1, script sha256 `5453c82f…`. |

| E026 | 2026-09-08 | V16 seed replication + lever-overlap: is `mask_ratio=1` real, and is it redundant with V13's box `iou=0.3`? | Seed 2 full V16 rerun (`seed=20260908`, kernel `abhimanyu122/solar-filament-v16-mask-ratio-seed2` v1, run identity `503d1bb8…`), same 566/141 grouped split and same frozen control as seed 1; plus a zero-GPU cache-only comparison against V13's real `iou=0.3` caches (141/141 validation files, re-pulled) | Two parts. **(1) Replication:** identical harness to seed 1, only the training seed differs (checkpoints differ: arm_r1 `097a57ed…` vs `ae476e66…`); paired per-image bootstrap of arm_r1 vs frozen control, 10,000 resamples, seed 42, resampling **files** — plus a cross-seed sign test. **(2) Lever overlap** (`scripts/analyze_lever_overlap.py`, no GPU): grouped pre-refiner recall / meanIoU / oracle PQ for A = frozen V6 @ box `iou=0.7`, B = frozen V6 @ box `iou=0.3` (V13's shipped lever), C = arm_r1 `mask_ratio=1` @ `iou=0.7`, on the same 141 files and same GT | **Harness anomaly was a one-off.** arm_r2 (same `mask_ratio` as control) vs control: seed 1 **−0.007677**, seed 2 **−0.000715** — seed 2 reproduces the control, so the harness is sound, but the noise floor is ~±0.007 tune PQ. **Raw proposal IoU replicates:** arm_r1 mean +0.010864 / +0.015282, max +0.010979 / +0.008367 vs control — both seeds, both axes positive. **Tune PQ does not:** arm_r1 − control = **−0.001220** (seed 1) vs **+0.001972** (seed 2), sign flips. Every bootstrap interval includes zero (seed1/tune CI [−0.009095, +0.006881] P(>0)=0.389; seed2/tune CI [−0.005993, +0.010044] P(>0)=0.692). **Lever overlap, tune group:** A recall 0.7416 / meanIoU 0.6371 / oraclePQ 0.5426; B recall 0.7002 / 0.6366 / 0.5243; **C recall 0.7795 / 0.6442 / 0.5644** | Not submitted (neither seed `promotable` or `export_authorized`; both "below submit bar; do not export") | **`mask_ratio=1` is a real but small proposal-quality lever whose downstream PQ effect is unmeasurable at this sample size — and it is NOT redundant with the lever already shipped.** The two questions separate cleanly. **On reality:** the mechanistic, single-mechanism measurement (raw proposal IoU, pre-refiner, 1,593 proposals) moves in the predicted direction in **both** seeds on **both** mean and max, and the 0.8734 cap that `ceiling_analysis` §4 treated as structural is exceeded in both (0.8844, 0.8818). arm_r1 also beats arm_r2 on tune PQ in both seeds (+0.006458 / +0.002687). That is four consistent signs across two independent draws. **On magnitude:** the end-to-end tune PQ delta vs the frozen control is **not** distinguishable from zero — it changes sign between seeds and every 95% interval straddles zero, because run-to-run detector-training noise (±0.007, now priced by seed 1's arm_r2) is several times the effect. So the honest statement is that the lever exists at the proposal stage and its PQ consequence is unmeasured, not that it is worth +0.002. **On redundancy — the decisive new evidence:** C > A > B on **recall, meanIoU, and oracle PQ simultaneously**. `mask_ratio=1` at the default box `iou=0.7` beats V13's shipped `iou=0.3` by **+0.0793 recall** and **+0.0401 oracle PQ** while emitting *more* proposals (3,400 vs 2,347), and it wins at **every** confidence 0.25–0.55 in the sweep. The two levers are therefore not two routes to the same recall: `mask_ratio=1` dominates the one currently in the pipeline. Note also that B (`iou=0.3`) scores *below* A (`iou=0.7`) on this pre-refiner oracle metric, which is consistent with E023's finding that its gain is confidence-coupled and appears only after the refiner and the 0.30 floor. **Reading for V17:** the case for a full-pipeline re-measurement rests on the pre-refiner dominance result, not on the arm-vs-control PQ deltas, which are noise. A V17 must retrain the refiner on `mask_ratio=1` proposals (V12/V13's frozen refiner never saw them) and, to be worth its GPU, must be powered against the ±0.007 floor now measured — a single-seed tune-PQ comparison cannot clear the +0.010 promotion gate credibly. **Two self-inflicted measurement defects found and fixed before any conclusion was drawn.** (a) The first version of `analyze_v16_seed_replication.py` **overwrote rather than summed** the multiple `image_id` rows per `file_name` (annotator variants), yielding a confident but wrong seed1/tune interval (observed +0.003929 against the true −0.001220). Fixed by summing, and `_assert_matches_selection()` now verifies the re-aggregated PQ equals `v16_selection.json` **exactly** before any bootstrap runs — the V14-class failure mode (a diagnostic describing a different population than it claims) is now caught loudly rather than reported. (b) `outputs/kaggle_v13_cache_pull` held only **108 of 141** required caches (an interrupted earlier download, compounded by the ops CLI's default `--file-pattern` excluding `.gz`); re-pulled to 141/141, and seed 1's `v16_image_diagnostics.csv` had never been pulled at all. Both analyses were re-run only after the inputs were complete. Cost: **zero GPU** for both analyses (seed 2's training was already complete and unpulled). Quota 51,996 s of 108,000 remaining, refresh 2026-09-12. Local suite **304 passed / 1 skipped** (from 186), including 13 new tests in `tests/test_v16_seed_replication.py`. Artifacts `outputs/kaggle_v16_full_seed2/`, `outputs/v16_proposal_analysis/seed_replication.json`, `outputs/v16_proposal_analysis/lever_overlap.json`. |

| E027 | 2026-09-08/09 | V17 composition submitted and measured: `mask_ratio=1` detector x V13's recipe | Frozen V6 detector + both V16 `arm_r1` checkpoints (seed 42 `ae476e66…`, seed 20260908 `097a57ed…`), V11 `proposal_centered_e4` refiner (`fae496b9…`), canonical annotations; 94 tune / 47 audit; **no training** | 84 predeclared candidates = 2 box-`iou` cells {0.3, 0.7} x 6 thresholds x 7 confidences, per seed; cache identity spans `detector_sha \| mask_ratio \| cell \| split \| name`; tune-only selection on the pre-registered primary seed; SUBMIT_BAR tune ≥ 0.441777 and audit ≥ 0.450130; PROMOTION_GATE tune ≥ V13 + 0.010 and audit ≥ 0.464644 | Selected `mr1_iou030/t0.45_c0.30`: tune **0.448703**, audit **0.470763**, 704 TP / 306 FP / 430 FN, recall 0.6208, meanIoU 0.6833. Beats **all five** parents on **both** groups (vs V13 +0.006926 / +0.006119; vs E015 +0.015879 / +0.020634). Seed 2 corroborates: tune 0.444860 / audit 0.468300, spread 0.003843, both seeds independently clear SUBMIT_BAR | **0.39 public** (`56103401`) — **ties, no improvement; fifth consecutive 0.39** | **SUBMIT_BAR cleared, PROMOTION_GATE not, and the submission answered its actual question in the negative.** V17's design doc justified one submission as "a calibration point on whether a proposal-stage gain transfers where inference-stage gains did not" — its edge was mechanistically grounded and **replicated across two independent training seeds** (E026's raw proposal IoU: mean +0.0109/+0.0153, max breaking the 0.8734 cap in both), unlike V13's inference-side rearrangement. **It transferred no better.** The decisive consequence is not about `mask_ratio=1` but about the measurement model: **`ceiling_analysis_2026-09-07.md` §1's constant local→public offset of −0.050 ± 0.003 is falsified.** Local tune PQ has now risen **+0.015880** (0.432823 → 0.448703) across five submissions while the public score moved exactly once, by one display step. `scripts/analyze_public_transfer.py` proves the feasible constant-offset region is **empty** under both round-to-nearest and truncation display models, while being **non-empty for the first four points** — so V17 specifically is the falsifier (pinned by test). For affine `public = a·local + b`, feasible slopes are **a ∈ [0.01, 0.87]** with **a = 1.00 ruled out**; at the best feasible slope local 0.60 yields public ≈ **0.527**, and public 0.55 requires local ≥ **0.6269**. **Every target in `ceiling_analysis` §8 was therefore computed from a broken transfer function and understates what is required.** The same script also finds **selection is overfitting the 141-image tune set**: 17 of 84 seed-1 candidates sit within E026's ±0.007677 training-noise floor of the top, and seed 2's tune-argmax lands on a **grid edge** (`t0.60`) while disagreeing with its own audit-argmax (giving up 0.002157 audit). Verification was clean: the frozen control ran first and reproduced E015's exact counts (tune `[426.5473399071962, 626, 211, 508]`, audit `[182.75261599994462, 265, 88, 194]`) with its CSV hashing to the accepted 0.38 file `bf11690e…` byte-for-byte; independent validation PASSED (1,215 masks / 177 populated images). Cost: inference only. [Package and receipt](../submission_ready/v17_2026-09-08/README.md). |

| E028 | 2026-09-09 | Is the 0.39 plateau structural? Detector complementarity, a realistic union ensemble, and the V18 feasibility post-mortem | **(a,b)** the six proposal populations already on disk — V6@`iou`{0.7, 0.3} and `arm_r1` seed1/seed2 @`iou`{0.7, 0.3} — on the same 141 files with the same GT, grouped pre-refiner; **(c)** returned logs of the two errored V18 kernels | Three parts, **all zero GPU for (a) and (b)**. **(a)** `analyze_detector_complementarity.py`: per-GT-instance match record across all six sources at IoU>0.5, giving per-source recall, union recall, the count missed by *every* source, and union oracle PQ. **(b)** `analyze_union_ensemble.py`: the honest ensemble — concatenate all six pools, **cross-source NMS at IoU 0.5**, then sweep the shipped 7-confidence grid and compare against the best single source measured identically. **(c)** parse both V18 failure traces and compare the full run's instances-per-batch against what the smoke probes observed | **(a)** tune: best single `MR1s2_iou070` recall 0.7875; **UNION recall 0.8131** (+0.0256); **212 of 1134 (18.7%) missed by every source**; union oracle PQ **0.5849 @ zero FP**, **0.5092 @ V17's FP=306**. audit: union recall 0.8410, 73 of 459 (15.9%) missed by all. **(b)** UNION_all beats the best single source at **1 of 7** tune confidences (max **+0.0005**) and 2 of 7 audit. **(c)** both runs OOMed in `single_mask_loss`'s BCE; smoke probes saw ≤24 and ≤13 instances/batch vs the full run's **31** | Not submitted (no candidate produced) | **The last cheap lever is now measured and dead, and the plateau is substantially structural.** **On complementarity:** the six detectors genuinely fail on *different* instances — union recall exceeds the best single source by +0.0256 — so `ceiling_analysis` §8's untested "E-H fold ensemble" was not a priori worthless. **But its ceiling is too low to matter:** with *perfect* selection and *zero* false positives the union pool yields only **0.5849** local PQ, and at V17's actual FP count **0.5092**. Both sit below the 0.60 that even the now-falsified transfer model demanded, and far below the ≥0.6269 that E027's affine bound implies for public 0.55. **A flawless six-way ensemble cannot reach the target.** **On the realistic version:** it is a clean negative — merging the pools raises recall but the added false positives cost more than the recall earns, so the union loses to a single detector at 6 of 7 tune confidences. This is the same recall/precision trade that killed the naive threshold drop (E015) and 1536 inference (E022), now measured for proposal union. **E-H is closed.** Note the honest floor this establishes: **18.7% of tune instances are invisible to every detector we have**, so no selection rule, refiner or ensemble over these proposals can exceed 0.8131 recall. **On V18:** both full runs are **resource failures, not defects** — each verified the frozen control *exactly* (tune 0.432823, audit 0.450130, raw mean IoU 0.5253, raw max 0.8734) before dying in epoch 1 with `torch.OutOfMemoryError` inside Ultralytics' `binary_cross_entropy_with_logits`. The first ran at `GLOBAL_BATCH=2`, took Ultralytics' auto-fallback to `batch=1` at step 7, and still OOMed near step 555/937 (1.17 GiB requested, 1.16 free); revA set `batch=1` from the start **and** `expandable_segments:True` and still OOMed (2.85 GiB requested, 1.91 free) at t=1842 s. **Root cause of the bad feasibility call:** the smoke probes ran at `fraction=0.2`/`0.02` and observed at most 24 and 13 instances per batch, while the full 937-image run reaches **31**; at `mask_ratio=1` the loss holds masks at full proto resolution so its memory scales with **instances per image**, exactly the quantity subsampling understated. Since `batch=1` is already the floor, **`imgsz=1536` + `mask_ratio=1` does not fit a 14.56 GiB T4 at any batch size** — this configuration needs a larger accelerator, not a retry. A retry at unchanged settings would burn quota to reproduce the same traceback. Cost ~1,550 GPU-s for the two failures, **zero** for (a) and (b). Quota 45,292 s of 108,000, refresh 2026-09-12. Local suite **385 passed / 1 skipped** (from 304), adding 25 tests in `tests/test_transfer_and_ensemble_analyses.py` which pin the NMS/FP arithmetic and assert the committed reports still support these conclusions. Also caught a repeat of E026's cache trap: `outputs/kaggle_v17_full/proposal_cache` held only **323 of 744** signatures with partial coverage *per cell* (a silently truncated pull); re-pulled complete to `outputs/kaggle_v17_cache_pull/proposal_cache` and every source is now asserted complete before any number is computed. Artifacts `outputs/public_transfer_2026-09-09.json`, `outputs/v16_proposal_analysis/detector_complementarity.json`, `outputs/v16_proposal_analysis/union_ensemble.json`. |

| E029 | 2026-09-10 | V19: Mask2Former (ResNet/Swin-T query-based segmentation) as the "genuinely different detector family" `RESUME_TOMORROW.md` (2026-09-09) identified as the only lever with any remaining chance | Kaggle T4, `facebook/mask2former-swin-tiny-coco-instance` checkpoint (2-class head reinitialised, COCO backbone retained), same 566 train / 94 tune / 47 audit split as every V6-V18 run, same `src/pq_eval.py` PQ implementation. `imgsz=1024` train, 100 object queries, batch 2 / grad-accum 4 | Smoke test first (worst-case 26-instance image, peak GPU 11.39 GiB of 14.56 GiB budget — cleared, unlike V18). Timing probe: ~872 s/epoch measured on real data, authorized a 20-epoch full run under a 32,400 s (9 h) timeout. Full run: `abhimanyu122/solar-filament-v19-mask2former-full`, actual wall time 16,177 s (~4.5 h), well inside budget and quota | **Best tune PQ 0.153578 (epoch 13 of 20), audit PQ 0.150346** (343 TP / 1634 FP / 385 FN at best epoch; audit 158 TP / 782 FP / 183 FN). Per-epoch tune PQ oscillated noisily between 0.0752 and 0.1536 with no clear convergence trend across 20 epochs; train loss fell 31.5 → ~23 and plateaued by epoch ~10 while PQ kept swinging, so the loss curve and the PQ curve decouple. `cleared_submit_bar=False`, `cleared_promotion_gate=False` | Not submitted (SUBMIT_BAR gates CSV export; not cleared) | **SUBMIT_BAR not cleared, by a wide margin — roughly a third of the 0.448703 bar, not a close miss.** This is the predeclared "what happens if this also fails" case from `v19_design_2026-09-09.md`: the doc's own protocol says report it as a second local→public transfer data point and move on, not iterate on this configuration. Two things distinguish this from a training bug worth debugging further before concluding: **(1) FP volume is structurally out of proportion.** At its best epoch, V19's tune FP is 1634 against V17's control FP of 211 (frozen) / 306 (mask_ratio=1 composed) — roughly 5-8x higher, on a *smaller* effective proposal count (100 queries/image vs YOLO's much larger raw proposal pool pre-NMS), so this is not "more proposals, more noise" but the query-based decoder placing a large fraction of its 100 queries on non-filament pixels even at epoch 13, the run's best point. **(2) 20 epochs from a COCO-pretrained backbone is likely undertrained for a 2-class head learning a domain this different from natural-image COCO categories** (707 total images, 566 train) — the per-epoch PQ trace never shows a stabilizing trend, consistent with a model still exploring rather than converging, but this project's own process discipline (`ceiling_analysis`, E025) is not to extrapolate a longer run would fix it without evidence, since V14/V15's "just needs more capacity" instinct was wrong twice already for reasons diagnosed only after the fact. **Verdict: do not extend this run (more epochs, hyperparameter sweep) without a specific, evidence-backed hypothesis for the FP volume** — that is a new, costed experiment, not a continuation of this one. What this run *does* establish cleanly: the "different architecture family" bet, run exactly as designed (smoke-tested correctly, unlike V18; timing-probed correctly; full run completed without a resource failure), is a clean negative at the scale tested. Combined with E027/E028, **every lever identified by this project across 19 versions has now been tried or explicitly closed by measurement**, and none has moved the public score off 0.39. Cost: ~16,177 s GPU (~4.5 h) for the full run, plus ~1,200 s for the cancelled timing probe and a few minutes for the smoke test. Artifacts `outputs/kaggle_v19_full_pull/v19_mask2former/` (training_history.json, v19_full_result.json, best_model/). Quota consumption not yet re-verified locally; recheck the live allowance before deciding on further Kaggle spend. |
