# Solar Filament Segmentation Challenge 2026

Repository for the Kaggle Solar Filament Segmentation Challenge 2026.

**Active Azure run (2026-09-14): V41 learned contextual texture/topology G2 is training on one isolated Tesla T4.** The frozen runner is detached under `/mnt/solar-v41-context-topology` on shared VM `rogii-gpu`, physical GPU 0. Runtime/hash/firewall checks passed; step 0 produced PQ 0 with 0/42/308 TP/FP/FN and finite optimizer logs were observed through at least step 100. GPU 0 was active at 7,019 MiB/100%, while GPUs 1-3 remained idle. No result exists yet. Do not relaunch or deallocate the shared VM. Exact monitoring/retrieval: [`docs/v41_azure_handoff_2026-09-14.md`](docs/v41_azure_handoff_2026-09-14.md). Full training remains forbidden unless every bounded fit-only gate passes. V17 submission `56103401` remains public best at **0.39**.

**Active experiment (2026-09-14): V41 learned contextual texture/topology G2 is running privately.** V39+V40 leave 121/212 misses unsupported; V41 targets their small/thin, low-fixed-signal phenotype with four-channel raw/CLAHE/local-dark/Sobel input, residual channel attention, dilated context, foreground/spine/boundary supervision, and topology refinement. The frozen fit-only cohort is 16 distinct images / 308 GT (244 small+thin). Local target preflight and 21 V39-V41 tests passed. Private offline kernel `abhimanyu122/solar-filament-v41-context-topology-g2` version 1 remained running through 23:09 IST. This is not a quality result: full training stays forbidden unless fit PQ >=0.85, recall >=0.90, median IoU >=0.80, FP <=40, and all resource/firewall gates pass. No tune/audit/test/export/submission access exists. See [`docs/v41_design_2026-09-14.md`](docs/v41_design_2026-09-14.md), `outputs/v41_g2_kaggle/`, ledger E053. V17 submission `56103401` remains public best at **0.39**.

**Current result (2026-09-14): V40 fixed multiscale dark-ridge signal is closed at its zero-GPU opportunity gate.** Using unchanged V23 radial flattening/CLAHE and native Frangi scales 1/2/4/8, only **63/212 (29.7%)** all-detector misses and **40/113 (35.4%)** zero-fragment misses were ridge-addressable; found-instance control coverage was only **226/922 (24.5%)**. Addressable zero-fragment cases were strong (median AUC 0.811, q90 coverage 0.408), but perfect zero-FP recovery projects V17 PQ only **0.448703 → 0.473003**, below 0.50. V39 graph plus V40 ridge evidence covers 91/212 misses and leaves 121 unsupported. No fit/audit/test pixels, GPU, training, CSV, export, or submission were accessed. Close fixed ridge filtering; future work needs learned contextual foreground features. Evidence: [`docs/v40_design_2026-09-14.md`](docs/v40_design_2026-09-14.md), `outputs/v40_multiscale_ridge_signal/result.json`, ledger E052. Preserve V17 submission `56103401` at public **0.39**.

**Current result (2026-09-14): V39 latent-fragment graph assembly is closed at its zero-GPU opportunity gate.** The canonical tune audit reproduced all six source counts and the 922/1,134 union exactly. Only **40/212 (18.9%)** all-source misses had at least two distinct pure fragments from at least two sources covering >=65% of both GT area and expert spine. Although addressable misses had median area/spine coverage **0.796/0.789**, perfect zero-FP recovery at IoU 0.65 projects V17 PQ only **0.448703 → 0.464294**, below 0.50. **113/212 misses had no admissible fragment at all.** No image pixels, audit, test, GPU, training, CSV, export, or submission were accessed. Close graph assembly over the existing six pools; any new path must generate new faint/small-filament foreground. Evidence: [`docs/v39_design_2026-09-14.md`](docs/v39_design_2026-09-14.md), `outputs/v39_fragment_graph_opportunity/result.json`, ledger E051. Preserve V17 submission `56103401` at public **0.39**.

**Current result (2026-09-14): V38 orientation-to-overpass is closed at its zero-GPU opportunity gate.** The frozen canonical-tune audit reproduced all six proposal-source counts and the 922/1,134 union exactly, then found only **7/212 (3.3%)** all-detector misses within the predeclared close/parallel/crossing/terminus geometry. Although this was enriched 2.77x versus found instances, perfect zero-FP recovery at IoU 0.65 would project V17 tune PQ only **0.448703 → 0.451473**, below the 0.50 authorization bar. No image pixels, audit, test, GPU, training, CSV, export, or submission were accessed. Close this path; preserve V17 submission `56103401` at public **0.39**. Evidence: [`docs/v38_design_2026-09-14.md`](docs/v38_design_2026-09-14.md), `outputs/v38_orientation_opportunity/result.json`, ledger E050.

**Current result (2026-09-14): V37 RF-DETR-Seg-Nano is closed at its bounded memorization gate.** The exact Kaggle-compatible Torch 2.10/CUDA 12.8 runtime passed G0 and the corrected fixed-768 G1 on an Azure Tesla T4: finite losses, successful checkpoint reload, 113.79 s, and 5,785 MiB external peak. The frozen fit-only G2 then completed all 125 epochs / 2,000 steps and evaluated epochs 25/50/75/100/125 over confidence `{0.05,0.10,0.20,0.30,0.50}`. Best was epoch 125 / confidence 0.10 at **PQ 0.523400**, recall 0.894737, median matched IoU 0.746306, and 51/36/6 TP/FP/FN. This missed all three quality gates (0.85 / 0.90 / 0.80), while finite-loss, memory (11,177 MiB external peak), step-budget, and data-firewall gates passed. Therefore `full_design_authorized=false`: do not rescue V37, train the full population, read tune/audit/test, export, or submit. Evidence: `outputs/v37_rfdetr_azure/`, [`docs/v37_design_2026-09-14.md`](docs/v37_design_2026-09-14.md), ledger E049. V17 submission `56103401` remains the validated public best at **0.39**.

**Latest completed result (2026-09-14): V34 expert-spine bilateral width profiles are closed at the CPU representation gate.** A contamination-safe oracle reconstructed all 7,740 fit+tune source instances from ordered expert spines plus separately measured left/right normal widths. It achieved combined PQ **0.945963**, SQ 0.946085, recall 0.999871, 7,739/1/1 TP/FP/FN, and median matched IoU 0.958979; tune PQ was 0.948178. The symmetric-width control reached only 0.764011, confirming a large +0.181952 value from bilateral asymmetry. Nevertheless the frozen combined gate was >=0.95, missed by 0.004037, so no GPU/image model/audit/test/export/submission is authorized. Evidence: [`docs/v34_design_2026-09-14.md`](docs/v34_design_2026-09-14.md), `outputs/v34_width_profile_oracle/`, ledger E046. Preserve V17 submission `56103401` at public **0.39**.

**Latest completed result (2026-09-13): V33 balanced foreground-gated Omnipose FPN is closed at its bounded overfit gate.** Exact V32-cohort analysis measured only **0.1436%** foreground pixels (~695:1 background:foreground), motivating a separate stride-4 FPN with explicit foreground and balanced sparse losses. Two immutable private runs remained far below the PQ 0.85 gate: fix1 reached step-1,000 PQ **0.041703**, recall 0.087719, 5/70/52 TP/FP/FN, with three fail-closed upstream decoder errors and 1,677.74 MiB peak; step 1,500 then hit a second NaN-derived upstream decoder exception. Per the frozen stop rule, no third run, full training, tune/audit/test read, CSV, export, or submission is authorized. Closure: [`docs/v33_design_2026-09-13.md`](docs/v33_design_2026-09-13.md), `outputs/v33_closure_2026-09-13.json`, ledger E045. Preserve V17 submission `56103401` at public **0.39**.

**Latest completed result (2026-09-13): V32 Omnipose passed its exact-field oracle but failed the bounded image-model overfit gate.** G1 reconstructed all 1,087 fit+tune annotator records / 7,740 GT at combined PQ **0.922493**, fit **0.920227**, tune **0.935838**, recall **0.995220**, and 7,703/969/37 TP/FP/FN, passing every gate. The frozen 16-record DeepLabV3-R50 G2 then completed all 2,000 steps on a private Kaggle T4 after a batch-one BatchNorm runtime fix. Memory was safe at **2,820 MiB** and losses were finite, but decoded PQ stayed **0.0** at all six checkpoints with zero predictions and 57 FN. `full_training_authorized=false`; no tune/audit/test, CSV, export, or submission occurred. Close V32 without threshold/loss/architecture rescue. Evidence: `outputs/v32_g2_kaggle_fix1_download/v32_g2_result.json`; design: [`docs/v32_design_2026-09-13.md`](docs/v32_design_2026-09-13.md); ledger E044. Preserve V17 submission `56103401` at public **0.39**.

**Latest completed result (2026-09-13): V31 expert-spine SDT stopped at the CPU target oracle.** The hash-pinned, contamination-safe oracle evaluated all 566 fit + 94 tune files (1,087 annotator records / 7,740 GT). All 7,740 expert spines projected validly. Standard DT failed badly at selected seed 0.85: PQ **0.242650**, recall 0.652584, 5,051/20,329/2,689. Expert-spine SDT at seed 0.65 reached PQ **0.968866**, SQ 0.992263, recall 0.997804, 7,723/356/17; tune alone was 0.978734. G1 required both targets to reach PQ >=0.97 and recall >=0.99. SDT missed combined PQ by -0.001134 and DT failed decisively, so `gpu_authorized=false`. No GPU, audit/test, export, or submission occurred. Do not rescue V31 by changing the frozen gate or decoder. Result: [`docs/v31_result_2026-09-13.md`](docs/v31_result_2026-09-13.md); evidence: `outputs/v31_target_oracle/`; ledger E043. Next permitted representation research is Omnipose. Preserve V17 submission `56103401` at public **0.39**.

**Prior completed result (2026-09-13): V29/V30 native YOLOv8l-2048 reproductions both failed tune gates.** V29 tested all 937 annotator records and reached direct PQ **0.430489** and frozen-V17-refined PQ **0.446350**, both below V17's 0.448703. V30 used one maximum-instance representative target per physical image (566 files / 4,452 masks), completed 40 epochs, and evaluated eight predeclared checkpoints on the exact 94-file / 150-record / 1,134-GT tune population. Its best direct result was epoch 20 at **0.423207** (626/234/508); the frozen V17 refiner improved it to **0.438633** (710/385/424), still -0.010070 versus V17. Both kept audit/test/export sealed; no CSV or submission was created. The attempted V29 continuation did not reach 40 epochs because Ultralytics restored its stored 20-epoch contract; no V29 40-epoch result exists. Evidence: [`docs/v30_result_2026-09-13.md`](docs/v30_result_2026-09-13.md), `outputs/v30_representative_azure/`; ledger E041/E042. Preserve V17 submission `56103401` at public **0.39**.

**Prior completed result (2026-09-13): V28 proposal utility passed OOF feasibility but failed end-to-end tune gates.** The exact V26 serving cohort covered 566 training files / 14,127 proposals. Physical-file-grouped OOF product scoring beat confidence by +0.012028 AUC, +0.017659 utility correlation, and +0.007562 expected-PQ surrogate, authorizing a separately frozen tune run. The final fixed scorer reproduced V17 at **0.448703** and V26 hybrid at **0.432869**, but reached only **0.433172** (635 TP / 245 FP / 499 FN): +0.000303 versus V26 and -0.015531 versus V17. Three gates failed, audit remained sealed, and no test/CSV/export/submission occurred. Artifacts: `outputs/v28_utility_azure/`; design: [`docs/v28_design_2026-09-13.md`](docs/v28_design_2026-09-13.md); ledger E040. V17 submission `56103401` remains best at **0.39**.

**Prior result (2026-09-12): V27 slicing-aided fine-tuning failed replicated tune gates.** Four matched two-epoch/11,244-step runs compared repeated full-image continuation against full+owner-tile fine-tuning at two seeds. Primary sliced PQ was **0.398535** versus matched control **0.434496**; replica sliced PQ was **0.406910** versus control **0.429310**. Sliced fine-tuning reduced hybrid raw recall to ~0.73 and historical rescue to 48-49, losing V26's inference-only recall breakthrough. Every substantive gate failed, audit remained sealed, no test/CSV/export occurred, and nothing was submitted. Artifacts: `outputs/v27_sliced_azure/`; design: [`docs/v27_design_2026-09-12.md`](docs/v27_design_2026-09-12.md); ledger E039. V17 submission `56103401` remains best at **0.39**.

**Prior result (2026-09-12): V26 native sliced inference broke the raw-recall wall but failed final PQ gates.** Exact V17 control reproduced at tune PQ **0.448703** (704/306/430). Full+tile hybrid raw recall rose **0.728395 → 0.805996**, bottom-area recall **0.604106 → 0.821114**, and 90/228 historical misses were rescued; nevertheless selected hybrid PQ fell to **0.432869** (678/339/456). Both PQ gates failed, so audit remained sealed, no test inference/CSV/export occurred, and nothing was submitted. Artifacts: `outputs/v26_sliced_azure/`; design/research: [`docs/v26_design_2026-09-12.md`](docs/v26_design_2026-09-12.md), [`docs/breakthrough_reaudit_2026-09-12.md`](docs/breakthrough_reaudit_2026-09-12.md); ledger E038. V17 submission `56103401` remains best at **0.39**.

**Research consequence:** native slicing is a real proposal-discovery breakthrough, not a submission breakthrough. Any continuation must address tile-aware score calibration/refinement or use separately frozen sliced fine-tuning; do not tune V26 geometry post hoc.

**Prior result (2026-09-12): V25 omitted-rater uncertain-background loss improved raw recall slightly but failed every tune gate.** A custom pinned Ultralytics loss preserved all 6,606 positives and downweighted only negative anchor classification loss inside 2,659 known omitted-rater boxes to 0.1, against an identical weight-1.0 runtime control at two seeds/four T4s. Primary raw recall improved **0.733686 → 0.739859** (+0.006173; seven fewer misses), but selected tune PQ fell **0.448441 → 0.443053**; the replica also fell **0.442704 → 0.438742**. All absolute, matched-control, raw-recall, and replication gates failed. Audit remained sealed, no test inference/CSV/export occurred, and nothing was submitted. Artifacts: `outputs/v25_missing_label_azure/`; design/research: [`docs/v25_design_2026-09-12.md`](docs/v25_design_2026-09-12.md), [`docs/v25_research_2026-09-12.md`](docs/v25_research_2026-09-12.md); ledger E037. V17 submission `56103401` remains best at **0.39**.

**Prior result (2026-09-12): V24 multi-annotator consensus-union training improved raw recall slightly but failed every tune gate.** Four matched 20-epoch T4 runs compared the current incomplete all-annotator targets against one constrained medoid consensus target per physical image, at two seeds. Primary raw recall improved **0.733686 → 0.739859** (+0.006173; seven fewer misses), but tune PQ fell **0.448441 → 0.445858** because FP increased; the replica also moved negatively (**0.442704 → 0.441403**). Required absolute, matched-control, +0.01 raw-recall, and replication gates all failed. Audit remained sealed, no test inference/CSV/export occurred, and nothing was submitted. Artifacts: `outputs/v24_consensus_azure/`; design/research: [`docs/v24_design_2026-09-12.md`](docs/v24_design_2026-09-12.md), [`docs/v24_research_2026-09-12.md`](docs/v24_research_2026-09-12.md); ledger E036. V17 submission `56103401` remains best at **0.39**.

**Prior result (2026-09-12): V23 radial-flattening + CLAHE was closed at its predeclared resource gate.** Competition-only preprocessing and exact matched raw/transformed datasets were prepared successfully, and four independent one-epoch T4 smokes at the final `imgsz=896`, batch-1 fallback all completed with finite results. External peaks were **14,403 / 14,775 / 14,355 / 14,671 MiB**, all above the strict `<14,336 MiB` safety requirement; batch 1 is already the floor. Consequently no 20-epoch continuation, tune/audit evaluation, test inference, CSV export, or submission was authorized. Evidence is in `outputs/v23_resource_gate_azure/`, design/result in [`docs/v23_design_2026-09-12.md`](docs/v23_design_2026-09-12.md), ledger E035. V17 submission `56103401` remains the public best at **0.39**.

**Prior result (2026-09-12): V22 PointRend is complete and gate-failed.** A
four-T4, 20-epoch / 4,700-step official COCO-pretrained PointRend Mask R-CNN
R50-FPN run tested a new RPN/RoIAlign proposal family with 224x224 adaptive masks.
Frozen tune selection chose epoch 15 / confidence 0.50: tune **0.353718**
(SQ 0.648661, RQ 0.545304; 659 TP / 624 FP / 475 FN) and audit **0.403023**
(291 / 198 / 168). Both V17 bars (0.448703 / 0.470763) failed, so no CSV was
generated and nothing was submitted. The result closes this PointRend
configuration: recognition errors, not mask rendering alone, remained limiting.
Artifacts are in `outputs/v22_pointrend_azure/`; frozen design and result are in
[`docs/v22_design_2026-09-12.md`](docs/v22_design_2026-09-12.md), ledger E034.
V17 submission `56103401` remains the public best at **0.39**. Do not extend or
submit V22 without a new mechanism-specific hypothesis.

**Resume here (updated 2026-09-11 23:22 IST):** the public best remains
**0.39**. V21 completed the previously planned final full-data refit of V17's
frozen recipe on all **707 files / 1,154 annotator records / 8,199 annotations**.
The verified final detector/refiner SHA-256 values are
`5cc803e4f3ca334c471fc1012ef1e27a142b59b085831fc09d5e35d621ca5629` and
`26013cec76b7c329f9891fdbf90def19abc60a696d134273837cf5dc3f5f0792`.
Test inference used all four T4s as deterministic 45-image shards and completed
all 180 images in **72–77 seconds**. The merged V17-recipe candidate contains
**1,091 masks across 174 populated images** and passed two independent validators:
all RLEs decode, IDs are known/sequential, areas are >=20, and masks are disjoint.
The submission-ready package is
[`submission_ready/v21_full_refit_2026-09-11/`](submission_ready/v21_full_refit_2026-09-11/);
CSV SHA-256 is
`98f58d2e115ebb6ae9c4f270aa4f03e46d75c279ad6a5c5c09afefb9121e9016`.
V21 was submitted once as Kaggle submission **56170647** and scored **0.38**, below
V17's 0.39. V17 submission `56103401` remains the public best. Because all labels
were used for fitting, V21 has no held-out local PQ claim; the public result is the
only final-fit measurement. Do not re-upload V21. Deadline 2026-11-15. Full state:
[`../RESUME_TOMORROW.md`](../RESUME_TOMORROW.md).

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
