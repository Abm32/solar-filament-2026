# Optimization plan toward public PQ 0.55 — 6 September 2026

Synthesis of three research inputs (local public-notebook analysis, external
web/literature research, our own quantitative bottleneck analysis) into one
prioritised plan. Current public best **0.39** (V11 candidate `56049359`,
unpromoted; gate FAILED at +0.005177 tune vs required +0.01). Accepted frozen
baseline is the 0.38 pipeline. This document proposes work; it launches nothing.

Constraints assumed throughout: Kaggle T4 x2, ~84,000 GPU-s remaining until the
2026-09-12 refresh; a full V6-style 80-epoch 1024 train cost ~5,900 s; local env
has no torch; 3 Kaggle submissions/day resetting 00:00 UTC. Evaluation stays
canonical: global PQ, all 217 validation annotator entries scored separately,
strict IoU > 0.5, 94 tune / 47 audit filenames (audit is reused screening
evidence, NOT an untouched holdout).

---

## 1. Honest gap analysis: what 0.39 → 0.55 actually requires

**The public target is far outside anything the three reports can numerically
justify as reproducible.** Be blunt about this before ranking any experiment.

### 1.1 The local→public mapping is unknown and the only calibration points are tiny

We have exactly two dated (local tune-PQ gain → public score) observations:

| Version | Local tune-PQ gain vs V6 | Public score |
| --- | ---: | ---: |
| V10 | +0.0012 | 0.38 |
| V11 | +0.0052 | 0.39 |

Our current local canonical PQ is ~0.439 while public is 0.39 — a **local-minus-public
gap of ~0.05**. Two points cannot establish a transfer function. What they weakly
suggest is that small local gains (well under the old +0.01 gate) do move public
score, and that public sits ~0.05 below local. **If** that gap held constant
(it may not), public 0.55 would need local canonical PQ ≈ **0.60**. We are at
0.439. That is a **+0.16 local PQ** requirement.

### 1.2 What +0.16 local PQ decomposes into (our own algebra, strongest evidence)

From our bottleneck analysis (Q6 algebraic ceilings, actual V11 counts fixed):

| Counterfactual | Local PQ |
| --- | ---: |
| Baseline (V11 counts) | 0.438 |
| Perfect FP filter (delete all FP) | 0.489 |
| Perfect masks, counts fixed (mean IoU → 1.0) | 0.639 |
| Full GT recovery at current mean IoU | 0.628 |
| Perfect FP filter **and** full recovery | **0.686** |

The **absolute oracle ceiling is ~0.686 local** — combining a perfect FP filter
with recovering every recoverable GT instance at today's mask quality. Public 0.55
(≈ local 0.60 under the assumed constant gap) is therefore **not impossible in
principle, but it sits at ~87% of the way to the joint oracle.** No single lever
gets there:

- FP filtering alone caps at **0.489** (+0.05). Insufficient, and our own analysis
  says we are **not** FP-limited.
- Mask quality alone (SQ 0.684 → required 0.778 for local 0.50) caps below what we
  need even at perfect masks unless recall also moves.
- Recall recovery is the largest single lever: **+0.19 local PQ of headroom** vs
  +0.05 for FP. Our decomposition (Q3) says 278 instances are lost at the 0.40
  cutoff and 228 have no raw proposal at all.

**Conclusion: 0.39 → 0.55 is implausible from any one change and implausible from
frozen-weight postprocessing.** It requires (a) materially higher recall AND (b)
higher mask quality, delivered together, and almost certainly requires **retraining
the detector** (the 228 "no raw proposal" instances cannot be recovered by any
refiner or postprocess — the detector never proposed them). A realistic honest
framing: the reproducible, evidence-backed levers below can plausibly stack to the
**0.42–0.47 local / ~0.41–0.44 public** range. Reaching 0.55 public on top of that
depends on the resolution lever (V12) delivering a large, currently-unmeasured
recall+SQ gain that only HDJoJo's private 2048 checkpoint circumstantially supports.
Treat 0.55 as a stretch research objective, not a forecast.

### 1.3 Where the three reports disagree, and which is stronger

| Question | local_notebook | web_research | our_bottleneck | Verdict |
| --- | --- | --- | --- | --- |
| Primary bottleneck | Resolution (native-2048) | SQ (mask heads); SQ 0.684→0.778 | **RECALL** (278 cutoff + 228 missing-raw); we are NOT SQ/FP-limited | **our_bottleneck** — it is our own canonical measurement on our own counts; the other two are priors/analogy |
| Single highest-impact change | Native-2048 detector (V12) | Point-based mask head + IoU head | **Learned proposal-quality/IoU head** to recover the 278 | Convergent: web #3 == our_bottleneck. Strongest combined support |
| clDice/topology | medium (ruhul used it) | WEAK/contested; EdgeAttNet used only BCE+Dice | not raised | **web** — topology optimises connectivity, not IoU; PQ rewards IoU |
| Instance conditioning (seed channel) | (implied via #1 refiner) | strong mechanism prior | — | **Overridden by local evidence**: E019 already tested this and it was the weakest arm every epoch. REJECTED. |

The reports agree on one thing worth acting on immediately: a **learned mask/proposal
quality head** (web #3 Mask Scoring R-CNN; our_bottleneck's single highest-impact
change). web_research over-weights SQ because it reasons from generic segmentation
priors; our own counts show SQ is real but secondary to recall. Where they conflict,
our canonical measurement wins.

---

## 2. Ranked experiments (expected value per GPU-hour)

Ordering is by expected local-PQ gain per GPU-hour, then by informativeness.
GPU budget reference: 84,000 s ≈ 23.3 GPU-h remaining before refresh; V6-style
1024 train ≈ 5,900 s ≈ 1.6 GPU-h; V11 refiner continuation ≈ 5,080 s.

### E-A. Cheap inference flags on the frozen detector (retina_masks, agnostic_nms, iou)
- **Hypothesis:** full-resolution mask upsampling (`retina_masks=True`) and
  `agnostic_nms=True` raise matched IoU on thin bodies, converting some of the
  199 fragile TPs (IoU 0.50–0.60) and 113 near-misses (0.40–0.50) over 0.5.
- **Exact change:** in the frozen inference path only, sweep `retina_masks`
  {False,True} × `agnostic_nms` {False,True} × box `iou` {default, 0.0-aggressive},
  no retrain, no threshold-grid expansion beyond these predeclared flags.
- **Mechanism:** SQ primarily (finer mask raster), minor RQ (near-misses crossing 0.5).
- **Success gate:** ≥ +0.003 tune PQ with nonnegative audit delta; else drop.
- **GPU cost:** ~0.2–0.4 GPU-h (inference-only, cached raw where possible).
- **Falsify:** if tune PQ moves < +0.001 or audit regresses, the flags do not help
  our raster path — abandon.
- **Evidence:** MEDIUM (hdjojo, ektarr use them). **Worth 0.01–0.02 at most.**

### E-B. Off-limb / solar-disk false-positive suppression — **REJECTED, MEASURED 2026-09-06**

**Do not implement. The diagnostic was run and returned zero.** Reproduce with
`.venv/bin/python scripts/analyze_offdisk_predictions.py`; results in
`outputs/offdisk_2026-09-06/offdisk_summary.json`.

Measured on the submitted 980-mask V11 CSV, with the solar disk fitted per test
image (Otsu, largest component, least-squares enclosing circle; median disk
fraction 0.618 of the frame):

| Quantity | Result |
| --- | ---: |
| Predictions with **any** pixel beyond the fitted limb | **0 / 980** |
| Predictions with centroid radius > 0.95 | **0** |
| GT instances (1,490 over 120 files) with any pixel beyond the limb | **0** |
| GT max centroid radius / max any-pixel radius | 0.912 / 0.962 |
| GT centroid radius p95 / p99 | 0.820 / 0.860 |

The detector never fires off-disk, so there are no off-disk FP to remove and the
gate (">= ~30 of the FP off-disk") fails outright. The original hypothesis was
wrong, consistent with our bottleneck analysis showing we are not FP-limited.

A secondary read is also informative: predictions stop short of the limb just as
GT does, so there is **no near-limb recall deficit** either. Limb geometry is not
a lever in this pipeline in any direction. Cost of learning this: minutes of CPU,
zero GPU.

### E-C. Learned proposal-quality / IoU head to replace the 0.40 consensus cutoff  ⭐ — **CHEAP VARIANT MEASURED AND REJECTED 2026-09-06 (E021)**

**Status:** the cheap fixed-feature variant is dead. Reproduce with
`.venv/bin/python scripts/analyze_ec_feasibility.py` then
`scripts/analyze_ec_oof.py`; results in
`outputs/ec_feasibility_2026-09-06/`. At a matched accepted-proposal budget of
544, the best out-of-fold linear probe (`consensus_only`, AUC 0.7252) recovers
**405 positives vs the gate's 406** — one *fewer*. `consensus_plus_geometry`
collapses to AUC 0.4535 / 171 positives; `geometry_only` to 0.4244 / 157. No cheap
geometric feature carries incremental signal over the consensus score itself.
**Only the CNN mask-scoring variant — a head reading actual mask pixels rather
than summary statistics — remains untested**, and the negative cheap result
weakens its prior. The original specification follows unchanged.
- **Hypothesis:** the fixed 0.40 consensus gate discards 278 grouped proposals that
  already have a mask with IoU > 0.5 (directly recoverable TP). A learned scorer that
  predicts per-proposal mask-IoU recovers a size-aware subset without the FP cost that
  killed the naive threshold drop.
- **Exact change:** train a small mask-scoring/IoU-regression head (Mask Scoring
  R-CNN-style) on **training-only, out-of-fold** detections; inputs = mask/image
  features + four-view agreement + refiner support; target = actual mask IoU vs the
  matched canonical annotation. Replace the raw consensus gate with a
  predicted-IoU rank at a **matched accepted-proposal budget** (same count as the
  0.40 gate), then re-run ownership. Make it size-aware (losses concentrate in the
  smallest area decile — 78% FN-proxy — and category 3).
- **Mechanism:** **RQ primarily.** This is the single largest recall lever short of
  detector retraining. Also protects the 199 fragile TPs by better ranking.
- **Success gate:** ≥ +0.01 tune PQ over frozen V6 with **nonnegative** audit delta
  AND FP count not exceeding the 0.40-gate FP count. (Recalibrated intent: even
  +0.005 here would, on the V10/V11 evidence, likely move public score — but hold
  the screening bar at +0.01 for promotion; treat +0.005–0.01 as "measure publicly.")
- **GPU cost:** MEDIUM. Head training on cached features ≈ 1–2 GPU-h; needs an
  out-of-fold detector-proposal generation pass. Budget ~3–4 GPU-h total including
  fold proposals.
- **Falsify:** if the learned scorer at the matched budget cannot beat the 0.40 gate
  on tune PQ (i.e. the 278 pool is not separable from FP by learnable features),
  then recall recovery is genuinely detector-limited and effort must shift to V12.
- **Evidence:** **STRONGEST convergent signal** — web_research #3 (standard technique)
  AND our_bottleneck's single highest-impact change AND already scoped as V11-follow-up
  in `score_050_research`. **Could plausibly close a meaningful part of the gap**
  (recall headroom is +0.19 local); realistically worth 0.02–0.05 local if the pool
  is separable. NOT yet run.

### E-D. V12 detector training-resolution comparison (1024 vs 1536, controlled)  ⭐ — **MEASURED AND REJECTED 2026-09-06 (E022)**

**Status:** run to completion, gate FAILED, nothing promoted, nothing submitted.
Artifacts in `outputs/kaggle_v12_full/` (`v12_selection.json`, `v12_candidates.csv`,
`v12_manifest.json`). Control tune PQ 0.432823 / audit 0.450130. Best arm
`v12_res1024_infer1536`: tune **0.432243** (delta **-0.000580**), audit **0.430845**
(delta **-0.019284**). Full matrix: `res1536_infer1536` 0.428023, `res1024_infer1024`
0.425146, `res1536_infer1024` 0.415018. Training cost 1,388.81 s (1024) + 2,759.04 s
(1536).

**E-D's own falsification condition was met:** the 1536-*trained* arm regressed
against the 1024 control at **every** inference resolution, so "the resolution lever
is dead and HDJoJo's advantage is not resolution." Higher *inference* resolution
does move recall (TP 626→637, FN 508→497) but adds more FP than it recovers TP
(211→249), netting negative PQ. Do not progress to 2048. The original specification
follows unchanged.
- **Hypothesis:** training the detector at higher spatial resolution recovers part of
  the 228 "no-raw-proposal" small instances (78% FN in the smallest area decile) that
  no refiner or scorer can reach. This is the only lever that attacks the missing-raw pool.
- **Exact change:** matched-budget continuation of the pinned V6 checkpoint at 1024 vs
  1536, per the V12 audit config (20 epochs, AdamW 1e-4, identical augmentation, equal
  optimizer-step schedule, `resume=False`, isolated run dirs). Evaluate each checkpoint
  at both 1024 and 1536 inference (2×2) to separate train-res from infer-res. Progress
  to 2048 **only** if 1536 wins and a memory smoke passes.
- **Mechanism:** **RQ (new small-instance detections) + SQ (finer masks).** This is the
  mechanism behind HDJoJo's private 2048 checkpoint.
- **Success gate:** the higher-res arm delivers ≥ +0.01 tune PQ over the 1024 control
  at its best inference resolution, nonnegative audit delta. Predeclare the comparison
  checkpoint policy before looking at scores.
- **GPU cost:** HIGH. Two arms; 1536 at batch-2 grad-accum will exceed the ~5,900 s
  1024 baseline. Budget ~4–6 GPU-h for the 1024/1536 pair; 2048 adds substantially more.
- **Falsify:** V9 changed **inference** resolution only and regressed — this does NOT
  refute training at higher res, but if the 1536 **trained** arm also regresses or ties
  the 1024 control at every inference resolution, the resolution lever is dead and
  HDJoJo's advantage is not resolution.
- **Evidence:** MEDIUM-STRONG mechanism (only credible real-model 0.55 lead, hdjojo),
  but **circumstantial** — the checkpoint name implies 2048 training, it is not proven.
  **This is the only lever with a plausible path to a LARGE gap-closing gain**, and
  also the most expensive and least certain. High variance bet.

### E-E. CLAHE / multi-channel detector input (separate ablation)
- **Hypothesis:** a CLAHE (and/or local-contrast) channel improves faint/small and
  category-3 ("Unidentifiable", 38.4% accepted-anchor opportunity) contrast, raising
  raw recall.
- **Exact change:** retrain the detector arm with an added CLAHE channel, zero-init the
  new channel weights, preserve RGB path. Keep it a **standalone** change (not combined
  with V12 res in the same run).
- **Mechanism:** RQ via contrast on faint filaments; minor SQ.
- **Success gate:** ≥ +0.005 tune PQ vs same-recipe RGB control, nonnegative audit.
- **GPU cost:** LOW-MED, ~1.6 GPU-h (one detector train).
- **Falsify:** if it ties the RGB control, the near-universal notebook use is habit, not
  signal.
- **Evidence:** MEDIUM (near-universal: timgluhkikh, ektarr, shoaib, ruhul). Note
  Ultralytics may already inject Albumentations CLAHE at p=0.01 — confirm before
  claiming novelty. **Worth 0.005–0.02.** Best done as a channel inside the V12 winner.

### E-F. Focal-Tversky (β>α) loss in the refiner/mask head
- **Hypothesis:** FN-weighted Tversky improves recall of thin bodies/barbs, pushing
  near-miss masks (IoU 0.40–0.50) over 0.5.
- **Exact change:** swap the refiner's 0.5 BCE + 0.5 Dice for Focal-Tversky with β>α,
  matched budget vs the frozen-recipe control.
- **Mechanism:** RQ (recovers near-misses) + minor SQ.
- **Success gate:** ≥ +0.005 tune PQ; converts a measurable share of the 113 near-misses.
- **GPU cost:** LOW (loss swap), ~1–1.6 GPU-h.
- **Falsify:** if matched IoU distribution below 0.5 is unchanged, the loss is not the lever.
- **Evidence:** MEDIUM (strong in medical/thin-structure, not solar). **Worth 0.005–0.015.**
  Best combined with V12, not the saturated 256 refiner.

### E-G. Point-based / high-resolution mask head (PointRend-style)
- **Hypothesis:** coarse mask rasters over-smooth thin/bent parts; a point-based head
  lifts SQ toward the 0.778 target.
- **Exact change:** add a PointRend-style head inside the refiner path.
- **Mechanism:** SQ primarily, some RQ (near-misses over 0.5).
- **Success gate:** ≥ +0.005 tune PQ AND measurable Boundary-IoU improvement (internal
  diagnostic).
- **GPU cost:** MEDIUM-HIGH (architecture change + retrain).
- **Falsify:** if SQ (matched mean IoU) does not rise, the head is not helping.
- **Evidence:** STRONG generic, on-domain (thin structures), but web_research over-weights
  SQ relative to our measured recall bottleneck. **Worth 0.01–0.03 SQ-side**, but our
  analysis says recall matters more — do this AFTER E-C/E-D.

### E-H. Multi-fold instance ensemble (cross-fold IoU clustering + majority vote)
- **Hypothesis:** K-fold agreement filters FP in a PQ-aware way and firms up borderline TP.
- **Mechanism:** RQ via FP control + stability.
- **Success gate:** ≥ +0.005 tune PQ over the single best fold.
- **GPU cost:** HIGH (K× training). **Defer** — expensive, and both public impls that use
  it (ektarr, sivapnlr80) score ≤ us and siva's validation is leaky.
- **Evidence:** MEDIUM but weak score proof. **Worth 0.005–0.015. Deprioritise until after
  the resolution lever is settled.**

### Explicitly deprioritised / rejected (do not re-propose without new justification)
- Instance conditioning via proposal-seed channel — **E019 rejected**, weakest arm every epoch.
- Broad threshold grids; fixed-weight upscaled full-frame inference (V9 regressed);
  automatic disconnected-component splitting (E016 regressed); dropping category 3
  (its raw match rate is fine — loss is downstream); concatenating CSVs; equally
  weighting V6/V8; tuning on the public LB.
- Spine-bridging morphology + sliding-window semantic inference (contrary local evidence).
- clDice / Skeleton-Recall topology loss — contested externally, optimises connectivity
  not IoU; **at most a small ablation, never a headline bet**.
- Chasing lamhuy's 1,342-row count or any embedded CHAMPION_PAYLOAD — unverified, not a model.

---

## 3. Speculative vs evidence-backed; small vs large

**Evidence-backed (our own canonical measurement or a standard technique + convergent reports):**
- E-C (learned IoU head) — strongest; both web and our-own converge; addresses the
  measured 278-cutoff pool directly.
- E-A / E-B — cheap, mechanistically grounded, near-zero risk.

**Evidence-backed mechanism but no solar PQ guarantee (strong prior, unmeasured here):**
- E-D (V12 resolution) — the only lever with a plausible LARGE gain, but circumstantial
  (HDJoJo's checkpoint name); highest variance and cost.
- E-G (point-based head), E-F (Focal-Tversky), E-E (CLAHE) — solid generic evidence,
  domain-transfer uncertain.

**Speculative / contested:**
- clDice/topology, E-H fold ensemble (weak score proof), 2048 training (only after 1536 wins).

**Likely worth only 0.01–0.02 each (do them cheaply, do not bet the plan on them):**
E-A, E-B, E-E, E-F, E-G, E-H.

**Could plausibly close a large part of the 0.39→0.55 gap (but uncertain):**
- E-C, IF the 278-cutoff pool is learnably separable from FP (recall headroom +0.19 local).
- E-D, IF higher-resolution training recovers a large share of the 228 missing-raw
  instances (this is the only lever that can touch that pool at all).

**Honest bottom line:** no evidence-backed lever guarantees 0.55. The credible path is
E-C + E-D stacked, with E-A/E-B/E-E/E-F as cheap additive gains. That realistically
targets ~0.42–0.47 local; 0.55 public depends on E-D delivering an outsized,
currently-unmeasured recall+SQ gain.

---

## 4. Recommended immediate next action (cheap and informative)

**Both cheap diagnostics proposed here have now been executed. Section retained for
the record; the live recommendation is at the end.**

1. **~~CPU off-disk FP diagnostic~~ — DONE, returned zero.** 0 of 980 predictions
   have any pixel beyond the fitted limb, and neither does GT. E-B is dead in both
   directions (no off-disk FP, no near-limb recall deficit). See E-B above.

2. **~~`gt_centered_e4` public measurement~~ — DONE, scored 0.39.** Submitted as
   `56050554`. It **ties** `proposal_centered_e4` despite 0.000688 lower local
   tuning PQ, in a strict single-variable comparison. **Verdict: the V11 public gain
   was continuation training, not proposal centering.** Both of V11's substantive
   hypotheses are now unsupported. Two further conclusions: two-decimal public
   scoring cannot resolve ~0.0007 local deltas, so stop spending submissions on
   near-identical arms; and the calibration set is now +0.0012 → 0.38,
   +0.0045 → 0.39, +0.0052 → 0.39.

### Live recommendation — SUPERSEDED 2026-09-06 (kept for the record)

The recommendation below was to commit the next GPU block to E-C. **Both E-C
(cheap-feature variant) and E-D were subsequently run on 6 September and both
failed.** See the [experiment ledger](experiments.md) E021 and E022:

- **E021 / E-C cheap-feature — NOT SUPPORTED.** At a matched accepted-proposal
  budget of 544, the best out-of-fold linear probe over summary geometry features
  recovers **one fewer** positive than the fixed 0.40 gate (`consensus_only`,
  OOF AUC 0.7252, delta -1). Adding geometry to the consensus score collapses AUC
  to 0.4535 and costs 235 positives. The 278-cutoff pool is **not separable from
  FP by cheap fixed features**. Per E-C's own falsification clause this shifts
  effort to E-D — which was then also run and also failed. Only the CNN
  mask-scoring variant (reading real mask pixels, not summary features) is still
  untested.
- **E022 / E-D V12 resolution — gate FAILED.** Best arm `res1024_infer1536`: tune
  delta **-0.000580**, audit delta **-0.019284**. The 1536-*trained* arm lost to
  the 1024-trained arm at both inference resolutions, satisfying E-D's own
  falsification condition: **the resolution lever is dead and HDJoJo's advantage
  is not resolution.** Higher inference resolution trades recall for precision at
  a net PQ loss (TP 626→637, FN 508→497, FP 211→249).

**Current live recommendation:** run **E-A** (inference flags on the frozen
detector, ~0.2–0.4 GPU-h) — the only cheap unrun lever. After that the ranked
remainder is E-F (Focal-Tversky, cheap loss swap aimed at the 113 near-misses),
then E-C-as-CNN-head, then E-E/E-G/E-H. Section 1's honest bottom line needs
revising downward: it said the credible path was "E-C + E-D stacked, targeting
~0.42–0.47 local". With both eliminated, the reachable band on remaining
evidence-backed levers is narrower, and **0.55 public should now be treated as
out of reach absent a genuinely new mechanism.**

---

### Original recommendation (pre-E021/E022, do not act on)

Commit the next GPU block to **E-C (learned proposal-quality/IoU head)**. It is the
highest expected-value-per-GPU-hour experiment, attacks the measured primary
bottleneck, and is the convergent recommendation of the two strongest reports.
Nothing measured today touched the 278 cutoff-lost or 228 missing-raw instances, so
the bottleneck analysis stands unchanged and E-C's rationale is if anything
strengthened: the frozen-detector refiner track has now been shown to yield only
rounding-level public movement from continuation alone.

If the GPU budget is tight, run **E-A** (inference flags on the frozen detector,
~0.2–0.4 GPU-h) first — it is the only remaining cheap unrun lever.

Reserve the larger, higher-variance **E-D (V12 resolution)** spend until E-C's
fold-proposal infrastructure is in place, since E-D is the only path to a large
gap-closing gain and should not be rushed against the ~84,000 s quota.
