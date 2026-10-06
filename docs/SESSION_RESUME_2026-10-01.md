# Session Resume — Solar Filament Segmentation (2026-09-30 → 2026-10-01)

**Status: public best still 0.39 (V17 submission `56103401`). Nothing beat it. No
new submission was made.** This document captures the full session so work can
resume cleanly.

---

## 0. TL;DR

- Ran experiments **V42–V49** on the shared Azure VM `rogii-gpu` (4× Tesla T4).
- **V42** (pretrained YOLO-seg) broke the recall wall on the *fit* cohort
  (recall 0.997, PQ 0.768 on 16 memorized images) — but it was **memorization**.
- **V44** (full-population train) proved it does **not generalize**: held-out tune
  PQ ~0.35, vs V17's **0.4487**.
- **V43, V45, V47, V48, V49** all tried to close the gap (TTA, ensembles,
  inference gating, EdgeAttNet preprocessing, cross-architecture ensemble). All
  capped at **~0.35–0.37 tune PQ**. None beat V17.
- **V17's exact 0.4487 pipeline is NOT reconstructable on the VM** (its SHA-pinned
  6-source caches/checkpoints were on Kaggle `/kaggle/input`, now gone).
- **Decision: hold at public 0.39.** The only remaining path with measured headroom
  needs V17's original proposal caches retrieved from Kaggle.

---

## 1. Environment / access (verify first on resume)

- SSH host: `rogii` (`~/.ssh/config` → 20.51.160.183, user `azureuser`,
  key `~/.ssh/rogii_vm_key`). `az` CLI logged in (tenant tech@anukritiai.com).
- VM is **shared with Rogii** — do NOT deallocate without confirming Rogii idle.
- Repo on VM: `~/solar-filament-2026` (durable root disk, 95–96% full, ~12G free).
  venv: `.venv` → torch 2.4.1+cu121, 4 T4 visible.
  Installed this session: `scikit-image==0.24.0`, `ultralytics==8.4.166`
  (torch stayed pinned).
- `/mnt` = 2.6T **ephemeral** scratch (wiped on VM reboot!). All `/mnt/v4*`
  work dirs + checkpoints are at risk — durable copies are under
  `~/solar-filament-2026/outputs/`.
- `/mnt` dirs are root-owned by default: create work dirs with
  `sudo mkdir -p /mnt/X && sudo chown azureuser:azureuser /mnt/X`.
- Data (pinned): `~/solar-filament-2026/data/raw/filament-segmentation-2026/`
  `MAGFiLO_1.0_Kaggle_2026/` → `train/MAGFiLO_1.0_Annotations_kaggle2026_train.json`
  SHA256 `5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1`;
  `train/train_images` (707), `test/test_images` (180).
- Kaggle CLI present on VM and locally (`~/miniconda3/bin/kaggle`), account
  `abhimanyu122`. Competition slug: `filament-segmentation-2026`.
- Local repo: `/home/abhimanyu/Desktop/digital_agency/kaggle/solar/solar-filament-2026`
  (all docs + result JSONs mirrored here).

### Canonical split (frozen, used everywhere)
```python
files = sorted(by_file); random.Random(42).shuffle(files)
tune = sorted(files[:141])   # 141 held-out files
fit  = sorted(files[141:])   # 566 fit files, 4452 instances (one max-instance record/file)
```
V17 tune PQ bar = **0.448703**. Project PQ = instance IoU>0.5 greedy one-to-one
matching (`src/pq_eval.py`). Target disjoint, min area 20, cols
`[filament_id, segmentation_rle]`, ids `{stem}_{n}` sequential from 1.

---

## 2. Experiments this session

### V42 — pretrained YOLO-seg bounded overfit gate (fit cohort, 16 imgs/308 GT)
- Hypothesis: pretrained objectness breaks the recall wall V41 (from-scratch)
  failed. Controlled A/B — same frozen 16-file cohort / gate / firewall as V41.
- **FIRST ATTEMPT BUG:** `~/weights/yolo26n.pt` is a **detection** model (no mask
  head) → PQ 0 despite conf 0.988 boxes. Fixed by downloading proper
  `yolo11n/s/m/l-seg.pt` (now in `~/weights/`).
- **Eval OOM bug:** `retina_masks=True` blows T4 memory → fixed with
  `retina_masks=False` + per-mask CPU upsample.
- Resolution ladder (yolo11s-seg, fit cohort), the key finding:
  | imgsz | PQ | recall | medIoU | FP |
  |---|---|---|---|---|
  | 1024 | 0.601 | 0.899 | 0.662 | 26 |
  | 1536 | 0.725 | 0.990 | 0.734 | 2 |
  | 2048 | 0.752 | 0.990 | 0.770 | 3 |
  | 2048/300ep | **0.768** | **0.997** | **0.781** | **1** |
- **Resolution is the lever; capacity is NOT** (n/s/m at 1024 = 0.59/0.60/0.57;
  m/l at 2048 worse AND exceed 14.3G mem gate).
- Fit gate (PQ≥0.85) NOT passed, but recall/FP gates passed hugely. Breakthrough
  was real on fit. Docs: `docs/v42_result_2026-09-30.md`,
  `outputs/v42_resolution_ladder/`, `outputs/v42_native2048_refine/`,
  best ckpt `outputs/v42_best_checkpoint/v42_s_seg_2048_300ep_best.pt`
  (SHA `b28afa5e...`).

### V43 — TTA / multiscale mask refinement (fit cohort)
- TTA (flips) + soft-mask merge. **Clean NEGATIVE:** monotonically worse
  (no-TTA 0.768 → hv 0.682/FP68 → 4-view 0.619/FP124). Sub-pixel-misaligned
  flipped masks inject duplicate FPs. Docs: `docs/v43_result_2026-09-30.md`,
  `outputs/v43_mask_refine/`.

### V44 — full-population train (566 fit files, 200 epochs) + held-out tune eval
- yolo11s-seg @ 2048. **The decisive generalization test.**
- Held-out tune PQ: e31=0.351, e65=0.357, e102=0.348, **e200=0.351**, recall ~0.48,
  medIoU 0.654. Plateau; `beats_v17=FALSE`.
- V42's fit numbers were **memorization**; did not generalize. Submitting would
  regress <0.39. Docs: `docs/v44_result_2026-09-30.md`,
  `outputs/v44_full_population/` (incl. `v44_full_best.pt`).

### V45 — same-family union ensemble (s+m+l full-pop models)
- Complementarity REAL (union recall 0.51–0.55 > single 0.48) but PQ caps 0.370
  (agreement filter conf0.05/min_agree2/high_conf0.9). Correlated FPs.
  Docs: `docs/v45_result_2026-09-30.md`, `outputs/v45_ensemble/`.

### V46 — tune-error decomposition (the key analysis)
- V44 model, 141 tune files, conf 0.20: TP=550, **near_miss(0.3–0.5 IoU)=120**
  (mean 0.41), **hard_fn(≤0.3)=475**, **FP=359**, mean matched IoU 0.655.
- Counterfactual PQ ceilings: baseline 0.351; promote near-miss 0.392;
  remove FP 0.425; matched IoU→0.80 0.428; **promote+removeFP (oracle) 0.470**.
- Oracle beats V17, but it's not realizable (below).
  Evidence: `outputs/v46_research/v46_errors.json`.

### V47 — realizable inference-side FP gating + mask cleanup
- Sweep conf × min-area × morphology on V44. Best **0.360** (far below oracle
  0.425). FPs not threshold-separable. Negative.
  `outputs/v46_research/v47_sweep.json`.

### V48 — EdgeAttNet/V23 preprocessing retrain (radial-flatten + CLAHE)
- Retrained full-pop on preprocessed inputs. Tune PQ **0.357** @ e34, recall 0.494
  — **zero recall lift** vs raw. Instance-detector recall is NOT contrast-limited.
  Negative. `outputs/v46_research/v48_preprocessed_tune_e34.json`.

### V49 — V17 reconstruction + cross-architecture ensemble
- `v17_compose.py` present but needs SHA-pinned V6/V16/V11 checkpoints + **321
  raw proposal caches** — NONE on VM (wiped). V17 UNRECONSTRUCTABLE here.
- V21 full-refit ckpts present (`outputs/v21_full_refit_persistent/checkpoints/`,
  scored public 0.38). V21 detector = loadable seg model, tune PQ 0.351,
  **recall 0.559** (higher than V44).
- Cross-arch ensemble V44 + V21 = **0.370**, still ~0.08 below V17.
  Docs: `docs/v49_result_2026-10-01.md`, `outputs/v49_crossarch/`.

---

## 3. Research findings (external)

- **EdgeAttNet** (arXiv 2509.02964, SOTA on MAGFiLO): *semantic* U-Net + edge-guided
  attention, from scratch; best mIoU_pairwise 0.6451 (PIXEL IoU, not instance PQ).
  Transferable idea = radial-flatten+CLAHE preprocessing (= project V23); tested in
  V48, gave no lift. github: dasjar/EdgeAttNet.
- **clDice** (arXiv 2003.07311): topology-preserving soft-skeleton loss for tubular
  structures — untested; relevant only if training a semantic model.
- **SAM/SAM2 limitation** (arXiv 2412.04243): foundation models fail on "tree-like,
  low-contrast" objects = solar filaments → unlikely to recover the hard-FN bucket.
- MAGFiLO dataset origin: Nature Sci Data s41597-024-03876-y (GONG Hα, >10k filaments).

---

## 4. Why nothing beat 0.39 (the firm conclusion)

Six independent methods cap at **~0.35–0.37 tune PQ**. Root causes (measured, not
guessed):
1. **475 hard-FN** = genuinely-ambiguous faint filaments (the project's known
   ~18.7% "invisible to every source"). Not contrast-limited (V48). SAM won't help.
2. **Correlated false positives** across models (same- AND cross-architecture) that
   agreement/gating cannot remove → ensembles add recall but also FPs, net ~+0.01.
3. **V17's 0.4487 is irreducibly a property of its specific 6-source cache
   ensemble**, which cannot be rebuilt on the VM.

This independently reconfirms project ledger **E027** (local PQ decoupled from
public) and **E028** (realistic NMS ensembling barely beats best single source).

---

## 5. New code added this session (local + VM `scripts/`, `tests/`)

- `scripts/train_v42_pretrained_detector_overfit.py` (+ `tests/test_v42_...py`)
- `scripts/train_v43_mask_refine_overfit.py` (+ `tests/test_v43_...py`)
- `scripts/train_v44_full_population.py` (+ `tests/test_v44_...py`)
  — modes: train / tune / test; disjoint-mask CSV export matching
  `validate_submission.py`. **This is the submission-ready runner** if ever needed.
- `scripts/eval_v45_union_ensemble.py` — multi-model union w/ agreement voting.
- `scripts/analyze_v46_tune_errors.py` — PQ error decomposition + counterfactuals.
- `scripts/analyze_v47_precision_cleanup.py` — FP-gating + morphology sweep.
- `scripts/train_v48_preprocessed.py` — radial-flatten+CLAHE train/tune.
All ruff-clean; focused tests pass locally. (Local venv lacks torch/ultralytics,
so GPU runs happen on the VM `.venv`.)

Trained weights (durable, on VM `~/solar-filament-2026/outputs/`):
- `v42_best_checkpoint/v42_s_seg_2048_300ep_best.pt` (fit-overfit, SHA b28afa5e)
- `v44_full_population/v44_full_best.pt` (full-pop, tune PQ ~0.351)
`/mnt/v44_m`, `/mnt/v44_l` (full-pop m/l) are EPHEMERAL — may be gone on reboot.

---

## 6. NEXT STEPS (resume here)

**The only path with measured headroom: retrieve V17's original artifacts, then add
V44/V21 as additive orthogonal sources.**

1. Check if Kaggle still has the pinned V17 inputs:
   ```
   kaggle kernels list -m --user abhimanyu122
   kaggle kernels output abhimanyu122/<v17-kernel> -p /mnt/v17_recover
   # look for: yolo_best.pt (V6), arm_r1_best.pt (V16 x2), *_e4.pt (V11 refiner),
   #           proposal_cache/ (321 raw caches), v17_results.json
   ```
   Also check competition datasets / the user's Kaggle datasets for cached proposals.
2. If recovered → run `kaggle_inference/v17_compose.py` to reproduce the 6 sources,
   then fold in V44 (and/or V21) as a 7th source via `eval_v45_union_ensemble.py`
   logic (agreement filter). Evaluate tune PQ; **submit only if > 0.4487**.
3. If NOT recoverable → the realistic ceiling on this VM is ~0.37; **keep 0.39**.
   Alternative long shots (lower EV, documented as untried):
   - clDice-trained semantic model as a NEW orthogonal source (different failure
     modes than YOLO) — could decorrelate FPs.
   - Retrain V44 with a precision-focused config (higher conf, FN-weighted loss) to
     cut the 359 FP directly.

**Firewall discipline maintained all session:** no tune labels leaked into training;
no test/CSV/export/submission was created. Keep this on resume.

**Guardrail:** do NOT submit anything that does not beat tune PQ 0.4487 (it would
regress public 0.39). Do NOT deallocate the shared `rogii-gpu` VM.

---

## 7. ADDENDUM 2026-10-02 — V17 IS reconstructable (V50)

§0/§2-V49 were wrong: V17's artifacts were never lost — they were in the LOCAL repo.
All pinned SHAs match V17's manifest:
- V6 detector `0abf568f` → `outputs/v6_inference_source/yolo_best.pt`
- V16 seed1 `ae476e66` → `outputs/kaggle_v16_full/checkpoints/arm_r1_best.pt`
- V16 seed2 `097a57ed` → `outputs/kaggle_v16_full_seed2/checkpoints/arm_r1_best.pt`
- V11 refiner `fae496b9` → `outputs/kaggle_v11_full/checkpoints/proposal_centered_e4.pt`
- canonical refiner `8d008053` → `outputs/kaggle_coco_refiner/best_refiner_coco.pt`
- 744 signed V17 caches → `outputs/kaggle_v17_cache_pull/proposal_cache/` (all hash +
  cache_id verified: seed1 iou030 train141/test180, iou070 train141; seed2 both cells train141)
- 321 frozen V6 caches → `outputs/kaggle_detector_tta/detector_proposal_cache/`

**V50 result:** `scripts/reproduce_v17_from_cache.py` (signed caches → V11 refiner → PQ;
no detector run) on rogii, 4 shards, ~90 s: selected `seed1/mr1_iou030/t0.45_c0.30` =
`[481.0093, 704, 306, 430]` → **tune PQ 0.448703, exact**. All 168 candidates: 0 TP/FP/FN
differences; 14 seed1 t0.40 candidates differ in iou_sum by −0.00029 (max ΔPQ 3.3e-7).
Torch 2.4.1 on VM vs 2.10 on Kaggle does not matter. Audit not run (still sealed).
Evidence: `outputs/v50_v17_reproduction/`. All these artifacts are now on the VM under
`~/solar-filament-2026/outputs/` (durable disk).

Next: add V44/V21 proposals as an extra source on top of the V17 proposal set (tune only),
submit only if tune > 0.448703.

---

## 8. ADDENDUM 2026-10-02 — V51 V17+V44 fusion: NEGATIVE (gate failed)

`scripts/v51_fusion.py` on rogii, 4 shards, 94 tune files, 714 candidates
(extra_set {s, sml} × alpha {0,.25,.5} × beta {0,.5,1} × V17's 6×7 t/c grid).
Extra caches: `/mnt/v51/cache/{s,m,l}` (94 each, ephemeral).

- Control `s/a0.00_b0.00` reproduced V50/V17 **exactly** at all 42 t/c cells
  (`[481.0093, 704, 306, 430]`, PQ 0.448703) → `control_verified=True`.
- **Selected = the control itself** (delta 0.0). Promotion bar 0.458703 → `audit_authorized=False`.
  Audit stays sealed; no test/CSV/submission.
- Best non-control policies (all @ t0.45):
  | policy | PQ | TP/FP/FN |
  |---|---|---|
  | sml a0 b0.5 | 0.448285 | 704/308/430 |
  | s a0 b0.5 | 0.447923 | 705/311/429 |
  | sml a0.25 b0 | 0.445010 | 657/239/477 |
  | s a0 b1.0 | 0.442923 | 707/339/427 |
  | s a0.5 b0 | 0.436614 | 699/365/435 |
- Mechanism: adding V44's unmatched clusters (beta) gains at most +3 TP and costs
  +5 to +33 FP. Blending V44 scores into V17 (alpha) cuts FP only by losing TP at
  a worse rate. V44 is **nested inside V17's proposal set**: what it finds, V17
  already has, and its extras are mostly FPs. This reconfirms E028/V45/V49
  (correlated errors) on top of the exact V17 pipeline.
- Evidence: `outputs/v51_fusion/` (local + VM).

Conclusion: the "add V44/V21 as an extra source" path in §6/§7 is closed. With the
current model families, no measured lever remains with positive expected value.
Keep V17 `56103401` at public **0.39**.

---

## 9. ADDENDUM 2026-10-03 — V52 clDice semantic source + V53 fusion: NEGATIVE (gate failed, first +delta)

Pursued the §6.3 long shot (clDice semantic model as an orthogonal source). Full
detail in `docs/v53_result_2026-10-03.md`. Firewall intact throughout.

**V52 training** (`scripts/train_v52_cldice_semantic.py`, pure-torch U-Net, BCE +
soft-Dice + soft-clDice, FIT-only). 4 variants trained+tuned on rogii:
| variant | best thr | standalone tune PQ | recall |
|---|---|---|---|
| w1 (clDice-w1.0) | 0.60 | **0.3766** | 0.562 |
| base (w0.5) | 0.40 | 0.3687 | 0.624 |
| b48 (base48) | 0.30 | 0.3696 | 0.604 |
| r768 | 0.40 | 0.3401 | 0.533 |
Weights on VM `outputs/v52_cldice*/weights/v52_cldice_best.pt`. Proposal dumps (94
fusion-tune files each): `/mnt/v52/cache/{w1_t060, base32_t040}` (EPHEMERAL).

**Diagnostic** (`scripts/diagnose_v52_v17_overlap.py`): V52 is the FIRST orthogonal
source found — FP correlation with V17 only **19–24%** (vs V44 nested in V17);
base32_t040 recovers **93 of V17's 317** missed GT instances. The grid was warranted.

**V53 fusion** (`scripts/v53_fusion.py` + `tests/test_v53_fusion.py`, 7 tests pass;
4 shards, 94 files, 756 candidates). Control a0_b0 per set reproduced V50/V17 exactly
→ `control_verified=True`.
- **Selected `base32_t040/a0.25_b0.00/t0.45_c0.40` → PQ 0.449278** (tp675/fp249/fn459).
  Delta **+0.00058** vs V17; bar +0.010 → `audit_authorized=False`. Audit sealed; no
  test/CSV/submission.
- Mechanism (monotone across grid): **beta/recall-add is strictly harmful** (V52's 519
  FPs flood faster than its 93 recoverable TPs; PQ → 0.38 at b1.0). **alpha=0.25
  agreement up-weighting** is the only positive lever (+0.0006), precision-preserving
  but sub-noise (ledger E024: <0.01 local never moves public).
- Evidence: `outputs/v53_fusion/tune_merged.json`, `outputs/v53_diagnostic/`.

**Takeaway:** source correlation is no longer the blocker (V52 solved that). The new
blocker is **FP/TP separability of V52's recovered instances** — mean-prob score can't
gate them. Keep V17 `56103401` at public **0.39**.

### NEXT STEPS (resume here, by remaining EV)
1. **Recalibrate V52's unmatched-proposal score** before the beta-add: fit a logistic
   gate on cheap per-proposal features (compactness, skeleton length, mean_prob×area)
   on FIT only, re-dump scores, re-run the v53 grid. This directly attacks the one
   thing that killed beta>0. Low cost (reuses V17 cache + V52 weights). Untried.
2. **Precision-focused V52 retrain** (w1 already cut FP 519→364 vs base): push with
   FN-weighted BCE and/or full-2048 training to sharpen thin structures and cut FPs.
3. **V52 variant intersection** (w1 ∩ base) as a cleaner, higher-precision extra source.
Guardrails unchanged: submit only if tune PQ > 0.4487; do NOT deallocate rogii.

---

## 10. ADDENDUM 2026-10-03 — V54 score recalibration (rescue V52 fusion recall): NEGATIVE

Attacked the §9 next-step #1: V53 showed V52 is orthogonal but its mean-prob score
can't separate recovered TPs from FPs (so beta>0 hurt). V54 fit a FIT-only logistic
quality gate on 10 cheap features to replace that score. Full detail in
`docs/v54_result_2026-10-03.md`. Firewall intact; GPUs idle; rogii not deallocated.

- FIT training set: 5318 V52(base32@0.40) proposals, 3148 TP / 2170 FP (566 fit files).
- Gate AUC **0.8265** vs mean-prob baseline **0.8224** = **+0.004 only**. mean_prob
  weight 1.571 dominates; all mask-geometry features |w|≤0.35.
- **All +0.004 is from `max_prob` alone**: gate without max_prob = AUC 0.8225 = baseline.
  Mask-geometry (compactness, skeleton, elongation, solidity, extent) adds ZERO signal.
- max_prob needs a prob map → re-running V52 on tune, which is NON-DETERMINISTIC (all
  141 re-dumps had different proposal sets than the originals) → confounds score vs
  proposals. The only deterministic-from-dumps gate ≈ mean-prob → would reproduce V53's
  +0.0006. Did NOT run the grid (negative EV + confounded); removed confounded dumps.
- **Conclusion:** V52's FPs are not geometrically distinguishable from its TPs — the
  hard-FN wall also blocks post-hoc gating of an orthogonal source. Keep public **0.39**.

Artifacts: `scripts/{dump_v54_fit_features,fit_v54_gate,rescore_v54_tune,v54_fusion}.py`,
`tests/test_v54_gate.py` (4 pass), `outputs/v54_gate/{gate.json,fit_features.npz}`.

### NEXT STEPS (resume here)
Post-hoc scoring is exhausted; the separation info isn't in cheap geometry. Remaining:
1. Bake the quality objective INTO V52 training (FN-weighted BCE / focal) — change what
   the model is confident about, not how it's scored.
2. Full-2048 V52 retrain (sharpen thin structures; attack FP precision directly).
3. Prob-map features (gradient along skeleton, boundary sharpness) + make V52 inference
   deterministic (seeded, sorted CC, no TTA resize jitter) to enable a clean score A/B.
Guardrails unchanged: submit only if tune PQ > 0.4487; do NOT deallocate rogii.

---

## 11. ADDENDUM 2026-10-03 — V55 precision-focused V52 retrain: NEGATIVE

Attacked §10 next-step #1/#2: retrain V52 with a precision loss to cut FP at the source.
Full detail `docs/v55_result_2026-10-03.md`; mid-training analysis
`docs/v55_learnings_2026-10-03.md`. Firewall intact; rogii not deallocated; 147G free
(biohub 136G deleted this session).

Loss = focal_bce + tversky(α_fp>β_fn, precision bias) + cldice_w·clDice
(`scripts/train_v55_precision_semantic.py`, imports V52, V52 untouched; V52-compatible
ckpts → V52 `--mode tune`). `tests/test_v55_precision_loss.py` 4 pass. 4 variants, 140ep.

Standalone tune PQ (141 files): V52 w1 **0.3766 fp364** (still best) >> V55 tv73 0.3709
**fp632** > V55 tv82 0.3609 **fp648**; clDice-w2.0 variants collapsed (tv64_cl2 PQ 0.0056,
oscillating/unstable). **Precision training RAISED instance-FP and cut recall — opposite
of intent.**

Mechanism: pixel-level Tversky FP-penalty shrinks/fragments masks → fewer TP pixels but
MORE boundary-fragment FP *instances* after threshold→connected-components. V52's FP are
confident filament-shaped hallucinations (a confidence/localisation error), untouchable
by pixel reweighting. **Pixel-precision ≠ instance-precision** on a semantic→CC pipeline.

Decision: did NOT run V55→V17 fusion (every variant worse than V52 w1 as a source → would
only reproduce V53's +0.0006). V55 closed as negative. Keep public **0.39**.

The three orthogonal-source attempts now bound the problem: V53 (V52 orthogonal but FP
not gatable → +0.0006), V54 (no cheap post-hoc score helps), V55 (precision *training*
backfires on instance FP). NEXT (lower EV): instance-aware head (not pixel loss) so the
FP penalty acts on instances; confidence calibration (ensembles/MC-dropout) vs low-contrast
hallucinations; Skeleton-Recall loss (noise-robust) at clDice≤0.5 w/ warm-start + γ=4/3.
Guardrails unchanged: submit only if tune PQ > 0.4487; do NOT deallocate rogii.

---

## 12. ADDENDUM 2026-10-04 — V56 Mask R-CNN pivot + SPARSE-ANNOTATION breakthrough

Full detail: `docs/v56_result_2026-10-04.md`, `docs/annotation_sparsity_finding_2026-10-04.md`.
Firewall intact (FIT-only); rogii not deallocated; disk 146G free (biohub 136G deleted).

### Leaderboard reframing (first look at the actual competition)
Live LB: **top 0.63, dense cluster 0.55–0.63**. Our 0.39 is far down the board. The whole
"0.4487 ceiling / E027-E028 hard-FN wall" describes V17's YOLO pipeline, NOT the
competition. Public "0.70"/"0.55" kernels = FAKE embedded-payload CSVs (verified). One
REAL competitive kernel: `maskrcnn_resnet50_fpn_v2` COCO, 1024px.

### V56 Mask R-CNN (`scripts/train_v56_maskrcnn.py`, 7 tests, ruff clean) — single-GT NEGATIVE
Sweep (141 tune, single-annotator GT): all variants **0.354–0.383** PQ, below V17 0.4487
and not reliably above V52 w1 0.3766. Recall caps ~0.55–0.58, FP 468–520 (V17 306).
CORRECTION: an earlier "resolution scaling law" (1536=0.3833) was VARIANCE — 1536_e40 same
config scored 0.3579. Instance-aware architecture alone does not break the wall. No submission.

### THE BREAKTHROUGH — MAGFiLO annotations are SPARSE (BioHub transfer, user's steer)
BioHub (royerlab Cell Tracking) baseline: sparse GT, "unannotated cells IGNORED during
training." Tested on MAGFiLO and CONFIRMED:
- 707 files but 1154 image-records = ~1.63 annotators/file; 296 files multi-annotator.
- min/max annotator coverage ratio **0.677** (each misses ~32% of what another marked).
- UNION vs single-max-annotator = **1.347×** (~35% more real filaments, each human-marked).
Every model (V17/V52/V55/V56) trains + scores on ONE max-annotator record → punished for
detecting real-but-unannotated filaments → suppressed recall + inflated FP. Partly
explains the ~0.35–0.38 ceiling AND E027 (public likely union-scored → aggressive-recall
wins there, loses on our single-annotator tune).

### Union support built + cheap confirmation
Added `union_rles`/`union_instance_masks`, `--target {single,union}`, `--eval-gt
{single,union}` (FIT-only firewall preserved). Union-GT re-eval of existing single-trained
weights: TP↑ (664→694) FP↓ (448→418) — confirms ~30 "FP"/variant are real filaments — but
PQ dips (0.3833→0.3746) because FN↑ (481→603): single-trained model can't cover the 35%
larger union set. => must TRAIN on union.

### RUNNING NOW
`v56_mrcnn_1536_union` + `v56_mrcnn_2048_union` (`--target union`, FIT-only, ~1.5–2h). The
decisive test: does union-TRAINING lift recall/PQ? Tune-eval each vs union AND single when
done. If union-trained recall climbs materially → real lever + submission candidate (gate:
tune > 0.4487+0.010 OR strong public-transfer rationale → ask user before a daily submission).

### NEXT STEPS (resume here)
1. Tune-eval `v56_mrcnn_{1536,2048}_union` vs union + single GT; compare to single-trained.
2. If union-trained union-PQ >> single-trained, consider: (a) union-train V52 semantic too;
   (b) a submission (recall-oriented, since public is likely union-scored — the strongest
   public-transfer rationale we've had). 3. Also worth: sparse-supervision loss MASK
   (ignore unannotated regions) rather than hard union, the exact BioHub technique.
Guardrails unchanged: submit only on gate/evidence + user go-ahead; do NOT deallocate rogii.

---

## 13. ADDENDUM 2026-10-04 (cont.) — union-TRAINED CONFIRMED NEGATIVE; data-lever closed

Retrained Mask R-CNN `--target union` (FIT-only), 4 configs, tune-eval vs union GT:
- single-trained union-GT: 1536 **0.3746**, 2048 **0.3665**
- union-trained  union-GT: 1536 **0.3576**, 2048 **0.3564**, 1536_e40 0.3495 (2048_lr2e3 same trend)

ALL union-trained variants BELOW single-trained. Recall FLAT ~0.52–0.54 (no more real
filaments found), FP UP ~545. Multi-variant confirmation (not a single-point read).

**Final:** the sparse-annotation finding is real (1.63 annotators/file, 0.677 coverage,
1.35× union) but **hard-union targets do NOT convert to score** — the extra union
instances are too noisy (one annotator made, another rejected) → model fires looser →
more FP, flat recall. Best standalone instance model = single-trained 1536 single-GT
0.3833, still < V17 0.4487. **No submission** (nothing beat the bar).

### Whole V53–V56 arc (this session) — all negative vs V17 0.39, but each informative
- V53 fuse V52(clDice) into V17: +0.0006 (orthogonal but FPs not gatable).
- V54 recalibrate V52 score: no cheap score beats mean-prob.
- V55 precision-Tversky retrain: raised instance-FP (pixel≠instance precision).
- V56 Mask R-CNN pivot: 0.354–0.383, below V17 at all res/epoch/LR.
- Sparse-annotation finding (BioHub transfer): REAL + measured, but hard-union training negative.

### NEXT (open, by EV)
1. **Sparse-supervision LOSS MASK** (BioHub's real technique, untried): ignore unannotated
   regions in the loss (not pos, not neg) instead of forcing noisy union positives. Needs
   custom Mask R-CNN RPN/ROI target assignment — bigger build.
2. **≥2-annotator-union** (confidence-weighted): add only union instances ≥2 annotators
   agree on — higher-precision subset than full union. Cheap (reuses the union builder).
3. Accept the data-lever as explored and hold V17 at public 0.39.

Guardrails: submit only on gate (tune>0.4587) or strong public-transfer evidence + user
go-ahead; do NOT deallocate rogii; firewall FIT-only. All docs synced; weights durable.
