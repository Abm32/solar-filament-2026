# SESSION CONTEXT — Solar Filament 2026 (complete state as of 2026-10-06 ~08:00 IST)

Written to be read cold. Covers the whole arc from the 0.39 plateau through today. Numbers quoted
here come from result files/docs listed in §10; where I only have an inference it is marked
INFERRED. Nothing below authorises a submission.

## 0. TL;DR
- Competition: Kaggle "Solar Filament Segmentation Challenge 2026" (GONG H-alpha, 2048px, metric =
  Panoptic Quality, PQ). Host: Azim Ahmadzadeh. Deadline 2026-11-15.
- Best CLEAN public score: **0.39** (V17: YOLO11s-seg mask_ratio=1 detector + 4-view flip
  consensus + V11 refiner; submission 56103401). Every V18-V60 lever failed to beat it.
- Yesterday/today found the FIRST real detector-level gain: adding a **P2/stride-4 head level with
  stride-2 mask prototypes to YOLO11s-seg ("V62 p2first")**. Fold 0: direct-detector PQ
  0.3763 -> 0.3967 (+0.020 at the baseline cut), passed both pre-registered gates. **5-fold OOF
  is running now**; verdict rule is pre-registered. **No submission has been made or is justified yet.**
- Reliable OOF baseline (V17 recipe, direct detector, no refiner, all 707 files): **PQ 0.3856**
  @ conf 0.35, 95% CI [0.376, 0.395]. The old "tune 0.4487" is NOT comparable (refiner + a 94-file
  set that was overfit by selection).
- Rules of the road: firewall (no test labels, no test-GT leak file), do NOT deallocate the VM
  `rogii`, submit only on a pre-registered gate + user go-ahead.

## 1. Environment and state
- VM: `ssh rogii` (Azure, 4x Tesla T4 16GB, 64 CPU, 432GB RAM). Repo `~/solar-filament-2026`, venv
  `.venv` (torch 2.4.1+cu121, ultralytics 8.4.138, Python 3.10, pydensecrf2 1.1). Local mirror:
  `/home/abhimanyu/Desktop/digital_agency/kaggle/solar/solar-filament-2026`. Sync is manual (`scp`).
- **/mnt/v61 is EPHEMERAL** (Azure temp disk; lost on reallocation). Finished runs were copied to the
  persistent `~/solar-filament-2026/outputs/v61_persistent/` (hash-verified) by
  `scripts/_persist_v61.sh`; re-run it as folds finish. Fold manifest `outputs/v61_folds.json`.
- Data: `data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/{train,test}`; 707 train files,
  1154 annotator records, 8199 annotations; 180 test images. Annotation SHA256
  `5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1` (pinned in all scripts).
- Kaggle API creds on the VM (`~/.kaggle/kaggle.json`); submission list readable; kernels output
  download works (used to recover the V16 detector).
- Recurring tooling hazards: (a) batched parallel `strReplace` edits silently drop; apply edits one
  at a time and GREP to verify, or use a python script with `assert count == 1`. (b) inline python
  through ssh heredocs breaks on nested quotes (Py3.10 f-strings) -> write a script file instead.
  (c) `ssh ... sleep` beyond ~60s can time out on detached launches; verify with `pgrep`. (d) Tool
  calls occasionally fail with "missing field path": always re-check a file exists after a create.

## 2. Public leaderboard history (what actually scored)
V17 0.39 (56103401), V21 0.38 (full-data refit of the same recipe), V13/V11 0.39, E015 0.38,
V20 CondInst 0.30, later experimental submissions v2 0.22, v4 0.25, v7 0.30. Five consecutive local
gains (+0.016 total tune) moved the public score by one display step. Public has 1 displayed
decimal -> cannot resolve effects < ~0.01.
Leaderboard pollution (host-confirmed, see `docs/kaggle_discussion_scan_2026-10-04.md`): scores
>~0.41-0.45 used the public test annotations (the MAGFiLO v1.0 Dataverse file contains test GT; we
set it aside and never trained/tuned on it); the board mixed two metric versions and will be
rescored. Host: "anything beyond 0.35 PQ is of great interest", judged by a committee on
predictions and method. Credible clean frontier ~0.46 strict-OOF (a competitor), self-reported.

## 3. Earlier arc (V1-V56), one paragraph each — details in docs/
V1-V17 built the YOLO detector + crop refiner stack; V17 = `arm_r1` seed1 YOLO11s-seg
(mask_ratio=1 continuation) @ box iou 0.3, 4-view flip_consensus, V11 `proposal_centered_e4`
refiner, thresholds t0.45/c0.30; local tune PQ 0.448703 (94 files), 704 TP/306 FP/430 FN.
V18-V49: larger/other detectors (YOLOv8l@2048, RF-DETR, Mask2Former 0.154, CondInst, ensembles,
union pools) and inference tricks — all <= V17. Measured plateau facts: ensemble oracle PQ ceiling
0.585 at zero FP; 18.7% of GT invisible to every detector.
V52-V56 (Oct 1-4): clDice semantic model fused into V17 (+0.0006), score recalibration, precision
Tversky (raised FP), Mask R-CNN pivot (0.354-0.383, below V17), sparse-annotation finding (1.63
annotators/file, union-GT training = negative), external MAGFiLO (zero net-new data; test GT leak
set aside), SWEFil (HF `antonio-reche/SWEFil`, CC-BY, firewall-clean, at `/mnt/swefil`).

## 4. V58-V60 (Oct 4-5): post-hoc levers, all closed
- **V58/59 SWEFil verifier**: +0.0282 PQ on weak OOF detectors but **-0.070 on V17** (does not
  transfer). Spine/topology head contraindicated (V17 fragmentation/merge only ~2.5%; aggressive
  thresholds give FP explosion, not fragmentation).
- **V60 tile detector** (native-res 768 tiles, Mask R-CNN, 4 OOF folds): union oracle recall
  0.728 -> 0.860 BUT a 4-model union inflated it (single tile model 0.688 < V17 0.728; V17+1 model
  0.806). Add-only fusion -0.062 PQ; consensus ranking refuted (correct proposal rank 5-6);
  consensus masks -0.031. CORRECTION I made: the "NMS collapse 108->1" was mis-attributed - the
  add-only V17-overlap rule rejected the tile-only GT by construction (105/149 are V17 near-misses).
  The baseline tile trainer also had a bug (28% empty tiles turned into full-tile "filament" masks;
  22.5% of detections were whole-tile blobs, 0 true positives).
- **V60A mask refiner** (1.9M U-Net): 17/132 reservoir pushed over 0.5 but damaged 172/607 good
  masks; best replacement rule = off. Identity-seed + correction head: AUC 0.51 (chance) -> killed.
  Fixed a seed bug first (fixed erosion emptied 92% of seeds) and an RNG replay bug.
- **V60D one-to-one tile loss** (D0-D3): within-tile duplication is already ~1.0; duplicate penalty
  trades recall for fewer proposals along the same curve -> killed.
- **V61 DenseCRF**: paper-scale settings lowered IoU on good masks and reservoir -> killed.
Literature/datasets searched (EdgeAttNet, MORDEN/AHAS, Diercke 2024, UniInst): no route past V17;
EdgeAttNet splits the full MAGFiLO incl. our test images so its weights are not usable.

## 5. Phase A (Oct 5): clean OOF baseline
- 5 folds over all 707 files, by file (`scripts/make_v61_folds.py`, `outputs/v61_folds.json`),
  group_days=0 (each file its own group; chosen after measuring that test files sit as close to
  train files as train files do to each other). Balanced ~1640 annotations/fold; files
  141/141/142/142/141.
- Recipe = EXACT V21/V17 detector recipe (`scripts/train_v21_full_detector.py`, byte-identical to
  the file that produced V21): YOLO11s-seg from COCO ckpt sha 1caa81c0, 80 ep mask_ratio=2 then 20
  ep mask_ratio=1 (AdamW 1e-4), imgsz 1024, batch 4, single class, final fixed checkpoints.
- Scoring path `scripts/v61_toolkit.py` replicates V17's 4-flip generation + flip_consensus +
  pixel ownership + per-record pq_counts; validated against V17's signed caches (24/24 views
  identical, worst IoU 0.99975, PQ counts identical at 8 confidences, 6 files).
- **Result (direct detector, 707 files): pooled PQ 0.3856 @0.35 (CI [0.376,0.395]); 0.3855 @0.40;
  TP/FP/FN 4503/2442/3696; per-fold 0.376-0.397.** Caches `outputs/v61_oof_cache/<recipe>/fold<K>`.
- Seed-noise control (baseline_seed2, seed 20261005) fold 0: A 0.3763 vs A' 0.3813 (+0.0049, CI
  [-0.005,+0.016]); per-file delta sd 0.083, 25 files move >0.10. Only ONE seed pair.

## 6. Phase B (Oct 5): Focal + Dice mask loss — KILLED
- Found Ultralytics 8.4.138 mask loss is plain BCE (no Dice/Focal). Wrote `scripts/v61_seg_loss.py`
  (BCE + 1.0 Focal(gamma 2) + 1.0 Dice), tested bit-identical to upstream at zero weights, caught a
  `crop_mask` in-place bug via a gradient test, verified it is invoked in real training.
- Pre-registered (`docs/v61_phaseB_preregistration_2026-10-05.md`) BEFORE results. Fold 0: B 0.3810
  (own best cut 0.40) / 0.3779 (baseline cut) vs A 0.3763; B-A +0.0047 (CI [-0.006,+0.016], P 0.80)
  = same as seed noise +0.0049; B-A' -0.0003. Verdict **kill**. Fine-structure recall did not
  improve more than normal recall. No weight sweep, no 5-fold B.
- Incident: B stage 2 (mask_ratio=1) OOM'd (loss memory 2.5x upstream at 1024x1024 masks); fixed
  with 16-instance activation-checkpointed chunking (equal value+gradient, tested), restarted stage 2
  from the finished stage-1 checkpoint; disclosed in `docs/v61_B_oom_incident_2026-10-05.md`.

## 7. V62 (Oct 5-6): P2 head level — the current lead
**Architecture** (`scripts/v62_arch.py`, configs `configs/yolo11s-p2{first,last}-seg.yaml`): append
to the stock yolo11-seg graph: Upsample(layer16) -> Concat(backbone P2 = layer 2) -> C3k2[128]
(layers 23-25); Segment head over 4 inputs. Ultralytics builds mask prototypes from `x[0]` only.
- `p2last` [P3,P4,P5,P2]: adds detection at stride 4; prototypes stay stride 4 (256x256@1024).
- `p2first` [P2,P3,P4,P5]: prototypes at stride 2 (512x512@1024) + detection at stride 4.
**Three implementation problems found and fixed before training:**
1. Inserting layers moves the head 23->26, so the stock loader would silently drop ALL head
   weights (confound). `scripts/v62_init.py` remaps COCO head tensors (555/639 transfer for
   p2last, 509/639 for p2first at nc=1; tested incl. real-model value equality).
2. p2first's cls branch is sized by the first input (64 vs COCO 128ch) -> cannot transfer; a
   disclosed, unavoidable init disadvantage.
3. Ultralytics validator hard-codes 4x proto scale; stride-2 protos crash `mask_iou` (and the trainer
   validates even with val=False). `scripts/v62_patch.py` sets factor 2 for p2first. Both variants
   pass a tiny train+predict smoke test. Multi-GPU is NOT usable (DDP workers lose in-process
   patches) -> one GPU per run.
**Pre-registered** (`docs/v62_preregistration_2026-10-05.md`, before any V62 number): structural
gate on the consensus pool (small+thin oracle recall gain >=+0.030 AND >=2x seed noise; proposals
and FP/image at conf 0.35 <= 1.30x baseline), then PQ gate (>=+0.010, >=2x0.0049, P>=0.90,
positive at baseline cut).
**Fold-0 results** (`docs/v62_fold0_result_2026-10-06.md`, 141 files, direct detector):
| | pool small+thin recall | props/img@0.35 | FP/img@0.35 |
|---|---|---|---|
| A baseline | 0.580 | 6.1 | 1.7 |
| A' seed2 | 0.608 (noise +0.028) | 6.4 | 1.8 |
| p2last | 0.588 (+0.008) FAIL; seed2 +0.019 FAIL | 6.1 | 1.7 |
| p2first | 0.648 (+0.068) PASS; seed2 +0.059 PASS | 6.2 | 1.6 |
PQ: p2first 0.3967 @0.35 (+0.0204), 0.3991 @0.30 (+0.0227, CI [+0.010,+0.036], P 1.000) vs A 0.3763;
mean matched IoU 0.6444 -> 0.6521; small recall 0.390 -> 0.440; thin recall DID NOT improve (0.616
-> 0.581, n=86). Seed-matched confirmation p2first_seed2 vs baseline_seed2: +0.0128 (CI
[+0.0013,+0.0245], P 0.985). Mechanical verdict: continue. Averaging the two pairs suggests the
true effect is closer to ~+0.018 (INFERRED). Attribution hypothesis (NOT proven): stride-2
prototypes, not P2 detection, carry the gain; confounded by init differences. Cost: p2first stage 2
~298 s/epoch, 12-13 GiB on a T4.

## 8. RUNNING RIGHT NOW (as of 08:00 IST)
- `p2first` folds 1-4 training, one per GPU (GPU k-1), started ~07:40 IST; monitor
  `python3 scripts/_v62_monitor.py` (checks process + results freshness + log tail; the standard
  since an unmonitored OOM went unnoticed on Oct 5). ETA ~11:00 IST.
- Chains (`scripts/_v62_chain.sh <fold> <gpu>`, running detached): wait for p2first fold K ->
  held-out inference -> train `baseline_seed2` fold K -> inference. Each writes
  `outputs/v62_chain_fold<K>.log` and `.done`. seed-2 baselines finish ~14:30 IST.
- Pre-registered 5-fold rule (`docs/v62_5fold_preregistration_2026-10-06.md`), computed by
  `scripts/eval_v62_5fold.py` (tested pure `verdict()`): primary cut 0.35; CONTINUE iff pooled
  d >= max(+0.010, 2n), P(d>0) >= 0.90, CI lower > 0, >= 4/5 folds positive, small recall rises,
  B-own-best d > 0, proposals and FP/image <= 1.30x. KILL iff d < +0.005 or < 3/5 folds positive.
  Otherwise inconclusive (no further compute). My stated expectation: pooled +0.010 to +0.018.
- To run the verdict once everything is cached:
  `python scripts/eval_v62_5fold.py --a baseline --b p2first --noise baseline_seed2 --out outputs/v62_5fold_p2first.json`
- Persist checkpoints: `bash scripts/_persist_v61.sh` after folds finish.

## 9. Decisions, open questions, and what is NOT established
- **Can we submit p2first? No, not yet** (user asked 2026-10-06 07:48; I advised against). It is a
  one-fold, detector-only result; the 0.3967 is on a different scale from V17's refiner-included
  0.4487/0.39; the public board (1 decimal) cannot resolve a +0.01-0.02 effect. A valid submission
  needs: (1) 5-fold verdict = continue, (2) a refiner retrained on p2first proposals and evaluated
  OOF (the V11 refiner was trained on the old detector's proposals; transfer untested), (3) an
  all-707-file p2first retrain (~3.5 h), (4) test export with the same recipe + repo validator
  `scripts/validate_submission.py`, (5) one submission with user go-ahead. Optional cheap probe
  (user has NOT chosen it): fold-0 p2first + existing V11 refiner on test, expected ~0.38-0.39,
  tests the pipeline not the idea. Export code to adapt: `scripts/export_v21_shard.py`.
- Not established: that the effect survives 5 folds; that the refiner adds the same ~+0.016 on
  top of p2first; that stride-2 prototypes are the cause; any thin-filament benefit.
- Next ideas, ONLY after the 5-fold result (user-ranked): better use of P2 features for mask
  formation; P2-aware training target for fine extensions; then magnetogram/unlabeled-GONG
  training-only auxiliary signal (host-approved; never at inference; neighbouring frames at
  inference are forbidden). Do not mix with V62.
- Rejected/closed lines (do not revisit without new evidence): tile selection/NMS/consensus,
  verifier reranking on V17, refiner gating, DenseCRF, one-to-one duplicate loss, Focal+Dice
  (this formulation), spine/topology head, hard-union annotator training, 1536/2048 imgsz alone.

## 10. File map
Docs (all under docs/): `SESSION_RESUME_2026-10-01.md` (older, ends Oct 4 11:18),
`kaggle_discussion_scan_2026-10-04.md`, `v60_segmentation_first_reframing_2026-10-04.md`,
`v60a_refiner_result_2026-10-04.md`, `v61_phaseB_preregistration_2026-10-05.md`,
`v61_phaseB_fold0_result_2026-10-05.md`, `v61_B_oom_incident_2026-10-05.md`,
`v62_preregistration_2026-10-05.md`, `v62_fold0_result_2026-10-06.md`,
`v62_5fold_preregistration_2026-10-06.md`, and this file.
Key scripts: `make_v61_folds.py`, `train_v61_fold.py` (recipes: baseline, baseline_seed2, focal_dice,
p2last, p2first, +_seed2 variants), `eval_v61_oof.py` (infer/pool/diff), `eval_v61_compare.py`
(fold comparison + verdict), `analyze_v61_seed_noise.py`, `analyze_v62_structural.py`,
`eval_v62_5fold.py`, `v61_toolkit.py`, `v61_seg_loss.py`, `v62_arch.py`, `v62_init.py`,
`v62_patch.py`, `_v62_monitor.py`, `_v62_chain.sh`, `_persist_v61.sh`.
Tests: tests/test_v61_*.py, test_v62_*.py (all passing when last run).
Outputs on VM: `outputs/v61_oof_cache/`, `outputs/v62_structural_fold0*.json`,
`outputs/v62_compare_*.json`, `outputs/v61_compare_fold0.json`, `outputs/v61_persistent/`.

## 11. Resume checklist for a fresh session
1. `ssh rogii`; `python3 scripts/_v62_monitor.py`; `ls outputs/v62_chain_fold*.done`.
2. If folds crashed, resume the stage from `base/final.pt` (trainer skips finished stages).
3. `bash scripts/_persist_v61.sh`.
4. When all five folds of baseline, baseline_seed2 and p2first are cached:
   `python scripts/eval_v62_5fold.py ...` and record the mechanical verdict in a new doc.
5. Only after a `continue`: build the matched pipeline (refiner on p2first proposals, OOF-check it,
   all-data retrain, export, validate, ONE submission with user approval).
