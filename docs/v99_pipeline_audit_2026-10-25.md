# V99 pipeline audit (2026-10-25): what was verified, what was closed on weak evidence, what was never run

Three independent read-only audits (detector + labels; refiner + selection + export; metric + protocol),
each with file:line citations, plus a check of the experiment ledger. No training, no test data.
Context: OOF PQ 0.459, public 0.39, clean frontier 0.39–0.42 (`docs/v98_leaderboard_audit_2026-10-25.md`).

## 1. Verified correct (no action)

- Scorer = host self-evaluation notebook rule for rule: hit matrix `iou > 0.5`, no 1:1 matching, FP = zero
  columns, FN = zero rows, pooled micro counts, k copies of each prediction for k annotator records,
  strict `>`, ID parsing. Identical TP/FP/FN on 707 files, IoU sums to 1e-6 (`v61_toolkit.py:121-150`,
  `v69_metric_audit.py`, `docs/v69_kaggle_metric_audit`).
- Export path = OOF path bit-for-bit on 707 files (V87); host-notebook score of the CSV = internal counts.
- Fold split: unit = physical file, all annotator records together; test is a random file sample of the
  same observations; same-day neighbour effect ≤ 0.003 (V85).
- Non-overlap enforcement costs nothing recoverable by rule (V88, 5 ownership rules, all ≤ 0).
- Labels: every annotation is one polygon, no holes, no crowd; the `len(poly) >= 6` filter drops nothing.
- Closed on 707-file 5-fold evidence: refiner threshold, clamp size, ±1 px width (also on public), min-area,
  global cut (also on public), soft/union targets, crop 512, encoder, SSL init, pooled-source selector,
  longer detector schedules (two negatives), yolo11m, limb division, label transfer, annotator-style
  training (V96/V97).

## 2. Closed on weak evidence (single fold, proxy gate, or a frozen downstream stage)

| item | what happened | why it is weak |
|---|---|---|
| "Refiner absorbs detector gains" (1280 +0.0142 direct 5/5 folds → −0.003 refined; cp; thin; E_p2x2) | judged through a refiner trained on p2first-1024 proposals and frozen | `v77_result`: "The refiner was not retrained on 1280 proposals; that variant is untested." The only co-trained case (r3cp) has no recorded number. 1280 OOF caches for all 5 folds and the 5 fold checkpoints survive in `outputs/v61_oof_cache/p2first_1280`, `outputs/v68_persistent`. |
| `p2first_flip` (fliplr/flipud 0.5 at train) | fold 0 +0.0068, P 0.89, "inconclusive", never followed | training uses `fliplr=0` while inference averages 4 flip views, so 3 of 4 views are out of distribution |
| Seed variance of the deployed refiner | V89: seeds 43/44/45 = 0.4561/0.4574/0.4559 vs deployed seed 42 = 0.4590 | the shipped refiner is the luckiest of four draws (+0.0025 above the fresh-seed mean). V90 `ens4` was +0.0028 vs that mean but KILLed against the lucky seed, without demote refit. Every refiner bootstrap excludes training-seed variance; V84 passed its gate by +0.00004. V82/V84 quote "retraining noise 0.0001" from a same-seed rerun. |
| Stage A/B schedule | one comparison (r1 vs r3) on 141 files | never swept on 707 files |
| Multi-component masks / duplicate fragments | E016, V71 R1 on V4/r3-era refiners, 141 files | never re-run on the V84 refiner |
| Gate design | every V68+ arm screened on fold 0 (141 files, CI ±0.011) with CONTINUE at +0.010 | pooled seed noise is 0.0015; arms worth +0.003..+0.008 are discarded individually and never stacked. `v69_history_audit` recommended stack-then-test; it was never run. |

## 3. Never run

| item | headroom (measured) | cost |
|---|---|---|
| Refiner trained with hard-negative crops + veto (refiner never sees a junk proposal; empty output falls back to the seed, so every seed survives) | 811 pure-junk kept masks = +0.037 exact; oracle selection on our own pool 0.528 (+0.074) | 1 fold-0 train (~25 min GPU) + CPU extract |
| Split masks that swallow two GT filaments | 305 FN + 21 FP = +0.019 exact (V85) | CPU on the V88 extract |
| Refiner co-trained on the 1280 detector's proposals (then flip, cp) | 1280 is +0.0142 direct, 5/5 folds | 1 fold-0 train + extract, caches exist |
| Rot90/rot270 test-time views on p2first (8 views) | proposed in `v69_literature_research` #3, never run; `p2first_rot` tested train-time rotation, not TTA | OOF inference 707 files (~40 min GPU) + refined harness |
| Score saved intermediate detector checkpoints / average last stage-2 checkpoints | two negatives for *longer* training suggest the fixed 100-epoch point is past the optimum; no epoch curve exists (`val=False`) | inference only, if `save_period` checkpoints survived the Oct 17 wipe |
| Union-of-annotators labels with per-record repetition (V81 `union` arm, built, never launched) | removes 2,764 positive-vs-background conflicts on 296 multi-record files | 1 fold-0 detector run (~4 h) |
| Demote feature shift under the all-data refiner (`demote_all` fit on fold-refiner features) | unquantified; a free export fix if non-zero | 30 min CPU |
| Hole filling on final masks; INTER_AREA crop downsampling | expected ≈ 0 | CPU minutes |
| Demote λ at the test PQ level (τ = PQ_test/2 ≈ 0.195 vs deployed 0.229) | protocol mismatch; < 0.005 expected | CPU |
| Private kernel vs public notebook (pooled vs per-image mean; 1:1 matching) | every "host-exact" claim is notebook-exact; V92/V95 only bound it indirectly | one public probe (split-half by density), submissions currently paused |

## 4. Reading

Nothing found explains the 0.07 OOF→public offset; all three audits agree it is test-side. What the audits
found is ~+0.01 to +0.02 of OOF left on the table by the project's own gate design: several +0.003..+0.008
levers were killed one at a time against a lucky seed or through a frozen refiner, and two mechanisms with
the largest measured headroom (refiner veto of junk seeds, splitting swallowed masks) were never built.
At the usual 0.85× transfer that is +0.01 to +0.015 public, i.e. 0.39 → 0.40 at best and inside the board's
±0.012 noise. Standing after cleanup is decided on the private half and the 30 % qualitative rubric.

Plan (V99, pre-registered per arm, fold-0 screens judged against the fresh-seed mean 0.4565 as well as the
deployed 0.4590, then stack-then-test on 5 folds):
1. `soft1280`: V84 soft refiner retrained on p2first_1280 fold-0 proposals, judged with 1280 views.
2. `hardneg`: stage B adds unmatched proposals (max IoU < 0.1) with empty target; fallback → drop.
3. rot90 TTA on the existing p2first fold models.
4. Positive arms stacked into one bundle, 5-fold, gate once at P ≥ 0.95 on 707 files.
Compute: GPUs 1–3 free; disk 8.8 GB free of 248 GB (`outputs/v61_densecrf.collected.pkl` is 32 GB and
obsolete; `~/.cache` is 61 GB).
