# V86 pre-registration: pooled multi-detector candidates + learned selector (2026-10-12)

Written BEFORE any V86 number exists (no V86 extract or selector run). GPUs 0-2 only (never 3); this experiment is
CPU only. No submission, no test data/labels, no MAGFiLO Dataverse test GT, no leaked CSVs, no SWEFil.
Code: `scripts/v86_extract.py` (stage 1), `scripts/v86_select.py` (stage 2). Outputs: `outputs/v86/`.

## Sources (inventory, done before writing this)
Raw 4-flip OOF views are cached in `outputs/v61_oof_cache/<recipe>/fold{0..4}` (V61 harness). Full 5-fold coverage
(141/141/142/142/141 = 707 files): `p2first` (V61/V65), `p2first_1280` (V68b), `baseline` (V61), plus
`p2first_cp` (V75 member) and `baseline_seed2`. `p2first_ns` (V76 noisy student) exists for fold 0 only
(`/mnt/v76/p2first_ns_fold0`, no fold 1-4 weights) and is EXCLUDED (not regenerated: would need 4 new pseudo-label
student trainings). Every source uses the same V61 fold map, so the fold-k proposals of every source are OOF.

## Stage 1: per-source refined candidates (exactly the V84 path)
For every file and every source: 4-flip views -> flip_consensus groups (match 0.5, score = sum/4) -> V84 soft fold-k
refiner (leak-free, trained without fold k), t = 0.50, k7 clamp, greedy pixel ownership within the source ->
candidates = alive groups with consensus score >= 0.15 and the 28 V83 features (`v83_extract.refine_feats` /
`candidate_rows`), plus the post-ownership RLE and per-record IoU vs GT.
Reproduction assertion: for `p2first`, per-file counts at cut 0.30 match `outputs/v84/soft_fold*.json`
(k7, t0.50_c0.30) within 1e-3 (as V84).

## Stage 2: pooled selector
- Pool P = union of the chosen sources' candidates per file.
- Features = the 28 V83 features + source one-hot + cross-source agreement: for every source j in the pool, max IoU
  of this candidate with source j's candidates (own source excluded for itself = 1.0) and the score / p_in of that
  best match; n sources with an IoU > 0.5 match (incl. own); mean consensus score over the matching sources; number of
  pooled candidates with IoU > 0.1; pooled candidate count in the file.
- Target / weight = V84 (min(1, sum over records of matched IoU>0.5 / R), weight R). LightGBM = V83 params,
  300 rounds (`outputs/v70/pylib`).
- Cross-fitting by the 5 V61 folds: the model scoring fold k is trained only on folds != k; the lambda for fold k is
  the Dinkelbach fixed point on inner cross-fit predictions (models trained on folds not in {k, j} predicting fold
  j), over folds != k only. Fold-k labels are never used for fold-k decisions.
- Selection per file (non-overlapping): candidates in descending y-hat; each takes its pixels minus already-owned
  pixels; it is skipped if the remaining area < 200 px (MIN200) or < 50% of its own area (cross-source duplicate);
  otherwise it is accepted and owns its pixels. Keep the accepted masks with y-hat > lambda/2 (since acceptance is in
  y-hat order, this is a prefix; skipped masks never own pixels). Eligibility = all candidates (score >= 0.15).
  The 0.5 retain fraction and 200 px are fixed here and not tuned.
- Scoring = exact per-record pooled PQ on the trimmed accepted masks (TP iff IoU > 0.5).

## Arms (incremental)
- P1 = p2first only (control for the selector change: same pool as V84, new selector/selection).
- P2 = P1 + p2first_1280. **P3 = P2 + baseline (PRIMARY: all task-named sources with full coverage).**
- P4 = P3 + p2first_cp, P5 = P4 + baseline_seed2 (secondary, report only).

## Primary (one comparison)
- Baseline = V84 bundle per-file counts (`outputs/v84/export/parity/bundle_ref.pkl`, pooled PQ 0.45898, asserted).
- d = PQ(P3) - PQ(V84), paired file bootstrap (2000 resamples, seed 86), 95% CI and P(d>0); per-fold d reported.
- **Gate: CONTINUE iff d >= +0.010 AND P(d>0) >= 0.90.** Otherwise KILL (pooled selector not adopted).

## Secondary (report only, no decision)
- P1, P2, P4, P5 vs V84 (same bootstrap); P3 vs P1 (the gain attributable to extra sources, not selector change).
- Oracle ceiling: selection with true y in place of y-hat and Dinkelbach lambda on all 707 files, for P1, P3, P5.
- Pool sizes (candidates, per-source), LightGBM feature importance (P3).

## Disclosed simplifications
- Each source is refined with its own within-source ownership first (V84 path), then re-trimmed in the pool. Target y
  is computed on the untrimmed source mask, scoring on the trimmed mask.
- Oracle = the same greedy-ownership selection rule driven by true y; it is not a global optimum.
