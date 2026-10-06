# Kaggle discussion scan — filament-segmentation-2026 (2026-10-04)

Pulled via authenticated Kaggle internal API (forumId 11317517,
`discussions.DiscussionsService/GetForumTopicById` with `includeComments:true`).
20 topics; the high-signal ones below. Host = **Azim Ahmadzadeh (@azimahmadzadeh)**.

## A. SCORE FRONTIER — our leaderboard-pollution read is CONFIRMED by host + top competitors
- **Alisher Myrgyyassov:** "any score above ~0.45 (or even 0.41) has used the publicly
  available test dataset annotations." Judges aware.
- **Host (on the 0.55 CSV):** "You are not wrong. Please do not let random scores discourage
  you. We are only interested in real solutions, and we can easily weed out the fake ones."
- **Leaderboard mixes two metrics** (topic 733912): same byte-identical file scored 0.66 →
  0.31 after a metric update. Host: Kaggle will re-score the whole board; new metric now live.
- Host repeatedly: **"Anything beyond 3.5 PQ [0.35] is of great interest to us"**, judging is
  by a committee looking at predictions + algorithm logic, NOT the single PQ number.
=> Our recalibrated target (clean ~0.45–0.47 OOF, NOT the polluted 0.55–0.63) is correct.
   V17's clean 0.39 is already in the "of great interest" band and judged on method.

## B. STRONGEST LEGITIMATE COMPETITOR — "Spaghetto99" (topic 744765) & "_.tushhaaarrrr"
- **Spaghetto99:** self-reported PQ **0.4843 → 0.4871** on a *casual* 106/707 (then 170)
  validation split (TP~855-863, FP~293-313, FN~395-403, SQ~0.68, RQ~0.70). Real-time model
  (<2s/image). Key quotes:
  - **"I used a DECOUPLED pipeline to get past 0.44 PQ."** (not Diercke et al. 2024)
  - **"In my new model I'm giving 100% weight to segmentation only"** (dropped spine weight).
  - "adjusting Focal Loss + Dice Loss is key."
  - Confirmed his val images are drawn from train (no val folder) → split differs per person,
    so his 0.48 is NOT directly comparable to a strict-OOF number.
- **_.tushhaaarrrr:** **0.4600 STRICTLY OUT-OF-FOLD** (5-fold CV, every stacked input/scorer/
  threshold cross-fitted; pooled over every annotator, matched at IoU 0.5). States own split
  variance ~**0.012**, up to **0.02–0.03 if the split clusters by annotator group**. Told
  Spaghetto99 "you're clearly ahead on DETECTION; you've found more of the filaments."
=> IMPORTANT: our tune is 141 random-split; expect ±0.01–0.03 noise. The credible clean
   frontier is ~0.46 OOF (tushhaaarrrr), and the gap vs our V17 0.4487-tune is "finding more
   filaments" = DETECTION/recall — EXACTLY the reservoir V60A tiling opened (0.728→0.86 oracle).
   Decoupled pipeline + segmentation-weighted + Focal+Dice independently matches our direction.

## C. PQ SCORING MECHANICS — confirms our harness assumptions
- **Siva Kumar (735377):** raw Mask R-CNN ERRORs under PQ because instances share pixels; PQ
  requires **pixel-disjoint** predictions per image. Fix = score-ordered painting (sort by
  score desc, each instance claims only unclaimed pixels). Host confirmed: "we do not accept
  overlapping predictions." => our owned-pixel-suppression PQ scorer is correct + necessary.
- **Rahul S. (740400):** argues the metric matches a loose multi-annotator consensus, so a
  physically-correct mask can be an FP; detecting a real low-agreement filament can be net
  negative. Host pushback: that variance is the real-world problem; multi-value rubric + human
  judging mitigates it. => aligns with our "faint/low-consensus FN partly irreducible" finding.

## D. FIREWALL / RULES — host rulings (direct quotes), decisive for our data use
- **spine/bbox/area/category_id ARE allowed as training supervision** (736543): host —
  "You can use other meta data available in the dataset (spine, bbox, area, category_id)."
  spine & category_id are manual; bbox & area derived. Auxiliary targets from masks
  (skeleton/medial-axis/boundary/distance-transform) implicitly fine. => our spine-aux head
  and any mask-derived target are LEGAL. (Moot given spine was contraindicated, but confirmed.)
- **External UNLABELED GONG H-alpha for pretraining = YES** (737215, 744047, 743385): self-
  supervised / temporal-consistency pretraining on public GONG frames OK **if** no test GT
  leaks and **inference uses ONLY the supplied test image**.
- **Co-temporal GONG magnetograms (bzi) as TRAINING-ONLY aux signal = YES** (743521) — e.g.
  a PIL/polarity-inversion-line loss to separate filaments from sunspots. Never at inference.
- **Neighbouring frames AT INFERENCE = NO** (743385, 744047): "no extra observations during
  inference" — single-image task, not video. Temporal info is training-only.
- **Grayscale→3-channel replication = allowed** but host notes 3× compute for no clear gain
  (744544). => validates our 1-channel-native choice for the refiner/tiler.
- What's forbidden: "any piece of information obtained from external sources leaking the
  ground-truth data of the test set." => our SWEFil use (disjoint images, own labels) and
  MAGFiLO-superset handling (test GT set aside, never trained) remain firewall-clean.

## E. NEW LEGITIMATE LEVER surfaced by the host rulings (not yet tried)
- **Magnetogram / unlabeled-GONG training-only aux supervision** is explicitly blessed and
  physically motivated (filaments lie on polarity-inversion lines; sunspots are single-polarity
  flux — the exact confuser our verifier couldn't separate from MAGFiLO alone). This is a
  real, legal, untried recognition lever IF detection recall becomes the binding constraint
  and we can obtain co-temporal bzi frames. Parking it as a candidate, not a commitment.

## BOTTOM LINE vs our program
1. Pollution read confirmed; don't chase >0.45 LB. Clean ~0.46 OOF is the real bar
   (tushhaaarrrr), and V17 0.39-tune is judged on method merit ("beyond 0.35 of great interest").
2. The credible top (0.46–0.48) got there by **better DETECTION via a decoupled, segmentation-
   weighted pipeline** — same conclusion our V60 reframing + oracle breakthrough reached.
3. Our firewall is intact under the host's explicit rulings; spine-aux and SWEFil both legal.
4. The current V60C question (multi-view consensus to convert the 0.86 oracle recall into PQ)
   is precisely the "find more filaments / better localization" axis the leader is ahead on.
