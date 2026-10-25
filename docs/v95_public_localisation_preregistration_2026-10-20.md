# V95 pre-registration: where on the public board does the −0.07 live? (2026-10-20, before any result)

Hash recorded in `outputs/v95/prereg.sha256` before the first probe is submitted.

## Question
OOF 0.459 → public 0.39 is a 15 % multiplicative loss shared by every clean team (V69, V85, V92). V92 showed
that mask width, confidence cut and all-data refits all move the public board by the same amount they move
OOF, i.e. the loss is not a tunable at the V84 operating point. Two readings remain:
- **H_uniform**: test records differ in annotator style everywhere; every image and every kind of mask carries
  the same 15 % loss. Nothing in the export can recover it; only better generalisation can.
- **H_local**: the loss is concentrated: a subset of test images (one observatory, one period, one label
  batch) or one class of mask (small, low-confidence, near-limb, fragments) is nearly unmatched on the public
  board, while the rest behaves like OOF. The algebra of pooled PQ makes this indistinguishable from
  H_uniform on the V84 submission alone: a fraction f of records with zero matches lowers PQ by exactly
  (1 − f) because the denominator (G + P)/2 does not depend on matching.

Checked today without submitting (CPU, train only): all 1154 training records sit on dark structure (ring
contrast on the limb-flattened image min 0.19 sd, median 1.20; flipped/transposed geometries ≈ 0.0), so
there are no misregistered label records in train (`scripts/v95_record_contrast.py`,
`outputs/v95/record_contrast.json`). Train/test split is random in time (runs test z = 0.73), site and year.

## Fixed design
Every probe is a one-knob removal or merge applied to the exact V84 submission
(`outputs/v84/export/submission_v84.csv`, sha256 prefix recorded in `outputs/v95/probes.sha256`), with
`scripts/v95_probe_ops.py`. The same op applied to the 707-file OOF V84 CSV (`outputs/v92/oof_v84.csv`) and
scored with the host scorer gives OOF_probe (`outputs/v95/oof_scores.json`, written before any public number
is read). The **uniform prediction** for each probe is

    P_u = 0.85 × OOF_probe       (0.85 = 0.39 / 0.459, the V84 ratio)

read at the board's 2 decimals. Paired noise between two of our submissions on the same 90 images is ≈0.004
(V85), rounding ±0.005.

Probes (14 + 4):
- Blank one observatory: `dropBh dropCh dropLh dropMh dropTh dropUh` (predictions removed for every test
  image of that site; 26–34 test images each).
- Blank one period: `dropyears` 2011–2012 / 2013–2014 / 2015–2017 / 2018–2022.
- Remove small masks: `minarea500`, `minarea1000`.
- Keep only the k highest-ranked masks per image: `topk3`, `topk5` (rows are already in ownership = score
  order).
- Remove near-limb masks: `maxrr090`, `maxrr080` (centroid r/R from the V94 Otsu disk fit on the image).
- Merge masks that touch after a G-px dilation: `merge10`, `merge25` (union of originals, no new pixels).

## Reading rule (fixed)
For each probe, d = public_probe − P_u.
- |d| ≤ 0.01: consistent with H_uniform for that axis.
- d ≥ +0.02 on a removal probe: the removed masks were mostly non-matching on test → the test labels lack
  that class of object (or that subset of images is unmatched). This is evidence for H_local on that axis.
- d ≤ −0.02 on a removal probe: the removed masks were *more* valuable on test than on OOF.
- For the six site blanks jointly: under H_uniform the six d's are all within ±0.01. One site with
  d ≥ +0.03 (its blanking costs less than half the predicted drop) is the H_local signature.
- For `merge`: d ≥ +0.02 means test labels are merged-style relative to train.

## What a result authorises
- Any axis with d ≥ +0.02 may be turned into **one** export change (the same op or its obvious inverse) and
  **one** confirmation submission; it must also be run on OOF and reported with both numbers.
- Nothing here touches training, test labels, the public MAGFiLO release or the leaked CSV. Probes are
  removals of our own predictions.
- All 14 + 4 probes are submitted regardless of early results (no stopping on a lucky read). Five per day.

## Budget
Deadline 2026-11-15 06:00 UTC, five submissions per day: ~130 remain. This plan uses 18 (+ ≤ 4
confirmations). Order: six site blanks, four year blanks, then the object-class probes.

## Amendment 2026-10-20 05:30 UTC (after the five site reads and the OOF table, before any other public number)
- Site blanks read: Bh 0.35, Ch 0.36, Lh 0.35, Mh 0.35, Th 0.36 against P_u 0.348 / 0.347 / 0.350 / 0.356 / 0.352.
  All |d| ≤ 0.013. H_uniform on the site axis. dropUh is not submitted (five sites already decide it).
- Year blanks withdrawn: the year-composition effect was computed directly instead (OOF PQ per year bin,
  reweighted to the test year mix: 0.4581 vs 0.4590, `outputs/v95/oof_subsets.json`), so the four year
  blanks cannot carry information the OOF does not already give. They are not submitted.
- maxrr090, merge10, merge25 withdrawn as unreadable (|ΔOOF| ≤ 0.011 → predicted public change < 0.01).
- Remaining probes, in order, with P_u fixed now: minarea500 (0.373), minarea1000 (0.306), topk5 (0.340),
  topk3 (0.272), maxrr080 (0.381). These are the only probes whose positive d would also *raise* the public
  score of a final export (an object class the test labels lack).
