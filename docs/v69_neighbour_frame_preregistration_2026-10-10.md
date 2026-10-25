# V69 neighbour-frame check: pre-registration (2026-10-10, before any result)

## Question
Do the labels on a training frame still hold on nearby GONG archive frames from the same site, and
from another site? If yes, nearby frames are extra labelled training data (5-10x), which the host
allows for training (unlabelled GONG frames and temporal information in training only).

## Facts checked first
The archive copy of a Kaggle train image (`url` field, gong2.nso.edu/HA/hag) has the same geometry:
phase-correlation shift (0.003, -0.006) px. On the disk it matches closely: correlation 0.98, mean
abs diff ~2 grey levels (re-encoded). Off the disk it differs (corr 0.33). Frames come every minute
at each of 6 sites.

## Data (firewall)
Only the 141 fold-0 held-out TRAIN files. Buckets, each the nearest archive frame within +-90 s of
the target: o0 (archive copy of the labelled frame, re-encoding control), m2/p2, m10/p10, m30/p30
min, same site; xsite = nearest frame from another site within 15 min. No test date is queried.
Script: `scripts/fetch_v69_nbr.py` (run locally; the VM cannot reach gong2.nso.edu). Files go in
`/mnt/v69_nbr/<bucket>/<kaggle name>`. Ground truth is carried over UNCHANGED: no rotation
correction, so any drop with offset includes solar rotation (<= ~4.5 px at disk centre at 30 min).

## Measurement
p2first fold-0 model, V17 cell, direct detector (`eval_v61_oof.py infer --image-dir`), compared with
p2first on the original images over the files present in every same-site bucket
(`eval_v61_compare.py --only`). Primary contrast: PQ(bucket) - PQ(o0), at conf 0.35.

## Gate
- LABELS TRANSFER (design a fold-0 training arm with neighbour frames as extra data) iff the mean of
  m10 and p10 minus o0 >= -0.020. xsite is reported but not gating; if it also passes (>= -0.020),
  the training arm may use other-site frames.
- If m2/p2 already lose more than 0.020 against o0, frame-to-frame seeing variation is large. That
  is reported as a finding: neighbour frames are then augmentation-like, not duplicate labels.
- Nothing here authorises a submission.
