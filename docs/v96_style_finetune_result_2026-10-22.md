# V96 result: annotator-style fine-tune of the p2first detector (2026-10-22)

**Status: KILL on all three arms.** Fine-tuning the fold-0 p2first detector for 20 more epochs on
hard-annotator records (010101 + 010401), on annotation-group-01 records, or on all records (control)
lowers refined PQ on every subset, including the subset each arm was trained for. Gate was CONTINUE at
d ≥ +0.020 with P(d>0) ≥ 0.90, KILL below +0.005. Nothing trained beyond fold 0, nothing exported or
submitted. Per the pre-stated reading, "train on hard-annotator style" is closed as a lever.

- Pre-registration: `docs/v96_style_finetune_preregistration_2026-10-22.md`, sha256 `3ef6e574…`
  (`outputs/v96/prereg.sha256`, written before training).
- Code: `scripts/_v96_chain.sh` (poll GPU, train, infer, gate), `scripts/v96_refined_fold0.py`
  (V81/V94 harness, seed 96), `STYLE_ARMS` in `scripts/train_v61_fold.py`.
- Outputs: `outputs/v96/{refined_fold0.json,chain.log,refined.log,train_*.log,infer_*.log}` (VM + local);
  checkpoints `outputs/v96/work/<arm>_fold0/mask_ratio1/final.pt` (sha256 hard `eeea7ad7…`,
  g01 `7a209b9a…`, more `e6c4358b…`); views `outputs/v61_oof_cache/<arm>/fold0` (141 files each).
- Harness reproduced V73 base fold 0 exactly (PQ 0.453259, max per-file diff 0.0).
- Training: continuation stage only (20 epochs, mask_ratio 1) warm-started from
  `outputs/v61_persistent/p2first_fold0/final.pt`, GPU 2, 17:58 → 20:51 UTC. The chain waited 10 h for a
  GPU behind the vLLM server.

## Numbers (fold 0, 141 held-out files, host-exact per-record PQ, refined)

Subsets: hard = annotators 010101/010401 (26 records), g01 = group 01 (88), rest (131), all (219).

| arm | all | hard | g01 | rest |
|---|---:|---:|---:|---:|
| p2first (V73 base) | 0.4533 | 0.3998 | 0.4265 | 0.4709 |
| p2first_hard (119 rec) | 0.4462 | 0.3861 | 0.4170 | 0.4654 |
| p2first_g01 (428 rec) | 0.4504 | 0.3928 | 0.4253 | 0.4671 |
| p2first_more (935 rec, control) | 0.4503 | 0.3935 | 0.4221 | 0.4688 |

Delta vs p2first, per-file bootstrap (2000 draws, seed 96). Primary subset in bold.

| arm | subset | d | 95% CI | P(d>0) |
|---|---|---:|---|---:|
| p2first_hard | **hard** | **−0.0138** | [−0.0278, +0.0005] | 0.03 |
| p2first_hard | all | −0.0071 | [−0.0148, +0.0003] | 0.03 |
| p2first_hard | g01 | −0.0095 | [−0.0234, +0.0028] | 0.06 |
| p2first_hard | rest | −0.0055 | [−0.0144, +0.0041] | 0.13 |
| p2first_g01 | **g01** | **−0.0012** | [−0.0113, +0.0089] | 0.41 |
| p2first_g01 | all | −0.0029 | [−0.0087, +0.0033] | 0.18 |
| p2first_g01 | hard | −0.0070 | [−0.0217, +0.0083] | 0.17 |
| p2first_g01 | rest | −0.0039 | [−0.0109, +0.0036] | 0.16 |
| p2first_more | **all** | **−0.0030** | [−0.0059, +0.0003] | 0.04 |
| p2first_more | hard | −0.0063 | [−0.0140, −0.0015] | 0.00 |
| p2first_more | g01 | −0.0044 | [−0.0093, −0.0000] | 0.03 |
| p2first_more | rest | −0.0021 | [−0.0058, +0.0022] | 0.16 |

TP/FP/FN, all subset: p2first 1028/431/612; hard 1005/438/635; g01 1002/403/638; more 1027/453/613.
Matched pred/GT area ratio median: 1.055 base, 1.052 / 1.059 / 1.057 for the arms. Kept pred area median:
1281 base, 1395 / 1323 / 1288.

## Reading

- The hard arm is the worst arm on the hard subset (−0.014, P 0.03). 119 records for 20 epochs does not
  move extent or confidence toward the hard style; it loses recall everywhere (TP 1028 → 1005) and does
  not reduce FP on hard records (60 → 63). Matched area ratio is unchanged (1.055 → 1.052), so the
  detector did not learn tighter extents.
- The control arm is also negative on every subset (all −0.003, P 0.04): 20 extra continuation epochs on
  the full set cost a little. So the style arms' losses are the sum of "extra epochs hurt" and "fewer
  records hurt", with no style signal visible on top.
- Consistent with V95 addendum 2: hard-annotator loss is recall on zero-overlap misses plus extent
  disagreement that the refiner already absorbs. Detector fine-tuning on the style does not reach either.
- Closed: training-set reweighting toward hard annotators as a lever on the 0.39 public score. Still
  open and still untestable without a submission: whether the test style is the hard style at all.

Not touched: test images, test labels, MAGFiLO release, leaked CSV. No Kaggle submission.
