# V84 result: the polish bundle passes the gate, by a hair (CONTINUE, 2026-10-11)

**Verdict: CONTINUE.** The full bundle (soft refiner + t0.50 + MIN200 + cross-fitted demote rule) gains
dPQ = **+0.00504** (95% CI [+0.0024, +0.0077], P(d>0) = 0.9995, 5/5 folds positive) over the deployed-equivalent
refined OOF (0.45393 to 0.45898, 707 files). It meets the pre-registered +0.005 bar with only 0.00004 to spare.
That margin is smaller than the retraining noise measured in V82 (about 0.0001).

- **Pre-registration:** `docs/v84_polish_bundle_preregistration_2026-10-11.md`, sha256 `94d75213…cabb79d`
  (`outputs/v84_prereg.sha256`), written 16:14 UTC before any V84 number.
- **Code:** `scripts/_v84_train.sh`, `scripts/v84_extract.py`, `scripts/v84_bundle.py`.
- **Outputs:** `outputs/v84/{soft_fold1-4*, extract_{base,soft}.pkl, result.json, bundle.log}`. Folds 0 are symlinks to `outputs/v82/soft_fold0*`.
- **Run conditions:** GPUs 0-2 only, CPU pool of 20, no submission, no test data.

## Checks
- **Smoke:** a CPU base-target smoke (fold 1, 2 files, 1+1 epochs) of the V82 script vs `train_v73_refiner.py` matches.
  Losses agree to 1e-7, and records_a/b (21/18) and PQ are identical.
- **Soft folds 1-4:** these used the V73 recipe (imagenet/resnet18/256, p2first, seed 42, 8 A + 12 B epochs). Their
  records_b (6111/6074/6089/6083) equal V73 base exactly.
- **Reproduction:** the extract reproduces V73 base at t0.45/c0.30 = **0.45393** (max per-file diff 1.1e-4), and at t0.50
  (diff 0). The soft extract matches each training job's own scoring at t0.45 (diff 0) and t0.50 (diff 8e-4).
- `only_demote` reproduces V83 demote (+0.0030).
- **Disclosed fix:** V83's regression target (sum of IoU>0.5 over records / R) exceeded 1 for one candidate in each
  t0.50 extract. That candidate matched two overlapping GTs of one record, and LightGBM cross_entropy refused it. The target is now clipped to 1.
  This changes no rule or parameter, and it is a no-op on the t0.45 data.

## Primary and secondary (707 files, pooled per-record PQ, paired file bootstrap 2000, seed 84)
| arm | PQ | dPQ | 95% CI | P(d>0) | per-fold dPQ |
|---|---|---|---|---|---|
| baseline (V73 base, t0.45, c0.30) | 0.45393 | – | – | – | – |
| **bundle** | **0.45898** | **+0.0050** | [+0.0024, +0.0077] | 0.9995 | +.0057 +.0097 +.0028 +.0027 +.0045 |
| − soft | 0.45810 | +0.0042 | [+0.0016, +0.0068] | 0.999 | +.0019 +.0078 +.0019 +.0054 +.0038 |
| − t0.50 | 0.45775 | +0.0038 | [+0.0012, +0.0064] | 0.999 | +.0049 +.0074 −.0005 +.0033 +.0042 |
| − MIN200 | 0.45899 | +0.0051 | [+0.0023, +0.0077] | 0.9995 | +.0060 +.0097 +.0028 +.0025 +.0045 |
| − demote | 0.45673 | +0.0028 | [+0.0013, +0.0044] | 0.9995 | +.0041 +.0034 +.0029 +.0016 +.0021 |
| only soft | 0.45504 | +0.0011 | [+0.0001, +0.0021] | 0.982 | |
| only t0.50 | 0.45462 | +0.0007 | [−0.0001, +0.0015] | 0.94 | |
| only MIN200 | 0.45602 | +0.0021 | [+0.0014, +0.0029] | 1.000 | |
| only demote | 0.45693 | +0.0030 | [+0.0007, +0.0053] | 0.993 | |

## Reading
- **Overlap:** the singles sum to +0.0069 and the bundle keeps +0.0050 (73%).
- **MIN200 is redundant once the demote rule is on.** The "− MIN200" arm is +0.00002 better. The rule already
  drops the small false positives.
- **Soft and t0.50 add more inside the bundle than alone** (+0.0009 and +0.0012 by leave-one-out).
- **Biggest marginal component:** demote (+0.0022 by leave-one-out).
- **Caution:** the pass margin (+0.00004) is below the noise from retraining a refiner. The gate is met as
  pre-registered, but the expected test gain is about +0.005 ± 0.003, not a sure +0.005.

## Export plan (not run)
The final export needs these four changes.
1. **All-data soft refiner.** The OOF evidence is for the V73 two-stage recipe (imagenet, stage A 8 ep at 3e-4 on GT crops,
   then stage B 12 ep at 1e-4 with soft targets). The deployed r3all was trained by `train_v63_refiner.py --recipe p2first
   --tag r3all --train-all --epochs 12 --lr 1e-4` (canonical init, stage B only, all 707 files).
   - **Needs new code.** Add a `--train-all` path to `train_v73_refiner.cmd_fold`: fit = all 707, no scoring. Run it
     through `train_v82_refiner_targets.py` so the soft hooks apply.
   - Save the plain `state_dict` (export_v74 loads `CropUNet` from a bare state dict, while V73/V82 save `{"model": sd}`).
   - About 25 min on one GPU:
     `CUDA_VISIBLE_DEVICES=0 python scripts/train_v82_refiner_targets.py fold --arm soft --fold 0 --train-all --out outputs/v84/export`
     The `--fold` value is ignored with `--train-all`.
2. **t0.50:** `export_v74.py --mask-threshold 0.50 --refiner outputs/v84/export/soft_all_refiner_state.pt`.
3. **MIN200:** `--min-area 200`. This is the existing default and is harmless, though redundant with the demote rule.
4. **Demote model.** Fit the V83 LightGBM on all 707 soft/t0.50 candidates (`extract_soft.pkl[0.5]`). Set λ by Dinkelbach on the
   pooled 707 OOF (cross-fitted) predictions. Drop test masks with score >= 0.30 and ŷ <= λ/2.
   - **Needs new code in export_v74:** compute the V83 per-candidate features (`v83_extract.refine_feats`, plus group, neighbour
     and file features) on test proposals, then apply the saved model and λ.
