# V61 Phase B, fold 0: Focal + Dice result (2026-10-05) — pre-registered verdict: KILL

Pre-registration: `docs/v61_phaseB_preregistration_2026-10-05.md` (fixed before any OOF number).
Comparison: `scripts/eval_v61_compare.py` -> `outputs/v61_compare_fold0.json`. 141 held-out files,
direct detector (no refiner), same files / consensus / ownership / per-record scorer for A, A', B.

## Numbers (verbatim from the comparison output)
| recipe | cut | conf | PQ | TP | FP | FN | mean matched IoU |
|---|---|---|---|---|---|---|---|
| A baseline | own best = baseline | 0.35 | 0.3763 | 864 | 455 | 776 | 0.6444 |
| A' seed 2 | own best = baseline | 0.35 | 0.3813 | 903 | 511 | 737 | 0.6448 |
| B focal_dice | own best | 0.40 | 0.3810 | 854 | 400 | 786 | 0.6456 |
| B focal_dice | baseline conf | 0.35 | 0.3779 | 890 | 502 | 750 | 0.6437 |

Paired differences on the same files (bootstrap over files, 2000 resamples):
- A' - A (seed noise): +0.0049, 95% CI [-0.0049, +0.0159], P(>0) 0.829  -> n = 0.0049
- B - A, own best cut: +0.0047, CI [-0.0064, +0.0161], P(>0) 0.804
- B - A, baseline cut: +0.0016, CI [-0.0089, +0.0116], P(>0) 0.618
- B - A' (derived from the table above): 0.3810 - 0.3813 = -0.0003

Recall by GT category (B own best vs A; n_GT thin 86, small 661, faint 53, normal 840):
thin 0.547 vs 0.616, small 0.399 vs 0.390, faint 0.453 vs 0.509, normal 0.618 vs 0.626.
Pooled small+thin+faint gain -0.0037; normal gain -0.0083.

## Gate (unchanged) applied mechanically
- d = +0.0047 < +0.005  -> KILL trigger 1
- d = +0.0047 < n = 0.0049 -> KILL trigger 2
- (also fails: d >= +0.010, d >= 2n, P(d>0) >= 0.90)
Verdict: `kill`. No close-enough exception, no weight sweep.

## Reading
- B is statistically indistinguishable from an unchanged baseline retrained with a different seed:
  B - A (+0.0047) matches A' - A (+0.0049), and B - A' is -0.0003.
- The threshold-shift control shows no hidden gain: at the baseline's own confidence B - A is
  +0.0016.
- B did not change the error structure in the hoped-for direction: pooled fine-structure recall did
  not rise more than normal recall (both slightly negative). The thin-recall drop (0.616 -> 0.547)
  rests on 86 GT in one fold and is weak evidence either way.

## Limits of this result (not softening the verdict)
- One fold, one A/A' seed pair: n = 0.0049 is itself a noisy estimate. Fold-0 is a direction, not
  a result; the kill rule was fixed for this reason and is applied as written.
- Direct detector only. One formulation (BCE + 1.0 Focal + 1.0 Dice, gamma 2); weights were
  deliberately not tuned. The result kills THIS formulation, not every segmentation objective.

## Disclosure (kept with the result, per plan)
B stage 1 trained with the un-chunked loss. B stage 2 crashed with CUDA OOM in epoch 2, was
restarted from the finished stage-1 checkpoint with the activation-checkpointed, 16-instance-chunked
loss (`chunk=16`). Tests show equal value and gradient for the chunked vs un-chunked loss and
bit-identical upstream behaviour at zero weights, so this is an implementation change, not a change
of treatment. Details: `docs/v61_B_oom_incident_2026-10-05.md`. There is no evidence it affected
the outcome, and the result is a null, so a hidden implementation effect would have to have
suppressed a gain; the chunked-vs-unchunked tests give no reason to expect that.

## Decision
Loss line closed. Not running five-fold B. Next (regardless of B, as planned): the
reduced-early-downsampling / higher-effective-feature-resolution experiment, keeping the V17
instance formulation, judged against the 5-fold baseline OOF (direct PQ 0.3856, CI [0.376, 0.395]).
