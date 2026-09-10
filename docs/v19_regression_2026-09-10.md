# V19 regression fix — eval/train resolution mismatch + LR warmup

> Status: fixes applied to `scripts/train_v19_mask2former.py`, not yet retrained.
> Written 2026-09-10 after E029 (docs/experiments.md) recorded V19's first full
> run failing SUBMIT_BAR by roughly 3x (best tune PQ 0.153578 vs required
> 0.448703).

## Diagnosis (zero GPU cost, read from E029's returned artifacts)

Two concrete bugs in the training script, not evidence against Mask2Former as
an architecture choice:

**1. Model-selection evaluation ran at 2x the training resolution, single-shot,
not the sliding-window inference the design doc specified.** `run_full_training`
trained at `IMG_SIZE_TRAIN=1024` but called `evaluate_pq(..., IMG_SIZE_INFER)`
with `IMG_SIZE_INFER=2048` for **every** tune/audit evaluation, including the
one used to pick the best checkpoint. `evaluate_pq` resizes the *whole* image
through the processor at the requested `infer_size` and resizes predictions
back down — there is no sliding-window logic anywhere in the file, despite
`v19_design_2026-09-09.md` explicitly calling for "full-resolution
sliding-window inference at 2048×2048 at test time." Mask2Former's query
decoding and mask embeddings are scale-sensitive; evaluating a model at 2x the
scale it was ever trained at, every single epoch, both explains the outsized FP
volume (1386–3214 vs the YOLO control's 211–306) and decouples the loss curve
(monotonically falling, plateauing ~23 by epoch 10) from the PQ curve
(oscillating 0.075–0.154 with no visible trend across all 20 epochs) — the
model may have been improving at 1024 the whole time while being scored at a
resolution where that improvement didn't show up the same way.

**2. No LR warmup or decay.** `AdamW(lr=1e-4)` flat for 20 epochs on a backbone
that is pretrained (COCO) but has a freshly reinitialized 2-class head
(`class_predictor.{weight,bias}`, `criterion.empty_weight` all shape-mismatched
and randomly initialized per the smoke log). Fine-tuning a mostly-pretrained
transformer with a flat, undecayed LR and no warmup is a well-known recipe for
early instability that can persist for many epochs — consistent with the
observed non-convergence.

## Fix applied

- `evaluate_pq` calls in `run_full_training` (both per-epoch tune eval and the
  final audit eval) now pass `IMG_SIZE_TRAIN`, not `IMG_SIZE_INFER`. Model
  selection and gate-checking (SUBMIT_BAR / PROMOTION_GATE) are now internally
  consistent with what the model was trained at. `IMG_SIZE_INFER` (2048,
  sliding-window) is deferred to a **separate submission-export step**, to be
  built only if/when SUBMIT_BAR clears at 1024 — it was never validated as
  correct at any resolution and conflating it with model selection was exactly
  the bug.
- Added linear warmup (`WARMUP_FRACTION = 0.05` of total optimizer steps) then
  cosine decay to ~0, via `torch.optim.lr_scheduler.LambdaLR`. `train_one_epoch`
  now takes an optional `scheduler` argument and steps it once per
  `optimizer.step()` call (i.e. once per `GRAD_ACCUM` micro-batches, plus the
  unconditional end-of-epoch flush) — not once per micro-batch, which would run
  the schedule `GRAD_ACCUM`x too fast.
- `v19_full_result.json` now also records `eval_resolution` so any future
  artifact self-documents which resolution model selection used.

## What this fix does NOT change

- `NUM_QUERIES=100`, `BATCH_SIZE=2`, `GRAD_ACCUM=4`, `EPOCHS=20`, the backbone
  choice, and the loss function are all unchanged. This is deliberately a
  minimal, targeted fix for two identified bugs, not a broader hyperparameter
  sweep — per this project's own repeated lesson (E025, "V14 measures three of
  its own bugs, not its hypotheses"), fix known defects and re-measure before
  reaching for a bigger change.
- If this fix alone does not clear SUBMIT_BAR, the next diagnostic step is to
  inspect the corrected per-epoch trace for a real convergence trend (the
  original symptom this fix targets) before adding further changes — not to
  assume more epochs or a hyperparameter grid would help, which is exactly the
  assumption V14/V15 made incorrectly about their own designs.

## Verification

Added `tests/test_v19_mask2former.py::test_full_training_evaluates_model_selection_at_train_resolution`
(pins that all `evaluate_pq` calls inside `run_full_training` use
`IMG_SIZE_TRAIN`, never `IMG_SIZE_INFER`) and
`test_lr_schedule_warms_up_then_decays_to_near_zero` /
`test_train_one_epoch_steps_scheduler_on_optimizer_step_boundaries` (pin the
warmup/decay shape and the step-boundary timing). Full local suite: 410 passed,
1 skipped (up from 407).
