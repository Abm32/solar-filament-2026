# V61 Phase B — incident note: stage-2 OOM in B and its fix (2026-10-05 ~17:20 IST)

## What happened
B (focal_dice, fold 0) completed stage 1 (80 ep, mask_ratio=2, 9538 s). Stage 2 (mask_ratio=1)
completed epoch 1, then crashed in epoch 2 with CUDA OOM inside `single_mask_loss`
(`gc = crop_mask(gt_mask.clone(), ...)`: "tried to allocate 884 MiB ... 15.03 GiB in use").
A' and A4 (upstream loss, same stage) did not OOM. The crash happened after the 16:55 check (B
was still finishing stage 1 then) and was found at the 17:16 check because GPU 0 read 0 %.

## Cause (measured, not assumed)
`scripts/_probe_v61_loss_mem.py`, 1024x1024 masks (stage-2 geometry), fp16 autocast:
    n=100 instances: upstream 1.83 GiB, naive Focal+Dice 5.01 GiB   (2.5-2.7x)
    n=200 instances: upstream 3.58 GiB, naive Focal+Dice 9.89 GiB
Focal+Dice materialises several extra full-size (n,H,W) temporaries that autograd keeps for
backward (sigmoid, focal, two clones for Dice). At mask_ratio=1 that is ~2.5x the loss memory, on
top of the model, so images with many instances exceed 15.6 GiB. Stage 1 (mask_ratio=2, 512x512)
is 4x smaller and fit.

## Fix
`make_single_mask_loss(..., chunk=16)`: the extra terms are computed 16 instances at a time under
activation checkpointing. The result is the same sum over instances (per-instance terms do not
interact), so only memory changes. Measured peak with chunk=16: 1.82 GiB (n=100), 3.58 GiB (n=200)
= 1.00x upstream.
Tests (12 pass): chunked == unchunked in value AND gradient for chunk in {1,2,3,16}; single-term
cases match; zero weights still bit-identical to upstream even with chunking; install() defaults to
chunked. `chunk` is a memory knob, not a hyperparameter: formulation B (BCE + 1.0 Focal + 1.0 Dice,
gamma 2, smooth 1) is unchanged, so the pre-registration stands as written.

## What B's result now consists of
- Stage 1: trained with the un-chunked loss (mathematically identical to the chunked one).
- Stage 2: restarted from the finished stage-1 checkpoint (`base/final.pt`, sha af524f78a21b...)
  with the chunked loss; the crashed partial run is kept in `mask_ratio1/run_crashed_oom_unchunked`.
  The restart used the same stage arguments and seed as the baselines.
- One difference to disclose: stage 2 of B was restarted from scratch of that stage (no partial
  epoch was reused), so it is a clean 20-epoch stage; nothing else differs from A/A'.
Timing: B stage 2 ~410 s/epoch vs ~230 s for the baselines (checkpointing recompute + the extra
terms), so ~2.3 h; expected completion ~19:50 IST.

## Process fix
Status checks must verify process liveness and the log tail, not just results.csv (which goes
stale silently when a job dies). `scripts/_v61_health.py` reads results.csv only; the 17:16 check
that caught this also inspected GPU utilisation and the process list. Note: an earlier draft of
this note wrongly blamed the 15:30 and 16:55 checks; B was alive and in stage 1 at both.
