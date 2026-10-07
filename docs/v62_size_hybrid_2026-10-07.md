# V62 size-gated hybrid: the refiner does not erase p2first's small masks (2026-10-07)

Harness: `scripts/eval_v62_size_hybrid.py` -> `outputs/v62_size_hybrid.json` (Azure `rogii`, GPU 1).
Same 141 canonical held-out files, OOF proposals and V17 cell (t0.45/c0.30) as
`v62_refiner_test_2026-10-07.md`. The `refined` arm reproduces that test exactly (both recipes).

Pre-registered before the run: PRIMARY = p2first/hybrid - baseline/refined; PASS iff delta >= +0.010
and P(delta>0) >= 0.90. Hybrid = proposals under 1,000 px keep their unrefined flip-consensus mask.

| recipe | arm | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| baseline | refined | 0.4517 | 0.6886 | 0.6560 | 1004 | 464 | 589 |
| baseline | hybrid | 0.4297 | 0.6750 | 0.6367 | 977 | 499 | 616 |
| baseline | unrefined | 0.3940 | 0.6495 | 0.6067 | 930 | 543 | 663 |
| p2first | refined | 0.4566 | 0.6876 | 0.6641 | 1016 | 451 | 577 |
| p2first | hybrid | 0.4380 | 0.6760 | 0.6480 | 993 | 479 | 600 |
| p2first | unrefined | 0.4155 | 0.6595 | 0.6300 | 967 | 510 | 626 |

- PRIMARY: **-0.0137**, CI [-0.0247, -0.0022], P(>0) 0.01. **FAIL.**
- Secondary (p2first/hybrid - baseline/hybrid): +0.0083, CI [-0.0020, +0.0192], P 0.93.

## What this says
1. Hypothesis "the refiner overwrites p2first's finer small masks" is falsified. The refiner helps
   small proposals too: removing it below 1,000 px costs about 0.02 PQ for both detectors.
2. The refiner lifts both mask quality (SQ +0.039 / +0.028) and detection (more TPs, fewer FPs,
   because masks cross IoU 0.5 and overlap cleaning changes).
3. p2first's edge without the refiner is mostly SQ (+0.010) plus RQ (+0.023). After the refiner, SQ is
   equal (0.6876 vs 0.6886) and only the RQ edge survives (+0.008, TP +12). The refiner equalises
   mask quality; it does not remove p2first's extra detections.
4. Implication for "refiner trained on p2first proposals": the remaining headroom is the RQ edge,
   which the current refiner already keeps. Expect little from re-fitting the same refiner design.
