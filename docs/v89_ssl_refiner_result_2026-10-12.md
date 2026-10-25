# V89 SSL-pretrained refiner: result (2026-10-12)

Pre-registration: `docs/v89_ssl_refiner_preregistration_2026-10-12.md` (filed 05:00 UTC; Amendment 1 records the
restart of both SSL runs at ~05:45 UTC with the unchanged single-GPU recipe; DDP attempt abandoned, no outputs kept).

Data: 2,063 unlabelled images (707 train pixels + 1,356 V76 GONG archive frames, every frame >= 28.7 h from any
test timestamp). No test images, no labels in SSL.

Arms (ImageNet ResNet18 init, 40k steps, seed 89), fine-tuned with the exact V84 soft recipe on fold 0:
- `mim`: masked-image modelling of the full U-Net (64 x 256 crops/step); loss 0.41 -> 0.275.
- `moco`: MoCo v2 on the encoder (128/step); loss 7.40 -> ~5.7.

Note: the first `moco` fine-tune failed at 08:46 on an init sanity assert (torch does not report missing
`num_batches_tracked` buffers). The assert was corrected to ignore those buffers; loading was unchanged; rerun 08:48.

## Fold 0, full V84 path (t0.50, cut 0.30, MIN200, cross-fitted demote), 141 files

| arm | PQ | TP/FP/FN | d vs V84 seed 42 | P(d>0) | verdict |
|---|---|---|---|---|---|
| V84 soft (seed 42, control) | 0.45899 | 982/312/658 | - | - | - |
| mim | 0.45791 | 977/313/663 | -0.0011 | 0.28 | FAIL |
| moco | 0.45628 | 974/318/666 | -0.0027 | 0.02 | FAIL |
| ctrl seed 43 | 0.45609 | 973/314/667 | -0.0029 | 0.09 | (noise) |
| ctrl seed 44 | 0.45739 | 981/314/659 | -0.0016 | 0.24 | (noise) |
| ctrl seed 45 (addition) | 0.45587 | 971/309/669 | -0.0031 | 0.10 | (noise) |

Gate (pre-registered): d >= +0.008 and P >= 0.90. **Both arms FAIL -> KILL. No 5-fold run.**

Secondary (not a gate): vs the mean of the three fresh-seed controls (0.45645), mim is +0.0015 and moco -0.0002,
both inside seed noise (+-0.0015 range of the controls). Domain SSL on ~2k frames does not move the refined PQ.

Interpretation: consistent with the V76/V77/V81/V82 pattern. The refiner's output is set by the annotator-label
ceiling and its recipe, not by encoder features; better in-domain features add nothing measurable.
