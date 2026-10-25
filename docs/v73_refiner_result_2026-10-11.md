# V73 result: refiner variants (crop512, rn34, dil15). All KILL (2026-10-11)

Pre-registration: `docs/v73_refiner_preregistration_2026-10-10.md`. Output: `outputs/v73/compare.json`. 707 files, 5-fold, cell t0.45/c0.30.
Base r3 (k7) PQ 0.4539.
| arm | d vs base | 95% CI | P(d>0) | folds + | verdict |
|---|---|---|---|---|---|
| crop512 | +0.0000 | [-0.0013, +0.0014] | 0.53 | 3 | KILL |
| rn34 | +0.0006 | [-0.0005, +0.0017] | 0.86 | 4 | KILL |
| dil15 | +0.0006 | [-0.0001, +0.0013] | 0.95 | 4 | KILL |
Refiner capacity, crop size, and seed dilation are all exhausted as levers.
