# V78 result: label transfer to nearby GONG frames. All arms KILL (2026-10-11)

Pre-registration: `docs/v78_label_transfer_preregistration_2026-10-11.md`. Outputs: `outputs/v78/compare_*.json`. Fold 0, 141 files, ref p2first 0.3991 @0.30.
| arm | PQ (own best cut) | d | 95% CI | P(d>0) | verdict |
|---|---|---|---|---|---|
| p2first_lt1 (+-1 min) | 0.3882 | -0.0109 | [-0.0217, +0.0010] | 0.04 | kill |
| p2first_lt2 (+-2 min) | 0.3997 | +0.0006 | [-0.0125, +0.0131] | 0.56 | kill |
| p2first_lt5 (+-5 min) | 0.3927 | -0.0063 | [-0.0184, +0.0054] | 0.14 | kill |
Seed noise (p2first_seed2): -0.0050. Copying labels onto frames 1-5 minutes away adds no usable signal; at +-1 min it hurts
(near-duplicate images double the weight of the same label noise). Line closed.
