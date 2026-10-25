# V69 neighbour-frame check: result (2026-10-10). FAIL, line closed.

Pre-registration: `docs/v69_neighbour_frame_preregistration_2026-10-10.md`. p2first fold-0 model,
direct detector, V17 cell, scored with `eval_v61_compare.py --only` on each bucket's own file subset
(S_B = files having both the bucket frame and the o0 archive copy). Outputs in
`outputs/v69_nbr/compare_*.json` on rogii. Nothing trained, nothing submitted.

| bucket | files | PQ bucket | PQ o0 (same files) | bucket - o0 |
|---|---|---|---|---|
| m2 | 50 | 0.3311 | 0.4089 | -0.078 |
| p2 | 135 | 0.3582 | 0.4025 | -0.044 |
| m10 | 51 | 0.2921 | 0.3993 | -0.107 |
| p10 | 132 | 0.3106 | 0.4000 | -0.089 |
| m30 | 51 | 0.1927 | 0.3993 | -0.207 |
| p30 | 133 | 0.1985 | 0.3971 | -0.199 |
| xsite | 127 | 0.1998 | 0.4085 | -0.209 |

The re-encoding control holds: o0 scores the same as the Kaggle originals (within +-0.01).

**Gate:** mean(m10, p10) - o0 = -0.098, far below -0.020, so the result is a FAIL. No training arm
with neighbour frames. m2/p2 already lose more than 0.020, so frames vary a lot even within minutes.

**Cause is not geometry.** Phase correlation against o0 on 12 files: shifts <= 0.6 px for every
bucket, so registration and solar rotation are not the cause. On-disk image correlation with the
labelled frame falls fast: about 0.90 at 2 min, 0.85 at 10 min, 0.77 at 30 min, 0.46-0.86 for
another site. Frame-to-frame seeing and quality change is large; the labelled frames are probably
selected good-quality frames.

**What this does not separate.** The drop could be invalid labels on the neighbour frame or the
detector failing on lower-quality frames. The pre-registered gate does not distinguish the two, and
either way carried-over labels are not near-duplicate data. Because the host ruled out extra
ground-truth data on 2026-10-06, the line stays closed unless the host explicitly allows label
transfer.
