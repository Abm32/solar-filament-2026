# V92 public-board probes (2026-10-19)

Question: is the OOF 0.459 → public 0.39 offset a tunable (mask width, keep threshold, all-data refit) or
test-label content? Five one-knob perturbations of the V84 export were submitted and read against the same
perturbations on the 707-file OOF (host scorer). Public shows 2 decimals; only ±0.01 moves are readable.

| probe | rows | public | Δ public | OOF (707) | Δ OOF |
|---|---:|---:|---:|---:|---:|
| V84 as submitted (reference) | 1112 | 0.39 | — | 0.4590 | — |
| dilate 1 px (`v92_width_probe.py`) | 1112 | 0.37 | −0.02 | 0.4383 | −0.021 |
| erode 1 px, MIN200 | 1101 | 0.36 | −0.03 | 0.4218 | −0.037 |
| conf 0.20, demote off | 1446 | 0.38 | −0.01 | 0.4465 | −0.013 |
| conf 0.40, demote on | 989 | 0.39 | 0.00 | 0.4508 | −0.008 |
| fold-0 p2first + fold-0 soft refiner + demote_all, conf 0.30 | 1049 | 0.39 | 0.00 | n/a (single fit) | — |

Reading: every knob moves the public board by the same amount it moves OOF. There is no mask-width style
shift in the test labels, the V84 operating point is already at the test optimum, and the all-data detector /
refiner fits behave like the fold models. The offset is test-label content (recall-side), not a tunable.

Also closed: a row-level public/private split of the solution (`v92_rowsplit_sim.py`) would give 0.32, not
0.39; by-image 0.458 and by-record 0.460 both match OOF.

Artifacts (rogii): `outputs/v92/{c20,c40,fold0}/probe_*.csv`, `outputs/v92/probe_{dilate1,erode1}.csv`,
`outputs/v92/conf_sweep.json`, `outputs/v92/oof_width.json`, `outputs/v92/rowsplit_sim.json`.
Note: `/mnt` was wiped at the 2026-10-17 reboot; the deployed detector now lives at
`outputs/v61_persistent/p2first_fold99/mask_ratio1/final.pt` (sha 7ff39296… unchanged). `_v74_export.sh` gained a
`CONF` env var (default 0.30, unchanged behaviour).
