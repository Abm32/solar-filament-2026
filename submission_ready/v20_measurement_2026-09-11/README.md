# V20 CondInst public measurement package

This package contains the exact gate-failed V20 candidate explicitly authorized
by the user for one Kaggle public-leaderboard measurement on 2026-09-11.

- Selected model: CondInst epoch 10, `model_0002349.pth`
- Checkpoint SHA-256: `e18bc02c655f52d7eccf8c16ea6c919cbc2d286f42452746a3500b9bfa797a78`
- Confidence: `0.50`
- Local tune/audit PQ: `0.358191 / 0.397240`
- Required local bars: `0.448703 / 0.470763`
- Gate status: failed; this is a measurement override, not a promoted model
- CSV: 981 masks, 176 populated test images, all 180 images processed
- CSV SHA-256: `247e1d0a7cfc0ca6ded5843e6549db8478ab71fce170d1ac210ccd55e6f2814e`

The exporter first refused without `--measurement-override`. Independent and
repository validators both confirmed complete RLE decoding, valid IDs, minimum
mask area, and zero per-image overlap. Upload this file at most once.
