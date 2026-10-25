"""Overlay fitted disk circle and warped |Bz|>150 G contours on the H-alpha frame (train smoke)."""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v67_register as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
MAG = ROOT / "data/external/gong_bz"
PICK = ["20110109104734Ch", "20140218230234Lh", "20150409081954Uh", "20180926014350Uh"]

records = json.loads((MAG / "manifest_smoke.json").read_text())
man = {r["image"][:16]: r for r in records if r["file"]}
fig, ax = plt.subplots(1, 4, figsize=(24, 6.5))
for a, key in zip(ax, PICK, strict=True):
    r = man[key]
    gray = cv2.imread(str(next(TRAIN.rglob(r["image"]))), cv2.IMREAD_GRAYSCALE)
    disk = R.fit_disk(gray)
    bz, geom = R.read_mag(MAG / r["file"])
    w = R.warp_mag(bz, geom, disk)
    a.imshow(gray, cmap="gray")
    a.add_patch(plt.Circle(disk[:2], disk[2], fill=False, color="yellow", lw=0.8))
    a.contour(np.abs(w) > 150, levels=[0.5], colors="red", linewidths=0.8)
    a.set_title(f"{key} disk=({disk[0]:.0f},{disk[1]:.0f},r{disk[2]:.0f})", fontsize=9)
    a.axis("off")
plt.tight_layout()
plt.savefig("/tmp/overlay_v67.png", dpi=45)
