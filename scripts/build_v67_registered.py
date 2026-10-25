"""V67: register every fetched train magnetogram onto the 512x512 H-alpha grid and save as one npz.

Runs locally (needs astropy, see /tmp/magvenv); the VM then needs only numpy to read the result.
Output: data/external/gong_bz/registered512.npz with arrays
  names (N,) str      train image file names that have a magnetogram
  bz    (N,512,512) float16   Gauss, on the 2048/4 H-alpha grid, 0 off-disk
  gap   (N,) int16    minutes between magnetogram and H-alpha frame
  disk  (N,3) float32 fitted H-alpha disk (cx, cy, r) in 2048 px
Train files only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v67_register as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
MAG = ROOT / "data/external/gong_bz"
SMALL = 512


def main():
    assert "test" not in str(TRAIN)
    records = [r for r in json.loads((MAG / "manifest.json").read_text())
               if r["file"] and (MAG / r["file"]).exists()]
    names, bzs, gaps, disks = [], [], [], []
    for i, r in enumerate(records):
        gray = cv2.imread(str(next(TRAIN.rglob(r["image"]))), cv2.IMREAD_GRAYSCALE)
        disk = R.fit_disk(gray)
        bz, geom = R.read_mag(MAG / r["file"])
        full = R.warp_mag(bz, geom, disk)
        small = cv2.resize(full, (SMALL, SMALL), interpolation=cv2.INTER_AREA)
        names.append(r["image"])
        bzs.append(small.astype(np.float16))
        gaps.append(int(r["gap_min"]))
        disks.append(disk)
        if i % 100 == 0:
            print(i, len(records), r["image"], np.round(disk, 1), flush=True)
    np.savez_compressed(MAG / "registered512.npz", names=np.array(names), bz=np.stack(bzs),
                        gap=np.array(gaps, np.int16), disk=np.array(disks, np.float32))
    radii = [d[2] for d in disks]
    print("saved", len(names), "files; disk r range", round(float(min(radii)), 1),
          round(float(max(radii)), 1))


if __name__ == "__main__":
    main()
