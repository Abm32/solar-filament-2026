"""Registration sanity check: |Bz| vs smoothed H-alpha brightness (plage), correct vs flipped rows.

Active-region plage is bright in H-alpha and co-located with strong |Bz|. If the registration and
row orientation are right, corr(|Bz|, local H-alpha excess) is clearly higher than with rows
flipped or the magnetogram shifted. Train smoke files only.
"""
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


def excess(gray, disk):
    """Smoothed H-alpha minus a large-scale background, on the disk."""
    f = gray.astype(np.float32)
    fine = cv2.GaussianBlur(f, (0, 0), 6)
    coarse = cv2.GaussianBlur(f, (0, 0), 60)
    return fine - coarse


def main():
    man = [r for r in json.loads((MAG / "manifest_smoke.json").read_text()) if r["file"]]
    rows = []
    for r in man:
        gray = cv2.imread(str(next(TRAIN.rglob(r["image"]))), cv2.IMREAD_GRAYSCALE)
        disk = R.fit_disk(gray)
        bz, geom = R.read_mag(MAG / r["file"])
        ex = excess(gray, disk)
        res = {}
        for tag, flip, dx in (("correct", True, 0), ("rows_flipped", False, 0),
                              ("shift_+60px", True, 60)):
            w = np.abs(R.warp_mag(bz, geom, (disk[0] + dx, disk[1], disk[2]), flip_rows=flip))
            on = w > 0
            inner = on & (ex != 0)
            res[tag] = float(np.corrcoef(w[inner], ex[inner])[0, 1])
        rows.append(res)
        print(f"{r['image'][:14]} disk (cx {disk[0]:.0f}, cy {disk[1]:.0f}, r {disk[2]:.0f}) "
              + "  ".join(f"{k} {v:+.3f}" for k, v in res.items()))
    print("MEAN  " + "  ".join(f"{k} {np.mean([x[k] for x in rows]):+.3f}" for k in rows[0]))


if __name__ == "__main__":
    main()
