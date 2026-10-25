"""V92 public-board width probe: dilate or erode every mask of a submission CSV by 1 px (3x3 cross).
Row order within an image is kept as the ownership order (earlier row wins contested pixels), so the
output stays non-overlapping. Masks that fall below MIN_AREA after erosion are dropped. CPU, test images never read.
  python scripts/v92_width_probe.py outputs/v84/export/submission_v84.csv dilate outputs/v92/probe_dilate1.csv
  python scripts/v92_width_probe.py outputs/v84/export/submission_v84.csv erode outputs/v92/probe_erode1.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_submission import validate_submission

MIN_AREA = 200
src, mode, dst = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
op = {"dilate": ndi.binary_dilation, "erode": ndi.binary_erosion}[mode]
df = pd.read_csv(src, dtype=str, keep_default_na=False)
df["image"] = df.filament_id.str.split("_").str[0]
rows, stats = [], {"in": len(df), "out": 0, "dropped": 0, "px_in": 0, "px_out": 0}
for image, g in df.groupby("image", sort=False):
    occupied = np.zeros((2048, 2048), bool)
    n = 0
    for counts in g.segmentation_rle:
        m = mask_utils.decode({"size": [2048, 2048], "counts": counts.encode()}).astype(bool)
        stats["px_in"] += int(m.sum())
        m = op(m) & ~occupied
        a = int(m.sum())
        if a < MIN_AREA:
            stats["dropped"] += 1
            continue
        occupied |= m
        n += 1
        stats["px_out"] += a
        r = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
        rows.append({"filament_id": f"{image}_{n}", "segmentation_rle": r["counts"].decode()})
stats["out"] = len(rows)
dst.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(dst, index=False)
stems = {p.stem for p in (Path(__file__).resolve().parents[1]
                          / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images").glob("*.jpeg")}
print(stats)
print(validate_submission(dst, stems, min_area=MIN_AREA))
