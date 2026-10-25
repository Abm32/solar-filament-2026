"""Compare submission CSV stats (rows, images, instances/image, mask areas). python v84_sub_stats.py a.csv b.csv ..."""
import json
import sys

import numpy as np
import pandas as pd
from pycocotools import mask as mu

for p in sys.argv[1:]:
    df = pd.read_csv(p, dtype=str)
    img = df.filament_id.str.rsplit("_", n=1).str[0]
    a = np.asarray([mu.area({"size": [2048, 2048], "counts": c.encode()}) for c in df.segmentation_rle])
    per = img.value_counts()
    print(json.dumps(dict(file=p, rows=len(df), images=int(img.nunique()), inst_per_img_mean=round(float(per.mean()), 3),
                          inst_per_img_median=float(per.median()), area_min=int(a.min()), area_p10=float(np.percentile(a, 10)),
                          area_median=float(np.median(a)), area_mean=round(float(a.mean()), 1), total_px=int(a.sum()))))
