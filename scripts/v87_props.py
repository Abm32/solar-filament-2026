"""V87: image properties as the export path reads them (cv2 color for detector, cv2 gray for refiner, PIL header).
Prints 5 train + 5 test examples and aggregates over all 707 train / 180 test. CPU only, images only.
  CUDA_VISIBLE_DEVICES= python scripts/v87_props.py
"""
import collections
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
OUT = ROOT / "outputs/v87"


def props(p):
    c = cv2.imread(str(p))
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    im = Image.open(p)
    return dict(name=p.name, ext=p.suffix, bytes=p.stat().st_size, pil_mode=im.mode, pil_size=im.size,
                fmt=im.format, progressive=bool(im.info.get("progressive")), exif=bool(im.getexif()),
                dpi=str(im.info.get("dpi")), subsampling=getattr(__import__("PIL.JpegImagePlugin").JpegImagePlugin,
                                                                 "get_sampling")(im),
                color_shape=c.shape, dtype=str(c.dtype), channels_equal=bool((c[..., 0] == c[..., 1]).all()
                                                                             and (c[..., 1] == c[..., 2]).all()),
                gray_shape=g.shape, gray_min=int(g.min()), gray_max=int(g.max()), gray_mean=round(float(g.mean()), 2),
                gray_eq_c0=bool((g == c[..., 0]).all()), frac_zero=round(float((g == 0).mean()), 4))


res = {}
for split, sub in (("train", "train/train_images"), ("test", "test/test_images")):
    files = sorted((D / sub).iterdir())
    rows = [props(p) for p in files]
    res[split + "_examples"] = rows[:: len(rows) // 5][:5]
    agg = {}
    for k in ("ext", "pil_mode", "pil_size", "fmt", "progressive", "exif", "dpi", "subsampling", "color_shape",
              "dtype", "channels_equal", "gray_shape", "gray_eq_c0"):
        agg[k] = dict(collections.Counter(str(r[k]) for r in rows))
    for k in ("bytes", "gray_mean", "gray_max", "frac_zero"):
        v = np.array([r[k] for r in rows], float)
        agg[k] = dict(p5=float(np.percentile(v, 5)), median=float(np.median(v)), p95=float(np.percentile(v, 95)))
    res[split + "_agg"] = dict(n=len(rows), **agg)
(OUT / "props.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
print(json.dumps(res, indent=1, default=str))
