"""V79 audit: train vs test image/metadata distribution shift. CPU only.
python scripts/audit_v79_shift_images.py  -> outputs/v79_audit/shift_images.{csv,json}"""
import json, os, sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
OUT = ROOT / "outputs/v79_audit"


def quality(im):
    # ponytail: IJG-standard luma table ratio estimate; ~1 point accurate, fine for shift detection
    q = im.quantization.get(0) if hasattr(im, "quantization") else None
    if not q:
        return -1
    std = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
           14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
           49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]
    s = 100.0 * np.mean(np.asarray(q, float)) / np.mean(std)
    return round(200 - s / 2) if s <= 100 else round(5000 / s)


def work(args):
    split, path = args
    im = Image.open(path)
    rec = dict(split=split, name=path.name, mode=im.mode, w=im.width, h=im.height, fmt=im.format,
               bytes=path.stat().st_size, q=quality(im), prog=int(bool(im.info.get("progressive"))),
               sub=str(getattr(im, "layer", ""))[:30])
    a = np.asarray(im.convert("L"), np.float32)
    if im.mode == "RGB":
        c = np.asarray(im, np.float32)
        rec["chan_maxdiff"] = float(np.abs(c[..., 0] - c[..., 1]).max())
    disk = a > 0.3 * np.percentile(a, 99)
    ys, xs = np.nonzero(disk)
    cy, cx = ys.mean(), xs.mean()
    r = np.sqrt(disk.sum() / np.pi)
    rr = np.hypot(*np.mgrid[:a.shape[0], :a.shape[1]] - np.array([cy, cx])[:, None, None]) / r
    inside = a[rr < 0.95]
    prof = [float(a[(rr >= lo) & (rr < lo + 0.1)].mean()) for lo in np.arange(0, 1.0, 0.1)]
    rec.update(cx=float(cx), cy=float(cy), r=float(r), sky=float(a[rr > 1.1].mean()),
               mean=float(inside.mean()), std=float(inside.std()),
               p1=float(np.percentile(inside, 1)), p50=float(np.percentile(inside, 50)),
               p99=float(np.percentile(inside, 99)), limb=prof[9] / max(prof[0], 1),
               sat=float((a >= 254).mean()), uniq=int(len(np.unique(a))),
               hp=float(np.abs(np.diff(inside[:4096])).mean()))
    n = path.stem
    rec.update(year=int(n[:4]), month=int(n[4:6]), hour=int(n[8:10]), site=n[14])
    return rec


def main():
    jobs = [("train", p) for p in sorted((D / "train/train_images").glob("*"))]
    jobs += [("test", p) for p in sorted((D / "test/test_images").glob("*"))]
    with Pool(16) as pool:
        recs = pool.map(work, jobs, chunksize=8)
    OUT.mkdir(parents=True, exist_ok=True)
    import csv
    keys = sorted({k for r in recs for k in r})
    with open(OUT / "shift_images.csv", "w") as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(recs)
    summ = {}
    for s in ("train", "test"):
        R = [r for r in recs if r["split"] == s]
        summ[s] = dict(n=len(R),
                       **{k: dict(Counter(r[k] for r in R)) for k in ("site", "year", "mode", "fmt", "w", "q", "prog")},
                       **{k: [round(float(np.percentile([r[k] for r in R], p)), 3) for p in (10, 50, 90)]
                          for k in ("hour", "month", "bytes", "cx", "cy", "r", "sky", "mean", "std", "p1", "p50",
                                    "p99", "limb", "sat", "uniq", "hp")})
    from scipy.stats import ks_2samp
    tr = [r for r in recs if r["split"] == "train"]; te = [r for r in recs if r["split"] == "test"]
    summ["ks_p"] = {k: float(ks_2samp([r[k] for r in tr], [r[k] for r in te]).pvalue)
                    for k in ("year", "month", "hour", "bytes", "cx", "cy", "r", "mean", "std", "p1", "p99",
                              "limb", "sky", "uniq", "hp")}
    (OUT / "shift_images.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
